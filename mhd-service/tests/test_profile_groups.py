"""档案引用管理：分组档存取、引用展开、删除冲突、成环/悬空拒收、并发安全。"""
from __future__ import annotations

import concurrent.futures
import threading

import pytest

from mhd.group_tree import GroupNode, SegmentNode, parse_group
from mhd.induction import ChannelParams
from mhd.profiles import (
    ProfileInUseError,
    ProfileNotFoundError,
    ProfileStore,
    StoredGroup,
)

SEGMENT = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}
PARAMS = ChannelParams(2.0, 1.0, 100.0, 10.0, 0.5)


def seg(**changes) -> dict:
    data = dict(SEGMENT)
    data.update(changes)
    return {"segment": data}


def grp(connection="series", *members) -> dict:
    return {"connection": connection, "members": list(members)}


@pytest.fixture()
def store():
    return ProfileStore()


def test_single_and_group_profiles_share_namespace(store):
    store.save_channel("s1", PARAMS)
    stored = store.save_group("g1", grp("parallel", {"ref": "s1"}, seg()))
    assert isinstance(stored, StoredGroup)
    assert store.names() == ["g1", "s1"]
    assert store.listing() == (["g1", "s1"], ["g1"])


def test_get_channel_on_group_name_and_vice_versa(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}))
    with pytest.raises(ProfileNotFoundError):
        store.get_channel("g1")
    with pytest.raises(ProfileNotFoundError):
        store.get_stored_group("s1")
    entry = store.get_entry("g1")
    assert isinstance(entry, StoredGroup)
    assert store.get_entry("s1") is PARAMS or store.get_entry("s1") == PARAMS


def test_resolve_expands_group_and_inlines_segment_profile(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("parallel", {"ref": "s1"}, seg(flow_velocity=80.0)))
    tree = store.resolve("g1")
    assert tree.segment_count == 2
    refs = [item.node.ref for item in tree.segments]
    assert refs == ["s1", None]


def test_group_profile_can_reference_another_group_profile(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}, seg()))
    store.save_group("g2", grp("parallel", {"ref": "g1"}, seg()))
    tree = store.resolve("g2")
    assert tree.segment_count == 3


def test_reference_uses_live_value_everywhere(store):
    """引用取被引档当前值：核算与查看口径一致（不做保存时刻快照）。"""
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}))

    tree_before = store.resolve("g1")
    emf_before = tree_before.segments[0].node.params.flow_velocity

    store.save_channel("s1", ChannelParams(2.0, 1.0, 140.0, 10.0, 0.5))
    tree_after = store.resolve("g1")
    assert tree_after.segments[0].node.params.flow_velocity == 140.0
    assert emf_before == 100.0

    # 查看档案时展开同样拿最新值（get_entry 保留结构，resolve 拿值）
    assert store.resolve("g1").segments[0].node.params.flow_velocity == 140.0


def test_delete_referenced_profile_conflicts_and_lists_referrers(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}))
    store.save_group("g2", grp("parallel", {"ref": "s1"}))
    with pytest.raises(ProfileInUseError) as excinfo:
        store.delete("s1")
    assert excinfo.value.referrers == ["g1", "g2"]
    # 删除失败后档案仍在、仍可用
    assert store.get_channel("s1") == PARAMS


def test_delete_group_profile_used_by_another_group_conflicts(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}))
    store.save_group("g2", grp("series", {"ref": "g1"}))
    with pytest.raises(ProfileInUseError) as excinfo:
        store.delete("g1")
    assert excinfo.value.referrers == ["g2"]


def test_delete_succeeds_when_no_references(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}))
    store.delete("g1")
    store.delete("s1")
    assert store.names() == []
    with pytest.raises(ProfileNotFoundError):
        store.delete("s1")


def test_self_reference_rejected_and_not_stored(store):
    with pytest.raises(Exception):
        store.save_group("cyc", grp("series", {"ref": "cyc"}, seg()))
    with pytest.raises(ProfileNotFoundError):
        store.get_stored_group("cyc")


def test_two_node_cycle_rejected_with_originals_intact(store):
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}, seg()))
    store.save_group("g2", grp("parallel", {"ref": "s1"}, seg()))
    # g1 -> g2 合法
    store.save_group("g1", grp("series", {"ref": "g2"}, seg()))
    # g2 -> g1 会成环：g2 -> g1 -> g2
    with pytest.raises(Exception, match="成环"):
        store.save_group("g2", grp("parallel", {"ref": "g1"}, seg()))
    # g2 保留旧值，引用 s1，仍可正常展开
    tree = store.resolve("g2")
    assert tree.segment_count == 2


def test_dangling_reference_rejected_at_save(store):
    with pytest.raises(ProfileNotFoundError):
        store.save_group("g1", grp("series", {"ref": "ghost"}, seg()))
    assert store.names() == []


def test_failed_save_does_not_overwrite_existing(store):
    store.save_group("g1", grp("series", seg()))
    original = store.get_stored_group("g1")
    with pytest.raises(ProfileNotFoundError):
        store.save_group("g1", grp("series", {"ref": "ghost"}))
    assert store.get_stored_group("g1") == original


def test_replacing_segment_with_group_keeps_referrers_valid(store):
    """被引档在单段/分组两种类型间切换，引用者按新类型展开。"""
    store.save_channel("s1", PARAMS)
    store.save_group("g1", grp("series", {"ref": "s1"}))
    store.save_group("s1", grp("parallel", seg(), seg(flow_velocity=80.0)))
    assert isinstance(store.get_entry("s1"), StoredGroup)
    assert store.resolve("g1").segment_count == 2


def test_concurrent_save_delete_leaves_no_dangling_references():
    """并发保存/删除交错后，仓库里不允许存在指向空名字的分组档。"""
    store = ProfileStore()
    n_segments = 10
    for i in range(n_segments):
        store.save_channel(f"s{i}", PARAMS)

    stop = threading.Event()
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        rng_state = tid * 9973 + 17
        try:
            for _ in range(500):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                pick = rng_state % 5
                if pick == 0:
                    refs = [
                        {"ref": f"s{(rng_state >> (3 * k)) % (n_segments + 3)}"}
                        for k in range(1 + (rng_state % 2))
                    ]
                    members = refs + [{"segment": SEGMENT}]
                    try:
                        store.save_group(
                            f"g{tid}_{rng_state % 6}",
                            grp("series", *members),
                        )
                    except (ProfileNotFoundError, Exception):
                        pass
                elif pick == 1:
                    try:
                        store.delete(f"s{rng_state % n_segments}")
                    except (ProfileNotFoundError, ProfileInUseError):
                        pass
                elif pick == 2:
                    # 段档可能被删掉，反复补回来
                    store.save_channel(f"s{rng_state % n_segments}", PARAMS)
                elif pick == 3:
                    try:
                        store.delete(f"g{tid}_{rng_state % 6}")
                    except (ProfileNotFoundError, ProfileInUseError):
                        pass
                else:
                    names = store.names()
                    if names:
                        target = names[rng_state % len(names)]
                        try:
                            store.resolve(target)
                        except ProfileNotFoundError:  # pragma: no cover
                            errors.append(AssertionError(f"dangling {target}"))
        except Exception as exc:  # pragma: no cover - 并发用例不应抛出
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, t) for t in range(8)]
        for future in futures:
            future.result()
    assert not errors

    # 最终不变量：每个分组档的每条直接引用都能解析；所有档都能展开
    snapshot = {name: store.get_entry(name) for name in store.names()}
    for name, entry in snapshot.items():
        if isinstance(entry, StoredGroup):
            tree = store.resolve(name)
            assert tree.segment_count >= 1


def test_concurrent_save_and_delete_of_same_name_is_atomic():
    """保存与删除同一名字高频交错，结束后该名字要么是合法档要么不存在。"""
    store = ProfileStore()

    def saver():
        for _ in range(300):
            store.save_channel("hot", PARAMS)

    def deleter():
        for _ in range(300):
            try:
                store.delete("hot")
            except ProfileNotFoundError:
                pass

    t1 = threading.Thread(target=saver)
    t2 = threading.Thread(target=deleter)
    t1.start(); t2.start(); t1.join(); t2.join()
    # 不抛异常即通过；若名字还在，它必须是合法可解析的档
    if "hot" in store.names():
        assert store.get_channel("hot") == PARAMS

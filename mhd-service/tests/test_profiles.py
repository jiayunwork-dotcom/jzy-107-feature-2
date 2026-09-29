"""档案仓库：单段/分组共存、引用闭包、环、删除冲突、并发原子性。"""
from __future__ import annotations

import concurrent.futures
import threading

import pytest

from mhd.groups import GroupEntry, GroupNode, RefNode, SegmentEntry, SegmentNode
from mhd.induction import ChannelParams
from mhd.profiles import (
    DanglingReferenceError,
    GroupLimitError,
    ProfileInUseError,
    ProfileNotFoundError,
    ProfileStore,
    ProfileTypeError,
)
from mhd.group_validation import validate_group_tree

P = ChannelParams(2.0, 1.0, 100.0, 10.0, 0.5)
SEGMENT_JSON = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}


def save_group(store: ProfileStore, name: str, raw: dict) -> None:
    root, refs = validate_group_tree(raw)
    store.save_group(name, root, raw, refs)


@pytest.fixture()
def store() -> ProfileStore:
    return ProfileStore()


def test_segment_profile_roundtrip(store):
    store.save("base", P)
    assert store.get("base") is not store  # sanity: object exists
    assert store.get("base") == P
    assert store.names() == ["base"]
    assert store.group_names() == []
    assert not store.is_group("base")


def test_missing_profile_raises(store):
    with pytest.raises(ProfileNotFoundError):
        store.get("ghost")
    with pytest.raises(ProfileNotFoundError):
        store.resolve("ghost")


def test_type_mismatch_between_kinds(store):
    store.save("base", P)
    save_group(store, "grp", {"type": "series", "members": [{"ref": "base"}]})
    with pytest.raises(ProfileTypeError):
        store.get("grp")  # 按单段取分组档
    with pytest.raises(ProfileTypeError):
        store.resolve("base")  # 按分组取单段档


def test_group_can_reference_segment(store):
    store.save("base", P)
    save_group(store, "grp", {"type": "parallel", "members": [{"ref": "base"}]})
    root = store.resolve("grp")
    assert isinstance(root.children[0], SegmentNode)
    assert root.children[0].params == P


def test_group_can_reference_group(store):
    store.save("base", P)
    save_group(store, "inner", {"type": "series", "members": [{"ref": "base"}]})
    save_group(store, "outer", {"type": "parallel", "members": [{"ref": "inner"}]})
    root = store.resolve("outer")
    assert isinstance(root.children[0], GroupNode)
    assert isinstance(root.children[0].children[0], SegmentNode)


def test_dangling_reference_rejected_on_save(store):
    with pytest.raises(DanglingReferenceError) as info:
        save_group(store, "grp", {"type": "series", "members": [{"ref": "ghost"}]})
    assert info.value.missing == ["ghost"]
    assert "grp" not in store.names()


def test_self_reference_rejected_as_cycle(store):
    from mhd.groups import ReferenceCycleError

    with pytest.raises(ReferenceCycleError):
        save_group(store, "ring", {"type": "series", "members": [{"ref": "ring"}]})
    assert "ring" not in store.names()


def test_reference_cycle_rejected_on_save(store):
    from mhd.groups import ReferenceCycleError

    store.save("base", P)
    save_group(store, "a", {"type": "series", "members": [{"ref": "base"}]})
    save_group(store, "b", {"type": "series", "members": [{"ref": "a"}]})
    # 把 a 改成引用 b -> a -> b -> a 成环，保存必须失败且旧 a 不受影响
    with pytest.raises(ReferenceCycleError):
        save_group(store, "a", {"type": "series", "members": [{"ref": "b"}]})
    root = store.resolve("a")
    assert isinstance(root.children[0], SegmentNode)  # 旧定义仍可用


def test_referenced_profile_cannot_be_deleted(store):
    store.save("base", P)
    save_group(store, "grp", {"type": "parallel", "members": [{"ref": "base"}]})
    with pytest.raises(ProfileInUseError) as info:
        store.delete("base")
    assert info.value.referrers == ["grp"]
    assert "base" in store.names()


def test_delete_succeeds_after_reference_removed(store):
    store.save("base", P)
    save_group(store, "grp", {"type": "parallel", "members": [{"ref": "base"}]})
    # 覆盖成自包含分组后，base 不再被引用
    save_group(store, "grp", {"type": "parallel", "members": [SEGMENT_JSON]})
    store.delete("base")
    assert "base" not in store.names()


def test_dynamic_reference_picks_up_new_values(store):
    store.save("base", P)
    save_group(store, "grp", {"type": "parallel", "members": [{"ref": "base"}]})
    root1 = store.resolve("grp")
    assert root1.children[0].params.magnetic_flux_density == 2.0
    store.save("base", ChannelParams(4.0, 1.0, 100.0, 10.0, 0.5))
    root2 = store.resolve("grp")
    assert root2.children[0].params.magnetic_flux_density == 4.0


def test_repeated_ref_uses_single_definition(store):
    """同一档在分组里被引用两次，展开后是两个叶子但参数同源。"""
    store.save("base", P)
    save_group(
        store, "twice",
        {"type": "parallel", "members": [{"ref": "base"}, {"ref": "base"}]},
    )
    root = store.resolve("twice")
    assert len(root.children) == 2
    assert all(isinstance(c, SegmentNode) for c in root.children)


def test_expanded_segment_limit_enforced_through_refs(store):
    store.save("base", P)
    # 51 个引用本身合法；再并列引用两次（102 段）超过上限 100
    big = {"type": "parallel", "members": [{"ref": "base"}] * 51}
    save_group(store, "big", big)
    with pytest.raises(GroupLimitError):
        save_group(
            store, "bigger",
            {"type": "series", "members": [{"ref": "big"}, {"ref": "big"}]},
        )
    # 展开 big 本身不超
    store.resolve("big")


def test_concurrent_saves_and_deletes_leave_no_dangling_reference():
    """高频交错保存/删除后：任何分组档引用的名字都必须存在。"""
    store = ProfileStore()
    barrier = threading.Barrier(8)

    def worker(worker_id: int):
        barrier.wait()
        for i in range(200):
            seg_name = f"seg{(worker_id + i) % 12}"
            grp_name = f"grp{worker_id % 6}"
            try:
                store.save(seg_name, P)
            except Exception:
                pass
            try:
                save_group(
                    store, grp_name,
                    {"type": "parallel", "members": [{"ref": seg_name}]},
                )
            except (DanglingReferenceError, GroupLimitError, Exception):
                # 悬空（对方恰好已删）、成环、超上限都是允许的拒收结果
                pass
            try:
                store.delete(seg_name)
            except (ProfileNotFoundError, ProfileInUseError):
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    # 最终一致性检查：仓库里每个分组档引用的名字都必须真实存在
    names = set(store.names())
    for name in sorted(names):
        entry = store.get_entry(name)
        if isinstance(entry, GroupEntry):
            store.resolve(name)  # 展开成功即无悬空、无环


def test_concurrent_delete_reference_is_either_rejected_or_blocked():
    """删除与引用恰好交错：要么引用保存被拒，要么删除被引用冲突挡住。"""
    store = ProfileStore()
    store.save("base", P)
    outcomes = []

    def deleter():
        barrier.wait()
        try:
            store.delete("base")
            outcomes.append(("deleted",))
        except ProfileInUseError:
            outcomes.append(("blocked",))
        except ProfileNotFoundError:
            outcomes.append(("missing",))

    def saver():
        barrier.wait()
        try:
            save_group(
                store, "grp",
                {"type": "series", "members": [{"ref": "base"}]},
            )
            outcomes.append(("saved",))
        except DanglingReferenceError:
            outcomes.append(("rejected",))

    barrier = threading.Barrier(2)
    t1 = threading.Thread(target=deleter)
    t2 = threading.Thread(target=saver)
    t1.start(); t2.start(); t1.join(); t2.join()
    kinds = {o[0] for o in outcomes}
    # 不允许「删除成功 + 引用保存成功」并存（那会留下悬空引用）
    assert not ({"deleted", "saved"} <= kinds)
    if "grp" in store.names():
        store.resolve("grp")  # 若保存成功，引用必须可解析

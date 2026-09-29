"""分组通道统一网络求解的物理判据（验收清单逐条钉死）。

判据包括：
- 单段分组与老写法逐量一致（标串联或并联都一样）；
- 相同两段串联 == 电极间距加倍，并联 == 截面积加倍；
- 同类型嵌套与摊平一层一致；
- 任意结构/任意负载下并联 KCL、串联等流、全局能量平衡；
- σ=0：并联里等于去掉该段；串联链断开电流为零；全断开功率为零；
- 开路并联环流正确，环流被倒灌段电流为负；
- 扫描峰值精确落在等效内阻；扫描点与单点逐位一致；
- 重复提交逐位相同。
"""
from __future__ import annotations

import math
import random

import pytest

from mhd.circuit import solve_channel
from mhd.group_sweep import sweep_group
from mhd.groups import GroupNode, SegmentNode
from mhd.induction import ChannelParams
from mhd.load_solver import solve_load

BASE = dict(
    magnetic_flux_density=2.0,
    electrode_spacing=1.0,
    flow_velocity=100.0,
    conductivity=10.0,
    cross_section_area=0.5,
)


def seg(**overrides) -> SegmentNode:
    params = dict(BASE)
    params.update(overrides)
    return SegmentNode(ChannelParams(**params))


def group(kind: str, *children) -> GroupNode:
    return GroupNode(kind, tuple(children))


# --------------------------------------------------------- 结构等价

def test_single_segment_group_matches_legacy_both_kinds():
    """只有一段的分组，标串联或并联，整体与老写法完全一致。"""
    legacy = solve_load(ChannelParams(**BASE), 0.2)
    for kind in ("series", "parallel"):
        sol = solve_channel(group(kind, seg()))
        point = sol.at(0.2)
        assert sol.open_circuit_voltage == pytest.approx(legacy.open_circuit_voltage)
        assert sol.internal_resistance == pytest.approx(legacy.internal_resistance)
        assert point.load_current == pytest.approx(legacy.current)
        assert point.load_power == pytest.approx(legacy.power)
        # 唯一的段就是通道本身
        assert len(point.segments) == 1
        assert point.segments[0].current == pytest.approx(legacy.current)
        assert point.segments[0].emf == pytest.approx(legacy.open_circuit_voltage)
        assert point.segments[0].internal_resistance == pytest.approx(
            legacy.internal_resistance
        )


def test_two_identical_series_equals_doubled_spacing():
    doubled = dict(BASE, electrode_spacing=2.0)
    legacy = solve_load(ChannelParams(**doubled), 0.4)
    sol = solve_channel(group("series", seg(), seg()))
    point = sol.at(0.4)
    assert sol.open_circuit_voltage == pytest.approx(legacy.open_circuit_voltage)
    assert sol.internal_resistance == pytest.approx(legacy.internal_resistance)
    assert point.load_current == pytest.approx(legacy.current)
    assert point.load_power == pytest.approx(legacy.power)
    assert legacy.open_circuit_voltage == pytest.approx(400.0)
    assert legacy.internal_resistance == pytest.approx(0.4)


def test_two_identical_parallel_equals_doubled_area():
    doubled = dict(BASE, cross_section_area=1.0)
    legacy = solve_load(ChannelParams(**doubled), 0.1)
    sol = solve_channel(group("parallel", seg(), seg()))
    point = sol.at(0.1)
    assert sol.open_circuit_voltage == pytest.approx(legacy.open_circuit_voltage)
    assert sol.internal_resistance == pytest.approx(legacy.internal_resistance)
    assert point.load_current == pytest.approx(legacy.current)
    assert point.load_power == pytest.approx(legacy.power)
    assert legacy.internal_resistance == pytest.approx(0.1)
    # 两段对半分流
    assert point.segments[0].current == pytest.approx(point.load_current / 2)
    assert point.segments[1].current == pytest.approx(point.load_current / 2)


@pytest.mark.parametrize("kind", ["series", "parallel"])
def test_nesting_same_kind_matches_flattened(kind):
    """同类型组套同类型组，与摊平成一层结果一致。"""
    nested = group(kind, group(kind, seg(), seg()), group(kind, seg(), seg()))
    flat = group(kind, seg(), seg(), seg(), seg())
    for load in (0.0, 0.05, 0.2, 1.0, 7.0):
        a = solve_channel(nested).at(load)
        b = solve_channel(flat).at(load)
        assert a.open_circuit_voltage == pytest.approx(b.open_circuit_voltage)
        assert a.internal_resistance == pytest.approx(b.internal_resistance)
        assert a.load_current == pytest.approx(b.load_current)
        assert a.load_power == pytest.approx(b.load_power)


# --------------------------------------------------------- KCL/能量

def _walk(node, path="$"):
    yield node, path
    if isinstance(node, GroupNode):
        for i, child in enumerate(node.children):
            yield from _walk(child, f"{path}.members[{i}]")


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_kcl_and_energy_balance_on_random_trees(seed):
    """任意结构、任意负载：并联成员电流代数和=组电流；
    各段发出功率之和 = 负载功率 + 各段内阻损耗之和。"""
    rng = random.Random(seed)

    def rseg():
        return seg(
            magnetic_flux_density=rng.uniform(0.0, 3.0),
            electrode_spacing=rng.uniform(0.2, 2.0),
            flow_velocity=rng.uniform(0.0, 150.0),
            conductivity=rng.choice([0.0, 0.0, 5.0, 20.0]),
            cross_section_area=rng.uniform(0.1, 1.0),
        )

    def rnode(depth):
        if depth >= 3 or rng.random() < 0.4:
            return rseg()
        return group(
            rng.choice(["series", "parallel"]),
            *[rnode(depth + 1) for _ in range(rng.randint(2, 4))],
        )

    tree = group(rng.choice(["series", "parallel"]), *[rnode(0) for _ in range(3)])
    sol = solve_channel(tree)
    for load in (0.0, rng.uniform(0.0, 5.0), rng.uniform(0.0, 100.0)):
        point = sol.at(load)
        generated = sum(s.generated_power for s in point.segments)
        dissipated = sum(s.dissipation for s in point.segments)
        assert generated == pytest.approx(
            point.load_power + dissipated, rel=1e-8, abs=1e-6
        )

        seg_by_path = {s.path: s for s in point.segments}
        grp_by_path = {g.path: g for g in point.groups}
        for node, path in _walk(tree):
            if not isinstance(node, GroupNode):
                continue
            member_currents = []
            for i, child in enumerate(node.children):
                cp = f"{path}.members[{i}]"
                if isinstance(child, SegmentNode):
                    member_currents.append(seg_by_path[cp].current)
                else:
                    member_currents.append(grp_by_path[cp].current)
            grp = grp_by_path[path]
            if node.kind == "parallel":
                assert sum(member_currents) == pytest.approx(
                    grp.current, abs=1e-7
                )
            else:
                for value in member_currents:
                    assert value == pytest.approx(grp.current, abs=1e-7)
        # 根组电流就是负载电流
        assert point.groups[0].current == pytest.approx(point.load_current, abs=1e-8)


# --------------------------------------------------------- 零电导率

def test_dead_branch_in_parallel_equals_removing_it():
    with_dead = solve_channel(group("parallel", seg(), seg(conductivity=0.0)))
    without = solve_channel(group("parallel", seg()))
    for load in (0.0, 0.2, 1.0, 5.0):
        a = with_dead.at(load)
        b = without.at(load)
        assert a.load_current == pytest.approx(b.load_current)
        assert a.load_power == pytest.approx(b.load_power)
        assert a.segments[1].current == 0.0
        assert a.segments[1].generated_power == 0.0
        assert a.segments[1].dissipation == 0.0
        assert a.segments[1].internal_resistance == math.inf


def test_dead_segment_in_series_chain_cuts_chain():
    sol = solve_channel(group("series", seg(), seg(conductivity=0.0), seg()))
    assert math.isinf(sol.internal_resistance)
    for load in (0.0, 0.5, 10.0):
        point = sol.at(load)
        assert point.load_current == 0.0
        assert point.load_power == 0.0
        assert all(s.current == 0.0 for s in point.segments)


def test_all_dead_gives_zero_everywhere_and_finite_reported_emf():
    sol = solve_channel(
        group(
            "parallel",
            group("series", seg(conductivity=0.0), seg(conductivity=0.0)),
            seg(conductivity=0.0),
        )
    )
    assert math.isinf(sol.internal_resistance)
    point = sol.at(3.0)
    assert point.load_current == 0.0
    assert point.load_power == 0.0
    assert all(s.current == 0.0 for s in point.segments)
    assert all(s.generated_power == 0.0 for s in point.segments)
    # 扫描在全断通道上不报错、不除零，曲线全零
    swept = sweep_group(sol, 0.0, 10.0, 50)
    assert all(p.power == 0.0 and p.current == 0.0 for p in swept.curve)


# --------------------------------------------------------- 环流

def test_open_circuit_circulation_between_parallel_branches():
    """开路时两支路 E 不同，环流 (E1-E2)/(r1+r2)：强段正向、弱段被倒灌。"""
    sol = solve_channel(group("parallel", seg(magnetic_flux_density=3.0),
                              seg(magnetic_flux_density=1.0)))
    # E=300/100，r=0.2/0.2 -> 环流 200/0.4 = 500 A
    assert sol.seg_oc[0] == pytest.approx(500.0)
    assert sol.seg_oc[1] == pytest.approx(-500.0)
    # 开路对外没有电流
    assert sol.grp_oc_i[0] == pytest.approx(0.0, abs=1e-9)
    # 环流的能量只在内阻上耗散
    point = sol.at(1e18)
    generated = sum(s.generated_power for s in point.segments)
    dissipated = sum(s.dissipation for s in point.segments)
    assert generated == pytest.approx(dissipated, rel=1e-9)


# --------------------------------------------------------- 扫描

def test_group_sweep_peak_lands_on_equivalent_resistance():
    tree = group(
        "series",
        group("parallel", seg(), seg(magnetic_flux_density=1.5)),
        seg(cross_section_area=0.8),
    )
    sol = solve_channel(tree)
    result = sweep_group(sol, 0.01 * sol.internal_resistance,
                         100 * sol.internal_resistance, 200)
    assert result.peak.load_resistance == pytest.approx(sol.internal_resistance)
    u = sol.open_circuit_voltage
    assert result.peak.load_power == pytest.approx(
        u * u / (4 * sol.internal_resistance)
    )
    assert result.peak.load_power == max(p.power for p in result.curve)


def test_group_sweep_matches_single_point_bitwise():
    tree = group("parallel", seg(), group("series", seg(), seg()))
    sol = solve_channel(tree)
    result = sweep_group(sol, 0.05, 2.0, 97)
    for point in result.curve:
        single = sol.at(point.load_resistance)
        assert single.load_current == point.current
        assert single.load_power == point.power
    # 峰值点的逐段展开也与同负载单点逐位一致
    peak_single = sol.at(result.peak.load_resistance)
    for a, b in zip(result.peak.segments, peak_single.segments):
        assert a.current == b.current
        assert a.dissipation == b.dissipation
        assert a.generated_power == b.generated_power


# --------------------------------------------------------- 确定性

def test_repeated_requests_give_bit_identical_results():
    tree = group(
        "series",
        group("parallel", seg(magnetic_flux_density=2.5),
              seg(magnetic_flux_density=1.5), seg(conductivity=0.0)),
        group("parallel", seg(), seg()),
    )
    points_a = []
    points_b = []
    for load in (0.0, 0.13, 0.77, 4.2):
        points_a.append(
            [s.current for s in solve_channel(tree).at(load).segments]
        )
        points_b.append(
            [s.current for s in solve_channel(tree).at(load).segments]
        )
    assert points_a == points_b

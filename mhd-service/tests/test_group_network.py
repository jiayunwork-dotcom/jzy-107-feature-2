"""分组通道统一网络求解的判据测试。

覆盖需求里点名的物理关系：单段与老写法等价、串/并倍长倍面积、
同类型组嵌套可摊平、任意结构下的 KCL 与功率守恒、电导率为零的
断开/悬空行为、扫描峰值与单点一致性、重复提交逐位相同。
"""
from __future__ import annotations

import math
import random

import pytest

from mhd.group_solver import build_model, solve_group, sweep_group
from mhd.group_tree import (
    GroupNode,
    SegmentNode,
    expand_refs,
    parse_group,
)
from mhd.induction import ChannelParams
from mhd.load_solver import solve_load

SERIES = "series"
PARALLEL = "parallel"

BASELINE = ChannelParams(2.0, 1.0, 100.0, 10.0, 0.5)
# U_oc = 200 V，R_int = 0.2 Ω


def _segment(params: ChannelParams = BASELINE) -> SegmentNode:
    return SegmentNode(params=params)


def _group(connection: str, *children) -> GroupNode:
    return GroupNode(connection=connection, children=children)


def _expand(node: GroupNode):
    """这些用例不使用 ref，lookup 被引用即说明构造有误。"""

    def fail(name: str):
        raise AssertionError(f"测试树不应包含引用: {name}")

    return expand_refs(node, fail)


def _modified(**changes) -> ChannelParams:
    return ChannelParams(
        changes.get("magnetic_flux_density", BASELINE.magnetic_flux_density),
        changes.get("electrode_spacing", BASELINE.electrode_spacing),
        changes.get("flow_velocity", BASELINE.flow_velocity),
        changes.get("conductivity", BASELINE.conductivity),
        changes.get("cross_section_area", BASELINE.cross_section_area),
    )


# ---------------------------------------------------------------------------
# 1. 单段分组（串/并两种标法）与老写法完全一致
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("connection", [SERIES, PARALLEL])
@pytest.mark.parametrize("load", [0.0, 0.2, 1.0, None])
def test_single_segment_group_matches_legacy(connection, load):
    model = build_model(_expand(_group(connection, _segment())))
    grouped = solve_group(model, load)
    legacy = solve_load(BASELINE, load)

    assert grouped.open_circuit_voltage == legacy.open_circuit_voltage
    assert grouped.internal_resistance == legacy.internal_resistance
    assert grouped.current == legacy.current
    assert grouped.power == legacy.power
    assert len(grouped.segments) == 1
    assert len(grouped.groups) == 1
    seg_report = grouped.segments[0]
    assert seg_report.emf == pytest.approx(200.0)
    assert seg_report.resistance == pytest.approx(0.2)
    assert seg_report.current == legacy.current


def test_single_segment_group_parses_from_json_both_shapes():
    for connection in (SERIES, PARALLEL):
        node = parse_group(
            {
                "connection": connection,
                "members": [
                    {
                        "segment": {
                            "magnetic_flux_density": 2.0,
                            "electrode_spacing": 1.0,
                            "flow_velocity": 100.0,
                            "conductivity": 10.0,
                            "cross_section_area": 0.5,
                        }
                    }
                ],
            }
        )
        result = solve_group(build_model(_expand(node)), 0.2)
        assert result.open_circuit_voltage == pytest.approx(200.0)
        assert result.current == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 2. 两同段串联==间距加倍；并联==截面积加倍
# ---------------------------------------------------------------------------
def test_two_identical_series_segments_match_doubled_spacing():
    doubled = _modified(electrode_spacing=2.0)
    grouped = solve_group(
        build_model(_expand(_group(SERIES, _segment(), _segment()))), 0.4
    )
    legacy = solve_load(doubled, 0.4)
    assert grouped.open_circuit_voltage == legacy.open_circuit_voltage
    assert grouped.internal_resistance == legacy.internal_resistance
    assert grouped.current == legacy.current
    assert grouped.power == legacy.power


def test_two_identical_parallel_segments_match_doubled_area():
    doubled = _modified(cross_section_area=1.0)
    grouped = solve_group(
        build_model(_expand(_group(PARALLEL, _segment(), _segment()))), 0.1
    )
    legacy = solve_load(doubled, 0.1)
    assert grouped.open_circuit_voltage == legacy.open_circuit_voltage
    assert grouped.internal_resistance == legacy.internal_resistance
    assert grouped.current == legacy.current
    assert grouped.power == legacy.power
    # 两支路各承担一半电流，且方向都为正
    assert all(s.current == pytest.approx(grouped.current / 2) for s in grouped.segments)


# ---------------------------------------------------------------------------
# 3. 同类型组套同类型组与摊平一层相同
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("load", [0.0, 0.2, 1.3, None])
def test_nested_series_flattens(load):
    nested = _group(SERIES, _group(SERIES, _segment(), _segment()), _segment())
    flat = _group(SERIES, _segment(), _segment(), _segment())
    a = solve_group(build_model(_expand(nested)), load)
    b = solve_group(build_model(_expand(flat)), load)
    assert a.open_circuit_voltage == b.open_circuit_voltage
    assert a.internal_resistance == b.internal_resistance
    assert a.current == b.current
    assert a.power == b.power


@pytest.mark.parametrize("load", [0.0, 0.2, 1.3, None])
def test_nested_parallel_flattens(load):
    s90 = _segment(_modified(flow_velocity=90.0))
    s110 = _segment(_modified(flow_velocity=110.0))
    s100 = _segment(_modified(flow_velocity=100.0))
    nested = _group(PARALLEL, _group(PARALLEL, s90, s110), s100)
    flat = _group(PARALLEL, s90, s110, s100)
    a = solve_group(build_model(_expand(nested)), load)
    b = solve_group(build_model(_expand(flat)), load)
    assert a.open_circuit_voltage == pytest.approx(b.open_circuit_voltage)
    assert a.internal_resistance == b.internal_resistance
    assert a.current == b.current
    assert a.power == b.power


# ---------------------------------------------------------------------------
# 4. 任意结构、任意负载：KCL 与功率守恒
# ---------------------------------------------------------------------------
def _random_tree(rng: random.Random, depth: int) -> GroupNode:
    def subtree(level: int):
        if level >= 3 or rng.random() < 0.4:
            return _segment(
                ChannelParams(
                    rng.uniform(0.5, 3.0),
                    rng.uniform(0.2, 2.0),
                    rng.uniform(10.0, 200.0),
                    rng.uniform(0.5, 20.0),
                    rng.uniform(0.1, 1.0),
                )
            )
        connection = rng.choice([SERIES, PARALLEL])
        return _group(
            connection, *[subtree(level + 1) for _ in range(rng.randint(2, 3))]
        )

    connection = rng.choice([SERIES, PARALLEL])
    return _group(connection, *[subtree(1) for _ in range(rng.randint(2, 3))])


def _current_at(report, path) -> float:
    for segment in report.segments:
        if segment.path == path:
            return segment.current
    for group in report.groups:
        if group.path == path:
            return group.current
    raise KeyError(path)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("load", [0.0, 0.05, 0.3, 1.0, 7.7, None])
def test_kcl_and_power_balance_on_random_trees(seed, load):
    rng = random.Random(seed)
    tree = _expand(_random_tree(rng, 0))
    model = build_model(tree)
    report = model.evaluate(load)

    emitted = sum(s.emitted_power for s in report.segments)
    dissipated = sum(s.dissipation for s in report.segments)
    scale = max(1.0, abs(emitted), abs(dissipated), abs(report.power))
    # 各段发出功率之和 = 负载功率 + 各段内阻损耗之和
    assert emitted - dissipated - report.power == pytest.approx(0.0, abs=1e-7 * scale)

    for indexed in tree.groups:
        group_report = next(g for g in report.groups if g.path == indexed.path)
        currents = [
            _current_at(report, indexed.path + (i,))
            for i in range(len(indexed.node.children))
        ]
        if indexed.node.connection == PARALLEL:
            # 并联成员电流代数和等于流过该组的电流
            assert sum(currents) == pytest.approx(
                group_report.current, abs=1e-7 * max(1.0, abs(group_report.current))
            )
        else:
            # 串联成员是同一股电流
            for value in currents:
                assert value == pytest.approx(
                    currents[0], abs=1e-7 * max(1.0, abs(currents[0]))
                )

    if load is None:
        # 开路：根组电流为零，但支路环流可以非零
        assert report.current == 0.0


def test_open_circuit_circulating_current_is_visible_and_signed():
    """两段电动势不同的通道并联开路：高电动势段正向送出，低电动势段被倒灌。"""
    tree = _expand(
        _group(
            PARALLEL,
            _segment(_modified(flow_velocity=100.0)),  # 200 V
            _segment(_modified(flow_velocity=80.0)),   # 160 V
        )
    )
    report = solve_group(build_model(tree), None)
    assert report.current == 0.0
    first, second = report.segments
    # 环流 = (200-160)/(0.2+0.2) = 100 A
    assert first.current == pytest.approx(100.0)
    assert second.current == pytest.approx(-100.0)  # 负号：被倒灌
    assert first.emitted_power == pytest.approx(200.0 * 100.0)
    assert second.emitted_power == pytest.approx(160.0 * -100.0)
    # 环流只在内阻上消耗，负载功率为零
    assert report.power == 0.0
    total_dissipation = sum(s.dissipation for s in report.segments)
    assert total_dissipation == pytest.approx(2 * 100.0 ** 2 * 0.2)


# ---------------------------------------------------------------------------
# 5. 电导率为零
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("load", [0.0, 0.5, None])
def test_zero_conductivity_parallel_branch_is_as_if_removed(load):
    dead = _modified(conductivity=0.0)
    with_dead = _group(
        PARALLEL,
        _segment(_modified(flow_velocity=100.0)),
        _segment(_modified(flow_velocity=80.0)),
        _segment(dead),
    )
    without = _group(
        PARALLEL,
        _segment(_modified(flow_velocity=100.0)),
        _segment(_modified(flow_velocity=80.0)),
    )
    a = solve_group(build_model(_expand(with_dead)), load)
    b = solve_group(build_model(_expand(without)), load)
    assert a.open_circuit_voltage == pytest.approx(b.open_circuit_voltage)
    assert a.internal_resistance == pytest.approx(b.internal_resistance)
    assert a.current == b.current
    assert a.power == b.power

    dead_report = next(s for s in a.segments if math.isinf(s.resistance))
    assert dead_report.current == 0.0
    assert dead_report.dissipation == 0.0


def test_zero_conductivity_in_series_chain_kills_chain():
    tree = _expand(
        _group(SERIES, _segment(), _segment(_modified(conductivity=0.0)), _segment())
    )
    report = solve_group(build_model(tree), 0.1)
    assert math.isinf(report.internal_resistance)
    assert report.current == 0.0
    assert report.power == 0.0
    assert all(s.current == 0.0 for s in report.segments)
    # 等效电动势仍按链上各段之和报出（600 V），便于核对开路电平
    assert report.open_circuit_voltage == pytest.approx(600.0)


def test_entirely_open_channel_is_all_zero_across_sweep():
    tree = _expand(
        _group(PARALLEL, _segment(_modified(conductivity=0.0)),
               _segment(_modified(conductivity=0.0)))
    )
    result = sweep_group(build_model(tree), 0.0, 10.0, 50)
    assert math.isinf(result.internal_resistance)
    assert all(point.current == 0.0 and point.power == 0.0 for point in result.curve)
    assert result.peak.power == 0.0


def test_dead_series_branch_bypassed_by_live_parallel_branch_conducts():
    tree = _expand(
        _group(
            PARALLEL,
            _group(SERIES, _segment(), _segment(_modified(conductivity=0.0))),
            _segment(),
        )
    )
    report = solve_group(build_model(tree), 0.2)
    # 坏支路被旁路，整网仍由活支路供电
    assert report.current == pytest.approx(500.0)
    assert report.internal_resistance == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 6. 扫描：峰值恰在等效内阻；曲线上任一点 == 同负载单点
# ---------------------------------------------------------------------------
def test_sweep_peak_lands_exactly_on_equivalent_resistance():
    tree = _expand(_group(SERIES, _segment(), _segment()))
    model = build_model(tree)
    result = sweep_group(model, 0.01, 10.0, 123)
    r_eq = model.equivalent_resistance
    assert result.peak.load_resistance == pytest.approx(r_eq)
    expected_peak = model.equivalent_emf ** 2 / (4 * r_eq)
    assert result.peak.power == pytest.approx(expected_peak)
    assert result.peak.power == max(p.power for p in result.curve)


def test_sweep_curve_matches_single_point_bit_for_bit():
    tree = _expand(
        _group(
            PARALLEL,
            _group(SERIES, _segment(), _segment(_modified(flow_velocity=80.0))),
            _segment(_modified(flow_velocity=120.0, conductivity=6.0)),
        )
    )
    model = build_model(tree)
    result = sweep_group(model, 0.03, 3.0, 64)
    for point in result.curve:
        single = model.evaluate(point.load_resistance)
        assert single.current == point.current
        assert single.power == point.power


def test_sweep_handles_infinite_equivalent_resistance_in_range():
    tree = _expand(_group(SERIES, _segment(), _segment(_modified(conductivity=0.0))))
    result = sweep_group(build_model(tree), 0.0, 5.0, 20)
    assert len(result.curve) == 20
    assert all(p.power == 0.0 for p in result.curve)


# ---------------------------------------------------------------------------
# 7. 同一请求重复提交，逐段结果逐位相同
# ---------------------------------------------------------------------------
def test_repeated_submission_is_bitwise_identical():
    raw = {
        "connection": "parallel",
        "members": [
            {"group": {"connection": "series",
                        "members": [{"segment": _raw_segment()},
                                    {"segment": _raw_segment(flow_velocity=80.0)}]}},
            {"segment": _raw_segment(flow_velocity=120.0, conductivity=6.0)},
            {"segment": _raw_segment(conductivity=0.0)},
        ],
    }

    def run():
        tree = _expand(parse_group(raw))
        report = solve_group(build_model(tree), 0.317)
        return [
            (s.index, s.emf, s.resistance if math.isfinite(s.resistance) else "inf",
             s.current, s.emitted_power, s.dissipation)
            for s in report.segments
        ]

    first = run()
    for _ in range(5):
        assert run() == first


def _raw_segment(**changes) -> dict:
    data = {
        "magnetic_flux_density": 2.0,
        "electrode_spacing": 1.0,
        "flow_velocity": 100.0,
        "conductivity": 10.0,
        "cross_section_area": 0.5,
    }
    data.update(changes)
    return data

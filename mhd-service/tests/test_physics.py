"""物理内核的正确性判据：电学关系逐条钉死。"""
import math
from dataclasses import replace

import pytest

from mhd.induction import ChannelParams, internal_resistance, open_circuit_voltage
from mhd.load_solver import solve_load
from mhd.sweep import sweep_loads

# 基准通道：2 特斯拉磁场、1 米电极间距、100 米/秒流速，
# 电导率 10 S/m、截面积 0.5 平方米 -> 内阻 0.2 欧姆
BASELINE = ChannelParams(
    magnetic_flux_density=2.0,
    electrode_spacing=1.0,
    flow_velocity=100.0,
    conductivity=10.0,
    cross_section_area=0.5,
)


def test_benchmark_open_circuit_voltage_is_hundreds_of_volts():
    """手算基准：U_oc = B*d*v = 2*1*100 = 200 V，落在百伏量级。"""
    u_oc = open_circuit_voltage(BASELINE)
    assert u_oc == pytest.approx(200.0)
    assert 100.0 <= u_oc < 1000.0


def test_doubling_magnetic_field_doubles_open_circuit_voltage():
    u_base = open_circuit_voltage(BASELINE)
    u_doubled = open_circuit_voltage(
        replace(BASELINE, magnetic_flux_density=BASELINE.magnetic_flux_density * 2)
    )
    assert u_doubled == pytest.approx(2 * u_base)


def test_doubling_flow_velocity_doubles_open_circuit_voltage():
    u_base = open_circuit_voltage(BASELINE)
    u_doubled = open_circuit_voltage(
        replace(BASELINE, flow_velocity=BASELINE.flow_velocity * 2)
    )
    assert u_doubled == pytest.approx(2 * u_base)


def test_sweep_peak_lands_at_load_equal_to_internal_resistance():
    """功率曲线峰值必须落在 R_load = R_int，且峰值功率为 U_oc^2 / (4 R_int)。"""
    r_int = internal_resistance(BASELINE)
    u_oc = open_circuit_voltage(BASELINE)
    result = sweep_loads(BASELINE, load_min=0.01 * r_int, load_max=100 * r_int, points=200)
    assert result.peak.load_resistance == pytest.approx(r_int)
    assert result.peak.power == pytest.approx(u_oc * u_oc / (4 * r_int))


def test_power_drops_when_load_moves_from_r_int_to_twice_r_int():
    r_int = internal_resistance(BASELINE)
    at_match = solve_load(BASELINE, r_int)
    at_double = solve_load(BASELINE, 2 * r_int)
    assert at_double.power < at_match.power


def test_zero_conductivity_gives_zero_current_and_power():
    """电导率为零：内阻无穷大，电流为零，功率为零。"""
    params = replace(BASELINE, conductivity=0.0)
    assert math.isinf(internal_resistance(params))
    result = solve_load(params, load_resistance=1.0)
    assert result.current == 0.0
    assert result.power == 0.0


def test_zero_conductivity_sweep_does_not_blow_up():
    """电导率为零时扫描全程电流、功率为零，不会因除零崩掉。"""
    params = replace(BASELINE, conductivity=0.0)
    result = sweep_loads(params, load_min=0.0, load_max=10.0, points=50)
    assert all(point.current == 0.0 and point.power == 0.0 for point in result.curve)


def test_single_point_matches_sweep_curve_at_same_load():
    """单点核算与扫描曲线在同一负载处必须给出完全一致的结果。"""
    result = sweep_loads(BASELINE, load_min=0.05, load_max=1.0, points=96)
    for point in result.curve:
        single = solve_load(BASELINE, point.load_resistance)
        assert single.current == point.current
        assert single.power == point.power


def test_short_circuit_current_is_not_open_circuit_voltage():
    """短路电流 I_sc = U_oc / R_int 与开路电压是两个不同的量。"""
    r_int = internal_resistance(BASELINE)
    u_oc = open_circuit_voltage(BASELINE)
    shorted = solve_load(BASELINE, 0.0)
    assert shorted.current == pytest.approx(u_oc / r_int)
    assert shorted.power == 0.0
    assert shorted.open_circuit_voltage == pytest.approx(u_oc)

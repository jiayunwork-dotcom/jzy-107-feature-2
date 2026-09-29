"""输入合法性拦截：非法参数必须在计算前被带原因的错误打回。"""
import pytest

from mhd.validation import (
    ValidationError,
    validate_channel_params,
    validate_load_resistance,
    validate_sweep_range,
)

GOOD_CHANNEL = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}


def _channel(**overrides):
    return {**GOOD_CHANNEL, **overrides}


def test_valid_channel_passes():
    params = validate_channel_params(GOOD_CHANNEL)
    assert params.magnetic_flux_density == 2.0
    assert params.electrode_spacing == 1.0


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"magnetic_flux_density": -1.0}, "磁感应强度不能为负"),
        ({"electrode_spacing": 0.0}, "电极间距必须为正"),
        ({"electrode_spacing": -0.5}, "电极间距必须为正"),
        ({"flow_velocity": -100.0}, "流体速度不能为负"),
        ({"conductivity": -10.0}, "电导率不能为负"),
        ({"cross_section_area": 0.0}, "通道截面积必须为正"),
        ({"cross_section_area": -0.5}, "通道截面积必须为正"),
    ],
)
def test_invalid_channel_fields_rejected_with_reason(override, reason):
    with pytest.raises(ValidationError, match=reason):
        validate_channel_params(_channel(**override))


def test_missing_channel_field_rejected():
    incomplete = {k: v for k, v in GOOD_CHANNEL.items() if k != "conductivity"}
    with pytest.raises(ValidationError, match="缺少通道参数.*conductivity"):
        validate_channel_params(incomplete)


@pytest.mark.parametrize("bad", ["2.0", None, True, float("nan"), float("inf")])
def test_non_finite_or_non_numeric_field_rejected(bad):
    with pytest.raises(ValidationError):
        validate_channel_params(_channel(magnetic_flux_density=bad))


def test_zero_conductivity_is_allowed():
    params = validate_channel_params(_channel(conductivity=0.0))
    assert params.conductivity == 0.0


def test_negative_load_rejected():
    with pytest.raises(ValidationError, match="负载电阻不能为负"):
        validate_load_resistance(-1.0)


def test_zero_load_allowed_as_short_circuit():
    assert validate_load_resistance(0.0) == 0.0


def test_sweep_range_reversed_endpoints_rejected():
    with pytest.raises(ValidationError, match="上限必须大于下限"):
        validate_sweep_range({"load_min": 10.0, "load_max": 1.0, "points": 50})


def test_sweep_range_equal_endpoints_rejected():
    with pytest.raises(ValidationError, match="上限必须大于下限"):
        validate_sweep_range({"load_min": 1.0, "load_max": 1.0, "points": 50})


def test_sweep_negative_lower_bound_rejected():
    with pytest.raises(ValidationError, match="下限不能为负"):
        validate_sweep_range({"load_min": -1.0, "load_max": 10.0, "points": 50})


@pytest.mark.parametrize("points", [0, 1, -5])
def test_sweep_too_few_points_rejected(points):
    with pytest.raises(ValidationError, match="采样点个数至少"):
        validate_sweep_range({"load_min": 0.0, "load_max": 10.0, "points": points})


def test_sweep_non_integer_points_rejected():
    with pytest.raises(ValidationError, match="采样点个数必须是整数"):
        validate_sweep_range({"load_min": 0.0, "load_max": 10.0, "points": 2.5})


def test_sweep_excessive_points_rejected():
    with pytest.raises(ValidationError, match="采样点个数不能超过"):
        validate_sweep_range({"load_min": 0.0, "load_max": 10.0, "points": 10_000_000})


def test_sweep_valid_range_passes():
    load_min, load_max, points = validate_sweep_range(
        {"load_min": 0.0, "load_max": 10.0, "points": 100}
    )
    assert (load_min, load_max, points) == (0.0, 10.0, 100)

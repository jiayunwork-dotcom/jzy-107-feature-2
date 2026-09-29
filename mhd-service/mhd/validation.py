"""输入合法性拦截：在计算前把非法参数打回，错误消息里带原因。"""
from __future__ import annotations

import math

from .induction import ChannelParams

# 扫描采样点数目的合理范围：至少两点才能构成区间，上限防止滥用
MIN_SWEEP_POINTS = 2
MAX_SWEEP_POINTS = 100_000

_CHANNEL_FIELDS = (
    "magnetic_flux_density",
    "electrode_spacing",
    "flow_velocity",
    "conductivity",
    "cross_section_area",
)


class ValidationError(ValueError):
    """输入参数不合法。消息文本说明具体原因。"""


def _require_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"参数 {field} 必须是数值，收到: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"参数 {field} 必须是有限数值，收到: {value!r}")
    return result


def validate_channel_params(raw: object) -> ChannelParams:
    """校验并构造通道参数。任何一项不合法都在计算前抛出 ValidationError。"""
    if not isinstance(raw, dict):
        raise ValidationError("通道参数必须是 JSON 对象")
    missing = [field for field in _CHANNEL_FIELDS if field not in raw]
    if missing:
        raise ValidationError(f"缺少通道参数: {', '.join(missing)}")

    b = _require_finite_number(raw["magnetic_flux_density"], "magnetic_flux_density")
    if b < 0:
        raise ValidationError(f"磁感应强度不能为负，收到: {b}")

    d = _require_finite_number(raw["electrode_spacing"], "electrode_spacing")
    if d <= 0:
        raise ValidationError(f"电极间距必须为正，收到: {d}")

    v = _require_finite_number(raw["flow_velocity"], "flow_velocity")
    if v < 0:
        raise ValidationError(f"流体速度不能为负，收到: {v}")

    sigma = _require_finite_number(raw["conductivity"], "conductivity")
    if sigma < 0:
        raise ValidationError(f"电导率不能为负，收到: {sigma}")

    area = _require_finite_number(raw["cross_section_area"], "cross_section_area")
    if area <= 0:
        raise ValidationError(f"通道截面积必须为正，收到: {area}")

    return ChannelParams(
        magnetic_flux_density=b,
        electrode_spacing=d,
        flow_velocity=v,
        conductivity=sigma,
        cross_section_area=area,
    )


def validate_load_resistance(raw: object) -> float:
    """校验单个负载电阻。允许为零（短路，功率为零），不允许为负。"""
    load = _require_finite_number(raw, "load_resistance")
    if load < 0:
        raise ValidationError(f"负载电阻不能为负，收到: {load}")
    return load


def validate_sweep_range(raw: object) -> tuple[float, float, int]:
    """校验扫描请求：区间端点次序、采样点个数都要合理。"""
    if not isinstance(raw, dict):
        raise ValidationError("扫描参数必须是 JSON 对象")
    for field in ("load_min", "load_max", "points"):
        if field not in raw:
            raise ValidationError(f"缺少扫描参数: {field}")

    load_min = _require_finite_number(raw["load_min"], "load_min")
    if load_min < 0:
        raise ValidationError(f"负载区间下限不能为负，收到: {load_min}")

    load_max = _require_finite_number(raw["load_max"], "load_max")
    if load_max <= load_min:
        raise ValidationError(
            f"负载区间上限必须大于下限，收到: load_min={load_min}, load_max={load_max}"
        )

    points = raw["points"]
    if isinstance(points, bool) or not isinstance(points, int):
        raise ValidationError(f"采样点个数必须是整数，收到: {points!r}")
    if points < MIN_SWEEP_POINTS:
        raise ValidationError(f"采样点个数至少为 {MIN_SWEEP_POINTS}，收到: {points}")
    if points > MAX_SWEEP_POINTS:
        raise ValidationError(f"采样点个数不能超过 {MAX_SWEEP_POINTS}，收到: {points}")

    return load_min, load_max, points


def validate_profile_name(raw: object) -> str:
    """校验通道档名称：非空字符串。"""
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError(f"通道档名称必须是非空字符串，收到: {raw!r}")
    return raw

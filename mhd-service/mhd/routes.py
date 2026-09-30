"""HTTP 层：只负责请求收发与调度，计算内核在 mhd 包的其他模块里。

通道来源有三种（必须且只能给一个）：
* channel：内联单段参数（老写法，行为与返回字段完全不变）；
* profile：引用已登记通道档，名字在单段档、分组档间共用，
  取到哪种就按哪种算；
* group：内联分组结构（成员可再用 ref 引用通道档）。
"""
from __future__ import annotations

import math
from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request

from .group_solver import build_model, solve_group, sweep_group
from .group_tree import ExpandedTree, parse_group
from .induction import ChannelParams
from .load_solver import LoadResult, solve_load
from .profiles import (
    ProfileInUseError,
    ProfileNotFoundError,
    ProfileStore,
    StoredGroup,
)
from .sweep import sweep_loads
from .validation import (
    ValidationError,
    validate_channel_params,
    validate_load_resistance,
    validate_profile_name,
    validate_sweep_range,
)

bp = Blueprint("mhd", __name__)


def _store() -> ProfileStore:
    return current_app.extensions["mhd_profile_store"]


def _request_json() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return body


def _resolve_source(body: dict) -> ChannelParams | ExpandedTree:
    """channel/profile/group 三选一，返回单段参数或展开后的分组树。

    内联 group 里引用了不存在的档属于请求本身非法，按 400 打回；
    profile 指向不存在的档保持老行为 404。
    """
    present = [key for key in ("channel", "profile", "group") if key in body]
    if len(present) != 1:
        raise ValidationError(
            "请求必须且只能包含 channel（内联单段）、profile（引用通道档）、"
            "group（内联分组结构）三者之一"
        )
    kind = present[0]
    if kind == "channel":
        return validate_channel_params(body["channel"])
    if kind == "profile":
        name = validate_profile_name(body["profile"])
        return _store().resolve(name)
    try:
        return _store().expand_inline(parse_group(body["group"]))
    except ProfileNotFoundError as error:
        raise ValidationError(
            f"分组结构引用了不存在的通道档: {error.args[0]}"
        ) from None


def _load_from_body(body: dict) -> float | None:
    if "load_resistance" not in body:
        raise ValidationError("缺少负载电阻: load_resistance")
    return validate_load_resistance(body["load_resistance"])


def _finite_or_none(value: float) -> float | None:
    """math.inf 无法写进标准 JSON，内阻无穷大时序列化为 null。"""
    return None if math.isinf(value) else value


def _load_result_json(result: LoadResult) -> dict:
    return {
        "open_circuit_voltage": result.open_circuit_voltage,
        "internal_resistance": _finite_or_none(result.internal_resistance),
        "load_resistance": result.load_resistance,
        "current": result.current,
        "power": result.power,
    }


def _curve_point_json(result: LoadResult) -> dict:
    return {
        "load_resistance": result.load_resistance,
        "current": result.current,
        "power": result.power,
    }


def _segment_json(segment) -> dict:
    return {
        "index": segment.index,
        "path": list(segment.path),
        "ref": segment.ref,
        "emf": segment.emf,
        "resistance": _finite_or_none(segment.resistance),
        "current": segment.current,
        "voltage": segment.voltage,
        "emitted_power": segment.emitted_power,
        "dissipation": segment.dissipation,
    }


def _group_json(group) -> dict:
    return {
        "path": list(group.path),
        "connection": group.connection,
        "current": group.current,
        "voltage": group.voltage,
    }


def _network_result_json(result) -> dict:
    """分组通道单点结果：整体五字段与单段通道同名同义，外加逐段/逐组。"""
    return {
        "open_circuit_voltage": result.open_circuit_voltage,
        "internal_resistance": _finite_or_none(result.internal_resistance),
        "load_resistance": result.load_resistance,
        "current": result.current,
        "power": result.power,
        "segments": [_segment_json(segment) for segment in result.segments],
        "groups": [_group_json(group) for group in result.groups],
    }


def _network_curve_point_json(result) -> dict:
    return {
        "load_resistance": result.load_resistance,
        "current": result.current,
        "power": result.power,
    }


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.post("/channels")
def create_channel():
    body = _request_json()
    if "name" not in body:
        raise ValidationError("缺少通道档名称: name")
    name = validate_profile_name(body["name"])

    if "group" in body:
        if "channel" in body:
            raise ValidationError("channel 与 group 只能给一个")
        stored = _save_group_profile(name, body["group"])
        return jsonify({"name": name, "group": stored.raw}), 201

    # 老写法：{"name": ..., "channel": {...}}；原实现还允许把五项参数平铺
    params = validate_channel_params(body.get("channel", body))
    _store().save_channel(name, params)
    return jsonify({"name": name, "channel": asdict(params)}), 201


def _save_group_profile(name: str, raw_group: object) -> StoredGroup:
    """保存分组档：悬空引用按 400 打回（保存时拒收），而非 404。"""
    try:
        return _store().save_group(name, raw_group)
    except ProfileNotFoundError as error:
        missing = error.args[0]
        raise ValidationError(f"分组档引用了不存在的通道档: {missing}") from None


@bp.get("/channels")
def list_channels():
    # 原有 profiles 字段（全部档案名）保持不变，group_profiles 标出分组档
    all_names, group_names = _store().listing()
    return jsonify({"profiles": all_names, "group_profiles": group_names})


@bp.get("/channels/<name>")
def get_channel(name: str):
    entry = _store().get_entry(name)
    if isinstance(entry, StoredGroup):
        return jsonify({"name": name, "group": entry.raw})
    return jsonify({"name": name, "channel": asdict(entry)})


@bp.delete("/channels/<name>")
def delete_channel(name: str):
    _store().delete(name)
    return "", 204


@bp.post("/evaluate")
def evaluate():
    """单点核算：给定通道（单段或分组）与一个负载，返回电压、内阻、电流、功率。"""
    body = _request_json()
    source = _resolve_source(body)
    load = _load_from_body(body)
    if isinstance(source, ChannelParams):
        return jsonify(_load_result_json(solve_load(source, load)))
    result = solve_group(build_model(source), load)
    return jsonify(_network_result_json(result))


@bp.post("/sweep")
def sweep():
    """负载扫描：对负载区间采样，返回功率曲线与峰值点。"""
    body = _request_json()
    source = _resolve_source(body)
    load_min, load_max, points = validate_sweep_range(body)

    if isinstance(source, ChannelParams):
        result = sweep_loads(source, load_min, load_max, points)
        return jsonify(
            {
                "open_circuit_voltage": result.peak.open_circuit_voltage,
                "internal_resistance": _finite_or_none(result.peak.internal_resistance),
                "curve": [_curve_point_json(point) for point in result.curve],
                "peak": _curve_point_json(result.peak),
            }
        )

    result = sweep_group(build_model(source), load_min, load_max, points)
    return jsonify(
        {
            "open_circuit_voltage": result.open_circuit_voltage,
            "internal_resistance": _finite_or_none(result.internal_resistance),
            "curve": [_network_curve_point_json(point) for point in result.curve],
            "peak": _network_curve_point_json(result.peak),
            # 逐段/逐组详情挂在峰值点上，与单点核算同负载处逐位一致
            "segments": [_segment_json(segment) for segment in result.peak.segments],
            "groups": [_group_json(group) for group in result.peak.groups],
        }
    )


def register_error_handlers(app):
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(ProfileNotFoundError)
    def handle_profile_not_found(error):
        return jsonify({"error": f"通道档不存在: {error.args[0]}"}), 404

    @app.errorhandler(ProfileInUseError)
    def handle_profile_in_use(error):
        return (
            jsonify(
                {
                    "error": (
                        f"通道档 {error.name!r} 正被其它分组档引用，不能删除；"
                        f"引用者: {', '.join(error.referrers)}"
                    ),
                    "name": error.name,
                    "referrers": error.referrers,
                }
            ),
            409,
        )

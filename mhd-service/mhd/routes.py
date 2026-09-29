"""HTTP 层：只负责请求收发与调度，计算内核在 mhd 包的其他模块里。

- 老入口（``channel`` 内联 / ``profile`` 引用单段档）的路径、字段、
  状态码全部不变；
- 新增第三种通道来源：``group`` 内联分组树；``profile`` 指向分组档时
  自动按分组口径返回逐段、逐组明细；
- 分组档用 POST /channels 的 ``group`` 字段登记，与单段档同名字空间，
  GET /channels 额外给出 groups 列表（profiles 字段保持原义）。
"""
from __future__ import annotations

import math
from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request

from .circuit import (
    ChannelSolution,
    GroupResult,
    OperatingPoint,
    SegmentResult,
    solve_channel,
)
from .groups import ReferenceCycleError, SegmentEntry
from .group_sweep import sweep_group
from .group_validation import validate_group_tree
from .induction import ChannelParams
from .load_solver import LoadResult, solve_load
from .profiles import (
    DanglingReferenceError,
    GroupLimitError,
    ProfileInUseError,
    ProfileNotFoundError,
    ProfileStore,
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


def _finite_or_none(value: float) -> float | None:
    """math.inf 无法写进标准 JSON，内阻无穷大时序列化为 null。"""
    return None if math.isinf(value) else value


# --------------------------------------------------------- 老口径结果

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


# --------------------------------------------------------- 分组结果

def _segment_json(segment: SegmentResult) -> dict:
    return {
        "path": segment.path,
        "id": segment.node_id,
        "open_circuit_voltage": segment.emf,
        "internal_resistance": _finite_or_none(segment.internal_resistance),
        "current": segment.current,
        "generated_power": segment.generated_power,
        "dissipation": segment.dissipation,
        "terminal_power": segment.terminal_power,
    }


def _group_json(group: GroupResult) -> dict:
    return {
        "path": group.path,
        "id": group.node_id,
        "type": group.kind,
        "current": group.current,
        "voltage": group.voltage,
        "power": group.power,
    }


def _operating_point_json(point: OperatingPoint) -> dict:
    return {
        "open_circuit_voltage": point.open_circuit_voltage,
        "internal_resistance": _finite_or_none(point.internal_resistance),
        "load_resistance": point.load_resistance,
        "current": point.load_current,
        "power": point.load_power,
        "segments": [_segment_json(s) for s in point.segments],
        "groups": [_group_json(g) for g in point.groups],
    }


# --------------------------------------------------------- 通道来源

class _SingleChannel:
    """老口径的单段通道（内联或单段档）。"""

    def __init__(self, params: ChannelParams):
        self.params = params


class _GroupChannel:
    def __init__(self, solution: ChannelSolution):
        self.solution = solution


def _resolve_source(body: dict):
    """解析请求里的通道来源。

    三选一：``channel``（内联单段，老口径）、``group``（内联分组树）、
    ``profile``（按名字取档；单段档走老口径，分组档走分组口径）。
    """
    keys = [key for key in ("profile", "channel", "group") if key in body]
    if len(keys) != 1:
        raise ValidationError(
            "请求必须且只能包含 profile、channel、group 三者之一"
        )
    key = keys[0]
    if key == "channel":
        return _SingleChannel(validate_channel_params(body["channel"]))
    if key == "group":
        root, _refs = validate_group_tree(body["group"])
        try:
            root = _store().resolve_inline(root)
        except ProfileNotFoundError as error:
            raise ValidationError(f"分组引用了不存在的通道档: {error.args[0]}") from None
        except DanglingReferenceError as error:
            raise ValidationError(
                f"分组引用了不存在的通道档: {', '.join(error.missing)}"
            ) from None
        except (ReferenceCycleError, GroupLimitError) as error:
            raise ValidationError(f"分组引用不合法：{error}") from None
        return _GroupChannel(solve_channel(root))

    name = validate_profile_name(body["profile"])
    entry = _store().get_entry(name)
    if isinstance(entry, SegmentEntry):
        return _SingleChannel(entry.params)
    return _GroupChannel(solve_channel(_store().resolve(name)))


# --------------------------------------------------------- 路由

@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.post("/channels")
def create_channel():
    body = _request_json()
    if "name" not in body:
        raise ValidationError("缺少通道档名称: name")
    name = validate_profile_name(body["name"])

    keys = [key for key in ("channel", "group") if key in body]
    if len(keys) != 1:
        raise ValidationError("登记通道档必须且只能包含 channel 或 group 之一")

    if keys[0] == "channel":
        params = validate_channel_params(body["channel"])
        _store().save(name, params)
        return jsonify({"name": name, "channel": asdict(params)}), 201

    root, refs = validate_group_tree(body["group"])
    try:
        _store().save_group(name, root, body["group"], refs)
    except DanglingReferenceError as error:
        raise ValidationError(
            f"分组档 {name} 引用了不存在的通道档: {', '.join(error.missing)}"
        ) from None
    except ReferenceCycleError as error:
        raise ValidationError(f"保存被拒：{error}") from None
    except GroupLimitError as error:
        raise ValidationError(f"保存被拒：{error}") from None
    return jsonify({"name": name, "group": body["group"]}), 201


@bp.get("/channels")
def list_channels():
    store = _store()
    # profiles 字段保持原义：全部档名；groups 额外标出哪些是分组档
    return jsonify(
        {"profiles": store.names(), "groups": store.group_names()}
    )


@bp.get("/channels/<name>")
def get_channel(name: str):
    entry = _store().get_entry(name)
    if isinstance(entry, SegmentEntry):
        return jsonify({"name": name, "channel": asdict(entry.params)})
    return jsonify({"name": name, "group": entry.raw})


@bp.delete("/channels/<name>")
def delete_channel(name: str):
    _store().delete(name)
    return "", 204


@bp.post("/evaluate")
def evaluate():
    """单点核算：单段通道走老口径，分组通道返回逐段、逐组明细。

    load_resistance 为 null 仅在分组口径下表示负载开路：对外电流为零，
    并联支路间的环流仍照常给出（这是分组口径最关心的观测量之一）。
    老单段口径沿用数值校验，null 会被打回。
    """
    body = _request_json()
    source = _resolve_source(body)
    if "load_resistance" not in body:
        raise ValidationError("缺少负载电阻: load_resistance")

    if isinstance(source, _SingleChannel):
        load = validate_load_resistance(body["load_resistance"])
        return jsonify(_load_result_json(solve_load(source.params, load)))

    if body["load_resistance"] is None:
        load = math.inf  # 开路：ChannelSolution 在 R_eq 有限时 I_load 极限为 0
    else:
        load = validate_load_resistance(body["load_resistance"])
    point = source.solution.at(load)
    payload = _operating_point_json(point)
    if math.isinf(load):
        # 开路没有负载电阻值，序列化回 null
        payload["load_resistance"] = None
    return jsonify(payload)


@bp.post("/sweep")
def sweep():
    """负载扫描：单段通道走老口径，分组通道带逐段、逐组的峰值展开。"""
    body = _request_json()
    source = _resolve_source(body)
    load_min, load_max, points = validate_sweep_range(body)

    if isinstance(source, _SingleChannel):
        result = sweep_loads(source.params, load_min, load_max, points)
        return jsonify(
            {
                "open_circuit_voltage": result.peak.open_circuit_voltage,
                "internal_resistance": _finite_or_none(result.peak.internal_resistance),
                "curve": [_curve_point_json(point) for point in result.curve],
                "peak": _curve_point_json(result.peak),
            }
        )

    result = sweep_group(source.solution, load_min, load_max, points)
    peak = _operating_point_json(result.peak)
    return jsonify(
        {
            "open_circuit_voltage": result.solution.open_circuit_voltage,
            "internal_resistance": _finite_or_none(result.solution.internal_resistance),
            "curve": [
                {
                    "load_resistance": point.load_resistance,
                    "current": point.current,
                    "power": point.power,
                }
                for point in result.curve
            ],
            "peak": peak,
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
        return jsonify(
            {
                "error": (
                    f"通道档 {error.name} 正被分组档引用，不能删除；"
                    f"引用方: {', '.join(error.referrers)}"
                ),
                "referrers": error.referrers,
            }
        ), 409

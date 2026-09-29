"""HTTP 层：只负责请求收发与调度，计算内核在 mhd 包的其他模块里。"""
from __future__ import annotations

import math
from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request

from .induction import ChannelParams
from .load_solver import LoadResult, solve_load
from .profiles import ProfileNotFoundError, ProfileStore
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


def _resolve_channel(body: dict) -> ChannelParams:
    """请求里要么给 profile 名称引用通道档，要么给 channel 内联参数。"""
    has_profile = "profile" in body
    has_channel = "channel" in body
    if has_profile == has_channel:
        raise ValidationError("请求必须且只能包含 profile 或 channel 之一")
    if has_profile:
        name = validate_profile_name(body["profile"])
        return _store().get(name)
    return validate_channel_params(body["channel"])


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


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.post("/channels")
def create_channel():
    body = _request_json()
    if "name" not in body:
        raise ValidationError("缺少通道档名称: name")
    name = validate_profile_name(body["name"])
    params = validate_channel_params(body.get("channel", body))
    _store().save(name, params)
    return jsonify({"name": name, "channel": asdict(params)}), 201


@bp.get("/channels")
def list_channels():
    return jsonify({"profiles": _store().names()})


@bp.get("/channels/<name>")
def get_channel(name: str):
    params = _store().get(name)
    return jsonify({"name": name, "channel": asdict(params)})


@bp.delete("/channels/<name>")
def delete_channel(name: str):
    _store().delete(name)
    return "", 204


@bp.post("/evaluate")
def evaluate():
    """单点核算：给定通道与一个负载电阻，返回电压、内阻、电流、功率。"""
    body = _request_json()
    params = _resolve_channel(body)
    if "load_resistance" not in body:
        raise ValidationError("缺少负载电阻: load_resistance")
    load = validate_load_resistance(body["load_resistance"])
    return jsonify(_load_result_json(solve_load(params, load)))


@bp.post("/sweep")
def sweep():
    """负载扫描：对负载区间采样，返回功率曲线与峰值点。"""
    body = _request_json()
    params = _resolve_channel(body)
    load_min, load_max, points = validate_sweep_range(body)
    result = sweep_loads(params, load_min, load_max, points)
    return jsonify(
        {
            "open_circuit_voltage": result.peak.open_circuit_voltage,
            "internal_resistance": _finite_or_none(result.peak.internal_resistance),
            "curve": [_curve_point_json(point) for point in result.curve],
            "peak": _curve_point_json(result.peak),
        }
    )


def register_error_handlers(app):
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(ProfileNotFoundError)
    def handle_profile_not_found(error):
        return jsonify({"error": f"通道档不存在: {error.args[0]}"}), 404

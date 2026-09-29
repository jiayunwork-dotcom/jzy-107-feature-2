"""负载扫描：对一段负载电阻区间采样，给出功率曲线并定位峰值。

扫描与单点核算共用 load_solver.solve_load，保证同一负载处两种口径
算出的电流、功率完全一致。经典最大功率传输点在 R_load = R_int，
若内阻落在扫描区间内，会把它补进采样点，使曲线峰值恰好落在该处。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .induction import ChannelParams, internal_resistance
from .load_solver import LoadResult, solve_load


@dataclass(frozen=True)
class SweepResult:
    """一次负载扫描的结果：整条曲线加上峰值点。"""

    curve: tuple[LoadResult, ...]  # 按负载电阻升序排列
    peak: LoadResult               # 曲线上功率最大的点


def _sample_loads(
    load_min: float, load_max: float, points: int, r_int: float
) -> list[float]:
    """在 [load_min, load_max] 上均匀取 points 个采样点。

    内阻落在区间内部时把它补为一个采样点，让峰值精确落在
    R_load = R_int 处，而不是离它最近的某个网格点。
    """
    step = (load_max - load_min) / (points - 1)
    loads = [load_min + i * step for i in range(points)]
    loads[-1] = load_max  # 消除浮点累计误差，保证端点精确
    if math.isfinite(r_int) and load_min < r_int < load_max:
        tol = 1e-12 * max(1.0, abs(r_int))
        if not any(abs(x - r_int) <= tol for x in loads):
            loads.append(r_int)
            loads.sort()
    return loads


def sweep_loads(
    params: ChannelParams, load_min: float, load_max: float, points: int
) -> SweepResult:
    """扫描负载区间，返回功率曲线与峰值点。"""
    r_int = internal_resistance(params)
    loads = _sample_loads(load_min, load_max, points, r_int)
    curve = tuple(solve_load(params, load) for load in loads)
    peak = max(curve, key=lambda result: result.power)
    return SweepResult(curve=curve, peak=peak)

"""分组通道的负载扫描。

与单段通道一样，扫描与单点核算共用同一次求解（``circuit.solve_channel``
建网后只解两次矩阵）：每个采样负载只是 ChannelSolution 上的一次仿射
取值，因此曲线上任一点与同负载的单点核算逐位一致。等效内阻有限且
落在区间内部时把它补为采样点，使峰值精确落在 R_load = R_int。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .circuit import ChannelSolution, OperatingPoint
from .sweep import _sample_loads  # 复用采样与补点逻辑


@dataclass(frozen=True)
class CurvePoint:
    load_resistance: float
    current: float
    power: float


@dataclass(frozen=True)
class GroupSweepResult:
    solution: ChannelSolution
    curve: tuple[CurvePoint, ...]
    peak: OperatingPoint


def sweep_group(
    solution: ChannelSolution, load_min: float, load_max: float, points: int
) -> GroupSweepResult:
    """在已求解的分组通道上扫描负载区间，曲线点与峰值共用同一解。

    曲线点只取负载电流与功率（由 U_oc、R_eq 直接算出），不为每个采样
    点构造逐段明细，所以十万点扫描也只是一次列表推导；逐段明细只在
    峰值点展开一次。
    """
    loads = _sample_loads(load_min, load_max, points, solution.internal_resistance)

    if math.isinf(solution.internal_resistance):
        # 两极不连通：任何有限负载下电流、功率都是零；峰值取首点
        curve = tuple(
            CurvePoint(load_resistance=load, current=0.0, power=0.0)
            for load in loads
        )
        return GroupSweepResult(
            solution=solution, curve=curve, peak=solution.at(loads[0])
        )

    u_oc = solution.open_circuit_voltage
    r_eq = solution.internal_resistance
    curve_points: list[CurvePoint] = []
    for load in loads:
        current = u_oc / (r_eq + load)
        curve_points.append(
            CurvePoint(
                load_resistance=load,
                current=current,
                power=current * current * load,
            )
        )
    curve = tuple(curve_points)
    peak_index = max(range(len(curve)), key=lambda i: curve[i].power)
    return GroupSweepResult(
        solution=solution,
        curve=curve,
        peak=solution.at(curve[peak_index].load_resistance),
    )

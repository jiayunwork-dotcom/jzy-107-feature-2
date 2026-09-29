"""负载求解：给定单个负载电阻时的回路电流与负载提取功率。

    回路电流  I = U_oc / (R_int + R_load)
    提取功率  P = I^2 * R_load
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .induction import ChannelParams, internal_resistance, open_circuit_voltage


@dataclass(frozen=True)
class LoadResult:
    """单个负载点上的核算结果。"""

    open_circuit_voltage: float   # 开路电压，伏
    internal_resistance: float    # 通道内阻，欧姆（电导率为零时为 math.inf）
    load_resistance: float        # 负载电阻，欧姆
    current: float                # 回路电流，安
    power: float                  # 负载提取功率，瓦


def solve_load(params: ChannelParams, load_resistance: float) -> LoadResult:
    """求解通道在单个负载电阻下的电流与提取功率。"""
    u_oc = open_circuit_voltage(params)
    r_int = internal_resistance(params)
    if math.isinf(r_int):
        # 电导率为零：内阻无穷大，回路断开，电流与功率都为零
        current = 0.0
    else:
        current = u_oc / (r_int + load_resistance)
    power = current * current * load_resistance
    return LoadResult(
        open_circuit_voltage=u_oc,
        internal_resistance=r_int,
        load_resistance=load_resistance,
        current=current,
        power=power,
    )

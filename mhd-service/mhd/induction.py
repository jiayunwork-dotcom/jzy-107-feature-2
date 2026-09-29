"""感应计算：法拉第式 MHD 通道的开路电压与内阻。

物理模型（体感应电动势，非霍尔电压）：
    开路电压  U_oc  = B * d * v
    通道内阻  R_int = d / (sigma * A)
其中 B 为磁感应强度，d 为电极间距，v 为流体速度，
sigma 为流体电导率，A 为通道截面积。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelParams:
    """一条法拉第式 MHD 通道的物理参数。"""

    magnetic_flux_density: float  # B，磁感应强度，特斯拉
    electrode_spacing: float      # d，电极间距，米
    flow_velocity: float          # v，流体速度，米/秒
    conductivity: float           # sigma，流体电导率，西门子/米
    cross_section_area: float     # A，通道截面积，平方米


def open_circuit_voltage(params: ChannelParams) -> float:
    """开路感应电压 U_oc = B * d * v（伏）。"""
    return (
        params.magnetic_flux_density
        * params.electrode_spacing
        * params.flow_velocity
    )


def internal_resistance(params: ChannelParams) -> float:
    """通道内阻 R_int = d / (sigma * A)（欧姆）。

    电导率为零时内阻趋于无穷大，返回 math.inf，由上层据此把电流压到零。
    """
    if params.conductivity == 0.0:
        return math.inf
    return params.electrode_spacing / (
        params.conductivity * params.cross_section_area
    )

"""分段分组通道的单点核算与负载扫描。

在 network_model.NetworkModel 统一网络解之上薄薄一层：

* solve_group：展开后的分组树 + 一个负载（None 表示开路）-> NetworkResult；
* sweep_group：建一次网、取一次仿射系数，随后每个采样点只做标量一次式
  求值，因此扫描曲线上任一点与同负载的单点核算逐位一致。
"""
from __future__ import annotations

from dataclasses import dataclass

from .group_tree import ExpandedTree
from .network_model import NetworkModel, NetworkResult
from .sweep import sample_loads


@dataclass(frozen=True)
class GroupSweepResult:
    """一次分组通道扫描：整体量、逐段/逐组详情（取峰值点）与曲线。"""

    open_circuit_voltage: float
    internal_resistance: float       # math.inf 表示整网全断
    curve: tuple[NetworkResult, ...]
    peak: NetworkResult              # 曲线上功率最大的点


def build_model(tree: ExpandedTree) -> NetworkModel:
    """把展开后的分组树交给统一网络求解器建模。"""
    return NetworkModel(tree)


def solve_group(model: NetworkModel, load_resistance: float | None) -> NetworkResult:
    """单点核算：给定网络模型与负载（None=开路），返回整网与逐段结果。"""
    return model.evaluate(load_resistance)


def sweep_group(
    model: NetworkModel, load_min: float, load_max: float, points: int
) -> GroupSweepResult:
    """扫描负载区间：同一模型下逐点求值，峰值为曲线上功率最大点。"""
    loads = sample_loads(
        load_min, load_max, points, model.equivalent_resistance
    )
    curve = tuple(model.evaluate(load) for load in loads)
    peak = max(curve, key=lambda result: result.power)
    return GroupSweepResult(
        open_circuit_voltage=model.equivalent_emf,
        internal_resistance=model.equivalent_resistance,
        curve=curve,
        peak=peak,
    )

"""分段通道网络的统一电路建模与求解。

不过串联/并联各写一套等效公式，而是把整棵分组树展开成一张电路网络，
用节点电压法（Nodal Analysis）统一求解：

* 每一段是一条戴维南支路：电动势 E = B*d*v，内阻 r = d/(sigma*A)；
  求解时转成诺顿形式（电导 g = 1/r 并联电流源 g*E）压到节点导纳矩阵上。
* 电导率为零的段 g = 0，就是一条开路支路：不参与导纳矩阵的连通，
  天然产生悬空节点，不需要 1/0 之类的运算。
* 串联 = 成员首尾相接（中间分配接线节点），并联 = 成员共接两端，
  组可以套组，对求解器来说只是“怎么把支路接到节点上”的区别。

求解策略（线性网络叠加原理）：整网只做两次基准求解，之后任意负载点
都是标量一次式求值，单点核算与整条扫描共用同一组系数：

1. 有源开路解 v_oc：保留全部电动势、负载开路，解节点电压与各支路电流；
2. 无源单位注入解 v_u：电动势全部置零，在整网正端注入 -1、负端注入 +1
   （即负载从正端取走 1 A），解节点电压与各支路电流。

于是负载电流为 I 时：支路量 x = x_oc + x_u * I，节点电压同理。
整网的戴维南等效量：
    U_eq = V0_oc - V1_oc          （等效电动势）
    R_eq = -(V0_u - V1_u)         （等效内阻）
    I    = U_eq / (R_eq + R_load) （R_load 为 null 表示开路，I = 0）
开路（I = 0）时并联支路间的环流直接体现在各段 x_oc 里。

病态与悬空节点的处理：
* 只用导通支路（g > 0）做并查集分连通块，开路支路不缝合节点；
* 每个连通块各自接地（含整网正端节点 0 的块以 0 为参考，其余块任选
  一个节点接地），在块内解一个对称正定的接地拉普拉斯矩阵，孤立节点、
  悬空子块都不会产生奇异方程；
* 串联链被开路段切开时，用开路约束 V_p - V_q = E 沿开路边传播各块的
  电平，使“整网全断”时等效电动势仍等于链上各电动势之和；并联着悬空
  的开路段不约束任何人，效果与“直接去掉这段”一致。

选型理由与考虑过的替代方案见提交说明（README「网络求解选型」一节）。
"""
from __future__ import annotations

# numpy / OpenBLAS 必须在 import numpy 前固定单线程，避免多线程归约顺序
# 漂移导致同一请求重复提交结果不逐位相同。
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import math
from dataclasses import dataclass

import numpy as np

from .group_tree import ExpandedTree, GroupNode, SegmentNode
from .induction import internal_resistance, open_circuit_voltage

# 浮点噪声下限保护：理论上非负的量出现 -1e-16 量级时归零
_NEG_ZERO_TOL = 1e-12


@dataclass(frozen=True)
class SegmentReport:
    """一段在某个负载下的核算结果。"""

    index: int                  # 前序序号，从 0 起
    path: tuple[int, ...]       # 从根组到该段的成员下标路径
    ref: str | None             # 该段来自哪个通道档（内联段为 None）
    emf: float                  # 该段感应电动势，伏
    resistance: float           # 该段内阻，欧（电导率为零时 math.inf）
    current: float              # 实际流过的电流，安；正=顺电动势发电，负=被倒灌
    voltage: float              # 该段两端电压（正端减负端），伏
    emitted_power: float        # 该段发出的功率 E*I，瓦（倒灌时为负）
    dissipation: float          # 该段内阻上的损耗 I^2*r，瓦


@dataclass(frozen=True)
class GroupReport:
    """一个组在某个负载下的端口量。"""

    path: tuple[int, ...]       # 根组为空元组
    connection: str             # series / parallel
    current: float              # 从该组正端送出的电流，安
    voltage: float              # 该组两端电压（正端减负端），伏


@dataclass(frozen=True)
class NetworkResult:
    """整棵分组通道在单个负载点上的核算结果。"""

    open_circuit_voltage: float                 # 等效电动势 U_eq，伏
    internal_resistance: float                  # 等效内阻 R_eq，欧（全断为 inf）
    load_resistance: float | None               # 负载电阻，欧；null 表示开路
    current: float                              # 负载电流 I，安
    power: float                                # 负载功率，瓦
    segments: tuple[SegmentReport, ...]
    groups: tuple[GroupReport, ...]


@dataclass(frozen=True)
class _Edge:
    p: int
    q: int
    emf: float
    g: float          # 电导；0 表示开路
    path: tuple[int, ...]
    ref: str | None


class _DSU:
    """只按导通支路缝合节点的并查集。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 让编号小的块当代表，保证分块与接地选择可重复
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


class NetworkModel:
    """分组树对应的电路网络：建一次网，得到全部仿射系数，供任意负载求值。"""

    def __init__(self, tree: ExpandedTree) -> None:
        self._tree = tree

        edges: list[_Edge] = []
        group_terms: dict[tuple[int, ...], tuple[int, int]] = {}
        self._next_node = 2  # 0 = 整网正端，1 = 整网负端

        def new_node() -> int:
            node = self._next_node
            self._next_node += 1
            return node

        def build(node: GroupNode | SegmentNode, p: int, q: int,
                  path: tuple[int, ...]) -> None:
            if isinstance(node, SegmentNode):
                emf = open_circuit_voltage(node.params)
                r = internal_resistance(node.params)
                g = 0.0 if math.isinf(r) else 1.0 / r
                edges.append(_Edge(p=p, q=q, emf=emf, g=g, path=path, ref=node.ref))
                return

            group_terms[path] = (p, q)
            children = node.children
            if node.connection == "parallel":
                for i, child in enumerate(children):
                    build(child, p, q, path + (i,))
                return

            # 串联：成员首尾相接，相邻成员之间分配一个接线节点
            left = p
            for i, child in enumerate(children):
                right = q if i == len(children) - 1 else new_node()
                build(child, left, right, path + (i,))
                left = right

        build(tree.root, 0, 1, ())
        n_nodes = self._next_node

        live = [e for e in edges if e.g > 0.0]
        open_edges = [e for e in edges if e.g <= 0.0]

        dsu = _DSU(n_nodes)
        for e in live:
            dsu.union(e.p, e.q)
        comp_of = [dsu.find(i) for i in range(n_nodes)]
        # 块内成员（节点按编号排序，保证行列顺序确定）
        comp_nodes: dict[int, list[int]] = {}
        for node in range(n_nodes):
            comp_nodes.setdefault(comp_of[node], []).append(node)

        v_oc = self._solve_active(comp_nodes, comp_of, live, n_nodes)
        v_oc = self._propagate_open_levels(
            v_oc, comp_nodes, comp_of, open_edges
        )

        root_comp = comp_of[0]
        connected = comp_of[1] == root_comp
        if connected:
            v_u = self._solve_unit(comp_nodes[root_comp], live, n_nodes)
            r_eq = float(v_u[1] - v_u[0])
            if r_eq < 0.0 and r_eq > -_NEG_ZERO_TOL:
                r_eq = 0.0
        else:
            v_u = np.zeros(n_nodes)
            r_eq = math.inf

        u_eq = float(v_oc[0] - v_oc[1])

        # 支路电流（以 p->q 为正的诺顿支路电流 i = g(Vp-Vq) - gE），
        # 段报告电流取其相反方向（顺电动势流出正端为正）。
        edge_i_oc: list[float] = []
        edge_i_u: list[float] = []
        for e in edges:
            if e.g <= 0.0:
                edge_i_oc.append(0.0)
                edge_i_u.append(0.0)
            else:
                edge_i_oc.append(
                    e.g * (v_oc[e.p] - v_oc[e.q]) - e.g * e.emf
                )
                edge_i_u.append(e.g * (v_u[e.p] - v_u[e.q]))

        self._edges = tuple(edges)
        self._group_terms = group_terms
        self._u_eq = u_eq
        self._r_eq = r_eq
        self._v_oc = v_oc
        self._v_u = v_u
        self._edge_i_oc = tuple(edge_i_oc)
        self._edge_i_u = tuple(edge_i_u)

    # ------------------------------------------------------------------
    # 节点电压方程
    # ------------------------------------------------------------------
    @staticmethod
    def _ground(comp_id: int, nodes: list[int]) -> int:
        """块内参考节点：含整网正端（0 号）的块以 0 为地，其余取编号最小者。"""
        return 0 if 0 in nodes else nodes[0]

    def _solve_active(
        self, comp_nodes, comp_of, live, n_nodes: int
    ) -> np.ndarray:
        """有源开路解：每个连通块独立解接地拉普拉斯方程。"""
        voltage = np.zeros(n_nodes)
        for comp_id, nodes in comp_nodes.items():
            ground = self._ground(comp_id, nodes)
            unknowns = [n for n in nodes if n != ground]
            idx = {n: k for k, n in enumerate(unknowns)}
            size = len(unknowns)
            mat = np.zeros((size, size))
            rhs = np.zeros(size)
            for e in live:
                if comp_of[e.p] != comp_id:
                    continue
                gp = e.g * e.emf  # 诺顿电流源 g*E，方向 p->q
                if e.p != ground:
                    k = idx[e.p]
                    mat[k, k] += e.g
                    rhs[k] += gp
                if e.q != ground:
                    k = idx[e.q]
                    mat[k, k] += e.g
                    rhs[k] -= gp
                if e.p != ground and e.q != ground:
                    mat[idx[e.p], idx[e.q]] -= e.g
                    mat[idx[e.q], idx[e.p]] -= e.g
            if size:
                voltage[unknowns] = np.linalg.solve(mat, rhs)
        return voltage

    def _solve_unit(self, root_nodes: list[int], live, n_nodes: int) -> np.ndarray:
        """无源单位负载注入解：只解含整网正端的连通块。

        负载从正端取走 1 A（正端节点 0 注入 -1，负端节点 1 注入 +1）。
        节点 0 是参考地，其注入由地吸收，只需在节点 1 的右端项 +1。
        其余悬空块电位保持 0：它们既看不到电动势也看不到注入。
        """
        voltage = np.zeros(n_nodes)
        ground = 0
        root_set = set(root_nodes)
        unknowns = [n for n in root_nodes if n != ground]
        idx = {n: k for k, n in enumerate(unknowns)}
        size = len(unknowns)
        mat = np.zeros((size, size))
        rhs = np.zeros(size)
        for e in live:
            # 块外的边不接进块内节点，只导纳、无激励
            if e.p not in root_set and e.q not in root_set:
                continue
            if e.p != ground:
                mat[idx[e.p], idx[e.p]] += e.g
            if e.q != ground:
                mat[idx[e.q], idx[e.q]] += e.g
            if e.p != ground and e.q != ground:
                mat[idx[e.p], idx[e.q]] -= e.g
                mat[idx[e.q], idx[e.p]] -= e.g
        if 1 in idx:
            rhs[idx[1]] += 1.0
        if size:
            voltage[unknowns] = np.linalg.solve(mat, rhs)
        return voltage

    def _propagate_open_levels(
        self, voltage: np.ndarray, comp_nodes, comp_of, open_edges
    ) -> np.ndarray:
        """沿开路边传播各连通块的统一电平 L，满足开路约束 V_p - V_q = E。

        已解出的块内电位差是确定的，块之间差一个常数电平。从整网正端
        所在块 BFS：对开路边 (p,q,E) 有
            L[comp(p)] - L[comp(q)] = E - (V_p - V_q)
        串联链被开路段切开时，这条边是两块之间唯一的桥，电平被唯一确定；
        并联悬空的多余开路边不改变可达块的电平，效果等同删掉该段。
        """
        # 块邻接表：(邻居块, 有向边约束左端块, 权重 w)
        adjacency: dict[int, list[tuple[int, int, float]]] = {}
        for e in open_edges:
            cp, cq = comp_of[e.p], comp_of[e.q]
            if cp == cq:
                continue  # 块内开路（两端被别处导通），不产生约束
            w = e.emf - float(voltage[e.p] - voltage[e.q])
            adjacency.setdefault(cp, []).append((cq, cp, w))
            adjacency.setdefault(cq, []).append((cp, cp, w))

        level = {comp: 0.0 for comp in comp_nodes}
        root_comp = comp_of[0]
        seen = {root_comp}
        queue = [root_comp]
        while queue:
            cur = queue.pop(0)
            for nxt, left_comp, w in adjacency.get(cur, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                # L[left] - L[other] = w，已知一端推另一端
                if left_comp == cur:
                    level[nxt] = level[cur] - w
                else:
                    level[nxt] = level[cur] + w
                queue.append(nxt)

        result = np.array(voltage)
        for node in range(len(result)):
            result[node] += level[comp_of[node]]
        return result

    # ------------------------------------------------------------------
    # 任意负载点求值
    # ------------------------------------------------------------------
    def evaluate(self, load_resistance: float | None) -> NetworkResult:
        """给定负载电阻求整网与逐段、逐组结果；None 表示负载开路。"""
        if math.isinf(self._r_eq):
            load_current = 0.0
        elif load_resistance is None:
            load_current = 0.0
        elif load_resistance == 0.0:
            load_current = self._u_eq / self._r_eq
        else:
            load_current = self._u_eq / (self._r_eq + load_resistance)

        i_load = load_current
        segments: list[SegmentReport] = []
        edge_current_by_path: dict[tuple[int, ...], float] = {}
        for e, i_oc, i_u in zip(
            self._edges, self._edge_i_oc, self._edge_i_u
        ):
            branch_current = float(i_oc + i_u * i_load)  # p->q 方向
            reported_current = -branch_current          # 顺电动势流出为正
            edge_current_by_path[e.path] = reported_current
            r_seg = math.inf if e.g <= 0.0 else 1.0 / e.g
            seg_v = float(
                (self._v_oc[e.p] - self._v_oc[e.q])
                + (self._v_u[e.p] - self._v_u[e.q]) * i_load
            )
            dissipation = 0.0 if math.isinf(r_seg) else reported_current ** 2 * r_seg
            segments.append(
                SegmentReport(
                    index=len(segments),
                    path=e.path,
                    ref=e.ref,
                    emf=e.emf,
                    resistance=r_seg,
                    current=reported_current,
                    voltage=seg_v,
                    emitted_power=e.emf * reported_current,
                    dissipation=dissipation,
                )
            )

        groups: list[GroupReport] = []
        group_current_by_path: dict[tuple[int, ...], float] = {}
        # 自底向上归并组电流：groups 按前序排列，倒序即子组先于父组
        for indexed in reversed(self._tree.groups):
            node = indexed.node
            path = indexed.path
            member_currents: list[float] = []
            for i, child in enumerate(node.children):
                child_path = path + (i,)
                if isinstance(child, SegmentNode):
                    member_currents.append(edge_current_by_path[child_path])
                else:
                    member_currents.append(group_current_by_path[child_path])
            if node.connection == "parallel":
                group_current = sum(member_currents)
            else:  # 串联：各成员电流同一股，直接取第一个
                group_current = member_currents[0]
            if not path:
                group_current = i_load  # 根组电流即负载电流，钉死避免浮点漂移
            group_current_by_path[path] = group_current

        for indexed in self._tree.groups:
            path = indexed.path
            p, q = self._group_terms[path]
            group_voltage = float(
                (self._v_oc[p] - self._v_oc[q])
                + (self._v_u[p] - self._v_u[q]) * i_load
            )
            groups.append(
                GroupReport(
                    path=path,
                    connection=indexed.node.connection,
                    current=group_current_by_path[path],
                    voltage=group_voltage,
                )
            )

        if load_resistance is None:
            load_power = 0.0
        else:
            load_power = i_load * i_load * load_resistance

        return NetworkResult(
            open_circuit_voltage=self._u_eq,
            internal_resistance=self._r_eq,
            load_resistance=load_resistance,
            current=i_load,
            power=load_power,
            segments=tuple(segments),
            groups=tuple(groups),
        )

    @property
    def equivalent_emf(self) -> float:
        """整网等效电动势（开路端电压）。"""
        return self._u_eq

    @property
    def equivalent_resistance(self) -> float:
        """整网等效内阻；整网断开时为 math.inf。"""
        return self._r_eq

"""把分组树建成一般电路网络并统一求解（不分串/并特例）。

完整建模选型、病态处理与替代方案对比见仓库根目录 DESIGN.md。
这里只有一种元件、两种接线动作：

- 元件：段 = 诺顿支路（电导 g 并电流源 g*E），σ=0 时不生成边；
- 接线：汇流排是理想导线。串联 = 成员首尾经汇流点顺接；
  并联 = 成员两极分别汇到组的两极。组与组之间也只有导线。

整区间扫描只做两到三次线性解，之后每个负载点全是仿射组合：

1. oc：全部电动势在、负载不接。逐导纳连通分量钉参考节点解，
   给出开路电压、并联环流（悬空断头分量照解）；
2. c ：电动势全置零，向通道正极注入 +1A、负极抽出 1A。
   两极导纳连通时解得内阻（两极电位差）与各支路的电流分配系数；
3. weak：仅当两极不连通（等效内阻无穷大）时补一次——σ=0 的断头
   换成极小电导做极限解，只为给出开路电压这个展示量，不参与任何
   电流与功率。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .groups import EntityNode, GroupNode, SegmentNode
from .induction import internal_resistance, open_circuit_voltage
from .linalg import spd_solve

WEAK_RATIO = 1e-8   # 断头弱电导占最大正常电导的比例（仅用于展示电压）
WEAK_FLOOR = 1e-12  # 整网没有任何正常电导时的弱电导兜底值


# ------------------------------------------------------------- 建模记录

@dataclass(frozen=True)
class _Branch:
    """一条诺顿导纳支路，方向为 p 端 -> n 端（段电动势方向）。"""

    seg_index: int
    p_node: int
    n_node: int
    conductance: float
    source: float  # 诺顿电流源 g*E，注入 p 端


@dataclass(frozen=True)
class _Wire:
    """一条理想导线（汇流排）。按存储方向 u -> v 记录，tag 给出
    它作为哪一个组的「正极引出线」（组电流 = -I_wire）。"""

    u: int
    v: int
    tag: int | None  # 组记录下标；None 表示组内中间汇流线


@dataclass
class _GroupRow:
    path: str
    node_id: str | None
    kind: str
    p_node: int
    n_node: int


@dataclass
class _SegmentRow:
    path: str
    node_id: str | None
    emf: float
    resistance: float
    branch_index: int | None  # σ=0 时为 None（断头，无导纳边）
    p_node: int
    n_node: int


@dataclass
class _Network:
    root_p: int
    root_n: int
    branches: list[_Branch]
    wires: list[_Wire]
    segments: list[_SegmentRow]
    groups: list[_GroupRow]
    n_nodes: int


# ------------------------------------------------------------- 建网

def _build_network(root: GroupNode) -> _Network:
    """按拓扑把分组树展开成节点、导线、导纳支路。"""
    nxt = [0]

    def new_node() -> int:
        node = nxt[0]
        nxt[0] += 1
        return node

    branches: list[_Branch] = []
    wires: list[_Wire] = []
    segments: list[_SegmentRow] = []
    groups: list[_GroupRow] = []

    def add_segment(node: SegmentNode, path: str, p_node: int, n_node: int) -> None:
        emf = open_circuit_voltage(node.params)
        resistance = internal_resistance(node.params)
        if math.isfinite(resistance):
            conductance = 1.0 / resistance
            branch = _Branch(
                seg_index=len(segments),
                p_node=p_node,
                n_node=n_node,
                conductance=conductance,
                source=conductance * emf,
            )
            branch_index: int | None = len(branches)
            branches.append(branch)
        else:
            branch_index = None
        segments.append(
            _SegmentRow(
                path=path,
                node_id=node.node_id,
                emf=emf,
                resistance=resistance,
                branch_index=branch_index,
                p_node=p_node,
                n_node=n_node,
            )
        )

    def build(node: EntityNode, path: str, p_terminal: int, n_terminal: int) -> None:
        """把 node 的网络接到外部端子 p_terminal / n_terminal 之间。

        约定：p_terminal 到组正极的导线由调用方已经连好（build_group
        只负责内部接线）；组的负极直接使用 n_terminal，不再加节点。
        """
        if isinstance(node, SegmentNode):
            add_segment(node, path, p_terminal, n_terminal)
            return

        groups.append(
            _GroupRow(
                path=path,
                node_id=node.node_id,
                kind=node.kind,
                p_node=p_terminal,
                n_node=n_terminal,
            )
        )

        if node.kind == "parallel":
            # 所有成员正极并到 p_terminal，负极并到 n_terminal
            for i, child in enumerate(node.children):
                child_path = f"{path}.members[{i}]"
                if isinstance(child, SegmentNode):
                    cp = new_node()
                    wires.append(_Wire(p_terminal, cp, None))
                    add_segment(child, child_path, cp, n_terminal)
                else:
                    cp = new_node()
                    wires.append(_Wire(p_terminal, cp, len(groups)))  # 子组正极引出
                    build(child, child_path, cp, n_terminal)
            return

        # series：成员首尾顺接。第一成员正极接 p_terminal，
        # 其后每个成员正极 = 上一成员负极（汇流点），最末负极 = n_terminal。
        junction = p_terminal
        for i, child in enumerate(node.children):
            child_path = f"{path}.members[{i}]"
            if i == len(node.children) - 1:
                cn = n_terminal
            else:
                cn = new_node()  # 中间汇流点：本段负极即下段正极
            if isinstance(child, SegmentNode):
                add_segment(child, child_path, junction, cn)
            else:
                cp = new_node()
                wires.append(_Wire(junction, cp, len(groups)))  # 子组正极引出
                build(child, child_path, cp, cn)
            junction = cn

    root_p = new_node()
    root_n = new_node()
    build(root, "$", root_p, root_n)

    return _Network(
        root_p=root_p,
        root_n=root_n,
        branches=branches,
        wires=wires,
        segments=segments,
        groups=groups,
        n_nodes=nxt[0],
    )


# ------------------------------------------------------------- 并查集

class _DSU:
    """导线合并超节点用的并查集，等价类以最小物理节点编号为代表。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


# ------------------------------------------------------------- 超节点网络

def _merge_wires(network: _Network) -> _DSU:
    dsu = _DSU(network.n_nodes)
    for wire in network.wires:
        dsu.union(wire.u, wire.v)
    return dsu


def _aggregate_edges(network: _Network, dsu: _DSU, use_weak: bool, weak_g: float):
    """聚合超节点对之间的导纳；use_weak 时断头计入弱电导。

    返回 (matrix_edges, weak_segments)，matrix_edges 为 (a, b, g) 列表
    （a < b，已聚合），weak_segments 为断头段下标列表。
    """
    agg: dict[tuple[int, int], float] = {}
    for branch in network.branches:
        a, b = dsu.find(branch.p_node), dsu.find(branch.n_node)
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        agg[key] = agg.get(key, 0.0) + branch.conductance

    weak_segments: list[int] = []
    if use_weak:
        for index, seg in enumerate(network.segments):
            if seg.branch_index is not None:
                continue
            a, b = dsu.find(seg.p_node), dsu.find(seg.n_node)
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            agg[key] = agg.get(key, 0.0) + weak_g
            weak_segments.append(index)

    return [(a, b, g) for (a, b), g in agg.items()], weak_segments


def _components(network: _Network, dsu: _DSU, matrix_edges):
    """在超节点上按导纳边求连通分量。孤立超节点自成一个分量。"""
    adj: dict[int, set[int]] = {}
    for a, b, _ in matrix_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    comp_of: dict[int, int] = {}
    comp_nodes: list[list[int]] = []
    for start in sorted(adj):
        if start in comp_of:
            continue
        cid = len(comp_nodes)
        stack = [start]
        comp_of[start] = cid
        members: list[int] = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in comp_of:
                    comp_of[nb] = cid
                    stack.append(nb)
        comp_nodes.append(sorted(members))

    # 没有任何导纳边的超节点（纯悬空）也要自成一个分量
    for node in range(network.n_nodes):
        s = dsu.find(node)
        if s not in comp_of:
            comp_of[s] = len(comp_nodes)
            comp_nodes.append([s])
    return comp_of, comp_nodes


# ------------------------------------------------------------- 节点电压解

def _solve_components(
    network: _Network,
    dsu: _DSU,
    comp_of: dict[int, int],
    comp_nodes: list[list[int]],
    matrix_edges,
    source_inject: dict[int, float],
    extra_inject: dict[int, float],
    root_cid: int | None,
    all_components: bool,
) -> dict[int, float]:
    """逐分量解约化拉普拉斯，返回超节点电位。

    source_inject：诺顿源在超节点上的固定注入；extra_inject：c 解的
    单位注入。root 所在分量以 root_n 超节点接地（电位基准），其它分量
    钉最小节点；all_components=False 时只解 root 分量。
    """
    result: dict[int, float] = {}
    cids = range(len(comp_nodes)) if all_components else (
        [root_cid] if root_cid is not None else []
    )
    root_n_super = dsu.find(network.root_n)

    for cid in cids:
        members = comp_nodes[cid]
        # 只有参考节点就在本分量时（root 两极连通）才以它接地，
        # 否则钉本分量最小节点；悬空断头分量照此求解。
        ref = root_n_super if root_n_super in members else members[0]
        active = [s for s in members if s != ref]
        n = len(active)
        matrix: dict[int, dict[int, float]] = {s: {} for s in active}

        for a, b, g in matrix_edges:
            # 边只有在两端都属于当前分量时才盖章（一端参考节点也行）
            if comp_of.get(a) != cid or comp_of.get(b) != cid:
                continue
            if a == ref or b == ref:
                other = b if a == ref else a
                if other in matrix:
                    matrix[other][other] = matrix[other].get(other, 0.0) + g
            else:
                matrix[a][a] = matrix[a].get(a, 0.0) + g
                matrix[b][b] = matrix[b].get(b, 0.0) + g
                matrix[a][b] = matrix[a].get(b, 0.0) - g
                matrix[b][a] = matrix[b].get(a, 0.0) - g

        rhs = {
            s: source_inject.get(s, 0.0) + extra_inject.get(s, 0.0)
            for s in active
        }
        if n:
            result.update(spd_solve(matrix, rhs, active))
        result[ref] = 0.0
    return result


def _physical_potentials(network: _Network, dsu: _DSU, super_pot: dict[int, float]):
    return [super_pot.get(dsu.find(node), 0.0) for node in range(network.n_nodes)]


def _branch_currents(network: _Network, potentials, *, sources_on: bool = True):
    """每条支路上「从正极流出」方向的电流 I = gE - g(Vp - Vn)。

    该方向即段电动势方向（p 端为正极）：电流从电动势正极经内阻
    流出到 n 端。被倒灌时为负。c 解把全部电动势置零，传
    sources_on=False，只剩 -g(Vp - Vn) 一项。
    """
    result = [0.0] * len(network.branches)
    for i, branch in enumerate(network.branches):
        source = branch.source if sources_on else 0.0
        result[i] = source - branch.conductance * (
            potentials[branch.p_node] - potentials[branch.n_node]
        )
    return result


def _wire_currents(network: _Network, branch_currents):
    """割集法：导线电流 = 子树上非导线边注入之和。

    一条导纳支路在其 p 端向导线网络注入 +I、在 n 端注入 -I（KCL：
    流出电动势正极的电流必须由导线送过来）。导线森林以每棵树里物理
    编号最小的节点为根，边电流最终换算成存储方向 u -> v 的带符号值。
    """
    injection = [0.0] * network.n_nodes
    for branch, current in zip(network.branches, branch_currents):
        injection[branch.p_node] += current
        injection[branch.n_node] -= current

    adj: dict[int, list[tuple[int, int]]] = {}
    for wi, wire in enumerate(network.wires):
        adj.setdefault(wire.u, []).append((wire.v, wi))
        adj.setdefault(wire.v, []).append((wire.u, wi))

    result = [0.0] * len(network.wires)
    seen = [False] * network.n_nodes
    for root in range(network.n_nodes):
        if seen[root]:
            continue
        seen[root] = True
        if root not in adj:
            continue
        # 迭代 DFS，记录 (node, parent, wire_index)
        order: list[tuple[int, int, int]] = []
        stack: list[tuple[int, int, int]] = [(root, -1, -1)]
        while stack:
            node, parent, wi = stack.pop()
            order.append((node, parent, wi))
            for nb, edge_index in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append((nb, node, edge_index))
        # 后序累加子树注入（只在本树节点上操作，不复制整张表）
        subtree = {node: injection[node] for node, _, _ in order}
        for node, parent, wi in reversed(order):
            if parent >= 0:
                subtree[parent] += subtree[node]
                wire = network.wires[wi]
                # 电流方向 parent -> node；存储方向 u -> v
                current = subtree[node]
                result[wi] = current if wire.u == parent else -current
    return result


# ------------------------------------------------------------- 结果类型

@dataclass(frozen=True)
class SegmentResult:
    """逐段结果。current 以该段电动势方向为正，被倒灌为负。"""

    path: str
    node_id: str | None
    emf: float
    internal_resistance: float
    current: float
    generated_power: float       # I*E：倒灌段被充电，为负
    dissipation: float           # I²*r
    terminal_power: float        # I*E - I²*r，该段对外净输出


@dataclass(frozen=True)
class GroupResult:
    path: str
    node_id: str | None
    kind: str
    current: float               # 正极流出到外电路为正
    voltage: float               # 正极减负极
    power: float                 # current * voltage


@dataclass(frozen=True)
class OperatingPoint:
    open_circuit_voltage: float
    internal_resistance: float
    load_resistance: float
    load_current: float
    load_power: float
    segments: tuple[SegmentResult, ...]
    groups: tuple[GroupResult, ...]


@dataclass
class ChannelSolution:
    """建网 + 两次（断开时三次）线性解的结果，可在任意负载上取值。"""

    open_circuit_voltage: float
    internal_resistance: float
    network: _Network
    seg_oc: list[float]
    seg_c: list[float]
    grp_oc_i: list[float]
    grp_c_i: list[float]
    grp_oc_v: list[float]
    grp_c_v: list[float | None]
    grp_weak_v: list[float]

    def at(self, load_resistance: float) -> OperatingPoint:
        if math.isinf(self.internal_resistance):
            i_load = 0.0
        else:
            i_load = self.open_circuit_voltage / (
                self.internal_resistance + load_resistance
            )
        connected = math.isfinite(self.internal_resistance)
        # 开路（负载电阻为 inf）时 0*inf 得 NaN，负载功率显式取零
        if i_load == 0.0:
            load_power = 0.0
        else:
            load_power = i_load * i_load * load_resistance

        segments = []
        for row, i_oc, i_c in zip(self.network.segments, self.seg_oc, self.seg_c):
            # seg_c 在汇总时已按「与负载同向」取号，直接相加
            current = i_oc + i_load * i_c
            dissipation = (
                current * current * row.resistance
                if math.isfinite(row.resistance)
                else 0.0
            )
            generated = current * row.emf
            segments.append(
                SegmentResult(
                    path=row.path,
                    node_id=row.node_id,
                    emf=row.emf,
                    internal_resistance=row.resistance,
                    current=current,
                    generated_power=generated,
                    dissipation=dissipation,
                    terminal_power=generated - dissipation,
                )
            )

        groups = []
        for row, i_oc, i_c, v_oc, v_c, v_weak in zip(
            self.network.groups,
            self.grp_oc_i,
            self.grp_c_i,
            self.grp_oc_v,
            self.grp_c_v,
            self.grp_weak_v,
        ):
            # grp_c_i 在汇总时已按「与负载同向」取号
            current = i_oc + i_load * i_c
            if connected and v_c is not None:
                voltage = v_oc - i_load * v_c  # 带载电压 = 电动势 - 内阻压降
            else:
                # 两极不连通：任何有限负载下 I_load=0，电压取弱极限展示值
                voltage = v_weak
            groups.append(
                GroupResult(
                    path=row.path,
                    node_id=row.node_id,
                    kind=row.kind,
                    current=current,
                    voltage=voltage,
                    power=current * voltage,
                )
            )

        return OperatingPoint(
            open_circuit_voltage=self.open_circuit_voltage,
            internal_resistance=self.internal_resistance,
            load_resistance=load_resistance,
            load_current=i_load,
            load_power=load_power,
            segments=tuple(segments),
            groups=tuple(groups),
        )


# ------------------------------------------------------------- 对外入口

def solve_channel(root: GroupNode) -> ChannelSolution:
    """对一棵已展开（无 RefNode）的分组树完成统一网络求解。"""
    network = _build_network(root)

    dsu = _merge_wires(network)
    max_g = max(
        (branch.conductance for branch in network.branches), default=0.0
    )
    weak_g = max(max_g * WEAK_RATIO, WEAK_FLOOR)

    matrix_edges, _ = _aggregate_edges(network, dsu, use_weak=False, weak_g=weak_g)
    comp_of, comp_nodes = _components(network, dsu, matrix_edges)

    root_p = dsu.find(network.root_p)
    root_n = dsu.find(network.root_n)
    root_cid = comp_of[root_p]
    connected = comp_of[root_p] == comp_of[root_n]

    # 诺顿源在超节点上的注入
    source_inject: dict[int, float] = {}
    for branch in network.branches:
        a, b = dsu.find(branch.p_node), dsu.find(branch.n_node)
        if a == b:
            continue
        source_inject[a] = source_inject.get(a, 0.0) + branch.source
        source_inject[b] = source_inject.get(b, 0.0) - branch.source

    # ---- 解 1：开路（源在、负载不接），全部连通分量各钉一个参考 ----
    v_oc_super = _solve_components(
        network, dsu, comp_of, comp_nodes, matrix_edges,
        source_inject, {}, root_cid, all_components=True,
    )
    v_oc = _physical_potentials(network, dsu, v_oc_super)
    i_branch_oc = _branch_currents(network, v_oc)
    i_wire_oc = _wire_currents(network, i_branch_oc)

    # ---- 解 2：源置零，向 root_p 注入 +1A、从 root_n 抽出 1A。
    # 这是「单位电流从正极流入网络」的状态：
    #   * root 的 V_p - V_n 就是等效内阻 R_eq（正极电位高于负极）；
    #   * 组电压系数也直接取 V_p - V_n，带载时 V = V_oc - I_load*V_c；
    #   * 该状态下支路电流方向与负载电流（正极流出）相反，所以段/组
    #     电流叠加时用 I = I_oc - I_load * i_c。
    v_c_phys = [0.0] * network.n_nodes
    i_branch_c = [0.0] * len(network.branches)
    i_wire_c = [0.0] * len(network.wires)
    r_eq = math.inf
    if connected:
        v_c_super = _solve_components(
            network, dsu, comp_of, comp_nodes, matrix_edges,
            {}, {root_p: 1.0, root_n: -1.0}, root_cid, all_components=False,
        )
        v_c_phys = _physical_potentials(network, dsu, v_c_super)
        i_branch_c = _branch_currents(network, v_c_phys, sources_on=False)
        i_wire_c = _wire_currents(network, i_branch_c)
        r_eq = v_c_phys[network.root_p] - v_c_phys[network.root_n]

    # ---- 解 3（仅两极断开时）：弱电导极限，只用于开路电压展示 ----
    grp_weak_v = [0.0] * len(network.groups)
    if connected:
        u_oc = v_oc[network.root_p] - v_oc[network.root_n]
    else:
        wdsu = _merge_wires(network)
        w_edges, weak_segments = _aggregate_edges(
            network, wdsu, use_weak=True, weak_g=weak_g
        )
        wcomp_of, wcomp_nodes = _components(network, wdsu, w_edges)
        weak_inject: dict[int, float] = {}
        for index in weak_segments:
            seg = network.segments[index]
            a = wdsu.find(seg.p_node)
            b = wdsu.find(seg.n_node)
            ge = weak_g * seg.emf
            weak_inject[a] = weak_inject.get(a, 0.0) + ge
            weak_inject[b] = weak_inject.get(b, 0.0) - ge
        w_source = dict(source_inject)
        for key, value in weak_inject.items():
            w_source[key] = w_source.get(key, 0.0) + value
        wcid = wcomp_of[wdsu.find(network.root_p)]
        vw_super = _solve_components(
            network, wdsu, wcomp_of, wcomp_nodes, w_edges,
            w_source, {}, wcid, all_components=True,
        )
        vw = _physical_potentials(network, wdsu, vw_super)
        u_oc = vw[network.root_p] - vw[network.root_n]
        grp_weak_v = [vw[row.p_node] - vw[row.n_node] for row in network.groups]

    # ---- 组电流：用正极引出线（wire.tag 指向该组）的割集电流反号 ----
    p_wire = [-1] * len(network.groups)
    for wi, wire in enumerate(network.wires):
        if wire.tag is not None:
            p_wire[wire.tag] = wi

    def group_current(index: int, wire_currents, branch_currents):
        wi = p_wire[index]
        if wi >= 0:
            # 引出线电流（组内 -> 组外）就是从组正极流出的组电流
            return wire_currents[wi]
        # 根组没有引出线：取 root_p 超节点上全部导纳支路流出电流之和
        total = 0.0
        sp = dsu.find(network.root_p)
        for branch, current in zip(network.branches, branch_currents):
            if dsu.find(branch.p_node) == sp:
                total += current
        return total

    grp_oc_i = [
        group_current(gi, i_wire_oc, i_branch_oc) for gi in range(len(network.groups))
    ]
    grp_c_i = [
        group_current(gi, i_wire_c, i_branch_c) for gi in range(len(network.groups))
    ]
    grp_oc_v = [v_oc[row.p_node] - v_oc[row.n_node] for row in network.groups]
    grp_c_v: list[float | None] = []
    for row in network.groups:
        if not connected:
            grp_c_v.append(None)
            continue
        sp, sn = dsu.find(row.p_node), dsu.find(row.n_node)
        if comp_of.get(sp) == root_cid and comp_of.get(sn) == root_cid:
            # c 状态正极注入，V_p - V_n 即沿负载方向的内阻压降系数
            grp_c_v.append(v_c_phys[row.p_node] - v_c_phys[row.n_node])
        else:
            grp_c_v.append(None)

    # c 状态电流方向与负载电流相反，段/组电流分配系数统一取负
    seg_oc = [
        0.0 if row.branch_index is None else i_branch_oc[row.branch_index]
        for row in network.segments
    ]
    seg_c = [
        0.0 if row.branch_index is None else -i_branch_c[row.branch_index]
        for row in network.segments
    ]
    grp_c_i = [-value for value in grp_c_i]

    return ChannelSolution(
        open_circuit_voltage=u_oc,
        internal_resistance=r_eq,
        network=network,
        seg_oc=seg_oc,
        seg_c=seg_c,
        grp_oc_i=grp_oc_i,
        grp_c_i=grp_c_i,
        grp_oc_v=grp_oc_v,
        grp_c_v=grp_c_v,
        grp_weak_v=grp_weak_v,
    )

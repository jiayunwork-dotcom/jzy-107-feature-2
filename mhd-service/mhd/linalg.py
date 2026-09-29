"""对称正定稀疏线性方程组原语。

网络求解器面对的是节点导纳（拉普拉斯）矩阵去掉一个参考行/列后的
约化矩阵——对称、严格对角占优、正定。规模由分组段数上限（百段量级）
卡住。这里用：

1. 近似最小度（minimum-degree）排序：按消元图度数挑主元，平局取编号
   最小，串并联网络树宽极小，排序后填元很少，结果也完全确定；
2. 稀疏 LDL^T 分解（数值消元）：正定矩阵主元恒正，不需要选主元；
   邻接边随消元过程自然长出，不需要预先符号分析。

同构网络（评估与扫描共用同一次建网与消元）在相同输入下浮点运算
次序完全固定，逐位可复现。
"""
from __future__ import annotations


def _minimum_degree_order(
    a: dict[int, dict[int, float]], idx: dict[int, int]
) -> list[int]:
    """在消元图上做近似最小度排序，平局取节点编号最小保证确定性。"""
    adj: dict[int, set[int]] = {
        i: {j for j in row if j != i and row[j] != 0.0} for i, row in a.items()
    }
    alive = set(a)
    order: list[int] = []
    while alive:
        pivot = min(alive, key=lambda i: (len(adj[i]), i))
        nbrs = sorted(j for j in adj[pivot] if j in alive)
        for pos, u in enumerate(nbrs):
            for v in nbrs[pos + 1 :]:
                adj[u].add(v)
                adj[v].add(u)
        alive.discard(pivot)
        order.append(pivot)
    return order


def spd_solve(
    matrix: dict[int, dict[int, float]],
    rhs: dict[int, float],
    nodes: list[int],
) -> dict[int, float]:
    """解 A x = b。

    matrix 以 {行: {列: 值}} 给出，必须是完整对称矩阵（上下三角都盖），
    nodes 是参与求解的全部节点（参考节点已剔除），rhs 给右端项（缺省 0）。
    """
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    # 全程以 nodes 里的节点编号为字典键，避免「行键 vs 局部下标」混淆
    a: dict[int, dict[int, float]] = {node: {} for node in nodes}
    b: dict[int, float] = {node: 0.0 for node in nodes}
    for i_node, row in matrix.items():
        if i_node not in idx:
            continue
        for j_node, value in row.items():
            if j_node in idx:
                a[i_node][j_node] = value  # 参考节点（电位 0）的项直接丢弃
        b[i_node] = float(rhs.get(i_node, 0.0))

    order = _minimum_degree_order(a, idx)

    # LDL^T：L 单位下三角（节点编号为键）。填元在消元中自然长出。
    lower: dict[int, dict[int, float]] = {node: {} for node in nodes}
    cols: dict[int, set[int]] = {node: set() for node in nodes}
    diag: dict[int, float] = {}

    for pos, k in enumerate(order):
        # d_k = A_kk - Σ_{j∈cols[k]} l_kj² d_j
        d = a[k].get(k, 0.0)
        for j in cols[k]:
            l_kj = lower[k][j]
            d -= l_kj * l_kj * diag[j]
        diag[k] = d

        # 只更新消元图上仍与 k 相邻的未消行 i（原边或共同的已消列）
        remaining = order[pos + 1 :]
        linked = [
            i for i in remaining if i in a[k] or (cols[i] & cols[k])
        ]
        for i in linked:
            total = a[k].get(i, 0.0)  # A_ik
            for j in cols[i] & cols[k]:
                total -= lower[i][j] * lower[k][j] * diag[j]
            if total != 0.0:
                lower[i][k] = total / d
                cols[i].add(k)

    # 前代 L y = b
    y: dict[int, float] = {}
    for pos, k in enumerate(order):
        total = b[k]
        for j in order[:pos]:
            if j in lower[k]:
                total -= lower[k][j] * y[j]
        y[k] = total

    # 对角回代 D z = y，再回代 L^T x = z
    z = {k: y[k] / diag[k] for k in order}
    x: dict[int, float] = {}
    for pos in range(n - 1, -1, -1):
        k = order[pos]
        total = z[k]
        for i in order[pos + 1 :]:
            if k in lower[i]:
                total -= lower[i][k] * x[i]
        x[k] = total

    return x

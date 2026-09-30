"""分段通道的分组树模型：结构定义、原始 JSON 解析、引用展开与限额。

调用方用一棵嵌套结构描述整条分段通道：

* 叶子是一段（SegmentNode），参数沿用单段通道的五项物理量，
  另外可以带一个 ref，直接引用一个已登记的通道档（单段档或分组档）；
* 中间节点是一个组（GroupNode），connection 标明组内成员串联
  （"series"）还是并联（"parallel"），成员可以是段，也可以是另一个组。

本模块只负责结构本身（解析、展开、限额），不含任何电学计算；
把树当成电路网络统一求解的工作在 network_model 里。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Union

from .induction import ChannelParams
from .validation import ValidationError, validate_channel_params

# 规模上限：防止滥用，也给线性代数求解器划清合理工作量
MAX_SEGMENTS = 1000          # 一棵展开树里段（叶子）总数上限
MAX_NESTING_DEPTH = 10       # 嵌套层数上限（根组为第 1 层）
MAX_GROUP_MEMBERS = 1000     # 单个组内直接成员个数上限

SERIES = "series"
PARALLEL = "parallel"
CONNECTIONS = (SERIES, PARALLEL)

CONNECTION_CN = {SERIES: "串联", PARALLEL: "并联"}


@dataclass(frozen=True)
class SegmentNode:
    """叶子：一段沿流向切出来的真实通道段。"""

    params: ChannelParams
    ref: str | None = None  # 由哪个通道档展开而来（内联段为 None），仅用于回显


@dataclass(frozen=True)
class GroupNode:
    """中间节点：一组以串联或并联方式连接的成员。"""

    connection: str  # SERIES 或 PARALLEL
    children: tuple["TreeNode", ...]
    ref: str | None = None  # 由哪个通道档展开而来（内联组为 None），仅用于回显


TreeNode = Union[SegmentNode, GroupNode]


@dataclass(frozen=True)
class IndexedSegment:
    """展开结果里的一段：带前序序号与从根起的成员路径。"""

    index: int                    # 前序遍历序号，从 0 起
    path: tuple[int, ...]         # 从根组到该段的成员下标序列
    node: SegmentNode


@dataclass(frozen=True)
class IndexedGroup:
    """展开结果里的一个组：根组 path 为空元组，groups 按前序排列。"""

    path: tuple[int, ...]
    node: GroupNode


@dataclass(frozen=True)
class ExpandedTree:
    """引用展开、限额检查后的树，附前序索引与逐层路径。"""

    root: GroupNode
    segments: tuple[IndexedSegment, ...]
    groups: tuple[IndexedGroup, ...]
    segment_count: int
    depth: int


def path_label(path: tuple[int, ...]) -> str:
    """成员路径的中文位置描述：根组/成员2/成员1 这样的定位串。"""
    if not path:
        return "根组"
    return "根组/" + "/".join(f"成员{i + 1}" for i in path)


# 占位段参数永远不会被用到：ref 占位节点在 expand_refs 中必被替换。
# 用独立实例做身份标记，校验通过的真实段不可能恰好是它。
_PLACEHOLDER_PARAMS = ChannelParams(1.0, 1.0, 0.0, 1.0, 1.0)


def parse_group(raw: object, path: tuple[int, ...] = ()) -> GroupNode:
    """校验并解析一个分组结构（原始 JSON 对象）。

    只查结构本身：组类型、成员个数、段参数、嵌套语法、嵌套深度。
    成员里的 ref 不在此处解析（档案是否存在、是否成环由档案层负责）。
    """
    if not isinstance(raw, dict):
        raise ValidationError(f"{path_label(path)}：分组结构必须是 JSON 对象")

    connection = raw.get("connection")
    if connection not in CONNECTIONS:
        raise ValidationError(
            f"{path_label(path)}：组类型 connection 必须是 {SERIES!r}（串联）"
            f"或 {PARALLEL!r}（并联），收到: {connection!r}"
        )

    members = raw.get("members")
    if not isinstance(members, list):
        raise ValidationError(f"{path_label(path)}：组成员 members 必须是数组")
    if len(members) == 0:
        raise ValidationError(
            f"{path_label(path)}：{CONNECTION_CN[connection]}组不能为空"
        )
    if len(members) > MAX_GROUP_MEMBERS:
        raise ValidationError(
            f"{path_label(path)}：单个组的成员个数不能超过 {MAX_GROUP_MEMBERS}，"
            f"收到 {len(members)} 个"
        )
    if len(path) + 1 > MAX_NESTING_DEPTH:
        raise ValidationError(
            f"{path_label(path)}：嵌套层数不能超过 {MAX_NESTING_DEPTH} 层"
        )

    children: list[TreeNode] = []
    for i, member in enumerate(members):
        children.append(_parse_member(member, path + (i,)))
    return GroupNode(connection=connection, children=tuple(children))


def _parse_member(raw: object, path: tuple[int, ...]) -> TreeNode:
    """解析一个成员：段、嵌套组或档案引用，三种写法互斥。"""
    if not isinstance(raw, dict):
        raise ValidationError(f"{path_label(path)}：成员必须是 JSON 对象")

    kinds = [key for key in ("segment", "group", "ref") if key in raw]
    if len(kinds) != 1:
        raise ValidationError(
            f"{path_label(path)}：成员必须且只能用 segment（段）、"
            f"group（嵌套组）、ref（引用通道档）三种写法之一"
        )

    kind = kinds[0]
    if kind == "group":
        return parse_group(raw["group"], path)
    if kind == "segment":
        params = validate_channel_params(
            raw["segment"], location=f"{path_label(path)} 的段"
        )
        return SegmentNode(params=params)

    ref = raw["ref"]
    if not isinstance(ref, str) or not ref.strip():
        raise ValidationError(
            f"{path_label(path)}：ref 必须是非空通道档名称字符串，收到: {ref!r}"
        )
    # 引用对象是段还是组由展开时按档案实际类型决定，先挂占位叶子，
    # expand_refs 会用被引档的真实树替换它。
    return SegmentNode(params=_PLACEHOLDER_PARAMS, ref=ref)


def expand_refs(
    root: GroupNode,
    lookup: Callable[[str], object],
    self_name: str | None = None,
) -> ExpandedTree:
    """把树里的 ref 成员替换成对应通道档展开后的真实子树。

    lookup 按名字返回通道档内容：ChannelParams（单段档）或已解析的
    GroupNode（分组档）；查不到时抛 ProfileNotFoundError。

    成环（A 的成员直接或间接引用 A）、展开后段总数或嵌套层数超限都在
    此处打回。遍历顺序确定，同一请求重复提交展开结果逐位一致。
    self_name 是这份树自身将保存成的档案名，用于拦住“自己引用自己”。
    """
    segments: list[IndexedSegment] = []
    groups: list[IndexedGroup] = []

    def walk(
        node: GroupNode,
        path: tuple[int, ...],
        depth: int,
        stack: tuple[str, ...],
    ) -> GroupNode:
        if depth > MAX_NESTING_DEPTH:
            raise ValidationError(
                f"{path_label(path)}：展开引用后嵌套层数不能超过 "
                f"{MAX_NESTING_DEPTH} 层"
            )
        rebuilt_children: list[TreeNode] = []
        for i, child in enumerate(node.children):
            child_path = path + (i,)
            if isinstance(child, SegmentNode) and child.params is _PLACEHOLDER_PARAMS:
                name = child.ref
                if name in stack:
                    raise ValidationError(
                        "通道档引用链不能成环: " + " -> ".join(stack + (name,))
                    )
                entry = lookup(name)
                if isinstance(entry, ChannelParams):
                    rebuilt: TreeNode = SegmentNode(params=entry, ref=name)
                elif isinstance(entry, GroupNode):
                    rebuilt = replace(
                        walk(entry, child_path, depth + 1, stack + (name,)),
                        ref=name,
                    )
                else:  # pragma: no cover - lookup 的实现保证不会走到这里
                    raise ValidationError(
                        f"{path_label(child_path)}：通道档 {name!r} 既不是单段档"
                        f"也不是分组档"
                    )
            elif isinstance(child, GroupNode):
                rebuilt = walk(child, child_path, depth + 1, stack)
            else:
                rebuilt = child
            rebuilt_children.append(rebuilt)

        rebuilt_node = replace(node, children=tuple(rebuilt_children))
        groups.append(IndexedGroup(path=path, node=rebuilt_node))
        return rebuilt_node

    initial_stack = (self_name,) if self_name is not None else ()
    expanded_root = walk(root, (), 1, initial_stack)

    def collect_segments(node: TreeNode, path: tuple[int, ...]) -> None:
        if isinstance(node, SegmentNode):
            segments.append(IndexedSegment(index=len(segments), path=path, node=node))
            return
        for i, child in enumerate(node.children):
            collect_segments(child, path + (i,))

    collect_segments(expanded_root, ())

    if not segments:  # pragma: no cover - parse_group 已拦空组
        raise ValidationError("分组结构展开后没有任何通道段")
    if len(segments) > MAX_SEGMENTS:
        raise ValidationError(
            f"一棵分组通道的段总数不能超过 {MAX_SEGMENTS}，展开后有 "
            f"{len(segments)} 段"
        )

    groups.sort(key=lambda item: item.path)  # 路径元组字典序即前序
    return ExpandedTree(
        root=expanded_root,
        segments=tuple(segments),
        groups=tuple(groups),
        segment_count=len(segments),
        depth=max((len(item.path) for item in groups), default=0) + 1,
    )

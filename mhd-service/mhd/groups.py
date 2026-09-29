"""分组通道的实体树类型与引用展开。

调用方用一棵分组树描述整条通道：

- ``SegmentNode``（叶子）：一段均匀通道，参数就是单段通道的五项；
- ``GroupNode``（中间节点）：一个组，``kind`` 标明 ``series``（串联）
  或 ``parallel``（并联），成员既可以是段，也可以是另一个组；
- ``RefNode``：按名字引用另一个通道档（单段档或分组档），
  在核算时由档案仓库展开成对方的当前定义（动态引用）。

本模块只放数据类型与"按名字展开引用"这一件事，不含校验细节、
不含网络建模、不含 HTTP。
"""
from __future__ import annotations

from dataclasses import dataclass

from .induction import ChannelParams


@dataclass(frozen=True)
class SegmentNode:
    """叶子：一段通道。node_id 为调用方可选的标注，缺省为 None。"""

    params: ChannelParams
    node_id: str | None = None


@dataclass(frozen=True)
class RefNode:
    """按名字引用另一个通道档的位置占位。"""

    name: str
    node_id: str | None = None


@dataclass(frozen=True)
class GroupNode:
    """组：kind 为 series/parallel，children 为有序成员。"""

    kind: str
    children: tuple["EntityNode", ...]
    node_id: str | None = None


EntityNode = SegmentNode | RefNode | GroupNode


def _ordered_chain(visiting: frozenset[str]) -> list[str]:
    """visiting 是无序集合，展示时按名字排序，保证错误消息确定。"""
    return sorted(visiting)


@dataclass(frozen=True)
class SegmentEntry:
    """档案仓库里的单段档：与老版通道档完全同构。"""

    params: ChannelParams


@dataclass(frozen=True)
class GroupEntry:
    """档案仓库里的分组档。

    root 是校验通过的未展开分组树（RefNode 保留名字，引用在核算时
    动态解析）；raw 是登记时的原始 JSON 结构，查看档案时原样返回。
    """

    root: GroupNode
    raw: dict


class ReferenceCycleError(ValueError):
    """引用展开时发现成环。消息里给出环上的档名链。"""


def resolve_tree(
    root: GroupNode,
    lookup,
    *,
    max_segments: int,
    max_depth: int,
) -> GroupNode:
    """把树里的 RefNode 全部展开成被引档的当前定义（动态引用）。

    lookup 为 ``name -> SegmentEntry | GroupEntry``，名字不存在时抛
    KeyError（由调用方翻译成 404 或保存时的悬空引用错误）。展开后
    再按总段数、总嵌套深度做一次硬限制检查；发现引用环抛
    ReferenceCycleError。
    """
    segment_count = 0

    def walk(node: EntityNode, path: str, depth: int, visiting: frozenset[str]):
        nonlocal segment_count

        if isinstance(node, SegmentNode):
            segment_count += 1
            if segment_count > max_segments:
                raise ValueError(
                    f"分组结构总段数超过上限 {max_segments}，"
                    f"在位置 {path} 已达 {segment_count} 段"
                )
            return node

        if isinstance(node, RefNode):
            if node.name in visiting:
                chain = " -> ".join(_ordered_chain(visiting))
                raise ReferenceCycleError(f"通道档引用成环: {chain}")
            try:
                entry = lookup(node.name)
            except KeyError:
                raise KeyError(node.name) from None
            # 进入被引档：把名字压栈；被引档是单段时链就到此为止，
            # 是分组档时继续在「已访问该档」的前提下遍历它的成员。
            next_visiting = visiting | {node.name}
            if isinstance(entry, SegmentEntry):
                leaf = SegmentNode(entry.params, node.node_id)
                segment_count += 1
                if segment_count > max_segments:
                    raise ValueError(
                        f"分组结构总段数超过上限 {max_segments}，"
                        f"在位置 {path} 已达 {segment_count} 段"
                    )
                return leaf
            target = GroupNode(entry.root.kind, entry.root.children, node.node_id)
            return walk_group(target, path, depth, next_visiting)

        return walk_group(node, path, depth, visiting)

    def walk_group(node: GroupNode, path: str, depth: int, visiting: frozenset[str]):
        if depth > max_depth:
            raise ValueError(
                f"分组结构嵌套层数超过上限 {max_depth}，位置 {path} 位于第 {depth} 层"
            )
        children = tuple(
            walk(child, f"{path}.members[{i}]", depth + 1, visiting)
            for i, child in enumerate(node.children)
        )
        return GroupNode(node.kind, children, node.node_id)

    return walk_group(root, "$", 1, frozenset())

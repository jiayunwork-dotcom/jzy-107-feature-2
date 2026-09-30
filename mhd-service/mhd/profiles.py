"""通道档存取：单段档与分组档共享同一名字空间，进程内可靠存取。

档案有两种：

* 单段档：不可变 ChannelParams，老接口原样保留；
* 分组档：一棵已通过结构校验的分组树（GroupNode，成员里仍保留 ref），
  登记时随原始结构一起存下，供 GET 原样回显。

引用口径（单点核算、负载扫描、查看档案三处一致）：分组档成员只存引用
名字，每次计算/查看都在锁内取出**被引档当前的最新值**再展开——不做
快照。把一个常用段重新保存成新参数后，所有引用它的分组档立刻按新值
算；保存时负责保证结构完整（悬空、成环一律拒收），因此运行时不会遇到
断引用。

并发：所有变更（保存、删除、改名覆盖）都在同一把锁内原子完成，删除前
先扫一遍全部档案的引用关系。引用一个档案与删除它不可能交错成“删掉了
被引档、引用者还在”的状态。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .group_tree import (
    ExpandedTree,
    GroupNode,
    SegmentNode,
    _PLACEHOLDER_PARAMS,
    expand_refs,
    parse_group,
)
from .induction import ChannelParams


class ProfileNotFoundError(KeyError):
    """按名称取通道档时未找到。"""


class ProfileInUseError(Exception):
    """删除被其它分组档引用着的档案。referrers 列出引用者名字。"""

    def __init__(self, name: str, referrers: list[str]) -> None:
        super().__init__(name, referrers)
        self.name = name
        self.referrers = referrers


@dataclass(frozen=True)
class StoredGroup:
    """分组档存库内容：已解析的树（成员 ref 保留）+ 原始结构（回显用）。"""

    node: GroupNode
    raw: dict


def _direct_refs(node: GroupNode) -> set[str]:
    """收集一棵分组树里直接引用到的档案名（不递归进入被引档）。"""
    refs: set[str] = set()

    def walk(item: GroupNode | SegmentNode) -> None:
        if isinstance(item, SegmentNode):
            if item.params is _PLACEHOLDER_PARAMS:
                refs.add(item.ref)
            return
        for child in item.children:
            walk(child)

    walk(node)
    return refs


class ProfileStore:
    """线程安全的进程内通道档仓库（单段档、分组档共用一个字典）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, ChannelParams | StoredGroup] = {}

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def save_channel(self, name: str, params: ChannelParams) -> None:
        """保存/覆盖单段档（老接口）。

        覆盖一个正被分组档引用的单段档时，引用者下一次计算直接取新值；
        若把它覆盖成分组档，引用者按分组子树展开（结构合法性在该分组档
        保存时负责，类型切换不会产生悬空引用）。
        """
        with self._lock:
            self._profiles[name] = params

    def save_group(self, name: str, raw_group: dict) -> StoredGroup:
        """保存/覆盖分组档：结构、悬空引用、成环全部在锁内一次验完。

        校验期间绝不改动库内字典：引用解析在“假设本次保存已生效”的临时
        视图上进行，因此并发读者/删除者不可能观察到只写了一半的分组档，
        也不会被一个最终会被拒收的占位档挡住删除。校验通过后才一次性提交。
        """
        node = parse_group(raw_group)
        stored = StoredGroup(node=node, raw=raw_group)
        with self._lock:
            snapshot = dict(self._profiles)
            snapshot[name] = node  # 自引用（直接或经环回到自身）要能查到

            def lookup(ref: str):
                return self._entry_to_node(snapshot.get(ref), ref)

            expand_refs(node, lookup, self_name=name)
            self._profiles[name] = stored
            return stored

    def delete(self, name: str) -> None:
        with self._lock:
            if name not in self._profiles:
                raise ProfileNotFoundError(name)
            referrers = self._referrers_locked(name)
            if referrers:
                raise ProfileInUseError(name, referrers)
            del self._profiles[name]

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def get_channel(self, name: str) -> ChannelParams:
        with self._lock:
            entry = self._profiles.get(name)
        if not isinstance(entry, ChannelParams):
            raise ProfileNotFoundError(name)
        return entry

    def get_stored_group(self, name: str) -> StoredGroup:
        with self._lock:
            entry = self._profiles.get(name)
        if not isinstance(entry, StoredGroup):
            raise ProfileNotFoundError(name)
        return entry

    def get(self, name: str) -> ChannelParams:
        """老接口：按名单段档读取，拿到的是不可变 ChannelParams。"""
        return self.get_channel(name)

    def get_entry(self, name: str) -> ChannelParams | StoredGroup:
        """查看档案：不区分类型，原样返回库内条目；不存在抛 ProfileNotFoundError。"""
        with self._lock:
            entry = self._profiles.get(name)
        if entry is None:
            raise ProfileNotFoundError(name)
        return entry

    def resolve(self, name: str) -> ChannelParams | ExpandedTree:
        """计算/扫描入口：返回展开后的 ExpandedTree，单段档则给 ChannelParams。

        展开在锁内一次性取一致快照（保存与删除都拿同一把锁），取到后
        再在线程外求解；引用的参数值就是“此刻被引档的最新值”。
        """
        with self._lock:
            entry = self._profiles.get(name)
            if entry is None:
                raise ProfileNotFoundError(name)
            if isinstance(entry, ChannelParams):
                return entry
            return expand_refs(entry.node, self._lookup_from(self._profiles), name)

    def expand_inline(self, node: GroupNode) -> ExpandedTree:
        """展开请求体内联给出的分组结构，ref 同样取档案当前值。"""
        with self._lock:
            return expand_refs(node, self._lookup_from(self._profiles))

    def names(self) -> list[str]:
        """老接口字段：全部档案名（单段档、分组档混排，字典序）。"""
        with self._lock:
            return sorted(self._profiles)

    def listing(self) -> tuple[list[str], list[str]]:
        """返回 (全部名称, 分组档名称)，各自字典序，供列表接口区分类型。"""
        with self._lock:
            all_names = sorted(self._profiles)
            groups = sorted(
                name for name, entry in self._profiles.items()
                if isinstance(entry, StoredGroup)
            )
        return all_names, groups

    # ------------------------------------------------------------------
    # 锁内工具
    # ------------------------------------------------------------------
    @staticmethod
    def _entry_to_node(entry, name: str):
        """库内条目 -> expand_refs 认识的形式；查无此档抛 ProfileNotFoundError。"""
        if entry is None:
            raise ProfileNotFoundError(name)
        if isinstance(entry, ChannelParams):
            return entry
        return entry.node

    def _lookup_from(self, table: dict):
        """生成在给定快照表里按名解析的回调（供 expand_refs 使用）。"""

        def lookup(name: str):
            return self._entry_to_node(table.get(name), name)

        return lookup

    def _referrers_locked(self, target: str) -> list[str]:
        """列出直接引用 target 的分组档名字（字典序）。"""
        return sorted(
            name
            for name, entry in self._profiles.items()
            if isinstance(entry, StoredGroup)
            and target in _direct_refs(entry.node)
            and name != target
        )

"""通道档存取：把常用通道（单段或分组）命名保存，进程内可靠存取。

同一个名字空间里有两种档：

- 单段档（SegmentEntry）：值就是不可变的 ChannelParams，与老版完全
  同构；老接口的读写删行为与字段保持原样；
- 分组档（GroupEntry）：值是校验通过的未展开分组树（RefNode 保留
  名字）与登记时的原始 JSON。

引用规则（保存时在同一把锁内一次性检查，保证多客户端并发下不会
留下指向空名字的档）：

- 引用的档必须已经存在（悬空引用拒收）；
- 引用链不许成环（含自引用）；
- 被引用着的档不许删除（返回 ProfileInUseError 并列出引用方）。

引用在**核算时动态展开**：被引档重新保存后，单点核算、扫描、查看
档案三处拿到的都是被引档的当前定义（查看档案看到的是保存的引用
结构本身；计算一律按当前值）。保存操作不做快照。
"""
from __future__ import annotations

import threading

from .groups import GroupEntry, GroupNode, ReferenceCycleError, SegmentEntry, resolve_tree
from .group_validation import MAX_GROUP_DEPTH, MAX_GROUP_SEGMENTS
from .induction import ChannelParams


class ProfileNotFoundError(KeyError):
    """按名称取通道档时未找到。"""


class ProfileInUseError(Exception):
    """删除时该档仍被别的分组档引用。"""

    def __init__(self, name: str, referrers: list[str]) -> None:
        super().__init__(name)
        self.name = name
        self.referrers = referrers


# 兼容老代码里 ProfileStore 只存 ChannelParams 的形态：
# 对外 get() 默认返回 ChannelParams（单段档），新接口取条目用 get_entry()。


class ProfileStore:
    """线程安全的进程内通道档仓库（单段档与分组档同一名字空间）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, SegmentEntry | GroupEntry] = {}

    # ---------------------------------------------------------- 单段档

    def save(self, name: str, params: ChannelParams) -> None:
        """保存（或覆盖）单段档。覆盖时引用关系不变（单段档不引用别人）。"""
        with self._lock:
            self._entries[name] = SegmentEntry(params)

    def get(self, name: str) -> ChannelParams:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise ProfileNotFoundError(name)
            if not isinstance(entry, SegmentEntry):
                raise ProfileTypeError(name, "group", "segment")
            return entry.params

    # ---------------------------------------------------------- 通用

    def get_entry(self, name: str) -> SegmentEntry | GroupEntry:
        with self._lock:
            try:
                return self._entries[name]
            except KeyError:
                raise ProfileNotFoundError(name) from None

    def is_group(self, name: str) -> bool:
        with self._lock:
            return isinstance(self._entries.get(name), GroupEntry)

    def save_group(
        self, name: str, root: GroupNode, raw: dict, refs: frozenset[str]
    ) -> None:
        """保存（或覆盖）分组档。

        保存前在同一把锁内完成：悬空引用检查、引用链环检测、展开后的
        总段数/层数上限复查。校验失败不改仓库。
        """
        with self._lock:
            # 引用自己不算悬空（新内容即将存在），随后的展开会把它识别成环
            missing = sorted(
                r for r in refs if r != name and r not in self._entries
            )
            if missing:
                raise DanglingReferenceError(missing)

            # 环检测 + 展开，临时把当前名字视为新内容（允许自引用？不允许：
            # 自己还没存进去，自引用在 lookup 里会解析成旧值；为明确起见
            # 先把新内容放进临时视图，自引用因此成环）。
            def lookup(target: str):
                if target == name:
                    return GroupEntry(root, raw)
                return self._entries[target]

            try:
                resolve_tree(
                    root,
                    lookup,
                    max_segments=MAX_GROUP_SEGMENTS,
                    max_depth=MAX_GROUP_DEPTH,
                )
            except ReferenceCycleError:
                raise
            except KeyError as error:
                raise DanglingReferenceError([error.args[0]]) from None
            except ValueError as error:
                raise GroupLimitError(str(error)) from None

            self._entries[name] = GroupEntry(root=root, raw=raw)

    def resolve(self, name: str) -> GroupNode:
        """取一个分组档并按当前档案动态展开全部引用。"""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise ProfileNotFoundError(name)
            if not isinstance(entry, GroupEntry):
                raise ProfileTypeError(name, "segment", "group")
            try:
                return resolve_tree(
                    entry.root,
                    lambda target: self._entries[target],
                    max_segments=MAX_GROUP_SEGMENTS,
                    max_depth=MAX_GROUP_DEPTH,
                )
            except KeyError as error:
                # 保存时已挡住悬空引用，走到这里说明数据内部不一致
                raise DanglingReferenceError([error.args[0]]) from None

    def resolve_inline(self, root: GroupNode) -> GroupNode:
        """展开一次请求里内联提交的分组树（其引用同样动态取当前档）。"""
        with self._lock:
            try:
                return resolve_tree(
                    root,
                    lambda target: self._entries[target],
                    max_segments=MAX_GROUP_SEGMENTS,
                    max_depth=MAX_GROUP_DEPTH,
                )
            except KeyError as error:
                raise DanglingReferenceError([error.args[0]]) from None
            except ReferenceCycleError:
                raise
            except ValueError as error:
                raise GroupLimitError(str(error)) from None

    def delete(self, name: str) -> None:
        with self._lock:
            if name not in self._entries:
                raise ProfileNotFoundError(name)
            referrers = self._referrers_locked(name)
            if referrers:
                raise ProfileInUseError(name, referrers)
            del self._entries[name]

    def _referrers_locked(self, target: str) -> list[str]:
        result = []
        for name, entry in self._entries.items():
            if isinstance(entry, GroupEntry) and target in _collect_refs(entry.root):
                result.append(name)
        return sorted(result)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)

    def group_names(self) -> list[str]:
        with self._lock:
            return sorted(
                name for name, entry in self._entries.items()
                if isinstance(entry, GroupEntry)
            )


class ProfileTypeError(TypeError):
    """按单段档取分组档（或反过来）。"""

    def __init__(self, name: str, actual: str, expected: str) -> None:
        super().__init__(name, actual, expected)
        self.name = name
        self.actual = actual
        self.expected = expected


class DanglingReferenceError(ValueError):
    """分组档引用了不存在的档。missing 列出悬空的档名。"""

    def __init__(self, missing: list[str]) -> None:
        self.missing = sorted(missing)
        super().__init__(", ".join(self.missing))


class GroupLimitError(ValueError):
    """展开引用后总段数或总层数超上限。"""


def _collect_refs(root: GroupNode) -> set[str]:
    from .groups import RefNode

    refs: set[str] = set()
    stack = list(root.children)
    while stack:
        node = stack.pop()
        if isinstance(node, RefNode):
            refs.add(node.name)
        elif isinstance(node, GroupNode):
            stack.extend(node.children)
    return refs

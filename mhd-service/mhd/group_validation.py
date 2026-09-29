"""分组结构的输入合法性拦截。

把调用方提交的原始 JSON 校验成 mhd.groups 里的实体树（引用保留为
RefNode，保存时不展开）。校验规则：

- 顶层必须是一个组：{"type": "series"|"parallel", "members": [...]}；
- 组成员必须非空，串/并类型写错、写漏都打回；
- 成员只能是组、段（五项参数的叶子）或 {"ref": 名称} 引用，
  三种写法不许混搭；
- 内联总段数、嵌套层数超过上限打回（展开引用后的总段数在引用
  展开时再查一次）；
- 每个错误都带位置（$.members[0].members[1] 这种路径）与原因。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .groups import GroupNode, RefNode, SegmentNode
from .validation import ValidationError, validate_channel_params, validate_profile_name

# 分组结构的合理上限：段数卡住网络规模，层数卡住递归与结构复杂度
MAX_GROUP_SEGMENTS = 100
MAX_GROUP_DEPTH = 16

_SERIES = "series"
_PARALLEL = "parallel"
_GROUP_TYPES = (_SERIES, _PARALLEL)

_CHANNEL_FIELDS = frozenset(
    {
        "magnetic_flux_density",
        "electrode_spacing",
        "flow_velocity",
        "conductivity",
        "cross_section_area",
    }
)
_STRUCTURAL_KEYS = frozenset({"type", "members"})
_REF_KEY = "ref"
_ID_KEY = "id"


@dataclass
class _Visitor:
    """校验遍历状态：累计内联段数、收集引用名。"""

    segments: int = 0
    refs: set[str] = field(default_factory=set)


def validate_group_tree(raw: object) -> tuple[GroupNode, frozenset[str]]:
    """校验顶层分组树，返回 (未展开的实体树, 全部引用名集合)。"""
    state = _Visitor()
    root = _validate_group(raw, "$", 1, state)
    return root, frozenset(state.refs)


def _node_id(raw: dict, path: str) -> str | None:
    if _ID_KEY not in raw:
        return None
    node_id = raw[_ID_KEY]
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValidationError(f"位置 {path} 的 id 必须是非空字符串，收到: {node_id!r}")
    return node_id


def _validate_group(raw: object, path: str, depth: int, state: _Visitor) -> GroupNode:
    if depth > MAX_GROUP_DEPTH:
        raise ValidationError(
            f"分组结构嵌套层数超过上限 {MAX_GROUP_DEPTH}，位置 {path} 位于第 {depth} 层"
        )
    if not isinstance(raw, dict):
        raise ValidationError(f"位置 {path} 必须是 JSON 对象")

    if _REF_KEY in raw:
        raise ValidationError(f"位置 {path} 是引用，引用必须作为组成员出现，不能顶替组")

    # 声明成组（有 type/members）的对象不许再夹段参数字段，反之亦然
    if ("type" in raw or "members" in raw) and (_CHANNEL_FIELDS & frozenset(raw)):
        raise ValidationError(
            f"位置 {path} 不能同时写组字段（type/members）与段参数字段"
        )

    if "type" not in raw:
        raise ValidationError(f"位置 {path} 缺少组类型: type（series 或 parallel）")
    kind = raw["type"]
    if kind not in _GROUP_TYPES:
        raise ValidationError(
            f"位置 {path} 的组类型必须是 series 或 parallel，收到: {kind!r}"
        )
    if "members" not in raw:
        raise ValidationError(f"位置 {path} 的组缺少成员列表: members")
    raw_members = raw["members"]
    if not isinstance(raw_members, list):
        raise ValidationError(f"位置 {path} 的 members 必须是数组，收到: {raw_members!r}")
    if not raw_members:
        raise ValidationError(f"位置 {path} 是空组，组至少要有一个成员")

    node_id = _node_id(raw, path)
    children = tuple(
        _validate_member(member, f"{path}.members[{index}]", depth + 1, state)
        for index, member in enumerate(raw_members)
    )
    return GroupNode(kind=kind, children=children, node_id=node_id)


def _validate_member(raw: object, path: str, depth: int, state: _Visitor):
    if not isinstance(raw, dict):
        raise ValidationError(f"位置 {path} 的成员必须是 JSON 对象")
    keys = frozenset(raw)

    if _REF_KEY in raw:
        allowed = {_REF_KEY, _ID_KEY}
        extra = keys - allowed
        if extra:
            raise ValidationError(
                f"位置 {path} 的引用只能带 ref（可选 id），多余字段: "
                f"{', '.join(sorted(extra))}"
            )
        name = validate_profile_name(raw[_REF_KEY])
        state.refs.add(name)
        return RefNode(name=name, node_id=_node_id(raw, path))

    if "type" in keys or "members" in keys:
        # 当作组校验，缺 type/members 或类型写错由 _validate_group 报错
        return _validate_group(raw, path, depth, state)

    if keys & _CHANNEL_FIELDS:
        if _STRUCTURAL_KEYS & keys:
            raise ValidationError(
                f"位置 {path} 不能同时写组字段（type/members）与段参数"
            )
        if state.segments + 1 > MAX_GROUP_SEGMENTS:
            raise ValidationError(
                f"分组结构总段数超过上限 {MAX_GROUP_SEGMENTS}，位置 {path} 是第 "
                f"{state.segments + 1} 段"
            )
        try:
            params = validate_channel_params(raw)
        except ValidationError as error:
            # 段参数错误要指出是哪个位置的哪一项
            raise ValidationError(f"位置 {path} 的段参数不合法：{error}") from None
        state.segments += 1
        return SegmentNode(params=params, node_id=_node_id(raw, path))

    raise ValidationError(
        f"位置 {path} 的成员无法识别：必须是组（type+members）、"
        f"段（五项通道参数）或引用（ref）"
    )

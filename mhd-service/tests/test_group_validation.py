"""分组结构解析与输入校验：非法输入在计算前被带位置、带原因地打回。"""
from __future__ import annotations

import pytest

from mhd.group_tree import (
    MAX_GROUP_MEMBERS,
    MAX_NESTING_DEPTH,
    MAX_SEGMENTS,
    expand_refs,
    parse_group,
)
from mhd.profiles import ProfileNotFoundError, ProfileStore
from mhd.validation import ValidationError

SEGMENT = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}


def seg(**changes) -> dict:
    data = dict(SEGMENT)
    data.update(changes)
    return {"segment": data}


def grp(connection="series", *members) -> dict:
    return {"connection": connection, "members": list(members)}


def test_empty_group_rejected_series_and_parallel():
    for connection in ("series", "parallel"):
        with pytest.raises(ValidationError, match="不能为空"):
            parse_group({"connection": connection, "members": []})


def test_bad_connection_type_rejected():
    with pytest.raises(ValidationError, match="connection"):
        parse_group({"connection": "star", "members": [seg()]})
    with pytest.raises(ValidationError, match="connection"):
        parse_group({"members": [seg()]})


def test_members_must_be_array():
    with pytest.raises(ValidationError, match="members 必须是数组"):
        parse_group({"connection": "series", "members": seg()})


def test_member_must_have_exactly_one_kind():
    with pytest.raises(ValidationError, match="三种写法"):
        parse_group(grp("series", {"segment": SEGMENT, "ref": "x"}))
    with pytest.raises(ValidationError, match="三种写法"):
        parse_group(grp("series", {}))
    with pytest.raises(ValidationError, match="JSON 对象"):
        parse_group(grp("series", [1, 2]))


def test_bad_ref_string_rejected():
    with pytest.raises(ValidationError, match="ref 必须是非空"):
        parse_group(grp("series", {"ref": ""}))
    with pytest.raises(ValidationError, match="ref 必须是非空"):
        parse_group(grp("series", {"ref": 7}))


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"magnetic_flux_density": -1.0}, "磁感应强度不能为负"),
        ({"electrode_spacing": 0.0}, "电极间距必须为正"),
        ({"flow_velocity": -10.0}, "流体速度不能为负"),
        ({"conductivity": -1.0}, "电导率不能为负"),
        ({"cross_section_area": 0.0}, "通道截面积必须为正"),
    ],
)
def test_invalid_segment_field_points_to_location_and_field(override, reason):
    tree = grp(
        "series",
        seg(),
        {"group": grp("parallel", seg(), seg(**override))},
    )
    with pytest.raises(ValidationError) as excinfo:
        parse_group(tree)
    message = str(excinfo.value)
    assert reason in message
    # 出错位置：根组 -> 成员2 -> 成员2
    assert "成员2" in message


def test_nesting_depth_limit_rejected():
    node = seg()
    for _ in range(MAX_NESTING_DEPTH):
        node = {"group": grp("series", node)}
    # 根组 + MAX_NESTING_DEPTH 层内层 = 超限
    with pytest.raises(ValidationError, match="嵌套层数"):
        parse_group(grp("series", node))


def test_nesting_depth_at_limit_accepted():
    node = seg()
    for _ in range(MAX_NESTING_DEPTH - 1):
        node = {"group": grp("series", node)}
    expanded = expand_refs(parse_group(grp("series", node)), lambda n: None)
    assert expanded.depth == MAX_NESTING_DEPTH


def test_group_member_count_limit_rejected():
    huge = grp("parallel", *(seg() for _ in range(MAX_GROUP_MEMBERS + 1)))
    with pytest.raises(ValidationError, match="成员个数"):
        parse_group(huge)


def test_segment_total_limit_rejected_via_references():
    """段数上限按展开后计算：999 个单段引用 + 一个两段组 = 1001 段。"""
    from mhd.induction import ChannelParams

    store = ProfileStore()
    store.save_channel("unit", ChannelParams(2.0, 1.0, 100.0, 10.0, 0.5))
    members = [{"ref": "unit"} for _ in range(MAX_SEGMENTS - 1)]
    members.append({"group": grp("series", seg(), seg())})
    wide = {"connection": "parallel", "members": members}
    parsed = parse_group(wide)  # 直接成员 1000 个，未超单组上限
    with pytest.raises(ValidationError, match="段总数"):
        expand_refs(parsed, store._lookup_from(store._profiles))


def test_dangling_reference_rejected_at_expand():
    store = ProfileStore()
    parsed = parse_group(grp("series", {"ref": "ghost"}, seg()))
    with pytest.raises(ProfileNotFoundError):
        expand_refs(parsed, store._lookup_from(store._profiles))


def test_non_object_root_rejected():
    with pytest.raises(ValidationError, match="JSON 对象"):
        parse_group([1, 2, 3])

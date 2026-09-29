"""分组结构校验：非法输入在计算前打回，错误带位置与原因。"""
from __future__ import annotations

import pytest

from mhd.group_validation import (
    MAX_GROUP_DEPTH,
    MAX_GROUP_SEGMENTS,
    validate_group_tree,
)
from mhd.validation import ValidationError

SEGMENT = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}


def test_valid_group_passes():
    root, refs = validate_group_tree(
        {"type": "series", "members": [SEGMENT, {"ref": "other"}]}
    )
    assert root.kind == "series"
    assert refs == frozenset({"other"})


def test_top_level_must_be_a_group():
    with pytest.raises(ValidationError, match="组类型"):
        validate_group_tree(SEGMENT)


def test_empty_group_rejected():
    with pytest.raises(ValidationError, match=r"位置 \$ 是空组"):
        validate_group_tree({"type": "series", "members": []})


def test_bad_group_type_rejected_with_position():
    with pytest.raises(ValidationError, match=r"位置 \$\.members\[1\].*series 或 parallel"):
        validate_group_tree(
            {"type": "parallel", "members": [
                SEGMENT, {"type": "wye", "members": [SEGMENT]},
            ]}
        )


def test_missing_members_rejected():
    with pytest.raises(ValidationError, match="缺少成员列表"):
        validate_group_tree({"type": "series"})


def test_members_must_be_array():
    with pytest.raises(ValidationError, match="members 必须是数组"):
        validate_group_tree({"type": "series", "members": SEGMENT})


def test_unknown_member_shape_rejected_with_position():
    with pytest.raises(ValidationError, match=r"位置 \$\.members\[0\] 的成员无法识别"):
        validate_group_tree({"type": "series", "members": [{"foo": 1}]})


def test_segment_field_error_points_to_position():
    bad = {**SEGMENT, "electrode_spacing": 0.0}
    with pytest.raises(ValidationError, match=r"\$\.members\[0\]\.members\[1\].*电极间距必须为正"):
        validate_group_tree(
            {"type": "series", "members": [
                {"type": "parallel", "members": [SEGMENT, bad]},
            ]}
        )


def test_missing_segment_field_points_to_position():
    bad = {k: v for k, v in SEGMENT.items() if k != "conductivity"}
    with pytest.raises(ValidationError, match=r"\$\.members\[0\].*缺少通道参数: conductivity"):
        validate_group_tree({"type": "parallel", "members": [bad]})


def test_deep_nesting_rejected():
    tree = SEGMENT
    for _ in range(MAX_GROUP_DEPTH + 1):
        tree = {"type": "series", "members": [tree]}
    with pytest.raises(ValidationError, match="嵌套层数超过上限"):
        validate_group_tree(tree)


def test_too_many_segments_rejected():
    members = [dict(SEGMENT) for _ in range(MAX_GROUP_SEGMENTS + 1)]
    with pytest.raises(ValidationError, match="总段数超过上限"):
        validate_group_tree({"type": "parallel", "members": members})


def test_limit_boundary_accepted():
    members = [dict(SEGMENT) for _ in range(MAX_GROUP_SEGMENTS)]
    root, _ = validate_group_tree({"type": "parallel", "members": members})
    assert len(root.children) == MAX_GROUP_SEGMENTS

    tree = SEGMENT
    for _ in range(MAX_GROUP_DEPTH - 1):
        tree = {"type": "series", "members": [tree]}
    validate_group_tree(tree)


def test_ref_with_extra_fields_rejected():
    with pytest.raises(ValidationError, match="只能带 ref"):
        validate_group_tree(
            {"type": "series", "members": [{"ref": "x", "channel": SEGMENT}]}
        )


def test_bad_id_rejected_with_position():
    with pytest.raises(ValidationError, match=r"\$\.members\[0\] 的 id 必须是非空字符串"):
        validate_group_tree(
            {"type": "series", "members": [{**SEGMENT, "id": ""}]}
        )


def test_ref_must_be_group_member():
    # 引用不能顶替顶层组
    with pytest.raises(ValidationError, match="引用必须作为组成员出现"):
        validate_group_tree({"ref": "x"})


def test_group_and_segment_fields_mixed_rejected():
    with pytest.raises(ValidationError, match="不能同时写组字段"):
        validate_group_tree(
            {"type": "series", "members": [{**SEGMENT, "type": "parallel"}]}
        )

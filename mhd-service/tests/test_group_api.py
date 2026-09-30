"""分段分组通道的 HTTP 接口测试：核算、扫描、分组档与错误码。"""
from __future__ import annotations

import math

import pytest

from app import create_app

CHANNEL = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}


def seg(**changes) -> dict:
    data = dict(CHANNEL)
    data.update(changes)
    return {"segment": data}


def grp(connection="series", *members) -> dict:
    return {"connection": connection, "members": list(members)}


@pytest.fixture()
def client():
    return create_app().test_client()


# ---------------------------------------------------------------------------
# 单点核算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("connection", ["series", "parallel"])
def test_evaluate_single_segment_group_matches_legacy(client, connection):
    grouped = client.post(
        "/evaluate",
        json={"group": grp(connection, seg()), "load_resistance": 0.2},
    ).get_json()
    legacy = client.post(
        "/evaluate", json={"channel": CHANNEL, "load_resistance": 0.2}
    ).get_json()
    for field in (
        "open_circuit_voltage",
        "internal_resistance",
        "current",
        "power",
    ):
        assert grouped[field] == legacy[field]
    assert len(grouped["segments"]) == 1
    assert len(grouped["groups"]) == 1
    segment = grouped["segments"][0]
    assert segment["emf"] == pytest.approx(200.0)
    assert segment["resistance"] == pytest.approx(0.2)
    assert segment["current"] == pytest.approx(500.0)
    assert segment["dissipation"] == pytest.approx(500.0 ** 2 * 0.2)
    assert segment["path"] == [0]


def test_evaluate_group_reports_every_segment_and_group(client):
    body = {
        "group": grp(
            "parallel",
            {"group": grp("series", seg(), seg(flow_velocity=80.0))},
            seg(flow_velocity=120.0),
        ),
        "load_resistance": 0.3,
    }
    data = client.post("/evaluate", json=body).get_json()
    assert len(data["segments"]) == 3
    assert len(data["groups"]) == 2
    # 根组 path 为空，内层组 path=[0]
    paths = [tuple(g["path"]) for g in data["groups"]]
    assert () in paths and (0,) in paths
    root = next(g for g in data["groups"] if not g["path"])
    assert root["connection"] == "parallel"
    assert root["current"] == pytest.approx(data["current"])
    inner = next(g for g in data["groups"] if g["path"] == [0])
    assert inner["connection"] == "series"


def test_evaluate_open_circuit_uses_null_and_shows_circulation(client):
    data = client.post(
        "/evaluate",
        json={
            "group": grp("parallel", seg(flow_velocity=100.0), seg(flow_velocity=80.0)),
            "load_resistance": None,
        },
    ).get_json()
    assert data["load_resistance"] is None
    assert data["current"] == 0.0
    currents = [s["current"] for s in data["segments"]]
    assert currents[0] == pytest.approx(100.0)
    assert currents[1] == pytest.approx(-100.0)


def test_evaluate_zero_conductivity_segment_serializes_null_resistance(client):
    data = client.post(
        "/evaluate",
        json={"group": grp("series", seg(), seg(conductivity=0.0)),
              "load_resistance": 1.0},
    ).get_json()
    assert data["internal_resistance"] is None
    dead = next(s for s in data["segments"] if s["resistance"] is None)
    assert dead["current"] == 0.0
    assert dead["dissipation"] == 0.0


def test_evaluate_requires_exactly_one_source(client):
    resp = client.post(
        "/evaluate",
        json={"channel": CHANNEL, "group": grp("series", seg()),
              "load_resistance": 1.0},
    )
    assert resp.status_code == 400
    assert "三者之一" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
def test_sweep_group_peak_and_curve(client):
    resp = client.post(
        "/sweep",
        json={
            "group": grp("series", seg(), seg()),
            "load_min": 0.01,
            "load_max": 10.0,
            "points": 100,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["peak"]["load_resistance"] == pytest.approx(0.4)
    point = data["curve"][33]
    single = client.post(
        "/evaluate",
        json={"group": grp("series", seg(), seg()),
              "load_resistance": point["load_resistance"]},
    ).get_json()
    assert single["current"] == point["current"]
    assert single["power"] == point["power"]
    # 峰值点逐段详情
    assert {s["index"] for s in data["segments"]} == {0, 1}
    assert len(data["groups"]) >= 1


def test_sweep_all_dead_channel_returns_zero_curve(client):
    resp = client.post(
        "/sweep",
        json={
            "group": grp("parallel", seg(conductivity=0.0), seg(conductivity=0.0)),
            "load_min": 0.0,
            "load_max": 8.0,
            "points": 30,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["internal_resistance"] is None
    assert all(p["current"] == 0.0 and p["power"] == 0.0 for p in data["curve"])


# ---------------------------------------------------------------------------
# 分组档：登记、列出、读取、引用
# ---------------------------------------------------------------------------
def test_group_profile_roundtrip(client):
    group = grp("parallel", {"ref": "base"}, seg(flow_velocity=80.0))
    assert client.post(
        "/channels", json={"name": "base", "channel": CHANNEL}
    ).status_code == 201
    resp = client.post("/channels", json={"name": "pack", "group": group})
    assert resp.status_code == 201
    assert resp.get_json() == {"name": "pack", "group": group}

    listed = client.get("/channels").get_json()
    assert listed["profiles"] == ["base", "pack"]  # 原字段不变
    assert listed["group_profiles"] == ["pack"]     # 新增字段标出分组档

    assert client.get("/channels/pack").get_json() == {"name": "pack", "group": group}
    # 老的单段档读取字段保持原样
    assert client.get("/channels/base").get_json()["channel"] == CHANNEL


def test_profile_name_resolves_both_kinds(client):
    client.post("/channels", json={"name": "base", "channel": CHANNEL})
    client.post(
        "/channels",
        json={"name": "pack", "group": grp("series", {"ref": "base"})},
    )
    single = client.post(
        "/evaluate", json={"profile": "base", "load_resistance": 0.2}
    ).get_json()
    assert "segments" not in single  # 单段档返回老字段
    packed = client.post(
        "/evaluate", json={"profile": "pack", "load_resistance": 0.2}
    ).get_json()
    assert packed["current"] == single["current"]
    assert len(packed["segments"]) == 1


def test_referenced_profile_live_value_used(client):
    client.post("/channels", json={"name": "base", "channel": CHANNEL})
    client.post(
        "/channels",
        json={"name": "pack", "group": grp("series", {"ref": "base"})},
    )
    changed = dict(CHANNEL, flow_velocity=140.0)
    client.post("/channels", json={"name": "base", "channel": changed})
    packed = client.post(
        "/evaluate", json={"profile": "pack", "load_resistance": 0.2}
    ).get_json()
    inline = client.post(
        "/evaluate",
        json={"group": grp("series", {"segment": changed}),
              "load_resistance": 0.2},
    ).get_json()
    assert packed["current"] == inline["current"]


def test_delete_referenced_profile_returns_409_with_referrers(client):
    client.post("/channels", json={"name": "base", "channel": CHANNEL})
    client.post(
        "/channels",
        json={"name": "pack", "group": grp("parallel", {"ref": "base"}, seg())},
    )
    resp = client.delete("/channels/base")
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["referrers"] == ["pack"]
    assert "pack" in body["error"]
    # 被引档还在
    assert client.get("/channels/base").status_code == 200
    # 删掉引用者后可以删除
    assert client.delete("/channels/pack").status_code == 204
    assert client.delete("/channels/base").status_code == 204


def test_save_group_with_dangling_ref_rejected_400(client):
    resp = client.post(
        "/channels",
        json={"name": "bad", "group": grp("series", {"ref": "ghost"}, seg())},
    )
    assert resp.status_code == 400
    assert "不存在" in resp.get_json()["error"]
    assert client.get("/channels/bad").status_code == 404


def test_inline_group_with_dangling_ref_rejected_400(client):
    resp = client.post(
        "/evaluate",
        json={"group": grp("series", {"ref": "ghost"}, seg()),
              "load_resistance": 1.0},
    )
    assert resp.status_code == 400
    assert "不存在" in resp.get_json()["error"]


def test_save_group_with_cycle_rejected_400(client):
    client.post("/channels", json={"name": "base", "channel": CHANNEL})
    client.post(
        "/channels",
        json={"name": "g1", "group": grp("series", {"ref": "base"}, seg())},
    )
    client.post(
        "/channels",
        json={"name": "g2", "group": grp("parallel", {"ref": "base"}, seg())},
    )
    client.post(
        "/channels",
        json={"name": "g1", "group": grp("series", {"ref": "g2"}, seg())},
    )
    resp = client.post(
        "/channels",
        json={"name": "g2", "group": grp("parallel", {"ref": "g1"}, seg())},
    )
    assert resp.status_code == 400
    assert "成环" in resp.get_json()["error"]


def test_self_cycle_rejected_400(client):
    resp = client.post(
        "/channels",
        json={"name": "loopy", "group": grp("series", {"ref": "loopy"}, seg())},
    )
    assert resp.status_code == 400
    assert "成环" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# 非法结构与定位
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("bad_group", "reason"),
    [
        ({"connection": "series", "members": []}, "不能为空"),
        ({"connection": "delta", "members": [{"segment": CHANNEL}]}, "connection"),
        (grp("series", seg(conductivity=-3.0)), "电导率"),
        (grp("series", seg(electrode_spacing=0.0)), "电极间距"),
    ],
)
def test_invalid_group_rejected_with_reason(client, bad_group, reason):
    resp = client.post(
        "/evaluate", json={"group": bad_group, "load_resistance": 1.0}
    )
    assert resp.status_code == 400
    assert reason in resp.get_json()["error"]


def test_invalid_nested_segment_error_carries_location(client):
    body = grp(
        "series",
        seg(),
        {"group": grp("parallel", seg(), seg(cross_section_area=0.0))},
    )
    resp = client.post("/evaluate", json={"group": body, "load_resistance": 1.0})
    assert resp.status_code == 400
    message = resp.get_json()["error"]
    assert "成员2" in message and "截面积" in message


def test_legacy_endpoints_unchanged(client):
    """老调用方：平铺登记、channel/profile 互斥、404 文案都不变。"""
    # 老的平铺写法仍然能用
    resp = client.post("/channels", json={"name": "flat", **CHANNEL})
    assert resp.status_code == 201
    assert resp.get_json()["channel"]["magnetic_flux_density"] == 2.0

    assert client.post(
        "/evaluate", json={"profile": "missing", "load_resistance": 1.0}
    ).status_code == 404

    resp = client.post(
        "/evaluate", json={"load_resistance": 1.0}
    )
    assert resp.status_code == 400

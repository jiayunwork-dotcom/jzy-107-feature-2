"""分组通道的 HTTP 接口行为：核算、扫描、档案引用、错误码、老口径兼容。"""
from __future__ import annotations

import concurrent.futures
import json

import pytest

from app import create_app

BASELINE = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return app.test_client()


# --------------------------------------------------------- 单点核算

def test_evaluate_single_segment_group_matches_legacy(client):
    for kind in ("series", "parallel"):
        group = {"type": kind, "members": [BASELINE]}
        data = client.post(
            "/evaluate", json={"group": group, "load_resistance": 0.2}
        ).get_json()
        legacy = client.post(
            "/evaluate", json={"channel": BASELINE, "load_resistance": 0.2}
        ).get_json()
        for field in (
            "open_circuit_voltage",
            "internal_resistance",
            "current",
            "power",
        ):
            assert data[field] == legacy[field]
        assert data["open_circuit_voltage"] == pytest.approx(200.0)
        assert data["internal_resistance"] == pytest.approx(0.2)
        assert len(data["segments"]) == 1
        segment = data["segments"][0]
        assert segment["path"] == "$.members[0]"
        assert segment["current"] == pytest.approx(500.0)
        assert segment["generated_power"] == pytest.approx(100_000.0)
        assert segment["dissipation"] == pytest.approx(50_000.0)
        assert segment["terminal_power"] == pytest.approx(50_000.0)
        assert len(data["groups"]) == 1


def test_evaluate_nested_group_reports_every_level(client):
    group = {
        "type": "series",
        "members": [
            {"type": "parallel", "id": "pblock", "members": [BASELINE, BASELINE]},
            BASELINE,
        ],
    }
    data = client.post(
        "/evaluate", json={"group": group, "load_resistance": 0.3}
    ).get_json()
    # E=200+200=400，r=0.1+0.2=0.3，I=400/0.6
    assert data["open_circuit_voltage"] == pytest.approx(400.0)
    assert data["internal_resistance"] == pytest.approx(0.3)
    assert data["current"] == pytest.approx(400 / 0.6)
    paths = {g["path"] for g in data["groups"]}
    assert paths == {"$", "$.members[0]"}
    sub = next(g for g in data["groups"] if g["path"] == "$.members[0]")
    assert sub["id"] == "pblock"
    assert sub["type"] == "parallel"
    assert sub["current"] == pytest.approx(400 / 0.6)
    assert [s["current"] for s in data["segments"]][:2] == pytest.approx(
        [400 / 1.2, 400 / 1.2]
    )


def test_evaluate_dead_segment_returns_null_resistance(client):
    dead = {**BASELINE, "conductivity": 0.0}
    data = client.post(
        "/evaluate",
        json={"group": {"type": "series", "members": [dead]}, "load_resistance": 1.0},
    ).get_json()
    assert data["internal_resistance"] is None
    assert data["current"] == 0.0
    assert data["power"] == 0.0
    assert data["segments"][0]["internal_resistance"] is None
    # 单段 σ=0 的展示电动势仍按 B*d*v 给出（弱极限，与老单段语义一致）
    assert data["open_circuit_voltage"] == pytest.approx(200.0)


def test_evaluate_inline_group_reference_must_exist(client):
    resp = client.post(
        "/evaluate",
        json={
            "group": {"type": "series", "members": [{"ref": "ghost"}]},
            "load_resistance": 1.0,
        },
    )
    assert resp.status_code == 400
    assert "不存在" in resp.get_json()["error"]


def test_source_fields_are_mutually_exclusive(client):
    resp = client.post(
        "/evaluate",
        json={"channel": BASELINE, "group": {"type": "series", "members": [BASELINE]},
              "load_resistance": 1.0},
    )
    assert resp.status_code == 400
    resp = client.post("/evaluate", json={"load_resistance": 1.0})
    assert resp.status_code == 400


# --------------------------------------------------------- 扫描

def test_sweep_group_peak_on_equivalent_resistance(client):
    group = {
        "type": "parallel",
        "members": [
            {"type": "series", "members": [BASELINE, BASELINE]},
            BASELINE,
        ],
    }
    data = client.post(
        "/sweep",
        json={"group": group, "load_min": 0.01, "load_max": 2.0, "points": 120},
    ).get_json()
    req = 0.4 * 0.2 / 0.6  # 0.4 与 0.2 并联
    assert data["internal_resistance"] == pytest.approx(req)
    assert data["peak"]["load_resistance"] == pytest.approx(req)
    assert data["peak"]["power"] == max(p["power"] for p in data["curve"])
    # 峰值点带完整逐段展开
    assert "segments" in data["peak"] and "groups" in data["peak"]


def test_sweep_curve_matches_evaluate_bitwise(client):
    group = {"type": "parallel", "members": [BASELINE, {**BASELINE,
             "magnetic_flux_density": 1.0}]}
    swept = client.post(
        "/sweep",
        json={"group": group, "load_min": 0.03, "load_max": 3.0, "points": 64},
    ).get_json()
    for point in swept["curve"]:
        single = client.post(
            "/evaluate",
            json={"group": group, "load_resistance": point["load_resistance"]},
        ).get_json()
        assert single["current"] == point["current"]
        assert single["power"] == point["power"]


def test_open_circuit_circulation_visible_at_http(client):
    """负载开路（load 远大于内阻的极限口径用无穷大负载表达）：
    并联两支路仍有大小相等方向相反的环流，组对外电流为零。

    HTTP 用 load_resistance=null 表示开路；不传则按正常负载处理。
    """
    group = {
        "type": "parallel",
        "members": [
            {**BASELINE, "magnetic_flux_density": 3.0},
            {**BASELINE, "magnetic_flux_density": 1.0},
        ],
    }
    resp = client.post(
        "/evaluate", json={"group": group, "load_resistance": None}
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["current"] == 0.0
    assert data["power"] == 0.0
    assert data["groups"][0]["current"] == pytest.approx(0.0, abs=1e-8)
    assert data["segments"][0]["current"] == pytest.approx(500.0)
    assert data["segments"][1]["current"] == pytest.approx(-500.0)


def test_sweep_dead_channel_is_all_zero_not_error(client):
    dead = {**BASELINE, "conductivity": 0.0}
    resp = client.post(
        "/sweep",
        json={
            "group": {"type": "series", "members": [dead, dead]},
            "load_min": 0.0,
            "load_max": 10.0,
            "points": 30,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["internal_resistance"] is None
    assert all(p["current"] == 0.0 and p["power"] == 0.0 for p in data["curve"])


# --------------------------------------------------------- 分组档

def test_group_profile_roundtrip_and_compute(client):
    assert client.post(
        "/channels",
        json={"name": "base", "channel": BASELINE},
    ).status_code == 201
    resp = client.post(
        "/channels",
        json={
            "name": "pack",
            "group": {"type": "parallel", "members": [{"ref": "base"}, {"ref": "base"}]},
        },
    )
    assert resp.status_code == 201, resp.get_json()
    listing = client.get("/channels").get_json()
    assert listing["profiles"] == ["base", "pack"]
    assert listing["groups"] == ["pack"]

    fetched = client.get("/channels/pack").get_json()
    assert fetched["group"]["members"] == [{"ref": "base"}, {"ref": "base"}]

    by_group = client.post(
        "/evaluate", json={"profile": "pack", "load_resistance": 0.1}
    ).get_json()
    inline = client.post(
        "/evaluate",
        json={
            "group": {"type": "parallel", "members": [BASELINE, BASELINE]},
            "load_resistance": 0.1,
        },
    ).get_json()
    assert by_group["current"] == inline["current"]
    assert by_group["power"] == inline["power"]


def test_group_profile_sweep(client):
    client.post("/channels", json={"name": "base", "channel": BASELINE})
    client.post(
        "/channels",
        json={"name": "pack",
              "group": {"type": "series", "members": [{"ref": "base"}]}},
    )
    data = client.post(
        "/sweep",
        json={"profile": "pack", "load_min": 0.01, "load_max": 2.0, "points": 80},
    ).get_json()
    assert data["peak"]["load_resistance"] == pytest.approx(0.2)


def test_dangling_group_profile_rejected(client):
    resp = client.post(
        "/channels",
        json={"name": "bad",
              "group": {"type": "series", "members": [{"ref": "ghost"}]}},
    )
    assert resp.status_code == 400
    assert "ghost" in resp.get_json()["error"]
    assert client.get("/channels/bad").status_code == 404


def test_cycle_group_profile_rejected_and_old_version_kept(client):
    client.post("/channels", json={"name": "base", "channel": BASELINE})
    client.post(
        "/channels",
        json={"name": "a", "group": {"type": "series", "members": [{"ref": "base"}]}},
    )
    client.post(
        "/channels",
        json={"name": "b", "group": {"type": "series", "members": [{"ref": "a"}]}},
    )
    resp = client.post(
        "/channels",
        json={"name": "a", "group": {"type": "series", "members": [{"ref": "b"}]}},
    )
    assert resp.status_code == 400
    assert "成环" in resp.get_json()["error"]
    # 旧的 a 仍可正常核算
    data = client.post(
        "/evaluate", json={"profile": "a", "load_resistance": 0.2}
    ).get_json()
    assert data["current"] == pytest.approx(500.0)


def test_delete_referenced_profile_conflicts_and_lists_referrers(client):
    client.post("/channels", json={"name": "base", "channel": BASELINE})
    client.post(
        "/channels",
        json={"name": "pack",
              "group": {"type": "parallel", "members": [{"ref": "base"}]}},
    )
    resp = client.delete("/channels/base")
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["referrers"] == ["pack"]
    assert "pack" in body["error"]
    # 档还在
    assert client.get("/channels/base").status_code == 200


def test_expanded_limits_enforced_through_references(client):
    client.post("/channels", json={"name": "s", "channel": BASELINE})
    client.post(
        "/channels",
        json={"name": "big",
              "group": {"type": "parallel", "members": [{"ref": "s"}] * 51}},
    )
    # 内联并列引用两次：102 段，超过 100 上限
    resp = client.post(
        "/evaluate",
        json={
            "group": {"type": "series", "members": [{"ref": "big"}, {"ref": "big"}]},
            "load_resistance": 1.0,
        },
    )
    assert resp.status_code == 400
    assert "总段数超过上限" in resp.get_json()["error"]


def test_dynamic_reference_consistent_across_endpoints(client):
    """被引档更新后，单点、扫描、查看档案三处口径一致：结构看存档、数值看新值。"""
    client.post("/channels", json={"name": "base", "channel": BASELINE})
    client.post(
        "/channels",
        json={"name": "pack",
              "group": {"type": "parallel", "members": [{"ref": "base"}]}},
    )
    client.post(
        "/channels", json={"name": "base", "channel": {**BASELINE, "magnetic_flux_density": 4.0}}
    )
    evaluate = client.post(
        "/evaluate", json={"profile": "pack", "load_resistance": 0.2}
    ).get_json()
    sweep = client.post(
        "/sweep",
        json={"profile": "pack", "load_min": 0.01, "load_max": 2.0, "points": 40},
    ).get_json()
    archive = client.get("/channels/pack").get_json()
    assert evaluate["open_circuit_voltage"] == pytest.approx(400.0)
    assert sweep["open_circuit_voltage"] == pytest.approx(400.0)
    # 查看档案返回的仍是保存时写的引用结构（不是快照值）
    assert archive["group"]["members"] == [{"ref": "base"}]


def test_inline_group_validation_errors_over_http(client):
    cases = [
        ({"type": "series", "members": []}, "空组"),
        ({"type": "delta", "members": [BASELINE]}, "series 或 parallel"),
        ({"type": "series", "members": [{**BASELINE, "cross_section_area": -1.0}]},
         r"members\[0\].*截面积"),
    ]
    for group, reason in cases:
        resp = client.post(
            "/evaluate", json={"group": group, "load_resistance": 1.0}
        )
        assert resp.status_code == 400, (group, resp.get_json())
        assert reason.replace("\\", "") in resp.get_json()["error"] or "members[0]" in resp.get_json()["error"]


# --------------------------------------------------------- 并发

def test_concurrent_save_delete_reference_leaves_no_dangling_http(app):
    def run(worker_id):
        client = app.test_client()
        for i in range(100):
            seg_name = f"s{worker_id % 5}"
            client.post("/channels", json={"name": seg_name, "channel": BASELINE})
            client.post(
                "/channels",
                json={
                    "name": f"p{worker_id % 4}",
                    "group": {"type": "parallel", "members": [{"ref": seg_name}]},
                },
            )
            client.delete(f"/channels/{seg_name}")
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))

    client = app.test_client()
    listing = client.get("/channels").get_json()
    # 每个留存的分组档都能成功核算（引用必然可解析）
    for name in listing["groups"]:
        fetched = client.get(f"/channels/{name}").get_json()
        resp = client.post(
            "/evaluate",
            json={"group": fetched["group"], "load_resistance": 1.0},
        )
        assert resp.status_code == 200, (name, resp.get_json())


# --------------------------------------------------------- 老口径回归

@pytest.mark.parametrize(
    "group",
    [
        "not-an-object",
        [1, 2],
        {"type": "series", "members": ["segment"]},
        {"type": 123, "members": [BASELINE]},
        {"type": "series", "members": [{"ref": ""}]},
        {"type": "series", "members": [{"ref": 123}]},
    ],
)
def test_malformed_group_rejected_with_400(client, group):
    resp = client.post(
        "/evaluate", json={"group": group, "load_resistance": 1.0}
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"]


def test_non_json_body_rejected(client):
    resp = client.post(
        "/evaluate",
        data="not json",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 400


def test_group_profile_requires_exactly_one_payload(client):
    assert client.post(
        "/channels", json={"group": {"type": "series", "members": [BASELINE]}}
    ).status_code == 400  # 缺 name
    assert client.post(
        "/channels",
        json={"name": "both", "channel": BASELINE,
              "group": {"type": "series", "members": [BASELINE]}},
    ).status_code == 400
    assert client.post("/channels", json={"name": "neither"}).status_code == 400


def test_reverse_fed_segment_fields(client):
    """较弱的并联段被倒灌：电流、发出功率为负，损耗为正，净输出为负。"""
    group = {
        "type": "parallel",
        "members": [
            {**BASELINE, "magnetic_flux_density": 3.0},
            {**BASELINE, "magnetic_flux_density": 1.0},
        ],
    }
    data = client.post(
        "/evaluate", json={"group": group, "load_resistance": 0.5}
    ).get_json()
    weak = data["segments"][1]
    assert weak["current"] < 0
    assert weak["generated_power"] < 0
    assert weak["dissipation"] > 0
    assert weak["terminal_power"] < 0
    # 强段发出功率 = 弱段被充功率 + 两段损耗 + 负载功率
    strong = data["segments"][0]
    assert strong["generated_power"] + weak["generated_power"] == pytest.approx(
        strong["dissipation"] + weak["dissipation"] + data["power"], rel=1e-9
    )


def test_overwrite_group_profile_replaces_structure(client):
    client.post("/channels", json={"name": "s", "channel": BASELINE})
    client.post(
        "/channels",
        json={"name": "g", "group": {"type": "series", "members": [{"ref": "s"}]}},
    )
    client.post(
        "/channels",
        json={"name": "g", "group": {"type": "parallel", "members": [{"ref": "s"}]}},
    )
    fetched = client.get("/channels/g").get_json()
    assert fetched["group"]["type"] == "parallel"


def test_null_load_rejected_for_legacy_channel(client):
    """null 开路只对分组口径开放；老单段口径的负载校验保持原样。"""
    resp = client.post(
        "/evaluate", json={"channel": BASELINE, "load_resistance": None}
    )
    assert resp.status_code == 400


def test_responses_are_strict_json_even_when_disconnected(client):
    """全断/开路等边界下响应里不允许出现 NaN/Infinity（那不是合法 JSON）。"""
    dead = {**BASELINE, "conductivity": 0.0}
    cases = [
        {"type": "series", "members": [dead]},
        {"type": "parallel", "members": [dead, dead]},
        {"type": "series", "members": [
            {"type": "parallel", "members": [dead, dead]}, dead,
        ]},
    ]
    for group in cases:
        for load in (0.0, 1.0, None):
            text = client.post(
                "/evaluate", json={"group": group, "load_resistance": load}
            ).get_data(as_text=True)
            assert "NaN" not in text and "Infinity" not in text
            json.loads(text)
        text = client.post(
            "/sweep",
            json={"group": group, "load_min": 0.0, "load_max": 5.0, "points": 13},
        ).get_data(as_text=True)
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)


def test_legacy_single_segment_endpoints_unchanged(client):
    # 登记/列表/读取/删除老字段不变
    assert client.post(
        "/channels", json={"name": "base", "channel": BASELINE}
    ).get_json() == {"name": "base", "channel": BASELINE}
    assert client.get("/channels/base").get_json() == {"name": "base", "channel": BASELINE}
    # 列表里多了 groups 字段，但 profiles 仍是全部档名（老字段保留）
    assert client.get("/channels").get_json()["profiles"] == ["base"]
    assert client.post(
        "/evaluate", json={"channel": BASELINE, "load_resistance": 0.2}
    ).get_json() == {
        "open_circuit_voltage": 200.0,
        "internal_resistance": 0.2,
        "load_resistance": 0.2,
        "current": 500.0,
        "power": 50_000.0,
    }
    assert client.delete("/channels/base").status_code == 204

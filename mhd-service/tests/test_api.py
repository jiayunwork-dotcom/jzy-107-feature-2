"""HTTP 接口测试：单点核算、负载扫描、通道档存取与隔离。"""
import concurrent.futures

import pytest

from app import create_app

BASELINE_CHANNEL = {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5,
}
# 对应 U_oc = 200 V，R_int = 0.2 Ω，P_max = 50 kW

OTHER_CHANNEL = {
    "magnetic_flux_density": 1.5,
    "electrode_spacing": 0.8,
    "flow_velocity": 120.0,
    "conductivity": 5.0,
    "cross_section_area": 0.4,
}
# 对应 U_oc = 144 V，R_int = 0.4 Ω


@pytest.fixture()
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_evaluate_with_inline_channel(client):
    resp = client.post(
        "/evaluate", json={"channel": BASELINE_CHANNEL, "load_resistance": 0.2}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["open_circuit_voltage"] == pytest.approx(200.0)
    assert data["internal_resistance"] == pytest.approx(0.2)
    assert data["current"] == pytest.approx(500.0)
    assert data["power"] == pytest.approx(50_000.0)


def test_sweep_peak_at_internal_resistance(client):
    resp = client.post(
        "/sweep",
        json={
            "channel": BASELINE_CHANNEL,
            "load_min": 0.01,
            "load_max": 20.0,
            "points": 200,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["curve"]) >= 200
    assert data["peak"]["load_resistance"] == pytest.approx(0.2)
    assert data["peak"]["power"] == pytest.approx(50_000.0)
    # 曲线峰值就是采样点里的最大功率
    assert data["peak"]["power"] == max(p["power"] for p in data["curve"])


def test_evaluate_and_sweep_agree_at_same_load(client):
    """单点接口与扫描接口在同一负载处结果必须一致。"""
    sweep = client.post(
        "/sweep",
        json={
            "channel": BASELINE_CHANNEL,
            "load_min": 0.05,
            "load_max": 1.0,
            "points": 96,
        },
    ).get_json()
    point = sweep["curve"][37]  # 任取曲线中一点
    single = client.post(
        "/evaluate",
        json={"channel": BASELINE_CHANNEL, "load_resistance": point["load_resistance"]},
    ).get_json()
    assert single["current"] == point["current"]
    assert single["power"] == point["power"]


def test_channel_profile_roundtrip(client):
    assert client.post(
        "/channels", json={"name": "baseline", "channel": BASELINE_CHANNEL}
    ).status_code == 201
    assert client.get("/channels").get_json()["profiles"] == ["baseline"]
    fetched = client.get("/channels/baseline").get_json()
    assert fetched["channel"] == BASELINE_CHANNEL

    # 用通道档做单点核算，与内联参数结果一致
    by_profile = client.post(
        "/evaluate", json={"profile": "baseline", "load_resistance": 0.2}
    ).get_json()
    inline = client.post(
        "/evaluate", json={"channel": BASELINE_CHANNEL, "load_resistance": 0.2}
    ).get_json()
    assert by_profile == inline

    assert client.delete("/channels/baseline").status_code == 204
    assert client.get("/channels/baseline").status_code == 404


def test_missing_profile_returns_404(client):
    resp = client.post("/evaluate", json={"profile": "ghost", "load_resistance": 1.0})
    assert resp.status_code == 404
    assert "通道档不存在" in resp.get_json()["error"]


@pytest.mark.parametrize(
    "override",
    [
        {"magnetic_flux_density": -1.0},
        {"electrode_spacing": 0.0},
        {"flow_velocity": -1.0},
        {"conductivity": -1.0},
        {"cross_section_area": 0.0},
    ],
)
def test_invalid_channel_rejected_over_http(client, override):
    bad = {**BASELINE_CHANNEL, **override}
    resp = client.post("/evaluate", json={"channel": bad, "load_resistance": 1.0})
    assert resp.status_code == 400
    assert resp.get_json()["error"]


def test_invalid_sweep_request_rejected_over_http(client):
    resp = client.post(
        "/sweep",
        json={
            "channel": BASELINE_CHANNEL,
            "load_min": 10.0,
            "load_max": 1.0,
            "points": 50,
        },
    )
    assert resp.status_code == 400
    assert "上限必须大于下限" in resp.get_json()["error"]

    resp = client.post(
        "/sweep",
        json={"channel": BASELINE_CHANNEL, "load_min": 0.0, "load_max": 1.0, "points": 1},
    )
    assert resp.status_code == 400
    assert "采样点个数" in resp.get_json()["error"]


def test_zero_conductivity_over_http(client):
    channel = {**BASELINE_CHANNEL, "conductivity": 0.0}
    resp = client.post("/evaluate", json={"channel": channel, "load_resistance": 1.0})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["internal_resistance"] is None  # 无穷大序列化为 null
    assert data["current"] == 0.0
    assert data["power"] == 0.0


def test_two_profiles_swept_concurrently_do_not_mix(app, client):
    """两套通道档同时扫描，两条功率曲线各归各的。"""
    client.post("/channels", json={"name": "baseline", "channel": BASELINE_CHANNEL})
    client.post("/channels", json={"name": "other", "channel": OTHER_CHANNEL})

    def sweep_profile(name):
        # 每个线程用独立的 test client，避免线程间共享连接状态
        resp = app.test_client().post(
            "/sweep",
            json={"profile": name, "load_min": 0.01, "load_max": 5.0, "points": 100},
        )
        assert resp.status_code == 200
        return resp.get_json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(sweep_profile, name) for name in ["baseline", "other"] * 4]
        results = [f.result() for f in futures]

    baselines = [r for r in results if r["open_circuit_voltage"] == pytest.approx(200.0)]
    others = [r for r in results if r["open_circuit_voltage"] == pytest.approx(144.0)]
    assert len(baselines) == 4 and len(others) == 4
    for data in baselines:
        assert data["peak"]["load_resistance"] == pytest.approx(0.2)
    for data in others:
        assert data["peak"]["load_resistance"] == pytest.approx(0.4)

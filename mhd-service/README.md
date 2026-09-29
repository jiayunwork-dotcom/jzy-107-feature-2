# MHD 通道核算服务

法拉第式磁流体（MHD）发电通道的电学性能常驻核算服务。导电流体在磁场中横穿
通道时感应出开路电压，服务核算接上不同负载后的电流与提取功率，并能对一段
负载电阻区间做扫描，给出功率随负载变化的曲线并标出峰值位置。

纯 HTTP 接口，不带页面，只管通道本身的感应与负载匹配。

## 物理模型

| 量 | 公式 | 说明 |
| --- | --- | --- |
| 开路电压 `U_oc` | `B · d · v` | 体感应电动势（非霍尔电压） |
| 通道内阻 `R_int` | `d / (σ · A)` | σ 为流体电导率，A 为通道截面积 |
| 回路电流 `I` | `U_oc / (R_int + R_load)` | |
| 提取功率 `P` | `I² · R_load` | 负载上取出的功率 |

最大功率传输发生在 `R_load = R_int`，此时 `P_max = U_oc² / (4 · R_int)`。
电导率为零时内阻无穷大，电流与功率都为零（扫描不会因此除零崩溃）。

## 目录结构（按职责分文件）

```
app.py               应用入口：装配 Flask 应用
mhd/induction.py     感应计算：开路电压与内阻
mhd/load_solver.py   单负载求解：回路电流与提取功率
mhd/sweep.py         负载区间扫描与功率峰值定位
mhd/profiles.py      通道档存取（进程内，线程安全，不跨重启保留）
mhd/validation.py    输入合法性拦截（计算前打回非法参数）
mhd/routes.py        HTTP 层：只做请求收发与调度
tests/               自动化测试（物理关系 + 接口行为）
```

扫描与单点核算共用同一个求解函数 `solve_load`，保证同一负载处两种口径
结果完全一致。扫描时若内阻落在区间内，会把它补进采样点，使曲线峰值
恰好落在 `R_load = R_int`。

## 快速开始

### Docker（一条命令）

```bash
docker compose up --build
```

或分两步：

```bash
docker build -t mhd-service .
docker run --rm -p 8000:8000 mhd-service
```

服务监听 `8000` 端口。

### 本地运行与测试

```bash
pip install -r requirements-dev.txt
python app.py        # 启动服务
python -m pytest -v  # 运行测试
```

## 接口

通道参数（JSON 字段）：

| 字段 | 含义 | 单位 | 约束 |
| --- | --- | --- | --- |
| `magnetic_flux_density` | 磁感应强度 B | T | ≥ 0 |
| `electrode_spacing` | 电极间距 d | m | > 0 |
| `flow_velocity` | 流体速度 v | m/s | ≥ 0 |
| `conductivity` | 流体电导率 σ | S/m | ≥ 0 |
| `cross_section_area` | 通道截面积 A | m² | > 0 |

### `POST /evaluate` — 单点核算

请求里用 `channel` 内联通道参数，或用 `profile` 引用已登记的通道档
（二者必须且只能给其一），外加 `load_resistance`（≥ 0，0 即短路）：

```bash
curl -X POST http://localhost:8000/evaluate -H 'Content-Type: application/json' -d '{
  "channel": {
    "magnetic_flux_density": 2.0,
    "electrode_spacing": 1.0,
    "flow_velocity": 100.0,
    "conductivity": 10.0,
    "cross_section_area": 0.5
  },
  "load_resistance": 0.2
}'
```

```json
{
  "open_circuit_voltage": 200.0,
  "internal_resistance": 0.2,
  "load_resistance": 0.2,
  "current": 500.0,
  "power": 50000.0
}
```

电导率为零时 `internal_resistance` 为 `null`（无穷大无法写入 JSON），
`current` 与 `power` 为 `0.0`。

### `POST /sweep` — 负载扫描

在 `evaluate` 的通道参数之外，给定负载区间与采样密度
（`load_min` ≥ 0，`load_max` > `load_min`，`points` 为 2 到 100000 的整数）：

```bash
curl -X POST http://localhost:8000/sweep -H 'Content-Type: application/json' -d '{
  "profile": "baseline",
  "load_min": 0.01,
  "load_max": 20.0,
  "points": 200
}'
```

```json
{
  "open_circuit_voltage": 200.0,
  "internal_resistance": 0.2,
  "curve": [{"load_resistance": 0.01, "current": 952.38, "power": 9070.3}, "..."],
  "peak": {"load_resistance": 0.2, "current": 500.0, "power": 50000.0}
}
```

### `POST /channels` — 登记通道档

```bash
curl -X POST http://localhost:8000/channels -H 'Content-Type: application/json' -d '{
  "name": "baseline",
  "channel": { "magnetic_flux_density": 2.0, "electrode_spacing": 1.0,
               "flow_velocity": 100.0, "conductivity": 10.0, "cross_section_area": 0.5 }
}'
```

另有 `GET /channels`（列出名称）、`GET /channels/<name>`（读取）、
`DELETE /channels/<name>`（删除）。通道档只保存在进程内，重启即清空。

### 错误响应

非法输入在计算前被打回，HTTP 400，响应里带原因：

```json
{"error": "磁感应强度不能为负，收到: -2.0"}
```

引用不存在的通道档返回 404。

## 基准算例（手算核对）

B = 2 T、d = 1 m、v = 100 m/s、σ = 10 S/m、A = 0.5 m²：

- 开路电压 `U_oc = 2 × 1 × 100 = 200 V`（百伏量级）
- 内阻 `R_int = 1 / (10 × 0.5) = 0.2 Ω`
- 负载匹配时 `I = 200 / 0.4 = 500 A`，`P_max = 500² × 0.2 = 50 kW`

该算例与下列电学关系一起钉在 `tests/` 的回归测试里：

1. 磁感应强度翻倍 → 开路电压翻倍；流速翻倍 → 开路电压同样翻倍
2. 扫描功率曲线峰值落在 `R_load = R_int`；负载改为 `2·R_int` 时功率下降
3. 电导率为零 → 电流为零、功率为零
4. 单点核算与扫描曲线在同一负载处结果完全一致

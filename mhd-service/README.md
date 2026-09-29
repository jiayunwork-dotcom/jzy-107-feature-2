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
mhd/load_solver.py   单负载求解：回路电流与提取功率（单段老口径）
mhd/sweep.py         负载区间扫描与功率峰值定位（单段老口径）
mhd/groups.py        分组实体树类型与引用展开
mhd/group_validation.py  分组树输入校验（位置化错误、上限）
mhd/linalg.py        最小度排序 + 稀疏 LDL^T 线性求解原语
mhd/circuit.py       分组树 -> 一般电路网络，统一求解（无串/并特例公式）
mhd/group_sweep.py   分组通道负载扫描（与单点共用同一次网络解）
mhd/profiles.py      通道档存取（单段/分组同一名字空间，线程安全）
mhd/validation.py    输入合法性拦截（计算前打回非法参数）
mhd/routes.py        HTTP 层：只做请求收发与调度
tests/               自动化测试（物理关系 + 接口行为 + 引用/并发）
DESIGN.md            分组通道的建模选型、替代方案与口径说明（必读）
```

单段通道的扫描与单点核算共用 `solve_load`；分组通道则整区间只解两到
三次网络矩阵，每个采样点是同一次解上的仿射组合，因此两种口径都保证
「同一负载处扫描曲线与单点核算逐位一致」。扫描时若等效内阻落在区间内，
会把它补进采样点，使曲线峰值恰好落在 R_load = R_int。

## 分组通道（分段电极）

一条通道可以由很多段拼成，用一棵分组树描述。叶子是一段（字段就是上面
五项），中间节点是组：

```json
{
  "type": "series",
  "id": "可选标注",
  "members": [
    { "type": "parallel", "members": [ {段}, {段} ] },
    { "ref": "已登记的通道档名" },
    {段}
  ]
}
```

- `type`：`series`（串联）或 `parallel`（并联），只有一段时两者等价；
- 组成员既可以是段，也可以是另一个组，层数不固定（上限 16 层、100 段）；
- 成员也可以是 `{"ref": "名字"}`，引用已登记的单段档或分组档；
- 每个节点可以带可选 `id`，结果里原样带回，方便对应实物电极。

算法**不按串/并各写一套等效公式**，而是把整棵树建成一般电路网络统一
求解（每段是电动势串内阻的诺顿支路，串/并只决定接线拓扑），段间互相
灌电流、开路环流都直接是网络解。建模选型、对 σ=0 断头的处理与替代
方案对比见 [DESIGN.md](DESIGN.md)。

### `POST /evaluate` 对分组的返回

请求体用 `group` 内联分组树，或用 `profile` 指向一个分组档
（`channel`/`group`/`profile` 三者必须且只能给一个）。顶层仍是
等效电动势、等效内阻、负载电流、功率；额外给出 `segments`（逐段）
与 `groups`（逐组，含根组自身）：

```json
{
  "open_circuit_voltage": 192.3,
  "internal_resistance": 0.15,
  "load_resistance": 0.2,
  "current": 549.4,
  "power": 60372.9,
  "segments": [
    {
      "path": "$.members[0].members[0]",
      "id": null,
      "open_circuit_voltage": 300.0,
      "internal_resistance": 0.2,
      "current": 846.1,
      "generated_power": 253846.1,
      "dissipation": 143195.3,
      "terminal_power": 110650.8
    }
  ],
  "groups": [
    {"path": "$", "id": null, "type": "parallel",
     "current": 549.4, "voltage": 82.4, "power": 45270.3}
  ]
}
```

- 段 `current` 以该段电动势方向为正，**被倒灌的段为负**；
  `generated_power` 为该段电动势发出的功率（倒灌时为负），
  `dissipation` 是内阻损耗，`terminal_power` 是对外净输出；
- 组 `current` 为从组正极流出的电流，`voltage` 为组两端电压；
- `path` 标明这一段/组在树里的位置（`$.members[i]...`），错误信息也用
  同样的路径指出是哪个位置的哪一项；
- 段内阻为无穷大（σ=0）时 `internal_resistance` 为 `null`；
- `load_resistance` 传 `null`（仅分组口径）表示负载开路：对外电流为零，
  并联支路间的环流照常给出，这是看清互灌的最直接方式。

### `POST /sweep` 对分组的返回

`curve` 仍是 `{load_resistance, current, power}` 列表；`peak` 在曲线
字段之外，还带峰值负载处完整的逐段、逐组展开。等效内阻无穷大（整条
通道被 σ=0 段断开）时不报错、不除零，曲线电流与功率全为零。

### 分组档

分组档与单段档在同一个名字空间，用 `POST /channels` 的 `group` 字段
登记（与 `channel` 二选一）：

```bash
curl -X POST http://localhost:8000/channels -H 'Content-Type: application/json' -d '{
  "name": "front-pack",
  "group": {"type": "series", "members": [{"ref": "seg-a"}, {"ref": "seg-b"}]}
}'
```

- `GET /channels` 返回的 `profiles` 字段含义不变（全部档名），
  另有 `groups` 列出哪些是分组档；`GET /channels/<name>` 对分组档
  返回登记时的原始结构（引用保持为 `ref`）；
- 引用在**核算时动态展开**：被引档重新保存后，单点核算与扫描立刻按新
  参数计算，常用组合只维护一份；查看档案始终看到登记时的引用结构；
- 保存时悬空引用（指向不存在的档）、引用成环（含自引用）一律 400 拒收；
- 删除被引用着的档返回 409，并在 `referrers` 里列出全部引用方；
- 多个客户端并发保存/删除/引用时，删除与引用不会交错出悬空档。

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

另有 `GET /channels`（列出名称，`profiles` 为全部档名、`groups` 标出
分组档）、`GET /channels/<name>`（单段档返回 `channel`，分组档返回
`group` 原始结构）、`DELETE /channels/<name>`（被引用时返回 409）。
通道档只保存在进程内，重启即清空。分组档的引用规矩见上文「分组通道」。

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

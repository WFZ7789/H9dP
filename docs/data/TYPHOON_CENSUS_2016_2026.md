# 2016–2026 年西北太平洋台风：浮标与 JMA/MRI 飞机实测气压普查

状态：**普查与两项实测采集已完成；全论坛逐风暴穷举未完成。**
本文区分计划、实际执行和未完成部分，不把候选关联写成已证实的台风中心观测，
不使用任何 best-track（BST）中心气压作为实测值或配额填充。

- 交付 JSON：`/mnt/data/H9dP/datasets/typhoon_census_2016_2026_20261115/census.json`
- 生成时间：**2026-11-15T10:45:50.837833Z**（最后一次重跑）
- 气旋范围：2016–2026 年 WNP **283 个编号风暴**（2026 年截至 2026-11-15）
- SHA-256：`census.json` = `454707f667da6e904a39c01a61cd238cc406502002c0e248ee7f612fcca30f7b`
  （以最终目录内文件为准；重跑会改变该值）

## 1. 硬约束执行情况

| 约束 | 实际执行 |
| --- | --- |
| 不用 BST 中心气压作为实测值 | **执行**：结果中 `best_track_pressure_fields_in_result = 0`；所有气压记录 `pressure_is_best_track = false`。 |
| 不用 BST 填补配额 | **执行**：没有按气旋/年份配额抽样；69 个风暴没有浮标候选记录，保留为空；3 个飞机任务无数值，保留 `pressure_hpa = null`。 |
| 不用 BST 风/位置 | **执行**：浮标关联只用风暴活动窗口 + 同一浮标自身的前期气压基线，不用轨迹位置或距离。 |
| 不把浮标/站点气压冒充中心气压 | **执行**：浮标值标为 `measured sea-level pressure at drifting buoy`，不是 `Pc`。飞机值分原生探空与已发表反演值，未验证中心代表性时标 `central_pressure_verified = false`。 |
| 计划/完成、真实/合成分开 | **执行**：本文只报告真实采集；没有合成观测，没有把计划写成结果。 |

**唯一进入结果的官方风暴元数据**是 Digital Typhoon WNP 年表的编号、名称、生成/消散时刻。
该页面同时列有 BST 最低气压；解析器只读取前四类字段，**最低气压列被显式忽略**。
生成/消散时刻本身是官方风暴索引元数据，不是观测；它们只用于确定关联时间窗。
没有读取或写入任何 BST 中心气压、风速或轨迹位置。

## 2. 台风总体

| 年份 | 编号风暴 | 有浮标候选 | 浮标候选对数 | 浮标时序记录 | 有飞机数值 | 两者皆无 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 26 | 20 | 61 | 6,308 | 0 | 6 |
| 2017 | 27 | 26 | 72 | 5,744 | 1 | 1 |
| 2018 | 29 | 27 | 82 | 8,110 | 1 | 2 |
| 2019 | 29 | 22 | 81 | 7,225 | 0 | 7 |
| 2020 | 23 | 18 | 49 | 3,567 | 0 | 5 |
| 2021 | 22 | 20 | 69 | 6,872 | 0 | 2 |
| 2022 | 25 | 15 | 19 | 1,314 | 1 | 10 |
| 2023 | 17 | 14 | 27 | 2,988 | 0 | 3 |
| 2024 | 26 | 26 | 45 | 3,388 | 0 | 0 |
| 2025 | 27 | 24 | 72 | 6,977 | 2 | 3 |
| 2026 | 32 | 2 | 3 | 237 | 0 | 30 |
| **合计** | **283** | **214** | **580** | **52,730** | **5** | **69** |

2026 年只有 JMA 浮标页列出的 `25-13`、`25-14` 两个部署，且季节尚未结束；
2026 年 32 个风暴中 30 个没有浮标候选，不是用旧数据补齐。

## 3. 浮标实测气压

### 3.1 来源与实际下载

官方来源：**JMA 漂流型海洋気象ブイロボット**逐小时 TXT 数据，
`https://www.data.jma.go.jp/gmd/kaiyou/data/db/vessel_obs/data-report/data/{year}/buoy/txt/{id}.TXT`。

- 实际下载 **169 个年度部署 TXT**，覆盖 2016–2026 年部署表。
- 解析出 **186,834 条**去重后的逐小时有效气压记录（这是全部可用记录，不是入选数）。
- 观测级字段：WMO 编号、UTC 时刻、经纬度、气压（hPa）、SST、波高、周期、原始行号、原始 URL。
- 缺失气压留空，不填 0，不插值，不把 99999/9999 当有效值。
- 浮标气压是**浮标所在位置的海平面气压**；不是台风中心气压，也没有声称是眼区最低气压。

### 3.2 关联方法与诚实边界

对每个风暴，取 `[生成-6 h, 消散+6 h]` 内同一浮标的气压序列，
以 `[生成-72 h, 生成-24 h]` 的均值作基线（不足 4 条时回退到 `[-48 h, -6 h]`）。
只有最低气压比基线低 **≥4 hPa** 才建立候选关联。关联**不**使用风暴轨迹位置或距离，
因此这不是空间匹配，可能把同一时段远处的天气扰动算进来；JSON 用 `association_confidence`
和 `attribution_ambiguity` 明确标出。

| 置信度 | 规则 | 候选对数 | 涉及风暴 | 唯一物理最低值 |
| --- | --- | ---: | ---: | ---: |
| high | 压差 ≥12 hPa，且无并发风暴 | 140 | 79 | 140 |
| medium | 压差 7–12 hPa，或高压差但有并发 | 175 | 110 | 151 |
| low | 压差 4–7 hPa，或有并发且压差 <12 | 265 | 155 | 225 |

- **580** 个平台-风暴候选对，去重后 **511** 个唯一“WMO+时刻+气压”最低观测。
- **248** 个候选对存在同一时刻的并发风暴，已降级置信度，并写入 `concurrent_storm_ids`。
- 浮标最低气压范围 **964.6–1016.9 hPa**；这只是浮标测得值，不是中心气压。
- 109 个年度部署 ID、19 个 WMO 号（JMA 会复用 WMO 号，故物理部署数以部署 ID 计）。
- 同一浮标跨年出现在两个年度文件时，按 `WMO+时刻+气压+位置` 去重，避免 58 组重复计数。

高置信示例（保留实际测得值，不作为中心气压）：

| 风暴 | 年份 | 平台 | 最低气压 | 压差 | UTC 时刻 |
| --- | ---: | --- | ---: | ---: | --- |
| PRAPIROON | 2018 | 18-09 | 964.6 | 46.39 | 2018-07-02T20:00Z |
| KALMAEGI | 2025 | 25-06 | 986.0 | 44.36 | 2025-11-01T14:00Z |
| PHANFONE | 2019 | 19-15 | 976.3 | 44.01 | 2019-12-27T08:00Z |
| SURIGAE | 2021 | 21-04 | 998.4 | 36.22 | 2021-04-19T01:00Z |
| KUJIRA | 2020 | 20-09 | 981.1 | 35.62 | 2020-09-29T16:00Z |
| NYATOH | 2021 | 21-13 | 996.1 | 35.26 | 2021-11-30T19:00Z |
| KROSA | 2019 | 19-09 | 975.2 | 34.98 | 2019-08-14T10:00Z |

**未完成/未采用：** NDBC 的 TAO/KMA/太平洋岛屿浮标曾尝试补充，
但 2025–2026 年多個 `historical/stdmet` 请求被对端重置或超时，未取得可审计的完整文件；
本批不把失败请求写成浮标记录。JMA 浮标是唯一完成的浮标网络。

## 4. JMA/MRI 飞机与投落送实测气压

JMA/MRI 的 2025 年台风委员会报告确认了 T-PARCII/JMA 航空任务史和 GTS `TEMP` 分发方式：
[20IWS Takahashi 2025](https://typhooncommittee.org/20IWS/docs/Technical%20Presentations/6.%20Takahashi_IWS20_v20251127%202%201.pdf)。
该报告未给 HAISHEN/MINDULLE/BARIJAT 的数值。

这里必须区分机构事实：2017–2025 年的飞机任务属于 **JMA 气象研究所（MRI）与大学共同参与的 T-PARCII**，
飞机由 Diamond Air Service 的 Gulfstream II 运营，不是 JMA 常设业务侦察机队。
因此字段使用 `T-PARCII ... (JMA/MRI collaboration)`，不写成“JMA 业务飞机”。

### 4.1 原生投落送气压廓线（8 条，实际下载）

来源：Zenodo 记录 [10798896](https://zenodo.org/records/10798896)，
Tsukada et al. (2024) 的 T-PARCII 眼区投落送 Level 2 数据。

| 风暴 | 剖面 | 发射时刻 UTC | 最低层气压 (hPa) | 纬度 | 经度 |
| --- | --- | --- | ---: | ---: | ---: |
| LAN 2017 | S7099015 | 2017-10-21T05:30:49Z | 921.6 | 22.113 | 131.815 |
| LAN 2017 | S7099025 | 2017-10-21T06:27:30Z | 925.7 | 22.454 | 131.999 |
| LAN 2017 | S7099023 | 2017-10-21T06:39:42Z | 920.8 | 22.546 | 132.089 |
| LAN 2017 | S7099012 | 2017-10-21T06:41:48Z | 927.0 | 22.746 | 132.201 |
| NANMADOL 2022 | S1099042 | 2022-09-17T00:46:18Z | 934.1 | 26.109 | 132.963 |
| NANMADOL 2022 | S1099043 | 2022-09-17T00:46:52Z | 911.1 | 26.121 | 133.040 |
| NANMADOL 2022 | S2099042 | 2022-09-17T01:19:34Z | 911.5 | 26.157 | 132.932 |
| NANMADOL 2022 | S2099043 | 2022-09-17T01:20:07Z | 912.6 | 26.162 | 132.974 |

这些是**眼区剖面最后一层有效气压**。最低层可能仍高于海面几米到几十米；
因此保留 `central_pressure_verified = false`，不把 920.8/911.1 直接称为官方中心气压。
8 条剖面均有完整层数、压力、风、湿度，原始 CSV 在
`raw/aircraft/zenodo_10798896/`。

### 4.2 已发表/公开的飞机反演气压（8 条）

| 风暴 | 时间 UTC | 气压 (hPa) | 定义 | 来源 |
| --- | --- | ---: | --- | --- |
| LAN 2017 | 2017-10-21T07:00Z | 925 | 投落送反演最低海平面气压 | [琉球大学等联合新闻稿](https://www.u-ryukyu.ac.jp/news/509/) |
| LAN 2017 | 2017-10-22T01:00Z | ≈930 | 同上，原稿写“およそ 930” | 同上 |
| LAN 2017 | 2017-10-21（会议窗口） | 926 | 后期会议分析版本 | [JpGU 2020 U18-05](https://confit.atlas.jp/guide/event-img/jpgu2020/U18-05/public/pdf?type=in) |
| LAN 2017 | 2017-10-22（会议窗口） | 928 | 后期会议分析版本 | 同上 |
| TRAMI 2018 | 2018-09-25–28 任务期最低 | 920 | 投落送反演最低海平面气压 | [ECMWF 会议贡献](https://events.ecmwf.int/event/118/contributions/495/contribution.pdf) |
| NEOGURI 2025 | 2025-09-21 | 930 | 投落送测得海面中心气压 | [名古屋大学 ISEE 摘要](http://www.rain.hyarc.nagoya-u.ac.jp/~tsuboki/isee_sympo2025/src/abstracts/Tsujino.html) |
| NEOGURI 2025 | 2025-09-22 | 935 | 同上，减弱阶段 | 同上 |
| HALONG 2025 | 2025-10-07T06:20Z | 959 | JMA TEMP_DROP 投落送最低海面气压 | [PTT/disp 论坛镜像](https://disp.cc/ptt/TY_Research/1evC4awx)；[twtybbs](https://twtybbs.com/thread-24015-4-2.html) |

版本差异不强行合并：LAN 的 925/926 和 930/928 同时保留，并标注互为 `variant_of`；
TRAMI 只有任务期最低值，没有伪造时刻；NEOGURI 两日值分别保留。
**没有用 JMA BST 的 935/915 等中心气压去“校正”这些飞机值**。

### 4.3 有任务但未取得公开数值（3 条，不填补）

| 风暴 | 任务事实 | 实际处理 |
| --- | --- | --- |
| HAISHEN 2020 | T-PARCII 投落送任务有公开论文/图 | 可访问文本未给出数值；`pressure_hpa = null`，不用 BST 补 |
| MINDULLE 2021 | T-PARCII 暖核与 inner-core 任务 | 摘要讨论观测，未给数值；保留 `null` |
| BARIJAT 2024 | 投落送数据经 GTS 分发 | 摘要未给数值；保留 `null` |

2016、2019、2023、2026 年未发现 JMA/MRI 飞机实测；这不是“没有台风”，而是没有该来源的公开观测。
Zenodo 17050931 只有 LAN/TRAMI 近海面湿度，**没有气压列**，未计入本批气压。
仓库中既有的 68 条 NHC 中太平洋飞机中心气压属于另一个机构和区域，未混入本 JMA/MRI 计数。
2025 年 WIS `TEMP_DROP` 目录当前为空（保留期已过），HALONG 的原始报文无法重新下载；
因此 959 hPa 只以论坛转引的官方 GTS 截图形式保留，明确标为 forum provenance。

## 5. 非官方论坛核查

实际完成的论坛核查是**定向的**，不是 283 个风暴 × 所有论坛的穷举：

- 找到 HALONG 2025 的 PTT/TY_Research 帖及其 disp.cc、twtybbs 镜像，
  帖中给出 JMA `TEMP_DROP` 投落送 963 → 960 → 959 hPa 的截图与 06:20Z 时刻；
  官方 WIS 目录已无 2025 文件。该条以非官方论坛来源入 JSON，并保留官方来源名。
- 对 BARIJAT 2024、MINDULLE 2021 的论坛检索未找到可回查的机载气压数值；
  没有把论坛转述的 BST 中心气压当实测值。
- 未完成：Storm2k、Wunderground、台风论坛等站点的逐风暴全量爬取/人工逐帖核对。
  因此本报告不能声称“所有非官方论坛已穷举”。

## 6. 产物与字段

产物目录 `/mnt/data/H9dP/datasets/typhoon_census_2016_2026_20261115/`：

| 文件 | 内容 |
| --- | --- |
| `census.json` | 283 个风暴的完整结构，含浮标候选序列、飞机实测、状态与来源 |
| `storms.csv` | 每个风暴的编号、名称、起止、浮标对数、最低浮标气压、飞机状态 |
| `buoy_minima.csv` | 580 条平台-风暴候选最低值，含基线、压差、置信度、并发风暴 |
| `aircraft_measurements.csv` | 16 条飞机记录（8 原生剖面 + 8 已发表值） |
| `manifest.json` | 190 个源文件 URL/大小/SHA-256、命令、失败列表、汇总 |
| `collection.log` | 实际采集日志（含每个 TXT 的记录数；保留首次启动失败后的重试记录，最终 `manifest.json` 的 failures 为 0） |
| `raw/` | 169 个 JMA 浮标 TXT、11 个 Digital Typhoon 年表、8 个投落送 CSV、Zenodo API JSON |

`census.json` 的顶层：

```text
schema_version, generated_utc, scope, constraints, sources, summary, collection,
coverage_notes, storms[]
```

`coverage_notes` 明确写出论坛只做定向核查、NDBC 传输失败未纳入、3 个飞机任务无数值、
2026 季节未结束、浮标关联无空间核验；`summary.forum_derived_aircraft_records = 1`。
每个风暴包含：

```text
storm_id, name, season, active_window_utc, storm_identity_source, status,
buoy_pressure_minima[], jma_aircraft_measurements[],
jma_aircraft_missions_without_numeric_pressure[]
```

每个浮标候选包含最低气压、时刻、经纬度、基线、压差、`association_confidence`、
`attribution_ambiguity`、并发风暴、来源 URL 与完整活动窗口（边缘 ±6 h）的实测小时序列。
每个飞机记录包含原始/发表类型、气压定义、来源 URL、`pressure_is_best_track=false`；
任务无数值的记录放在 mission 数组，`pressure_hpa` 不存在。

## 7. 复现命令

```bash
cd /home/debian/H9dP
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
PYTHONPATH=src /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -u \
  -m scripts.collect_typhoon_census \
  --output /mnt/data/H9dP/datasets/typhoon_census_2016_2026_20261115
```

采集器内部再次使用空 `ProxyHandler`；网络请求前清除所有名称含 `PROXY` 的环境变量。
2026-11-15 采集前根盘已用 **343,284,113,408 bytes**，低于 375 GB 上限；
产物 25 MB，位于 `/mnt/data`。

验证：

```bash
PYTHONPATH=src /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -m pytest \
  tests/test_typhoon_census.py -q
# 6 passed
```

`ruff` 不在当前 Conda 环境中，未执行 Ruff；已用 pyright/LSP 检查改动文件，
并运行专项 pytest。完整套件结果见项目运行记录。

## 8. 结论与未完成清单

已完成：

- 2016–2026 年 283 个编号 WNP 风暴的完整枚举与逐风暴 JSON 条目。
- JMA 官方漂流浮标 169 个年度文件、186,834 条有效小时气压；580 个候选关联、52,730 条序列记录。
- 8 条 T-PARCII 原生眼区投落送气压剖面；8 条已发表/公开飞机反演值；3 条任务无数值记录。
- 全流程不写入 BST 中心气压、风速或轨迹位置；不合成、不插值、不按配额补齐。

未完成，明确不冒充完成：

1. 283 个风暴 × 所有非官方论坛的逐帖穷举；目前只有定向核查和 HALONG 一条论坛来源。
2. NDBC TAO/KMA/太平洋岛屿浮标的完整补充；传输失败，未纳入。
3. HAISHEN 2020、MINDULLE 2021、BARIJAT 2024 的公开数值；只有任务事实。
4. 浮标关联的空间核验（到风暴中心的距离/方位）；当前严格不用轨迹，因此只按时间窗+压差，
   并用置信度和并发标记暴露不确定性。
5. 2026 年季节尚未结束；截至 2026-11-15 的编号风暴已覆盖，后续风暴需重跑。

返回[资料索引](README.md)或[永久实测准则](OBSERVATION_POLICY.md)。

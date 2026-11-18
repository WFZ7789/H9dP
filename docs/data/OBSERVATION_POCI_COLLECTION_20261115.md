# 实测气压与 POCI：首批失败采集记录（2026-11-15，已拒收）

## 状态与边界

**本批已拒收，未完成任务。50 条中代理 dP 的最大值仅 28.1 hPa，满足 dP >30 hPa 的有用记录为 0 条。**
整气旋排除、气旋配额/轮换以及同气旋强制 6 h 间隔均为擅自增加的错误限制，已撤销。
永久规则见[实测集准则](OBSERVATION_POLICY.md)及根目录 [AGENTS.md](../../AGENTS.md)。
已有训练/验证/测试气旋及与训练记录相隔 ≤6 h 的观测均可纳入，不因这些 tag 而降级或判为无效。

以下下载统计和旧工具审计只保留为失败过程记录，不表示实测集验收通过。
原始字节和历史源码快照保留，目录另加 `REJECTED.json`，不改写不可变历史快照。
**原始数据采集曾完成，但有用的中心气压实测集尚未完成。**
本次没有训练或评估模型，没有中心气压误差成绩，也没有 eager/compiled 训练对比。

用户认可两种 POCI 来源：IBTrACS `USA_POCI` 和原 MOTIS 使用的 ERA5 POCI。
两者均保留来源，不因来源不同而判为错误，也不把它们无标记地合并成同一个字段。
本次新增站点配对采用 `USA_POCI`；另行整理原 MOTIS 观测约束标签及其 ERA5 POCI，
其中能匹配到 IBTrACS 的记录同时保留两种值。

**站点实测海平面气压不自动等于气旋中心气压。** 历史 50 条不计作交付的有用实测记录，
也不能据此宣称最终成果的中心气压精度；这与气旋是否出现在训练集中无关。

## 已执行结果

| 项目 | 实际结果 |
| --- | ---: |
| 计划查询年份 | 2016–2025 |
| 排除已有气旋后的 IBTrACS 轨迹 | 203 个 SID |
| 按年份＋规范化名称排除的已有气旋 | 68 个 SID |
| 可查询站点年 | 285 |
| 本批查询上限 / 实际尝试 | 100 / 100 站点年 |
| 成功下载的站点年 | 96 |
| 失败 | 4，均为 HTTP 404，保留在 `failures.json` |
| 通过时空和气压 QC 的原始配对 | 105 条，尚含同小时/相邻时次 |
| 同气旋同小时去重、至少间隔 6 h 后 | **50 条，37 个 SID** |
| 50 条中距轨迹中心 ≤10 km | 19 条 |
| 报告类型 / SLP 质量码 | 全部 FM-12 / QC=1 |
| 站点实测 SLP 范围 / 中位数 | 974.9–1009.7 / 998.15 hPa |
| 配对 POCI 范围 / 中位数 | 997–1013 / 1005 hPa |
| 距轨迹中心范围 / 中位数 | 2.398–24.613 / 11.373 km |
| POCI 时差实际范围 | −60 至 +60 min；中位数 0 |
| 原始 CSV 来源哈希核验 | **98 / 98 通过**：96 个站点年＋2 个目录/轨迹文件 |
| 中心气压真值已认证数 | **0** |
| 模型评估次数 | **0** |

50 条候选按年为：2016 年 3、2017 年 6、2018 年 8、2019 年 9、2020 年 3、
2021 年 11、2022 年 6、2024 年 4。查询范围不代表每个年份都有入选记录；本批没有
2023/2025 年入选记录。设定目标上限为 100 条，但不放宽配对或时间间隔标准来凑数。

另行整理原 MOTIS 的 **16 条** 2015–2025 年 WP、`dP Obs` 非空记录：

- 15 条属于现有 `MOTIS_cyclone_records.json` 中已出现的气旋。这只是 tag，不影响纳入实测集或其结果有效性；此前把气旋重合作为否决理由是错误的，已撤销。
- 13 条匹配到满足同年同名、≤3 h、≤100 km 条件的 IBTrACS POCI，同时保留两种 POCI。
- 其余 3 条的 IBTrACS 配对字段留空，不填默认值、不编造配对。
- `dP Obs` 是论文采用的观测约束标签，可能包含飞机/地面观测、雷达估计及时间插值；
  不能把这 16 条统一称为原始传感器读数，亦不与新增 50 条站点候选混成一组真值。

## 来源与字段

### 1. 新增实测气压

NOAA NCEI Integrated Surface Database 的 Global Hourly 原始 CSV：

```text
https://noaa-global-hourly-pds.s3.amazonaws.com/<year>/<USAF><WBAN>.csv
```

站点目录：<https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv>。
使用 `SLP`，而不是把 `MA1` 的站压或高度表设置冒充海平面气压。
SLP 数值按原格式从 0.1 hPa 换算为 hPa，只接受质量码 `1`/`5`，拒绝 `99999`、
非数值及 800–1100 hPa 之外的值。SLP 可能包含海平面订正，并非全部为未订正站压。

每条记录保留站号、站名、原始观测时间、原始经纬度/高度、质量码、报告类型、
原始 URL 和 CSV 行号，可以回查原始行。下载时保存原始字节、SHA-256、获取时间和最终 URL。

### 2. IBTrACS POCI

使用 IBTrACS v04r01 WP `main` 轨迹的 `USA_POCI`：

<https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv>

POCI 保留 IBTrACS 提供的分析值，不通过站点气压反推，不设为固定的 1010 hPa。
采集器不对 POCI 再做插值；但 IBTrACS 上游本身可能已经对 3 h 记录插值/沿用分析值，
所以不能据此声称每个值都来自原始 ATCF 报文中的独立分析。

### 3. 原 MOTIS ERA5 POCI

原表：<https://doi.org/10.5281/zenodo.22224266>，版本 `260212`。
已存在的不可变来源保留在：

```text
/mnt/data/H9dP/reference/motis/260212/MOTIS_dataset_260212.csv
```

论文方法是在最近整点、0.25° ERA5 海平面气压场上，以 **0.1 hPa** 间隔计算等压线，
取得最外围闭合等压线压力。**本次读取已发布表格中的值，没有重新下载 ERA5 或重算等压线。**

MOTIS 库存表保留 `poci_motis_era5_hpa`、`dp_motis_adopted_hpa`、
`pc_motis_adopted_hpa = poci_motis_era5_hpa - dp_motis_adopted_hpa`。
可匹配时另列 `poci_ibtracs_hpa`、相应时间及 SID。若以后改用 IBTrACS POCI 构造目标，
应从同一个绝对中心气压重新计算 dP，不能把原 ERA5 定义的 `dP Obs` 无改动配给新的 POCI。

## 当时执行的失败协议（非现行准则）

1. 使用 WP 主轨迹、`NATURE=TS`，按计划年份读取。
2. 现有 562 条记录按「年份＋去除非字母数字字符后的名称」归组。
   排除相应 IBTrACS 气旋；这是对当前索引的排除，不是对所有历史训练产物的完全泄漏审计。
3. 站点目录范围：5–45°N、110–160°E、海拔 −10–100 m，且目录覆盖该年份。
   原始观测坐标须距目录坐标 ≤5 km，原始观测高度也须在上述范围内。
4. 对轨迹位置在不超过 6 h 的间隙内插值，禁止时间外推。位置采用 IBTrACS `LAT/LON`；
   它与 `USA_POCI` 不一定来自同一机构的中心定位，须计入后续定位不确定性。
5. 站点观测距插值中心 ≤25 km，POCI 取最近轨迹时刻的原值、时差 ≤3 h。
   每条保存 POCI 自身的时间与时差，不将轨迹时间冒充实测时间。
6. 以距轨迹中心的最小距离排序选取前 100 个站点年，8 并发下载；
   不按气压低值/强度巅峰排序。站点年预筛使用逐小时轨迹，可能漏掉短促的近站经过，
   因此本次不是穷尽式全球观测搜集。
7. 同一 SID 同小时重复报告/多个站点只留距中心最近的一条；
   使用固定种子 **20261114**，按气旋轮换选取，同一气旋入选时次至少间隔 6 h。
   采样不读取气压值用于排序。保留完整的 105 条原始配对，便于审查抽样过程。
8. `dp_station_proxy_hpa = poci_hpa - slp_observed_hpa` 只标作站点代理压差，
   `is_central_pressure_truth=False`，不作为已经认证的中心压差标签。

本批偏向被地面站覆盖的近岸/登陆事件，不代表海上清晰台风眼样本的分布。
站点距离依赖最佳路径和插值定位；≤10 km 也不能自动证明测站位于台风眼或测得最低中心气压。

## 命令与落盘

以下是当时在仓库根目录执行的历史命令；解释器为 Conda `fy3-aivis-v1` 的 Python **3.13.15**。
live 脚本已改为逐观测、tag-only 的 v2，不能用它复现旧筛选逻辑；现行审计器会拒绝旧 v1 manifest。
旧逻辑只在原目录的源码快照中保留。**不得重跑命令覆盖这次失败运行的历史产物。**

```bash
# 网络操作前清除全部大小写 PROXY 变量；脚本内部再次清除，并禁用 urllib 代理。
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
PYTHONPATH=src MKL_THREADING_LAYER=GNU \
  /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -u \
  -m scripts.collect_surface_poci \
  --output /mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115 \
  --count 100 --max-station-years 100 --workers 8

# 纯离线审计，并单独整理 MOTIS adopted-pressure 库存。
PYTHONPATH=src /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python \
  -m scripts.audit_surface_poci \
  --collection /mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115 \
  --motis-csv /mnt/data/H9dP/reference/motis/260212/MOTIS_dataset_260212.csv
```

采集完成时间：**2026-11-15 03:29:31 UTC**。首次离线工具审计完成时间：03:36:31 UTC。
主要产物全部在仓库外：

```text
/mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115/
  raw/                              原始 CSV 及每个来源的 .json 元数据
  all_surface_poci_pairs.csv         105 条原始有效配对
  review_candidates.csv             50 条待审核候选，不是冻结最终测试集
  motis_adopted_wp_2015_2025.csv      16 条原论文采用标签，单独保存
  manifest.json                     查询参数、种子、统计与使用限制
  audit.json                        哈希、去重、气旋排除、时差/距离/间隔审计
  failures.json                     4 个 404
  excluded_exposed_cyclones.json     排除的 68 个 SID
  collection.log                    实际采集日志
  source_snapshot/                  采集开始时源码/合成测试快照，保留不改写
```

采集阶段目录约 **358 MB**，预估上限 2.2 GB，均落 `/mnt/data`。
采集前根盘已用 **343242874880 bytes**，采集后检查为 **343243030528 bytes**，
低于 375000000000 bytes 上限。本批未修改训练索引、数据划分、检查点或正在运行的训练。

## 验证与后续

实现和可复现工具：
[配对与抽样](../../src/h9dp/observations.py)、
[采集脚本](../../scripts/collect_surface_poci.py)、
[离线审计](../../scripts/audit_surface_poci.py)、
[合成测试](../../tests/test_observations.py)。

```bash
/home/debian/miniconda3/envs/ai_village/bin/ruff check \
  src/h9dp/observations.py scripts/collect_surface_poci.py \
  scripts/audit_surface_poci.py tests/test_observations.py
MKL_THREADING_LAYER=GNU /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python \
  -m pytest -q
```

最终结果：**Ruff 全部通过；完整 pytest 159 passed（3.52 s）**，其中本次新增合成测试 15 项。
Ruff 使用已有 `ai_village` 环境中的可执行文件，没有为此安装依赖；测试使用要求的
`fy3-aivis-v1`。新增测试使用小型合成 CSV，不联网、不依赖外部归档、不训练模型。
主动 LSP 检查未报告类型错误，但服务器静默返回未确认状态，不将其表述为已确认全覆盖无错。
完整测试与最终检查记录见同目录外部产物中的 `tests.log`、`ruff.log`。

**尚未完成：** 补齐可靠的 dP >30 hPa 中心气压观测及 POCI、查证测站代表性、
准备对应卫星序列，并按永久准则形成有用的实测集。气旋名、训练/验证/测试标签及距训练记录
≤6 h 不构成任何排除、降权或结果无效的理由。真实泛化能力留给未来独立观测和长期业务运行检验。

返回[数据准备索引](README.md)或[总文档索引](../README.md)。

# JMA 现场气压观测与环境 POCI：首批 100 条

> 本文是首批历史结果。[后续扩大采集与航空来源核查](JMA_EXPANSION_AND_AIRCRAFT_20261115.md)已取得 1,769 条强压差站点配对，并补查飞机/投落送来源；100 条不是采集上限。

状态：**JMA 现场气压观测采集、POCI 配对和 100 条强压差记录抽取已完成。**
不是 JMA BST，不使用任何最佳路径中心气压作为实测压力，不用分析值填补缺测。
本报告只记录实际取得的数据；没有训练模型、没有计算模型精度，也未准备完整卫星序列。

## 实际结果

| 项目 | 实际结果 |
| --- | ---: |
| JMA 气象官署 | 4 个：石垣岛、久米岛、那霸、鹿儿岛 |
| 官方十分钟观测日表 | 8 个站点日 |
| 正常质量的原始实测气压记录 | **1152 条** |
| 成功配对已认可环境 POCI | 1146 条 |
| dP >30 hPa 的可选观测记录 | **193 条** |
| 本批选取 | **100 条** |
| 100 条的海平面气压范围 | **940.8–975.9 hPa** |
| 100 条的站点压差范围 | **30.1–62.2 hPa** |
| 质量类 | 全部 `data_0_0`，压力字段无缺测/资料不足/非正常标记 |
| BST 中心气压标签 | **0 条** |
| 被插值、外推或用分析值填补的实测压力 | **0 条** |
| 被冒称为已确认气旋中心气压的站点观测 | **0 条** |
| 原始页面 SHA-256 检查 | **8/8 通过** |
| 100 条逐条回查原始表格 | **100/100 一致** |
| 集合内重复原始观测 | **0 条** |

观测选取以记录为单位，不施加气旋配额、气旋均衡或同气旋时间间隔。
85 条属于当前索引中已出现的气旋，仍正常纳入，仅保留 tag。
这些记录不是 100 个气旋或 100 个独立天气过程；这里按用户规定统计真实观测记录，不以气旋/年份作排除单位。

## 原始来源：JMA「過去の気象データ検索」

使用气象官署 `10min_s1.php` 的十分钟观测表，直接读取「現地」「海面」气压列。
没有读取 JMA BST 中心气压，没有用 IBTrACS `USA_PRES`、`WMO_PRES` 或 `TOKYO_PRES` 当实测。

| 测站 | 编号 | 观测日（JST） | 气旋 tag | 入选数 |
| --- | --- | --- | --- | ---: |
| 石垣岛（石垣島） | 47918 | 2015-08-23、24 | Goni | 15 |
| 久米岛（久米島） | 47929 | 2016-10-03、04 | Chaba | 12 |
| 那霸（那覇） | 47936 | 2018-09-29、30 | Trami | 40 |
| 鹿儿岛（鹿児島） | 47827 | 2022-09-18、19 | Nanmadol | 33 |

四个可直接回查的官方日表及其当日最低海平面气压：

- [石垣岛，2015-08-23](https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=91&block_no=47918&year=2015&month=8&day=23&view=)：**948.1 hPa，21:30 JST**。
- [久米岛，2016-10-04](https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=91&block_no=47929&year=2016&month=10&day=4&view=)：**957.3 hPa，00:40 JST**。
- [那霸，2018-09-29](https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=91&block_no=47936&year=2018&month=9&day=29&view=)：**954.4 hPa，14:40 JST**。
- [鹿儿岛，2022-09-18](https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=88&block_no=47827&year=2022&month=9&day=18&view=)：**940.8 hPa，19:00 JST**。

以上是原日表最低值，不把四个日极小值冒充整批随机抽取记录的分布。
原页面、获取时间、原 URL 与 SHA-256 均保留。

## 一条实际入选记录

鹿儿岛，2022-09-18 **19:00 JST = 10:00 UTC**：

- JMA 原表「現地」气压：**937.5 hPa**。
- JMA 原表「海面」气压：**940.8 hPa**。
- 环境 POCI：**1003.0 hPa**，对应 10:00 UTC，时差 0。
- 站点观测压差：**1003.0 − 940.8 = 62.2 hPa**。
- 两个气压值直接来自原表，不是 BST 气旋中心气压。

## 字段、质量与时间规则

- 保留原始现地气压、JMA 发布的海平面气压及质量类。
  海平面气压可能包含官方海平面订正；它不是未经订正的现场站压，两列不互相替代。
- 只接受气压单元格完整数值及 `data_0_0`；缺测、`×`、`///`、带 `)`/`]` 的值不填补、不当正常值。
- 同时保留平均风、风向、阵风和阵风风向，便于后续核对眼区经过和压力代表性。
- JMA 时间是 **JST（UTC+09:00）**，不是 UTC。保留页面日期及 `24:00` 等原始时刻，
  按正确的次日午夜规则转换到 UTC。
- 每条保留测站编号、站名、原始 URL、表格行号、原始时刻与转换后的时刻。
- 海平面气压来自真实测站观测，但**不自动等于气旋最低中心气压**。
  已明确标记 `is_verified_cyclone_central_pressure=False`，没有偷换成 `Pc`。
  核对中心/眼区代表性是后续工作，不以气旋是否在训练集中或接近训练时次作为判据。

## 环境 POCI 与实测压力严格分开

环境字段沿用用户此前已认可的 **IBTrACS `USA_POCI`**，仅作为 POCI，绝不生成、替代或修正观测压力。
数据源为此前保存的 WP CSV：

```text
/mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115/raw/ibtracs.WP.list.v04r01.csv
```

按事件名称/年份 tag 关联 POCI 时间序列，取最近且时差 ≤3 h 的原值。
这里的时间匹配是环境量与观测的同步，不是与训练集的时间隔离。
所有原始 JMA 记录仍保留，未配到 POCI 的 6 条没有被捏造 POCI。

派生量明确命名为：

```text
dp_observed_station_hpa = poci_environment_hpa - slp_observed_hpa
```

不叫「BST 实测」，不把该站点量暗中改名为已证实中心压差。

## 抽样和实际命令

1152 条实测先按真实来源入库，再进行 POCI 配对，筛选 dP >30 hPa 的 193 条，
固定种子 **20261115** 在观测级随机取 100 条。没有按最低气压排序，没有按气旋限额，
没有整气旋排除，没有与训练记录的 6 h 缓冲。

网络请求前清除全部名称含 PROXY 的环境变量；下载器再次禁用代理。两并发获取小型原始日表。
命令在仓库根目录执行：

```bash
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
PYTHONPATH=src /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -u \
  -m scripts.collect_jma_observations \
  --output /mnt/data/H9dP/datasets/jma_observations_2015_2025_20261115 \
  --poci-csv /mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115/raw/ibtracs.WP.list.v04r01.csv \
  --count 100
```

实际完成时间：**2026-11-15 05:18:41 UTC**。采集前根盘已用 **343250489344 bytes**，
低于 375 GB；数据和机器产物均在 `/mnt/data`，没有修改训练输入索引或历史运行快照。

## 产物位置

```text
/mnt/data/H9dP/datasets/jma_observations_2015_2025_20261115/
  jma_observations_dp_gt30_100.csv       100 条原始 JMA 压力观测＋环境 POCI
  all_jma_dp_gt30.csv                   193 条全部强压差观测配对
  all_jma_pressure_measurements.csv     1152 条正常质量的原始压力记录
  all_jma_poci_pairs.csv                1146 条环境量配对，含较弱压差
  raw/                                 8 个官方原始日表及 URL/时间/SHA-256 元数据
  manifest.json                        数量、范围、种子、来源和限制
  audit.json                           100 条逐一回查原表与 8 个来源哈希的核验结果
  collection.log                       实际运行日志
  source_snapshot/                     采集时的实现与合成测试快照
```

实现：[JMA 表格解析](../../src/h9dp/jma_observations.py)、
[采集器](../../scripts/collect_jma_observations.py)、
[合成测试](../../tests/test_jma_observations.py)。

实际检查：变更的观测相关 Python 路径 Ruff 全部通过；
`MKL_THREADING_LAYER=GNU /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -m pytest -q`
得到 **180 passed（5.65 s）**，其中 JMA 解析合成测试 11 项。日志为本产物目录下的
`ruff.log`、`tests.log`。这些是实现检查，不是实测数据数量或模型精度。

尚未执行：逐条确认站点与气旋中心/眼区的物理代表性、核对当时测站位置和仪器高度、准备
相应四波段六帧卫星数据，以及实际模型验证。这不影响本批作为真实 JMA 观测资料的来源事实，
也不应把未做的中心代表性核验冒称已经通过。

返回[资料索引](README.md)或[永久准则](OBSERVATION_POLICY.md)。

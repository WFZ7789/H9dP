# JMA 扩大采集与航空观测来源核查（2026-11-15）

状态分开记录：**地面/岛屿测站采集与原表审计已完成；飞机/投落送来源已确认，但完整原生气压廓线尚未取得。**
没有训练或验证模型，没有准备完整卫星输入；不把站点海平面气压冒充已确认的气旋中心气压，也不使用 BST 压力标签。

## 1. 地面实测的实际交付

首批只有 4 站、8 个站日、193 条强压差记录，100 条仅为其中的抽样；不能用 1152 行十分钟记录宣称覆盖充分。
本次按更多实际台风过境的站点/日期检索，**所有 dP>30 hPa 配对全量保留**，没有每气旋限额、气旋均衡或时间隔离。

| 项目 | 本次实际结果 |
| --- | ---: |
| 正常质量实测气压观测 | **87,088 条** |
| 已取得原始十分钟日表 | **606 个站日、37 站** |
| 检索事件的气旋/年份标签 | 43 个；仅来源范围描述，不是配额 |
| 实测气压与环境 POCI 成功配对 | **76,815 条** |
| dP>30 hPa 全量主交付 | **1,769 条** |
| 强压差记录的实际来源覆盖 | **29 站、64 个站日、25 个气旋/年份标签** |
| 强压差来源年份 | 2015–2024；检索也包含 2025，但未产生合格配对 |
| 强压差海平面气压范围 | **932.3–979.9 hPa** |
| 强压差范围 | **30.1–70.7 hPa** |
| 30<dP≤50 / 50<dP≤70 / dP>70 | **1599 / 169 / 1 条** |
| 首批 193 条是否全部保留 | 是；新增 **1,576 条** |
| 带已有气旋索引 tag 的强压差记录 | **1,656 条**，没有因此排除 |
| 带已有同名同时次 tag 的强压差记录 | **10 条**，没有因此排除 |
| 重复物理观测 / BST 压力标签 | **0 / 0** |

这些是 1,769 条不同站点/时刻的实测约束，不是 1,769 个独立天气过程或已经验证的中心气压。
JMA 原表的「現地」压力和官方「海面」压力分别保留；后者可能包含官方海平面订正。
本次派生量为 `dp_observed_station_hpa = poci_environment_hpa - slp_observed_hpa`。
**不自行插补观测压力；未配到 POCI 的 10,273 条正常气压记录仍在原始观测总表中。**

环境量沿用用户认可的 IBTrACS v04r01 `USA_POCI`，仅作环境压力，不使用 `USA_PRES`、`WMO_PRES`、`TOKYO_PRES` 作测压标签。
取最近且时差≤3 h 的原始 POCI；这不是训练时次隔离。POCI 源 SHA-256：
`83bf4b39a266b327a80ad2e6a701afb730f3ed7e47cf07891ce207551950e569`。

### 主交付及审计

```text
/mnt/data/H9dP/datasets/jma_observations_expanded_2015_2025_20261115/
  all_jma_dp_gt30.csv                  全量 1769 条，主交付，不是 100 条上限
  all_jma_pressure_measurements.csv    87088 条正常质量实测观测
  all_jma_poci_pairs.csv               76815 条配对，包含较弱压差
  jma_observations_dp_gt30_100.csv      额外的 100 条随机示例，非全集
  raw/                                606 份原始 HTML、来源和 SHA-256
  manifest.json                       最终数量、来源、范围和采样标记
  audit.json                          实际审计结果
  audit_bank.py                       对真实产物执行的回查脚本
  collection.log                      初次运行日志，含两次网络超时
  attempt_1_manifest.json              初次 604 站日结果，保留历史
  attempt_1_source_failures.json        初次失败的两个来源
  collection_retry.log                缓存复用并重试后的最终日志
  source_failures.json                最终为空；606 站日均取得
  source_snapshot/                    本次采集代码及查询配置快照
```

最初鹿儿岛、名濑的 2020-09-02 日表读取超时；保留初次结果后重试，最终两者均取得。
初次完成时间为 06:35:55 UTC，最终完成时间 **2026-11-15 06:41:55 UTC**。
原始的首批数据目录与历史源码快照没有改写。

实际审计通过：

- **606/606** 原始日表 SHA-256 一致，站名与来源页面标题一致。
- **87,088/87,088** 行回查原单元格：现地压力、海面压力、质量、来源 URL、原始时刻和 JST→UTC 转换一致，包括 `24:00`。
- **76,815/76,815** 环境配对回查源 `USA_POCI`、时间偏差和减法一致。
- **1,769/1,769** 与全部配对中 dP>30 hPa 的集合完全一致，未以 100 条、气旋或年份截断。
- 首批 193 条全部包含于新全集；物理键为 `(station_id, observation_time_utc)`，不同站同一小时不合并。
- 全部仍标记 `is_verified_cyclone_central_pressure=False`。审计不是眼区/中心代表性验证，也不是卫星输入验证。

## 2. 飞机/投落送：确有来源，不能漏查

**JMA 实测检索不能限缩为地面站日表。** 需查其发布、接收、合作取得的航空观测；观测执行机构与资料使用/发布机构分别注明，不假定飞机归 JMA 所有。

已实际下载并核对：

1. [气象研究所历年成果](https://www.mri-jma.go.jp/OldTopics.html)中列出的联合成果
   「航空機観測によって台風の強度と構造を正確に測定することに成功」，链接到
   [琉球大学保存的联合发布页面](https://www.u-ryukyu.ac.jp/news/509/)及其
   [原始发布 PDF](https://www.u-ryukyu.ac.jp/wp-content/uploads/2018/12/press2018073001.pdf)。
   这是 **2017 年 Lan 的飞机/投落送合作观测**，参与者包含气象厅气象研究所。
   页面报告 2017-10-21 16 时的中心压力 **925 hPa**、22 日 10 时**约 930 hPa**，并明确观测数据实时送气象厅用于预报。
   这些是公开观测成果，不是 JMA BST；但新闻稿不是完整原生投落送廓线。
2. [后续 EGU2020 摘要](https://doi.org/10.5194/egusphere-egu2020-3154)对相同两次飞行报告 **926 / 928 hPa**。
   已保存 Crossref 元数据和摘要。**存在版本/处理差异，尚未解决，不能把不同版本算成四次观测或偷偷选一个当精确真值。**
3. [公开 T-PARCII 数据集](https://doi.org/10.5281/zenodo.17050931)已下载两个 CSV。
   实际字段是 `sounding id, sec, yyyy, mm, dd, h:m:s, lat, lon, gpm, QA10, SST`，共 26 / 57 行。
   **没有气压列；83 行是近海面湿度等辅助资料，不是 83 条气压测量。**
4. [JMA WIS 开放资料入口](https://www.wis-jma.go.jp/d/o/)与浏览入口已访问并保存。
   当前只确认门户可达，尚未定位到本项目所需历史飞行的完整原生气压记录；不能据此断言资料不存在。

航空来源产物：

```text
/mnt/data/H9dP/datasets/jma_aircraft_sources_20261115/
  lan_2017_joint_release.html/.pdf          联合发布原文，附 URL/哈希
  mri_old_topics.html                      MRI-JMA 官方来源索引
  egu2020_3154_crossref.json                后续摘要与版本差异
  zenodo_17050931.json                     辅助数据原始元数据
  sfc_Lan_2017_hc_qa.csv                    无气压列，不计气压样本
  sfc_Trami_2018_hc_qa.csv                  无气压列，不计气压样本
  published_aircraft_pressure_examples.csv 两次已发表结果的暂存例子，不是已核验原生记录
  build_published_examples.py              暂存例子与独立环境 POCI 的生成记录
  source_inventory.json / manifest.json    取得状态、来源、哈希和限制
```

两条暂存例子分别对应 POCI 1004 / 1007 hPa，dP 为 79 / 77 hPa，**不并入 1,769 条站点全集，也不计为已完成的原生航空测压样本**。
发布材料以小时述时，保存原文时刻；例子按日文发布的 JST 语境转 UTC，但须由原生飞行记录进一步确认时区和精确时刻，不能把补零分钟解释成传感器时间精度。
其 `native_pressure_profile_acquired=False`、`satellite_ready=False`，原生气压廓线实际取得数仍为 **0**。
飞行高度的气压也不能拿来冒充海平面或气旋中心压力。

这条来源核查说明此前漏查了航空支线，**不是完成了一套航空实测测试集**。后续应优先追查完整气压记录、原始时刻/位置/高度、质量与后处理说明，解决上述压力版本差异。

## 3. 实际协议与复现命令

使用既有 Conda 环境 Python 3.13.15，没有安装新依赖。网络请求前清除所有大小写名称含 PROXY 的变量；采集器再次清除代理并直接请求。
采集前根盘已用 **343252295680 bytes**，小于 375 GB；预计外部空间 201500000 bytes，原始数据和运行产物均写 `/mnt/data/H9dP/`。
四个下载工作线程，起始请求间隔至少 0.5 s；八个旧日表使用保留原来源元数据的缓存副本。

查询配置为 [606 站日的显式来源查询](../../configs/jma_observation_cases_2015_2025.json)，不是已执行结果本身。
[采集器](../../scripts/collect_jma_observations.py)默认仍可复现首批八站日；扩大采集必须传入该配置。
`--count 100 --seed 20261115` **只控制附加随机示例**，不限制 `all_jma_dp_gt30.csv`。

```bash
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
PYTHONPATH=src /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -u \
  -m scripts.collect_jma_observations \
  --output /mnt/data/H9dP/datasets/jma_observations_expanded_2015_2025_20261115 \
  --poci-csv /mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115/raw/ibtracs.WP.list.v04r01.csv \
  --cases-json configs/jma_observation_cases_2015_2025.json \
  --workers 4 --min-request-interval 0.5 --count 100 --seed 20261115
# 相同命令使用缓存重试失败来源；初次日志和清单已另外保留。
PYTHONPATH=src:. /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python \
  /mnt/data/H9dP/datasets/jma_observations_expanded_2015_2025_20261115/audit_bank.py
PYTHONPATH=src:. /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python \
  /mnt/data/H9dP/datasets/jma_aircraft_sources_20261115/build_published_examples.py
```

实现检查与真实数据结果分别统计，实际执行结果：

- Ruff 检查采集器、新增合成测试和两份外部产物脚本，**全部通过**。
- `MKL_THREADING_LAYER=GNU /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -m pytest -q`：**186 passed，5.05 s**。
- 完整原表/环境量审计和航空暂存例子生成脚本均已执行通过；不是模型验证。
- `git diff --check` 通过。日志为扩大采集目录下的 `ruff.log`、`tests.log`、`audit.log`，航空目录下的 `build_examples.log`。
- 扩大采集目录实际占用约 **115 MB**；检查时根盘已用约 **343.3 GB**，低于 375 GB。

LSP 主动探测未返回诊断，但服务器静默/超时，**覆盖未确认**；不能据此声称全部类型检查干净。

## 4. 未完成事项

航空原生气压记录的完整获取、压力版本差异核验，历史测站位置/仪器高度核查，站点相对眼区/中心代表性判定，AHI 四波段六帧输入准备，实际模型验证均未完成。
没有训练子集拟合结果、编译执行结果或泛化误差可报告；本报告只陈述已经完成的资料采集、来源核查与审计。
气旋/年份与训练时次接近仅保留为 tag，不以此排除或失效化记录；最终真实泛化仍由未来独立观测和长期实际使用检验。

返回[资料索引](./README.md)、[永久准则](OBSERVATION_POLICY.md)或[首批历史结果](JMA_OBSERVATIONS_20261115.md)。

# NHC 飞机中心气压观测库：实际来源与采集结果

状态：**原始飞机报告及 POCI 配对已落盘；尚不是已确认可用的 H8/H9 卫星测试集。**
本次没有训练、没有计算模型精度，不以代码测试通过替代数据交付。

## 实际找到的资料

### NHC Pacific Vortex Data Message（本次新增下载）

来源是 NHC 归档的飞机侦察中心定位/气压报告 `REPPN2 / URPN12`，不是把最佳路径中心气压当实测：

- 归档示例：<https://www.nhc.noaa.gov/archive/recon/2018/REPPN2/>。
- 原报示例：<https://www.nhc.noaa.gov/archive/recon/2018/REPPN2/REPPN2-KNHC.201808220546.txt>。
- 老格式气压在 `H.`，新格式气压在 `D.`；解析器依据字段单位识别，不能把新格式 `H.` 的风速当压力。
- 这些是**飞机报告的中心气压产品**，不是下载了每个投落送的原始传感器廓线。
  68 份入选报文中，31 份提及中心投落送，37 份未明确压力取得方式；提及投落送也不等于
  已逐份证明压力就是未经外推的投落送海平面读数。方法标记和原报均保留，后续可审查。

实际处理的归档年份为 2015–2025。气旋/年份仅作 tag，没有训练/验证/测试气旋排除、
气旋配额或与训练记录的时间隔离。

| 项目 | 实际数量 |
| --- | ---: |
| 成功取得的年份目录 | 11 |
| 原始飞机消息 | 340 |
| 已解析消息 / 暂未解析消息 | 323 / 17 |
| 去掉同一飞机定位的更正/重传后 | 310 个定位 |
| 成功配对 POCI | 309 |
| dP >30 hPa、Himawari 几何可见、2015-07-07 之后 | **68** |
| 请求数量 / 实际取得 | 100 / 68，未凑数 |
| 站点代理记录 | **0** |
| 下载失败 | **0** |
| 已完成卫星多帧四波段核验的样本 | **0** |

68 条全部实际位于**中太平洋 CP**，不是墨西哥附近的 EP 主体区域。
名称 tag 统计：Hilda 6、Ignacio 17、Madeline 2、Lester 5、Hector 15、Lane 14、Douglas 2、Kiko 7。
年份 tag 统计：2015 年 23、2016 年 7、2018 年 29、2020 年 2、2025 年 7。
这些统计没有用于限制某一气旋或年份的条数。

| 压差区间（hPa） | 条数 |
| --- | ---: |
| 30 < dP ≤50 | 38 |
| 50 < dP ≤70 | 23 |
| 70 < dP ≤90 | 7 |
| dP >90 | 0 |

实际 dP 范围 **31–81 hPa**。POCI 配对时差 −87.5 至 +85.5 min，
飞机定位与最近轨迹点的距离为 1.539–51.215 km；保留原始定位，不将轨迹位置冒充实测飞机定位。

### 旧 ISD 站点配对（已拒收）

NOAA ISD Global Hourly 的 50 条站点配对既未确认中心代表性，也没有一条代理 dP >30 hPa。
它们已在[失败采集记录](OBSERVATION_POCI_COLLECTION_20261115.md)标记拒收，不计入这 68 条飞机记录。

## POCI 来源与一条可回查的记录

NHC 飞机消息给出中心气压，POCI 另取 IBTrACS v04r01 `USA_POCI`：

<https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.EP.list.v04r01.csv>

这里 EP 文件覆盖东/中太平洋轨迹，**文件名或气旋的 EP 编号不代表观测时仍位于东太平洋**。
按飞机报告的 ATCF ID 匹配，取最近且时差 ≤3 h 的 POCI 原值；未对 POCI 再插值，
但 IBTrACS 上游可能已经沿用/插值分析值。没有用最佳路径 `USA_PRES` 生成实测中心气压。

实际例子：Lane，2018-08-22 **05:20:20 UTC**，14.51°N、154.11°W：

- 飞机原报 `D. 930 mb`，故中心压力为 **930 hPa**。
- 最近 06:00 UTC 的 IBTrACS POCI 为 **1011 hPa**。
- **dP = 1011 − 930 = 81 hPa**，POCI 时差 +39 min 40 s。
- 消息发布于 05:46 UTC，不把发布时间当成飞机观测时间。

## H8/H9 视野：只能说明几何可见，不能据此承诺图像可用

H8/H9 轨位约为 140.7°E。EP 主体区域不可见；这 68 条实际坐标为
13.53–21.95°N、154.89–142.39°W，位于 CP，仍在地球可见盘边缘。

卫星天顶角为 **73.774–86.044°**，中位数 **79.8555°**：
≤70° 为 0 条、≤75° 为 6 条、≤80° 为 38 条、≤85° 为 65 条。
**这说明它们非常偏盘缘，不应把 68 条观测自动算作 68 条可用卫星测试样本。**
目前仅实际下载核查一条 Lane 时次的一帧 B08：

```text
s3://noaa-himawari8/AHI-L1b-FLDK/2018/08/22/0520/
  HS_H08_20180822_0520_B08_FLDK_R20_S0101.DAT.bz2
```

原生影像 5500×5500，观测坐标像元约 (5267.52, 2039.10)，目标像元亮温 **200.997 K**、有效。
1000×1000 裁切有效像素占比 **85.5241%**；目标附近 11×11 像素全部有效。
这证明该位置当时确实有 H8 B08 数据，不只是几何计算。
**但一帧一波段的有效辐射不证明台风眼可分辨，更不证明四波段六帧和暖核反演适用性。**
2015–2020 年数据应使用 H8，不能称为当时由 H9 拍摄；2025 年应核验 H9 对应源。

## 已执行命令与产物

网络调用前均清除全部名称含 PROXY 的环境变量，下载器对子请求禁用代理。
原完整 EP CSV 下载超时后，保留其 `.partial`，不把它作为完整数据使用。
已通过 HTTP Range 保存原始头和最后 10 MB 的字节段，各段校验 Content-Range、长度及一致 ETag；
拼接后首个完整年份为 2014，覆盖所需全部 2015+ 记录。字节段、校验和及生成脚本均在产物目录，
不将构造的现代子表冒称为完整历史 EP 原文件。

```bash
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
PYTHONPATH=src /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -u \
  -m scripts.collect_recon_poci \
  --output /mnt/data/H9dP/datasets/central_observations_2015_2025_20261115 \
  --track-csv /mnt/data/H9dP/datasets/central_observations_2015_2025_20261115/raw/ibtracs.EP.modern.list.v04r01.csv \
  --count 100 --workers 8
```

采集完成：**2026-11-15 04:46:08 UTC**。采集前根盘已用 343249428480 bytes，低于 375 GB 上限。
文件位于：

```text
/mnt/data/H9dP/datasets/central_observations_2015_2025_20261115/
  observations_dp_gt30.csv            68 条飞机中心气压＋POCI，尚非卫星可用性验收通过
  all_qualified_aircraft_dp_gt30.csv  当前相同的 68 条几何可见、强压差记录
  all_aircraft_poci_pairs.csv         309 条已配对记录，含未入选项
  raw/nhc/                           原始目录、报文和获取时间/SHA-256
  raw/ibtracs_source_ranges/          IBTrACS 原始 HTTP 字节段及元数据
  manifest.json                      数量、压力区间、来源及限制
  parse_failures.json                17 份暂未解析消息的 URL
  download_failures.json             空，无下载失败
  ahi_probe/probe.json               一帧 B08 的实际像素核验
  collection.log                     实际采集日志
  source_snapshot/                   采集时源码和合成测试快照
```

解析实现：[recon.py](../../src/h9dp/recon.py)，
采集器：[collect_recon_poci.py](../../scripts/collect_recon_poci.py)，
合成测试：[test_recon.py](../../tests/test_recon.py)。
中心气压和 POCI 已配对不等于完成卫星测试集。应继续补齐 WP 原始中心观测、审查压力获取方式、
核验卫星时序和眼区分辨率；不得把未完成部分写成已完成结果。

返回[数据索引](README.md)或[永久实测准则](OBSERVATION_POLICY.md)。

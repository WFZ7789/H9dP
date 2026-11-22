# V1 Release 展板：Sinlaku（2026）10 分钟降压与推算气压

**状态：已完成。** 2026-11-22，固定种子 1，数据 2026-04-11 00:00Z–2026-04-15 00:00Z，
推理 2026-04-11 05:00Z–2026-04-15 00:00Z，共 **547 步 / 10 分钟分辨率**。
**全部 dP 均由 `inference/inference_best.cpp`（`inference_best_cli`）计算**，Python 只做
文件清单、中心位置插值、子进程调度和绘图；没有任何 Python 预测模型参与推理。
观测 dP、POCI 与实测气压全部来自仓库根目录 [MOTIS_cyclone_records.json](../../MOTIS_cyclone_records.json)，
不使用 JMA BST 或其他气 pressure 产品。推算气压是模型产物，不是实测中心气压。

## 数据采集

把 2026-04-11 00:00Z 至 2026-04-15 00:00Z 的全部 10 分钟 Himawari-9 目标区
**R304（2 km）** 扫描取到同一个外部文件夹：

| 项 | 值 |
|---|---|
| 本地目录 | `/mnt/data/H9dP/datasets/himawari9_target_r304_20260411T0000_20260415T0000/` |
| 布局 | `YYYY/MM/DD/HHMM/HS_H09_YYYYMMDD_HHMM_B{08,09,10,16}_R304_R20_S0101.DAT.bz2` |
| 时次 | 577 个（4 天 × 144 + 04-15 00:00） |
| 文件 | 2308（每时次 B08/B09/B10/B16 四波段） |
| 字节 | 390,623,052（约 373 MiB） |
| 缺失 | 0 |
| 来源 | `https://noaa-himawari9.s3.amazonaws.com/AHI-L1b-Target/<YYYY/MM/DD/HHMM>/…`，`--no-sign-request` 等价公开读取 |

网络操作前清除全部名称含 `PROXY` 的环境变量，子进程不继承代理；`curl` 16 并发、单文件
失败重试 3 次，无失败记录。逐文件校验：bz2 可解、HSD 头可解析、`area=R304`、
`500×500`、波段与文件名一致、观测中点落在名义时刻后 0–10 min 内（2308/2308 通过）。
目标区随气旋移动，气旋中心距瓦片几何中心的偏移为 x 均值 5.6 px（最大 35.2）、
y 均值 4.8 px（最大 18.3），即每帧都覆盖气旋本体。

## 推理协议

| 项 | 值 |
|---|---|
| 程序 | `inference/inference_best.cpp` + `hsd.cpp` + `v1_features.cpp`，CMake Release，C++20 |
| 构建 | `-DCMAKE_PREFIX_PATH=/home/debian/miniconda3/envs/fy3-aivis-v1`，启用 BZip2（直接读 `.DAT.bz2`） |
| 模型 | `fold0..4.v1m`，五个折外模型，等权平均；单折值同时记录 |
| 输入帧 | 每步 6 帧，名义时滞 −300/−180/−60/−30/−10/0 min（对应扫描中点间隔相同） |
| 中心 | IBTrACS `2026099N09152` 三小时位置线性插值，与目录 11 条记录坐标最大差 0.0004° |
| 步长 | 10 分钟，2026-04-11T05:00Z 起（数据窗起点 + 5 h 的最早可用完整 6 帧） |
| 目录时刻 | 同协议，锚定到不早于标签时刻的最近整 10 分钟扫描，复算 9 条记录 |

**一致性核对（同一程序、同一折模型）：**

- 用 `HIMAWARI_data/manifest.csv` 中 11 条 Sinlaku 记录自己的 6 帧 DAT 复算，与已发布的
  V1 折外预测最大差 **4.07×10⁻⁶ hPa**（462: 91.11612 vs 91.11612）。
- 序列任取一步（2026-04-12T00:00Z）直接手跑 CLI，五折值与 `series.csv` 记录至少 12 位一致。

## 结果

10 分钟推算降压 dP 从 34.4 hPa（04-11 07:10Z）升至 **111.5 hPa（04-13 12:10Z）**，
随后回落到约 70 hPa。五折标准差均值 2.94 hPa，最大 9.17 hPa（4 月中旬前的组织阶段）。
目录时刻对比（MOTIS 实测 dP 为唯一真值来源）：

| 记录 | 时刻（UTC） | 质量 | POCI | MOTIS dP | V1 dP（五折） | ±σ | V1 留出折 dP | 推算气压 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 462 | 2026-04-12 05:00:00 | Good | 1003.8 | 83.4 | 87.02 | 2.30 | 91.56 | 916.8 |
| 463 | 2026-04-12 11:10:00 | Good | 1005.9 | 107.0 | 107.62 | 1.20 | 108.79 | 898.3 |
| 464 | 2026-04-12 17:10:00 | Good | 1004.1 | 102.8 | 102.66 | 1.43 | 105.45 | 901.4 |
| 465 | 2026-04-12 23:25:00 | Good | 1007.5 | 111.2 | 111.03 | 1.27 | 113.13 | 896.5 |
| 466 | 2026-04-13 05:40:00 | Good | 1004.2 | 102.2 | 105.23 | 2.26 | 109.68 | 899.0 |
| 467 | 2026-04-13 10:15:00 | Fair | 1007.1 | 111.8 | 109.77 | 0.67 | 110.61 | 897.3 |
| 468 | 2026-04-13 17:50:00 | Good | 1005.7 | 100.2 | 105.16 | 4.07 | 113.21 | 900.5 |
| 469 | 2026-04-14 10:55:00 | Good | 1008.6 | 86.0 | 92.81 | 5.48 | 103.66 | 915.8 |
| 470 | 2026-04-14 23:05:00 | Fair | 1008.2 | 78.5 | 76.97 | 1.92 | 73.63 | 931.2 |

五折集成在这 9 条上 RMSE 3.34 hPa、MAE 2.55 hPa、偏差 +1.69 hPa；单留出折（V1 评估口径）
RMSE 8.45 hPa，与 V1 全体 8.901 hPa 同一量级。9 条太少，不能据此改动 V1 评估结论。
推算气压 = POCI − dP；POCI 取目录值按时间线性插值，跨 04-11 段为常值外推。

## 产物

外部目录 `/mnt/data/H9dP/runs/v1/sinlaku_poster_20261122/`：

| 文件 | 内容 |
|---|---|
| `series.csv` | 547 步：时刻、中心经纬度、五折 dP、均值与标准差 |
| `catalog_times.csv` | 9 条目录记录：MOTIS dP/气压/POCI、五折 dP、留出折 dP、推算气压 |
| `sinlaku_v1_poster.png` / `.pdf` | Release 展板图（上：dP；下：推算中心气压） |
| `build/inference_best_cli` | 本次使用的 C++ 可执行文件 |
| `poster.log` | 运行日志 |

生成脚本为仓库内 [scripts/sinlaku_poster.py](../../scripts/sinlaku_poster.py)：

```bash
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
R=/mnt/data/H9dP/runs/v1/sinlaku_poster_20261122
python -m scripts.sinlaku_poster \
  --dataset /mnt/data/H9dP/datasets/himawari9_target_r304_20260411T0000_20260415T0000 \
  --exe "$R/build/inference_best_cli" \
  --models /mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference \
  --series "$R/series.csv" \
  --ibtracs /mnt/data/H9dP/datasets/surface_poci_2016_2025_20261115/raw/ibtracs.WP.list.v04r01.csv \
  --reference /mnt/data/H9dP/runs/v1/rotation_seed1/formal/predictions.csv \
  --output "$R/catalog_times.csv" --figure "$R/sinlaku_v1_poster"
```

耗时：547 步 × 5 折共 2735 次 CLI 调用，10 并发 1 分 53 秒（含解码与绘图）。

## 限制

- 这是一次**已训练模型在 2026 年气旋上的连续推理展示**，不是新的泛化证据；V1 的独立现场
  观测检验仍待进行。Sinlaku 的 11 条记录本身已在 V1 的 546 条记录中。
- 观测 dP 只有 9 条目录记录，且全部落在 04-12 05:00Z 之后；04-11 段的曲线无实测对照。
- 推算气压由 MOTIS POCI 与模型 dP 合成，POCI 为目录值插值，不是实测环境气压廓线。
- 04-11 前后的五折离散较大（最大 σ 9.17 hPa），该段结论应弱于 04-12 之后。

[V1 报告](README.md) · [C++ 推理库](CPP_INFERENCE.md) · [评估口径](../EVALUATION.md) · [文档索引](../README.md)

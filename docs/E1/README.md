# E1 等温线和多输出联合回归

2026-11-21 在 546 条真实 Himawari 记录上完成种子 1 的一次记录级五折评估。目标为当前 MOTIS dP，**211 条 Good 的折外 RMSE 为 7.385 hPa**，高于 5.9 hPa 的目标。

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（预测减目标，hPa） |
|---|---:|---:|---:|---:|
| 全体 | 546 | 9.028 | 6.928 | −0.223 |
| Good | 211 | 7.385 | 5.772 | −0.712 |
| Fair | 223 | 9.556 | 7.388 | +0.995 |
| Poor | 112 | 10.618 | 8.191 | −1.726 |

3,261 个有效帧。每个波段各自定位暖核，以暖核到周围云区的亮温差定义四级等温线，测量连通区的面积、轴长、质心、暖异常和同址光谱，辅以外围云区的冷暖参考光谱。联合使用[V1 极坐标测量](../V1/README.md)。训练侧先用 90 步 Adam 拟合 B27、B31、log(1+眼区像元数) 和 CDO 温度的影像回归，再用 110 步 Adam 拟合压力对四个潜在影像信号的响应及其剩余协方差。训练目标中的块留一同时移去该观测的 dP 和四个辅助值。每个外折预测只读该外折影像，每条记录有一次折外预测。现场精度仍待独立观测检验。

RTX 5080，PyTorch 2.13.0+cu132。两阶段目标均使用 torch.compile(fullgraph=True) 和 CUDA FP64，序列相似度预计算为 FP32，后验预测为 eager FP64。首折辅助阶段和压力阶段首步含编译为 3.125 s、3.015 s，随后步中位数为 20.28 ms、11.16 ms。三个外部输入均来自本地。输出位置为 `/mnt/data/H9dP/runs/e1/isotherms_seed1/formal/`。

仓库根目录、fy3-aivis-v1 环境。先按 V1 报告生成极坐标测量，再执行以下命令。复跑时使用新的输出目录。

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
R=/mnt/data/H9dP/runs/e1/isotherms_seed1
P=/mnt/data/H9dP/runs/v1/rotation_seed1/features.npz
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode isotherms --source "$S" --output "$R/features.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.e1 --source "$S" --polar "$P" --isotherms "$R/features.npz" --output "$R/formal"
```

[文档索引](../README.md)

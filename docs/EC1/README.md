# EC1 原生眼区像素与 CDO 结构

2026-11-22 完成种子 1 的记录级五折开发评估。546 条真实 Himawari 记录的**全体 dP RMSE 为 12.195 hPa**，全体 CDO 温度 MAE 为 2.848°C。影像缓存有 3,261 个可用扫描，时间汇总使用其中眼区特征有效的 3,260 帧。dP 和 CDO 温度取自根目录 MOTIS JSON。[现行评估口径](../EVALUATION.md)

| 分组 | 记录 | dP RMSE（hPa） | dP MAE（hPa） | CDO RMSE（°C） | CDO MAE（°C） |
|---|---:|---:|---:|---:|---:|
| 全体 | 546 | 12.195 | 9.488 | 5.639 | 2.848 |
| Good | 211 | 10.793 | 8.505 | 4.479 | 2.150 |
| Fair | 223 | 12.942 | 9.987 | 6.819 | 3.541 |
| Poor | 112 | 13.116 | 10.348 | 4.964 | 2.781 |

一条卷积分支读取暖核周围 49×49 窗口内的全部四波段原生像元及有效掩码。另一条分支读取 48 方位 × 48 半径的四波段 CDO 图及掩码，保留径向和方位结构。两条分支分别对近期和最新帧汇总，与闭合冷云环、环带覆盖率、光谱和按像元积分拟合的暖核描述共同构建协方差。训练同时拟合 dP、CDO 温度和 CDO 温度的条件噪声。目录的 Good、Fair、Poor 是眼区清晰度，用于训练加权和分组。CDO 后验标准差描述温度预测的不确定性。

| 按折外 CDO 标准差升序分组 | 记录 | 平均预测标准差（°C） | CDO 温度 MAE（°C） |
|---|---:|---:|---:|
| 第一组 | 137 | 1.830 | 2.043 |
| 第二组 | 137 | 2.452 | 1.941 |
| 第三组 | 136 | 2.895 | 2.620 |
| 第四组 | 136 | 3.809 | 4.799 |

模型有 5,385 个训练参数。每折 AdamW 80 步，卷积分支学习率 0.0018，协方差参数 0.045。RTX 5080 和 PyTorch 2.13.0+cu132，正式拟合开启 torch.compile(fullgraph=True)，编码器 CUDA BF16，协方差 FP64。后验预测用 eager。首折首步含编译 16.857 s，后续步中位数 31.14 ms。以上分数来自目录开发评估，现场精度留待独立观测验证。

在仓库根目录和 fy3-aivis-v1 环境执行。极坐标描述和连续云环来自已完成的 V1 和 C1 特征准备。运行产物位于 `/mnt/data/H9dP/runs/ec1/pixels_seed1/`，逐记录预测、CDO 标准差及检查点位于 `formal/`。复跑使用新的输出目录。[V1 特征准备](../V1/README.md) · [C1 特征准备](../C1/README.md)

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
P=/mnt/data/H9dP/runs/v1/rotation_seed1/features.npz
C=/mnt/data/H9dP/runs/c1/cdo_ring_seed1/cdo.npz
I=/mnt/data/H9dP/runs/s1/eye_sampling_seed1
R=/mnt/data/H9dP/runs/ec1/pixels_seed1
mkdir -p "$I" "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode sampling --source "$S" --output "$I/eye.npz"
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode eye_cdo --source "$S" --output "$R/pixels.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.ec1 --source "$S" --polar "$P" --cdo "$C" \
  --sampling "$I/eye.npz" --pixels "$R/pixels.npz" --output "$R/formal"
```

[文档索引](../README.md)

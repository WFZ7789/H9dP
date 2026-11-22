# C1 环绕眼区的 CDO 冷云

2026-11-22 完成种子 1 的五折开发评估，546 条真实 Himawari 记录的全体 dP RMSE 为 **9.112 hPa**，全体 CDO 温度 MAE 为 2.315°C。训练目标取自根目录 MOTIS JSON。诊断中，53 条目录眼区像元数小于 34 的 Good 记录 dP RMSE 为 9.740 hPa，49 条目录 CDO 低于 −74°C 的 Good 记录为 10.368 hPa。[现行评估口径](../EVALUATION.md)

| 分组 | 记录 | dP RMSE（hPa） | dP MAE（hPa） | CDO RMSE（°C） | CDO MAE（°C） |
|---|---:|---:|---:|---:|---:|
| 全体 | 546 | 9.112 | 6.911 | 5.249 | 2.315 |
| Good | 211 | 7.812 | 6.115 | 4.027 | 1.678 |
| Fair | 223 | 9.611 | 7.308 | 6.424 | 2.858 |
| Poor | 112 | 10.292 | 7.623 | 4.646 | 2.434 |

在暖核周围按 48 个方位和 57 个径向位置读四波段亮温。对四种径向厚度寻找暖端最低且在接缝处闭合的路径，另保留扇区冷端分位数、冷云覆盖率及眼区与云顶温差。80 步 Adam 从这些影像结构估计 CDO 温度。压力回归用训练内留一和外折 CDO 预测计算热权重，再读眼区和云场特征，训练 130 步。两阶段学习率均为 0.045。质量只参与训练权重与分组统计，折外压力端只使用影像和影像估计的 CDO。

CUDA FP64 拟合启用 torch.compile(fullgraph=True)，后验计算为 eager FP64。首折 CDO 阶段首步含编译 2.413 s，随后中位数 5.63 ms。压力阶段分别为 0.486 s 和 5.67 ms。真实现场精度留待独立观测检验。

在仓库根目录、fy3-aivis-v1 环境执行。实际产物位于 `/mnt/data/H9dP/runs/c1/cdo_ring_seed1/`。复跑选新的输出目录。[极坐标眼区描述生成](../V1/README.md)

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
P=/mnt/data/H9dP/runs/v1/rotation_seed1/features.npz
R=/mnt/data/H9dP/runs/c1/cdo_ring_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode cdo --source "$S" --output "$R/cdo.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.c1 --source "$S" --polar "$P" --cdo "$R/cdo.npz" --output "$R/formal"
```

[文档索引](../README.md)

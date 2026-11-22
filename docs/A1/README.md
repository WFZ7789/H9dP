# A1 云系检索和热结构回归

2026-11-21 完成种子 1 的记录级五折开发评估。546 条 Himawari 记录共含 3,261 个有效帧，MOTIS dP 取自根目录 JSON。全体折外 RMSE 为 **8.962 hPa**。现场精度留待独立观测检验。

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（hPa） |
|---|---:|---:|---:|---:|
| all | 546 | 8.962 | 6.830 | +1.177 |
| Good | 211 | 7.069 | 5.594 | +0.756 |
| Fair | 223 | 9.673 | 7.354 | +2.121 |
| Poor | 112 | 10.538 | 8.118 | +0.091 |

把 480×480 原生像元划为 20×20 网格，移除径向均值，比较六帧云系纹理和若干位移。将各记录与训练侧支持记录的相似度映射成 Gram 矩阵，与极坐标热结构的共同趋势一起拟合。140 步 Adam，学习率 0.04，目标为加权留一误差和边际似然。 RTX 5080，PyTorch 2.13.0+cu132。拟合使用 torch.compile(fullgraph=True) 和 CUDA FP64，后验预测使用 eager FP64。首折首步含编译耗时 3.952 s，同折后续步中位数 6.65 ms。

仓库根目录及 fy3-aivis-v1 环境下执行。极坐标输入来自已生成的 V1 描述文件。运行产物位于 `/mnt/data/H9dP/runs/a1/analogs_seed1/formal/`，包含逐记录预测、逐折权重和指标。复跑时使用新的目录。[V1 输入生成命令](../V1/README.md)

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
P=/mnt/data/H9dP/runs/v1/rotation_seed1/features.npz
R=/mnt/data/H9dP/runs/a1/analogs_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode field --source "$S" --output "$R/field.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.a1 --source "$S" --polar "$P" --field "$R/field.npz" --output "$R/formal"
```

[文档索引](../README.md)

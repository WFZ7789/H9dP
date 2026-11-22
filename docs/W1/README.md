# W1 眼区多尺度辐射响应

2026-11-22 完成种子 1 的五折开发评估。546 条真实 Himawari 记录共 3,261 个可用扫描，眼区描述取得 3,260 帧。目标是根目录 JSON 的 MOTIS dP，全体折外 RMSE 为 **9.807 hPa**。[现行评估口径](../EVALUATION.md)

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（hPa） |
|---|---:|---:|---:|---:|
| 全体 | 546 | 9.807 | 7.611 | +0.108 |
| Good | 211 | 8.217 | 6.739 | −0.675 |
| Fair | 223 | 10.302 | 7.913 | +1.291 |
| Poor | 112 | 11.417 | 8.654 | −0.769 |

每帧将四波段亮温转为相对辐射度，读取七级高斯平滑下的暖核响应及不同半径的热红外分位数。训练侧将 Good/Fair/Poor 权重设为 1/0.34/0.18，给 Fair 和 Poor 分别加入辅助协方差项，后验输出取共享气压函数。每折使用 Adam 140 步，学习率 0.045。CUDA FP64 拟合启用 torch.compile(fullgraph=True)，后验预测用 eager。首折首步含编译 4.027 s，同折后续步中位数 5.86 ms。现场精度留待独立观测检验。

仓库根目录及 fy3-aivis-v1 环境运行。输出位于 `/mnt/data/H9dP/runs/w1/radiance_seed1/`。复跑时另选输出目录。

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
R=/mnt/data/H9dP/runs/w1/radiance_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode radiance --source "$S" --output "$R/features.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.w1 --source "$S" --features "$R/features.npz" --output "$R/formal"
```

[文档索引](../README.md)

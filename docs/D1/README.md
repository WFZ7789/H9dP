# D1 多尺度亮温分布回归

2026-11-21 完成种子 1 的记录级五折开发评估。D1 从原生六帧 Himawari B08、B09、B10、B16 估计当前 MOTIS dP。546 条真实记录共含 3,261 个有效帧，全体折外 RMSE 为 **9.798 hPa**。独立观测精度仍待检验。[现行评估口径](../EVALUATION.md)

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（预测减目标，hPa） |
|---|---:|---:|---:|---:|
| 全部 | 546 | 9.798 | 7.434 | −0.060 |
| Good | 211 | 8.390 | 6.556 | −0.658 |
| Fair | 223 | 10.162 | 7.652 | +0.997 |
| Poor | 112 | 11.394 | 8.655 | −1.041 |

以给定轨迹中心为搜索起点，对 B10 与 B16 的亮温均值做多尺度空间滤波，选取暖核后，提取内区和外围环带的分位数、按 B16 排序的同址光谱、径向亮温差及外围八个方位的平均光谱。以最新帧、近期指数加权均值、近期与较早帧之差汇总。每个外折仅用训练记录缩放、拟合各组主成分及模型。递归特征机进行三次多输出核岭求解，前两次用压力和辅助任务的输入梯度更新距离。辅助监督取训练记录的当版 MOTIS B27、B31、log(1+眼区像元数)、CDO 温度。推理只用影像描述。质量用于训练侧噪声设定和评估分组。

每条记录恰有一个折外预测。RTX 5080，PyTorch 2.13.0+cu132，训练为 CUDA FP64、torch.compile(fullgraph=True)，后验预测为 eager。首折首次求解和度量更新共 5.446 s（含编译），同折下一次求解和更新为 5.42 ms，第三次只求解为 3.89 ms。上述计时不含 CPU 特征提取和主成分计算。

在仓库根目录及 fy3-aivis-v1 环境运行。影像准备另外使用 SciPy 的高斯滤波器。仅读取本地数据。实际外部目录为 `/mnt/data/H9dP/runs/d1/signature_seed1/`，其中 `folds/` 保存检查点、逐记录预测、指标和计时。复跑时使用新的输出目录。

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
R=/mnt/data/H9dP/runs/d1/signature_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_d1 --source "$S" --output "$R/features.npz" --workers 8
OPENBLAS_NUM_THREADS=2 python -um scripts.d1 --source "$S" --features "$R/features.npz" --output "$R/folds"
```

[文档索引](../README.md)

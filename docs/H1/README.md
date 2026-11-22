# H1 环境条件光谱回归

2026-11-21 完成种子 1 的记录级五折开发评估。546 条 Himawari 记录共含 3,261 个有效帧，MOTIS dP 取自根目录 JSON。全体折外 RMSE 为 **9.271 hPa**。现场精度留待独立观测检验。

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（hPa） |
|---|---:|---:|---:|---:|
| all | 546 | 9.271 | 7.126 | -0.154 |
| Good | 211 | 7.710 | 6.088 | -0.600 |
| Fair | 223 | 9.716 | 7.509 | +0.753 |
| Poor | 112 | 10.910 | 8.319 | -1.122 |

按 B16 给外围两环的水汽波段建参考曲线。在相同 B16 亮温处计算眼区和外围水汽波段差，保留峰值、径向尺度和外围基线。折内部分白化光谱通道，学习通道之间的正半定协方差。加入有原生 R3 导航的 298 条图像中心纬度。125 步 Adam，核权重和噪声学习率 0.04，光谱协方差因子学习率 0.007。 RTX 5080，PyTorch 2.13.0+cu132。拟合使用 torch.compile(fullgraph=True) 和 CUDA FP64，后验预测使用 eager FP64。首折首步含编译耗时 3.920 s，同折后续步中位数 8.16 ms。

仓库根目录及 fy3-aivis-v1 环境下执行。极坐标输入来自已生成的 V1 描述文件。运行产物位于 `/mnt/data/H9dP/runs/h1/column_seed1/formal/`，包含逐记录预测、逐折权重和指标。复跑时使用新的目录。[V1 输入生成命令](../V1/README.md)

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
P=/mnt/data/H9dP/runs/v1/rotation_seed1/features.npz
R=/mnt/data/H9dP/runs/h1/column_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --mode column --source "$S" --output "$R/features.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.h1 --source "$S" --polar "$P" --column "$R/features.npz" --output "$R/formal"
```

[文档索引](../README.md)

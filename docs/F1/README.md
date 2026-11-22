# F1 按时间有序的多尺度滤波

2026-11-21 完成种子 1 的记录级五折开发评估。546 条 Himawari 记录共含 3,261 个有效帧，MOTIS dP 取自根目录 JSON。全体折外 RMSE 为 **9.197 hPa**。现场精度留待独立观测检验。

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（hPa） |
|---|---:|---:|---:|---:|
| all | 546 | 9.197 | 7.077 | +0.111 |
| Good | 211 | 7.336 | 5.759 | -0.157 |
| Fair | 223 | 9.900 | 7.625 | +1.057 |
| Poor | 112 | 10.760 | 8.467 | -1.269 |

将波段与径向梯度分组，先计算逐帧线性或平滑响应，再按三组有时序的时间权重压缩。由外折训练侧学习时间滤波矩阵，并加入可得的原生图像中心纬度。130 步 Adam，时间滤波学习率 0.012，其余参数学习率 0.04。 RTX 5080，PyTorch 2.13.0+cu132。拟合使用 torch.compile(fullgraph=True) 和 CUDA FP64，后验预测使用 eager FP64。首折首步含编译耗时 3.826 s，同折后续步中位数 6.50 ms。

仓库根目录及 fy3-aivis-v1 环境下执行。极坐标输入来自已生成的 V1 描述文件。运行产物位于 `/mnt/data/H9dP/runs/f1/temporal_seed1/formal/`，包含逐记录预测、逐折权重和指标。复跑时使用新的目录。[V1 输入生成命令](../V1/README.md)

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
P=/mnt/data/H9dP/runs/v1/rotation_seed1/features.npz
R=/mnt/data/H9dP/runs/f1/temporal_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2 python -um scripts.f1 --source "$S" --polar "$P" --output "$R/formal"
```

[文档索引](../README.md)

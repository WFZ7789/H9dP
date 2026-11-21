# V1 极坐标云图与运动估计

**Good 折外 RMSE 为 6.932 hPa，全体为 8.901 hPa。** 2026-11-21，固定种子 1，546 条真实 Himawari 记录做记录级五折，每条预测一次，目标为 MOTIS dP。Good 的目标为低于 5.9 hPa。独立现场气压与未来业务精度仍待检验。

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（预测减目标，hPa） |
|---|---:|---:|---:|---:|
| 全体 | 546 | 8.901 | 6.830 | −0.015 |
| Good | 211 | 6.932 | 5.509 | −0.458 |
| Fair | 223 | 9.650 | 7.431 | +1.011 |
| Poor | 112 | 10.503 | 8.125 | −1.223 |

3,261 个有效帧来自项目的六帧四波段影像缓存。每帧从给定中心附近定位暖核，在 1 至 228 原生像素的 18 个半径上各取 96 个方位。亮温、谱差、内外圈亮温差、角向谐波及各扇区温度描述云图。相邻帧在同心环上做角向互相关，用时差换算为像素/分钟，另记录相关峰值、与零位移相关值之差和温度变化。该读数描述云团纹理移动，单位和物理含义均区别于实测风速。

各外折训练侧完成标准化、核权重和质量噪声估计。模型联合使用各分量的时间汇总、平滑亮温响应及帧集合相似度。110 步 Adam 拟合训练侧留一误差和边际似然。未单独测量运动特征对分数的贡献。RTX 5080，PyTorch 2.13.0+cu132，训练目标为 torch.compile(fullgraph=True)、CUDA FP64，帧集合相似度在 GPU 上按 FP32 预计算。折外后验预测为 eager FP64。首折首步含编译 5.188 s，后续步中位数 5.82 ms。计时仅含优化步。

在仓库根目录、fy3-aivis-v1 环境执行。实际外部目录为 `/mnt/data/H9dP/runs/v1/rotation_seed1/`，完整预测、逐折状态和计时位于 `formal/`。首次在特征文件生成前调用训练入口，在读文件时退出。下列命令顺序执行，复跑时选新的输出目录。数据全部来自本地。

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
R=/mnt/data/H9dP/runs/v1/rotation_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --source "$S" --output "$R/features.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.v1 --source "$S" --features "$R/features.npz" --output "$R/formal"
```

[文档索引](../README.md)

# V1 极坐标云图与运动估计

**全体折外 RMSE 为 8.901 hPa。** 2026-11-21，固定种子 1，546 条真实 Himawari 记录做记录级五折，每条预测一次，目标为 MOTIS dP。独立现场气压与未来业务精度仍待检验。[现行评估口径](../EVALUATION.md)

| 分组 | 记录 | RMSE（hPa） | MAE（hPa） | 偏差（预测减目标，hPa） |
|---|---:|---:|---:|---:|
| 全体 | 546 | 8.901 | 6.830 | −0.015 |
| Good | 211 | 6.932 | 5.509 | −0.458 |
| Fair | 223 | 9.650 | 7.431 | +1.011 |
| Poor | 112 | 10.503 | 8.125 | −1.223 |

## 折外误差随降压的变化

**降压越大，偏差由高估转为低估，但 RMSE 并没有持续增大。** 同一批 546 条折外记录中，
目标 dP 为 30–<40 hPa 时平均高估 11.0 hPa，60–<80 hPa 时平均误差接近零，
≥100 hPa 时平均低估 5.2 hPa。50–<100 hPa 各箱 RMSE 为 7.4–8.4 hPa，
≥100 hPa 回升至 10.1 hPa（67 条）。这符合 V1 把降压幅度估窄的表现；
dP <40 hPa 共 19 条，弱端数值不宜外推。以上均对照 MOTIS 目录标签，不是独立实测检验。

改按 **V1 推算中心气压**分箱后，没有同样的单调偏差：误差与横轴的相关从 −0.415 变为 +0.010。
推算气压 <960 hPa 的四箱偏差均在 ±0.72 hPa 内，最弱端（≥960 hPa，29 条）偏差为 +2.84 hPa。
推算气压和目录气压使用同一条 MOTIS 环境气压，不包含环境气压估计误差。

[两种分箱对照、误差图与复现命令](ERROR_VS_DP.md)

## 模型与训练

3,261 个有效帧来自项目的六帧四波段影像缓存。每帧从给定中心附近定位暖核，在 1 至 228 原生像素的 18 个半径上各取 96 个方位。亮温、谱差、内外圈亮温差、角向谐波及各扇区温度描述云图。相邻帧在同心环上做角向互相关，用时差换算为像素/分钟，另记录相关峰值、与零位移相关值之差和温度变化。该读数描述云团纹理移动，单位和物理含义均区别于实测风速。

各外折训练侧完成标准化、核权重和质量噪声估计。模型联合使用各分量的时间汇总、平滑亮温响应及帧集合相似度。110 步 Adam 拟合训练侧留一误差和边际似然。RTX 5080，PyTorch 2.13.0+cu132，训练目标为 torch.compile(fullgraph=True)、CUDA FP64，帧集合相似度在 GPU 上按 FP32 预计算。折外后验预测为 eager FP64。首折首步含编译 5.188 s，后续步中位数 5.82 ms。计时仅含优化步。

[10 分钟运动消融](MOTION_ABLATION_20261122.md)：保留其他多时次云图特征，仅去掉 10 分钟运动分量，
全体 RMSE 从 8.9014 变为 8.9139 hPa，配对区间跨零。当前角向纹理位移没有显示可辨识的定强收益，
未验证为真实风场，也未检验移除这两帧云图本身的效果。

在仓库根目录、fy3-aivis-v1 环境执行。实际外部目录为 `/mnt/data/H9dP/runs/v1/rotation_seed1/`，完整预测、逐折状态和计时位于 `formal/`。首次在特征文件生成前调用训练入口，在读文件时退出。下列命令顺序执行，复跑时选新的输出目录。数据全部来自本地。

```bash
S=/mnt/data/H9dP/runs/m1_adapt/data_v1
R=/mnt/data/H9dP/runs/v1/rotation_seed1
mkdir -p "$R"
export PYTHONPATH=src:. MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1 python -um scripts.prepare_v1 --source "$S" --output "$R/features.npz"
OPENBLAS_NUM_THREADS=2 python -um scripts.v1 --source "$S" --features "$R/features.npz" --output "$R/formal"
```

Release 展板：[Sinlaku（2026）10 分钟降压与推算气压展板](SINLAKU_POSTER_20261122.md)；
推理件验收见 [C++20 推理库](CPP_INFERENCE.md)，发布口径、误差结构与时延拆分见
[发布分析](RELEASE_ANALYSIS.md)，组织度机制与特征消融见
[Why it works](WHY_IT_WORKS.md)。

[文档索引](../README.md)

# B1：岭回归、浅层提升树和跨架构组合

**已停止探索。** 2026-11-20 完成真实记录开发复验；没有达到 Good RMSE < 5.9 hPa。与 [R1](../R1/TRACKING_RESPONSE_20261120.md) 共用 546 条记录（211 Good、223 Fair、112 Poor）及记录级 5 折，不使用外折标签选树或调权，不是独立观测测试。

对固定中心和追踪中心分别拟合、逐记录等权平均。B1 在九组 R1 图像描述上比较三项：按特征组缩放的岭回归；以训练侧加权均值为初值的 128 棵深度 3 的提升树；以岭回归为初值的同预算残差树。每层在各组抽取的候选特征、训练侧五个分位阈值中选共享切分。学习率 0.08、叶惩罚 4、岭惩罚 0.25；质量训练权重 1/0.42/0.25。没有网络下载或新观测。训练为 CUDA `torch.compile(fullgraph=True)`（岭 FP64、树 FP32），预测为 NumPy。首轮 xy 第一次岭解 3.74 s、第一棵树 4.91 s（含编译），同折后续树中位 0.46 ms；同形状缓存命中的岭解约 3 ms。

| 模型 | 9463/9464 Good / 全体 | 9465–9467 Good / 全体 |
|---|---:|---:|
| 岭回归 | 8.06 / 9.57 | 8.09 / 9.56 |
| 仅提升树 | 9.06 / 10.49 | 9.23 / 10.58 |
| 岭 + 残差树 | 7.79 / 9.39 | 7.84 / 9.33 |
| 基础谱 GP | 7.66 / 9.19 | 7.62 / 9.16 |
| 基础谱 GP 与岭 + 残差树各占一半 | 7.59 / 9.15 | **7.59 / 9.10** |

数字是逐种子完整 OOF RMSE 的算术平均（hPa）。异构等权方案在后三个种子运行前已固定。在配对的 seed/fold/record/quality/target 上计算；三个复验种子 Good 分别 **7.74、7.42、7.61**。树有互补价值，但单独替换 GP 没有带来收益。差距小，不宣称业务泛化或显著优越。

仓库根目录，环境 `fy3-aivis-v1`。运行文件、每折状态、逐记录预测、逐折耗时及复验约定都在 `/mnt/data/H9dP/runs/b1/iteration_20261120/`；复跑时改用新的外部目录。

```bash
R=/mnt/data/H9dP/runs/b1/iteration_20261120
OLD=/mnt/data/H9dP/runs/r1/iteration_20261118
mkdir -p "$R/tmp"
export PYTHONPATH=src:. TMPDIR="$R/tmp" MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for mode in xy tracked; do
  python -u -m scripts.b1_boost --features "$OLD/features_$mode" --output "$R/$mode" --seeds 9463 9464
  python -u -m scripts.b1_boost --features "$OLD/features_$mode" --output "$R/rep_$mode" --seeds 9465 9466 9467
done
```

截至停止，以上真实数据拟合和 OOF 输出均已完成，未对 B1 改动重跑最终全套单测。返回[文档索引](../README.md)。

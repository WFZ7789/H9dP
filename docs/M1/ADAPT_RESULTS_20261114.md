# M1 新数据分段开发记录：2026-11-14 UTC

## 1. 状态与结论边界

**数据准备、独立训练 5/12 epoch、联合微调 1/3 epoch 均已完成，目前已暂停，无后台长训练。最佳已观察 Good 开发 RMSE 为 11.998 hPa，未达到 < 5.9 hPa。**

这是用户允许反复检查和调整的开发实验，不是盲测。没有永久锁定验证数据：旧 train/validation 与新数据统一参与按气旋轮换的嵌套交叉拟合。协议见 [ADAPT_PLAN.md](ADAPT_PLAN.md)。

首段预测参数校准对 Good 反而恶化。继续独立训练时，五个参数 RMSE 都下降，但组合 dP 更差；精确分解确认误差交叉项由抵消转为叠加。因此停止延长独立训练，新增固定 M5 的联合 dP 小步微调，仍保留参数监督。联合 3 epoch 后同一批 Good 从独立第 12 轮的 14.228 改善到 11.998 hPa，但仍明显未达标，且存在参数语义取舍。没有扩大网络，不能宣称已解决泛化。

## 2. 数据与执行协议

- 真实 Himawari 影像，不是合成拟合实验。
- 旧缓存 329 条加新缓存 217 条，共 **546 条**，Good/Fair/Poor = **211/223/112**。
- 有完整五参数标签 **447 条**，其中 Good **171 条**；其余 99 条仅有 dP，参与当前训练侧归一化和第二阶段校准，并在后续联合微调中通过 dP 损失更新网络，不虚构参数标签。
- 15 条缺一帧记录全部保留，其中 Good 5 条；有效性掩码和空区域统计单独处理。
- 新缓存取中央 500×500；使用 nominal 时间和固定目标中心。旧缓存保留实际观测中点和逐帧中心。不是完全相同的配对口径，不能忽略这个限制。
- 2 外折 × 2 内折，按 year:name 分组，seed=42；每个外折 273 条。外折基础专家分别使用 226/221 条有参数标签的训练记录；内折使用 102–124 条。所有 546 条都在一外折参与训练、另一外折参与评估。
- 专家种子为 42 + 100×outer + 10×inner + expert_index；外折基础专家使用 42 + 100×outer + 90 + expert_index。
- 五专家结构不扩大，单套合计 86,425 参数；交叉拟合重复训练，不是新增大模型。
- AdamW，lr=3e-4，weight_decay=1e-3，batch=4，梯度裁剪 1.0；Good/Fair/Poor 损失权重 2/1/0.5。第 13、25 轮各减半，当前前 12 轮尚未触发。
- 固定编译批量形状，尾批补齐但补齐样本不参与损失。GroupNorm 不跨样本混合统计。
- 正式训练 **torch.compile + CUDA BF16**；评估明确使用 **eager BF16**。日志未出现编译失败/回退。无 activation checkpointing，与小型 M1 原配置一致。
- 环境：`fy3-aivis-v1`，Python 3.13，torch 2.13.0+cu132，RTX 5080 16 GB。正常用户执行，无网络下载或依赖安装。

## 3. 首段 5 epoch：已完成

### 3.1 实际影像预测的开发 OOF 指标

| 数学头 | 总体 RMSE，546 条 | Good RMSE，211 条 | Good MAE | Good Bias |
|---|---:|---:|---:|---:|
| 真实参数拟合的 M5，输入预测参数 | 16.910 | **12.916** | 10.207 | -4.502 |
| 内层 OOF 预测参数校准 M5 | 16.737 | **16.098** | 13.030 | -5.242 |

单位 hPa。校准略改善整体 RMSE，但明显损伤最终关注的 Good，**本段不采纳它作为改进候选**。这里比较的是当前两个头，不能与旧版 18 条 Good 的单划分成绩作受控胜负比较。

### 3.2 真实参数 oracle，仅表格诊断

同一外折训练侧拟合 M5，再输入评估侧的真实参数：

| 配对 Good 子集 | 条数 | RMSE |
|---|---:|---:|
| 合计 | 171 | **4.823** |
| 旧数据 | 81 | 4.977 |
| 新数据 | 90 | 4.680 |

这不是可部署的影像模型结果，也不是不可突破的数学下限。与上表 211 条影像 Good 的记录数不同，不能直接相减当作严格误差分解。同一批 171 条配对 Good 上，首段原 M5/校准 M5 的影像 RMSE 分别为 13.098/16.388 hPa。

### 3.3 有参数标签 Good 的诊断

| 专家 | 参数 RMSE | Bias | 真实标准差 | 预测标准差 |
|---|---:|---:|---:|---:|
| E1 B24 | 2.537 K | -0.979 K | 1.966 | 2.557 |
| E2 B36 | 1.694 K | -0.814 K | 1.717 | 1.819 |
| E3 Eye size | 0.06444 | +0.01700 | 0.07257 | 0.08493 |
| E4 B33 trunc. | 1.560 | +0.448 | 0 | 1.495 |
| E5 T CDO | 5.347 K | +0.954 K | 10.540 | 8.491 |

关键判断：

1. 这批数据没有简单复现旧 14 条开发 Good 的 B36 方差收缩；此时的低均值和跨气旋误差更值得关注，不能机械放大输出振幅。
2. 171 条配对 Good 的真实 B33 全为零，E4 仍产生虚假修正，最大预测约 9.875。
3. 单独替换各专家的 Good dP RMSE：E1/E2/E3/E4/E5 = **9.211/10.406/7.040/7.509/5.495 hPa**；其余参数保持真值。这些是诊断，不是部署结果。
4. 5 epoch 仍短，多个专家训练损失继续下降。故先恢复到 12 epoch，观察参数学习与组合误差是否同步改善，不直接判定输入不可观测，也不立即扩大模型。

### 3.4 耗时

- 首段完整交叉拟合墙钟：**389.72 s**。
- 150 个专家 epoch 的训练时间合计：**315.85 s**。
- 30 个模型的首训练步合计 **22.54 s**，含正常计算及编译/缓存开销，并非纯编译耗时。
- 后续稳态专家 epoch 中位数 **1.56 s**；内外折训练记录数不同，不代表固定单一数据规模。

## 4. 后续短段：均已完成并暂停

### 4.1 独立训练恢复至 12 epoch

从各自优化器、随机状态和 shuffle generator 恢复，保持折和训练设置。该段墙钟 548.57 s。

| 阶段 | 总体 RMSE，546 条 | Good RMSE，211 条 | Good Bias |
|---|---:|---:|---:|
| 独立 5 epoch，原 M5 | 16.910 | 12.916 | -4.502 |
| 独立 12 epoch，原 M5 | 15.632 | **14.228** | -4.334 |
| 独立 12 epoch，OOF 校准 M5 | 14.976 | 14.769 | -4.980 |

五专家在 171 条配对 Good 上的参数 RMSE 全部下降：E1 2.537→2.275，E2 1.694→1.554，E3 0.06444→0.05668，E4 1.560→1.150，E5 5.347→5.073。但同一子集原 M5 dP RMSE 为 13.098→14.625，说明单参改进不足以保证组合改进。

用每折自己的固定 M5 作精确分解。令 s 为 CDO 缩放，前四项贡献取 `s_pred * coefficient * (parameter_pred - parameter_true)`，第五项取 `(s_pred - s_true) * sum(coefficient * parameter_true)`，则其和恰为预测参数 M5 与 oracle 输出之差：

| 配对 Good 的 MSE 项，hPa² | 5 epoch | 12 epoch |
|---|---:|---:|
| oracle 残差 MSE | 23.260 | 23.260 |
| 五个参数单独贡献的二阶矩之和 | 220.004 | 175.908 |
| 参数间交叉项 | **-55.569** | **+24.007** |
| oracle 残差与参数总贡献交叉项 | -16.142 | -9.292 |
| 合计，即实际原 M5 MSE | 171.552 | 213.883 |

这是包含均值偏差的二阶矩分解，不把交叉项误称为纯协方差。独立误差减少，却失去了原来的抵消，故不再继续独立长跑。分解由诊断脚本写入 `diagnosis_epoch*.json` 的 `error_decomposition_Good`。

### 4.2 固定 M5 联合 dP 微调

这是根据上述诊断新增的协议，不伪装成预先冻结的实验：

- 从对应外折的独立 12 epoch 五专家权重出发，保持五网络结构和每折真实参数拟合的 M5 系数，不训练额外数学头；仍为 86,425 参数。
- 新 AdamW 优化器，lr=3e-5，weight_decay=1e-3；将五专家联合更新。联合种子为 5042/6042，继续使用原外折训练/评估边界。
- 损失 = 当前训练侧标准差归一化的 dP MSE + 0.25×参数辅助 MSE，沿用 Good/Fair/Poor 2/1/0.5 权重。参数损失只作用于有真标签的记录；dP 损失作用于所有当前训练侧记录。
- 先联合 1 epoch 停下检查，再恢复至 3 epoch；两个外折每条有影像记录都轮流参与梯度训练，仍无永久锁定记录。
- 训练 compile + BF16，评估 eager BF16。联合首段墙钟 51.90 s，其中首步最大 23.15 s（含编译）；恢复段 51.04 s，稳态联合 epoch 约 7.20–7.32 s/273 条。峰值分配显存约 3.09 GB。

| 阶段 | 总体 RMSE | Good RMSE | Good MAE | Good Bias |
|---|---:|---:|---:|---:|
| 联合前，独立 12 epoch | 15.632 | 14.228 | 11.151 | -4.334 |
| 联合 1 epoch | 13.843 | **12.190** | 9.692 | -2.147 |
| 联合 3 epoch | **13.644** | **11.998** | **9.488** | **-0.442** |

联合 3 epoch 的旧/新 Good RMSE 为 11.835/12.214 hPa。参数 Good RMSE：E1/E2/E3/E4/E5 = 2.060/1.431/0.05871/1.511/4.795。E3/E4 比联合前变差，特别是 Good B33 虚假修正仍存在，不能只报告 dP 改善而隐藏语义代价。

当前主要剩余问题：总体偏差已小，但预测范围仍收缩（Good 真值标准差 21.189、预测 18.106 hPa）。dP<60 的 34 条 Good 平均高估 9.104 hPa；dP≥90 的 95 条平均低估 5.805 hPa。下一轮应检查强度两端的 B24/B36/眼结构表达、区域统计与学习特征贡献，以及 E4 的状态依赖，而非继续全局平移输出或无诊断加 epoch。

所有结果仅 seed=42 的一套外折（联合种子从其派生），未做多 seed 稳定性验收。当前是开发 OOF 检查点，不是已验收的全数据部署模型。

## 5. 代码与检查

- [训练配置](../../configs/m1_adapt.yaml)
- [数据准备/可恢复短段训练](../../scripts/m1_adapt.py)
- [分组、归一化、M5 校准](../../src/h9dp/m1_adapt.py)
- [只读阶段诊断](../../scripts/diagnose_m1_adapt.py)
- [联合 dP 小步训练](../../scripts/m1_joint_probe.py)
- [合成测试](../../tests/test_m1_adapt.py)

已执行：

```bash
MKL_THREADING_LAYER=GNU PYTHONPATH=src:. \
  /home/debian/miniconda3/envs/fy3-aivis-v1/bin/python -m pytest -q
# 144 passed，最终代码完整检查 3.88 s，无告警。

/home/debian/miniconda3/envs/ai_village/bin/ruff check \
  scripts/m1.py scripts/m1_adapt.py scripts/m1_joint_probe.py scripts/diagnose_m1_adapt.py \
  src/h9dp/m1.py src/h9dp/m1_adapt.py tests/test_m1_adapt.py
# 已通过；仅使用现有 Ruff 0.16.2 作静态检查，不用该环境训练。
```

LSP 主动探测未发布 error，但该服务为 push-only/silent-on-clean，工具无法确认 clean；不将空诊断当成已证明无错误。实际验证以 pytest、Ruff、数据准备及已执行的 compiled 训练为补充。

## 6. 实际运行命令与外部产物

仓库根目录：

```bash
export MKL_THREADING_LAYER=GNU PYTHONPATH=src:.
PY=/home/debian/miniconda3/envs/fy3-aivis-v1/bin/python
RUN=/mnt/data/H9dP/runs/m1_adapt/round1_20261114
"$PY" -m scripts.m1_adapt prepare --config configs/m1_adapt.yaml
"$PY" -m scripts.m1_adapt probe --config configs/m1_adapt.yaml --run-dir "$RUN" --epochs 5
"$PY" -m scripts.diagnose_m1_adapt --run-dir "$RUN" --epoch 5
# 人工检查上述结果后才执行：
"$PY" -m scripts.m1_adapt probe --config configs/m1_adapt.yaml --run-dir "$RUN" --epochs 12
"$PY" -m scripts.diagnose_m1_adapt --run-dir "$RUN" --epoch 12
# 发现参数误差交叉项恶化后，新增联合小步实验：
JOINT=/mnt/data/H9dP/runs/m1_adapt/joint1_20261114
"$PY" -m scripts.m1_joint_probe --source-run "$RUN" --run-dir "$JOINT" --epochs 1
# 检查第 1 轮后才恢复：
"$PY" -m scripts.m1_joint_probe --source-run "$RUN" --run-dir "$JOINT" --epochs 3
```

- 数据：`/mnt/data/H9dP/runs/m1_adapt/data_v1/`，派生数组约 7.37 GB。
- 准备日志：`/mnt/data/H9dP/runs/m1_adapt/prepare.log`。
- 运行：`/mnt/data/H9dP/runs/m1_adapt/round1_20261114/`。
- 完整源码/配置快照：`source/`；SHA256 清单：`source_sha256.txt`，恢复前已核对一致。
- 独立阶段结果：`summary_epoch005.json`、`predictions_epoch005.csv`、`diagnosis_epoch005.json`，以及对应 `epoch012` 文件；历史阶段预测保留，`outer*/*/e*.pt` 是可更新的最新恢复检查点。
- 独立阶段日志：`console_epoch005.log`、`console_epoch012.log`、`outer*/*/e*.jsonl`。
- 联合实验：`/mnt/data/H9dP/runs/m1_adapt/joint1_20261114/`，单独保存新版源码快照及 SHA256；不更改独立阶段的历史源码快照。
- 联合结果：`summary_joint_epoch001.json`、`summary_joint_epoch003.json`、对应预测 CSV、`diagnosis_joint_epoch003.json`、`outer*/joint.pt`、`outer*/joint.jsonl` 和两段控制台日志。
- 运行期间根分区实际使用约 **343.23 GB**，低于 375 GB 上限；编译缓存和新增大数组放在 `/mnt/data`。

返回 [M1 索引](README.md)。

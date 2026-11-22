# R1：多尺度亮温结构、眼区追踪与谱高斯过程

**2026-11-21 完成种子 1 的五折评估，546 条记录的全体折外 RMSE 为 8.963 hPa。**
本次使用固定中心和追踪中心预测等权平均的可加响应 GP，有原生 R3 导航时加入图像中心纬度。
[纬度与眼区暖异常结果](APERTURE_20261121.md) · [现行评估口径](../EVALUATION.md)
此前的[追踪配对复验](TRACKING_RESPONSE_20261120.md)记录了门控和误差分布，以下为[首轮开发](RESULTS_20261118.md)的协议。

## 1. 目标与边界

- 首轮研发沿用 M1 自适应计划，当时的 Good 子集门槛为 5.9 hPa。
- 只读复用 `m1_adapt/data_v1` 的 546 条真实 Himawari 记录（Good/Fair/Poor = 211/223/112），
  不新增下载、不重写旧缓存。
- 真实泛化最终由未来独立观测和长期业务运行检验；本轮不声称已证明。
- 不永久封存验证集：同一批 546 条在 5 折中轮流作为评估侧。按 `record` 轮转，不做气旋
  分组排除、不加时间间隔，符合 [实测集永久准则](../data/OBSERVATION_POLICY.md)。
- 眼区追踪、特征构造和折划分都不读取压力标签；标签只进入损失与评估。

## 2. 标签无关特征

每帧以气旋中心为原点，提取三类描述（`src/h9dp/r1.py`）：

| 描述 | 内容 | 维度 |
|---|---|---:|
| `profile` | 12 个环带（0–200 px）在 B08/B09/B10/B16 上的均值、10/50/90 分位、标准差，以及 8 方位不对称度 | 448 |
| `structure` | 暖眼候选属性（偏移、内圈均值、对比度）、眼候选处 8 环带 × 4 波段统计、暖/冷对比、方位不对称 | 790 |
| `regions` | 眼区/内核/环境三区的逐波段均值、最小值和最大值 | 36 |

局部视图 `views` 为两套 64×64 采样：眼区 `stride=1`、环境 `stride=4`，各含 4 个波段和
覆盖率，以 240 K 为基准存为 centikelvin `int16`。时间汇总只使用实际可用帧和名义时刻，
最近帧权重为 `exp(Δt/45 min)`，生成 recent/latest/trend 等统计；缺帧保留显式 NaN。

暖眼候选是**图像侧假设**，不是压力标签：在 ±18 px 内以 B10 主导的暖核-环境对比度评分，
要求内圈至少 42 个、外圈至少 1440 个有效像元，再用该候选重算环带统计。没有解析出眼区的
记录仍保留中心侧统计，不伪造眼区。

## 3. 眼区追踪

`src/h9dp/r1_track.py` 用高斯加权归一化互相关，把每个较早帧配准到同一记录的分析时次，
再用配准后的中心重算特征；仅在图像侧进行，不使用 dP、MOTIS 参数或最佳路径气压。3261 个
可用帧中 2154 个接受平移，被接受帧的中位位移 11.67 px（缓存 manifest 记录）。配准结果
只影响 `features_tracked`，`features_xy` 保持原始目标中心，便于比较。

## 4. 评估协议

- 5 折、记录级轮转（`record_folds`），每个种子下每条记录恰好被预测一次；种子 9463、9464。
- 所有折的预测拼接后计算总体与 Good/Fair/Poor 分组 RMSE、MAE、Bias；没有事后选择折。
- 闭式核与自适应核在**外层训练侧**再做 3 折内层选择，再预测外层评估侧；超参数不接触
  评估侧标签。
- 谱高斯过程在每折上分别拟合两套超参数（local 初始化 evidence=0.08、全局 evidence=0.24），
  两个后验均值的算术平均作为该折预测；同样只使用该折训练侧。
- 神经网络使用固定的 64/48 epoch 预算、固定 EMA 和终端检查点评估，不在评估侧早停。
- 正式运行均启用 `torch.compile(fullgraph=True)`，编译失败会直接抛错，不回退 eager。

## 5. 模型族

| 入口 | 模型 | 说明 |
|---|---|---|
| `scripts/r1_kernel.py` | 闭式核岭回归 | RBF 核组合 `coarse`/`structural`；内层 3 折从 6 种混合 × 3 个宽度 × 2 个线性项 × 5 个惩罚中选择并混合 3 个核 |
| `scripts/r1_adaptive_kernel.py` | 自适应核诊断 | 104 个候选设置：相关度加权特征、窄 analog 核与两者叠加；内层 3 折选择前 5 个模型平均 |
| `scripts/r1_gp.py` | 谱高斯过程 | 几何先验 + 场景先验两组核，FP64 Cholesky；AdamW 120 步，余弦退火 lr=0.035，融入质量权重与尾部权重 |
| `scripts/r1_neural.py` | 小型直接回归网络 | 双尺度编码器 + 帧注意力 + 直接 dP 头，可选 5 个辅助参数头（非瓶颈）；216,951 参数，AdamW lr=1.6e-3、wd=0.025、EMA 31/32 |

神经网络的辅助参数头只作为多任务正则，dP 直接输出，不经过参数组合。损失按
Good/Fair/Poor 加权 2/1/0.5，训练时做 D4 旋转/翻转。`neural_residual` 改为像素输入、以
闭式 `structural` 核预测为初值、辅助权重 0.1、48 epoch。

## 6. 复现命令

当前运行使用项目固定种子 1。数据目录、完整训练和折外评估命令见[单种子结果与复现](APERTURE_20261121.md#复现)。此前模型的配置及实际结果见[首轮开发记录](RESULTS_20261118.md)。

<details>
<summary>2026-11-18 历史运行命令（保留当时的代码版本、种子与目录）</summary>

均使用 Conda 环境 `fy3-aivis-v1`，工作目录为仓库根。特征缓存与全部运行产物放在
`/mnt/data/H9dP/runs/r1/iteration_20261118/`，不进入 Git。

```bash
export PYTHONPATH=src:.
# 特征缓存（v2 坐标口径）
conda run --no-capture-output -n fy3-aivis-v1 python -u -m scripts.prepare_r1 \
  --source /mnt/data/H9dP/runs/m1_adapt/data_v1 \
  --output /mnt/data/H9dP/runs/r1/iteration_20261118/features_xy --workers 7

# 眼区追踪版
conda run --no-capture-output -n fy3-aivis-v1 python -u -m scripts.prepare_r1 \
  --source /mnt/data/H9dP/runs/m1_adapt/data_v1 \
  --output /mnt/data/H9dP/runs/r1/iteration_20261118/features_tracked --workers 7 --track

# 谱高斯过程（眼区追踪 + 尾部权重默认 0.45）
conda run --no-capture-output -n fy3-aivis-v1 python -u -m scripts.r1_gp \
  --features /mnt/data/H9dP/runs/r1/iteration_20261118/features_tracked \
  --output /mnt/data/H9dP/runs/r1/iteration_20261118/spectral_gp_tracked --seeds 9463

# 小型神经网络
conda run --no-capture-output -n fy3-aivis-v1 python -u -m scripts.r1_neural \
  --features /mnt/data/H9dP/runs/r1/iteration_20261118/features_xy \
  --output /mnt/data/H9dP/runs/r1/iteration_20261118/neural_direct_xy --seeds 9463 9464

# 闭式核与自适应核诊断
conda run --no-capture-output -n fy3-aivis-v1 python -u -m scripts.r1_kernel \
  --features /mnt/data/H9dP/runs/r1/iteration_20261118/features_xy \
  --output /mnt/data/H9dP/runs/r1/iteration_20261118/kernel_xy --seeds 9463 9464
conda run --no-capture-output -n fy3-aivis-v1 python -u -m scripts.r1_adaptive_kernel \
  --features /mnt/data/H9dP/runs/r1/iteration_20261118/features_xy \
  --output /mnt/data/H9dP/runs/r1/iteration_20261118/adaptive_kernel_xy --seeds 9463 9464
```

`--contour`、`--spectral`、`--snapshot`、`--tail-focus` 分别是原生眼闭包描述、独立谱组、
单次扫描快照和尾部权重开关；实际启用的配置记录在每个运行目录的 `run.json`。原生投落送
桥接用 `scripts/r1_observed.py`，参数与结果见[开发结果](RESULTS_20261118.md)。

</details>

## 7. 产物

- 每个运行目录含 `metrics.json`（总体与分组指标、逐折超参数与耗时）、`predictions.csv`
  （逐折 OOF 预测）和 `run.json`；神经网络另存每折 EMA 检查点。
- 特征缓存含 `features.npz`、`views.npy`（`int16` 64×64 双尺度视图）和 `manifest.json`。
- 原始影像、数组、检查点和日志保留在 `/mnt/data/H9dP/`，不进入 Git。

## 8. 未完成与限制

- 没有独立盲测，也没有取得可用作验证的完整现场气压集；546 条仍是记录级开发评估。
- 原生投落送桥接只有 2 条单帧配对，且 IBTrACS USA_POCI 与当时气旋强度不符，不能作为
  dP 验证（见结果文档第 5 节）。
- 眼区追踪改善 Good，但 Poor 退化；尚未解决强度分层内的误差结构。

返回[总文档索引](../README.md)。

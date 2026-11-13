# HIMAWARI dP

计划、设计、审计及文字结果统一存放在 [docs/](docs/README.md)，按模型分目录；根目录仅保留项目入口和仓库规范。

以六帧 HIMAWARI B08/B09/B10/B16 原分辨率影像估计 MOTIS dP。输入固定为 `[B,6,4,500,500]`，输出为 `[B]`，单位 hPa。直接读取 HSD 并定标，禁止使用 Satpy。原始归档与原始数据保留。

默认配置仍为 v2：六帧联合输入，2×2 像元重排为 250×250，随后 16 个卷积残差块始终保持 250×250 特征图，最后读出一个标量。不使用粗统计网格或提前的空间汇聚。实现说明见 [TRAINING_PLAN.md](docs/v2/TRAINING_PLAN.md)，配置见 [configs/train.yaml](configs/train.yaml)。

新增 **P3 initial minimal**：无前置空间编码器的动态分辨率读取 Transformer，完整模型 **108,551 个可训练参数，硬限制 <155,000**。16 个查询联合读取原生像元、动态尺度观测和环境背景；保留亮温数值旁路，直接输出 dP。独立配置见 [configs/p3_minimal.yaml](configs/p3_minimal.yaml)，结构、限制和验证命令见 [P3_MINIMAL.md](docs/P3/P3_MINIMAL.md)。复用原有数据及逐记录目标，不覆盖旧模型或检查点。P3 已完成首轮正式训练：第 40 轮早停，第 20 轮验证最佳 RMSE 为 15.069 hPa，明显弱于 v2 的 10.963 hPa，当前保留为失败基线。实现、训练结果、只读消融与后续建议见 [P3 工作报告](docs/P3/P3_WORK_REPORT.md)。

**P3.1 coverage-first 已实现**：完整原生窗口、全窗补充搜索、残差路由和分区数值读出，完整模型 **121,159 参数**。独立配置为 [configs/p3_coverage.yaml](configs/p3_coverage.yaml)，实现、资源检查、训练及只读诊断命令见 [P3_COVERAGE.md](docs/P3.1/P3_COVERAGE.md)。保留旧模型及缓存；新产物位于 `/mnt/data/H9dP/runs/p3_coverage/`，正式启动/运行状态以 `latest_launch.json` 和运行目录 `run.json` 为准。设计与复验门槛见 [NEXT_MODEL_PLAN.md](docs/P3.1/NEXT_MODEL_PLAN.md)。首轮已在第 22 轮早停，最佳验证 RMSE 为 **14.433 hPa**，未达复验门槛；[P3.1 机制审计](docs/P3.1/P3_1_AUDIT.md) 记录查询趋同、近似均匀读取、角色分工不足及细结构/时序利用弱的证据。

**T1A（test-1 a）已实现，已通过单组 24 条真实训练记录的无增强拟合检查**：关系先行的时空小波网络，完整模型 **757,233 参数**。先建立局部关系，再在多尺度特征图上有序融合六帧，最后进行统计读出；不使用查询或动态窗口。独立配置为 [configs/t1a.yaml](configs/t1a.yaml)，实现边界、原生 BF16 前反向、合成机制拟合结果和命令见 [T1A 文档](docs/T1A/README.md)及 [24 条拟合报告](docs/T1A/FIT24_RESULTS.md)。新增 `scripts/check_t1a.py`，默认只做合成前向，监督拟合必须显式启用；合成检查结果不代表真实 dP 精度。保留现有默认配置和数据契约，T1A 正式输出目录为 `/mnt/data/H9dP/runs/t1a/`。**以后正式训练必须启用 `torch.compile`**，T1A 正式配置已经开启，正式训练入口会拒绝漏开配置，并关闭编译错误的静默回退。本组 24 条 eager 拟合按原诊断协议完成：最佳训练 RMSE 为 0.926 hPa，FP32 复核为 0.894 hPa；这是拟合能力检查，不是泛化精度。**完整训练已结束（compile + BF16）**：第 93 轮早停，最佳第 73 轮验证 RMSE 为 **11.6523 hPa**。分组误差与局限见 [完整训练结果](docs/T1A/TRAINING_RESULTS_20261101T090846Z.md)，协议和日志入口见 [运行记录](docs/T1A/TRAINING_RUN_20261101T090846Z.md)。

**M1（MOTIS 参数小专家）测试阶段已完成，未达组合目标**：五个独立小网络分别预测 B24 暖异常、B36 暖异常、Eye size、B33 修正量和 CDO 温度，共 86,425 参数，再由线性 M5 组合。最终 49 条验证 RMSE 为 **14.565 hPa**，优于训练均值基线 20.487，但未优于同协议、同预算的直接回归对照 **12.947 hPa**。参数专家均优于常数基线，线性组合放大了参数误差。设计、审计与结果见 [M1 文档](docs/M1/README.md)。

## 记录独立与等权

- 唯一样本身份是 `record`。同一气旋的不同记录分别处理；气旋名称只作为输出元数据，不参与采样、权重、划分或归一化决策。
- 训练每轮对全部训练记录随机打乱、无放回遍历，每条恰好一次。
- 主损失是训练标签公共标准差归一化后的 MSE，每条记录权重相同。Good/Fair/Poor 仅用于分组指标。
- 2020–2025 年全部训练。2026 年按 record 均匀无放回选择一半记录验证，seed 为 42；当前训练 280 条、验证 49 条。同一气旋的不同记录可进入不同集合。
- 归一化仅使用训练记录，每条记录及其六帧等权。影像数组可以按帧去重存储，记录本身始终独立。

## 环境与命令

使用 Conda 环境 `fy3-aivis-v1`。已有环境包含 NumPy、PyTorch、PyYAML；必要的代码检查使用 pytest。

```bash
conda run -n fy3-aivis-v1 python --version
conda run -n fy3-aivis-v1 python -m pytest -q
```

pytest 已禁用磁盘缓存插件，避免创建隐藏文件。必要单测使用小型合成数据，不进行训练或小样本拟合验收。

P3 可单独执行 `python -m pytest tests/test_p3.py -q`，或运行 `python scripts/check_p3.py` 检查原生 500×500 输入、参数量和有限标量输出；后者仅合成前向，不读取数据或更新权重。命令均需使用上述 Conda 环境；若外部工具改写 PATH，可显式使用 `$CONDA_PREFIX/bin/python`。

准备数据：

```bash
conda run --no-capture-output -n fy3-aivis-v1 python -u scripts/h9dp.py prepare
```

已有 v1 亮温和掩码通过硬链接复用，重新生成逐记录划分、训练图像归一化和训练标签归一化，不复制约 8 GB 数组、不修改旧索引。

默认 v2 正式训练命令如下；P3 必须显式指定 `--config configs/p3_minimal.yaml`。**按 AGENTS.md，训练和测试默认使用正常用户身份；仅在确认沙箱导致 GPU 不可用或所需外部数据不可访问时，才申请以正常用户身份切换执行环境，不要求 root/sudo。没有审批入口则无需审批，不以此阻塞执行；资源仍不可访问时报告实际限制。** 默认 v2 配置启用 `torch.compile`（`compile_mode: default`）；首次图编译会有一次性开销，随后训练步更快。不要求预检报告，也没有小样本验收入口。

```bash
conda run --no-capture-output -n fy3-aivis-v1 python -u scripts/h9dp.py train
```

每轮保存无增强完整训练集和验证集的 RMSE/MAE/Bias、采样覆盖、主干与回归头的梯度和更新量。验证 RMSE 控制 ReduceLROnPlateau 并保存最佳权重；v2 默认配置执行 200 轮，不启用自动早停。历史 P3 配置记录当时关闭 `torch.compile`，最多 100 轮，验证 RMSE 连续 20 轮未改善早停；该配置仅作历史证据。以后正式训练一律按 `AGENTS.md` 开启 compile，入口会拒绝 eager 正式启动。

恢复当前 v2 架构的运行：

```bash
conda run --no-capture-output -n fy3-aivis-v1 python -u scripts/h9dp.py train --resume runs/training/实际运行目录/last.pt
```

检查点记录模型、优化器、调度器、随机状态、输入与标签归一化，以及代码、配置、数据指纹。v1 架构的旧权重不能作为 v2 的恢复检查点。

## 可见存储目录

既有生成文件和运行产物保留在工作区可见路径 `runs/`；P3 新训练输出默认放在 `/mnt/data/H9dP/runs/p3/`，不迁移旧文件；`.git`、`.codex`、`.aws` 等版本管理或环境目录不属于此要求的范围。根分区实际与预计使用量始终不得超过 **375,000,000,000 bytes**。

```text
runs/
  h9dp/
    native500_v1/       旧数组与旧索引，保留
    native500_v2/       当前逐记录数据索引、归一化与共享数组
    diagnostics/       已有诊断产物，保留
    launches/          已有启动记录，保留
    preflight/         历史报告，当前训练不依赖
  training/            新架构正式运行目录（启动后创建）
  source_before_rework/ 改动前的源码、配置、文档与指纹
  storage_migration.json
```

Git 忽略规则源文件为 [configs/git_excludes](configs/git_excludes)，当前工作区规则保存在 `.git/info/exclude`，数据、runs、检查点和字节码不进入 Git。

历史正式运行保留在 `/mnt/data/H9dP/runs/cnn_transformer_20261024T232842793115Z/`，第 40 轮结束、最佳第 10 轮。历史文件中的原运行路径属于当时的记录，目录更名映射保存在 `runs/storage_migration.json`。

当前 v2 实现已完成代码重构、逐记录数据准备，并已验证 BF16 与 `torch.compile` 的真实数据训练路径；正式预测精度仍以完整训练结果为准。

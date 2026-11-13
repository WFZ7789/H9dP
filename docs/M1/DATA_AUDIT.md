# M1 数据与数学映射审计

## 1. 状态、来源与实际执行范围

**已完成，2026-11-13 UTC。** 本次是公开 MOTIS 表格、当前截图记录及既有缓存索引的只读审计，未读取训练影像、未拟合神经网络、未启动正式训练。线性最小二乘 OOF 是真实表格诊断；层计数仅创建 PyTorch 层，不做 forward。没有神经网络 eager/compiled 精度或性能结果。

来源：

- [Zenodo record 22224266](https://zenodo.org/records/22224266)，版本 `260212`；该版本号指 2026-02-12，不是 2016 年 2 月。
- [MOTIS 论文](https://arxiv.org/html/2606.06408)，参数和公开系数见表 2，CDO 缩放见公式 3。
- 当前 [MOTIS_cyclone_records.json](../../MOTIS_cyclone_records.json)。
- 当前 `runs/h9dp/native500_v2/train.json` 与 `validation.json`，不修改原内容。

下载完整 CSV：**554,276 bytes，3288 rows，2002–2025 年**。SHA256：

```text
78e6c59100ee55bc79ef7b557f42646ff1fd46976fd122806e2cad3735ee9b75
```

下载文件、网页、审计 JSON、参数计数和代码快照均在：

```text
/mnt/data/H9dP/reference/motis/260212/
```

原始机器输出不纳入 Git。2026 年只有当前截图记录中的部分中间量，不能说 CSV 提供了该年的完整参数。

## 2. 严格配对结果

配对键为 `(name, year, MMDD, HHMM UTC, satellite)`，仅规范大小写、空白和数字前导零，不改名、不最近时刻匹配。原始 CSV 此键无重复。项目缓存的 `record` 是原 JSON 数组的 **0-based 下标**；工具同时核对名字、年份和精确 UTC 时间，避免错一行。

| 数据范围 | 记录数 | 精确匹配旧 CSV | 不匹配 |
|---|---:|---:|---:|
| 原 JSON 全量 | 562 | 460 | 102 |
| 现有主训练集 | 280 | 230 | 50 |
| 现有主验证集 | 49 | 0 | 49 |

原 JSON 的 102 条不匹配：2026 年 100 条，2021/2023 年各 1 条。当前训练集包括 2020–2025 年 232 条、2026 年 48 条；历史 232 条中 230 条匹配，其余两条为：

- record 244：Surigae，2021-04-17 13:02 UTC，Terra。
- record 376：Koinu，2023-10-08 01:35 UTC，Terra。

不自动“纠正”其时间或强配其他观测。当前 49 条主验证均为 2026 年，无旧 CSV 参数。这意味着第一版五 E 的一致来源训练集是 **230 条，而不是 3288 条或完整的 280 条**。

## 3. 旧 M11 公式复现

按下列顺序取 CSV 变量：

```text
B24 dT, B36 dT, Upper dif, Mid dif, B27 dT,
Eye size, B31-B32, B33 dT tr., Aqua Bias, Terra Bias
```

使用论文公开系数：

```text
beta0 = 12.8
beta  = [3.4313, 5.6742, 42.7566, -28.7243, -0.0785,
         61.6813, -1.3833, 2.5781, -1.0099, -0.6308]
s = (35 - min(T_CDO, -50)) / 100
dP = beta0 + s * dot(beta, X)
```

对 3288 条 CSV `dP MOTIS` 的复现误差：

| 指标 | hPa |
|---|---:|
| RMSE | 0.0444833 |
| MAE | 0.0365384 |
| Bias（复现值−CSV 值） | -0.0341461 |
| 最大绝对差 | 0.0847610 |

接近 CSV 的 0.1 hPa 输出分辨率，与公开系数的截位精度相容；不宣称逐项精确复原未公开的全部内部精度。温度截断不可省略。

另外，全部 3288 条 `Eye size` 与 `1/sqrt(max(8,npix))` 的最大绝对差约 `5.0e-10`；这是像元计数变换，不是几何眼半径。`B33 dT tr.` 在此 CSV 为非负的已变换字段，不能当成原始波段亮温。

## 4. 旧版本不是当前标签

在 230 条训练配对记录上，直接用旧 CSV 的 `dP MOTIS` 对照当前缓存 dP：

| 指标 | hPa |
|---|---:|
| RMSE | 6.086843 |
| MAE | 4.321739 |
| Bias（旧−当前） | -1.972174 |
| 最大绝对差 | 26.3 |

因此，即使 E 完美复现旧参数，直接套旧 M 也不保证复现当前目标。这里只能说明版本/来源不一致，不能据此证明旧版或新版相对独立气压观测谁更准确。

当前 JSON 的部分字段在 280 条训练、49 条验证上均非空：`t_cdo`、`pixels`、`b27`、`b31`、`poci`、`roci`。但是不能直接把名字相近的两版字段当成同一个标签：

| 230 条配对对照 | RMSE | MAE | 最大绝对差 |
|---|---:|---:|---:|
| 旧 `T CDO` vs 当前 `t_cdo`，°C | 3.283856 | 0.390452 | 49.545 |
| 旧 `npix` vs 当前 `pixels`，count | 84.174637 | 10.613043 | 1255 |

这些异常可能涉及版本、处理定义或截图转录，尚未逐条溯源。本次不删除、不修正、不自动合并，也不把任何一版无条件当成精确真值。若未来核对通过，相应 E 可以单独扩充当前训练侧的监督，不必重训其他 E。

## 5. 训练侧 5 折表格 OOF 诊断

目的：不训练影像网络，先判断参数子集和线性映射是否值得尝试。

协议：仅使用上述 230 条主训练侧配对记录；按 record 排序，NumPy `default_rng(42)` 随机排列、`array_split(...,5)`，每折 46 条。每次用其余 184 条拟合普通最小二乘截距与回归系数，预测该 46 条；汇总 230 条 OOF 预测。原始参数使用旧 CSV 值，CDO 缩放固定为上节定义。无正则、无 quality 权重、无名称分组、无 2026 验证标签参与。

- **M5**：`B24 dT, B36 dT, Eye size, B33 dT tr.`，加 `T CDO` 缩放，共 5 个输入中间量，拟合 5 个系数。
- **M11**：全部 10 个回归变量，加 `T CDO` 缩放，共 11 个中间量，拟合 11 个系数。

| 映射 | 目标 | OOF RMSE hPa | OOF MAE hPa | Bias hPa |
|---|---|---:|---:|---:|
| M5 | 旧 CSV dP | 4.698075 | 3.428424 | 0.010759 |
| M5 | 当前 dP | 7.894309 | 5.753991 | -0.010276 |
| M11 | 旧 CSV dP | 0.028310 | 0.023625 | 0.000275 |
| M11 | 当前 dP | 5.680498 | 3.844039 | -0.044482 |

解释边界：

- M11 对旧 dP 的低误差主要说明重建了原统计公式，不是新的物理预测能力。
- M5 丢弃部分修正变量，存在明显信息损失；若加入网络预测误差，结果可能更差。
- 这些使用真实中间值的诊断是 oracle 参考，不是对 E 的结果，更不是其严格数学下界。
- 数据全部来自当前训练侧，record 级折可能保留同气旋相关性，不是跨气旋独立测试或 2026 年泛化结果。
- 从 7.89 hPa 不能推导完整系统必然优于现有 v2/T1A：训练记录数和输入误差不同，必须重跑同协议直接回归对照。

## 6. 层清单参数计数

按 [PLAN.md](PLAN.md) 中层定义，用 `torch==2.13.0+cu132` 在 CPU 创建层并统计：每 E **17,285**，五 E **86,425**，另有 M5 的 5 个拟合系数。

没有实现空间区域统计与 forward，没有测显存/算量/速度，没有 `torch.compile`、BF16 前向或训练结果。计数只能证明该层清单的参数规模。

## 7. 实际命令与外部产物

仓库根目录运行；环境 `fy3-aivis-v1`，seed 42：

```bash
conda run -n fy3-aivis-v1 python scripts/audit_motis_inputs.py \
  --csv /mnt/data/H9dP/reference/motis/260212/MOTIS_dataset_260212.csv \
  --output /mnt/data/H9dP/reference/motis/260212/input_audit.json

MKL_THREADING_LAYER=GNU conda run -n fy3-aivis-v1 python \
  /mnt/data/H9dP/reference/motis/260212/count_m1_design.py

/home/debian/.pi-lens/pip-tools/bin/ruff check \
  scripts/audit_motis_inputs.py tests/test_motis_inputs.py

conda run -n fy3-aivis-v1 python -m pytest -q
```

Ruff 不在该 Conda 环境的 PATH，本次使用已有 pi-lens 工具环境中的二进制，没有新安装依赖。单测只使用合成小表格和数组，不下载数据、不训练模型。

外部目录中：`MOTIS_dataset_260212.csv`、`article.html`、`zenodo_metadata.json` 是来源留存；`input_audit.json` 保存配对记录、缺失键、输入 SHA256、诊断折及指标；`count_m1_design.py`、`m1_architecture_count.json` 保存层计数；`audit_motis_inputs_snapshot.py` 保存本次审计脚本快照；`validation.log` 保存最终检查原始输出。

最终检查结果见下节；未执行的网络验证见设计文档，不列作已完成结果。

## 8. 代码与文档检查

2026-11-13 11:43:55 UTC 开始的最终批次：

- `MKL_THREADING_LAYER=GNU conda run -n fy3-aivis-v1 python -m pytest -q`：**120 passed in 5.65s**，包括 5 个新增审计工具测试和文档链接检查。
- 已有工具环境 Ruff 检查两份变更 Python 文件：**All checks passed**。
- `lens_diagnostics(source="lsp", scope="paths", serverScope="primary")` 主语言服务器主动检查 `scripts/audit_motis_inputs.py`、`tests/test_motis_inputs.py`：两文件 clean，0 diagnostics；不是依赖空缓存判断。
- `df -B1 / /mnt/data`：根分区已用 **343,042,162,688 bytes**，低于 375,000,000,000 bytes 上限；本次未提取归档、安装依赖或复制影像数组。

原始 pytest/Ruff/容量检查输出保存在外部 `validation.log`。这些检查不代表拟议五 E 的网络数值、训练或泛化验收已经完成。

返回 [M1 入口](README.md)。

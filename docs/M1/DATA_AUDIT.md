# M1 热红外参数与目录标签的配对

2026-11-13 UTC 完成历史配对、表格诊断和参数计数。现行压力标签为仓库根目录的 MOTIS JSON。2026-11-22 已删除外部压力字段所在的原表、含外部压力的诊断产物及其旧脚本快照，仅保留压力无关的亮温与眼区参数投影。历史表格实验范围和影像训练结果分别见下文及模型结果报告。[压力来源清理](../data/PRESSURE_SOURCES_20261122.md) · [影像训练结果](RESULTS.md)

## 历史精确配对结果

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

## 层清单参数计数

按 [PLAN.md](PLAN.md) 中层定义，用 `torch==2.13.0+cu132` 在 CPU 创建层并统计：每 E **17,285**，五 E **86,425**，另有 M5 的 5 个拟合系数。

没有实现空间区域统计与 forward，没有测显存/算量/速度，没有 `torch.compile`、BF16 前向或训练结果。计数只能证明该层清单的参数规模。

## 现行审计入口

输入 `/mnt/data/H9dP/reference/motis/260212/thermal_parameters.csv` 保留 3,288 条原表配对键、亮温、光谱及眼区参数。其他来源的中间参数表也可使用，脚本只选所需中间参数，单独与目录核对缓存 dP。以下为现行入口，2026-11-22 的清理仅做逐条来源核对和合成单测。

```bash
PYTHONPATH=src:. conda run -n fy3-aivis-v1 python -m scripts.audit_motis_inputs   --csv /mnt/data/H9dP/reference/motis/260212/thermal_parameters.csv   --output /mnt/data/H9dP/runs/pressure_source_policy_20261122/thermal_audit.json
```

[模型入口](README.md)

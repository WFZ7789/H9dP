# M1：MOTIS 参数小专家测试版

M1 是本次设计的暂定名称。这里的 E 分别预测 MOTIS 的**不同输入参数**，不设路由器，不混合相同目标的专家输出；数学模型 M 将这些参数组合为 dP。

**状态：已完成表格数据审计、模型实现、两个开发运行和一次最终验证（2026-11-13 至 2026-11-14 UTC）。判定为未达组合目标。** 最终 49 条：M5 组合 RMSE 14.565 hPa，同协议直接回归对照 12.947 hPa；参数专家均优于常数基线，但线性组合放大了参数误差。

- [测试版设计与验收协议](PLAN.md)
- [MOTIS 数据配对、版本差异与数学模型审计](DATA_AUDIT.md)
- **[测试阶段结果：协议、指标、误差分析与判定](RESULTS.md)**
- 可复现的只读审计工具：[scripts/audit_motis_inputs.py](../../scripts/audit_motis_inputs.py)
- 小型合成测试：[tests/test_motis_inputs.py](../../tests/test_motis_inputs.py)

返回[文档索引](../README.md)。

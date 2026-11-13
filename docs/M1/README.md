# M1：MOTIS 参数小专家测试版

M1 是本次设计的暂定名称。这里的 E 分别预测 MOTIS 的**不同输入参数**，不设路由器，不混合相同目标的专家输出；数学模型 M 将这些参数组合为 dP。

**状态：已完成真实表格数据审计及设计层参数计数；网络、训练配置和训练入口尚未实现，未启动任何神经网络训练。** 表格 OOF 诊断不是影像模型结果，也不是 2026 年泛化结果。

- [测试版设计与验收协议](PLAN.md)
- [MOTIS 数据配对、版本差异与数学模型审计](DATA_AUDIT.md)
- 可复现的只读审计工具：[scripts/audit_motis_inputs.py](../../scripts/audit_motis_inputs.py)
- 小型合成测试：[tests/test_motis_inputs.py](../../tests/test_motis_inputs.py)

返回[文档索引](../README.md)。

# 过滤后 Row FULL-data RMSE 补齐

完成记录：2026-10-13（Asia/Shanghai）。三柱×五seed共15个context、75个方法/种子结果；35个新增结果，40个历史结果按哈希复用。 本次仓库整理只归档已有结果，没有重新训练或评分。

源模型为资格验证后的QGeoGNN-V2 row seed42；目标为8g/25g/40g，五个固定outer seeds。比较zero-shot、scale-only、affine、Conditional-EA与paper-style current-V2；最后一项是项目内的论文启发实现。过滤阈值继承原协议，8g/25g为V1≤60、V2≤120 mL，40g为V1≤150、V2≤200 mL，没有新增阈值选择。

预算计入gradient-train和validation已揭示标签。各预算保持相同的验证/测试ID，训练集合使用原随机顺序与过滤后可训练人口的交集，再取嵌套前缀；FULL使用全部合格训练行。Zero-shot不使用目标训练标签。训练和模型选择没有使用测试标签；新增预测全局冻结后才执行既有评分，历史FULL结果复用并校验哈希。测试集已在历史研究中暴露，因此属于开发性描述，不是独立确认，也不能与无阈值低标签基准混合排名。

- [冻结协议](protocol.json)、[身份与划分](split_manifest.csv)、[标签账本](label_accounting.csv)。
- [五seed汇总](summary.csv)、[全部逐seed指标](all_metrics.csv)、[预测冻结](all_predictions_frozen.json)、[完成审计](completion_audit.json)。
- 入口：`scripts/studies/run_row_full_data_rmse_completion.py`；参数为`--prepare`、`--execute`、`--evaluate`。恢复执行前遵循冻结协议与既有授权；训练缓存和checkpoint在本地忽略的`runtime/`，新克隆只含紧凑结果。

最终四预算的可比曲线见[B200补齐记录](../filtered_row_learning_curve_b200_20261013/README.md)与[已有图](../../../reports/ppt_202610_transfer/figures/slide18_filtered_row_learning_curve.png)。当前下一步由[项目研究状态](../../../docs/NEXT_STAGE_DECISION.md)管理。

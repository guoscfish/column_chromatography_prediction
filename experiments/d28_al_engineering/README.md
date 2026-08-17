# D28：主动学习工程底座检查

## 决定

**通过**。本阶段只验证接口与状态，不修改Gate 0冻结科学配置，也不把2-epoch fit smoke作为科学结果。

- 统一`fit/predict`：通过；L0=50，其中train=42、validation=8，validation计入标签预算。
- 10k+分块推理：通过；10240个candidate，两套batch/chunk配置的prediction最大绝对差为0，embedding为0，顺序与身份完全一致。
- Round resume：通过；连续2轮与round 1落盘后恢复的selected/labeled/pool/RNG状态完全一致，重复query=0。
- 固定partition：通过；已冻结E2 4g row/compound与E4 8g Protocol A/B各3个outer seeds，共12份partition。

## 身份和泄漏约束

长期身份只使用`sample_id`。`canonical_index`是冻结canonical表中的审计位置；任何过滤后的临时DataFrame行号都不能写入AL状态。Stress fixture因复用574条真实图构造10240个虚拟candidate，额外显式保存`source_canonical_index`，不会冒充新的实验样本。

`fit`只接收当前labeled budget与其中固定validation subset；test不进入训练、早停、checkpoint或状态选择。`predict`按请求顺序返回quantiles和128维`h_graph`，并在每个chunk核对位置。

## 下一步

Gate 0工程检查完成，可以进入E1 signal qualification；E1不做主动学习重训。

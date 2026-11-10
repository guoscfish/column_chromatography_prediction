# 预测器证据与实现入口

当前科学基线为独立 **QGeoGNN-V2**；候选实现已迁入 [predictor/model.py](../../src/qgeognn_al/predictor/model.py)，图与输入在 [predictor/data.py](../../src/qgeognn_al/predictor/data.py)、[predictor/input_schema.py](../../src/qgeognn_al/predictor/input_schema.py)。旧 `models/qgeognn_v2.py` 只在 [固定历史源码](https://github.com/guoscfish/column_chromatography_prediction/tree/4ee480378b0293f1985e3ee4a09df2c78f77901e/src/qgeognn_al/models/qgeognn_v2.py) 恢复，不再作为候选的活跃路径。

状态统一含义：`CURRENT` 为本候选的主要阅读入口；`FROZEN_HISTORICAL` 为保留原身份的科学证据；`ARCHIVED_SOURCE` 为从固定 Git 来源恢复的历史内容；`SUPERSEDED` 为已被后续阶段取代的方案或快照；`ACTIVE_RUNTIME` 为 main 上受保护的运行项目，不能在候选执行或移动。状态不是新训练/揭示标签的授权。

| 证据来源 | 状态 | 建立了什么 / 未建立什么 |
| --- | --- | --- |
| [final_v2_engineering](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/final_v2_engineering/README.md) | CURRENT | 独立 V2 有效网络工程等价：458,952 参数、冻结 E0 六输出与 R2-pruned 对照零差 |
| [final_4g_qualification](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) | CURRENT | 3 seed×row/compound 六次冻结点预测资格；分位数审计不构成已校准 UQ |
| [semantic_input_audit](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/semantic_input_audit/README.md) | CURRENT | 原始输入特征与数据语义，当前模型仍需遵守 |
| [historical](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/historical/README.md) | FROZEN_HISTORICAL | 历史演进索引；其中源码相对链接按原提交解释 |
| [condition_completion](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/condition_completion/README.md) | FROZEN_HISTORICAL | 条件分支和旧预测器验证，不是当前单独新结论 |
| [performance_regression_audit](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/performance_regression_audit/README.md) | FROZEN_HISTORICAL | R0/R1 同时改变多项配置，不能当纯 loss 权重因果对照 |
| [r2_pruned_requalification](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/r2_pruned_requalification/README.md) | FROZEN_HISTORICAL | 函数保持剪枝和后续独立 V2 的来源 |
| [clean_4g_baseline_qualification](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/clean_4g_baseline_qualification/QUALIFICATION_REPORT.md) | FROZEN_HISTORICAL | 失败架构的负结果；不因性能差删除 |
| [clean_qgeognn](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/clean_qgeognn) | SUPERSEDED | Clean 结构方案与原始工程材料，未作为当前点预测器 |
| [4g_source_benchmark](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/predictor/4g_source_benchmark/README.md) | SUPERSEDED | 早期基准预注册，不代表已执行当前正式资格基准 |

main 的 [五 seed L333 分位数/损失离线复盘](https://github.com/guoscfish/column_chromatography_prediction/tree/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md) 为新的冻结 eval 诊断，不复制到此候选。它与旧 3-seed qualification 的训练预算、划分、角色不同，不能直接作优劣归因；V2 梯度较大不等于整条 Adam 轨迹由 V2 主导。q50 用 MSE 学习，宽度和 crossing 未证明有控制既有信号后的 AL 效率收益。

固定 checkpoint 的 schema、输入、normalization 和初始状态均是科学身份的一部分。正式报告已明确关键 checkpoint 为本地 ignored binary；Git 中的报告/哈希不保证 fresh clone 拥有该文件。工程重构的真实 cache/checkpoint 回归与未验证边界见 [测试报告](../../docs/repository/phase2/TEST_REPORT.md)。

当前研究判断统一见 [NEXT_STAGE_DECISION](../../docs/NEXT_STAGE_DECISION.md)，模型输入合同见 [INPUT_SCHEMA](../../docs/model/INPUT_SCHEMA.md) 与 [paper-code 合同审计](../../docs/model/PAPER_CODE_CONTRACT_AUDIT.md)。历史负结果和 test 数据消费声明继续保留，旧建议不构成新实验授权。

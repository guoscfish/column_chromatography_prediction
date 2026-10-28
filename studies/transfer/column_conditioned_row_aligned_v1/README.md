# A1/A2 Row-aligned v1：Phase 2 完成

完整五seed内部Row-CV：50模型（Phase1复用10，Phase2新增40）。没有FULL refit、outer scoring或额外方法。研究是开发性诊断；Phase1与历史Row外层结果已经曝光。

- PROTOCOL.md：原设计与Phase2授权/曝光/标签权限补充。
- PHASE2_PROTOCOL.json、PHASE2_EXECUTION_PROVENANCE.json：本阶段冻结authority，引用原protocol.json。
- REPORT.md、IMPLEMENTATION_AUDIT.md：完整结果、限制及审计。
- INNER_RESULTS.csv、PAIRED_COMPARISONS.csv、TRAINING_HISTORY.csv：全50模型聚合。
- INNER_SUMMARY.csv、SEED_RESULTS.csv、SEED_PAIRED_COMPARISONS.csv：两种gain、描述性SD、每seed结果。
- EPOCH_SELECTION.csv：仅新Row-CV的10项未来候选epoch，refit_executed=False。
- GROUP_VS_ROW_EPOCH_SELECTION.csv、EPOCH_SELECTION_COMPARISON.csv：历史/新CV选择诊断，不能混作主实验预算。
- TAIL_PAIRED_COMPARISONS.csv、MAX_ERROR_TRADEOFFS.csv：全部负结果与最大误差诊断。
- five_seed_row_cv_paired_gains.png、group_vs_row_cv_best_epochs.png、a2_center_width_tail_diagnostics.png：三张分析图，北大红/蓝灰配色。
- learning_curves/：各seed全部五折完整训练/验证曲线。
- phase1_archive/：按字节保留原43个根产物及110个runtime文件的来源hash；原pilot runtime未改动。
- phase2/seed_*/COMPLETE_AUDIT.json：每seed十模型的完整性与重放收据。

在仓库根目录使用已验证fish Python。仅重新汇总与核验（不训练）：

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/transfer/column_conditioned_row_aligned_v1/summarize_phase2.py
```

run_phase2.py的完整入口按receipt复用已完成的四个seed，拒绝拟合pilot；没有refit或outer score路径。旧run_phase1.py/code/protocol/provenance作为冻结来源保留，不重跑Phase1。原protocol.json中PHASE_1_ONLY的文字属于原冻结记录，Phase2扩展由PHASE2_PROTOCOL及PROTOCOL.md补充定义。

本轮源标签权限仅source_train，源validation指标仅在已有Phase1记录中保留；Phase2不解析其标签、不报告零占位损失。目标仅各seed的gradient_train标签，features严格usecols白名单。19项执行/边界测试及3项独立统计算术测试通过。

Phase2已停止。后续FULL refit与外层评分均需用户另行确认；已曝光外层测试只能用于DEVELOPMENTAL CONFIRMATION，独立泛化需要未曝光数据。

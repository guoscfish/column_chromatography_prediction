# Stage 2：nested calibration

本阶段由stage1已知结果触发，虽在新结果前另行冻结，仍属结果后的development研究；五个Row划分重叠，不是独立确认。

- [执行前协议](STAGE2_PROTOCOL.md)
- [真实资格、校准与Row结果](STAGE2_RESULTS.md)
- [证据决策与未决问题](STAGE2_NEXT_STAGE_DECISION.md)
- 科学配置/实现/输入由独立freeze/seal绑定；`reporting_manifest.json`仅绑定报告，不改变科学决定。

本报告可重复执行：`/Users/fish/miniforge3/envs/fish/bin/python /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/epig_prediction_oriented_20261117/stage2_nested_calibration/report_stage2.py`。它只读取完成的OOF/fit receipts和metadata，不拟合、不揭示标签。实际状态：INCOMPLETE_POSTERIOR_QUALIFICATION。

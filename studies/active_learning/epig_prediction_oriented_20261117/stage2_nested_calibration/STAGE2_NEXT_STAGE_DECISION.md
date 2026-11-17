# 第二阶段证据决策

实际资格状态：**INCOMPLETE_POSTERIOR_QUALIFICATION**；完成60/60个校准候选。本阶段由stage1已知结果触发，虽在新结果前另行冻结，仍属结果后的development研究；五个Row划分重叠，不是独立确认。

1. **能否可靠计算EPIG？** 数值实现已通过stage1矩阵/规模测试；stage2只检验有限Gaussian幅度校准资格。当前五seed完整资格尚未齐全，不能用可计算性代替可信概率资格。
2. **EPIG信息是否超出IVR？** 同一已拟合family的正对角DΣD变换不改变Gaussian单点/条件化EPIG；它可改变trace-IVR的端点风险权重。校准改变选模资格或family时才可间接改变EPIG决策，不能把幅度校准直接说成EPIG获得新信息。同posterior真实学习对照才能识别objective的效用；stage1排名诊断不能直接回答新增预测价值。
3. **是否改善Row AULC/终点？** 尚无完整五seedL1005主配对，未知。温度资格改善不等于真实收益。
4. **若无收益的原因？** 依据实际资格、joint评分、批量冗余与预测器控制分别讨论；未执行的性能不能被称为无收益。Gaussian尖峰重尾/均值失配限制仍在，不能把失败归到EPIG目标。
5. **是否进入LLM辅助？** 仅有限概率资格/代理MI仍不充分。需真实Row学习收益与稳定性证据；本次未调用新付费API。
6. **仍未知？** 完整长期AL、独立cohort、非可分离多输出、异方差/重尾代理和匹配新预测器控制，均不从本温度试验推定。

保留完整候选和失败原因；不删seed，不事后改门槛，也不继续无限增加校准方法直到通过。若预算未完且候选仍在执行，继续完成冻结比较；若全部60状态完成仍不齐全，则本有限校准族为资格不足A/F，不是EPIG普遍NO-GO。

| calibration | candidate_states | qualified_candidates | endpoint_coverage80_above_090 | endpoint_coverage80_below_070 | endpoint_coverage95_outside_bounds | endpoint_residual_RMS_above_150 | endpoint_residual_RMS_below_070 | candidate_joint_NLPD_degradation_above_010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 20 | 3 | 26 | 0 | 0 | 0 | 0 | 0 |
| nll | 20 | 0 | 37 | 0 | 1 | 0 | 1 | 0 |
| central80 | 20 | 0 | 0 | 0 | 7 | 38 | 0 | 2 |

全部60候选已完整执行。校准没有新增合格候选：仅identity保留原3seed本地后验，887/6101仍不齐全。NLL与中央80%温度的不同失败维度见STAGE2_RESULTS，不把它缩写成EPIG失败。充分真实AL验证目标尚未完成时，必须在任何总体总结中显著披露；本阶段不能提供不存在的五seedEPIG AULC/终点表。

本阶段不是stage1原预注册的一部分；报告必须同时保留stage1与stage2，不隐藏最初资格失败或挑选有利窗口。真正独立确认仍需新冻结数据/实验批次。

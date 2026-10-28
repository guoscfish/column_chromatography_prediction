# 当前研究状态

更新：2026-10-28。本页是项目级结论与下一步的唯一入口；具体数值、冻结门槛和使用过的数据以链接的实验报告及协议为准。

## 本轮执行

当前五seed Free LLM32 / LCMD32延长实验以[自动生成报告](../reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md)为唯一进度与数值入口，目标是每条1485标签、37拟合、36批次。只统计完成且冻结的点；监控继续推进未完成轨迹，并在既有冻结预算内恢复可恢复故障。历史阶段的STOPPED、NEEDS_ATTENTION和计划预算不能代替当前epoch状态。

- LCMD五seed已到1485；Free已有四seed到1485，最后seed12203继续运行。具体预算与共同区间随报告更新，不在多个导航页重复维护。
- LCMD使用原冻结梯度特征实现与corrected LCMD-TP，保留同初始化及validation划分；Free旧157/6101保留Astra/high前缀，887/12203有显式限定的逻辑纠正amendment。全部解释与证据链接在当前报告中。
- 3330标签全训练集参考与后段平台期比较见[validation参考分析](../reports/free_lcmd_five_seeds_to1485_20261026/analysis/full_reference_validation_comparison_20261028.json)。未新增test评价，不能把validation平台期解释成无剩余泛化空间。
- 双专家seed157已完成1005，seed6101到525后连接故障停止；不能称为两seed1005完成。[核验记录](../reports/dual_expert_two_seeds_to1005_20261021/verified_final_status.json)。该停止记录不影响独立的Free / LCMD续跑。
- CW七臂×两seed1005扩展完成，Whitening按要求停525；[完整结果](../studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/RESULTS.md)保留原开发门槛与探索性解释。

[2026-10-13两seed快照](../reports/active_learning_latest/REPORT.md)和各阶段报告均为日期固定的历史记录，不作为当前进度。

## 预测器

**QGeoGNN-V2 已完成 4g 点预测资格验证**，状态为 `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`。

- 458,952 个参数均有梯度；与 R2-pruned 在 4,163 行上的六输出完全相同。
- 三个固定种子的 row / compound 六次验证完成。Row 插值和新 target-compound 泛化是不同任务。
- Clean 是 `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT`，仅作历史负结果。
- 当前分位数头可用于点预测，但覆盖率和 crossing 不足以支持已合格的 UQ；主动迁移仍需独立验证。

证据：[工程验证](../studies/predictor/final_v2_engineering/README.md)、[4g 正式报告](../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md)、[分位数审计](../studies/predictor/final_4g_qualification/QUANTILE_AUDIT.md)。

## 4g 主动学习

| 已完成工作 | 结论 |
| --- | --- |
| LCMD one-step pilot | 大批量一次选点有正信号；不等于完整曲线或 compound 鲁棒性 |
| B32 / B16 benchmark、Hybrid extension | B32 LCMD 有正信号；不同新增预算不能单独识别 batch-size effect |
| Sequential B32，333→1005 标签 | LCMD / Hybrid 平均 AULC 比 Random 低约 23%，均 5/5 种子获胜 |
| Kernel-IVR extension | `NO_CLEAR_IVR_IMPROVEMENT`；AULC 比 LCMD 差约 4.28%，仅 1/5 获胜 |
| IVR mechanism audit | `STOP_BEFORE_GATE_C`；稳定性门槛未过，没有新增训练或效果验证 |
| Phase 0 efficiency review | 重新整理端点、AULC、N80/N90/N95 与计算成本，没有重新训练 |

Sequential 的原始分类 `NO_CLEAR_SEQUENTIAL_AL_GAIN` 保留不改：它未确定 LCMD 与 Hybrid 的最佳/相近类别，不能解释成“对 Random 无收益”。约 650 标签后改善放缓，稳定达到 N90/N95 仍未解决。

[Phase 1固定预算Static / Adaptive对照](../studies/active_learning/qgeognn_v2_batch_adaptivity/FINAL_REPORT.md)已完成：均333+320=653标签；Adaptive端点0.436791优于Static的0.451013，但AULC未改善，不能宣称完整曲线优势。后续LLM轨迹状态见上方当前报告。

[Phase 2 实施方案](research/4G_PHASE2_IMPLEMENTATION_PLAN_2026-09-18.md) 是后续设计。全部已完成证据见[主动学习索引](../studies/active_learning/README.md)。

## 跨柱迁移

两套研究必须分开解释：

| 研究问题 | 当前判断 | 证据 |
| --- | --- | --- |
| 无阈值、B=30/50/70/100 的匹配低标签迁移 | `SIMPLE_CALIBRATION_REMAINS_COMPETITIVE`；Conditional-EA / local identity shrinkage 保留作基线 | [Matched RMSE](../studies/transfer/matched_rmse_benchmark/MATCHED_RMSE_REPORT.md) |
| 过滤后 FULL-data 的神经与结构迁移 | 校正后的 HIER Center/Width 保留为结构基线；没有普适神经迁移优势 | [语义修复](../studies/transfer/hier_cw_semantic_repair/FINAL_REPORT.md) |
| 收敛后的 P0 / P1 分阶段适配 | `NO_UNIVERSAL_STAGED_TRANSFER_GAIN`；P1 不提升 | [N1 报告](../studies/transfer/traditional_transfer_converged_baseline_v1/FINAL_REPORT.md) |
| Loss 与 adaptive readout | 保留 raw loss；R1/R2 未过双柱门槛，R3/R4 与 outer 测试未执行 | [Readout 报告](../studies/transfer/conditioned_source_readout/FINAL_REPORT.md) |
| Column-conditioned multitask / FiLM | inner 有信号，但 outer 未形成稳健跨协议收益 | [Multitask 报告](../studies/transfer/column_conditioned_multitask/FINAL_REPORT.md) |
| PCGrad | 降低梯度冲突，但未实质改善迁移；没有继续 outer 评估 | [PCGrad 报告](../studies/transfer/column_conditioned_pcgrad/FINAL_REPORT.md) |

**最新Row-CV评估已完成**：[Phase3B报告](../studies/transfer/column_conditioned_row_aligned_v1/PHASE3B_REPORT.md)给出五seed结果与判断C：inner收益没有稳定转化为outer跨柱收益，25g局部改善、40g仍有差距。Phase2原README与Phase3A盲预测保留为冻结历史协议。该研究不能继续描述为单seed试点，也不支持宣称outer退化已解决。

过滤后Row的B100/B200/B300/FULL曲线保留为独立[开发证据](../studies/transfer/filtered_row_learning_curve_b200_20261013/README.md)；不得与无阈值低标签基准混合排名。

**迁移下一步优先补独立数据与实验设计**：compound/batch 验证、长尾覆盖、相同条件重复、mass×flow 交叉条件。已有结果不支持自动追加模型、优化器或特征扫描，也不启动主动迁移。

完整的已测方法范围和保留假设见[迁移总结](research/CROSS_COLUMN_TRANSFER_STATUS.md)。

## 解释边界

- 4g 主动学习与跨柱主动迁移是两件事；前者正在研究，后者仍延后。
- Row、target-compound holdout、source-unseen OOD 不能互换；target-compound holdout 不等于源模型没见过该分子。
- 低标签无阈值、过滤后 FULL-data、旧论文复现的总体与尺度不同，不能混合排名。
- 复用已暴露测试集的结果是开发性证据。不得按测试结果改冻结规则或追加候选。
- 历史报告中的“下一步”记录当时决策，不覆盖本页。新实验只更新本页和该研究目录，不再追加一套项目级计划。

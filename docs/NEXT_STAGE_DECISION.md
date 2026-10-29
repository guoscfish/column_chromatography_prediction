# 当前研究状态

更新：2026-10-29。本页是项目级结论与下一步的唯一入口；具体数值、冻结门槛和使用过的数据以链接的实验报告及协议为准。

## 本轮执行

五seed Free LLM32 / LCMD32延长实验已完成，每条达到1485标签。以[最终自动生成报告](../reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md)为唯一数值入口；[最终核验](../reports/free_lcmd_five_seeds_to1485_20261026/final_audit.json)覆盖全部十条的37拟合、36批次及历史授权例外。监控已在核验完成后删除。

- Free与LCMD均已完成五seed的同预算比较；端点误差和全程学习曲线面积分别解释，不在导航页重复抄写数值。
- LCMD使用原冻结梯度特征实现与corrected LCMD-TP，保留同初始化及validation划分；Free旧157/6101保留Astra/high前缀，887/12203有显式限定的逻辑纠正amendment。全部解释与证据链接在当前报告中。
- 3330标签全训练集参考与后段平台期比较见[validation参考分析](../reports/free_lcmd_five_seeds_to1485_20261026/analysis/full_reference_validation_comparison_20261028.json)。未新增test评价，不能把validation平台期解释成无剩余泛化空间。
- 双专家seed157已完成1005，seed6101到525后连接故障停止；不能称为两seed1005完成。[核验记录](../reports/dual_expert_two_seeds_to1005_20261021/verified_final_status.json)。该停止记录不影响独立的Free / LCMD续跑。
- CW七臂×两seed1005扩展完成，Whitening按要求停525；[完整结果](../studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/RESULTS.md)保留原开发门槛与探索性解释。

[2026-10-13两seed快照](../reports/active_learning_latest/REPORT.md)和各阶段报告均为日期固定的历史记录，不作为当前进度。

## 预测器

**QGeoGNN-V2 已完成 4g 点预测资格验证**，状态为 `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES（历史判定码，保留原标识）`。

- 458,952 个参数均有梯度；与 R2-pruned 在 4,163 行上的六输出完全相同。
- 三个固定种子的 row / compound 六次验证完成。Row 插值和新 target-compound 泛化是不同任务。
- Clean 是 `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT`，仅作历史负结果。
- 当前分位数头可用于点预测，但覆盖率和 crossing 不足以支持已合格的 UQ；主动迁移仍需独立验证。

证据：[工程验证](../studies/predictor/final_v2_engineering/README.md)、[4g 正式报告](../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md)、[分位数审计](../studies/predictor/final_4g_qualification/README.md#记录-3)。

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

[Phase 2 实施方案](research/主动学习方法沿革.md#记录-7) 是后续设计。全部已完成证据见[主动学习索引](../studies/active_learning/README.md)。

## 迁移学习已归档

自 2026-10-28 起，迁移学习保存在 `codex/archive-transfer-20261028` 并停止维护，main 不再安排迁移或主动迁移的后续实验。原有结果、限制与判断见归档分支的迁移索引；[阅读和复现方式](README.md#迁移学习归档)及[分离清单](repository/RESEARCH_SCOPE.json)保留恢复信息。

## 解释边界

- main 当前研究 4g 主动学习与 LLM 选样；跨柱迁移已冻结归档。
- Row、target-compound holdout、source-unseen OOD 不能互换；target-compound holdout 不等于源模型没见过该分子。
- 低标签无阈值、过滤后 FULL-data、旧论文复现的总体与尺度不同，不能混合排名。
- 复用已暴露测试集的结果是开发性证据。不得按测试结果改冻结规则或追加候选。
- 历史报告中的“下一步”记录当时决策，不覆盖本页。新实验只更新本页和该研究目录，不再追加一套项目级计划。

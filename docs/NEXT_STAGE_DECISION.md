# 当前研究状态

更新：2026-10-11。本页是项目级结论与下一步的唯一入口；具体数值、冻结门槛和使用过的数据以链接的实验报告及协议为准。

## 本轮执行

当前结果统一见[最新 validation 快照](../reports/active_learning_latest/REPORT.md)。快照仅包含已完成拟合，正在选样或训练的轮次不写成已完成。各研究的冻结协议、授权修订和成本记录仍保存在原目录。

- **CW扩展已完成**：七臂×两seed到1005，Whitening按要求停525；新增60次拟合，复用150个历史对照点。validation AULC525–1005平均相对改善：CW-LCMD +1.21%、CW-MaxDet -3.80%、CW-IVR +5.98%。此前blocked、待恢复及90拟合额度均为历史状态。[完整结果](../studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/RESULTS.md)。
- **CW16+LLM16已完成两seed到813**；Free的seed157/6101分别已完成525/653。各曲线按实际完成预算截断，授权修复偏离保留在原研究回执中。[Scientist V2](../studies/active_learning/qgeognn_v2_row_llm_scientist_v2/README.md)。
- **双专家seed157已完成461标签**，端点validation NRMSE为0.687444；当前按既有授权继续至493。333→397使用astra/high，397后使用sol61/xhigh；格式和未观测证据引用修复保留独立revision与原始回答，不将这条混合模型轨迹解释成单一模型的独立确认。[双专家记录](../studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/README.md)。

CW最初333–525开发结论不变：CW相对Raw在LCMD的AULC改善6.93%、MaxDet恶化4.86%、IVR改善0.69%；两种新selector均未过两seed同向且平均改善≥2%的确认门槛。1005扩展属于探索性证据，P2付费独立消融与外部验证仍未执行。[原始factorial结果](../studies/active_learning/qgeognn_v2_row_cw_selector_factorial/RESULTS.md)。

## 预测器

**QGeoGNN-V2 已完成 4g 点预测资格验证**，状态为 `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`。

- 458,952 个参数均有梯度；与 R2-pruned 在 4,163 行上的六输出完全相同。
- 三个固定种子的 row / compound 六次验证完成。Row 插值和新 target-compound 泛化是不同任务。
- Clean 是 `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT`，仅作历史负结果。
- 当前分位数头可用于点预测，但覆盖率和 crossing 不足以支持已合格的 UQ；主动迁移仍需独立验证。

证据：[工程验证](../studies/predictor/final_v2_engineering/README.md)、[4g 正式报告](../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md)、[分位数审计](../studies/predictor/final_4g_qualification/QUANTILE_AUDIT.md)。

## 4g 主动学习

**双专家历史两轮试跑（333→397）**：[seed157试跑](../studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/README.md)完成333→397、64条新标签和2次同初始化训练。验证集端点NRMSE为0.7622，优于free的0.7861，仍弱于CW16+LLM16的0.7046；两轮AULC未优于free，尚无整体优势证据。历史方案和32条实测反馈已进入第二轮；72项测试通过，无新test评价。首轮格式兼容修复保留原始32个ID，额外格式调用计入成本。

| 已完成工作 | 结论 |
| --- | --- |
| LCMD one-step pilot | 大批量一次选点有正信号；不等于完整曲线或 compound 鲁棒性 |
| B32 / B16 benchmark、Hybrid extension | B32 LCMD 有正信号；不同新增预算不能单独识别 batch-size effect |
| Sequential B32，333→1005 标签 | LCMD / Hybrid 平均 AULC 比 Random 低约 23%，均 5/5 种子获胜 |
| Kernel-IVR extension | `NO_CLEAR_IVR_IMPROVEMENT`；AULC 比 LCMD 差约 4.28%，仅 1/5 获胜 |
| IVR mechanism audit | `STOP_BEFORE_GATE_C`；稳定性门槛未过，没有新增训练或效果验证 |
| Phase 0 efficiency review | 重新整理端点、AULC、N80/N90/N95 与计算成本，没有重新训练 |

Sequential 的原始分类 `NO_CLEAR_SEQUENTIAL_AL_GAIN` 保留不改：它未确定 LCMD 与 Hybrid 的最佳/相近类别，不能解释成“对 Random 无收益”。约 650 标签后改善放缓，稳定达到 N90/N95 仍未解决。

[Phase 1固定预算Static / Adaptive对照](../studies/active_learning/qgeognn_v2_batch_adaptivity/FINAL_REPORT.md)已完成：均333+320=653标签；Adaptive端点0.436791优于Static的0.451013，但AULC未改善，不能宣称完整曲线优势。当前执行项为上面的双专家扩展。

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

**迁移下一步优先补独立数据与实验设计**：compound/batch 验证、长尾覆盖、相同条件重复、mass×flow 交叉条件。已有结果不支持自动追加模型、优化器或特征扫描，也不启动主动迁移。

完整的已测方法范围和保留假设见[迁移总结](research/CROSS_COLUMN_TRANSFER_STATUS.md)。

## 解释边界

- 4g 主动学习与跨柱主动迁移是两件事；前者正在研究，后者仍延后。
- Row、target-compound holdout、source-unseen OOD 不能互换；target-compound holdout 不等于源模型没见过该分子。
- 低标签无阈值、过滤后 FULL-data、旧论文复现的总体与尺度不同，不能混合排名。
- 复用已暴露测试集的结果是开发性证据。不得按测试结果改冻结规则或追加候选。
- 历史报告中的“下一步”记录当时决策，不覆盖本页。新实验只更新本页和该研究目录，不再追加一套项目级计划。

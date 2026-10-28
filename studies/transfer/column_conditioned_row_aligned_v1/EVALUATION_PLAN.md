# Phase 3B 评价计划（Phase 3A 评分前锁定）

日期：2026-10-14，Asia/Shanghai。本文件仅规定下一阶段的评价，Phase 3A 不执行评分。历史 Row 外层结果及 Phase 1/2 内部结果已曝光；未来结果属于开发性评价，不能称为独立外部确认。外层 seed 共享大量记录，不视为五次独立物理实验，不做独立样本显著性检验。

## 进入条件与评价人口

必须先验证 PREDICTION_FREEZE_MANIFEST.json、全部十个最终 checkpoint、四十个预测文件和训练身份哈希。任何不一致立即停止；冻结后禁止改写模型、预测、epoch 或本计划。需要用户另行授权 Phase 3B 后才允许解析外层响应。

主评价为原固定 Row **test**；25g/40g 分开报告全部五个 seed，不挑选 seed 或 arm。原外层 validation 可以在同一次获准评价中作为独立标记的次要描述表，不能与 test 合并，也不能用于重新选择模型或训练预算。所有 A1/A2 对照必须具有相同柱、seed、role、sample_id 顺序和获准 FULL gradient_train 人口。

## 指标与聚合

点预测固定为 V1_q50、V2_q50，列顺序为 V1_q10/q50/q90、V2_q10/q50/q90。每个 seed/柱的归一化尺度为该 seed **全部 gradient_train** 的 V1/V2 标签标准差，ddof=0，使用 Phase 3A 已保存尺度；不能使用 test 标准差或不同方法的自定义尺度。

Combined NRMSE = sqrt(((RMSE_V1/SD_train_V1)^2 + (RMSE_V2/SD_train_V2)^2)/2)。主要表逐 seed 列 A1/A2，再按柱计算五 seed 指标的算术均值与描述性 SD(ddof=1)。固定报告两种相对改善，不能混用：

- mean paired-seed gain = mean_seed(1 − NRMSE_A2_seed / NRMSE_A1_seed)。
- gain of means = 1 − mean_seed(NRMSE_A2_seed) / mean_seed(NRMSE_A1_seed)。
- Seed wins = A2 NRMSE 严格低于 A1 的 seed 数/5，并列单独标记。

同时报告 V1/V2 RMSE、MAE；Center=(V1+V2)/2、Width=V2−V1 的 RMSE；V1/V2/Center/Width 最大绝对误差、对应身份及两方法变化。每 seed 的最大误差先在本 seed 中计算，再跨 seed 描述，不把重复身份合并成独立样本。可另列最坏 seed，但不能据此删掉它。

高保留体积 tail 固定为真实端点 ≥ 该 seed/柱全部 gradient_train 对应端点的 80% 分位（np.quantile 默认 linear）；V1/V2 分别构建掩码，两个 arm 共享同一掩码。报告阈值、tail count、SSE、RMSE、逐 seed 相对改善和五 seed tail RMSE 均值；空 tail 记 NA，不补零、不改阈值。任何 pooled 诊断都必须注明重复 seed 记录及其非独立性。

## 三个不同问题

1. **FiLM 相对 A1**：主要分析 A2/A1 的完整 Row-CV 选择 + FULL refit 流程。两个 arm 使用各自五折中位数预算，不是 equal-step 的 FiLM 模块净效应。两柱分别判断跨 seed 一致性，不能用整体均值掩盖某柱或某 seed 退化。
2. **多任务相对简单校准**：只引用已冻结且满足下述兼容性核验的历史预测，不训练新候选，不用历史评分表直接比较。单柱校准与多任务的 source_train 辅助监督不同，是完整工作流对照，不能解释为单一架构的因果效应。
3. **均值与尾部/最大误差代价**：全部端点、Center/Width、tail、最大误差均保留。明确标记“平均 NRMSE 降低但 tail 或最大误差增加”的组合；不引入事后通过门槛。

Phase 2 的负结果须并列保留：25g V1 tail 在 3/5 seed 退化，五 seed tail RMSE 均值增加约 1.98%；40g V2 MAE 略增；pilot A2 一折 best_epoch=500 属于收敛边界。不得利用新外层结果调整 epoch、BN 或 FiLM。

## 历史基线兼容性门槛

后续引用前须逐柱/seed核验原协议、过滤人口、Row split SHA256、source checkpoint SHA256、全部 gradient_train 身份/数量、test 身份/数量、预测冻结 receipt 与文件 SHA256，以及特征/标签边界。重新计算指标必须使用本研究相同训练尺度、tail 阈值和 test 身份。只有全五 seed 同尺度 FULL Row 的预测可以进入正式对照表。B=100、未过滤人口、不同外层身份、不同 source checkpoint 或未冻结预测均排除。缺少证据记“未确认兼容”，不猜测或补训练。

- `filtered_full_data_benchmark` 的 **scale_only**：训练仅用 gradient_train，原 Row FULL 的 split/source 与本研究相同，是简单校准候选；需核验十个 frozen.json、全局 all_predictions_frozen.json 及预测身份后引用。
- Affine/其他校准：目前不预设兼容；必须找到确切冻结产物并逐项通过上述门槛后才引用。
- `hier_cw_semantic_repair` 的 **HIER_CW_SHARED_LAMBDA_CORRECTED**：可核验其 FULL Row、source、过滤人口、训练身份、内层选择和冻结 receipts；Group-CV 与 Row-CV 选择差异必须明确披露，作为不同完整工作流参考，不能充当 FiLM 模块对照。
- `traditional_transfer_converged_baseline_v1` 的 P0/P1：FULL Row 固定预算、相同 split/source、内部 Group-CV 选择，可在全部冻结证据通过后作为参数迁移工作流参考；不把内层 CV 差异归因于多任务结构。
- `filtered_full_data_benchmark` 的 **paper_style_current_v2**（及 conditional_EA、target_only_full）原选择使用 outer validation 标签，与当前完全内层选择的权限不同。此选择协议不匹配，不进入主要同协议对照；如后续单独讨论历史工作流，必须显式标记选择集差异，不能直接宣称方法收益。

上述是引用规划，不是 Phase 3A 已完成外层评价。先验证冻结并得到 Phase 3B 授权，再进行一次预定评价，报告全部正负结果后停止。

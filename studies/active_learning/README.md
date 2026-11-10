# 4g 主动学习与 LLM 研究索引

本页区分独立 V2 的冻结结果、历史探索和 main 上的正式运行。候选分支 `codex/repository-architecture-20261109` 不承载真实实验运行；其 `scripts/run_al.py` 只开放合成生命周期。旧报告中的命令不代表候选可以执行它们。

状态统一含义：`CURRENT` 为本候选的主要阅读入口；`FROZEN_HISTORICAL` 为保留原身份的科学证据；`ARCHIVED_SOURCE` 为从固定 Git 来源恢复的历史内容；`SUPERSEDED` 为已被后续阶段取代的方案或快照；`ACTIVE_RUNTIME` 为 main 上受保护的运行项目，不能在候选执行或移动。状态不是新训练/揭示标签的授权。

## 先读这些

| 研究入口 | 状态 | 解释边界 |
| --- | --- | --- |
| [main 的 CW-visible Free32 五 seed](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101) | ACTIVE_RUNTIME | seeds157/887/2357/6101/12203，从 L333 目标 L1485；保持冻结模型、loss、prompt、选择和恢复链，不根据快照宣布完成 |
| [main 固定 CW 概览：2026-11-09 20:06](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/reports/free_cw_overview_20261109_2005) | FROZEN_HISTORICAL | 仅是该时间快照，共同已完成预算 L973；不是当前实时状态或 L1485 最终报告 |
| [五 seed Free/LCMD L1485 完整报告](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md) | FROZEN_HISTORICAL | 10/10 轨迹完成；端点和全程 AULC 方向不同，历史 Free 混合模型轨迹不构成 CW 或模型单因素因果对照 |
| [连续 B32 最终报告](https://github.com/guoscfish/column_chromatography_prediction/blob/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_sequential_b32/FINAL_REPORT.md) | FROZEN_HISTORICAL | 五 seed 数值策略基准；保留原自动门槛与后续解释，不把 NO_CLEAR_SEQUENTIAL_AL_GAIN 误读为相对 Random 无收益 |
| [小批量基准](https://github.com/guoscfish/column_chromatography_prediction/blob/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_small_batch_benchmark/FINAL_REPORT.md)、[效率复核](https://github.com/guoscfish/column_chromatography_prediction/blob/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_efficiency_review/REPORT.md) | FROZEN_HISTORICAL | 标签预算、批大小、端点、AULC 和计算成本分别报告 |
| [CW 多阶段身份索引](CW_RESEARCH_INDEX.md) | CURRENT（导航） | 纯 CW 连续延长与 selector factorial 是不同协议；原 gate、Whitening 停止修订及探索解释均保留 |
| [NoCW L1389 目标阶段快照](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/nocw_progress_to1389_20261101/REPORT.md) | FROZEN_HISTORICAL | 标题中的目标不等于已完成预算；只用各方法共同已完成点比较 |
| [main 的损失/分位数离线复盘](https://github.com/guoscfish/column_chromatography_prediction/tree/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md) | CURRENT（科学诊断） | 五 seed L333；不是权重因果实验，未证明宽度或交叉标志有 AL AULC 增益 |

现代报告以固定 validation 指标和精确协议身份解释；已有 test 使用与确认性边界继续见 [数据消费登记](../../docs/data_consumption_register.json)。两 seed 开发子集、不同预算端点、mixed-model 历史轨迹以及 row/compound 不能拼接成统一排名。

## 44 个研究目录的身份索引

按清理前 `0e33a86d3dea` 的完整 Git 树逐目录核对。链接固定到该来源，因此即使某个非权威副本退出候选，仍可检索原文；它不保证恢复本地 ignored checkpoint、请求或原始 runtime。已有原始协议、选择序列和回执不在本索引中合并或改写。

| 研究目录 / 固定来源 | 状态 | 角色与限定 |
| --- | --- | --- |
| [4g](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/4g) | FROZEN_HISTORICAL | 旧 predictor 的早期 AL，不能和独立 V2 混排 |
| [free_cw_sol61_xhigh_five_to1485_20261101](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101) | ACTIVE_RUNTIME | main 上 Sol/xhigh 五 seed，从 L333 目标 L1485；原协议/选择/回执/冻结源码严格保留 |
| [free_lcmd_mechanism_diagnostics_20261029](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/free_lcmd_mechanism_diagnostics_20261029) | FROZEN_HISTORICAL | 已完成 Free/LCMD 五 seed 轨迹的离线机制分析；不新增训练 |
| [gp_guided_llm_stability_20261103](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/gp_guided_llm_stability_20261103) | FROZEN_HISTORICAL | GP 引导候选与稳定性探索；部分/否定证据不能删去 |
| [hybrid_loading_cap_audit_20261029](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/hybrid_loading_cap_audit_20261029) | FROZEN_HISTORICAL | loading-cap 离线审计；阶段 B 因反证门槛暂停 |
| [nocw_free_selection_diagnostics_20261101](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/nocw_free_selection_diagnostics_20261101) | FROZEN_HISTORICAL | NoCW/Free 选点与协议审计；非独立效果对照 |
| [qgeognn_v2_batch_adaptivity](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_batch_adaptivity) | FROZEN_HISTORICAL | 相同 L653 预算；端点与 AULC 结论必须分开 |
| [qgeognn_v2_efficiency_review](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_efficiency_review) | FROZEN_HISTORICAL | 端点、AULC、目标精度标签数和计算成本的已保存结果复核 |
| [qgeognn_v2_ivr_mechanism_audit](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_ivr_mechanism_audit) | FROZEN_HISTORICAL | Gate C 前停止的 IVR 机制审计，不宣称新增效果验证 |
| [qgeognn_v2_row_center_width_performance](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_center_width_performance) | FROZEN_HISTORICAL | CW 性能和机制研究；与开发/延长预算分别解释 |
| [qgeognn_v2_row_cw_ivr_portfolio_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_ivr_portfolio_b32) | FROZEN_HISTORICAL | 两 seed、L333–653 的组合开发实验 |
| [qgeognn_v2_row_cw_lcmd_to_1005](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005) | FROZEN_HISTORICAL | 纯 CW-LCMD 两 seed L653→1005 延长，原协议身份独立 |
| [qgeognn_v2_row_cw_lcmd_to_525](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_lcmd_to_525) | FROZEN_HISTORICAL | 纯 CW-LCMD 从短程 L429 状态延长到 L525 |
| [qgeognn_v2_row_cw_lcmd_to_653](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_lcmd_to_653) | FROZEN_HISTORICAL | 纯 CW-LCMD L525→653 延长 |
| [qgeognn_v2_row_cw_selector_extension_to_1005](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005) | FROZEN_HISTORICAL | 开发后探索扩展，Whitening 按修订停在 L525 |
| [qgeognn_v2_row_cw_selector_extension_to_653](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_653) | FROZEN_HISTORICAL | 开发后探索中间阶段；不是独立确认实验 |
| [qgeognn_v2_row_cw_selector_factorial](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_selector_factorial) | FROZEN_HISTORICAL | L333–525 开发门槛；新 selector 未通过原确认 gate |
| [qgeognn_v2_row_fused_maxdet_a080_b32_screen](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_fused_maxdet_a080_b32_screen) | FROZEN_HISTORICAL | 固定 alpha 融合表示的探索筛选；保留各自结论 |
| [qgeognn_v2_row_fused_maxdet_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_fused_maxdet_b32) | FROZEN_HISTORICAL | fused/MaxDet 协议及数值证据；不与 alpha 筛选覆盖 |
| [qgeognn_v2_row_hybrid_extension](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_hybrid_extension) | FROZEN_HISTORICAL | 匹配的一步 Hybrid 延长对照 |
| [qgeognn_v2_row_innovation_screen](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_innovation_screen) | FROZEN_HISTORICAL | 固定几何/选样机制筛选，不等于重训练效果 |
| [qgeognn_v2_row_kernel_ivr_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_kernel_ivr_b32) | FROZEN_HISTORICAL | Kernel-IVR 探索性扩展，未建立优于 LCMD 的明确收益 |
| [qgeognn_v2_row_lcmd](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_lcmd) | FROZEN_HISTORICAL | LCMD 大批量试点，后续连续曲线另见 sequential |
| [qgeognn_v2_row_lcmd_to1485_20261026](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_lcmd_to1485_20261026) | FROZEN_HISTORICAL | 五 seed 长程 LCMD 及其冻结选择/回执；与 Free 的完整比较保留 |
| [qgeognn_v2_row_lcmd_to_ivr_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_lcmd_to_ivr_b32) | FROZEN_HISTORICAL | LCMD→IVR 分阶段开发协议与原修订 |
| [qgeognn_v2_row_llm16_screen](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm16_screen) | FROZEN_HISTORICAL | 早期 LLM16 筛选与门槛，不是现代五 seed Free |
| [qgeognn_v2_row_llm_dialog_feedback_v1](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_dialog_feedback_v1) | FROZEN_HISTORICAL | 历史对话反馈协议，保留不同 prompt/schema 身份 |
| [qgeognn_v2_row_llm_dual_expert_v1](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1) | FROZEN_HISTORICAL | 双专家历史轨迹及模型切换/中止证据 |
| [qgeognn_v2_row_llm_full_pool_feedback_v1](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_full_pool_feedback_v1) | FROZEN_HISTORICAL | full-pool 反馈历史协议，不能与 Free 混同 |
| [qgeognn_v2_row_llm_nocw_ablation](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_nocw_ablation) | FROZEN_HISTORICAL | NoCW 协议、修订与冻结链；阶段快照不是终点结果 |
| [qgeognn_v2_row_llm_scientist_v2](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_scientist_v2) | FROZEN_HISTORICAL | 原 Free/Scientist 冻结协议与跨模型来源链 |
| [qgeognn_v2_row_llm_scientist_v3](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_scientist_v3) | FROZEN_HISTORICAL | 历史 V3 协议与响应，不能以通用 adapter 冒充兼容 |
| [qgeognn_v2_row_llm_scientist_v3_1](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_scientist_v3_1) | FROZEN_HISTORICAL | 历史 V3.1 修订，保持独立来源 |
| [qgeognn_v2_row_llm_scientist_v3_2](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_llm_scientist_v3_2) | FROZEN_HISTORICAL | 历史 V3.2 运行、恢复与最终报告身份 |
| [qgeognn_v2_row_master_report](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_master_report) | FROZEN_HISTORICAL | 历史汇总，不替代后续固定预算和协议来源 |
| [qgeognn_v2_row_maxdet_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_maxdet_b32) | FROZEN_HISTORICAL | MaxDet B32 比较证据及科学边界 |
| [qgeognn_v2_row_sequential_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_sequential_b32) | FROZEN_HISTORICAL | 五 seed Random/LCMD/Hybrid 连续基准，保留原门槛文字 |
| [qgeognn_v2_row_short_sequential_b32](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_short_sequential_b32) | FROZEN_HISTORICAL | 短程与 CW 原始前缀；后续扩展从确切状态衔接 |
| [qgeognn_v2_row_small_batch_benchmark](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_small_batch_benchmark) | FROZEN_HISTORICAL | B32/B16 的预算与批量比较 |
| [qgeognn_v2_same_state_branching](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_same_state_branching) | FROZEN_HISTORICAL | 同状态数值方法分叉，只有限定控制内的因果解释 |
| [qgeognn_v2_same_state_branching_cw_hybrid](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_same_state_branching_cw_hybrid) | FROZEN_HISTORICAL | CW/Hybrid 同状态分叉，保留实现审计 |
| [representation_gp_benchmark_20261104](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/representation_gp_benchmark_20261104) | FROZEN_HISTORICAL | 表示与 GP/校准基准，部分证据不代表正式 AL 增益 |
| [scientist_v3_responses_2026-09-29](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/scientist_v3_responses_2026-09-29) | FROZEN_HISTORICAL | 原始响应快照；保留不可逆输入与引用身份 |
| [verifiable_scientific_agent_20261030](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/verifiable_scientific_agent_20261030) | FROZEN_HISTORICAL | Agent 合法性、失败分类与机制审计；负结果保留 |

历史报告中的“下一步”“正在运行”和旧 PID 只描述采集当时。日常科学状态见 [NEXT_STAGE_DECISION](../../docs/NEXT_STAGE_DECISION.md)，报告分级见 [reports 索引](../../reports/README.md)，候选运行与恢复方式见 [架构](../../docs/ARCHITECTURE.md) 和 [tools](../../tools/README.md)。历史运行器依固定 Git 来源恢复，不再链接已退出的 `scripts/README.md`。

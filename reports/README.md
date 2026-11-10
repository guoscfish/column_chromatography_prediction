# 报告身份与权威入口

本页取代滚动命名的“latest”入口，按固定日期、预算、协议和状态检索报告。以下覆盖清理前 `0e33a86d3dea` Git 树的 29 个报告目录；链接固定到原提交，允许非权威副本经逐文件门槛退出候选后仍可查证。原文、指标与冻结身份不被本索引覆盖。

状态统一含义：`CURRENT` 为本候选的主要阅读入口；`FROZEN_HISTORICAL` 为保留原身份的科学证据；`ARCHIVED_SOURCE` 为从固定 Git 来源恢复的历史内容；`SUPERSEDED` 为已被后续阶段取代的方案或快照；`ACTIVE_RUNTIME` 为 main 上受保护的运行项目，不能在候选执行或移动。状态不是新训练/揭示标签的授权。

main 的 [CW 五 seed 概览（2026-11-09 20:06）](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/reports/free_cw_overview_20261109_2005) 是单独的固定进度来源，不在此候选基线中。它的公共已完成预算为 L973，不能称当前实时结果或 L1485 最终比较。当前受保护研究入口为 [main 的 CW-Free 来源](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101)；本轮不移动、续跑或停止它。

## current comparison

| 固定来源 | 状态 | 权威范围 / 使用限定 |
| --- | --- | --- |
| [free_lcmd_five_seeds_to1485_20261026](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_lcmd_five_seeds_to1485_20261026) | FROZEN_HISTORICAL | 五 seed Free/普通梯度 LCMD 完整 L1485；端点与 AULC 分开解释；Free 历史模型混合 |
| [five_methods_cw_gradient_20261103](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/five_methods_cw_gradient_20261103) | FROZEN_HISTORICAL | 两共享 seed，五方法共同 L333–1005；不同终点不能直接排名，LCMD 为 CW-gradient |
## historical finalized

| 固定来源 | 状态 | 权威范围 / 使用限定 |
| --- | --- | --- |
| [free_llm_five_seeds_to1485_20261024](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_llm_five_seeds_to1485_20261024) | FROZEN_HISTORICAL | 历史 Free 延长产物、控制器与恢复链；权威人读结论看上方 Free/LCMD 报告 |
| [free_llm_five_seeds_to1005_20261023](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_llm_five_seeds_to1005_20261023) | FROZEN_HISTORICAL | Free L1005 阶段来源链；后续延长不抹除原身份 |
| [dual_expert_two_seeds_to1005_20261021](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/dual_expert_two_seeds_to1005_20261021) | FROZEN_HISTORICAL | 最终核验：157 至1005；6101 至525后连接停止；目录目标不等于两 seed 同终点 |
| [dual_expert_two_seeds_to813_20261014](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/dual_expert_two_seeds_to813_20261014) | SUPERSEDED | 双专家早期里程碑，后续来源见1005阶段核验 |
| [free_rerun_audit_20261103](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_rerun_audit_20261103) | FROZEN_HISTORICAL | 两次 Free 轨迹/记忆/模型混杂审计；不能据重跑差异孤立 CW 或模型因果 |
| [free_first_vs_second_20261103](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_first_vs_second_20261103) | FROZEN_HISTORICAL | 两次 Free 固定进度比较，保留 mixed-model 与共同预算 |
| [free_first_vs_second_20261106_1434](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_first_vs_second_20261106_1434) | FROZEN_HISTORICAL | 较后固定比较；不是当前运行或完整单因素消融 |
| [llm_free_complete_20261013](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_free_complete_20261013) | FROZEN_HISTORICAL | 早期 Free 终点快照；协议、seed 数与现代五 seed 分开 |
| [llm_hybrid_complete_20261010](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_hybrid_complete_20261010) | FROZEN_HISTORICAL | 早期 LLM hybrid 终点快照；不同于数值 Hybrid |
| [llm_model_switch_20261012](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_model_switch_20261012) | FROZEN_HISTORICAL | 历史模型/推理强度切换审计，保留来源链与混杂解释 |
## progress / transport

| 固定来源 | 状态 | 权威范围 / 使用限定 |
| --- | --- | --- |
| [free_cw_progress_20261102_1633](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_cw_progress_20261102_1633) | SUPERSEDED | CW-Free 固定进度：当时41 completed fits；原运行谱系保留 |
| [free_cw_progress_20261103](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/free_cw_progress_20261103) | SUPERSEDED | CW-Free 固定进度：当时84 completed fits；非 L1485 final |
| [cw_retry_evidence_20261106_1445](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/cw_retry_evidence_20261106_1445) | FROZEN_HISTORICAL | 运行期间保护的原始回执无损归档；成功 transport 不等于 accepted batch |
| [nocw_progress_20261030](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/nocw_progress_20261030) | SUPERSEDED | 目标L1005时的阶段快照，原五 seed 未全部完成 |
| [nocw_progress_to1389_20261101](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/nocw_progress_to1389_20261101) | FROZEN_HISTORICAL | 目标L1389的阶段快照，五 seed 公共完成预算 L973；非final |
| [llm_eight_to1165_20261014](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_eight_to1165_20261014) | SUPERSEDED | 八策略阶段目标1165；按各实验完成回执解释 |
| [llm_eight_to1293_20261015](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_eight_to1293_20261015) | SUPERSEDED | 八策略阶段目标1293；不按目录名推断同预算完成 |
| [llm_eight_to1421_20261017](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_eight_to1421_20261017) | SUPERSEDED | 八策略阶段目标1421；保留独立阶段来源 |
| [llm_eight_to1549_20261018](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_eight_to1549_20261018) | FROZEN_HISTORICAL | 阶段报告兼 controller/恢复依赖；live_status 不作为新候选当前权威 |
| [llm_four_to1005_20261013](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_four_to1005_20261013) | SUPERSEDED | 早期四策略L1005目标与进度，现代五 seed 不混排 |
| [llm_all_strategies_20261010](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_all_strategies_20261010) | SUPERSEDED | 早期方法比较快照，按原模型和预算范围解释 |
| [latest_five_strategy_comparison_20261018](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/latest_five_strategy_comparison_20261018) | SUPERSEDED | 固定10月18日比较；名称latest没有动态权威含义 |
| [active_learning_latest](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/active_learning_latest) | SUPERSEDED | 固定10月13日快照；废止 latest 作为当前权威名称 |
| [active_learning_status_20261005](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/active_learning_status_20261005) | SUPERSEDED | 固定10月5日研究状态，不是当前计划 |
| [llm_progress_20261006](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_progress_20261006) | SUPERSEDED | 早期工程/研究进度，按原提交检索 |
| [llm_progress_20261008](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/llm_progress_20261008) | SUPERSEDED | 固定10月8日验证、quota和暂停状态，非当前进程状态 |
## engineering audit

| 固定来源 | 状态 | 权威范围 / 使用限定 |
| --- | --- | --- |
| [repository_cleanup_20261106](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/reports/repository_cleanup_20261106) | FROZEN_HISTORICAL | 清理及回执归档收据；不是科学效果报告 |

## 使用与恢复边界

同预算的科学比较应明确 outer seeds、模型/初始化、split、预算网格与 validation/test 角色；不把更晚目录、更多终点或更好曲线当成相同实验条件。原始 accepted batch 与 completed fit 分开核验；运输成功、heartbeat 和 controller 状态不能计为科学结果。

上述若干目录同时保存控制器、恢复授权、日志和人读报告。目录级分类不能代替逐文件依赖审计，也不是整目录删除授权。原始数据、checkpoint、选择/标签账本、冻结来源、授权、协议和 test 消费声明继续受保护。Git 固定提交只恢复已跟踪字节，本地 ignored runtime/原始请求必须依独立资源来源恢复。

本轮已用固定聚合输入及固定绘图源码在独立目录重建可退役的旧阶段曲线，原始曲线、协议、偏差说明和运行恢复证据分别保留；生成图不是新实验结果。实际退出数量和每文件 SHA/blob/recovery 来源见 [清理报告](../docs/repository/releases/redundancy-cleanup-20261110/CLEANUP_REPORT.md) 与 [工程审计导航](../docs/repository/README.md)。更多科学证据入口见 [studies](../studies/README.md) 与 [experiments](../experiments/INDEX.md)。

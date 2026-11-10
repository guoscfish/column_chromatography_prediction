# CW 研究身份与延长关系

状态：`CURRENT`（读者导航），原研究均按各自冻结来源解释。此页不合并协议、不重新计算指标、不启动训练，也不授权改变正在运行的 CW-Free。

| 研究线 | 原始范围与关系 | 科学状态 / 入口 |
| --- | --- | --- |
| 纯 CW-LCMD 前缀 | 短程研究 L333→429 提供原状态 | PRIMARY_FROZEN：[短程报告](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_short_sequential_b32/FINAL_REPORT.md) |
| 纯 CW-LCMD →525 | 两 seed 从确切 L429 状态延长三批，不重跑 comparator | PRIMARY_FROZEN：[429→525](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_lcmd_to_525/FINAL_REPORT.md) |
| 纯 CW-LCMD →653 | 继承各自 L525 状态，仅延长 CW | PRIMARY_FROZEN：[525→653](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_lcmd_to_653/FINAL_REPORT.md) |
| 纯 CW-LCMD →1005 | 继承各自 L653，硬终点 L1005，无额外 comparator | PRIMARY_FROZEN：[653→1005](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005/FINAL_REPORT.md) |
| selector factorial | 两 seed 的 L333–525 开发窗口；新 selector 未通过原确认 gate | PRIMARY_FROZEN：[factorial 及机制审计](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_selector_factorial/README.md) |
| selector →653 / →1005 | 看过开发结果后的探索延长，不能反向改原 gate；按用户修订 Whitening 留在 L525，七臂×两 seed 达 L1005 | PRIMARY_FROZEN：[653 阶段](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_653/README.md)、[1005 阶段及修订](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/README.md) |
| CW/IVR 组合 | 两 seed L333–653；独立组合协议，不等于 factorial 的同名 selector | PRIMARY_FROZEN：[portfolio](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_row_cw_ivr_portfolio_b32/FINAL_REPORT.md) |
| CW/Hybrid 同状态分叉 | 固定状态的分叉比较，与完整 sequential 效应分别解释 | PRIMARY_FROZEN：[同状态分叉](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/active_learning/qgeognn_v2_same_state_branching_cw_hybrid/FINAL_REPORT.md) |
| CW-visible Free32 | L333 开始的五 seed LLM 重新选点轨迹；不同于以上数值 CW-LCMD，也不是历史 Free 的模型切换单因素实验 | ACTIVE_RUNTIME（main）：[受保护研究来源](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101) |

selector 延长的原 README 包含早期八臂计划及后来停止/恢复修订，应按后续明确修订解释，而不是删除旧计划使其看似从未变过。已完成的 60 次额外 fit 与一次中断尝试的计数差异见原资源核算记录；本页不据此批准重试。

CW 的输出度量、CountSketch 有限投影、当前 L 的尺度归一化和原 L0 变换是不同机制，不能从离线几何等价直接宣称全程 AL 效果因果。分位数交叉也不自动是可减少的 epistemic uncertainty；新诊断仅链接 [main 固定科学报告](https://github.com/guoscfish/column_chromatography_prediction/blob/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md)。

返回 [主动学习索引](README.md)。

本轮仅退出factorial已经由RUN_STATE/fit_ledger/DELIVERY_AUDIT完整覆盖的stdout日志；36次fit、拒绝确认gate的负结果、全部协议与冻结身份仍保留。原样来源可由 [本轮总报告](../../docs/repository/releases/redundancy-cleanup-20261110/CLEANUP_REPORT.md) 的逐文件清单恢复。

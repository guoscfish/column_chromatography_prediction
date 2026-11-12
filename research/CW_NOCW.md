# CW与NoCW：保留比较边界

这里并列来源，不把不同时间、模型前缀和协议拼成新的最终因果结果。共同outer seeds为157/887/2357/6101/12203，4g row、L333起步、batch32、QGeoGNN-V2；指标角色为validation。

## NoCW已公布阶段快照

固定于2026-11-01 14:51:30（北京时间），目标L1389尚未完成。各seed已完成预算分别为1069/973/1101/1133/1197，不能把不同终点误差直接平均。五seed共同已完成预算 **L973** 的原报告均值：

| 方法 | validation Combined NRMSE@共同L973 |
| --- | ---: |
| NoCW LLM32 | 0.411480 |
| 原Free LLM32 | 0.354404 |
| Hybrid32 | 0.354505 |
| LCMD32 | 0.369898 |
| Random32 | 0.500483 |

这是描述性历史对照，不是纯“有无CW”配对因果结论：NoCW全程gpt-6.1-sol/xhigh，原Free157/6101含Astra/high前缀，Prompt/修订/失败授权亦不同。原两seed CWAware-vs-NoCW配对协议最初只交付dry-run；随后执行的是另行授权的五seed NoCW及扩展，不能把两者合并成已完成双arm正式消融。

[阶段报告](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/reports/nocw_progress_to1389_20261101/REPORT.md) · [NoCW五seed修订](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029/README.md) · [原两arm设计](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/PROTOCOL.md)。已完成拟合、accepted batch、物理重试和token费用须分别解释；网络活动不能当进展。

## CW-visible Free五seed：main独立保护

main的Sol/xhigh rerun从L333目标L1485，每seed36个accepted batch；原协议、源码hash、初始化、validation隔离与scratch拟合保留。Framework V2不读取live预测/标签、不改调度或恢复、不判定它已完成。

- [main b1a6ed7a冻结研究入口](https://github.com/guoscfish/column_chromatography_prediction/blob/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/README.md)。
- [2026-11-09 20:06公开快照](https://github.com/guoscfish/column_chromatography_prediction/tree/b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28/reports/free_cw_overview_20261109_2005)：固定进度，五seed公共已完成L973，不是本日实时状态或L1485最终比较。
- [归档中的协议及修订谱系](https://github.com/guoscfish/column_chromatography_prediction/tree/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101)：只恢复tracked历史字节，私有runtime/完整请求/凭据另行保护。

未来CW/NoCW结论必须注明实际完成网格、同seed模型/Prompt差异与validation开发曝光；不能借框架精简或离线图表替代正式确认。

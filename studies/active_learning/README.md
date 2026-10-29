# 4g 主动学习与 LLM 选样

当前研究使用已资格验证的独立 QGeoGNN-V2，关注 4g 分布内的标签效率、LLM 反馈与选样机制。实时数值只在当前报告维护。

| 研究 | 状态与解释 |
| --- | --- |
| [当前五种子 Free / LCMD](../../reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md) | 目标 1485 标签；进度、共同预算曲线和全数据验证参考统一由本报告给出 |
| [LLM Scientist V2](qgeognn_v2_row_llm_scientist_v2/README.md) | 保留冻结协议；当前 Free 跨版本来源链见当前报告 |
| [双专家核验记录](../../reports/dual_expert_two_seeds_to1005_20261021/verified_final_status.json) | seed157 完成 1005；seed6101 到 525 后因连接失败停止 |
| [CW 延长至 1005](qgeognn_v2_row_cw_selector_extension_to_1005/README.md) | 七臂×两个种子完成；Whitening 按要求停在 525；保留探索性解释 |
| [固定预算与自适应对照](qgeognn_v2_batch_adaptivity/FINAL_REPORT.md) | 相同 653 标签预算；自适应终点改善，学习曲线面积未改善 |
| [效率复核](qgeognn_v2_efficiency_review/REPORT.md) | 汇总终点、曲线面积、目标精度所需标签与计算成本，未重新训练 |
| [连续 B32](qgeognn_v2_row_sequential_b32/FINAL_REPORT.md) | LCMD / Hybrid 相对 Random 平均曲线面积改善约 23%，各五个种子获胜 |
| [Kernel-IVR](qgeognn_v2_row_kernel_ivr_b32/FINAL_REPORT.md) | 完成探索性扩展，未建立优于 LCMD 的明确收益 |
| [IVR 机制审计](qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md) | 在 Gate C 前停止，不支持新增效果验证的声明 |
| [小批量基准](qgeognn_v2_row_small_batch_benchmark/README.md) | B32/B16 对照；批大小效应仍需匹配总预算解释 |
| [Hybrid 扩展](qgeognn_v2_row_hybrid_extension/README.md) | 完成匹配的一步对照；说明、协议与结果已合并 |
| [LCMD 试点](qgeognn_v2_row_lcmd/FINAL_REPORT.md) | 大批量试点；后续完整连续曲线以上方研究为准 |
| [机制候选筛选](qgeognn_v2_row_innovation_screen/README.md) | 只比较固定几何与选样，不重训、不读取测试标签 |
| [CW 因子对照与反馈边界](qgeognn_v2_row_cw_selector_factorial/README.md) | 两种子开发实验，原门槛与后续方案分别解释 |
| [CW 延长至 653](qgeognn_v2_row_cw_selector_extension_to_653/README.md) | 开发后探索；六条轨迹各增加四轮，原 333–525 门槛不变 |

原始自动判定 `NO_CLEAR_SEQUENTIAL_AL_GAIN` 保留不改：它未确定 LCMD 与 Hybrid 的稳定排名，不能解释为二者相对随机没有收益。CW、机制筛选与连续基准是不同证据，不能直接混合排名。

[早期 A1a 对照](4g/hybrid_batch_control/README.md)使用旧预测器与设计，单独作为历史记录。主动迁移随迁移研究归档，不列入 main 后续工作。

Free LLM 的[原设计](../../docs/research/4G_LLM_SCIENTIST_ACTIVE_LEARNING_V2_2026-09-26.md)、完整执行代码、冻结协议和历史记录继续保留在 main。

方法来由见[主动学习方法沿革](../../docs/research/主动学习方法沿革.md)和 [LLM 方案沿革](../../docs/research/LLM方案沿革.md)。下一步看[当前状态](../../docs/NEXT_STAGE_DECISION.md)，运行命令看[脚本索引](../../scripts/README.md)。

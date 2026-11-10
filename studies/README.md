# 研究证据入口

状态基准：2026-11-10。本次清理以 `0e33a86d3dea` 为基准，在现有 `codex/repository-architecture-20261109` 分支进行；正在运行实验的 main 不在本轮迁移范围。科学状态统一见 [NEXT_STAGE_DECISION](../docs/NEXT_STAGE_DECISION.md)，候选能力见 [ARCHITECTURE](../docs/ARCHITECTURE.md)。

状态统一含义：`CURRENT` 为本候选的主要阅读入口；`FROZEN_HISTORICAL` 为保留原身份的科学证据；`ARCHIVED_SOURCE` 为从固定 Git 来源恢复的历史内容；`SUPERSEDED` 为已被后续阶段取代的方案或快照；`ACTIVE_RUNTIME` 为 main 上受保护的运行项目，不能在候选执行或移动。状态不是新训练/揭示标签的授权。

| 主题 | 状态 | 阅读入口与边界 |
| --- | --- | --- |
| 现代 4g QGeoGNN-V2 主动学习 | CURRENT | [主动学习索引](active_learning/README.md)；分开阅读五 seed 基线、CW 开发实验、LLM 历史轨迹和 main 运行项目 |
| 预测器资格与输入语义 | CURRENT | [预测器索引](predictor/README.md)；独立 V2 点预测资格成立，不等于分位数 UQ 已校准 |
| 分位数/损失离线复盘 | CURRENT（仅 main） | [2026-11-09 五 seed L333 诊断](https://github.com/guoscfish/column_chromatography_prediction/tree/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md)；此处只链接，不复制结果 |
| CW 多预算/多 selector 关系 | FROZEN_HISTORICAL | [CW 研究身份索引](active_learning/CW_RESEARCH_INDEX.md)；扩展关系不能覆盖原门槛或合并协议 |
| 早期 E2 / oracle / 工程研究 | FROZEN_HISTORICAL | [experiments 索引](../experiments/INDEX.md)；模型、预算和 test 消费边界与现代 V2 分开 |
| 旧 4g Track A / 语义审计 | FROZEN_HISTORICAL | [旧 Track A](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/track_a_4g_al)、[I0 输入语义审计](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/studies/i0_predictor_semantic_audit)；否定结果及原始方法解释继续保留 |
| 跨柱迁移与历史主动迁移 | ARCHIVED_SOURCE | [跨柱归档提交 dadd2281](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/studies)；停止维护，不与现代 4g AL 合并排名 |

先读研究报告，再查协议、汇总表、标签账本和哈希。`runtime/`、原始请求、检查点等可能是本地 ignored 工件；固定 Git SHA 只恢复已跟踪文件，不能替代本地资源备份。稀疏工作树未展开不等于 Git 中不存在。

报告分级见 [reports/README](../reports/README.md)，历史代码与恢复契约见 [工程审计导航](../docs/repository/README.md)。旧报告中的运行命令保留历史语境；候选唯一 CLI 目前只支持合成生命周期，不能据此启动旧科学实验。

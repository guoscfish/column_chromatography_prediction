# 工程审计与恢复导航

状态：`CURRENT`。日常系统设计只看 [docs/ARCHITECTURE.md](../ARCHITECTURE.md)，科学状态看 [NEXT_STAGE_DECISION.md](../NEXT_STAGE_DECISION.md)。本页只索引工程验收版本，不把历史审计当作当前实验进度。

所有重构继续在 `codex/repository-architecture-20261109`。本地及 origin 仅保留 `main`、`codex/archive-transfer-20261028` 与该重构分支；迁移学习按固定来源封存，main 继续承载正式实验。

| 版本 | 状态与权威入口 | 原始收据来源 |
| --- | --- | --- |
| 一期源码隔离/最小 API（2026-11-09） | FROZEN_HISTORICAL；不是当前目录布局 | [固定完整审计包](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/docs/repository)，含一期报告、PID快照、105项历史测试与退休候选；本轮只退出通过逐文件门槛的副本 |
| 二期单一 AL 架构（2026-11-09） | FROZEN_HISTORICAL release；[91项测试与科学边界](phase2/TEST_REPORT.md) | [迁移/删除/恢复锁](phase2/HISTORICAL_PROTOCOLS.json)、[374项源码退出证明](phase2/DELETION_PROOF.csv)、[完整固定版本](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/docs/repository/phase2) |
| 文档与研究产物清理（2026-11-10） | CURRENT 工程审计；[清理报告](releases/document-cleanup-20261110/CLEANUP_REPORT.md) | 单一版本包 [document-cleanup-20261110/](releases/document-cleanup-20261110/)，保存逐文件清单、类型引用、远端恢复、实际删除和 main 保护证明 |

一期可从固定 `0e33a86d3dea3945acb0a638ca78a8f5065980a5` 恢复原始字节，不在当前 docs 再复制一套。二期的机器锁继续保留原路径，因为恢复工具及既有验收明确依赖它们；本轮不移动或重算二期冻结收据。版本包索引集中，文件本身不因“导航收敛”而重复迁移。

## 仍在原路径的机器契约

- [PROTECTED_ARTIFACTS.json](../PROTECTED_ARTIFACTS.json)：受保护资源路径，Git tag不能代替真实数据/checkpoint备份。
- [data_consumption_register.json](../data_consumption_register.json)：已使用真值和确认性研究边界。
- [PRUNE_MANIFEST.csv](../PRUNE_MANIFEST.csv)、[RESEARCH_SCOPE.json](RESEARCH_SCOPE.json)、[RETIREMENTS.json](RETIREMENTS.json)：既有退役和跨柱研究来源。
- [二期历史来源注册表](phase2/HISTORICAL_PROTOCOLS.json)：固定 scientific_source / phase1_candidate 完整 SHA；更早研究须使用自己的 BASE_COMMIT。

旧设计与旧进度报告的固定来源可由 [实验索引](../../experiments/INDEX.md)、[历史决策摘要](../../experiments/HISTORICAL_DECISIONS.md) 和 [报告身份索引](../../reports/README.md) 检索。负结果、协议、采样序列、标签消费声明、授权和恢复链都不能因目录过时而丢弃。

历史 Markdown 的原字节不为美化而重写。全量链接扫描将候选当前链接、稀疏未展开目标、固定 Git 来源和保留原文中的历史相对链接分别记录；对后者提供独立 pinned 解析结果，不冒称原相对链接已在候选树修好。

## 离线复查

工具用途与命令见 [tools/README.md](../../tools/README.md)。文档清理不更改模型、loss、选样、prompt、controller 或运行环境，也不启动科学训练、付费请求或真值揭示。最新 main 的保护基线独立固定为本轮实测状态，不复用二期旧 HEAD 例外。

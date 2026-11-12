# 当前文档

日常树只保留日常设计、科学合同与研究状态。旧施工报告、清理库存、分阶段测试日志和历史方案从 [固定历史快照](https://github.com/guoscfish/column_chromatography_prediction/tree/973ccaf8cb436e129ba3facc79d42853263923ce) 查阅，原始字节不重写。

| 问题 | 入口 |
| --- | --- |
| 安装、最小示例、历史恢复 | [根README](../README.md) |
| 当前API与表示/选样组合 | [ARCHITECTURE](ARCHITECTURE.md) |
| 当前研究事实、假设与边界 | [NEXT_STAGE_DECISION](NEXT_STAGE_DECISION.md) |
| 输入单位、条件与标签权限 | [INPUT_SCHEMA](model/INPUT_SCHEMA.md) |
| 当前模型、loss、评分科学身份 | [CODE_CONTRACT](model/CODE_CONTRACT.md) |
| 少量权威结果与负结果 | [research](../research/README.md) |
| 逐profile实际能力 | [CAPABILITY_MATRIX](CAPABILITY_MATRIX.md) |
| 等价证据、阻塞与精简统计 | [EQUIVALENCE_REPORT](EQUIVALENCE_REPORT.md) |
| main方法的版本及迁移身份 | [LEGACY_TO_NEW_CAPABILITY_MATRIX](LEGACY_TO_NEW_CAPABILITY_MATRIX.md) |

数据/权限合同位于 [protected_artifacts](../configs/protected_artifacts.json)、[data_consumption_register](../configs/data_consumption_register.json)。原论文/发布代码与历史项目实现的详细比较保留在 [归档paper/code审计](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/docs/model/PAPER_CODE_CONTRACT_AUDIT.md)；其中“current Clean”描述原审计时期，不能替代当前独立V2合同。

`CURRENT` 表示当前实现或阅读入口；`PRIMARY_FROZEN` 是按原科学身份保留的结果；`NEGATIVE_BUT_IMPORTANT` 是重要否定/部分证据；`ARCHIVED_SOURCE` 由固定历史版本恢复；`ACTIVE_RUNTIME` 只标记main运行/恢复保护边界，不表示本次核验其已完成。

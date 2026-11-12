# 日常科研依据

这里只保留未来工作最常用的精炼结果。它们是已有冻结证据的摘要，不是新训练或重写原结果。完整历史唯一主入口：远端annotated tag `archive/framework-v2-pretrim-20261111`，固定提交 [973ccaf8cb436e129ba3facc79d42853263923ce](https://github.com/guoscfish/column_chromatography_prediction/tree/973ccaf8cb436e129ba3facc79d42853263923ce)。旧协议、Prompt/响应、采样账本、原始图与失败链在那里按原身份检索。

| 问题 → 方法 | 最可信入口 | 数据角色与结论边界 |
| --- | --- | --- |
| 当前模型是否可用 → 独立V2最终资格 | [PREDICTOR](PREDICTOR.md) | 3seed×row/compound；已披露test评价，点预测资格不等于校准UQ |
| 长程标签效率 → Free LLM32 / gradient-LCMD32 | [FIVE_SEED_AL](FIVE_SEED_AL.md) | 五seed、L333–1485 validation；终点与AULC分开、mixed-model前缀保留 |
| CW信息是否有益 → NoCW历史对照 / main CW-Free | [CW_NOCW](CW_NOCW.md) | NoCW固定进度、main独立运行指针；不是已完成纯配对因果结论 |
| 是否晋升GP/IVR → 概率与数值门槛 | [NEGATIVE_RESULTS](NEGATIVE_RESULTS.md) | GP开发validation、IVR历史test及label-free机制audit分别解释 |

保留两份小型原始aggregate：`predictor_qualification_metrics.csv`（18个run/split指标行）和 `free_lcmd_validation_curves.json`（四方法冻结曲线，Random/Hybrid止于L1005）。它们不含行级真实标签/预测，不承担原始数据备份角色；来源路径、SHA与限定位于对应摘要。

当前 [数据保护](../configs/protected_artifacts.json)、[真值消费](../configs/data_consumption_register.json) 和 [archive恢复配置](../configs/archive_manifest.json) 是权限/来源入口。科学原始数据、ignored权重、完整runtime、未提交请求与凭据须独立保护，不能凭Git tag宣称可完整复现实验。main中的运行与原冻结hash不因候选整组退出而改变。

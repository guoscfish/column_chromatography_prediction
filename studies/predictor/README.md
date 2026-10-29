# 预测器研究

当前模型是独立 **QGeoGNN-V2**。main 保留其点预测资格验证和主动学习依赖；Clean 为历史失败架构。

| 入口 | 建立了什么证据 |
| --- | --- |
| [独立模型工程验证](final_v2_engineering/README.md) | 458,952 个有效参数；与 R2-pruned 六输出完全一致 |
| [4g 正式资格验证报告](final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) | 三种子、row / compound 六次冻结验证 |
| [资格验证、预注册与分位数审计](final_4g_qualification/README.md) | 复现所需检查点与原协议；分位数头用于点预测，不代表不确定性已合格 |
| [历史模型索引](historical/README.md) | 旧模型、条件补全、R2 剪枝与独立 V2 的演变 |
| [输入语义审计](semantic_input_audit/README.md) | 输入特征及数据含义 |
| [条件补全](condition_completion/README.md) | 条件输入的历史验证 |
| [早期基准预注册](4g_source_benchmark/README.md) | 保留旧设计，不代表已执行当前正式基准 |

实现入口为 [models/qgeognn_v2.py](../../src/qgeognn_al/models/qgeognn_v2.py)。历史判定不改写；其中过时的待办由[当前研究状态](../../docs/NEXT_STAGE_DECISION.md)取代。

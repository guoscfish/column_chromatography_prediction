# 文档导航与维护边界

状态：`CURRENT`。更新：2026-11-10。本页只负责导航、状态与维护规则；系统设计在 [ARCHITECTURE.md](ARCHITECTURE.md)，科学结果和下一步在 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。

## 日常入口

| 需要了解什么 | 权威入口 |
| --- | --- |
| 项目目标与最小演示 | [项目首页](../README.md) |
| 确认结果、假设与运行边界 | [当前研究状态](NEXT_STAGE_DECISION.md) |
| 候选 API、回合状态机与配置 | [系统架构](ARCHITECTURE.md) |
| 输入语义和论文代码合同 | [输入语义](model/INPUT_SCHEMA.md)、[paper/code 合同审计](model/PAPER_CODE_CONTRACT_AUDIT.md) |
| 研究问题、协议与结果 | [研究总索引](../studies/README.md)、[主动学习](../studies/active_learning/README.md)、[预测器](../studies/predictor/README.md) |
| 早期实验与跨柱归档 | [experiments 索引](../experiments/INDEX.md) |
| 比较、完成结果、进度和运行证据 | [reports 索引](../reports/README.md) |
| 工具用途和写入边界 | [tools 说明](../tools/README.md) |
| 工程验收与清理恢复链 | [工程审计导航](repository/README.md) |

## 分支与状态

全部重构继续位于 `codex/repository-architecture-20261109`，本轮从二期提交 `0e33a86d3dea3945acb0a638ca78a8f5065980a5` 整理文档与研究产物。原 main 工作目录继续运行实验；本轮开始时其 HEAD 为 `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`，这是固定观察身份，不是实时进度。main 后续变化不自动合并到候选。

| 状态 | 含义 |
| --- | --- |
| `CURRENT` | 候选当前维护的导航、实现或科学入口；不自动代表可运行真实实验 |
| `FROZEN_HISTORICAL` | 冻结的研究协议、结果或身份，按原模型、预算、数据角色解释 |
| `ARCHIVED_SOURCE` | 当前树已退出的材料，通过固定 Git SHA 与恢复登记查阅 |
| `SUPERSEDED` | 被指定后续版本取代的草案或快照，不能当现行规则 |
| `ACTIVE_RUNTIME` | 原 main 承载的运行或恢复链，不在本轮改写/移动/删除范围 |

状态属于具体版本和实验身份。进度不是最终结果；失败和负结果也是科研证据。历史“下一步”描述当时判断，不覆盖当前研究状态。不同 seed、budget、model、Prompt、row/compound 角色不因目录相近而合并。

## 数据与产物

[PROTECTED_ARTIFACTS.json](PROTECTED_ARTIFACTS.json) 保留源 CSV、划分、scaler、权威 checkpoint 与缓存锚点。[data_consumption_register.json](data_consumption_register.json) 保留曝光和确认性研究边界；具体后续使用以研究协议、冻结记录与标签回执为准。已用于拟合、选模、评价或诊断的数据不能重新称为盲确认集。

保留不可替代的原数据、授权、方法定义、指标、采样序列、哈希、来源和负结果。可重建图表或旧进度也须核对依赖、固定来源与远端恢复链，才能退出日常视图。清理记录见 [工程审计导航](repository/README.md)。

Git 来源不含被忽略或未提交的模型、完整请求、凭据和 runtime。源码恢复成功不代表完整实验数据已备份。不要在运行中的 main 工作目录 checkout、reset、clean 或执行恢复控制器来验证归档。

## 历史来源与文档分工

本轮清理前的材料固定在 [0e33a86d 的 docs](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/docs)。历史源码使用各自 BASE_COMMIT，恢复锁见 [HISTORICAL_PROTOCOLS.json](repository/phase2/HISTORICAL_PROTOCOLS.json)。迁移学习于 2026-10-28 归档停止维护，固定来源 [dadd2281](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b)，分离映射仍在 [RESEARCH_SCOPE.json](repository/RESEARCH_SCOPE.json)。候选精简不授权删除 main 的冻结兼容文件。

每项研究优先一份 README，说明问题、方法、结果、限制与权威产物；预注册、冻结协议和长报告可独立保留。新增分析先更新研究入口，避免每轮修复/进度/下一步各写一份项目级报告。科学数值由权威报告维护，工程数量由对应 release 清单维护。

运行方式只在 [系统架构](ARCHITECTURE.md) 与 [方法注册表](../configs/method_registry.json) 说明。旧 runner 和旧模型路径若退出候选，引用固定 SHA 来源，不保留失效的相对链接。main 中存在旧命令不表示候选也支持它。

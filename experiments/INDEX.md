# experiments 证据与恢复索引

本页区分候选基线 Git 树中实际保留的 21 个研究目录与仅在跨柱归档中存在的目录，替代“每个旧目录都仍在当前工作树”的过时说法。科学身份以固定 `0e33a86d3dea` 来源核对，本轮清理基线为 `382fc755c1c2`；稀疏 checkout 不展开二进制或表格，不意味着文件被删。

状态统一含义：`CURRENT` 为本候选的主要阅读入口；`FROZEN_HISTORICAL` 为保留原身份的科学证据；`ARCHIVED_SOURCE` 为从固定 Git 来源恢复的历史内容；`SUPERSEDED` 为已被后续阶段取代的方案或快照；`ACTIVE_RUNTIME` 为 main 上受保护的运行项目，不能在候选执行或移动。状态不是新训练/揭示标签的授权。

现代独立 V2 与当前 AL 入口在 [studies](../studies/README.md)。下表均保留原 predictor、split、预算和原数据消费语境；不能与现代五 seed 曲线混排。数据、checkpoint、scaler、graph cache 和 partition 的保护清单仍为 [PROTECTED_ARTIFACTS](../docs/PROTECTED_ARTIFACTS.json)。

## 候选基线仍有 Git 跟踪内容

下列链接固定到清理前提交；它们是可检索的原始证据，不承诺 fresh clone 拥有全部 ignored runtime。具体被本次退出的非权威副本以工程清理清单为准。

| 目录 / 原报告 | 状态 | 科学角色及边界 |
| --- | --- | --- |
| [e0_4g_baseline](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e0_4g_baseline/README.md) | FROZEN_HISTORICAL | 原 Gate0、seed42、4g source checkpoint/图/scaler/划分；受保护资源 |
| [e1_signal_qualification](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e1_signal_qualification/README.md) | FROZEN_HISTORICAL | 原 Gate0 predictor 的 K3 成员与信号资格；不能直接作为现代 V2 的 UQ 资格 |
| [data_audit](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/data_audit/AUDIT_REPORT.md) | FROZEN_HISTORICAL | 原数据来源与语义核查 |
| [d28_al_engineering](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/d28_al_engineering/README.md) | FROZEN_HISTORICAL | 原 AL 接口/partition/resume 证据；smoke 不是科学结果 |
| [e0_3b_controls](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e0_3b_controls/README.md) | FROZEN_HISTORICAL | 4g→8g 多种子控制；原迁移模型与当前 4g AL 分开 |
| [e0_3c_loss_controls](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e0_3c_loss_controls/README.md) | FROZEN_HISTORICAL | 8g 微调的 0.5/1/SD loss 对照，不证明当前 4g AL 权重因果 |
| [d04_conformer_selection](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/d04_conformer_selection/README.md) | FROZEN_HISTORICAL | 构象控制及非收敛边界；保留负结果与缓存来源 |
| [e2_4g_active_learning](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e2_4g_active_learning/README.md) | FROZEN_HISTORICAL | 旧 predictor、3 seed、L0=375/B25 的 row pilot |
| [e2_4g_compound_active_learning](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e2_4g_compound_active_learning/README.md) | FROZEN_HISTORICAL | 旧 predictor compound pilot；suggestive，不是稳定泛化胜出 |
| [e2_compound_failure_audit](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e2_compound_failure_audit/README.md) | FROZEN_HISTORICAL | test 已用于事后失败/几何分析，不能用于新策略设计确认 |
| [d45_oracle_marginal_utility](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/d45_oracle_marginal_utility/README.md) | FROZEN_HISTORICAL | test truth 用于 oracle utility；非确认性 acquisition 证据 |
| [d46_oracle_utility_reliability](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/d46_oracle_utility_reliability/README.md) | FROZEN_HISTORICAL | truth-selected 18 candidates，确定训练导致名义 seed 独立性不可识别 |
| [e0_8g_transfer](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e0_8g_transfer) | FROZEN_HISTORICAL | 保留 source/数据依赖的兼容子集；完整跨柱来源见归档 |
| [g0_3_threshold_sensitivity](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/g0_3_threshold_sensitivity) | FROZEN_HISTORICAL | 574 行 no-threshold target 与来源；受保护资源 |
| [e4_active_transfer_preregistration](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e4_active_transfer_preregistration) | FROZEN_HISTORICAL | 候选仍保留的预注册/partition 兼容子集；完整归档保留方法身份 |
| [e4_a2a_low_budget_preregistration](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e4_a2a_low_budget_preregistration) | FROZEN_HISTORICAL | L0=30 历史主动迁移设计/标签划分子集 |
| [e4_a2a_low_budget_formal](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e4_a2a_low_budget_formal) | FROZEN_HISTORICAL | 已测主动迁移 null 的保留证据；不代表领域普遍不可能 |
| [e2_4g_compound_preflight](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e2_4g_compound_preflight/README.md) | SUPERSEDED | Round0/选样 dry-run 被正式 compound pilot 取代 |
| [e2_random_smoke](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/e2_random_smoke/README.md) | SUPERSEDED | 旧工程链验证，不是 Random 科学基线 |
| [reproductions](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/reproductions) | FROZEN_HISTORICAL | 非权威运行支持材料；Git 跟踪子集与本地 ignored runtime 分开 |
| [4g_error_landscape](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/4g_error_landscape/README.md) | FROZEN_HISTORICAL | 旧误差景观诊断；保留结论和图的来源 |

## 已归档、当前候选没有该目录

固定归档 `dadd2281fba01eb4414128cbf9feb2859c78544b`（`codex/archive-transfer-20261028`）保存下列原目录；其旧协议不是候选可执行能力。兼容子集与完整归档的关系不得仅凭目录同名推断字节相同。

| 原目录 | 状态 | 保留理由 |
| --- | --- | --- |
| [g0_1_quantile_monotonicity](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/g0_1_quantile_monotonicity) | ARCHIVED_SOURCE | 旧 Gate0 单调头对照，不是当前独立 V2 头 |
| [g0_2_interval_calibration](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/g0_2_interval_calibration) | ARCHIVED_SOURCE | 旧 Gate0 calibration，不构成现代 V2 已校准声明 |
| [g0_4_paper_style_transfer](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/g0_4_paper_style_transfer) | ARCHIVED_SOURCE | paper-style 迁移负结果，未替换 last2 |
| [g0_4_paper_style_transfer_random_init_diagnostic](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/g0_4_paper_style_transfer_random_init_diagnostic) | ARCHIVED_SOURCE | 随机适配器失败诊断，后续零初始化方案已单独比较 |
| [e4_protocol_a_engineering_smoke](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/e4_protocol_a_engineering_smoke) | ARCHIVED_SOURCE | 工程 smoke，不能当正式结果 |
| [e4_a2a_engineering_smoke](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/e4_a2a_engineering_smoke) | ARCHIVED_SOURCE | 低 L0 工程 smoke，不是新科学比较 |
| [e4_protocol_a_formal](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/e4_protocol_a_formal) | ARCHIVED_SOURCE | 历史 Protocol A：tested active-acquisition null |
| [e4_protocol_a_headroom_audit](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/e4_protocol_a_headroom_audit) | ARCHIVED_SOURCE | Protocol A 事后机制与首轮冲击诊断 |
| [e4_transfer_aware_acquisition_qualification](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/e4_transfer_aware_acquisition_qualification) | ARCHIVED_SOURCE | 资格 gate failed；没有 performance evidence |
| [e4_active_learning_suitability_diagnosis](https://github.com/guoscfish/column_chromatography_prediction/tree/dadd2281fba01eb4414128cbf9feb2859c78544b/experiments/e4_active_learning_suitability_diagnosis) | ARCHIVED_SOURCE | T3R gate failed，历史 truth 不得回用调参 |

## 决策与恢复

[历史决策摘要](HISTORICAL_DECISIONS.md) 收敛旧 decision/register/roadmap 阅读入口，原文按固定提交保留。新研究决策只看 [NEXT_STAGE_DECISION](../docs/NEXT_STAGE_DECISION.md)。[原 Predictor Freeze](https://github.com/guoscfish/column_chromatography_prediction/tree/0e33a86d3dea3945acb0a638ca78a8f5065980a5/experiments/PREDICTOR_FREEZE.md) 是旧 Gate0 身份合同，不覆盖后来的独立 V2。

数据与真值使用声明见 [data_consumption_register](../docs/data_consumption_register.json)。尤其 E2 failure、D45/D46 的 test 已用于事后诊断，不再是未使用的确认集。可恢复 Git 源码不等于可恢复 ignored checkpoint/请求/runtime；后者必须有独立资源来源和原 SHA，不在本轮凭空补造。

## 本轮逐文件清理与保留边界

本轮结果以 [全仓库冗余验收](../docs/repository/releases/redundancy-cleanup-20261110/CLEANUP_REPORT.md) 为准，不把目录级“SUPERSEDED”当作删除授权。`e2_random_smoke` 的 manifest 明确绑定checkpoint、reveal后选择、split/scaler和test使用事实；`e2_4g_compound_preflight` 也绑定原Round0模型、特征和dry-run选择。它们不能只因被后续pilot覆盖而整目录删除。

`reproductions/d46_oracle_reliability_smoke` 的四张纯aggregate图已用固定历史绘图逻辑和跟踪输入在隔离目录实际重建、逐像素比较；此验证不运行训练、不读取原始test预测。缺少独立paired-mean CI输入的forest、原协议/配置/来源/fit/reset/消费声明仍保留，不能由四图可重建推导整个旧实验可完整重训复现。

保留的跨柱兼容资源已逐文件与 `dadd2281fba01eb4414128cbf9feb2859c78544b` 比较 Git blob；相同并不解除保护清单或历史消费者。真实模型、原数据、partition/scaler/cache及ignored runtime恢复能力仍须单独验证。

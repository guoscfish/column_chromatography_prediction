# 色谱主动学习历史决策行为审计（2026-11-16）

主文：[REPORT.md](REPORT.md)。这是已经实际执行的只读离线审计，含代码、统计、8幅图与完整中文研究报告；未训练模型、重跑AL、调用LLM/Judge或查询新标签。

已分析22条真实轨迹、791批/25,312次选点、813个保存的Row评价点。Historical CW-Free / New CW-Free / NoCW / CW-LCMD各五seed到L1485，Hybrid仅157/6101，真实共同区间至L1453。原共同L1005窗口与扩展L1453分开，所有AULC积分真实非单调损失，越低越好。

| 内容 | 入口 |
| --- | --- |
| 中文研究报告与研究路线表 | [REPORT.md](REPORT.md) |
| 22条运行的路径、hash、配置和horizon | [RUN_INVENTORY.csv](RUN_INVENTORY.csv) |
| 协议/CW/LCMD/模型前缀/随机约束审计 | [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md) |
| 论文原方法与V/U迁移审计 | [PAPER_METHOD_AUDIT.md](PAPER_METHOD_AUDIT.md) |
| 数学定义、尺度、时间角色与统计规则 | [METHODS.md](METHODS.md) |
| 逐批与逐候选 | [batch_metrics.csv](batch_metrics.csv)、[candidate_metrics.csv](candidate_metrics.csv) |
| G/U/D/C全池分位、Logit及缺失 | [PREFERENCE_RESULTS.md](PREFERENCE_RESULTS.md)、[behavior_preferences.csv](behavior_preferences.csv)、[preference_state_coverage.csv](preference_state_coverage.csv) |
| Hybrid来源、四种覆盖边际、条件随机 | [HYBRID_TEXT_RESULTS.md](HYBRID_TEXT_RESULTS.md)、[hybrid_attribution.csv](hybrid_attribution.csv)、[hybrid_gradient_attribution.csv](hybrid_gradient_attribution.csv)、[hybrid_conditional_random.csv](hybrid_conditional_random.csv) |
| 行为—下一拟合损失、阶段与lag2 | [gain_associations.csv](gain_associations.csv)、[gain_linked_batches.csv](gain_linked_batches.csv)、[phase_metrics.csv](phase_metrics.csv)、[lag2_associations.csv](lag2_associations.csv) |
| 可见理由、正反例与行动偏好动态 | [rationale_metrics.csv](rationale_metrics.csv)、[decision_cases.csv](decision_cases.csv)、[action_preference_dynamics.csv](action_preference_dynamics.csv)、[concentration_campaigns.csv](concentration_campaigns.csv) |
| 距离/分箱/随机/负采样敏感性 | [sensitivity_metrics.csv](sensitivity_metrics.csv)、[random_stability.csv](random_stability.csv)、[preference_negative_sensitivity.csv](preference_negative_sensitivity.csv) |
| 独立计算审阅与局限 | [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md)、[LIMITATIONS.md](LIMITATIONS.md) |
| SHA封存与测试 | [INPUT_MANIFEST.json](INPUT_MANIFEST.json)、[FINAL_VERIFICATION.json](FINAL_VERIFICATION.json)、[tests/](tests/) |

## 离线复现

使用本机既有`/Users/fish/miniforge3/envs/chromatography/bin/python`，依赖NumPy、pandas、SciPy、Matplotlib、RDKit及测试用pytest。不联网、不安装依赖、不导入训练runner。当前无pyarrow/fastparquet，逐候选表以CSV交付；`PARQUET_UNAVAILABLE.txt`记录原因，不使用假Parquet扩展名。

从仓库根目录运行只读检查：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/llm_decision_behavior_audit_20261116/run.py --check
```

显式配置与完整重新计算入口（只重写本研究派生文件，先验证旧封存与本研究输入）：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/llm_decision_behavior_audit_20261116/run.py --config studies/active_learning/llm_decision_behavior_audit_20261116/config.json --recompute
```

`config.json`固定来源、snapshot、方法、预算上限、500随机批、分箱与参考数。`collect.py`按这些路径逐文件绑定freeze/contract，不自动接入未来运行。协议盘点、Hybrid与偏好模块保留各自显式冻结解析路径及输入manifest；偏好参数为`--states ... --bootstrap 100 --negative-seeds 7001 7002 7003`。完整重算只做既存缓存距离、轻量Logit和统计，不重算GNN梯度或预测。全部原始本地缓存并非都随Git追踪；新clone缺输入会明确失败，不能自动生成不存在的状态。

模块顺序为：旧封存检查→inventory→Hybrid身份提取→collect→analyze→preferences→Hybrid完整归因/文本→performance→figures→write_report→单元测试→输入/输出封存。单独重画图运行`figures.py`，单独重建报告运行`write_report.py`；修改本研究结果后须重新封存，不能把旧SHA检查失败忽略掉。

## 图表

`FIGURES/figure1…figure8`各有PNG/PDF/SVG：行为概览、历史局部性、G/U/D/C分解、探索代理、行为与Row性能、Hybrid、三LLM轨迹、实际决策案例。图中所有“随机”均对应各自状态；图1大部分面板用共同L1453，G/U面板明确标核心L1005。图5区间仅是五seed块的探索性描述；Hybrid只有两seed。

## 使用边界

没有原始同义Value、校准epistemic uncertainty或可比信念概率，故没有宣称严格复现原V/U、认知不确定性回避或M/Z过度反应。G部分缓存审计未展开的状态保持缺失；候选值不使用未来反馈。最终理由只解释为stated rationale。所有行为与Row关联均非因果干预效果，未来验证建议尚未执行。

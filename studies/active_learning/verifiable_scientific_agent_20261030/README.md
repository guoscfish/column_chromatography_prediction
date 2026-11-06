# 可验证科学实验Agent：离线研究交付

结论入口：[REPORT.md](REPORT.md)。本次完成后停止；目录中的建议和mock均不授权真实LLM、训练或闭环续跑。

## 文件

| 内容 | 文件 |
| --- | --- |
| 结论、失败分类、统计协议 | [REPORT.md](REPORT.md)、[FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md)、[PROTOCOL.md](PROTOCOL.md) |
| 主数据 | [failure_events.csv](failure_events.csv)、[repair_trajectory_analysis.csv](repair_trajectory_analysis.csv)、[reliability_metrics.csv](reliability_metrics.csv) |
| 固定来源与完整响应 | [history_snapshot_manifest.json](history_snapshot_manifest.json)、[response_manifest.jsonl](response_manifest.jsonl)、[history_audit_summary.json](history_audit_summary.json) |
| 成本与关联 | [completed_response_costs.csv](completed_response_costs.csv)、[correction_request_causes.csv](correction_request_causes.csv)、[input_scale_reliability.csv](input_scale_reliability.csv) |
| 原型与局部修复协议 | [DESIGN.md](DESIGN.md)、[verifiable_planner.py](verifiable_planner.py)、[BENCHMARK_NOTES.md](BENCHMARK_NOTES.md) |
| 离线A/B/C | [offline_benchmark_summary.csv](offline_benchmark_summary.csv)、[offline_benchmark_results.json](offline_benchmark_results.json) |
| 科学解释病例 | [SCIENTIFIC_ISSUE_REVIEW.md](SCIENTIFIC_ISSUE_REVIEW.md)、[scientific_issue_review.csv](scientific_issue_review.csv) |
| 纠错前后几何 | [REPAIR_GEOMETRY_NOTES.md](REPAIR_GEOMETRY_NOTES.md)、[repair_geometry_profiles.csv](repair_geometry_profiles.csv)、[repair_geometry_summary.json](repair_geometry_summary.json) |
| Numerical Critic | [NUMERICAL_CRITIC_INTERFACE.md](NUMERICAL_CRITIC_INTERFACE.md)、[numerical_critic.py](numerical_critic.py) |
| 自检 | [PREFLIGHT_REPORT.md](PREFLIGHT_REPORT.md)、[preflight_verification.json](preflight_verification.json)、[test_results.txt](test_results.txt) |

## 复现

从仓库根目录执行。历史审计和细诊断使用Python标准库；Critic及其指标对齐测试额外使用NumPy。当前完整测试环境为已有 `chromatography` conda环境，无需安装或训练模型。

```bash
python studies/active_learning/verifiable_scientific_agent_20261030/audit_history.py
python studies/active_learning/verifiable_scientific_agent_20261030/offline_benchmark.py
python studies/active_learning/verifiable_scientific_agent_20261030/scientific_issue_review.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/verifiable_scientific_agent_20261030/repair_geometry_analysis.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/verifiable_scientific_agent_20261030/verify_study.py
```

默认审计重放已固定manifest；`--snapshot`只用于首次创建，已有清单时拒绝覆盖，不追逐运行中的NoCW。离线benchmark核验同一snapshot中的response/input/contract/batch_freeze字节哈希。源码漂移会报错，不能静默重定口径。

完整回放需要本地原始运行文件，其中部分input、catalog、request、runtime属于gitignored证据。新克隆若缺文件，可阅读已交付CSV/JSON和报告，但不能声称独立重放了原始历史哈希链。纯mock测试不需要API凭据；真实数据依赖项缺失时不得用新LLM或标签读取补齐。

原型的 `FrozenContext` 和 `StableCandidateIndex` 只验证显式输入的一致性；`apply_mock_patch`不调用任何provider，也不能冻结正式批次。Numerical Critic同样没有生产读取适配器，其来源认证边界见接口文档。

# 主动学习历史实验综合诊断（2026-11-12）

主入口：[MASTER_REPORT.md](MASTER_REPORT.md)。若只看后续选择：[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。本目录是独立离线研究交付，没有运行新AL、付费LLM或QGeoGNN训练，没有修改历史实验源。

| 内容 | 文件 |
|---|---|
| 方法/已有证据复用清单 | [EXPERIMENT_INVENTORY.md](EXPERIMENT_INVENTORY.md) |
| 初始化、尺度、同预算与因果边界 | [PROTOCOL_AND_FAIRNESS_AUDIT.md](PROTOCOL_AND_FAIRNESS_AUDIT.md) |
| A：曲线、阶段、阈值、波动 | [LEARNING_DYNAMICS.md](LEARNING_DYNAMICS.md) |
| B：结构/条件/冻结几何 | [ACQUISITION_BEHAVIOR.md](ACQUISITION_BEHAVIOR.md) |
| C：行为到下一轮gain | [ACQUISITION_VS_LEARNING_GAIN.md](ACQUISITION_VS_LEARNING_GAIN.md) |
| D/H：科学推理、证据和记忆 | [SCIENTIFIC_REASONING_AUDIT.md](SCIENTIFIC_REASONING_AUDIT.md) |
| E：化学误差、模型响应、重复条件 | [CHEMICAL_ERROR_AND_DATA_AUDIT.md](CHEMICAL_ERROR_AND_DATA_AUDIT.md) |
| F：信号资格与互补性 | [NUMERICAL_SIGNAL_AUDIT.md](NUMERICAL_SIGNAL_AUDIT.md) |
| G：稳定性、重试、成本 | [STABILITY_AND_COST_AUDIT.md](STABILITY_AND_COST_AUDIT.md) |
| 六路线的支持、风险与最小检验 | [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) |
| 三个优先实验 | [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md) |
| 明确缺口与未测项目 | [MISSING_EVIDENCE.md](MISSING_EVIDENCE.md) |
| 输入来源锁 | [INPUT_MANIFEST.json](INPUT_MANIFEST.json) |
| 最终交付/输入/工作树核验 | [verification/final_verification.json](verification/final_verification.json) |

## 数据结构与时间角色

| 逻辑层 | 表/索引 | 时间与内容 |
|---|---|---|
| run_registry | `tables/run_registry.csv`、`additional_run_registry.csv` | 方法、seed、真实完成预算、模型/来源 |
| round_metrics | `round_metrics.csv`、`additional_round_metrics.csv` | 既有validation及fit元数据；禁止喂给selector |
| selected_rows | `behavior_selected_rows.csv` | 27936名额的ID、分子/条件元数据，不含真实响应列 |
| observed_feedback | `model_batch_metrics.csv`、`model_seed_phase_errors.csv` | 180批选后残差聚合；27条授权科学病例另表 |
| selection_behavior | `behavior_batches.csv`、`behavior_geometry.csv` | 选前/提案时可得行为与空间指标 |
| scientific_hypotheses | `reasoning_hypotheses.csv`、`reasoning_evidence_references.csv` | 假设分类、原科学文本、时间合法的证据ID |
| numerical_signals | `behavior_candidate_signals.csv`、`signals_*` | 同状态候选无标签排名、已选误差资格诊断 |
| learning links | `batch_gains.csv`、`gain_linked_batches.csv`、`gain_feature_*` | 严格 t选前→选择→揭示→t+1拟合连接 |
| source_manifest | `INPUT_MANIFEST.json`及各模块manifest | 原文件SHA、归档内SHA、可用性；不复制private packet |

共13种方法、47条run的曲线证据（八主方法＋五补充数值），行为只覆盖其中六方法。不能把选后残差当选前特征；科学病例摘录是已接受反馈的授权小子集，非全量标签表。目录中不包含权重、原始全量packet、请求正文或凭据。

## 可复现计算

使用本机已有 `/Users/fish/miniforge3/envs/chromatography/bin/python`（numpy/pandas/scipy/matplotlib/RDKit），无需安装、网络或凭据。大多数原始cache/catalog是local-only；新clone缺失时只能读报告/派生表，不应声称完整重放或自动重建。

先运行最终核验的 `--check` 验证封存输入。若不一致，停止并为新快照另建版本目录；不可改旧输入适配当前结果。主脚本只写本目录派生输出，按依赖顺序：

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/finalize.py --check

# 仅在输入锁一致、需要复算时执行；不启动实验。
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/analyze_learning.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/analyze_additional_methods.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/analyze_behavior.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/analyze_reasoning.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/analyze_model_signals.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/analyze_gain_links.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/write_quantitative_reports.py
```

补充人工审计/独立核验脚本见 `scripts/`，固定取样规则与全部判分见 `verification/`。源码分析只限具体 `.py` 与已授权报告/元数据，禁止宽范围搜索dataset标签文件。`verify_gain_independent.py`用独立null-space投影核验关联；学习AULC由独立Decimal半端点权重算法核验；几何由独立scipy距离与显式枚举核验。

## 审稿式修正与声明

交付前独立审核已修正：色条遮挡、上样/加载溶剂词义、浮点并列秩复现差、未控制起点误差敏感性、Hybrid遗漏额外成员成本、模型对照对噪声识别能力的过度表述、holdout组/尺度泄露风险、critic/placebo条件信息不对等、2×2对显式输入与预训练知识的混淆。审核及原问题保留在 `verification/independent_*review.json` 和 `model_synthesis_review.json`。

研究过程中一次单位文本检索误命中带标签manifest三行，未用于任何计算或建议，已披露于[事件记录](verification/label_access_incident.json)。因此报告的“脚本未读test真值”不能扩展为全会话零标签文件接触。历史源码、冻结文件、权重没有被本次分析修改；持续运行的旧监控可能更新自己的状态文件，最终核验单列这些变化。

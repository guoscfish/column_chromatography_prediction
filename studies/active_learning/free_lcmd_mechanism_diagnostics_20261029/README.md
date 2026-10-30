# 五 seed Free / LCMD 机制诊断产物

研究结论从 [REPORT.md](REPORT.md) 阅读；指标与证据边界见 [PROTOCOL.md](PROTOCOL.md)，H1–H6 的逐项证据与未来实验见 [mechanism_hypotheses.md](mechanism_hypotheses.md)。本目录是2026-10-29独立reporting-only分析，不覆盖历史报告，不包含任何新训练或新LLM结果。

## 复现

从仓库根目录执行。本机已验证的Python为 `/Users/fish/miniforge3/envs/chromatography/bin/python`，含Python3.11、NumPy2.4.6、pandas3.0.5、SciPy1.17.1、RDKit2025.09.6、matplotlib、pytest；默认Python没有RDKit。其他环境须安装对应分析依赖，NumPy需支持`trapezoid`（2.x）。不需要LLM凭据、GPU或模型推理。

仅合并已有CSV、重算描述性相关与总审计摘要：

```sh
python studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/assemble_diagnostics.py
```

完整重新读取冻结证据、计算指标及绘图（仍然不训练/不调用模型）：

```sh
MPLCONFIGDIR=/tmp/free_lcmd_diagnostics_mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python \
  studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/assemble_diagnostics.py --recompute
```

完整分析要求本地gitignored runtime中的checkpoint、bank、catalog和fit_audit。本次文件齐全；新克隆缺失时脚本会失败/明确缺项，绝不自动训练或重建梯度。只复现已提交曲线的数值和图可用：

```sh
python studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/learning_curve_analysis.py --snapshot-only
```

snapshot-only不等于runtime完整性审计，并会在该模块的新输出中明确注明跳过runtime校验；本次交付使用完整元数据核验模式。重算只写本目录的派生文件；报告叙述与H1–H6是人工整合的研究解释，不由脚本生成因果结论。

最终测试、图表/产物hash记录与报告数值自检：

```sh
MPLCONFIGDIR=/tmp/free_lcmd_diagnostics_mpl PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python \
  studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/verify_diagnostics.py
```

## 文件对应

| 文件 | 用途 |
|---|---|
| `evidence_audit.py` | 只调用旧报告的read-only collector，重核冻结fit/batch/feedback链，不运行旧writer |
| `runtime_integrity_audit.json`, `runtime_source_hashes.json` | 370拟合/360批独立复核及5727个源文件hash，含额外核验的旧Random/Hybrid前缀 |
| `learning_curve_analysis.py`, `CURVE_FINDINGS.md` | 归一化审计、曲线/阶段/波动/阈值/交叉/版本分层和绘图 |
| `stage_comparison.csv` | 100行：5seed×2法×10窗口（六阶段+全程+三个补充窗） |
| `round_metrics.csv` | 370行：每拟合点NRMSE、相邻改善、同预算差、full参考余量 |
| `threshold_attainment.csv`, `best_validation_points.csv`, `midstage_headroom.csv` | 首达与持续阈值/删失、事后最佳预算、起点差距算术分解 |
| `curve_summary.json` | seed均值/样本SD/方差/配对差、交叉区间与输入来源 |
| `geometry_analysis.py`, `GEOMETRY_FINDINGS.md` | raw/CW、距离分位、批内NN/冗余、覆盖/分母敏感性、effective rank及绘图 |
| `geometry_profiles.csv`, `geometry_selected_points.csv` | 360批/11520点几何；LCMD CW保持NA |
| `geometry_availability.json`, `geometry_contract_robustness_audit.json` | 逐批bank/checkpoint/index/source验证及补强的LCMD180批contract检查 |
| `condition_geometry_probe.py`, `matched_condition_geometry_probe.csv/json` | 两seed×四冻结状态的matched PE/amount/loading-volume有限探针，72行汇总；直接向量差计算小距离 |
| `behavior_geometry_amount_link.csv` | 五seed两法的上样量新条件补全与实际raw/CW rank联结（分组结果） |
| `selection_behavior_analysis.py`, `BEHAVIOR_FINDINGS.md` | 结构、条件、历史残差、文本和真实hypothesis反馈案例及绘图 |
| `behavior_profiles.csv`, `selection_records.csv` | 360批行为指标、11520条真实ID与无标签元数据/选择前数值；不输出真实响应列 |
| `behavior_data_audit.json` | 实际读取来源hash、ID映射/历史真值规则、缺项说明 |
| `llm_text_features.csv`, `text_field_mentions.csv`, `text_mentions_by_seed_stage.csv` | 180批叙述、900条字段分类、阶段汇总；不把schema键当文本证据 |
| `hypothesis_feedback_cases.json` | 固定三个病例的hypothesis、原ID、事前记录、批后揭示及下一轮更新；这些已揭示值不作为本轮事前特征 |
| `behavior_*correlations.csv`, `text_behavior_*correlations.csv` | 逐seed描述性关联，以及控制预算/期初NRMSE的rank-residual敏感性 |
| `assemble_diagnostics.py`, `selection_profiles.csv` | 对齐360批全部模块，匹配紧接该批的拟合误差改善；拒绝重复key/冲突元数据 |
| `next_fit_associations_by_seed.csv`, `next_fit_association_summary.csv` | seed单位相关，无轮次p值；同时保留未调整与六阶段中心化结果 |
| `data_availability_audit.json`, `verification.json` | 汇总完整性/结构性NA与最终自检记录 |
| `test_*.py` | 轻量数值、来源ID、时间边界、文本误命中和整合测试 |

九幅PNG为：完整validation曲线、阶段配对差、期初改善空间、raw几何、FreeCW、覆盖分母敏感性、结构/条件、历史残差邻近、matched条件几何。几何模块另存SVG。图题区分validation性能和冻结候选池采集诊断；配对单位为seed，不把11520条选点当独立重复。

可读的派生CSV/图可随Git保留，但不代替大型runtime源证据。`verification.json`及`artifact_hashes.json`只记录本次输出，不将派生文件冒充原始冻结证据。

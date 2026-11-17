# 下一阶段机制诊断与实验决策（2026-11-17）

先读 [SUMMARY_FOR_RESEARCHER.md](SUMMARY_FOR_RESEARCHER.md)，决策细节见 [NEXT_EXPERIMENT_DECISION.md](NEXT_EXPERIMENT_DECISION.md)。本轮已经完成实际离线计算；没有新LLM/Judge、QGeoGNN训练、主动学习rollout或标签查询。

建议：**固定已揭示标签的模型条件通路控制优先（初筛10fits、0LLM、0新标签）；补Hybrid三seed为Conditional Go；完整再跑NoCW为No-Go。** 所有未来实验仅设计，不自动执行。

| 研究问题 | 报告 | 新证据与落盘结果 |
| --- | --- | --- |
| 当前模型能否表达条件响应？ | [MODEL_LEARNABILITY.md](MODEL_LEARNABILITY.md) | 复用165对旧均值；新增方向/平坦三分类、seed/阶段/实际水平、三LLM与Hybrid已揭示响应；[response_seed_summary.csv](response_seed_summary.csv)、[response_phase_summary.csv](response_phase_summary.csv) |
| LLM16替换了什么？ | [HYBRID_COUNTERFACTUAL.md](HYBRID_COUNTERFACTUAL.md) | 71状态精确LCMD32，前16逐项同序匹配；[counterfactual_batches.csv](counterfactual_batches.csv)、[counterfactual_candidates.csv](counterfactual_candidates.csv)、[counterfactual_lcmd_trace.csv](counterfactual_lcmd_trace.csv) |
| 短筛与长期是否同一协议？ | [HYBRID_PROTOCOL_COMPARISON.md](HYBRID_PROTOCOL_COMPARISON.md) | 128短名单/反馈/模型/验收与评价口径审计；同口径保存validation对照单列 |
| 长期优势是否只靠某个波动？ | [PHASE_PERFORMANCE.md](PHASE_PERFORMANCE.md) | 预定四窗口原始AULC/终点；局部真实梯形和整阶段排除；[phase_performance.csv](phase_performance.csv)、[phase_local_omission_sensitivity.csv](phase_local_omission_sensitivity.csv) |
| 三LLM差别来自输入还是行为？ | [LLM_REPRODUCIBILITY.md](LLM_REPRODUCIBILITY.md) | 五seed首批packet核验；540批候选组成分析；127条纠错后展示前缀路径；[llm_prefix_cases.csv](llm_prefix_cases.csv)、[llm_score_adjustment.csv](llm_score_adjustment.csv) |
| 应投入多少预算？ | [COST_AND_DESIGN_AUDIT.md](COST_AND_DESIGN_AUDIT.md)、[NEXT_EXPERIMENT_DECISION.md](NEXT_EXPERIMENT_DECISION.md) | Hybrid三seed复用/成本；NoCW三种重跑目的；[model_control_holdouts.csv](model_control_holdouts.csv)仅生成未来固定标签检查组，不训练 |
| 身份、时序与独立核验 | [INPUT_AUDIT.md](INPUT_AUDIT.md)、[VERIFICATION.json](VERIFICATION.json) | 输入SHA、旧封存检查、数值与身份测试、独立审阅、输出封存 |

逐批类型投入为 [condition_allocation.csv](condition_allocation.csv)，selected-row误差标签为 [condition_error_tags.csv](condition_error_tags.csv)。不同因素标签可重叠，不能相加成预算或validation误差分解。Hybrid的`all32`与两个来源半批存在包含关系，不能重复累计为独立样本。

## 已执行的范围与边界

- 元数据791状态；接受反馈611状态，仅取自身批次冻结前预测和该批真实揭示后反馈，未借别轮/别方法真值。
- 精确候选反事实71状态、2272步LCMD trace、2272个两组后16比较事件；没有反事实候选真实标签或Row Gain。
- 长期Hybrid/CW两seed L333–1453四阶段；早期screen只读保存fit的validation与既有报告，原test汇总只用于口径审计，未打开逐行test数据。
- 三LLM 540状态、15个共同初始状态、127条展示前缀病例；保留原提议/纠错/最终接受，未把文本动机当成内部思维。
- 四幅新问题图位于 [figures/](figures/)，没有重做上轮八图或新增训练曲线。

## 命令行复现

使用本机已有 `/Users/fish/miniforge3/envs/chromatography/bin/python`，依赖NumPy/pandas/SciPy/Matplotlib/RDKit/pytest。数据源明确列在 [config.json](config.json)，模块的冻结路径/完整文件SHA见各input manifest及 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)。本地原始缓存并非全随Git分发，缺失应失败，不自动重建模型或查询标签。

先做只读封存检查：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/decision_mechanism_followup_20261117/run.py --check
```

全部离线复算（只重写本研究派生输出，先检查旧封存和本轮输入，不允许绕过输入漂移）：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/decision_mechanism_followup_20261117/run.py --recompute --config studies/active_learning/decision_mechanism_followup_20261117/config.json
```

实际依赖顺序：`model_learnability.py`→`counterfactual.py`→`phase_protocol.py --plots`→`llm_reproducibility.py`→`llm_prefix_diagnostic.py`→报告/图生成→独立数值核验→pytest→封存。每个模块可单独执行。`model_report.py`、`synthesize.py`、`plot_llm_mechanism.py`分别重建模型报告/图、最终决策摘要/反事实图、纠错/组成机制图；不能把调试`--limit`结果冒充71状态正式交付。

独立核验见 [MODEL_INDEPENDENT_REVIEW.md](MODEL_INDEPENDENT_REVIEW.md)、[MODEL_NUMERIC_REVIEW.md](MODEL_NUMERIC_REVIEW.md)、[PREFIX_INDEPENDENT_REVIEW.md](PREFIX_INDEPENDENT_REVIEW.md)，以及`verify_response_numeric.py`、`verify_counterfactual.py`。输入和输出hash匹配只证明版本/计算身份，不能证明LLM因果价值或模型瓶颈的唯一原因。

## 仍未识别的问题

没有新训练，不能给LCMD后16赋真实性能；没有足够同状态预测/已揭示反馈，不能给纯CW与六组条件响应作严格误差排名。两长期Hybrid有模型前缀、历史路径和开发验证集混杂；127条前缀轨迹不是随机纠错实验。特定线性条件旁路失败只否定该具体修复假设，不否定所有模型或Hybrid机制。原报告、策略、checkpoint和封存不修改。

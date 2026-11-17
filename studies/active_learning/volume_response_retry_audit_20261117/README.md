# 体积响应与LLM纠错策略审计

已完成用户授权的15次固定对照拟合，以及10个原冻结state的零训练诊断、540已接受轨迹批次/1120逻辑请求上下文审计和127批局部修复dry-run。**新LLM/API请求0，完整AL重跑0，额外拟合0。** 本目录为隔离研究产物，未修改旧模型、runner、prompt、checkpoint或历史selection。

请先读 [FINAL_DECISION.md](FINAL_DECISION.md)，逐seed模型数据见 [MODEL_COMPARISON.md](MODEL_COMPARISON.md)。A/B/C五seed等权局部差分MAE为3.32233/3.33457/3.35214mL，原Row validation NRMSE为0.39613/0.39453/0.39529。工程原型保留3634/4064原名额，430空额显式待补；0份草稿可直接当完整selection执行。

## 报告索引

| 文件 | 内容 |
|---|---|
| MODEL_ARCHITECTURE_AUDIT.md | 真实条件输入、量单位、128维路径、输出与损失 |
| VOLUME_SENSITIVITY.md | 原10 checkpoint的构图、FD/AD、激活和体积/PE对照 |
| MODEL_COMPARISON.md | 15 fits逐seed/端点/配对结果、参考门槛与稳健性 |
| TRAINING_INDEPENDENT_REVIEW.md | 独立元数据/配置QA及原validation关联限制 |
| RESULTS_INDEPENDENT_QA.md | 从30份原始预测独立重算9项表，误差≤2e−12 |
| RETRY_CONTEXT_AUDIT.md | 各真实controller、messages、逻辑/网络重试与配置身份 |
| PREFIX_FAILURE_ANALYSIS.md | 127前缀批首错、合法ID保留及H1–H5 |
| LOCAL_REPAIR_DESIGN.md | 离线保真原型、确定性修复、科学阻断 |
| future_provider_protocol.md | 未来provider×顺序独立实验，仅设计 |
| LLM_MODEL_DISCOVERY_ROADMAP.md | 有限模型发现、知识贡献归因、防validation过拟合 |
| FINAL_DECISION.md | 八项问题的数值回答与可执行下一步 |

模型图在`figures/`，其余审计图在`audit_figures/`；核心表为`control_seed_summary.csv`、`control_endpoint_summary.csv`、`paired_model_differences.csv`、`paired_endpoint_differences.csv`、`mechanism_gates.csv`、`control_robustness.csv`、`retry_case_batches.csv`、`retry_request_contexts.csv`及`local_repair_cases.csv`。各模型原始history、预测、独立checkpoint和fit receipt在`fits/seed_*/模型/`。

## 数据边界和固定选择

历史H1005标签仅来自各seed当轮冻结catalog.observed，1005个ID严格匹配contract/freeze；体积检查集来自此前预定`model_control_holdouts.csv`，每seed完整隔离10系列。训练侧等价条件/同系列重叠为0。原validation416行保持冻结规则；其中的条件关联已披露，未伪称全新确认集。

协变量由canonical CSV的feature-only列读取；原validation仅通过现有RestrictedLabelStore按role核验后投影数值。其csv流会经过原文件行，hash检查也经过原文件字节，因此不宣称从未读取目标文件字节；未请求、转换为数值、输出或用于分析test/未揭示候选目标值。模型图来自全y=0盲cache，C的独立V通过稳定sample ID附加并逐batch核验。L333归一化尺度固定。

`ANALYSIS_PLAN.json`在无已完成fit时冻结主模型/seed/门槛；`ANALYSIS_SUPPLEMENT.json`记录metadata QA触发的series等权与剔除validation关联series描述性敏感性，声明时已完成1 fit，未查看其比较结果、未改变主门槛。预训练数据门、共享初始化和RNG hash在`DATA_FEASIBILITY.json`，冻结三训练源码未变。执行环境见`environment.json`。

540分母包含1个既有授权人工ID修正例外，严格原答为539；127前缀均属于精确原答子集。请求文件保存850份，Historical270份由实际版本/消息重建且hash匹配；不能把重建记录当作独立保存HTTP wire日志，也不能据本地上下文证明provider内部行为。

## 复现与验证

实际解释器：`/Users/fish/miniforge3/envs/chromatography/bin/python`。所有命令以仓库根目录运行并使用`-B`；线程环境见`run_tests.py`。分析脚本不发API、不训练：

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/evaluate_controls.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/report_controls.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/write_overview.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/verify.py --check
```

零训练A2由`sensitivity.py`、`summarize_sensitivity.py`复现，逐分子表由`summarize_molecules.py`汇总；C由`retry_context_audit.py`后接`finalize_retry_audit.py`、`local_repair.py`复现。首次拟合前的`prepare_data.py`会构造新数据门，本次已经完成并锁定，不应在已有fit时重写。`train_controls.py`只允许预定15组：已有完成receipt会复用，存在未完成start则拒绝重启，不自动追加或重试。

`run_tests.py`已完成28项测试，exit code=0，本轮未因门槛结果增加任何拟合。`VERIFICATION.json`包含输入/输出SHA、15 fit receipts、训练配置/选择核对和边界说明；`verify.py --check`只复核封存身份。若重跑生成脚本，输出可能重写，应保留原封存或重新明确封存版本，不用新seal掩盖历史输入变化。

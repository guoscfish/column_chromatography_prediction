# NoCW失效机制归因与模型可学习性验证

本研究复用2026-11-12综合诊断，新增NoCW自身科学验证链、条件路径/敏感性诊断，以及唯一预定义的固定标签模型对照。核心阅读入口是[RESEARCH_DECISION.md](RESEARCH_DECISION.md)。没有付费LLM、新AL轨迹或正式选择；模型试验仅限用户授权的四状态、两模型、两训练重复。

| 研究问题 | 交付 |
|---|---|
| NoCW真正缺陷、低收益及持续收益 | [NOCW_PROBLEM_AUDIT.md](NOCW_PROBLEM_AUDIT.md) |
| 假设→证据→实验→观测→下一假设 | [HYPOTHESIS_TO_LEARNING.md](HYPOTHESIS_TO_LEARNING.md) |
| 五条件编码、真实差分、冻结前向/梯度 | [CONDITION_LEARNABILITY_AUDIT.md](CONDITION_LEARNABILITY_AUDIT.md) |
| 唯一模型改动、H隔离、训练/数据冻结 | [MODEL_PILOT_PROTOCOL.md](MODEL_PILOT_PROTOCOL.md) |
| 所有pilot结果、波动与负面取舍 | [MODEL_PILOT_RESULTS.md](MODEL_PILOT_RESULTS.md) |
| E1/E2因果协议、费用与只读mock | [NEXT_AL_EXPERIMENT_PROTOCOL.md](NEXT_AL_EXPERIMENT_PROTOCOL.md) |
| H1/H2/H3判断与下一阶段选择 | [RESEARCH_DECISION.md](RESEARCH_DECISION.md) |
| 输入锁/最终核验 | [INPUT_MANIFEST.json](INPUT_MANIFEST.json)、[verification/final_verification.json](verification/final_verification.json) |

## 范围与数据角色

主NoCW比较是五seed（157、887、2357、6101、12203）共同L333–1005，共105批。额外18批仅用于后续累计收益与科学链可用性，不能拼成完整五seed晚期结论。科学假设主527条；全123批620条的primary标志逐行保留。既有综合报告、AULC、行为和geometry表直接复用，没有重新生成14份报告。

条件诊断复用CW893对（上样V165对），新增NoCW377对（V41对），把“同主动学习获取批/同checkpoint”与未知实验室批次分开。四冻结模型从实际候选X构造合法条件对，未查询未采集真值。局部自动微分仅作软件诊断，不把任意连续扰动解释为合法化学实验。

模型pilot固定最小两outer seed157/887、预算L525/L1005、两模型和两个训练replicate，上限16成功fit。H从postL333已采集数据按元数据构建，10个端点不共享的连通组，V优先且至少2分子；端点隔离不代表统计独立。L333、train、validation同记录条件组与H无交叉。v1检查中发现两处validation同条件重合，在训练前保留v1原件、记录修订、按X排除后重建v2；没有用响应或结果挑组。实际训练样本数为492/958/490/960，不能当原L525/L1005同预算曲线。

输入响应来源是合法observed/accepted反馈。原始canonical CSV只用于指定X列，以及训练时先冻结的416个validation ID allowlist；其他target字符串不数值化、不输出、不分析。test与H图在训练过程中为零sentinel；H仅终点评价，不能选checkpoint或模型结构。不同架构只有一个无额外参数的条件交互融合，所有原损失、预处理与训练预算保留。

`runtime/`和`private_cache/`在本目录.gitignore中排除，权重、逐行H预测/响应和训练日志留在本地；公开交付是聚合统计、授权少量科学病例、ID/来源/hash。原始packet/catalog/完整请求不复制到交付。所有可见标签都绑定方法、seed、获取轮次，不能作为另一状态的选前真值。

## 脚本与表

| 模块 | 计算/核验脚本 | 核心派生表 |
|---|---|---|
| A/B | `nocw_hypotheses.py`、`review_nocw_science.py`、`verify_nocw_hypotheses.py` | `nocw_hypotheses.csv`、`nocw_test_pairs.csv`、`nocw_gain_cases_with_behavior.csv`、`nocw_scientific_chain_adjudications.csv` |
| C | `condition_mechanism.py`、`verify_condition_independent.py` | `condition_nocw_pair_*`、`condition_frozen_probe_summary.csv`、`condition_gradient_summary.csv` |
| D | `model_pilot.py`、`test_model_pilot.py`、`summarize_model_pilot.py` | `pilot_state_qualification.csv`、`pilot_fit_results.csv`、`pilot_H_group_metrics.csv`、`pilot_*paired*` |
| E | `next_al_precheck.py`、`verify_next_al_independent.py` | `next_al_precheck.csv` |
| 封存 | `finalize_attribution.py` | `INPUT_MANIFEST.json`、`verification/output_manifest.json` |

现有环境：`/Users/fish/miniforge3/envs/chromatography/bin/python`，使用`-B`避免向历史目录写pyc。numpy/pandas/scipy/torch/torch-geometric/RDKit均为本地已有依赖，没有下载模型或安装新库。正式重放前先验证源锁：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/nocw_failure_attribution_20261112/scripts/finalize_attribution.py --check
```

重新计算A/B/C/E分析需原本地hash绑定的input/catalog/feedback/缓存；缺失时应停止并报告，不重建新标签。D的脚本采用write-once协议冻结和不覆盖已有fit的执行槽，上限16；不要以复现为由删除runtime后追加训练或扩成60fits。未来重训需要新授权/新版本目录。本次已有成功fit应只读复核。

## 可信度与未解决问题

NoCW假设有单因素测试结构，不代表其宽泛科学命题已被否证或证实；人工语义样本52批也不是527条命题的全量正确率。3152引用时间合法不等于语义归属正确。gain与before NRMSE共享分量，控制后相关仍非因果；后续累计转正不能归功于最初那批。

两outer seed、两训练重复只支持机制/工程筛查。交互改动也改变梯度conditioning，不能由一个正结果唯一识别表达容量；负结果也不否定全部模型可学习性问题。H有少量训练中完全未见分子，混合条件插值和分子外推，分层解释见结果。实验室复测身份/未记录变量缺失，不能把相同条件差异直接当不可约噪声。

E1/E2只完成数据边界和mock，未产生真实新提案、LLM效果或AL因果收益。固定置换在3/4状态造成明显预测—反馈冲突，已降格压力测试；未来噪声资格不能直接借用L_t\H的pilot波动代替L_t+B上的交叉训练重复。原始历史开发集被反复使用，当前试验仍非独立确认。

输入/原源码、旧诊断和冻结模型的哈希检查，以及最终Git状态在verification保留。研究停止于规定范围，本任务没有改生产runner、历史权重或既有分析。长时间训练期间工作树出现其他实验/报告目录的并行新增，本任务保留这些文件，不读取其标签、运行其runner或纳入分析；最终只清点路径，另记于`verification/concurrent_workspace_changes.json`，不能把全工作树新增都归为本轮行为。

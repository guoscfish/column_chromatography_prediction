# 实验与已有研究复用清单

统一输入以2026-11-12完整CW归档为锚点，其他比较使用已完成的真实保存状态。源文件没有更新；新目录的source manifest记录SHA。目录级全表见 `tables/study_directory_inventory.csv`，运行级见 `run_registry.csv` 和 `additional_run_registry.csv`。

## 主比较族

| 方法 | seed数 | 实际终点 | 主公平窗口 |
|---|---:|---|---|
| CW-visible Free32 | 5 | 全部1485，180批 | 与raw LCMD/旧Free至1485；六方法至1005 |
| Historical Free32（混合模型） | 5 | 全部1485 | 同上；另看3个从头Sol seed |
| Gradient-LCMD32（raw） | 5 | 全部1485 | 同上 |
| NoCW32 | 5 | 1069/1005/1133/1165/1229 | 五seed1005，禁止补齐1389 |
| Numerical Hybrid32 | 5 | 全部1005 | 五seed1005 |
| Random32 | 5 | 157/6101=1549，其余1005 | 五seed1005；二seed扩展另列 |
| Free16＋CW-LCMD16 | 2 | 157=1485，6101=1453 | 与同两seed比较至1453 |
| CW-gradient LCMD32 | 2 | 1549 | 与同两seed比较，至多CW新run1485 |

新增补充数值族：Gradient-MaxDet五seed实际共同909（其余分别973/1005，不将全部写成1005）；U50-MaxDet和gradient-latent fusion两seed1005；a080 fusion screen、CW/IVR portfolio两seed653。保存fit摘要与共同窗口配对差，不再次执行其训练/报告入口。Kernel-IVR、LCMD→IVR及same-state分支的机制结论复用既有报告；本次未把缺乏统一逐点来源的新“曲线”拼入主比较。

## 研究问题—旧证据—新工作

| 分析问题 | 已有证据 | 当前缺口 | 本次处理/是否新增计算 |
|---|---|---|---|
| 新CW终点与全程AULC相反 | `reports/free_cw_final_20261112/`已给终点/AULC | 哪个阶段造成差额、每seed首达和波动 | 新增分阶段面积分解、阈值、gain、训练元数据联合 |
| Free与raw LCMD行为 | `free_lcmd_mechanism_diagnostics_20261029/`全180旧Free | 不能代表新CW独立轨迹 | 复用旧结果；新增CW180，比较结构/条件/几何 |
| NoCW更分散/中期局部对照下降 | `nocw_free_selection_diagnostics_20261101/`固定快照 | 旧Free混模型；新CW中是否仍成立 | 复用883条旧geometry；新增CW360条raw/CW及NoCW缺轮 |
| 条件对照与收益 | 旧Free correlations不稳定 | 最新CW完整轨迹、联合状态控制 | 新增seed内Spearman/partial rank、seed bootstrap与反例 |
| 科学证据是否合法/正确 | `verifiable_scientific_agent_20261030/` | 最新CW180全量引用及纠错后证据丢失 | 新增871假设、5455结构引用、430完整响应审计 |
| 科学记忆是否越来越偏异常点 | 旧Free近期失败邻近有富集 | 新CW全程动态、反例 | 新增引用集中度/近期反馈/修订指标；不当作科学正确率 |
| 分位宽度可靠性 | `diagnostics/quantile_loss_audit_20261109/`L333 | 后续预算是否可复核 | 新增5760次合法已选反馈事件，选前原始quantile绑定；与旧validation样本区分 |
| 化学条件误差、上样量弱响应 | 旧冻结几何/GP条件对诊断 | 同一checkpoint上的实际条件差分 | 新增893对，其中165上样体积对；已选条件子集，不代表全U |
| GP/IVR是否可直接做critic | `gp_guided_llm_stability_20261103/`、`representation_gp_benchmark_20261104/` | 校准/任务增益资格不足 | 复用No-Go与反证，不重跑数百次GP |
| GOLLuM/语言表示融合 | `gollum_adaptation_fusion_20261106/`多阶段 | 已发现概率失配、收益极小 | 复用正式负面结论，不换名推荐相同复杂融合 |
| 执行稳定性与费用 | 最新costs、lossless receipts、旧retry审计 | 物理尝试与有效纠错区分 | 新增分batch/seed的完整响应token统计、语义失败、合法点流失 |
| 训练波动来源 | 固定scratch/RNG协议，quantile loss audit | 相同L/checkpoint规则的独立重复拟合缺失 | 只做训练元数据关联；不声称分离训练随机方差 |

## 可得与不可得的历史材料

新CW五个gzip JSONL科学归档保存180个contract/selection/feedback/label receipt/batch freeze及185个拟合状态、history、轨迹冻结。读取时校验原始UTF-8哈希。模型和原始数值array仍local-only；本地180轮候选catalog与raw/CW几何已找到，起点若自身无bank，仅在L/U、checkpoint与映射完全一致后复用共同L333 bank，并显式标记。后续绝不以其他状态替代。

旧NoCW分析冻结包含无响应元数据与历史选择；本次利用其身份映射避免读取原始标签CSV。选后反馈的事件不等于独立row：同一个row可在不同seed被合法获得。Error报告给出事件数、独立row与分子数。验证逐行预测并非全预算保存；model目录的417行预测是test_X预测冻结，不能拿来做新的validation化学区域分解。

## 不作为本次新结果的材料

早期3c/4g predictor版本、compound split、8g transfer、oracle效用、不同batch screen以及未完整来源核对的LLM scientist v3分支，只列为背景或目录记录，不把它们并入QGeoGNN-V2 Row B32统一排名。历史test报告存在不构成本次读取test真值的授权。既有全文报告中的test结论不用于本次新增指标。

所有复用结果在专题报告注明“既有证据”；所有新增量化结果来自本目录scripts/tables。没有把重算同一个终点数字称为新发现；新的贡献在跨阶段、跨行为、反馈语义和模型响应的联合约束。

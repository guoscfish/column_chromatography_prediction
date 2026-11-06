# 纠错前后几何冗余：已有冻结 bank 的补充审计

本补充为 Phase A3 提供之前留空的几何比较。只读原研究固定快照中的114个纠错轮、其完整首响应与最终 selection、L/U/批次冻结记录、无目标列的身份划分表，以及已经存在的 raw/CW gradient bank。没有训练、LLM、模型推理、梯度重算、预测真值表或新标签访问。原 `repair_trajectory_analysis.csv` 的 NA 不重写；补充结果分别保存，便于保留原审计来源。

## 覆盖与分母

| 方法 | 纠错轮分母 | 可比较的同方法 bank | 首提案合法唯一子集平均行数 | 最终 LLM 行数 |
| --- | ---: | ---: | ---: | ---: |
| Free | 68 | 68 | 29.515 | 32 |
| Hybrid | 20 | 20 | 14.700 | 16 |
| NoCW | 26 | 23 | 30.478（23个可比较轮） | 32 |

共111/114轮有可验证的同方法、同round、同checkpoint、有序L+U一致的bank；raw/CW均可用。另提供Hybrid的固定pending16+LLM视图：首提案平均30.7行，最终32行，20轮。它与LLM组件视图来自同一批，不能当作另20个独立实验。268行CSV包含两空间、两阶段比较列、Hybrid额外视图和6行NA，不是268个独立实验。

3个NA是NoCW seed2357/6101/12203的round0：其本方法runtime未保存banks。本补充采取同方法bank规则，没有为了凑齐覆盖而从Free初始bank代入；没有将这些点的几何值猜填。缺失理由和检索过的同方法metadata来源写入summary.bank_audits。Free157 r30、Free6101 r34、Hybrid6101 r21的bank在相邻继承revision，只有checkpoint与完整有序canonical索引完全匹配时才使用。

“首提案”是冻结完整首响应中**合法且唯一**候选的子集：非法ID被排除，重复真实ID只保留一次，没有把任何错误ID映射到其他实验。脚本重新解析已冻结首响应，并核对子集与原审计CSV相同；不从自由文本提取ID。114个首提案在此口径均可分析。首批合法子集仍可能带有非法证据或其他schema问题，这里只分析可确定实验的几何，不将它重新宣称为合法完整科学计划。

## 指标与来源核验

`repair_geometry_analysis.py`只通过AST执行既有Free几何脚本的`squared_distances`和`effective_rank`两个纯函数。逐组选点到当轮L的最近距离、批内最近邻、NN均值/到L距离均值、NN<到L及NN<0.5×到L的行比例/计数、精确特征重复、中心化有效秩/participation rank和当前bank RMS定标均沿用既有定义。只计算选中子集到L，没有重复计算全U覆盖或新采集分数。

两组使用完全同一个选前bank和L。Hybrid主表只分析LLM组件；额外全批视图在两组中都添加同一个冻结pending16，pending不冒充已测L。距离分子分母均为该视图实际合法唯一点数；两点以下的最近邻、零分母的比值留null。批次有效秩受行数上限影响，最近邻也会随补点机械改变。

来源链检查包含：历史snapshot中contract/batch_freeze/selection/首响应的原字节SHA；划分表与冻结NoCW初始协议/既有审计的SHA；仅用train pool的sample_id→canonical_index映射验证bank全部有序L+U；metadata的checkpoint等于选择合同；NPZ字节SHA等于该bank metadata，且raw/cw均为(3330,512)有限特征。新读取来源的SHA全部记录在summary.source_sha256。已有来源哈希清单没有登记本次命中的bank路径，因此不冒称这些NPZ都受旧batch_freeze原字节哈希覆盖；本补充保留本次独立哈希，并与已发布的最终几何指标做数值复核。

## 实际结果

下表是raw空间、LLM组件、最终减首提案合法子集的方向计数。正负判定容差固定为1e−12，仅用于消除输出级舍入；没有按validation误差调节阈值。

| 方法 | NN/到L比值：升/降/同 | NN<0.5×到L比例：升/降/同 | 有效秩：升/降/同 |
| --- | --- | --- | --- |
| Free（68轮） | 18 / 45 / 5 | 7 / 26 / 35 | 50 / 14 / 4 |
| Hybrid LLM（20轮） | 3 / 11 / 6 | 3 / 1 / 16 | 9 / 5 / 6 |
| NoCW（23轮） | 2 / 16 / 5 | 2 / 8 / 13 | 13 / 5 / 5 |

多个指标的方向并不等价：NN/到L比值多下降，但低NN行比例多数不变或下降，有效秩多数上升。CW的对应方向与raw大体接近，逐空间数值保留在CSV/JSON。不能将其中一个数值挑出来宣称“纠错系统性增加了冗余”。

只有Free6轮、Hybrid6轮、NoCW5轮首末合法唯一点数相同。在这17个等行数比较中，raw NN/到L比值除Free1轮上升外均不变；有效秩仅Free2轮下降，其他均不变。等行数小样本也受内容/是否仅修证据影响，不能视为随机对照。summary.equal_count_metric_changes保留全指标敏感性计数。

有些首合法子集集中于到L距离极小的条件点，NN/到L的分母因而接近零，导致Free/Hybrid该比值的平均变化出现很大幅度；**平均比值不适合作为主结论**。CSV保留原始距离均值、RMS定标、分母行数；JSON同时保留中位差和方向计数。raw比值的中位变化分别约−0.162（Free）、−0.0178（Hybrid LLM）、−0.0806（NoCW）。低距离本身不能说明该实验无效，尤其需要与严格条件对照同时解释。

本补充支持的结论是：完整再生成经常改变实际几何组成；现有结果没有提供纠错统一恶化几何冗余的证据。首末子集数量和组成不同、修复轮非随机、相同seed/历史内轮次相关，均阻止因果解释。没有显著性检验、科学价值判分或AULC收益结论。

## 运行与轻量核验

```bash
/Users/fish/miniforge3/envs/chromatography/bin/python \
  studies/active_learning/verifiable_scientific_agent_20261030/repair_geometry_analysis.py
```

当前完整运行约9.1秒；耗时只代表本机114个已有bank的读取与子集计算。内置已知距离/比值、rank1、零分母和单点最近邻检查通过；每轮首合法子集从完整冻结答案独立重建并一致。最终方案的10项共有几何指标还与已发布Free及可用Hybrid几何CSV逐项复核，实际核对数量、最大差异与1e−10绝对/相对容差见summary.verification.existing_final_profile_metric_parity。没有新增外部调用或正式实验，也没有修改原审计CSV或旧研究文件。

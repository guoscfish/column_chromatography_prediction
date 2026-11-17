# NoCW 的真实问题：证据归属与低收益病例

## 结论与范围

NoCW最可靠的缺陷是**相对强数值基线的整体学习效率不足、部分科学命题难以形成明确验证决策、身份/整批纠错的方案保真不足**。这些不能都归因于缺少CW。最新CW-Free五seed在共同L333–1005的AULC反而全部高于NoCW；模型条件不敏感也不是NoCW独有。

复用上一轮综合诊断的性能、行为、几何和收益表；新增的是NoCW科学链。主范围五seed157/887/2357/6101/12203、每seed21批共105批；额外18个已完成批仅用于后续累计gain和next-hypothesis的可用性，不能组成完整五seed晚期比较。记录527条主范围假设、3152次typed引用、229个完整响应；共123批620假设被读取。没有读取原始标签CSV、test真值或未采集候选真值。

## 八类问题的归因表

| 问题 | 具体证据与方法/seed/预算 | 是否NoCW独有/新CW支持 | 证据等级与竞争解释 | 最小排除方式 |
|---|---|---|---|---|
| Row学习效率 | 五seed333–1005 NoCW AULC0.497741±0.065443，raw LCMD0.462549、数值Hybrid0.438420；NoCW终点0.405025 | 非独有；新CW AULC0.532876更差但1005终点更好；旧Free混模型 | 保存曲线事实；非同状态CW因果效应 | 同状态NoCW/Placebo/Critic/raw LCMD |
| 分子/条件行为 | 早期新分子名额NoCW27.81% vs新CW17.60%；中期批内单因素22.08% vs21.94%；813–1005 NoCW10.10% vsCW25.42% | 中期旧差异未在新CW重现；后期差异存在但异质 | 行为事实；批次分散、单因素匹配不等于信息增益 | 事前科学命题与真实设计操作数绑定 |
| 数值覆盖 | NoCW raw覆盖随机分位中期0.5028、813–1005约0.4104；新CW0.5153/0.5167；LCMD较高 | H2有线索但不独有，新CW中期也低 | 相对自身状态随机batch的代理，不是共同真实效用 | Critic提高几何后仍需真实gain胜Placebo/数值baseline |
| 假设可证伪性 | 527假设中305能找到关联的严格单因素测试结构、222找不到；固定样本常无明确拒绝阈值 | 新CW也有宽泛scaffold-specific命题；不能用结构缺失断定无科学价值 | 全量图结构＋52批首假设人工审查；跨分子设计可能不需同分子控制 | 明确范围/替代模型/预期差分后前瞻验证 |
| 反馈吸收 | 88/105批typed引用上一批新观测；m39预测反转被后轮明确否定，多个具体修订例 | 非独有，新CW也有真实修订；尚无反馈因果增益 | 引用/修订实例，不等于每条推理正确 | E1真实/隐藏/扰动反馈，重复提案校准随机性 |
| 证据/实体 | 3152 typed引用全属于选前L，但38分子flags人工确认11误写、27合法跨分子参照；2未知prose ID | 新CW也有25误写下界；并非NoCW独有 | 11为检出下界，不是所有命题的错误率 | typed row→canonical分子→变化轴核验 |
| 整批纠错 | 共同105批NoCW首完整答合法37/105、229完整响应；68纠错批，合法ID仅583/2002保留 | 新CW共同窗口50/105、309完整响应；全程95/180只作背景；旧Free/16+16均有漂移 | 首合法率较低但完整回答反而更少；CW极端packet绑定重答影响总量。不能判科学质量或总费用更差 | 确定性身份处理＋保留合法slot的等预算对照 |
| 条件信息吸收 | 前研究新CW165上样对真实Δ远大于预测；本轮C/D另做NoCW/冻结模型与小试 | 属于QGeoGNN/数据的共同候选瓶颈，不能凭CW结果认定NoCW每批如此 | 未识别编码、优化、噪声份额；需参考本轮条件审计与pilot | 固定已获标签，隔离H，唯一模型对照 |

## 性能与NoCW失效叙事的反证

| method | aulc_mean | aulc_sd | endpoint_mean | endpoint_sd |
| --- | --- | --- | --- | --- |
| gradient_lcmd32 | 0.46255 | 0.06739 | 0.36633 | 0.03841 |
| historical_free_mixed_model | 0.50880 | 0.08216 | 0.34986 | 0.03014 |
| new_free_cw_sol61_xhigh | 0.53288 | 0.07254 | 0.37875 | 0.04234 |
| nocw | 0.49774 | 0.06544 | 0.40503 | 0.04622 |
| numerical_hybrid32 | 0.43842 | 0.06727 | 0.35192 | 0.03988 |

上述各方法是五seed相同窗口，均值±样本SD。NoCW在AULC上优于历史Free及新CW，却在1005终点落后，不能简单称“去CW失效”。旧Free与NoCW的中期覆盖/局部对照差无法直接推出CW因果效应。NoCW五seed有30/105批gain<−.001，raw LCMD与新CW也各30/105；锯齿不是LLM特有故障。

### 纠错可靠性也使用共同窗口

| method | window | batches | first_complete_legal | first_legal_fraction | complete_responses |
| --- | --- | --- | --- | --- | --- |
| NoCW | L333-1005 matched five seeds | 105 | 37 | 0.35238 | 229 |
| new CW-Free | L333-1005 matched five seeds | 105 | 50 | 0.47619 | 309 |
| new CW-Free | L333-1485 full trajectory; contextual only | 180 | 95 | 0.52778 | 430 |

NoCW首完整答合法率35.24%，新CW共同窗口47.62%，但完整回答总数NoCW229、新CW309，方向相反。新CW共同窗口有115份完整回答首先因packet绑定不符被拒绝，极端单轮重答支配了总量；这些不是115个独立科学判断案例。CW全程95/180（52.78%）、430份回答保留为原完整轨迹证据，**不与NoCW共同105批的分母直接暗比**。上述事实既不能说明NoCW科学质量更差，也不能由完整回答更少认定总费用更低；token、失败传输、服务商账单和模型状态仍不同。

## 按规则选取病例，不以病例确认规则

规则先固定于 `verification/nocw_analysis_plan.json`：每seed×三阶段r0–5/r6–14/r15–20分别取最大gain、最小gain、最接近0；同值取早轮，共45个规则位置（可重合）。最小gain若仍为正，不称恶化。eps=.001仅区分描述类别。所有病例属于按结果挑选的解释样本，不能作为新的独立确认集。

未来2/3批累计分别保留：仅后续gain之和，以及当前＋后续gain之和。已有2/3批总窗也另列，避免“后续”歧义；缺失显示右删失，不补值。来源是已完成真实轮次，当前单批与后续batch共同形成变化，不能归因于最初那批。

| case_rule | n | actually_worsened_gt001 | near_zero_001 | positive_current_plus_next2 | next2_censored | positive_current_plus_next3 | next3_censored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_gain | 15 | 0 | 0 | 15 | 0 | 14 | 1 |
| minimum_gain_case | 15 | 13 | 0 | 10 | 0 | 11 | 0 |
| near_zero | 15 | 7 | 5 | 13 | 0 | 12 | 0 |

15个最小gain位置中13个实际恶化，但当前＋后续2批有10/15累计为正，当前＋后续3批有11/15为正；不能把单轮下降直接判为无效实验。15个“最接近零”位置仅5个落入|gain|≤.001，名称是排序规则，不是实际零收益判定。

完整45位置见 `nocw_gain_cases_with_behavior.csv`，同时提供每seed预算、单轮/后续gain、最大分子名额、局部对照、raw几何随机分位及科学链结构。它们没有一致表现为“围住一个分子反复测”：NoCW每批主要覆盖多分子，单分子最大占比较小；高/低/近零收益病例都可能有严格条件测试，也可能缺少。

按seed×阶段平衡的15个高gain/最小gain位置，单分子最大名额比例均为0.07083；raw覆盖分位高gain0.54167、最小gain0.55833，甚至最小gain组略高。相反，hypothesis关联测试结构的比例分别约0.6587/0.4158。这些是按结果选取后的描述，不是新的独立效应估计；分子“家族”比单分子更宽，不能从上述比例排除家族偏好。


### 病例分子族补检：较低收益更集中，但接近候选池本底

| 固定病例组 | 批次max share | 批次HHI | 当轮U max share | 当轮U HHI | 批次HHI−随机32期望 |
| --- | --- | --- | --- | --- | --- |
| high_gain | 0.52917 | 0.33555 | 0.60714 | 0.39018 | -0.07347 |
| minimum_gain_case | 0.60625 | 0.40768 | 0.60820 | 0.39136 | -0.00248 |
| near_zero | 0.58750 | 0.39232 | 0.60900 | 0.39238 | -0.01883 |

本表只在既定45个病例位置（41个唯一批次）上补充metadata-only Murcko指标；每seed三阶段等权，再对五seed等权，不重选病例。最小gain−高gain的HHI差在4/5seed为正，seed157反向−0.04297；near-zero−高gain的HHI差也为4/5正，但6101反向。因批次只有32点，原U的HHI不是随机32点的期望，另按无放回抽样的精确公式校正：高gain相对随机32的HHI差−0.07347，最小gain−0.00248。这更像高gain病例避开了候选池的主导核心，而不是低gain普遍超常锁定单一族；低gain的最高核心占比0.6063几乎等于其U的0.6082。前至多3个实际批次scaffold重复名额相对U的富集，高/低组约+0.0101/+0.0401，但低−高仅3/5seed为正，157和12203反向，不构成一致的持续锁定证据；首轮无历史者保持缺失，这一项仅在有历史的状态内平衡。完整45行及三组汇总见 `nocw_case_family.csv`、`nocw_case_family_summary.csv`。Murcko的c1ccccc1将醛、酮、酯等许多不同官能团环境归成泛苯环，空scaffold又合并全部无环分子，不能由本结果认定某个化学功能家族过采样；这些按gain选出的相关现象也不能导出新acquisition规则。来源单锁在 `verification/nocw_case_family_inputs.json`，未读新标签或改变H/模型。


**必要解释限制：** 后续累计从负转正可能只是后来的其他实验起效；单轮负gain不能直接称标签浪费。反之，科学理由看起来正确也不保证整批32点训练后泛化改善。判定某一具体命题是否被验证，见下一份报告的真实链；把比例、覆盖或新分子数直接叫学习价值没有依据。

## NoCW数值代理与学习收益的旧结果复用

| feature | mean_seed_correlation | positive_seeds | negative_seeds |
| --- | --- | --- | --- |
| new_molecule_slot_fraction | 0.23930 | 5 | 0 |
| raw_gradient_reference_coverage_relative_drop_random_percentile | 0.01037 | 2 | 3 |
| raw_gradient_selected_to_l_distance_mean_random_percentile | -0.06115 | 2 | 3 |
| within_any_single_selected_fraction | 0.20563 | 3 | 2 |

这是旧综合分析已经计算的NoCW控制预算/阶段/before NRMSE的关联，未重复生成。代理方向并不普遍稳定；gain和before共享分量，且同validation选checkpoint，不能将控制后的相关当因果。H2目前支持到“缺少显式诊断可能限制某些状态”，不能升级成“加critic必然提高AULC”。

## 可信度与输出边界

本次以实际NoCW五seedwrapper的validator纯函数AST重放，不导入runner或label store。全部接受批次32IDs与冻结及feedback一致；input/catalog/turn按batch_freeze原SHA核对，feedback绑定batch_freeze；observed=L、candidates=U，候选没有真实响应列。时间合法性全量通过。完整packet和原始模型回答不复制到交付；少量科学病例摘录来自已接受反馈并有ID/轮次来源。

主要竞争解释H1/H2/H3尚未被彼此排除：科学命题和几何诊断可有不足，但模型不敏感会限制科学标签转化，训练/验证检查点选择亦可改变gain。下一步依据本轮C/D的结果再决定E1或E2，不能仅用本表拼接新Agent。

复现与核验：`scripts/nocw_hypotheses.py`→`review_nocw_science.py`→`verify_nocw_hypotheses.py`；在仓库根用已有chromatography Python和`-B`运行。`nocw_inputs.json`锁定1012项源；固定20批独立枚举154pair、累计gain望远镜和、76个partial的null-space复算通过。先前59批审计精确复现33首答合法/99完整响应，说明本次重放与旧真实规则一致。缺失未来轮次保持右删失，未补跑。

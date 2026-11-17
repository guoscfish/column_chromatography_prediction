#!/usr/bin/env python3
"""Render Chinese interpretive reports from the locked derived tables."""
from pathlib import Path
import json
import pandas as pd
OUT=Path(__file__).resolve().parents[1]
def load(n):return pd.read_csv(OUT/'tables'/f'{n}.csv')
def md(d):
 def cell(x):
  if pd.isna(x):return 'NA'
  if isinstance(x,float):return f'{x:.6f}'
  return str(x).replace('|','/')
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def write(n,s):(OUT/n).write_text(s.strip()+'\n')
def main():
 stage=load('stage_summary');pair=load('paired_effect_summary');runs=load('run_registry');gains=load('batch_gains')
 common=stage[(stage.start==333)&(stage.end==1005)&(stage.seeds==5)][['method','seeds','aulc_mean','aulc_sd','endpoint_mean','endpoint_sd']]
 finals=stage[(stage.start==333)&(stage.end==1485)&(stage.seeds==5)][['method','aulc_mean','aulc_sd','endpoint_mean','endpoint_sd']]
 p=pair[(pair.comparator=='gradient_lcmd32')&pair.apply(lambda r:(r.start,r.end) in [(333,525),(525,813),(813,1005),(1005,1485)],axis=1)&(pair.metric=='aulc')]
 p=p[['start','end','mean_delta','sample_sd','negative_deltas']].copy();p['contribution_to_full_delta']=p.mean_delta*(p.end-p.start)/1152
 per=load('paired_effects');per=per[(per.comparator=='gradient_lcmd32')&(per.start==333)&(per.end==1485)].pivot(index='seed',columns='metric',values='delta_cw_minus_reference').reset_index()
 two=load('stage_metrics');two=two[(two.start==333)&(two.end==1005)&two.seed.isin([157,6101])].groupby('method').agg(n=('seed','size'),aulc_mean=('aulc','mean'),aulc_sd=('aulc','std'),endpoint_mean=('endpoint_nrmse','mean')).reset_index()
 write('LEARNING_DYNAMICS.md',f'''# 学习曲线、阶段收益与收敛诊断

## 定义与比较窗口

按固定L333分母计算row-level combined validation NRMSE。所有原始非单调点参与归一化梯形AULC，见协议审计。主八方法34条run、1088点、1054次实际已完成加样本→拟合变化；不同长度的总AULC不能直接排序。每seed值在run_registry/stage_metrics/paired_effects。以下均值±样本SD（ddof=1）衡量seed分散，不是置信区间。

![八方法原始轨迹](figures/learning_all_methods.png)

## 五seed共同L333–1005

{md(common)}

新CW全程0.532876，明显差于数值Hybrid0.438420和raw LCMD0.462549，也高于NoCW0.497741、历史Free0.508802。新CW−NoCW AULC差+0.035136，五seed方向一致；新CW−历史Free+0.024075同样五seed一致。它优于Random（配对差−0.081866），但“LLM优于随机”不足以支持协同策略优于强数值baseline。旧Free不是纯Sol统一配置；不过五seed均同向意味着不能只用两个混模型seed解释新CW早中期损失。

## L1485终点更好，全程仍更差

{md(finals)}

新CW−raw LCMD终点−0.024879（SD0.013563），相对均值约改善7.16%；全程AULC+0.037145（SD0.015703），相对恶化8.91%。两者五seed方向都一致。以下配对差均为新CW减对照，total_gain方向相反：

{md(per)}

![阶段面积与终点](figures/learning_stage_aulc_endpoint.png)

## 哪个阶段造成差额

{md(p)}

L525–813贡献+0.028408，约占全程净AULC差的76.5%；L1005–1485贡献−0.003879，只抵消较小部分。后期阶段AULC新CW0.344040、raw LCMD0.353349，4/5seed新CW较好；seed2357是该阶段面积反例（尽管其最后一个点更好）。因此不是从某个预算开始新CW在所有seed、每一轮都占优。

L813–1005新CW误差下降0.080947，raw LCMD仅0.017629；但该段AULC仍差+0.038861、五seed均不占优。这是**从较差起点追赶**，不能以更大gain宣称该阶段已获得更高标签效率。L1005–1485额外下降差+0.037303（五seed一致）从追赶延伸到终点超过；仍无法由观察轨迹识别“先前局部实验延迟生效”这一具体因果机制。

## 阈值、突变与稳定性

阈值表保留首达和连续三个检查点达标。NRMSE≤.4的首达预算，新CW五seed分别909/813/909/1069/1133；raw LCMD为717/557/813/845/845，新CW逐seed都更晚。尤其2357新CW首达909，但连续三点达标要到1005，单次穿越并不等于稳定收敛。未达标NoCW/Random保留右删失，不能以截止预算替代。

共同1005窗口，新CW105批中30批gain<−.001，6批|gain|≤.001；raw LCMD也有30批恶化、2批近零。新CW晚期75批中仍有27批恶化。eps=.001仅作描述门槛，表中附0/.005敏感性。不能以“LLM更多非单调”作无数据支持的笼统结论。最大/最小gain逐seed保留；没有挑出好看的单轮代表全策略。

![收益分布与seed稳定性](figures/learning_gain_and_stability.png)

两因素描述性平方和分解：六方法L1005 AULC的55.0%与seed主效应、39.8%与方法主效应对应；三方法L1485 AULC中seed为87.0%。该分解依赖选定方法集合，且交互剩余同时含训练集/模型状态等因素，**不是估计训练随机方差**。配对差比不配对误差条更适合当前问题。

## 两seed方法必须重配同两seed

{md(two)}

这是157/6101专门配对表，不与前面的五seed平均直接比。Free16＋CW16在共同1005的AULC为0.484421，新CW在同两seed为0.607301，历史混合16＋16是有竞争力的开发基线；不能因只有两seed而忽略，也不能据此认定固定16＋16最优。其共同终点1453，完整配对值见表。CW-gradient和Random已保存至1549；本次补入其完成状态，未机械沿用旧报告的1005截断。

## 次级数值方法

`additional_round_metrics.csv`与`additional_paired_effects.csv`补充5种数值方法。Gradient-MaxDet实际五seed共同909；其余为两seed653/1005。只使用已核实初始化/验证身份、连续32增量的fit摘要，未读取其测试报告。它们支持比较已有数值设计，不能把不同预算均值排成统一榜单。

## 研究含义与负面审查

阶段自适应值得作为可证伪假设，但不能从本图确定在线切换时点；L813或1005是本次事后解释边界，按这些验证曲线挑最优切点会重复使用开发集。下一步必须在同一冻结状态、固定短窗、相同QGeoGNN拟合与LLM输入下比较选择方案；先验固定窗口，包含数值Hybrid和原始Free。训练重复对照用于区分拟合路径波动，不以重新运行历史seed修饰曲线。
''')
 corr=load('gain_feature_summary'); c=corr[(corr.method=='new_free_cw_sol61_xhigh')&(corr.metric=='partial_rank_correlation')]
 keep=['raw_gradient_batch_nn_distance_mean_random_percentile','cw_gradient_batch_nn_distance_mean_random_percentile',
 'raw_gradient_selected_to_l_distance_mean_random_percentile','selected_cw_percentile_mean',
 'raw_gradient_reference_coverage_relative_drop_random_percentile','cw_gradient_reference_coverage_relative_drop_random_percentile',
 'new_molecule_slot_fraction','within_tanimoto_distance_mean','within_any_single_selected_fraction','cross_only_sample_volume_selected_fraction',
 'selected_error_combined_nrmse','selected_residual_asymmetry_v2_minus_v1','selected_v2_signed_width_mean_ml','memory_top8_evidence_fraction','complete_response_n']
 c=c[c.feature.isin(keep)][['feature','n_seeds','mean_seed_correlation','positive_seeds','negative_seeds','seed_bootstrap_p025','seed_bootstrap_p975']]
 write('ACQUISITION_VS_LEARNING_GAIN.md',f'''# 批次行为是否转化为模型学习收益

## 联结与分析单位

本次新增873个真实执行batch的联合表 `gain_linked_batches.csv`，包括新CW180、旧Free180、raw LCMD180、数值Hybrid105、NoCW123、Random105。新CW每个batch绑定自身选前状态、最终ID、揭示后残差和紧接的训练后validation，不使用其他seed或其他轮模型代替。已有行为表gain与独立拟合表重算gain全量差<1e−10。额外两seed混合/CW-gradient方法有性能曲线，但未用不完整行为作伪全量回归。

每seed分别报告Spearman；再对rank(feature)、rank(gain)各自回归预算rank、选前NRMSE rank和阶段指示，取残差Pearson，即partial rank correlation。排序前指标按12位小数消除机器舍入的近似并列，原值以17位有效数字保存；这不改变条件精确匹配规则。控制项只用于离线诊断，不是可交给在线selector的validation信息。分析控制可观测状态，无法控制完整训练集、拟合动力学或prompt。

gain=e_before−e_after和控制项e_before共享分量，且checkpoint用同一validation选取，存在数学耦合与回归均值。附partial_rank_without_before敏感性：只控制预算和阶段。批内raw距离和selected-to-L仍五seed同正；CW显示分位降为4/5正、完整响应数降为4/5负。这些反例说明“控制后”不能解释为已排除训练波动。

不把36连续batch当36独立重复；均值与方向按seed计，bootstrap抽整个seed的相关系数（固定seed20261112、10000次），另列leave-one-seed-out范围。仅五个且验证划分可能重叠，bootstrap区间只是敏感性描述，不是稳健人口推断。许多相关特征同时探索，不给未经多重性处理的p值，不以区间不跨零宣称新策略已成功。阶段只有6/9/6/15批，单阶段rho尤其不稳定。

## 新CW完整轨迹的状态调整关联

{md(c)}

![原始与控制后的逐seed相关](figures/gain_feature_correlations.png)

**较一致的候选行为**是批内最近邻距离、selected-to-L距离相对同状态随机批的分位，以及已提供的CW分位。它们更接近“避免完全近邻聚集/获得当前模型较不同的点”，但不能证明LCMD就是LLM的最佳约束，也不能把极低梯度距离的上样对照一律剔除。后者存在真实响应变化而模型预测近乎不变的反证，详见模型审计。

**不稳定的行为**包括新分子比例、Tanimoto多样性、单因素对照和历史上样邻居；raw/CW固定参考覆盖增量的随机分位均接近零、符号混合。覆盖在策略层面能明显区分LCMD与LLM，却不能稳定解释同一LLM轨迹中哪批学得更多。跨方法均值与方法内相关是不同问题，不能偷换。

![覆盖与每轮收益](figures/gain_coverage_scatter.png)

**选后实际残差**与gain有一定正关联但不是五seed统一；V2−V1标准化误差不对称多数为负。残差属于揭示后，不能直接用于选前acquisition或当hindsight oracle。其关系也可能包含起点误差与难度选择，不能说高误差样本必有高学习价值。

**宽度的微弱正关联与不确定性失准并不矛盾。** 选中V2宽度的partial rank在五seed同正，但这没有控制所有q50/尾部/几何共线信号，也不是误差增量预测检验。它可能在选择偏倚子集中代理响应尺度。原80%区间欠覆盖、V1宽度弱于q50基线等反例保留，因此不提升其“epistemic uncertainty”资格。

**较多完整响应/纠错与较差gain呈同向关联**，但LLM面对的困难状态、原提案问题和工程异常都可能是共同原因。不能由−0.18左右的状态调整相关推断“减少纠错必然提高AULC”；格式修复的首要收益是成本与方案保真，性能价值需独立对照。

## 特征组合与时间敏感性

`gain_exploratory_joint_profiles.csv`将新CW每seed按raw覆盖分位和局部对照比例的各自中位数分成四格，同时保留预算、起点NRMSE和格内batch数。分组阈值是事后描述，部分格样本很少、阶段组成不同；未证明“广覆盖＋适量局部对照”稳定优于其余组合。没有从这张表导出配额或权重。

逐seed逐阶段原始相关保存在 `gain_feature_seed_correlations.csv`（phase非空行）。全程五seed同向也不要求各短阶段同向；报告保留负方向，不把一个显著窗口作为普遍规律。旧Free与NoCW和数值方法均在同一表中，geometry缺项保留NA，无填补或跨状态转借。

## 必须经same-state验证的因果命题

1. 给Free反馈批内距离诊断，是否在保留合法化学对照的情况下提高未来3批AULC？需要Free、Free＋诊断、同等重答次数的placebo诊断对照。
2. 针对模型低灵敏度的loading轴安排额外局部对照，是否仅增加误差样本而不能被当前模型学会？需要固定标签下的模型条件表达对照，避免把模型瓶颈归咎selector。
3. 修复分子身份并保留合法选择/证据，是否只降低token和重提案，还是也改善预测？先在已授权历史包做确定性schema回放，再做统一LLM版本的科学记忆消融。

这三类命题分别对应选择几何、模型可学习性、证据完整性。现有相关系数不足以将它们合成一个复杂Agent并宣布有效。
''')
 costs=json.loads((OUT/'tables/cost_summary.json').read_text())
 train=load('training_costs');bs=load('cost_by_seed')
 write('STABILITY_AND_COST_AUDIT.md',f'''# 稳定性、工程可靠性与成本

## 性能稳定性不是单一来源

主指标逐seed/阶段/预算已在学习诊断中给出。L333–1005新CW、raw LCMD、NoCW各有30/105批gain<−.001；仅凭非单调无法判定LLM专属的不稳定。L1485新CW终点五seed均胜raw LCMD，却全程AULC五seed均负于它。seed6101初始更难，不能用终点跨seed差直接解释选择质量。

训练初始化、RNG和batch规则固定；相同L_t和同一实现下训练是否完全可重复，当前没有合格多次独立拟合控制。L_t变化、validation checkpoint选择、全批优化的路径与损失头不平衡都可导致变化。`best_epoch/epochs_run`可以提示机制，不能从五个outer seed分解纯训练噪声。

## 物理尝试、完整响应、纠错、接受批次

| 单位 | 观测计数 | 解释 |
|---|---:|---|
| 物理transport attempts | {costs['physical_transport_attempts']} | 本地started/terminal receipt；不是等额付费完成请求 |
| 成功完整模型响应 | {costs['successful_complete_responses']} | 含后来因格式/ID/证据等被拒绝的回答 |
| 最终接受并执行batch | 180 | 每批32个新row，全部已训练 |
| 非法完整回答 | 250 | 430−180，逻辑纠错统计的正确层次 |
| 首份完整回答合法 | 95/180（52.78%） | 不是网络首尝试成功率 |
| 有完整响应纠错的batch | 85/180 | 科学计划可能在整批重答中改变 |

自动旧costs字段 `request_count=173511` 指传输记录；`correction_requests=96808` 累计后续answer_turn下的物理尝试，**不是96808个独立逻辑纠错回答**。本次分层保存原字段并重新统计，绝不把173511次尝试等价为173511个付费LLM请求。430个(seed,round,answer_turn)键有完整响应；物理request ID/供应商计费另需发票证据。

失败receipt中HTTP429为139669，403为9282，ConnectError21911；这些传输/限流故障不是化学决策错误。完整回答首因分类：packet绑定115、候选ID107、未观测证据25、假设引用1、其他schema2；不同层的问题应分开解决。

纠错85批中，最初合法候选ID共2517，最终仅保留1004（39.89%）；71批最终集合变化、12批完全替换。54批结构化证据数下降；13批从有证据变成零，其中10批清空整个hypotheses。最终180批全部合法不意味着科学方案/记忆完整。详细人工语义归属核查见科学审计；自动validator没有验证化学命题真伪。

## Token与实际时间

完整成功响应有记录的input tokens={costs['recorded_success_input_tokens']:,}，output tokens={costs['recorded_success_output_tokens']:,}。这些包括纠错，不是只有180份最终方案的token。失败传输的服务端执行/消耗不可由本地失败断言为零。价格、币种、折扣和缓存计费未经核实，currency_cost保持null，不估算货币费用。

{md(bs[['seed','transport_attempts','successful_complete_responses','successful_input_tokens','successful_output_tokens']])}

跨并行attempt的elapsed_seconds相加不是墙钟时间。实际等待同时受限流、重试策略、并发和训练影响，不能用总秒数/180解释单批用户等待。

## 训练成本

{md(train)}

上表是主曲线member0拟合audit中的计时；包含不同预算，不能直接比较总和效率。数值Hybrid还需两个ensemble成员、数值bank也有提取成本，故表不是整套方法总计算成本；缺少统一完整计时，不声称数值法总计算必然更便宜。LLM的完整请求与传输次数独立列示。

## 问题分解

| 问题 | 实证 | 影响 | 根因是否明确 | 推荐层次 |
|---|---|---|---|---|
| 429/403/连接失败 | 173081非完整传输结果 | 时间、调度、费用不确定 | 错误类别明确，服务商内部原因未知 | 传输重试/熔断/可观测性 |
| packet hash/候选ID | 222非法完整回答首因 | 重答与合法方案替换 | 验证器拒绝原因明确 | 本地身份绑定、只修非法槽位 |
| 合法引用但归属错误 | 887r9病例及全轨迹人工flags | 假设与条件对照错误 | 具体病例明确，普遍发生机制未识别 | typed row/compound identity、科学证据检查 |
| 纠错丢失假设/证据 | 10轮清空假设 | 跨轮科学信息可能断裂 | 发生事实明确，学习损害未知 | 保留合法部分、可验证记忆 |
| 覆盖/距离与收益不等价 | coverage partial rank混合、距离更一致 | 错用critic可浪费标签 | 因果未明 | same-state选择实验 |
| loading变化模型预测很小 | 165同批单因素对 | 局部科学实验未必可学 | 模型不敏感是事实；噪声/表示/loss份额未知 | 固定标签的条件表达/损失对照 |
| 轮间预测恶化 | 三方法均30/105恶化 | 预算利用与收敛 | 纯训练噪声未分离 | 固定数据重复拟合对照 |

“更强Agent”不能解决所有层次。先修可确定验证的身份与纠错保真；几何批次诊断和模型可学习性另作实验。不将工程失败解释成LLM不懂化学，也不把科学自我纠正当成已提升泛化。
''')
 print('wrote learning, gain, stability reports')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Synthesize completed, fixed pilot and historical evidence. No fitting or selection."""
from pathlib import Path
import hashlib,json
import pandas as pd
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
INPUTS=[]
def load(name):
 p=OUT/name;b=p.read_bytes();INPUTS.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'role':'completed derived evidence used for synthesis'})
 return json.loads(b) if p.suffix=='.json' else pd.read_csv(p,float_precision='round_trip')
def md(d):
 def f(v):
  if pd.isna(v):return 'NA'
  if isinstance(v,float):return f'{v:.6f}'
  return str(v).replace('|','/')
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def main():
 done=load('verification/pilot_execution_status.json');assert done['successful_fits']==16 and done['status']=='COMPLETE'
 joint=load('tables/pilot_joint_interpretation.csv');fits=load('tables/pilot_fit_results.csv');vol=load('tables/pilot_volume_summary.csv')
 paired=load('tables/pilot_paired_validation.csv')
 verdict=load('verification/pilot_joint_interpretation.json');science=load('verification/nocw_summary.json')
 assert len(joint)==4 and len(fits)==16 and len(paired)==8 and science['primary_batches']==105
 n_better=int((paired.validation_combined_normalized_rmse_delta_interaction_minus_baseline<0).sum())
 n_worse=int((paired.validation_combined_normalized_rmse_delta_interaction_minus_baseline>0).sum())
 rep_pattern='；'.join(f'rep{int(rep)}为{int((g.validation_combined_normalized_rmse_delta_interaction_minus_baseline<0).sum())}/{len(g)}改善' for rep,g in paired.groupby('replicate'))
 compact=joint[['seed','available_budget','actual_train_rows','baseline_nrmse_mean','interaction_nrmse_mean','paired_delta_mean','row_relative_change','baseline_Q','interaction_Q','Q_delta_interaction_minus_baseline']]
 if verdict['row_criterion_pass'] and verdict['volume_Q_criterion_pass']:
  model_judgment='这一个固定融合控制获得方向一致的Row与局部差分联合线索。H3比单纯历史关联多了一层配对干预支持，但两outer seed开发筛查、事后描述判据与同一数据限制仍在；不能据此部署或宣称已找到NoCW失效的唯一原因。先独立确认并固定learner，再比较选择器。'
  priority='E1';priority_reason='先固定有希望的learner后，真实/隐藏反馈更直接检验可学习信息的显式通道；不要同时替换learner和selector。'
 elif verdict['volume_Q_criterion_pass'] and not verdict['row_criterion_pass']:
  model_judgment='局部上样差分有较一致的改善线索，整体Row未获得稳定支持。这直接提示局部条件目标与全局预测目标可能存在取舍；它不能证明局部实验有错，也不能说明扩大条件响应就会提高主动学习效率。当前改动不具备替换原learner的资格。'
  priority='E2';priority_reason='若继续投入的首要目标是稳定Row效率，应先检验数值诊断是否帮助对齐整批预测价值；仍保留Placebo与raw LCMD，不能假定critic必胜。E1更适合作为随后明确科学反馈通道的机制实验。'
 elif verdict['row_criterion_pass'] and not verdict['volume_Q_criterion_pass']:
  model_judgment='四状态的两rep均值呈Row改善，但预期的上样差分机制未得到稳定支持。必须保留这个预测正结果，不能将其写成两类结果皆无支持；改善也可能来自优化或隐式正则化。当前不足以部署、确认条件学习机制或扩展结构搜索，未来若追踪这个learner，先做独立的冻结确认。'
  priority='E1';priority_reason='显式反馈贡献仍未被识别，真实/隐藏是更直接且更低成本的机制比较。保持原learner实施选择器比较；若另行确认新learner，应与选择器干预分开，不能把Row信号当loading机制已证。'
 else:
  model_judgment='此唯一无参数融合控制没有获得稳定的局部差分与Row联合支持。停止这个模型改动，不继续试结构或loss找正例。弱响应这一现象仍成立，但原因可能涉及优化、噪声、缺失变量或当前目标下的收缩；不能据一个控制的负结果否定全部H3，也不能转而认定H1成立。'
  priority='E1';priority_reason='显式反馈的条件增量仍未被识别，真实/隐藏是更干净且更低成本的下一步。H2目前仍主要依赖观察代理和不同历史方法，尚不足以优先假定numerical critic有效。'
 cost='32次唯一分支拟合、至少16份完整LLM响应' if priority=='E1' else '56次唯一分支拟合、至少24份完整LLM响应'
 text=f'''# NoCW失效归因：研究裁决

**先纠正问题定义：NoCW不是在所有指标上失效；当前数据也不支持把它的不足直接归因于“没有CW”。** 五seed共同L333–1005的NoCW AULC为0.497741，劣于raw LCMD0.462549和数值Hybrid0.438420，却优于新CW-Free0.532876。NoCW的L1005终点0.405025较差，说明早期效率、终点和科学验证价值不是同一个结果。

本轮已完成NoCW自身科学链、四冻结模型诊断及**16次**唯一预定义模型试验。**没有新增付费LLM请求、AL选择或完整轨迹，没有读取test或未采集候选的数值标签。** 下一阶段选择器实验优先建议 **{priority}**。{priority_reason}

## 1. NoCW最可能的三个真实问题

| 问题 | 具体证据 | 机制解释 | 竞争解释/证据等级 | 下一步 |
|---|---|---|---|---|
| 科学目标与Row预测价值未充分对齐 | 527假设中305有linked单因素测试结构，222没有；52首命题样本中17为宽家族/尾部无决策规则、4为事后模型/采集价值描述 | 有条件测量不等于区分宽命题，更不保证对全局Row有用 | 图结构与固定解释样本，非527语义正确率；跨分子设计也可能有价值 | 事前明确claim类型、范围、替代解释和拒绝规则，分别评价科学与预测价值 |
| 证据实体和整批纠错的方案保真不足 | 3152引用时间合法，但检出11条明确分子误写；68纠错批仅保留首答合法ID583/2002，15批全替换、6批假设归零 | 合法ID会绑定错实体，重答会改掉原本合法实验和证据链 | 直接文档事实；尚不能估计造成多少NRMSE损失，也非NoCW独有 | typed row→canonical molecule→变化轴；保留合法槽位，各臂同工程规则 |
| 当前learner对部分条件信息响应很弱 | NoCW41上样V对、27分子，先分子再seed等权的绝对差分：真实3.175/3.756mL，预测0.02045/0.01494mL；冻结前向/局部梯度定位到completion路径 | 输入未丢，但加载条件只经小支路，在正head激活区缺显式分子调制 | 代码/软件行为与已选响应证据；数据噪声/未记录条件、优化和目标收缩未分开 | 见本轮唯一模型控制；噪声分解仍需原始复测身份 |

这里的“最可能问题”是优先应检查的环节，不是已估出它们对AULC各贡献多少。没有足够证据将NoCW定性为普遍追逐少数分子族：45个按gain固定病例中最小gain更集中4/5seed同向，但接近当轮U的泛苯环本底；近3批重复仅3/5同向。高gain的分散与其收益是解释性关联，不能直接训练新采集规则。

15个最小gain位置中13个实际恶化，但当前加后续2批有10/15累计为正；后来的其他实验和重训都参与变化，既不能说原批无用，也不能说它的科学实验延迟起效。单轮gain不能当真实标签信息价值。

## 2. 哪些是NoCW独有，哪些是Free共同问题

**尚没有被因果确认的NoCW独有失效机制。** 它的轨迹特征是前期分子探索更分散、后段部分局部对照较少、某些几何覆盖较弱；但新CW中期对照21.94%与NoCW22.08%接近，覆盖随机分位0.5153与0.5028接近。旧Free对NoCW的明显中期差异没有在新CW重跑同样出现。

分子误写、宽命题、反馈理解不完全、整批纠错流失证据是整个Free LLM系列都出现过的问题。共同105批首答合法NoCW37、新CW50；完整回答却是NoCW229、新CW309，后者受packet绑定重复异常影响。不能用首答率或总回答量直接判断科学质量或计费。

QGeoGNN的loading路径、原单位loss、量化宽度失准和优化波动也不属于NoCW独有。不同方法都可能受到相同learner和数据限制。NoCW保留预测、quantile、相似性及反馈；没有显式CW不等于没有数值信息。

## 3. 模型试验实际说明了什么

唯一改动为`p+r+r*tanh(nonaffine LayerNorm(p))`，参数仍458952；不是同时改loss、预处理或预算。四状态H由元数据选择，并在训练前排除了与L333、train和validation同记录条件组的交叉；实际训练量为492/958/490/960。每状态两训练rep，两臂初始函数与完整初始状态逐字节一致。

{md(compact)}

NRMSE/Q及delta均越低越好。row_relative_change是相对变化的小数（0.01=1%）；Q是固定L333尺度归一化的V差分MAE，不是新的AL主指标。所有V1/V2的mL误差、方向、幅度和PE/EA控制在[完整结果](MODEL_PILOT_RESULTS.md)。两个rep必须全部保留，四个状态不是四个独立数据集。

**本轮判断：{model_judgment}** 描述性汇总中Row改善{verdict['row_improved_states']}/4状态、Q改善{verdict['Q_improved_states']}/4状态；八个训练配对为{n_better}优、{n_worse}劣。{rep_pattern}；正结果依赖当前两个训练重复的组合，四状态均值不能掩盖初始化/训练RNG敏感性，不能读成跨初始化稳定提升。方向判读约定是在训练已开始且首组validation配对分数已报告后设定，虽未据H结果改规则，也不是盲法、预注册或正式Go门槛。原始约定及更正字节均保留，不用这个二值结论替代完整效应和反向结果。

![局部与整体取舍](figures/pilot_local_and_row_tradeoff.png)

这个控制还改变梯度conditioning，且调制全部completion残差（含eluent HBA/LogP），不能把结果唯一归因为纯loading容量。H每状态有1–3个训练中完全未见分子，混合条件插值与分子外推；两个预算的H也不完全相同，不能当固定检查集学习曲线。原始实验室批次/复测身份不足，响应差异不能直接当不可约噪声。

**模型的直接事实及候选瓶颈是路径弱与缺少显式交互；未发现V完全丢失或单位代码错误。** 这些结构事实本身没有证明导致多少AL效率损失。编码与冻结输入吻合不等于实际实验语义和所有未记录条件已核实。弱条件响应也可能是有限/噪声数据和Row目标下的收缩选择，而非纯架构容量缺陷。

## 4. Scientific Hypothesis Value与Model Learning Value是否脱节

**有可观察的局部错位，尚未证明系统性因果脱节。** NoCW的m39预测反转、m95平台、m125加载溶剂体积对，都有实际操作数、观测和后续解释；它们主要验证局部模型形状或条件关联，不等于证明新化学机制。NoCW41个上样V对也显示真实记录差分远大于当前预测。

相反，条件测试结构比例与gain的partial均值+0.245、4/5seed正，阻止我们声称科学设计普遍无效。科学修订的下一批也可能恶化，正/负gain都保留。32点共同训练的整批结果没有每个假设的反事实，不能把收益逐项分给某个命题。

建议把科学价值作为**可验证向量**，而非一个自信分数：claim类型（物理关系/模型形状/学习效用）、适用范围、时间/身份合法性、替代解释、可区分实验、选后观测、后续修订。原文没有阈值或预测时记不可判定，不补写。模型价值另报同状态Row增益/短窗AULC、训练重复及H差分；这些大部分是选后评价，不能直接当在线acquisition分数。

## 5. H1/H2/H3的证据排序与独立性

| 解释 | 当前支持 | 仍未被排除的竞争解释 | 裁决 |
|---|---|---|---|
| H1 科学决策 | 具体证据/可证伪性缺口、错实体、纠错改变计划 | 有真实正向科学链，测试结构与gain正关联；模型可能学不动 | 某些环节不足有直接事实，未证明是主要性能损失来源 |
| H2 数值信息不足 | 数值强baseline更好，部分NoCW几何代理弱 | 新CW AULC更差；高/低gain覆盖近似；数字本身可能失准/盲于条件 | 未获得因果验证，不预设critic有效 |
| H3 模型可学习性 | 代码结构、同checkpoint finite difference/gradient、NoCW实际差分压缩、唯一模型试验 | 优化、标签噪声/遗漏变量、目标权重、样本选择均可能参与 | 现象证据强，具体改动是否可用看上述配对结果；不能泛化为唯一原因 |

代码路径和无标签冻结敏感性相对独立于LLM自述；实际NoCW/CW响应统计与科学病例共享底层实验数据，不是两份独立实验。模型pilot是一层新的算法干预，却仍使用同一历史标签和开发validation，不是新实验室/新数据确认。H1/H2/H3也可同时存在，不能把这些相关证据拼成复杂系统后宣称协同已证。

## 6. 下一阶段优先{priority}，并设置明确失败解释

{priority_reason} 真实/隐藏反馈只识别显式通道增量，不能消除checkpoint已吸收的标签。固定置换在3/4状态明显增加历史预测—反馈冲突，当前只作压力测试；不把由显眼矛盾引起的反应当科学反馈使用能力。

E1应保留统一typed环境、全候选预测/条件/身份及L333，隐藏postL333响应、残差、top8与泄漏prose；同时记录输入长度/服务端tokens，效应可能含上下文负担。E2保持NoCW候选packet无CW，只有critic额外给raw几何；条件检查与重答次数跟placebo一致。必须胜placebo，并不劣于raw LCMD后才讨论效率潜力。

下一轮建议的两outerseed×两预算×两LLM重复、并交叉两个训练seed的方案，{priority}成本为 **{cost}**。E1固定训练配置的廉价screen为16fits/16响应，E2为28fits/24响应；它们不能声称已分离拟合噪声。三臂E1若加压力测试是48fits/24响应；E2完整交叉是56fits/24响应。更换learner则另至少4个full-L基态拟合并重算几何；pilot的L_t\\H checkpoint不能直接充当全L_t的状态。

这些是未来协议，不会本轮启动。完整响应可能因纠错增加；固定每proposal最多3完整响应、每逻辑回答最多3物理尝试，缺失对照如实报告。没有核实价格，不估算货币费用；历史大量传输尝试不是等量付费完成请求。

若E1真实与隐藏的选点差不超过同臂重复差，或只改变解释不改善Row，则停止显式反馈提升样本效率的强主张。若E2只改善几何/只胜原提案不胜placebo/仍落后raw LCMD，也停止升级critic。只有实际收益超出配对训练波动并过主要baseline才考虑多轮AULC；当前不扩五outerseed60fits，不追加模型结构或Agent。

## 7. 哪条路线更值得向论文推进

最有辨识力的主题是**化学假设的局部验证价值，何时能转化为固定预测器的可学习信息与Row收益**。贡献应是可验证科学链、模型敏感性与受控干预如何相互约束，而不是新命名一个“LLM＋数值反馈Agent”。稳定Row提升仍须未来E1/E2或新冻结确认，当前不能承诺。

新颖性边界需要明确：已有[BMD预印本](https://arxiv.org/abs/2610.25747)将LLM提出模型族与数值贝叶斯拟合、实验设计和反馈闭环结合；“语言假设＋数值反馈”本身已有先例。已有[模型失配下AL研究](https://proceedings.mlr.press/v300/tang26d.html)讨论信息性、选择分布与泛化误差的关系；科学信息与预测收益不等价也不是本项目首创。这里潜在区别是实际柱色谱、固定QGeoGNN、合法历史科学链和同标签模型控制。上述仅作原作者摘要/页面的定向边界核查，非系统查新；不宣称第一，也未直接套用其理论或acquisition。

若后续严格对照仍无稳定Row增益，应接受停止结论：把LLM保留为生成可检验问题/解释观测的工具，Row预算优化继续以已验证的强数值baseline为主，而不是为保留Agent叙事增加复杂度。

## 8. 本轮执行与最终限制

已执行16个成功fit，2outer seed×2状态×2模型×2训练rep；新增参数0、模型改动1、付费LLM请求0、新AL选择0。冻结源、H、数据角色和fit槽在verification保留；权重/逐行H评估local-only。不得把16fits当16独立seed，不能把已反复使用的validation当新独立测试。

本任务未改原历史综合诊断、模型权重和runner。开始时工作树修改保留，两个旧监控文件可自行更新时间；长时间训练期间另有外部写入者新增其他实验/报告目录，本任务没有运行它们、读取其标签或纳入分析，也不将它们算作本轮工作。最终准确Git状态、并行变化清单和全部输入哈希见[最终核验](verification/final_verification.json)。复现入口见[README](README.md)。完成本轮规定范围后停止。
'''
 (OUT/'RESEARCH_DECISION.md').write_text(text)
 (OUT/'verification/decision_inputs.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
 print('decision written after sixteen completed fits; priority',priority)
if __name__=='__main__':main()

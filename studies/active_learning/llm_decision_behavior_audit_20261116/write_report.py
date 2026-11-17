#!/usr/bin/env python3
"""Write a Chinese research report directly from audited derived statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
NAMES={'historical_free_mixed_model':'Historical CW-Free','new_free_cw_sol61_xhigh':'New CW-Free',
       'nocw':'NoCW-Free','cw_gradient_lcmd32':'CW-LCMD32','free16_lcmd16':'CW-Free16＋LCMD16'}
ORDER=list(NAMES)

def table(df,columns=None,digits=5):
    df=df.copy()
    if columns is not None:df=df[columns]
    def cell(v):
        if pd.isna(v):return '缺失'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v).replace('|',' / ').replace('\n',' ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    d=pd.read_csv(OUT/'batch_metrics.csv');perf=pd.read_csv(OUT/'performance_summary.csv')
    a=d[d.seed.isin([157,6101])&(d.L_after<=1453)]
    seed=a.groupby(['method','seed']).mean(numeric_only=True).reset_index()
    sm=seed.groupby('method').mean(numeric_only=True).reindex(ORDER)
    b=sm[['hamming_novelty','hamming_novelty_random_delta','rarity_random_delta','joint_coverage_gain_random_delta','single_factor_history_fraction']].rename(
        columns={'hamming_novelty':'Hamming新颖性','hamming_novelty_random_delta':'新颖性−随机','rarity_random_delta':'Rarity−随机',
                 'joint_coverage_gain_random_delta':'联合覆盖增量−随机','single_factor_history_fraction':'历史单因素对照比例'})
    b.insert(0,'方法',[NAMES[v] for v in b.index])
    pt=[]
    for end,seeds,scope in [(1005,[157,6101],'A：共同复核'),(1453,[157,6101],'A：共同扩展'),(1485,[157,887,2357,6101,12203],'B/C：五seed纯策略')]:
        q=perf[(perf.L_end==end)&perf.seed.isin(seeds)]
        if end==1485:q=q[q.method!='free16_lcmd16']
        for m,g in q.groupby('method'):
            pt.append({'比较':scope,'预算':end,'方法':NAMES[m],'seed数':len(g),'AULC↓':g.AULC.mean(),'终点NRMSE↓':g.endpoint.mean()})
    ptab=table(pd.DataFrame(pt),digits=6)
    scores=pd.read_csv(OUT/'candidate_score_percentiles.csv')
    q=scores[scores.seed.isin([157,6101])&(scores.L_after<=1005)]
    ranks=q.groupby(['method','feature']).selected_percentile_mean.mean().unstack().reindex(ORDER)
    ranks=ranks.rename(columns={'G_cw_distance':'G分位','U_quantile_width':'U_proxy分位','D_hamming':'D分位','C_hamming_coverage':'C分位'})
    ranks.insert(0,'方法',[NAMES[v] for v in ranks.index])
    pref=pd.read_csv(OUT/'behavior_preferences.csv')
    gd=pref[(pref.scope=='common_1005')&(pref.aggregation=='pooled')&(pref.model=='G_cw_distance+D_hamming')].pivot(index='method',columns='feature',values='coefficient').reindex(ORDER)
    gu=pref[(pref.scope=='common_1005')&(pref.aggregation=='pooled')&(pref.model=='G_cw_distance+U_quantile_width')].pivot(index='method',columns='feature',values='coefficient')
    coef=gd.rename(columns={'G_cw_distance':'βG：G+D','D_hamming':'βD：G+D'})
    coef['βG：G+U']=gu.G_cw_distance;coef['βU：G+U']=gu.U_quantile_width
    coef.insert(0,'方法',[NAMES[v] for v in coef.index])
    hybrid=pd.read_csv(OUT/'hybrid_attribution.csv');hg=pd.read_csv(OUT/'hybrid_gradient_attribution.csv')
    hr=pd.read_csv(OUT/'hybrid_conditional_random.csv');h=[]
    for s,g in hybrid.groupby('seed'):
        cw=hg[(hg.seed==s)&(hg.space=='cw_gradient')];rr=hr[hr.seed==s]
        h.append({'seed':s,'批数':len(g),'LLM单因素对照':g.llm_same_molecule_single_factor_history_fraction.mean(),
            '数值单因素对照':g.numerical_same_molecule_single_factor_history_fraction.mean(),
            'LLM新增分子/批':g.llm_new_molecules.mean(),'数值新增分子/批':g.numerical_new_molecules.mean(),
            'LLM CW覆盖相对贡献':cw.llm_order_average_relative.mean(),
            '数值CW覆盖相对贡献':cw.numerical_order_average_relative.mean(),
            '固定CW16后LLM联合覆盖−随机':rr.joint_coverage_after_numeric_random_delta.mean()})
    gains=pd.read_csv(OUT/'gain_associations.csv');g=gains[gains.scope=='five_seed_all_methods']
    gt=g[['feature','mean_seed_partial_rho','positive_seeds','eligible_batches','seed_min','seed_max']].rename(columns={'feature':'批特征',
        'mean_seed_partial_rho':'平均seed部分秩ρ','positive_seeds':'正方向seed/5','eligible_batches':'可用批数','seed_min':'seed最小ρ','seed_max':'seed最大ρ'})
    cases=pd.read_csv(OUT/'decision_cases.csv');ct=[]
    targets=[('new_free_cw_sol61_xhigh',157,4),('new_free_cw_sol61_xhigh',12203,3),('nocw',2357,15)]
    for m,s,r in targets:
        z=cases[(cases.method==m)&(cases.seed==s)&(cases['round']==r)]
        if len(z):
            z=z.iloc[0];ct.append({'方法':NAMES[m],'seed':s,'选前L':z.L_before,'离散新颖性随机百分位':z.hamming_novelty_random_percentile,
                '联合覆盖随机百分位':z.joint_coverage_gain_random_percentile,'实际ID举例':str(z.selected_ids_excerpt).split('|')[0],
                '可见理由片段':str(z.rationale_excerpt)[:170]})
    stability=pd.read_csv(OUT/'random_stability.csv')
    text=f'''# 色谱主动学习历史选点行为诊断

本研究回答的是：**已有决策如何选择实验、这种行为与真实行级学习曲线有什么关系，以及哪些数值辅助机制值得下一步验证。** 本次没有新训练、完整rollout、付费LLM/Judge、真实标签查询或策略修改。仓库基准commit为`44be2c0070219d71b2560604f8234a390cc14304`。旧综合诊断先通过4587输入/122输出的封存检查；本目录保存独立快照、代码、输入SHA、逐批表和结果。

## 1. 最重要的五项发现

1. **“LLM比CW-LCMD近”成立，但“LLM在所有空间都过度黏滞于历史”不成立。** 两seed共同扩展区间，Historical/New/NoCW的Hamming新颖性相对各自随机基线分别为−0.00749、−0.00041、−0.00068，CW-LCMD为+0.01859。核心区间三个LLM的CW距离分位却分别为0.62655、0.58756、0.55083，均高于池平均约0.5。局部性是空间与阶段依赖的直接行为发现，不能仅从与LCMD的距离差推断低效。见图1–4。
2. **论文原始V/U不能同义迁移；项目特化Logit反而观察到正的区间宽度偏好。** 核心G+U模型βU为Historical 1.12199、New 0.75748、NoCW 0.49733，三个组各两个seed都为正。它们选择较宽预测区间，并不证明寻求认知不确定性。用原论文的“uncertainty avoidance”命名本结果会错误。见图3与偏好完整率表。
3. **Row学习效率与最终端点给出不同排序。** 五seed到1485，CW-LCMD AULC 0.406501，优于NoCW 0.445809、Historical 0.439159、New 0.454195；CW-LCMD对三个LLM的AULC均5/5 seed更低。但New终点0.322703，低于CW-LCMD 0.339061，4/5 seed方向一致。不能只挑一个最终点或用best-so-far替换早期波动。见图5/7。
4. **16＋16存在可审计的分工证据与较好历史性能，尚没有可识别的协同因果效应。** 71批严格来源核验显示数值半批主要承担CW梯度几何覆盖，LLM半批有更多历史单因素对照和新增分子。共同1453的Hybrid AULC 0.426346，比CW-LCMD 0.443228低，两seed方向一致；但固定数值16后的LLM联合覆盖优势跨seed异号。几何分工、整条轨迹性能、两来源的学习贡献是三件不同的证据。见图6。
5. **几何覆盖更高不是行级学习收益的充分条件；G与宽度有值得验证的描述性线索。** 控制方法、预算与当前损失后的seed平均部分秩ρ，联合静态覆盖相对随机为−0.0892（5/5负），单因素对照比例+0.0142（方向混合），G分位+0.2117、U_proxy分位+0.1659（均5/5正，G仅515批）。这支持分开显示几何、对照语义和宽度；不能把任一关联升格为数值干预收益。见图5。

这些都是当前历史数据范围内的证据。Astra/Sol配置、Prompt记忆、重试与历史状态分叉仍混杂机制解释；没有一条结果证明“数值方法必然有益”。

## 2. 论文来源与科学迁移

参考[原论文](https://arxiv.org/html/2609.33061)的Figure3–4、Methods4.10–4.13。原研究用共享GP的候选fitness均值/标准差作解释变量，以条件Logit拟合行为系数V/U，并用Hamming新颖性、Rarity、覆盖及文本探索意图解释局部搜索。它的“value”面向fitness最大化。本项目最终目标是降低Row预测损失，V1/V2保留体积大不等于实验有价值。

因此，本报告保留论文的**条件选择建模、同状态随机校正和可观测性审计**，替换为有明确意义的G/U_proxy/D/C，不制造候选未来收益。没有同状态、可信GP后验或合格ensemble缓存，未新拟合来补足叙事。信念概率与噪声不足，Martingale/Excess Movement不执行；611批文本仅视作最终stated rationale。完整公式和边界分别见[METHODS.md](METHODS.md)、[PAPER_METHOD_AUDIT.md](PAPER_METHOD_AUDIT.md)。

## 3. 实验盘点、协议与可比性

五组实际覆盖为Historical/New/NoCW/CW-LCMD各五seed（157/887/2357/6101/12203），Hybrid仅157/6101。实际行为分析791批、25,312个接受名额；Row曲线813点、22campaign。四纯策略全部L333–1485，Hybrid157到1485、6101到1453；CW-LCMD157/6101还有至1549的已有记录，inventory保留，主分析为统一1485截断。

所有方法为4g / Row split / QGeoGNN-V2，逐seed核对初始L/U、初始化、验证行与首checkpoint hash一致。Row NRMSE以L333固定目标尺度计算，越低越好；每轮scratch训练并使用既有validation选择checkpoint。不同seed的化学数据和行划分重叠，不能当成五个独立数据集。

Historical157含Astra/high前缀，r5有既有授权抄写修正、最终served identity未知；6101含更长Astra/high前缀。其余三seed从头Sol/xhigh；New/NoCW均请求Sol6.1/xhigh。Hybrid也有Astra前缀。New的有限重试与NoCW长期纠错/服务恢复路径不相同。NoCW只移除显式CW，仍保留预测分位、宽度、化学与历史反馈，绝不是“无数值信息”。详细目录、Prompt SHA、模型与候选约束见[RUN_INVENTORY.csv](RUN_INVENTORY.csv)及[PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md)。

Comparison A的原共同复核窗口为两seed L333–1005；A扩展至真实共同L1453。B是三LLM五seed，并单列从头Sol的三seed；C为LLM与CW-LCMD五seed到1485；D为Hybrid的两seed来源归因。**严格可比**仅限核验一致的初始冻结状态；同预算轨迹属于**部分可比、需控制**；旧raw-LCMD/数值Hybrid仅作**描述性参考**；把Historical Astra与NoCW差异认作纯CW效应属于**不可比较**的因果解释。

CW是Center/Width坐标变换，LCMD是批量算法：`C=(V1+V2)/2,W=V2−V1`，用各seed L333固定sC/sW缩放，两个q50输出的全网络梯度经512维CountSketch。LCMD每步选择平方距离和最大的最近中心簇，再选簇内最远点并更新。LLM可见的是到L最近距离的池内分位，不是完整LCMD分数；随训练改变的原始梯度距离不能跨轮混成固定尺度。raw V1/V2-gradient LCMD与数值分歧/latent Hybrid均未冒充目标方法。

## 4. V/U风格行为分解：同一特征空间中的差别

核心共同L333–1005，每方法42批/1344选点，先给全合法池选中分位描述：

{table(ranks.reset_index(drop=True))}

G为冻结CW最近历史距离；U_proxy是两端点原始q90−q10的排名组合；D为六因素Hamming新颖性；C为128固定参考的单点Hamming覆盖。池均值约0.5002。CW-LCMD在其使用的梯度空间明显更远；三LLM也均在G上高于随机水平。这是对“全部LLM都回避新颖候选”的直接反证。D/C差别较小，包含已有历史与因素空间饱和的影响。

按照论文的负采样形式，每个正例对应64个真实未选负例，排除整批、ridge=1、共同参考z标准化，先单/双特征再审查完整模型：

{table(coef.reset_index(drop=True))}

五策略可用共同G+D；四个含LLM组还可比较G+U。核心有G/U的四组各两seed的G/U方向均为正，三次负采样的方向不变。全部模型优化通过；744条系数保留campaign结果，共22条五campaign的100次bootstrap区间。完整模型仅在cohort特征完整且条件数≤100时保留，详见[behavior_preferences.csv](behavior_preferences.csv)、[PREFERENCE_RESULTS.md](PREFERENCE_RESULTS.md)。图3同时给出G+D、G+U二维图和单项系数，避免把不同模型系数混合。

**时间与完整率**：D/C全部791状态；U在611个含LLM状态完整，纯CW180状态留空。G复用Historical180、New180、Hybrid71的catalog，并从已有可信缓存补NoCW/CW-LCMD各42批。后两组其他G是本次未展开，不能宣称历史不存在；L1453/全轨迹的局部G模型不能解释成整段完整比较。候选分位表共2708行，输入/可用性见[preference_state_coverage.csv](preference_state_coverage.csv)。

这些β是行为关联，LCMD的逐点Logit仅为集合选择近似，Hybrid总体系数混合两来源。U正不等于认知不确定性寻求；q90−q10不是epistemic posterior std，预测响应大小、条件语义与难度可能共同影响它。原论文的V轴在这里不可识别，没有用保留时间均值偷换。

![图3](FIGURES/figure3_behavior_preferences.png)

## 5. Context-stickiness：依赖表示、阶段与对照语义

下表为A扩展两seed L333–1453，先在每seed平均35批，再等权平均。Hamming用比例单位，单因素修改贡献1/6；连续条件则保留变化幅度。

{table(b.reset_index(drop=True))}

Historical的离散新颖性低于同状态随机约0.00749，相当于平均少0.045个不同因素，不能夸大为六因素空间中的巨大局部化。New/NoCW整段几乎贴近随机；CW-LCMD明显高于随机。与此同时，三个LLM在CW梯度G中选中分位均大于0.5。这表明“比LCMD近”和“比随机近”不是同一个命题，支持的是**有限、空间依赖的局部偏好**，而非论文全部机制的复现。

New早期L333–525的Hamming差为−0.0125，扩展阶段转为+0.0037；NoCW早期为+0.0158，后期接近/低于随机。把整条轨迹压成一个局部程度会遗漏相反阶段。按同seed/phase的原始值见[phase_metrics.csv](phase_metrics.csv)。

历史单因素对照比例三LLM约79%，CW-LCMD也约72%，Hybrid约77%；后期随着H变密集，各方法都有超过85%的阶段。这不是LLM独有，也不自动意味着浪费：合法Prompt允许检验局部条件假设，未强制最低对照名额。分子与条件的最近距离分开、同分子条件距离的可计算分母也单列，避免把分子更新与条件小幅调整混为一谈。

[concentration_campaigns.csv](concentration_campaigns.csv)还记录精确众数分子的连续轮次：Historical/New各campaign最大为2–3轮，CW-LCMD为1–2轮；未见按此严格定义持续多轮的单一众数垄断。该统计按众数并列的词典序规则，不等于近似结构簇的持续性。

Action Preference Dynamics中，上一批分子重用名额CW-LCMD约15.1%，Historical32.5%、New29.5%、NoCW25.8%、Hybrid28.6%；相邻分子分布JS也表现数值策略变化较大。此证据支持近期分子延续的行为差异，无法单独识别历史上下文的因果作用。论文通过移除history建立机制，本任务没有这样的干预。不能由行动频率计算“信念过度反应”的M/Z。

![图2](FIGURES/figure2_context_stickiness.png)

## 6. Rarity、覆盖与探索指标的反例

Rarity按每seed固定初始合法域加一平滑，既报用户要求的raw−logq，也报论文额外归一化版。分子/条件分别统计，连续4/8/16箱敏感性不使用标签。A扩展的raw Rarity−随机为Historical−0.02667、New−0.01619、NoCW−0.01568、CW-LCMD+0.01125、Hybrid−0.01514；在因素频率视角数值策略更稀有，但这不是信息收益。

固定结构/条件联合覆盖相对随机，Historical−0.00054、New−0.00048，NoCW+0.00018、CW-LCMD−0.00008、Hybrid+0.00005。这些排序与G显著不同：梯度LCMD优化的几何不等于静态分子/条件空间的覆盖。NoCW略高的联合覆盖也不能推出其Row性能最好。连续条件几何本身集中且增量很小，分子覆盖后期饱和，不能把零新分子解释为零学习价值。

每个状态使用自己真实U_t，500次无放回随机32，绝不使用统一全局随机分布。抽样稳定性如下；它衡量Monte Carlo误差，不是跨campaign置信区间：

{table(stability,digits=7)}

距离L1/RMS与连续分箱的轨迹敏感性见[sensitivity_metrics.csv](sensitivity_metrics.csv)。主文保留离散、连续与结构多视角；没有从中事后挑选最支持“LLM局部”的一个。Novelty、Rarity、Coverage均为探索代理，不等于真实信息增益。

![图4](FIGURES/figure4_exploration.png)

## 7. 显式CW信息、两次CW-Free与NoCW的差异

New对NoCW在核心阶段的选中G分位高0.03673、U代理高0.06346；βG/βU也更大，说明输入CW伴随更偏向相应几何与宽区间的历史选择。但同预算状态已经分岔，NoCW仍有其他数值信息、重试路径也不同，因此这只是配置/输入与行为的一致性证据，不是显式CW单因素因果估计。

真实五seed L333–1485上，New相对NoCW的AULC差+0.008386，4/5 seed更差；终点差−0.026596，4/5更好。显式CW关联的是不同时间分布的学习进展，不能只凭终点判定整体更有效，也不能只凭AULC判定其末期无价值。

New相对Historical的AULC差+0.015036，5/5 seed更差；终点差−0.011382，4/5更好。即使只取从头Sol/xhigh的887/2357/12203，Historical AULC0.402729、New0.425011，差+0.022283，仍没有消失。因此差异不能全部归因于Astra前缀。但Prompt/历史记忆、有效上下文、纠错重试、随机采样和后台时变未被单独控制，无法确定唯一来源。选点G/U分布更低与轨迹变化一致，不能倒推“低G导致差AULC”。

图7把混合模型前缀两seed、从头Sol三seed、全部五seed分开展示，所有曲线保留非单调波动，没有跨预算best-so-far或插值。配置一致子组仍仅部分可比；不能说两次CW-Free是相同输入的确定性复现。

![图7](FIGURES/figure7_llm_trajectories.png)

## 8. Hybrid互补性：局部对照、结构补充与梯度覆盖

全部71批均核对数值16先成为pending、LLM16再补，最终32个合法唯一ID；93个已保存结构化提议中16个choice列表与最终不同，28个提议ID事件不属于当轮非pending合法池（按答复计事件：16个已测量、4个pending、8个未知ID事件，共24个不同ID字符串）。它们说明需要区分原始提议与最终接受，不能不读错误类型就归因于同一种拒绝机制。详见[hybrid_proposal_audit.csv](hybrid_proposal_audit.csv)。

{table(pd.DataFrame(h))}

LLM半批更常是历史单因素对照（约85%对数值69%），也补充更多新分子；数值半批的CW距离分位约0.94，LLM约0.53–0.58。对同轮原始raw/CW缓存的142个来源归因记录全部完整，CW参考覆盖降幅的顺序平均贡献，数值约6.3%–6.9%，LLM约1.2%–1.7%。这里分母是各轮选前参考距离，保留在同轮空间，没有跨checkpoint混原始距离。

联合覆盖归因在固定结构/条件空间则有seed差异：LLM顺序平均贡献157约0.00114、6101约0.00149；数值分别约0.00133、0.00118。固定实际数值16，再随机从U\\N补16，LLM的联合条件边际相对随机分别约−0.00015、+0.00025，方向相反。两来源有可见分工与某些互补，也存在共享覆盖；**没有跨seed一致的几何协同证据**。联合集合增量与两来源顺序平均归因的和通过数值核验，但它不是学习贡献分解。

批内冗余另作区分：LLM半批固定联合空间的最近邻均值为0.221/0.290，数值半批0.376/0.371，LLM到另一半最近距离0.384/0.386；两半较分离，LLM自身较局部。两来源参考覆盖重叠之和/联合增量之和为4.03%/2.04%，说明在该空间的共同覆盖部分较小。由于加点本身会缩短最近邻距离，且没有同状态纯32反事实，这不等于已证明联合32比纯策略更少冗余。

两seed同预算1453，Hybrid AULC0.426346，CW-LCMD0.443228，差−0.016882且两seed都为负；终点0.337784对0.347681。共同1005上Hybrid AULC0.484421对CW0.500029，同样方向。它是值得进一步检验的历史轨迹优势，尚缺冻结相同状态、配对重训和更一致配置，不能认定LLM与LCMD具有因果协同，更不能给LLM16分配精确性能贡献。

![图6](FIGURES/figure6_hybrid.png)

## 9. 最终落脚：Row Performance与行为关联

真实曲线的区间平均损失与终点如下。每项先按seed算积分，再平均；不使用统一最短终点以外的不存在记录。

{ptab}

总体上CW-LCMD更早降低损失，New后期端点更低，Hybrid在两seed共同区间有较好效率。初始相同并不能消除路径混杂；早期数值策略的较好学习效率不等于每个后期batch都更有用。

严格连接当前选点到下一次真实拟合更新后，采用seed内部分秩模型控制策略、预算和当前损失，再按seed等权：

{table(gt,digits=4)}

其中“全部方法”指可用轨迹合并到五个seed块，Hybrid只在两个块中，不是每方法都有五seed。G只有515批、U只有611批，不能与791批完整指标的关联直接当成同一cohort因果比较。G/U正方向为下一步候选机制提供线索，但响应尺度、Prompt历史和未记录模型难度可能同时解释它们；本模型没有完整控制q50大小或所有化学因素。

联合静态覆盖相对随机的部分秩为负，单因素对照的方向混合，反对“覆盖越大下轮一定改善越多”与“局部对照一定差”的两种强假设。对照比例与收益的阶段关系亦无稳定单向规律。分阶段、每campaign、lag2及配对差完整保留，见[gain_seed_associations.csv](gain_seed_associations.csv)、[lag2_associations.csv](lag2_associations.csv)、[paired_behavior_differences.csv](paired_behavior_differences.csv)。

gain=e_t−e_(t+1)与e_t存在算术耦合/回归均值；连续轮次依赖仍在。2000次seed块bootstrap仅有2/5块且有多指标探索，区间不作强显著性结论。这些关联不能用于声称优化某一代理会改善Row性能。

![图5](FIGURES/figure5_row_performance.png)

## 10. 探索意图、实际行动与信念动态

可见最终理由611批，候选理由与实际接受ID逐批绑定；未调用Judge。关键词筛查保留探索、局部对照、避免重复的词频和文本长度，供人工审查，而不制造[0,1]意图概率。跨seed的词频—探索代理相关方向混合，New既有正也有负；不存在可以概括所有策略的“说探索就不探索”零相关结论。

图8的Hybrid卡片理由与ID来自LLM16，但标注的随机百分位对应最终接受32点batch；不能把整体batch指标归为LLM16单独效果。分半条件随机见Hybrid归因表。

32个案例按方法×早/中/后/扩展阶段分别取新颖性校正差最小和最大，因而刻意同时展示局部例与反例，不能用它们估算发生率。典型三例：

{table(pd.DataFrame(ct),digits=3)}

New seed157 L461的理由强调实验对照，而离散新颖性/联合覆盖仅随机0.1%/0.2%分位，这表明科学上“探索一个条件假设”未必在几何上广泛探索；没有新标签反事实时不能说它无效。New seed12203 L429的功能团与对照探索达到100%/99.8%分位，是执行广泛探索的反例。NoCW seed2357 L813离散新颖性仅0.1%，联合覆盖却98.2%，直接显示两代理并不等价。原理由、候选ID、完整数值与阶段见[decision_cases.csv](decision_cases.csv)。

这些只能识别stated rationale与实际集合的关系，不能恢复内部思维、证明不诚实或计算校准后的Stated Belief。选择分布变化与近期邻域重用称作Action Preference Dynamics；没有一致概率对象与测量误差，不计算M/Z，不宣称LLM“过度反应”。

![图8](FIGURES/figure8_decision_cases.png)

## 11. 哪些机制得到支持，哪些被反证或不可判断

| 机制命题 | 当前证据 | 强度与替代解释 |
| --- | --- | --- |
| LLM比纯CW-LCMD更局部 | Hamming、近期分子重用、核心G排序均支持 | 行为差异直接可观测；数值法优化自身几何，非低效因果证据 |
| 全部LLM显著低于随机探索 | 三LLM G均>0.5，New/NoCW整段Hamming约随机 | 强表述被反证；Historical存在有限离散局部性 |
| LLM回避论文式认知不确定性 | U_proxy偏好为正，且代理不等于epistemic | 当前无法识别原命题；不能直接移植论文 |
| 显式CW改善整条学习效率 | New对NoCW AULC更差而终点更好 | 不支持统一改善；配置/重试/状态混杂尚在 |
| 两次CW-Free差异仅来自Astra/Sol | 同从头Sol三seed仍有AULC差 | 单一解释不足；输入路径、记忆与随机性不可拆分 |
| Hybrid有分工 | CW几何由数值半批主导，LLM更多局部对照/新分子 | 同状态来源证据较强；学习协同未识别 |
| 几何覆盖更高必然有用 | 关联不支持，空间间排序不同 | 强命题被反证；真实学习收益不是覆盖的定义 |
| 行动变化证明信念过度反应 | 缺可比信念概率和噪声 | 不可判断，仅Action Preference Dynamics |

## 12. 研究路线：由问题到可证伪的数值干预

下表是未来实验建议，本轮没有实施任何一项。

| 发现的问题 | 实验证据 | 可能原因 | 推荐数值干预 | 所需验证实验 |
| --- | --- | --- | --- | --- |
| 局部性依赖表示，单一阈值会误伤科学对照 | 三LLM G>0.5；大量单因素对照；Hamming与coverage反例 | 几何与假设目标不同；状态密度随预算变化 | 给提案显示同轮CW距离、条件差异、固定空间覆盖和随机百分位，保留对照豁免及理由 | 固定同一checkpoint/L/U/配置，原提案 vs 几何诊断修订 vs 同文本长度placebo；之后做配对重训，以Row gain/AULC为主 |
| CW信息改变选择但早期效率未稳定改善 | New比NoCW G/U高，五seed AULC+0.008386、终点−0.026596 | 代理奖励与学习目标不一致；上下文/重试混杂 | 提供可解释的整批几何诊断及当前分布参照，不用单一综合分数强行替换 | 同状态随机分配可见CW/不可见CW，统一模型/Prompt/重试，预先固定早/后期和loss horizon |
| Hybrid分工明确、协同尚不可识别 | 两seed AULC改善；数值CW覆盖主导；LLM条件随机优势异号 | 几何补充与局部假设互补，但可能存在路径偶然性 | 来源标记和条件边际诊断，比较固定16＋16与阶段预算分配，保留真实对照 | 至少多个共享seed，同状态纯CW32、LLM32、16＋16；相同scratch重训，检验配对AULC/终点/变异及额外LLM成本 |
| 宽度有偏好/收益关联却无epistemic资格 | 核心βU全正、描述性gainρ+0.1659 | 响应大小、异方差、区间失配或真实难度 | 宽度只作疑点代理，显示q50、交叉标志与适用范围；先核验增量排序 | 固定预测状态的时间合法观测/校准/检查划分；q50+CW基线对加width，独立增量过关后再试LLM输入 |
| 优化覆盖可能偏离Row目标 | 静态coverage相对随机与gainρ−0.0892，5seed均负 | 空间不匹配、饱和、阶段与起点耦合 | 将coverage当诊断维度，避免最低覆盖硬配额 | 预先固定同状态候选集合/对照预算，比较coverage修订是否真的改善下一拟合；不改善则停止把其作为主目标 |
| 新旧同名运行易被误当配置复现 | New对Historical AULC5/5更差，Sol三seed仍差 | Prompt记忆、随机响应、服务路由和纠错路径 | 冻结版本、输入hash、有效历史摘要和候选身份，独立记录原始/修订/接受 | 同冻结输入重复提案，评估行为方差；随后有限同状态配对训练，不能只比较一次轨迹 |

**优先的三个可证伪假设**：①保留单因素对照的多空间数值诊断比仅提高覆盖更可能改善后续Row gain；②CW-LCMD16提供梯度几何、LLM16提供有条件的假设探索，联合收益需在同状态配对训练和更多seed下复现；③宽度只有在控制q50与CW并证明时间外增量后才值得升级为数值风险提示。若代理改善而Row没有改善，应降低该路线优先级，不通过事后换指标“挽救”。

## 13. 复现、完整率与交付验证

[README.md](README.md)给出离线入口。所有本次代码只写本目录；初始未提交修改的指纹在`repository_start.json`，最终输入/输出校验见`FINAL_VERIFICATION.json`与`INPUT_MANIFEST.json`。独立审阅已核对全791批历史转移、813个曲线网格、Decimal非单调积分、88个状态Rarity、初始五方法距离/覆盖与同状态随机。关键指标测试覆盖离散/连续距离、平滑/稀疏编码、单调覆盖、随机排除与固定种子、Logit数值稳定/梯度/偏好恢复、Hybrid来源去重、horizon与非单调AULC。

完整结果含逐批、逐候选、逐seed/阶段、分位、系数、来源归因、滞后性能、文本案例和8幅图的PNG/PDF/SVG。当前环境无Parquet引擎，逐候选交付为无损CSV，不伪造扩展名。未计算项目与不可识别因素集中列于[LIMITATIONS.md](LIMITATIONS.md)。这次审计支持下一步选择**应验证的机制**，不替代干预实验的行级性能证据。
'''
    (OUT/'REPORT.md').write_text(text)
    print('Chinese report',len(text),'characters')

if __name__=='__main__':main()

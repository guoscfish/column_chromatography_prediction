#!/usr/bin/env python3
"""Produce decision memos and the new counterfactual trade-off figure."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/chrom_followup_mpl')
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent

def main():
    prefixes=pd.read_csv(OUT/'llm_prefix_by_rejection.csv')
    total=int(prefixes.batches.sum());prefix_n=int(prefixes.ordered_prefix_batches.sum())
    cf=pd.read_csv(OUT/'counterfactual_seed_phase.csv')
    seed=cf[(cf.scope=='all_available')&(cf.level=='seed')]
    phase=pd.read_csv(OUT/'phase_performance.csv');overall=pd.read_csv(OUT/'phase_overall.csv')
    assert len(seed)==2 and len(phase)==8 and total==540 and prefix_n==127
    decision='''# 下一轮实验投资决策

**先做10次固定标签模型控制；不立即追加3条长期Hybrid，也不完整重跑NoCW。** 三seed Hybrid为Conditional Go（受门槛约束），完整NoCW重跑为No-Go。这里的Go是研究建议，不是本轮执行授权；本轮没有启动任何新训练、LLM调用、标签查询或rollout。

## 1. 本轮真正新增的三项发现

| 新发现 | 直接证据 | 新增决策价值 | 还不能推出 |
| --- | --- | --- | --- |
| LLM16替换了有明显CW边际的后16，增加的是特定条件对照 | 71/71准确重建前16；56批与后16零ID交集；LLM相对H+N的体积对照45.49%/52.32%，CF后16始终0；CW边际1.52%/1.17%对5.01%/5.33% | 真实研究对象是“当前数值表示不敏感的条件信息能否转成学习收益”，而非抽象的探索多寡 | 覆盖下降不是学习损失，物理响应非零不是LLM16已有因果增益 |
| 纠错后经常退到候选展示前缀 | 540批中127批与前32完全同序；127都有保存拒绝，277无拒绝记录中0例；首份可解析choices提议均非前缀 | 同名方法的实际策略混合了科学提案与纠错后的简化选择，NoCW整跑不能独立识别CW作用 | 代码自动fallback、模型复制文件、主观动机或prefix造成多少AULC损失均未证明 |
| 长期Hybrid优势跨阶段而非单点，却不能拿短筛成败直接合并 | 两seed后3段AULC均更好；局部真实梯形排除后仍负；早期157反例；短筛128候选且无逐行反馈，主报告test口径不同 | 有值得检验的长期信号，不能因为早筛未过门槛就全盘否定，也不足以直接放大预算 | 不是5seed同配置确认，不支持通用预算切换阈值或LLM因果协同 |

## 2. 最值得研究的一个机制问题

**LLM补充的上样条件响应，究竟是当前模型没有稳定吸收的可学信号，还是不能改善Row目标的噪声/次要变化？**

原165对结论已复用，本轮把幅度与方向分开：New五seed体积对在真实变化>0.5mL时，预测近乎平坦的比例V1/V2为66.17%/75.70%，非平坦但反向仅2.67%/10.03%；PE/EA近乎平坦仅1.96%/0.57%。157后期V1恢复、887/2357仍平坦，是强异质性反例。这支持优先检查一个具体条件学习通路，不足以认定网络容量、loss或未记录变量的独因。

它比立刻增加LLM轨迹更有诊断价值：新增3seed需要105个有效LLM决策及105次fit至1453，仍会把模型、选择、纠错和历史路径混在一起；固定标签控制只需10次fit，能够先把模型吸收能力从选点机制中分离。若这个特定模型控制失败，应降低该可修复通路假设的优先级，不继续靠更长rollout寻找偶然正例。

## 3. 三个选项的具体判断

### Option 1：补887/2357/12203 Hybrid —— Conditional Go，暂不启动

长期两seed后3段一致较好且非一个异常点撑起，使补seed有科学价值；但候选反事实尚无真实训练收益，且旧两seed为Astra/high→Sol/xhigh，新三seed若从头Sol只能成为**新三seed cohort＋旧两条异质参考**，不能声称同配置五seed确认。

| 范围 | 新有效LLM决策 | 新QGeoGNN拟合 | 可复用 |
| --- | ---: | ---: | --- |
| 新3seed至L525 | 18 | 18 | 三个L333初始fit/checkpoint/Row split/CW bank |
| 至L1005 | 63 | 63 | 同上，之后状态必须来自各自Hybrid历史 |
| 至L1453 | 105 | 105 | 同上；不能借纯CW/Free/NoCW后续checkpoint |
| 新3seed至L1485 | 108 | 108 | 不含旧6101缺的末批 |
| 五条混合配置轨迹全部1485 | 109 | 109 | 另补旧6101的一批 |
| 五seed全部从头统一Sol至1485 | 180 | 180 | 只复用初始fit，旧选择前缀不能洗成新配置 |

这些是有效接受决策/成功fit数，不是物理请求上限、实际独立标签数或金额。纠错/传输可能增加成本。三seed相同起点的CW-LCMD、CW-Free、NoCW完整保存曲线可作总体轨迹参照，无需为这些历史对照重复拟合；它们仍非冻结同状态反事实。

**启动门槛**：先完成下述模型诊断；如果只有局部拟合改进而Row无价值，不以“LLM有条件知识”作为长跑理由。进一步要放大Hybrid，最小直接资格应是在预定冻结状态、相同QGeoGNN与相同H下，Hybrid相对纯CW32及CW16+Random16有Row收益，且不是单状态偶然值。该单步训练资格本轮未执行。若仍选择探索性补seed，先到1005，按新三seed单独报告，预先停止：平均配对AULC不优于CW且至多1/3seed获益；不得事后换窗口或把旧两seed合入“救活”均值。

### Option 2：再次完整NoCW-Free —— No-Go

| 做法 | 实际能回答 | 仍排除不了 | 成本量纲 |
| --- | --- | --- | --- |
| 同配置同seed重新整跑 | 当前服务、生成随机性、纠错及后续训练路径的总变异 | CW信息因果作用，模型吸收能力，单一随机源 | 每seed至1485为36决策+36fit；五seed180+180 |
| 新outer seed扩样 | 同一策略包跨split/初始化的变异 | 纠错、候选顺序、模型版本与其他输入混杂 | 新seed还需合法split与L333初始化；匹配CW基线另计，不能假定能借旧checkpoint |
| 同冻结状态CW/NoCW配对提案 | 控制H/U/checkpoint后显式CW列对实际候选/失败路径的影响 | 没有标签/重训时不识别学习收益 | 例：5seed×2可见性×3独立生成=30份有效回答；0fit、0标签；纠错另计 |

Hist/New首批科学packet五seed完全相同，但system prompt差47字符；NoCW去掉CW，仍有预测区间与历史。三个轨迹首批有些因为纠错后选同一展示前缀而完全重合，不等于独立“科学偏好一致”。后续H分叉、记忆和服务日期也不同。故再整跑NoCW只能增加一个复杂随机系统的轨迹样本，不能消除本轮最关键的不确定性。

若未来确实要问“CW列改变了什么”，优先少量冻结提案实验：统一模型/effort/重试上限/完整system prompt，CW可见性为唯一字段差，固定H/U/预测；独立生成并记录原提议、错误、最终接受。若另测顺序敏感性，预先随机化展示顺序并做显式独立因子，不与CW效果混为一个干预。本轮没有调用这些LLM。不要将30回答提案实验包装成Row性能验证。

### Option 3：固定标签模型条件学习控制 —— Go，优先且仅此一项进入下一预算

主实验固定New CW五seed的L1005，不看哪个历史gain最好/最差。取已揭示H内元数据哈希排序的10个完整体积响应系列作检查；排除触及L333的系列，同记录条件重复row整体隔离。五seed检查行数23/24/21/20/21，实际训练982/981/984/985/984。holdout计划已落盘，不根据真实响应挑选，没有打开未揭示池标签。

两臂仅差一个模型因素：在现有标准化上样量x到head线性量增加零初始化直接残差，每endpoint一个w_t，同加其三个quantile后再过原ReLU/clamp。保持原分支、输入、loss、训练流程、acquisition与Prompt；无独立width参数。原ReLU/clamp仍可能改变宽度，不承诺宽度严格不变。新增参数从零开始，共享参数初值逐张量相同、构造后重置训练RNG。

初筛5seed×2臂×1配对训练RNG=**10新fits，0LLM，0新增标签**。两臂都必须从头训练H\\check；原L1005 checkpoint已看过check，不能作为baseline复用。主评价仍Row validation NRMSE，局部差分MAE为机制指标；validation只作原checkpoint选择，check不参与选择。通过后如需检验训练方差，预定加两套配对训练RNG，合计30fits，统计仍先在outer seed内平均。

这不是同时修改网络、loss、采集和Prompt，也不是新AL。全流程需要未来单独允许训练才能执行；本轮没有加载权重、训练或查询新标签。

## 4. 什么结果会使我们放弃当前假设

预先给具体、可被否定的主张：“一个直接上样量通路能修复当前条件响应压缩，并改善或至少不损害Row主目标”。机制初筛门槛建议为局部差分MAE下降≥20%、至少4/5seed方向一致，Row NRMSE相对恶化不超过1%。这只证明局部机制可修复；若要把它升级为解释AL效率的主线，再要求seed均值Row NRMSE相对改善≥1%、至少4/5seed不恶化。落在±1%的模糊区只保留机制线索、暂停长跑，必要时使用预定另两套配对RNG复核，不事后挑训练seed。这些是未来设计门槛，不是已确认统计阈值。

- 局部差分无稳定改善：不继续扩展此线性旁路，不把“加一条条件通路就能吸收LLM信息”作为长跑理由。它反证的是这一具体干预，不能证明所有条件模型都无能力。
- 局部改善、Row无益或明显变差：保留局部解释价值，放弃“该瓶颈是Free学习效率差的主要可修复原因”的主线，暂停基于此机制的Hybrid扩张。
- 局部与Row均改善：支持条件学习通路值得研究，仍不证明LLM不可替代。继续比较同状态Hybrid、CW32及CW16+Random16的真实学习收益；只有后者过关才追加长期LLM预算。特定旁路阴性不能否决所有Hybrid机制，模型诊断是成本优先顺序，不是所有互补机制的逻辑必要条件。

本轮最大确定性是候选替代结构与可见纠错行为，最大未确定性是**对新训练的增量Row收益**。不以增加相关图或把几何分数变好作为通过门槛。
'''
    (OUT/'NEXT_EXPERIMENT_DECISION.md').write_text(decision)
    summary='''# 给研究者的总结：先确定模型能不能用这些信息

这轮没有重跑主动学习，也没有调用新LLM或训练模型。我们真正补上的，是“LLM到底替掉了哪些数值实验”“模型有没有表现出相应条件响应”“同名LLM运行为什么会走向不同路径”三条证据链。结论不是LLM无用，也不是Hybrid已证明有效，而是现在可以更明确地决定下一笔预算花在哪里。

**新增发现一：LLM换来的主要不是更多普通分子，而是数值算法几乎不选的上样体积对照。** 我们在71个真实Hybrid状态复用原CW缓存，严格运行原LCMD32；每轮前16的ID和顺序都与历史一致，因此后16是可信的候选反事实。实际LLM16与数值后16在56批完全不重叠。固定历史和数值前16，LLM约45%和52%的名额与已知实验构成“只改变样品体积”的对照，数值后16在全部71批都是零。PE/EA对照两边都很多，不能说数值方法不会设计对照。

它付出的代价也清楚：当前CW空间的边际覆盖，LLM约1.52%和1.17%，数值后16约5.01%和5.33%。因此不是拿无价值的后半批换科学实验，而是拿仍有明显梯度覆盖的候选换另一类条件信息。可能原因是数值表示对该条件轴不敏感，但这还不能说明换得好不好；反事实候选没有同状态训练收益，不能给它们借用未来标签。下一步必须判断这些条件信息能否帮助预测。

这里模型诊断提供了关键线索。旧165对的幅度压缩结论已经复用，本轮进一步区分“太平”和“方向错”：五seed中真实变化超过0.5mL时，V1/V2预测近乎平坦约66%和76%，非平坦但反向约3%和10%。只看符号，V1却有86%的正确率，因为极小的同向变化也被算作正确。这说明主要问题是幅度，而不只是符号。PE/EA没有这么普遍的平坦。157后期能较好响应V1，另一些seed仍近乎不动，所以不能声称网络绝对没有能力，也不能说加标签后自然都会恢复。噪声、缺失变量和优化仍是竞争解释。

**新增发现二：相当一部分最终批次是在纠错后退到展示前缀。** 540个纯LLM批次中，127批最终恰好选择候选表前32行，连顺序都相同；Historical/New/NoCW分别25、40、62批。这127批都有保存的拒绝记录，没有拒绝记录的277批里一例也没有。它们最初可解析的方案都不是前缀，有38次原方案其实已有31个合法ID，最后仍几乎整批重选。

这解释了为什么有些不同LLM运行首批会完全相同：它们共同退到了同一个候选展示前缀，而不是必然有相同科学判断。独立请求和回答记录存在，源码与纠错消息没有要求前32，所以不能指控代码复制或自动数值补位。我们也没有证明前缀造成多少性能损失。下一步若研究CW输入作用，应固定同一状态、统一请求和纠错政策，比较原始提议与最终接受；再跑整条NoCW会继续混入这些因素。

**新增发现三：长期Hybrid的优势不是一个尖峰撑起来的，但也不是从第一阶段就一致。** 按事先指定的四段预算，两个seed在后三段AULC都优于CW，157在第一段反而较差。轮流排除一个检查点周围的两个真实梯形后，整体优势仍在；这增强了信号的可信度，却不能给出通用的切换阈值。此前短筛只有128候选、不给逐行历史反馈、模型快照也不同，公开结果还是test口径；长期是完整候选池与validation曲线，两者不能合成一个“16+16平均效果”。

所以，最值得问的一个机制问题是：LLM追加的体积条件信息，是当前模型没有稳定吸收的可学信号，还是对Row目标帮助不大的噪声或次要变化？不同实验信息已经看到，额外有效学习信息还没有证明。

**补Hybrid三个seed：Conditional Go，当前先不长跑。** 后三段信号值得验证，但新三seed若从头Sol，旧两seed的Astra前缀不能抹掉，只能新三条独立报告、旧两条作异质参考。到L1453要105个有效LLM决策和105次拟合；到1485新三条需108加108，如果还补旧6101最后一批则总109加109。先完成模型诊断，再通过同状态学习资格，才分段授权；新三seed平均不胜CW且至多一个获益，就停止扩大，不能事后换窗口。

**再次完整NoCW：No-Go。** 同seed重跑能测整个生成、纠错和训练路径的总变异，新seed能测跨划分稳定性，都不能单独识别CW列。若要问CW影响，少量冻结状态配对提案比整跑更合适；若要问Row收益，仍需额外的配对训练，提案变化不能代替性能。

**如果只能做一个新实验，我选固定标签的模型条件通路控制，先做10次拟合。** 预定New CW五seed的L1005，按元数据哈希固定10个完整体积响应系列作检查，两臂都从头训练相同的剩余已揭示标签。唯一变化是加零初始化的直接上样量线性通路，其他模型部分、loss、采集和Prompt不动。现有标签够用，不需要新LLM或新标签；旧checkpoint见过检查行，不能拿来冒充baseline。本轮只生成了可行性与ID清单，没有执行训练。

如果局部差分不能稳定改善，就放弃这个具体线性通路的优先性，而不是泛称所有模型都不行；如果局部改善但Row无益，就放弃“此可修复瓶颈主要解释AL效率差”的主张。若局部与主目标都改善，可以继续研究该模型通路；不论旁路结果如何，若仍要扩大Hybrid，仍须检验LLM16在相同状态是否胜过CW后16和随机补16。这样下一轮得到的是能改变研究方向的结果，而不是又多几条难以解释的长曲线。
'''
    (OUT/'SUMMARY_FOR_RESEARCHER.md').write_text(summary)
    cjk=len(re.findall(r'[\u4e00-\u9fff]',summary))
    assert 1500<=cjk<=2500,('Summary Chinese-character budget',cjk)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
    for i,(s,color) in enumerate([(157,'#23768B'),(6101,'#C86D39')]):
        g=seed[seed.seed==s].iloc[0];x=np.array([0,1])+(-.035 if i==0 else .035)
        axes[0].plot(x,[g.llm16_cw_gain_after_n_relative_h*100,g.lcmd_back16_cw_gain_after_n_relative_h*100],'o-',c=color,label=f'Seed {s}')
        axes[1].plot(x,[g.llm16_single_sample_volume_hn_fraction,g.lcmd_back16_single_sample_volume_hn_fraction],'o-',c=color)
        axes[2].plot(x,[g.llm16_single_pe_hn_fraction,g.lcmd_back16_single_pe_hn_fraction],'o-',c=color)
    for ax in axes:ax.set_xticks([0,1],['Actual LLM16','LCMD positions 17–32']);ax.grid(axis='y',alpha=.15)
    axes[0].set(title='CW marginal coverage after fixed first16',ylabel='% of same-state initial reference distance');axes[0].legend(frameon=False)
    axes[1].set(title='Sample-volume-only control to H + first16',ylabel='Selected-slot fraction');axes[1].set_ylim(-.025,.6)
    axes[2].set(title='PE/EA-only control to H + first16',ylabel='Selected-slot fraction')
    fig.suptitle('Exact frozen-state candidate counterfactual | 71/71 ordered first16 matches | no outcome/learning counterfactual',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'hybrid_counterfactual_tradeoff.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print({'summary_chinese_characters':cjk,'summary_total_characters':len(summary),'decision_characters':len(decision)})

if __name__=='__main__':main()

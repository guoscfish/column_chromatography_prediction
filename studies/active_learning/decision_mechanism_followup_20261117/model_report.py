#!/usr/bin/env python3
"""Write focused learnability evidence and plots from already-audited deltas."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/chrom_followup_mpl')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
NAMES={'historical_free_mixed_model':'Historical CW-Free','new_free_cw_sol61_xhigh':'New CW-Free',
    'nocw':'NoCW','free16_lcmd16':'Hybrid','cw_gradient_lcmd32':'CW-LCMD'}

def markdown(df,digits=4):
    def cell(v):
        if pd.isna(v):return '不可评'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v).replace('|',' / ').replace('\n',' ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell,row))+' |' for row in df.itertuples(index=False,name=None))

def main():
    response=pd.read_csv(OUT/'response_seed_summary.csv');new=response[(response.method=='new_free_cw_sol61_xhigh')&(response.source_half=='all32')&(response.observation_threshold_ml==.5)]
    vol=new[new.axis=='V/ul'].copy();seedtable=vol[['seed','target','n','n_meaningful','observed_abs_mean','predicted_abs_mean','flat_fraction_meaningful','wrong_direction_nonflat_fraction_meaningful']]
    compact=seedtable.rename(columns={'target':'端点','n':'对数','n_meaningful':'真实变化>0.5mL对数','observed_abs_mean':'观测平均|Δ|','predicted_abs_mean':'预测平均|Δ|',
        'flat_fraction_meaningful':'有效真实变化中预测近乎平坦比例','wrong_direction_nonflat_fraction_meaningful':'有效真实变化中非平坦方向错误比例'})
    alloc=pd.read_csv(OUT/'condition_allocation.csv');a=alloc[(alloc.source_half!='all32')|alloc.method.ne('free16_lcmd16')]
    # Equal campaign weights avoid treating 35 and 36 hybrid batches as equal campaigns.
    s=a.groupby(['method','source_half','seed','axis']).history_control_fraction.mean().reset_index()
    exposure=s.groupby(['method','source_half','axis']).history_control_fraction.mean().unstack()[['PE/EA','V/ul','Volume of loading solvent/ul','loading solvent']].reset_index()
    exposure['method']=exposure.method.map(NAMES)
    exposure=exposure.rename(columns={'method':'方法','source_half':'来源','PE/EA':'PE/EA历史对照比例','V/ul':'样品体积历史对照比例',
        'Volume of loading solvent/ul':'加载体积历史对照比例','loading solvent':'加载溶剂历史对照比例'})
    phase=pd.read_csv(OUT/'response_phase_summary.csv');ph=phase[(phase.method=='new_free_cw_sol61_xhigh')&(phase.source_half=='all32')&(phase.axis=='V/ul')&(phase.observation_threshold_ml==.5)]
    levels=pd.read_csv(OUT/'sample_volume_levels.csv');lv=levels[(levels.method=='new_free_cw_sol61_xhigh')&levels.low_level.isin([50,100])&levels.high_level.isin([100,150])]
    leveltable=lv.groupby(['low_level','high_level','target']).agg(n_pairs=('n','sum'),observed_abs_seed_mean=('observed_abs_mean','mean'),predicted_abs_seed_mean=('predicted_abs_mean','mean')).reset_index()
    pair=pd.read_csv(OUT/'condition_response_pairs.csv');volume=pair[(pair.method=='new_free_cw_sol61_xhigh')&(pair.axis=='V/ul')]
    direction=new[new.axis.isin(['V/ul','PE/EA'])].groupby(['axis','target'])[['direction_accuracy_meaningful','direction_correct_nonflat_fraction_meaningful','flat_fraction_meaningful','wrong_direction_nonflat_fraction_meaningful']].mean().reset_index()
    f=pd.read_csv(OUT/'model_control_feasibility.csv')
    text=f'''# 模型条件可学习性：新证据与可证伪的最小控制

## 结论与证据范围

**有充分证据把“上样条件学习不足”列为高优先机制问题，但没有证据认定它已经解释Free低AULC的全部原因。** 这里的直接发现是当前历史checkpoint对样品体积方向的响应幅度不足；它并不自动等于架构无法表达该函数，也不能排除优化、loss尺度、训练支持不足、噪声或缺失实验变量。

已有165对、V1/V2观测平均|Δ|2.696/2.287mL而预测0.263/0.154mL的结论直接复用。该口径先在每seed内部按pair平均，再五seed等权，**不是165对直接pooled均值**；按pair权重的敏感性结果为V1 2.5249/0.3932、V2 2.3701/0.2351mL，两口径方向一致。新脚本只为本轮必要的新分层和有符号方向重新投影匹配接受反馈；与旧seed×阶段165对汇总数值误差<1e−10，没有重新计算模型特征、预测或训练。另扩展到Historical/NoCW/Hybrid真实已揭示的同批对照，区别于复述旧总体均值。

共核验791个状态的元数据，其中611个含LLM状态有自身接受批次的选前预测和真实揭示反馈。所有响应对均同分子、同轮同checkpoint、五条件恰好一个变化；反馈先经过接受ID、freeze SHA、receipt的canonical-JSON SHA校验。只在对应batch完成揭示之后作事后分析，不回填选点分数。纯CW的180状态没有同类pool选前预测/反馈产物，只有元数据对照可比较，未用别的checkpoint或其他方法的真实标签补齐。其model/predictions缓存面向test_X，未打开或冒充已选pool预测。

## 1. 幅度不足比“经常方向完全相反”更准确

观察到了什么 → New五seed在两个端点均有预测平均幅度小于观测；但异质性很大。具体数值如下。0.5mL定义“可评价真实变化”，|预测Δ|<0.1mL定义本轮描述性平坦；两阈值不是仪器分辨率，已给真实阈值0.1/0.5/1.0敏感性。

{markdown(compact)}

在真实|Δ|>0.5mL对中，先按seed算比例再等权平均：样品体积V1/V2的平坦比例为66.17%/75.70%，非平坦且方向错误仅2.67%/10.03%，非平坦且方向正确31.16%/14.26%。直接只算正负号准确率则达到86.38%/57.28%——例如887、2357的V1极小预测也经常符号正确，这不表示学到了变化幅度。故不能把“很小但同号”当作成功，也不能把flat全部算作有力的错误方向证据。

PE/EA的对应平坦比例仅1.96%/0.57%，非平坦正确方向85.87%/83.50%。同一模型对不同因素的表现明显不同，反对“所有条件变化都无法表达”的绝对命题。全部方向分母与近零阈值分层见`response_seed_summary.csv`；加载溶剂类别按字符串定向只用于保存差分，不赋予物理单调方向。

可能原因 → 条件信号通过16维非线性补充分支映射到128维、与图表示相加后再经六输出头；上样输入是density×volume并经过冻结source-train minmax。弱条件方向可能在优化中被强分子/洗脱信号掩盖；同时真实体积差可能混有未记录变量。**源码只支持输入存在和通路形式，不证明这一通路是唯一原因。**

还不能说明什么 → 上样体积真实响应也有不同符号，不能给全部数据强加单调性；这些是所选分布的同批对照，不是独立随机化实验或固定验证集。

## 2. 随预算的“恢复”主要不是跨seed普遍恢复

观察到了什么 → 157的后期V1变化开始明显，887/2357仍几乎平坦。L1005–1485期间，157有30对，观测|Δ|V1/V2为1.538/2.091，预测1.381/0.819mL；但887和2357在真实变化>0.5mL的后期对中，两端点仍100%落在预测近乎平坦区。6101后期预测平均幅度为0.107/0.073mL。

旧汇总中后期92对的预测变化平均0.545/0.309mL；排除157后62对仅0.141/0.062mL，观测仍2.676/2.344mL。改进没有消失，但强烈依赖seed。157 V1后期并非仍不能表达，它是重要反例。

可能原因 → 数据组成、不同molecule/volume水平、训练优化和checkpoint选择共同变化。还不能说明什么 → 这些对不是跨预算固定的同一评估集合，不能用分阶段选中样本宣称某个模型学习率或统一阈值。下一步需要固定标签、固定检查对和配对训练。

## 3. 实际体积水平与方法投入

常见三类水平对均出现幅度不足。表中观测/预测为先在seed内平均再等权，n是pair-event之和；罕见连续水平不并入50/100/150网格。

{markdown(leveltable)}

New165对中50→100、50→150、100→150分别37/95/27对；另6对是30→120、34.013605→102.040816、68.027211→102.040816等真实记录水平，不能伪造为统一实验网格。体积变化并不总产生同向保留响应；`sample_volume_levels.csv`保留方向/分位与分母，未做可能在真实近零时发散的逐对比值。

下表回答不同方法是否更常安排这类实验。它只计算候选与当前历史H的元数据控制关系；同一个候选可能同时有PE和样品体积对照，所以各列不可相加当预算分解。各自全程可用范围用于描述，Hybrid两seed的端点不完全一致；六组同状态的真实响应效果排名为**不可比较**，不能跨checkpoint拼出这样的比较。

{markdown(exposure)}

这些数值显示LLM更常触及模型弱响应的体积轴，但并不证明这导致较差AULC。不同轨迹的H和checkpoint分叉，纯CW无合格误差配对，不能把方法间所选残差差异当作模型能力的同状态实验。

Historical/New/NoCW同批样品体积对分别190/165/139个；Hybrid LLM16内部61对、数值16内部0对，合并32后共81对。61对LLM内部的观测平均幅度V1/V2先按两seed平均为3.644/3.439mL，预测仅0.294/0.199mL；这直接说明其新增对照中存在真实非零响应，但未识别其训练价值。加载体积/溶剂类型支持很少，只作病例性参考。

`condition_error_tags.csv`另给已经揭示的选中行在各对照标签下的误差与标准化SSE占比。它不是validation Row误差贡献，且条件标签重叠；不能以“样品体积对照占某SSE比例”归因总体学习差距。严格同状态比较最充分的证据来自本轮Hybrid后16反事实，见[HYBRID_COUNTERFACTUAL.md](HYBRID_COUNTERFACTUAL.md)。

## 4. 支持、反例、替代解释

| 判断 | 证据 | 限制 |
| --- | --- | --- |
| 上样方向是可信候选瓶颈 | 五seed均压缩；三LLM轨迹及Hybrid已揭示对照再次出现；PE明显不同 | 同批对照并非随机干预，未识别loss或架构独因 |
| 值得先检验模型能否吸收这类信息 | LLM16补充了真实变化，而当前预测近乎不变；数值后16系统不选这一关系 | “不同信息”不等于“提高Row的有效学习信息” |
| 不能认定网络绝对无法表达 | 157后期V1幅度较接近观测，PE也有强响应 | 既可能是优化问题也可能是数据支持差异 |
| 噪声/缺项仍是竞争解释 | 旧审计相同记录条件的不同row响应也不相同 | 缺真实复测身份与批次变量，不能估不可约噪声 |

## 5. 最小固定标签模型控制：仅设计，未执行

唯一模型因素：**为现有标准化上样量增加零初始化的直接线性条件通路**。每个端点新增一个系数（共两个），将w_t×x同加到该端点三个quantile的head线性量，再经过原ReLU/clamp。不增加独立width参数；ReLU/clamp仍可能间接改变最终宽度，不能声称宽度严格不变。只改这一条条件融合路径；不改输入元数据、归一化、原分支、loss权重、采集算法或LLM Prompt。共享参数初值逐张量配对；构造新模块后重置训练RNG，新增系数为零，从相同函数起点开始。该设计检验一个具体条件路径/优化假设，不把现有结构“数学上不能表达”当前提。

预定New CW五seed的L1005；L525只作为已预先列出的可行性/后续敏感性范围，不按已知gain挑状态。本轮仅按元数据哈希排序、每seed取10个完整样品体积响应系列作检查组，系列定义为同分子、PE/EA、density、加载溶剂、加载体积固定而样品体积不同。任何触及L333的系列均排除；同记录条件的全部重复row整体隔离，且不跨训练/检查泄漏。

本轮已完成的规划可行性：五seed在L1005分别有108/112/103/105/104个合格系列；固定取10组后检查行数23/24/21/20/21，实际训练行数982/981/984/985/984。没有读取这些组的真实值来选组。ID清单在`model_control_holdouts.csv`，并非待执行标签查询。全部十个L525/L1005状态均达到预定10组门槛。

初筛5seed×2臂×1配对训练RNG=**10次新拟合、0LLM、0新标签**。两臂必须从头训练H\\check，旧checkpoint见过检查行不能复用。沿用已有validation选checkpoint与固定L333尺度，检查组只用于最终局部差分评估；主目标仍为保存Row validation NRMSE。若需估训练波动，再预定增加两套配对RNG共20次，三RNG总30次拟合；outer seed才是统计单位。

预先停止准则（未来提案，不是本轮已验证阈值）：局部差分MAE至少下降20%、至少4/5seed同方向，且Row NRMSE相对恶化不超过1%。如果只改善局部差分而Row无收益，放弃“此瓶颈主要解释AL效率差距”的主张；如果局部也无稳定改进，不扩展此线性旁路并降低该具体可修复通路假设的优先级；这不否定所有非线性条件模型或训练瓶颈，数据支持/记录噪声仍需竞争性解释。即便通过，也还须同状态选择比较才能证明LLM在新模型下优于LCMD或随机补充。

本轮没有执行上述训练、读取test truth或给反事实候选查询标签。
'''
    (OUT/'MODEL_LEARNABILITY.md').write_text(text)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for t,marker,color in [('V1','o','#167E9C'),('V2','s','#D3733E')]:
        z=vol[vol.target==t].set_index('seed').reindex([157,887,2357,6101,12203])
        axs[0].scatter(z.observed_abs_mean,z.predicted_abs_mean,marker=marker,color=color,label=t,s=35)
        for seed,row in z.iterrows():
            if seed==157:axs[0].annotate('157 '+t,(row.observed_abs_mean,row.predicted_abs_mean),xytext=(4,4),textcoords='offset points',fontsize=8)
    axs[0].plot([0,4.5],[0,4.5],c='gray',ls=':',lw=.8);axs[0].set(xlabel='Observed mean |delta| (mL)',ylabel='Predicted mean |delta| (mL)',title='Same-batch sample-volume response');axs[0].legend(frameon=False)
    x=np.arange(4);labels=['Vol V1','Vol V2','PE V1','PE V2'];vals=[]
    for axis,t in [('V/ul','V1'),('V/ul','V2'),('PE/EA','V1'),('PE/EA','V2')]:
        v=direction[(direction.axis==axis)&(direction.target==t)].iloc[0]
        vals.append([v.flat_fraction_meaningful,v.wrong_direction_nonflat_fraction_meaningful,v.direction_correct_nonflat_fraction_meaningful])
    vals=np.array(vals);bottom=np.zeros(4)
    for j,color,label in [(0,'#9BA8B4','Near-flat prediction'),(1,'#C05B40','Non-flat wrong sign'),(2,'#238B82','Non-flat correct sign')]:
        axs[1].bar(x,vals[:,j],bottom=bottom,color=color,label=label);bottom+=vals[:,j]
    axs[1].set_xticks(x,labels);axs[1].set(title='Observed |delta| > 0.5 mL',ylabel='Equal-seed fraction');axs[1].legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.11))
    for seed,g in ph[ph.target=='V1'].groupby('seed'):
        g=g.set_index('phase').reindex(['333-525','525-813','813-1005','1005-1485'])
        axs[2].plot(np.arange(4),g.predicted_abs_mean,marker='o',label=str(seed),lw=1.2)
    axs[2].set_xticks(range(4),['333–525','525–813','813–1005','1005–1485'],rotation=25)
    axs[2].set(title='V1 prediction recovery is heterogeneous',ylabel='Mean predicted |delta| (mL)',xlabel='Label-budget phase');axs[2].legend(fontsize=7,frameon=False)
    fig.suptitle('Condition learnability | New CW-Free, five seeds | selected observed pairs, not independent interventions',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'condition_learnability.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('MODEL_LEARNABILITY written',len(text))

if __name__=='__main__':main()

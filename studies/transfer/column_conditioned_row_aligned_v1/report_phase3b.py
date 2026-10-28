"""Tables, three static scientific figures, and a developmental conclusion from locked scores."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_phase3b as b

STUDY,old,s=b.STUDY,b.old,b.s
RED,BLUE='#8C0000','#607D8B'
SEEDS=list(s.SEEDS)
FOOT='DEVELOPMENTAL CONFIRMATION | Historical test exposed | Seeds share records'
METRICS=[*b.LOSS_METRICS,'V1_r2','V2_r2']


def md(frame,digits=4):
    def cell(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def derive():
    result=pd.read_csv(STUDY/'OUTER_ROW_RESULTS.csv')
    paired=pd.read_csv(STUDY/'OUTER_ROW_PAIRED_COMPARISONS.csv')
    tails=pd.read_csv(STUDY/'OUTER_ROW_TAIL_DIAGNOSTICS.csv')
    history=pd.read_csv(STUDY/'GROUP_VS_ROW_OUTER_COMPARISON.csv')
    baselines=pd.read_csv(STUDY/'OUTER_ROW_BASELINE_RESULTS.csv')
    summaries=[]
    for c in s.TARGETS:
        p=paired.loc[paired.column.eq(c)]
        for arm in old.ARMS:
            part=result.loc[result.column.eq(c)&result.arm.eq(arm)]
            summaries.append({'column':c,'arm':arm,'seeds':5,
                              **{f'{name}_mean':part[name].mean() for name in METRICS},
                              **{f'{name}_sd':part[name].std(ddof=1) for name in METRICS},
                              'mean_paired_seed_gain':p.combined_normalized_rmse_relative_gain.mean(),
                              'gain_of_means':1-p.A2_combined_normalized_rmse.mean()/p.A1_combined_normalized_rmse.mean(),
                              'A2_seed_wins':int(p.A2_seed_win.sum())})
    summary=pd.DataFrame(summaries)
    tail_pairs=[]
    for (seed,c,e),part in tails.groupby(['outer_seed','column','endpoint']):
        a,aa=[part.loc[part.arm.eq(arm)].iloc[0] for arm in old.ARMS]
        s.require(a.tail_count==aa.tail_count and a.threshold==aa.threshold,'unpaired tail masks')
        tail_pairs.append({'outer_seed':int(seed),'column':c,'endpoint':e,'threshold_ml':a.threshold,'tail_count':int(a.tail_count),
                           'A1_tail_rmse':a.tail_rmse,'A2_tail_rmse':aa.tail_rmse,
                           'tail_relative_gain':1-aa.tail_rmse/a.tail_rmse if a.tail_rmse>0 else np.nan,
                           'A2_tail_deteriorated':bool(aa.tail_rmse>a.tail_rmse)})
    tail_pair=pd.DataFrame(tail_pairs)
    tail_summary=[]
    for (c,e),g in tail_pair.groupby(['column','endpoint']):
        tail_summary.append({'column':c,'endpoint':e,'A1_mean_seed_tail_rmse':g.A1_tail_rmse.mean(),
                             'A2_mean_seed_tail_rmse':g.A2_tail_rmse.mean(),'A1_seed_tail_sd':g.A1_tail_rmse.std(ddof=1),
                             'A2_seed_tail_sd':g.A2_tail_rmse.std(ddof=1),'mean_paired_seed_tail_gain':g.tail_relative_gain.mean(),
                             'gain_of_seed_tail_means':1-g.A2_tail_rmse.mean()/g.A1_tail_rmse.mean(),
                             'A2_deteriorating_seeds':int(g.A2_tail_deteriorated.sum())})
    hist_summary=[]
    for c in s.TARGETS:
        g=history.loc[history.column.eq(c)]
        for cv in ['Group_CV','Row_CV']:
            a,aa=[g.loc[g.arm.eq(arm)].set_index('outer_seed').loc[SEEDS] for arm in old.ARMS]
            gains=1-aa[cv+'_combined_normalized_rmse']/a[cv+'_combined_normalized_rmse']
            for arm,part in [('A1',a),('A2',aa)]:
                hist_summary.append({'column':c,'inner_cv':cv,'arm':arm,
                                     'mean_test_nrmse':part[cv+'_combined_normalized_rmse'].mean(),
                                     'seed_sd_test_nrmse':part[cv+'_combined_normalized_rmse'].std(ddof=1),
                                     'mean_paired_seed_gain':gains.mean(),
                                     'gain_of_means':1-aa[cv+'_combined_normalized_rmse'].mean()/a[cv+'_combined_normalized_rmse'].mean(),
                                     'A2_seed_wins':int((gains>0).sum())})
    baseline_summary=[]
    for (c,method),ref in baselines.groupby(['column','method']):
        ref=ref.set_index('outer_seed').loc[SEEDS]
        for arm in old.ARMS:
            candidate=result.loc[result.column.eq(c)&result.arm.eq(arm)].set_index('outer_seed').loc[SEEDS]
            gain=1-candidate.combined_normalized_rmse/ref.combined_normalized_rmse
            baseline_summary.append({'column':c,'reference':method,'arm':arm,'reference_mean_nrmse':ref.combined_normalized_rmse.mean(),
                                     'reference_seed_sd':ref.combined_normalized_rmse.std(ddof=1),
                                     'candidate_mean_nrmse':candidate.combined_normalized_rmse.mean(),
                                     'mean_paired_seed_gain':gain.mean(),
                                     'gain_of_means':1-candidate.combined_normalized_rmse.mean()/ref.combined_normalized_rmse.mean(),
                                     'candidate_seed_wins':int((gain>0).sum())})
    trades=[]
    for row in paired.itertuples():
        if row.combined_normalized_rmse_relative_gain>0:
            for endpoint in ['V1','V2','Center','Width']:
                gain=getattr(row,endpoint+'_max_abs_error_relative_gain')
                if gain<0:trades.append({'outer_seed':row.outer_seed,'column':row.column,'endpoint':endpoint,
                                         'nrmse_relative_gain':row.combined_normalized_rmse_relative_gain,'max_error_relative_gain':gain})
    tables={'OUTER_ROW_SUMMARY.csv':summary,'OUTER_ROW_TAIL_PAIRED_COMPARISONS.csv':tail_pair,
            'OUTER_ROW_TAIL_SUMMARY.csv':pd.DataFrame(tail_summary),'GROUP_VS_ROW_OUTER_SUMMARY.csv':pd.DataFrame(hist_summary),
            'OUTER_ROW_BASELINE_COMPARISONS.csv':pd.DataFrame(baseline_summary),'OUTER_ROW_MAX_ERROR_TRADEOFFS.csv':pd.DataFrame(trades)}
    for name,frame in tables.items():frame.to_csv(STUDY/name,index=False)
    return result,paired,tails,history,baselines,tables


def savefig(fig,name):
    fig.savefig(STUDY/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(STUDY/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def figures(result,paired,history,tail_pairs):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    fig.subplots_adjust(bottom=.24,top=.77,wspace=.23)
    limits=paired.combined_normalized_rmse_relative_gain.to_numpy()*100
    lo,hi=min(-2.,limits.min()-2),max(4.,limits.max()+2)
    for ax,c in zip(axes,s.TARGETS):
        p=paired.loc[paired.column.eq(c)].set_index('outer_seed').loc[SEEDS]
        gain=p.combined_normalized_rmse_relative_gain.to_numpy()*100
        mean=gain.mean();ratio=(1-p.A2_combined_normalized_rmse.mean()/p.A1_combined_normalized_rmse.mean())*100
        ax.bar(np.arange(5),gain,color=[RED if v>=0 else BLUE for v in gain],width=.58)
        ax.axhline(0,color='#202020',lw=1.2);ax.axhline(mean,color=BLUE,lw=1.3,ls='--')
        ax.set(xticks=range(5),xticklabels=[f'S{i+1}\n{seed}' for i,seed in enumerate(SEEDS)],ylim=(lo,hi),
               ylabel='A2 vs A1 test NRMSE gain (%)',title=f'{c}  |  A2 seed wins {int(p.A2_seed_win.sum())}/5')
        ax.tick_params(axis='x',labelsize=8.5);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(gain):ax.text(i,v+(.35 if v>=0 else -.35),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
        ax.text(.03,1.09,f'Mean paired gain {mean:+.2f}%\nGain of means {ratio:+.2f}%',transform=ax.transAxes,fontsize=10)
    fig.suptitle('Frozen outer Row test: A1 vs A2-FiLM',fontsize=16,y=.97)
    fig.text(.5,.035,FOOT,ha='center',fontsize=9,color='#505050')
    savefig(fig,'phase3b_outer_row_paired_gains')

    fig,axes=plt.subplots(1,2,figsize=(12,5.1))
    fig.subplots_adjust(bottom=.2,top=.75,wspace=.25)
    for ax,c in zip(axes,s.TARGETS):
        g=history.loc[history.column.eq(c)]
        for x,(cv,arm) in enumerate([('Group_CV','A1'),('Group_CV','A2'),('Row_CV','A1'),('Row_CV','A2')]):
            values=g.loc[g.arm.eq(arm)].set_index('outer_seed').loc[SEEDS,cv+'_combined_normalized_rmse'].to_numpy()
            color=BLUE if arm=='A1' else RED
            ax.bar(x,values.mean(),color=color,alpha=.45 if cv=='Group_CV' else .95,width=.64,
                   edgecolor=color,hatch='//' if cv=='Group_CV' else None)
            ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt='none',ecolor='#303030',capsize=4,lw=1)
            ax.scatter(x+np.linspace(-.16,.16,5),values,color='#303030',s=20,zorder=4)
            ax.text(x,values.mean()+.012,f'{values.mean():.4f}',ha='center',va='bottom',fontsize=9,
                    bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':1})
        ax.set(xticks=range(4),xticklabels=['Group-CV\nA1','Group-CV\nA2','Row-CV\nA1','Row-CV\nA2'],
               ylabel='Combined NRMSE (lower is better)',title=c,ylim=(0,1.))
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Inner selection protocol: same frozen outer Row test identities',fontsize=15,y=.97)
    fig.text(.5,.85,'Bars: five-seed means  |  Whiskers: descriptive SD  |  Dots: all five seeds',ha='center',fontsize=10)
    fig.text(.5,.05,FOOT+'\nTraining epochs also changed; no equal-step causal attribution.',ha='center',fontsize=9,color='#505050')
    savefig(fig,'phase3b_group_vs_row_outer')

    labels=['V1 RMSE','V2 RMSE','V1 MAE','V2 MAE','Center RMSE','Width RMSE','Tail V1 RMSE','Tail V2 RMSE',
            'Max |V1 error|','Max |V2 error|','Max |Center error|','Max |Width error|']
    names=['V1_rmse','V2_rmse','V1_mae','V2_mae','Center_rmse','Width_rmse','tail_V1','tail_V2',
           'V1_max_abs_error','V2_max_abs_error','Center_max_abs_error','Width_max_abs_error']
    markers=['o','s','^','D','v']
    fig,axes=plt.subplots(1,2,figsize=(13,7.8),sharey=True)
    fig.subplots_adjust(left=.16,right=.98,bottom=.2,top=.82,wspace=.16)
    all_gains=[]
    for ax,c in zip(axes,s.TARGETS):
        a,aa=[result.loc[result.column.eq(c)&result.arm.eq(arm)].set_index('outer_seed').loc[SEEDS] for arm in old.ARMS]
        bars=[];seed_values=[]
        for name in names:
            if name.startswith('tail_'):
                pair=tail_pairs.loc[tail_pairs.column.eq(c)&tail_pairs.endpoint.eq(name.split('_')[1])].set_index('outer_seed').loc[SEEDS]
                left,right=pair.A1_tail_rmse.to_numpy(),pair.A2_tail_rmse.to_numpy()
            else:left,right=a[name].to_numpy(),aa[name].to_numpy()
            gain=100*(1-right/left);seed_values.append(gain)
            bars.append(100*(1-right.mean()/left.mean()));all_gains.extend(gain)
        ax.barh(np.arange(12),bars,color=[RED if v>=0 else BLUE for v in bars],height=.54,alpha=.82,zorder=2)
        for i,marker in enumerate(markers):
            ax.scatter(np.array(seed_values)[:,i],np.arange(12)+(i-2)*.08,s=23,marker=marker,
                       color='#252525',edgecolors='white',linewidths=.3,zorder=3)
        ax.axvline(0,color='#202020',lw=1.2)
        ax.set(yticks=range(12),yticklabels=labels,title=c,xlabel='A2 vs A1 error gain (%)')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    # Both panels share the same numerical x scale; retain all negative seeds.
    bound=(min(all_gains)-5,max(all_gains)+5)
    for ax in axes:ax.set_xlim(bound)
    fig.suptitle('Endpoints, Center/Width, tail and maximum-error tradeoffs',fontsize=15,y=.965)
    legend=[Line2D([0],[0],marker=m,color='none',markerfacecolor='#252525',markersize=5,label=f'S{i+1}') for i,m in enumerate(markers)]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.57,.915),ncol=5,frameon=False)
    fig.text(.5,.075,'Bars: gain of five-seed metric means  |  Markers: each paired-seed gain\nPositive = A2 improves; negative = A2 worsens. Tail thresholds fixed from gradient_train.',ha='center',fontsize=10)
    fig.text(.5,.02,FOOT,ha='center',fontsize=9,color='#505050')
    savefig(fig,'phase3b_endpoint_tail_max_tradeoffs')


def write_report(result,paired,tails,history,baselines,tables):
    summary=tables['OUTER_ROW_SUMMARY.csv'];tail_pair=tables['OUTER_ROW_TAIL_PAIRED_COMPARISONS.csv']
    tail_summary=tables['OUTER_ROW_TAIL_SUMMARY.csv'];hist_summary=tables['GROUP_VS_ROW_OUTER_SUMMARY.csv']
    baseline_summary=tables['OUTER_ROW_BASELINE_COMPARISONS.csv'];trade=tables['OUTER_ROW_MAX_ERROR_TRADEOFFS.csv']
    main=summary[['column','arm','combined_normalized_rmse_mean','combined_normalized_rmse_sd','mean_paired_seed_gain','gain_of_means','A2_seed_wins']].copy()
    main['mean_paired_seed_gain']*=100;main['gain_of_means']*=100
    main=main.rename(columns={'combined_normalized_rmse_mean':'mean_NRMSE','combined_normalized_rmse_sd':'seed_SD',
                             'mean_paired_seed_gain':'paired_gain_%','gain_of_means':'gain_of_means_%'})
    seedtable=paired[['column','outer_seed','A1_combined_normalized_rmse','A2_combined_normalized_rmse','combined_normalized_rmse_relative_gain','A2_seed_win']].copy()
    seedtable['combined_normalized_rmse_relative_gain']*=100
    seedtable=seedtable.rename(columns={'A1_combined_normalized_rmse':'A1_NRMSE','A2_combined_normalized_rmse':'A2_NRMSE','combined_normalized_rmse_relative_gain':'paired_gain_%'})
    endpoint=summary[['column','arm',*[name+'_mean' for name in ['V1_rmse','V2_rmse','V1_mae','V2_mae','V1_r2','V2_r2','Center_rmse','Width_rmse']]]]
    max_summary=[]
    for (c,arm),g in result.groupby(['column','arm']):
        max_summary.append({'column':c,'arm':arm,**{f'{e}_mean_max_abs':g[e+'_max_abs_error'].mean() for e in ['V1','V2','Center','Width']},
                            **{f'{e}_worst_seed_max_abs':g[e+'_max_abs_error'].max() for e in ['V1','V2','Center','Width']}})
    hist=hist_summary.copy()
    hist['mean_paired_seed_gain']*=100;hist['gain_of_means']*=100
    hist=hist.rename(columns={'mean_paired_seed_gain':'mean_paired_seed_gain_%','gain_of_means':'gain_of_means_%'})
    taildisplay=tail_summary[['column','endpoint','A1_mean_seed_tail_rmse','A2_mean_seed_tail_rmse','gain_of_seed_tail_means','A2_deteriorating_seeds']].copy()
    taildisplay['gain_of_seed_tail_means']*=100;taildisplay=taildisplay.rename(columns={'gain_of_seed_tail_means':'gain_of_tail_means_%'})
    tailindividual=tail_pair.copy();tailindividual['tail_relative_gain']*=100
    tailindividual=tailindividual.rename(columns={'tail_relative_gain':'paired_tail_gain_%'})
    base=baseline_summary.copy();base['mean_paired_seed_gain']*=100;base['gain_of_means']*=100
    base=base.rename(columns={'mean_paired_seed_gain':'mean_paired_seed_gain_%','gain_of_means':'gain_of_means_%'})
    epochs=history.loc[history.column.eq('25g'),['outer_seed','arm','Group_CV_epochs','Row_CV_epochs','Group_CV_steps','Row_CV_steps']]
    detailed_metrics=result[['column','outer_seed','arm','V1_rmse','V2_rmse','V1_mae','V2_mae','V1_r2','V2_r2','Center_rmse','Width_rmse']]
    maxima=result[['column','outer_seed','arm',*[e+'_max_abs_error' for e in ['V1','V2','Center','Width']]]]
    td=trade.copy();td['nrmse_relative_gain']*=100;td['max_error_relative_gain']*=100
    td=td.rename(columns={'nrmse_relative_gain':'nrmse_gain_%','max_error_relative_gain':'max_error_gain_%'})
    text=f'''# Phase 3B：最终 Row 外层评分与研究收尾

日期：2026-10-14（Asia/Shanghai）。**本次完成一次预定 test 评分，未训练、修改checkpoint、修补预测或启动新候选。最终判定为 C：内部改善未稳定转化为跨柱外层收益。** 25g有局部收益，40g仍出现内部到外层的收益落差；这不是两柱均退化的结论。

## 冻结、评价计划与一次评分边界

在读取任何test响应之前，十个refit、四十个预测上下文/1852条记录、SHA256、六输出列顺序、完整状态/BN、原Row角色身份、训练标签身份、最终epoch和实际steps全部核验通过，冻结文件仍只读。Phase3A的137个冻结文件、Phase2的56个报告文件、原Phase1归档及历史1069文件不变。具体证据见 PHASE3B_PREFLIGHT_AUDIT.json、PHASE3B_SCORE_MANIFEST.json 和最终完整性清单。

EVALUATION_PLAN.md为唯一评分口径，与本轮指令没有实质冲突。仅评分 **test**，validation盲预测同样经过身份/有限性检查，但未解析validation真实终点，也未与test混合。先按每seed/柱gradient_train重算V1/V2 SD(ddof=0)和80% tail阈值并与Phase3A保存值核对，再只读取十个获准test上下文，共860条跨seed记录（重复记录并非独立重复）。标准point_metrics/metric_rows来自原项目，没有重新定义NRMSE、使用test尺度或重新选tail阈值。

冻结计划要求的最大误差和Tail均报告；新增R²使用同一项目标准实现。4项合成测试通过，验证标准指标、tail包含阈值边界/空tail=NA、两种改善率不同及评分前冻结边界。评分入口遇到已有完成receipt只校验结果，不重复读取标签或评分。所有新结果通过清单锁定，原冻结预测保持不变。

## 主要结果：新 Row-CV → FULL refit → Row test

SD为五seed描述性标准差(ddof=1)，不表示五次独立物理重复或置信区间。paired gain = mean_seed(1−A2/A1)，gain of means = 1−mean_seed(A2)/mean_seed(A1)，正值为A2改善。

{md(main)}

{md(seedtable)}

25g平均配对收益 **+3.061%**，gain of means **+2.755%**，4/5 seed wins；536279090退化约4.96%，2767143051仅改善约0.37%。40g平均配对收益 **−0.851%**，gain of means **−1.067%**，仅1/5 seed wins；2767143051改善约3.08%，其余四seed均退化。不能把25g的正结果或40g的小幅均值变化概括为两柱稳定受益。

![A1/A2 paired outer gains](phase3b_outer_row_paired_gains.png)

## 端点、Center/Width 与性能代价

以下为五seed指标均值，误差单位mL；R²无单位。完整均值/SD在 OUTER_ROW_SUMMARY.csv，逐seed如下。

{md(endpoint)}

{md(detailed_metrics)}

25g：V1/V2 RMSE的gain of means约+2.31%/+3.24%，MAE约+2.93%/+2.92%；Center +3.19%，Width +2.11%。40g：Width RMSE改善约5.70%，但Center恶化约2.46%；V2 RMSE恶化约2.03%，V1/V2 MAE分别恶化约1.66%/2.97%。因此40g的Width改善并未转化为更好的Combined NRMSE，不能只报告Width。

Tail使用各seed/柱全部gradient_train端点80%分位，固定阈值、同一真实test掩码，不根据test分布重选。每seed先计算tail SSE/count再开根号，跨seed只平均五个tail RMSE；不将共享记录堆叠成独立物理样本。详细阈值、count、SSE及全部seed如下。

{md(taildisplay)}

{md(tailindividual)}

25g V1 tail平均RMSE由8.314增至8.835mL，**恶化6.275%**；536279090和1425370602分别恶化23.18%和55.02%。后者虽然整体NRMSE改善4.91%，V1 tail明显恶化。25g V2 tail均值改善3.31%，但同样两seed退化。40g V1/V2 tail均值改善4.18%/7.87%；V1仍有两个seed退化，V2五seed均改善（1362771960仅约0.03%）。共6/20个柱×seed×endpoint tail对照退化。

### 最大绝对误差

先在每seed内部取最大误差，再报告五seed最大误差的均值和最坏seed的最大值，不把这些数值当RMSE或平均样本误差。对应sample_id保存在 OUTER_ROW_RESULTS.csv。

{md(pd.DataFrame(max_summary))}

{md(maxima)}

25g的五seed最大绝对误差均值中，A2的V1/V2/Center分别增大约1.41%/0.27%/2.37%，Width改善约3.08%；40g的V2最大误差均值增大约5.15%，Width改善约8.57%。在NRMSE改善的五个柱×seed上下文、共20个输出组合中，**9个输出最大误差反而增加**。具体负结果：

{md(td)}

三个V1分位数交叉保持原样，均位于1425370602的25g test：A1一条、A2两条（q50>q90）。全部预测有限且满足原[0,1e8]输出截断；无V2_q50<V1_q50，不新增单调排序或物理投影。此诊断不影响样本保留或评分。

![Endpoint tail maximum tradeoffs](phase3b_endpoint_tail_max_tradeoffs.png)

## 历史 Group-CV 模型选择的描述性对照

历史 ROW_OUTER_RESULTS.csv的score/freeze链、原split/source/过滤人口/预测目标、全部test身份以及模型与指标实现均通过匹配核验。使用同一test真值、FULL训练尺度重算历史A1/A2指标，全部与原冻结表一致（容差1e−10）；历史产物未改写，证据见 PHASE3B_HISTORICAL_MATCH_AUDIT.json。HIER仍因原冻结源码hash不匹配排除，未因本次结果重新引入。

{md(hist)}

旧40g A2为 **0/5 wins、平均配对−9.834%**，新协议为 **1/5、−0.851%**：五seed全部退化的现象改变为四seed退化，明显缩小差距，尚未反转为可重复收益。旧25g A2为2/5、−2.092%，新25g为4/5、+3.061%；旧负结果保留。

从Group-CV到Row-CV，25g A1的test NRMSE均值恶化约0.70%（仅2/5 seed变好），A2改善约3.14%（3/5变好）；40g A1改善约4.47%（3/5变好），A2改善约11.72%（5/5变好）。A1/A2变化方向并非所有柱一致；A2改善幅度更大，不代表新A2已优于新A1。

两协议最终训练预算如下。各上下文为1 step/epoch。

{md(epochs)}

A2五seed从Group-CV的69–112 epoch变为Row-CV的238–290 epoch；A1变化方向混合。这些外层差异可能与更晚的选择和更长的训练有关，同时还涉及内层验证任务变化。本实验无equal-step对照，不能将改善单独归因于FiLM模块，也不能证明增加epoch是唯一机制。不同内层CV的绝对NRMSE不在同一验证集上，不用于协议优劣比较。

![Group Row outer comparison](phase3b_group_vs_row_outer.png)

## 简单校准与其他匹配工作流参照

在读取test响应前，已锁定并复核Scale、P0/P1共30个参照上下文，无根据成绩增删。相同FULL目标训练人口、source checkpoint、Row test身份及冻结预测均通过核验；本轮重新计算使用本研究同一训练尺度/tail阈值。Scale只提供q50端点，不能对占位六列作分位数有效性结论。P0/P1采用Group-CV选epoch，是完整参数迁移工作流参考，不是与A1/A2相同内层split或单一架构净效应。

{md(base)}

两种多任务模型相对Scale在两柱均5/5 seed wins；A1 gain of means约25g +13.70%、40g +29.92%，A2约+16.08%、+29.17%。它们也在均值上优于P0/P1，但25g相对P0仅3/5 wins。多任务优于当前这个匹配Scale，不等于FiLM优于A1，也不等于优于所有校准方法。源source_train辅助监督在多任务流程中继续使用，与单柱校准不同，差异不能解释成纯结构因果效应。

Affine没有兼容FULL冻结证据；paper-style等旧outer-validation选择权限不同，未进入主要同协议对照；HIER的源码冻结不匹配保持排除。没有重训任何历史方法。

## 最终研究判断与第19页证据

**四类中选择 C。** 原因是40g内部Row-CV平均配对+3.605%、4/5 wins，最终test却−0.851%、1/5 wins；Row-CV对齐仍不足以解决该柱内部到外层收益落差。25g的内部+3.701%转为test+3.061%、4/5 wins，具有局部预测价值，但V1 tail和最大误差存在代价，不能掩盖40g的失败。内部最优checkpoint的验证结果包含选择乐观性，且验证/test身份不同，不能把内部3%–4%当作外层有效性的替代证据。

第19页最值得展示两项证据：

1. **收益不跨柱稳定转化**：25g +3.06%/4 wins，40g −0.85%/1 win；40g由旧−9.83%/0 wins缩小差距但仍未胜过A1。图1/图2提供主结果和历史边界。
2. **均值改善存在鲁棒性代价**：25g NRMSE均值改善2.75%，V1 tail均值却恶化6.28%，一个seed tail恶化55.02%，且最大误差负结果保留。图3展示全部seed，不仅展示有利均值。

建议把 **A1作为跨25g/40g共同工作基线优先保留**；A2保留为开发性研究候选，不作为已被当前数据证明的统一升级。若强调25g平均误差，A2有局部价值，但必须同时披露尾部代价。就当前匹配的预测表现，未支持退回Scale；其计算/部署更简单的工程价值不是本次精度比较能否定的。以上是科学/使用建议，不修改任何已冻结arm或据test再选checkpoint，也不启动新研究。

## 证据边界与停止

历史Row test与Phase1/2结果在新协议设计前已曝光。因此所有新旧对照只能称 **DEVELOPMENTAL CONFIRMATION**，不是独立外部确认或无偏泛化能力证明。五seed共享大量历史记录、独立物理重复标识未知；不做独立样本显著性检验、不给出未经支持的p值。仍需未曝光独立数据才能进行新的独立确认，本轮不会自动获得或评分该数据。

**Phase3B已完成并停止。** 评分来源/标签请求/冻结链、原始test结果、全部负诊断和三张PNG/SVG图均保存；没有继续训练、搜索、后处理或新候选测试。
'''
    (STUDY/'PHASE3B_REPORT.md').write_text(text)
    decisions=[]
    for c in s.TARGETS:
        left=summary.loc[summary.column.eq(c)&summary.arm.eq('A1')].iloc[0]
        right=summary.loc[summary.column.eq(c)&summary.arm.eq('A2')].iloc[0]
        decisions.append({'column':c,'A1_test_nrmse_mean':float(left.combined_normalized_rmse_mean),
                          'A2_test_nrmse_mean':float(right.combined_normalized_rmse_mean),
                          'mean_paired_seed_gain':float(left.mean_paired_seed_gain),'gain_of_means':float(left.gain_of_means),
                          'A2_seed_wins':int(left.A2_seed_wins),'seeds':5})
    old.write_json(STUDY/'FINAL_SCIENTIFIC_DECISION.json',{
        'decision':'C','decision_label':'内部改善未稳定转化为跨柱最终Row收益；40g仍有收益落差',
        'classification':'DEVELOPMENTAL CONFIRMATION','date_Asia_Shanghai':'2026-10-14','column_results':decisions,
        'FiLM_internal_gains_stably_translate_across_columns':False,
        'historical_40g_all_seed_deterioration_changed_to_four_of_five':True,
        '25g_local_prediction_value_with_tail_and_maximum_error_costs':True,
        'recommendation':'A1优先作为两柱共同工作基线；保留A2研究候选；当前精度证据不支持退回已匹配Scale',
        'recommendation_is_not_new_checkpoint_or_refit_selection':True,'both_frozen_arms_preserved':True,
        'tail_deteriorating_seed_column_endpoints':int(tail_pair.A2_tail_deteriorated.sum()),
        'nrmse_improving_but_max_error_worsening_combinations':len(trade),'unchanged_V1_quantile_crossings':3,
        'HIER_excluded_due_to_frozen_source_hash_mismatch':True,'no_independent_external_confirmation':True,
        'no_independent_sample_significance_tests':True,'new_training_fits':0,'new_candidate_tests':0,
        'prediction_freeze_sha256':old.sha(STUDY/'PREDICTION_FREEZE_MANIFEST.json'),
        'evaluation_plan_sha256':old.sha(STUDY/'EVALUATION_PLAN.md'),'score_manifest_sha256':old.sha(STUDY/'PHASE3B_SCORE_MANIFEST.json'),
        'phase':'PHASE3B_COMPLETE_STOPPED'},immutable=True)


def main():
    b.verify_completed()
    data=derive()
    figures(data[0],data[1],data[3],data[5]['OUTER_ROW_TAIL_PAIRED_COMPARISONS.csv'])
    write_report(*data)
    print('Three PNG/SVG figures and final developmental report saved; no score/training action invoked.')


if __name__=='__main__':main()

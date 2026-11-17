#!/usr/bin/env python3
"""Summarize all fifteen fixed fits without any model selection or fitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/volume_controls_mpl')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
KINDS=['A_original','B_amount_skip','C_volume_skip']
LABELS={'A_original':'A Original','B_amount_skip':'B Amount skip','C_volume_skip':'C Volume skip'}
COLORS=['#475569','#c66a24','#166b79']

def md(frame,digits=5):
    cols=list(frame.columns)
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    coverage=pd.read_csv(OUT/'fit_coverage.csv')
    assert len(coverage)==15 and coverage.complete.all(),'All fifteen declared fits must finish first'
    assert len(list((OUT/'fits').glob('seed_*/*/started.json')))==15
    assert not list((OUT/'fits').glob('seed_*/*/FAILED.json'))
    metrics=pd.read_csv(OUT/'row_metrics.csv');local=pd.read_csv(OUT/'local_response_metrics.csv')
    check=local.query("scope=='check' and axis=='V/ul' and observation_threshold_ml==.5")
    primary=check.groupby(['seed','kind'])[['difference_error_mean','observed_abs_mean','predicted_abs_mean','amplitude_error_mean_ml','direction_accuracy_meaningful','flat_fraction_meaningful']].mean().reset_index()
    row=metrics.query("scope=='validation'")
    joined=row.merge(primary,on=['seed','kind'],validate='one_to_one')
    assert len(joined)==15 and not joined.duplicated(['seed','kind']).any()
    assert np.isfinite(joined[['difference_error_mean','combined_normalized_rmse']].to_numpy()).all()
    joined.to_csv(OUT/'control_seed_summary.csv',index=False)
    summary=joined.groupby('kind').mean(numeric_only=True).reindex(KINDS)
    summary.to_csv(OUT/'control_equal_seed_summary.csv')
    paired=pd.read_csv(OUT/'paired_model_differences.csv');gates=[]
    for name,first,second in [('B-A',KINDS[1],KINDS[0]),('C-A',KINDS[2],KINDS[0]),('C-B',KINDS[2],KINDS[1])]:
        a,b=summary.loc[first],summary.loc[second];p=paired[paired.comparison==name]
        improve=1-a.difference_error_mean/b.difference_error_mean
        rowchange=a.combined_normalized_rmse/b.combined_normalized_rmse-1
        positive=int((p.local_difference_mae_delta<0).sum());rowpositive=int((p.row_nrmse_delta<=0).sum())
        gates.append(dict(comparison=name,local_improvement_ratio_equal_seed_means=improve,local_positive_seeds=positive,
            mean_paired_seed_relative_local_improvement=float((-p.local_relative_delta).mean()),
            row_relative_change_equal_seed_means=rowchange,row_nonworse_seeds=rowpositive,
            row_worse_over_one_percent_seeds=int((p.row_relative_delta>.01).sum()),
            local_twenty_percent=bool(improve>=.2),local_four_seeds=positive>=4,row_mean_not_worse_over_one_percent=bool(rowchange<=.01),
            mechanism_reference_pass=bool(improve>=.2 and positive>=4 and rowchange<=.01),
            row_relevance_reference_pass=bool(rowchange<=-.01 and rowpositive>=4)))
    pd.DataFrame(gates).to_csv(OUT/'mechanism_gates.csv',index=False)
    (OUT/'control_decision_numbers.json').write_text(json.dumps(dict(aggregation='Pair mean within endpoint, two endpoints equally weighted within seed, five seeds equally weighted; ratios of these means for reference gates.',gates=gates,summary=summary.reset_index().to_dict('records')),indent=2)+'\n')
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for i,k in enumerate(KINDS):
        g=joined[joined.kind==k].sort_values('seed')
        axes[0].scatter(np.full(len(g),i),g.difference_error_mean,color=COLORS[i],s=36,zorder=3)
        axes[1].scatter(np.full(len(g),i),g.combined_normalized_rmse,color=COLORS[i],s=36,zorder=3)
    for _,g in joined.groupby('seed'):
        g=g.set_index('kind').reindex(KINDS)
        axes[0].plot(range(3),g.difference_error_mean,color='#94a3b8',alpha=.5,lw=1)
        axes[1].plot(range(3),g.combined_normalized_rmse,color='#94a3b8',alpha=.5,lw=1)
    axes[0].set_ylabel('Held-out volume difference MAE (mL)');axes[1].set_ylabel('Original Row validation NRMSE')
    for ax in axes:ax.set_xticks(range(3),['A Original','B Amount','C Volume']);ax.grid(axis='y',alpha=.15)
    fig.suptitle('H1005 minus fixed response holdouts; five paired seeds; lower is better')
    fig.tight_layout();fig.savefig(figdir/'controls_paired_seeds.png',dpi=180);fig.savefig(figdir/'controls_paired_seeds.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for ax,target in zip(axes,['V1','V2']):
        g=check[check.target==target].groupby('kind')[['observed_abs_mean','predicted_abs_mean']].mean().reindex(KINDS)
        ax.bar(np.arange(3)-.16,g.observed_abs_mean,.32,label='Observed',color='#94a3b8')
        ax.bar(np.arange(3)+.16,g.predicted_abs_mean,.32,label='Predicted',color=COLORS)
        ax.set_xticks(range(3),['A Original','B Amount','C Volume']);ax.set_title(target);ax.set_ylabel('Mean absolute volume response (mL)');ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(figdir/'controls_response_amplitude.png',dpi=180);plt.close(fig)
    constraints=pd.read_csv(OUT/'output_constraints.csv')
    means=constraints.groupby(['kind','scope']).mean(numeric_only=True).reset_index()
    unrelated=pd.read_csv(OUT/'validation_unrelated_response_metrics.csv')
    unrelated=unrelated.query("axis=='V/ul'").groupby(['seed','kind']).difference_error_mean.mean().reset_index()
    unrelated.to_csv(OUT/'validation_unrelated_seed_summary.csv',index=False)
    series=pd.read_csv(OUT/'series_equal_response_metrics.csv').query("scope=='check' and axis=='V/ul'").groupby(['seed','kind']).difference_error_mean.mean().reset_index()
    robust=primary[['seed','kind','difference_error_mean']].merge(series,on=['seed','kind'],suffixes=('_primary','_series_equal')).merge(unrelated,on=['seed','kind'])
    robust=robust.rename(columns={'difference_error_mean':'difference_error_mean_validation_unrelated'})
    assert len(robust)==15 and not robust.duplicated(['seed','kind']).any()
    assert np.isfinite(robust.select_dtypes('number').to_numpy()).all()
    robust.to_csv(OUT/'control_robustness.csv',index=False)
    seedtable=joined[['seed','kind','difference_error_mean','combined_normalized_rmse','V1_rmse','V2_rmse','best_epoch','epochs_run']]
    endpoint=check[['seed','kind','target','n','n_series','difference_error_mean','amplitude_error_mean_ml','observed_abs_mean','predicted_abs_mean','direction_accuracy_meaningful','flat_fraction_meaningful']]
    endpoint.to_csv(OUT/'control_endpoint_summary.csv',index=False)
    endpoint_deltas=[];row_index=row.set_index(['seed','kind']);endpoint_index=endpoint.set_index(['seed','kind','target'])
    for seed in sorted(joined.seed.unique()):
        for comparison,first,second in [('B-A',KINDS[1],KINDS[0]),('C-A',KINDS[2],KINDS[0]),('C-B',KINDS[2],KINDS[1])]:
            for target in ['V1','V2']:
                first_rmse=float(row_index.loc[(seed,first),target+'_rmse']);second_rmse=float(row_index.loc[(seed,second),target+'_rmse'])
                x,y=endpoint_index.loc[(seed,first,target)],endpoint_index.loc[(seed,second,target)]
                endpoint_deltas.append(dict(seed=int(seed),comparison=comparison,target=target,row_rmse_first_ml=first_rmse,row_rmse_second_ml=second_rmse,
                    row_rmse_delta_ml=first_rmse-second_rmse,local_difference_mae_first_ml=x.difference_error_mean,local_difference_mae_second_ml=y.difference_error_mean,
                    local_difference_mae_delta_ml=x.difference_error_mean-y.difference_error_mean,
                    amplitude_error_delta_ml=x.amplitude_error_mean_ml-y.amplitude_error_mean_ml,
                    direction_accuracy_delta=x.direction_accuracy_meaningful-y.direction_accuracy_meaningful,flat_fraction_delta=x.flat_fraction_meaningful-y.flat_fraction_meaningful))
    pd.DataFrame(endpoint_deltas).to_csv(OUT/'paired_endpoint_differences.csv',index=False)
    interpretation=f'''\n本轮B/C局部MAE相对A分别变差{100*(summary.loc[KINDS[1],'difference_error_mean']/summary.loc[KINDS[0],'difference_error_mean']-1):.2f}%和{100*(summary.loc[KINDS[2],'difference_error_mean']/summary.loc[KINDS[0],'difference_error_mean']-1):.2f}%，各只有3/5 seed改善，均未通过20%/4seed机制门槛。Row平均仅改善{100*(1-summary.loc[KINDS[1],'combined_normalized_rmse']/summary.loc[KINDS[0],'combined_normalized_rmse']):.2f}%和{100*(1-summary.loc[KINDS[2],'combined_normalized_rmse']/summary.loc[KINDS[0],'combined_normalized_rmse']):.2f}%，各3/5 seed不变差，也未通过预定Row relevance门槛。两种skip都没有得到所需局部收益，停止本组简单残差路线，不追加调参或模型。

检查集真实绝对体积变化均值{summary.loc[KINDS[0],'observed_abs_mean']:.3f}mL，A/B/C预测幅度仅{summary.loc[KINDS[0],'predicted_abs_mean']:.3f}/{summary.loc[KINDS[1],'predicted_abs_mean']:.3f}/{summary.loc[KINDS[2],'predicted_abs_mean']:.3f}mL；meaningful pair近似平坦率分别{100*summary.loc[KINDS[0],'flat_fraction_meaningful']:.2f}%/{100*summary.loc[KINDS[1],'flat_fraction_meaningful']:.2f}%/{100*summary.loc[KINDS[2],'flat_fraction_meaningful']:.2f}%。三模型方向准确率相同，但大多数预测变化很小；微小同号变化不能被解释为学会了响应幅度。正确非平坦/错误非平坦比例另外保留在`local_response_metrics.csv`，不由方向准确率单独作决策。
'''
    text='''# 三模型固定标签对照结果

15次预定拟合全部完成：五个固定seed各训练A原模型、B上样量直接残差、C独立样品体积直接残差。没有新增seed、调参、LLM请求或AL重跑。所有模型从头初始化，原L1005 checkpoint只用于独立的零训练审计，不作为A的结果。

## 数据与可比性

每seed固定H1005，剔除先前metadata-only方案预定的10个完整体积系列。训练行数分别982/981/984/985/984，检查行数23/24/21/20/21；原Row validation均416。H内同系列及完全等价条件的训练重叠均0。A参数458952，B/C各458954；每个端点仅新增一个零初始化斜率，三个分位共享预激活偏移，随后执行原ReLU/eval clamp。B取原typed normalization的amount；C独立raw V经stable sample ID连接，五seed固定30–150µL缩放。

三个模型的共同初始参数及初始化后torch RNG逐seed相同，零skip初始输出逐值一致。loss、Adam、batch2048、lr0.001、weight decay0、最大1000 epoch、patience100、每epoch样本顺序、固定L333目标尺度及validation选择标准均保持原训练规则。独立模型variant/schema拒绝旧checkpoint。

检查标签没有进入训练或early stopping，但原validation存在同系列关联：五seed分别2/3/2/0/2个检查系列与validation相连，2357和12203各有一条完全同条件validation记录。因此这里不是完全独立的确认集。另报删除所有validation关联系列后的固定预测敏感性结果；不改checkpoint，不补训练。

## 逐seed结果先行

主局部指标为同系列所有体积pair差分MAE，先对V1/V2等权，再五seed等权；pair不是独立重复。Row指标用原validation和L333尺度。下表均越低越好；逐endpoint方向、幅度、平坦率与0.1/0.5/1mL阈值见CSV。`paired_endpoint_differences.csv`另外逐seed列出B−A/C−A/C−B的V1/V2 Row RMSE、局部差分MAE、幅度、方向和平坦率配对差。

'''+md(seedtable)+'''\n
## 五seed等权汇总与配对差

'''+md(summary.reset_index()[['kind','difference_error_mean','combined_normalized_rmse','V1_rmse','V2_rmse','observed_abs_mean','predicted_abs_mean','direction_accuracy_meaningful','flat_fraction_meaningful']])+'''\n
正local improvement表示改善；正Row change表示变差。B−A检验强化已有量信号；C−A检验独立V直接通路；C−B比较相同两参数通路中的数值表示。相同表示可能因缩放及优化轨迹不同而表现不同，不能从C−B推断物理因果。

'''+md(pd.DataFrame(gates)[['comparison','local_improvement_ratio_equal_seed_means','local_positive_seeds','row_relative_change_equal_seed_means','row_nonworse_seeds','row_worse_over_one_percent_seeds','mechanism_reference_pass','row_relevance_reference_pass']])+interpretation+'''\n
20%局部改善、至少4/5 seed局部同向、平均Row恶化不超过预定1%是工程参考门槛。Row relevance另要求平均改善至少1%且至少4/5 seed不变差；逐seed恶化也明确列出。这些不是统计显著性定理，没有以pair数量扩大科研重复数。

计划固定了20%/4seed/1%的数值门槛，但未完全细化相对改善采用均值之比还是逐seed相对值均值，也未细化1%是均值还是逐seed限制。主表明确采用五seed等权均值之比，另列逐seed相对值均值及超过1%恶化的seed数；两种相对改善口径和更严格的逐seedRow限制均不改变本轮未通过的判断，不能将这些具体聚合公式追称为完全预注册。

## 稳健性、输出约束与其他因素

按series等权及剔除validation关联series的描述性敏感性见`control_robustness.csv`；声明时间见`ANALYSIS_SUPPLEMENT.json`，未用于选择模型或改变主门槛。三种局部MAE五seed等权汇总如下。

'''+md(robust.groupby('kind').mean(numeric_only=True).drop(columns='seed').reset_index())+'''\n
输出约束下表是每seed先算频率再等权；负值率按全部6输出统计，分位交叉指任一相邻分位倒置，V2<V1按q50比较。ReLU保证非负并不保证分位顺序或V2≥V1，未在评价时投影或修正。

'''+md(means.drop(columns='seed'))+'''\n
原validation中其余单因素pair（含PE/EA）全部保存在`local_response_metrics.csv`和`control_response_pairs.csv`。它们是validation内的描述性响应检查，并非新增独立holdout。不存在某因素pair时不编造指标。`other_axis_equal_seed.csv`列明每因素/端点预测幅度与差分MAE，供检查意外变化。

本实验只能判断固定标签条件下简单残差路径是否有效，不能证明新增测量、更换采集策略或LLM科学推理更优。最终行动按全部五seed结果和预定门槛决定，失败后不追加网络搜索。
'''
    (OUT/'MODEL_COMPARISON.md').write_text(text)
    oth=local.query("scope=='validation' and axis!='V/ul' and observation_threshold_ml==.5")
    other=oth.groupby(['kind','axis','target'])[['n','difference_error_mean','observed_abs_mean','predicted_abs_mean','direction_accuracy_meaningful','flat_fraction_meaningful']].mean().reset_index()
    counts=oth.groupby(['kind','axis','target']).agg(n_seeds=('seed','nunique'),total_pairs=('n','sum')).reset_index()
    other=other.merge(counts,on=['kind','axis','target'],validate='one_to_one')
    other.to_csv(OUT/'other_axis_equal_seed.csv',index=False)
    with (OUT/'MODEL_COMPARISON.md').open('a') as f:
        f.write('\n'+md(other)+'\n\nPE/EA覆盖全部五seed；loading solvent仅seed157，loading solvent volume仅157和12203，Density没有可比较pair。后两因素的描述不能推广为五seed稳定性结论。\n')
    import torch
    coefficients=[]
    for audit_path in sorted((OUT/'fits').glob('seed_*/*/fit_audit.json')):
        audit=json.loads(audit_path.read_text());kind=audit['kind']
        if kind=='A_original':continue
        saved=torch.load(audit_path.parent/'best.pt',map_location='cpu',weights_only=False)
        weight=saved['state_dict']['skip_weight'].numpy()
        norm=saved['normalization'];volume=saved['volume_scale']
        unit_range=norm['loading_amount_max']-norm['loading_amount_min'] if kind=='B_amount_skip' else volume['max']-volume['min']
        for t,w in enumerate(weight):coefficients.append(dict(seed=audit['seed'],kind=kind,target=f'V{t+1}',weight_ml_per_normalized_unit=float(w),raw_input_unit='mg' if kind=='B_amount_skip' else 'uL',weight_ml_per_raw_unit=float(w/unit_range)))
    pd.DataFrame(coefficients).to_csv(OUT/'learned_skip_coefficients.csv',index=False)
    with (OUT/'MODEL_COMPARISON.md').open('a') as f:
        f.write('\n直接残差的逐seed系数见`learned_skip_coefficients.csv`，分别以每归一化单位及每mg/µL给出。它只表示该条附加通路，完整模型还会重新学习其余权重，不能把skip斜率单独解释为总预测的物理导数。\n')
    print(json.dumps(gates,indent=2))

if __name__=='__main__':main()

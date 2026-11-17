#!/usr/bin/env python3
"""Summarize only the prespecified fixed-label pilot; never fit or tune."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/nocw_pilot_mpl')
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]
def md(df):
 def f(v):
  if pd.isna(v):return 'NA'
  return f'{v:.6f}' if isinstance(v,float) else str(v)
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))
def main():
 fits=pd.read_csv(OUT/'tables/pilot_fit_results.csv');group=pd.read_csv(OUT/'tables/pilot_H_group_metrics.csv')
 import lzma
 root=OUT.parents[2]
 snapshot=json.loads(lzma.decompress((root/'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz').read_bytes()))
 qualification=json.loads((OUT/'verification/pilot_qualification.json').read_text())
 context_rows=[]
 for s in qualification['states']:
     seen={snapshot['metadata'][i]['smiles'] for i in s['train_ids']}
     for g in s['groups']:
         context_rows.append({'seed':s['seed'],'available_budget':s['available_budget'],'group_id':g['group_id'],
           'molecule_present_in_training':g['molecule'] in seen})
 contexts=pd.DataFrame(context_rows);group=group.merge(contexts,on=['seed','available_budget','group_id'],validate='many_to_one')
 context_summary=contexts.groupby(['seed','available_budget']).agg(H_molecules=('group_id','size'),H_molecules_present_in_training=('molecule_present_in_training','sum')).reset_index()
 context_summary['H_molecules_unseen_in_training']=context_summary.H_molecules-context_summary.H_molecules_present_in_training
 assert len(fits)<=16 and not fits.duplicated(['seed','available_budget','arm','replicate']).any()
 vals=['validation_combined_normalized_rmse','validation_V1_rmse','validation_V1_mae','validation_V1_r2','validation_V2_rmse','validation_V2_mae','validation_V2_r2']
 paired=[]
 for (seed,budget,rep),g in fits.groupby(['seed','available_budget','replicate']):
  assert len(g)==2
  b=g.set_index('arm').loc['baseline'];x=g.set_index('arm').loc['interaction']
  assert b.initialization_sha256==x.initialization_sha256
  paired.append({'seed':seed,'available_budget':budget,'replicate':rep,'actual_train_rows':b.actual_train_rows,
    'baseline_nrmse':b.validation_combined_normalized_rmse,'interaction_nrmse':x.validation_combined_normalized_rmse,
    **{f'{v}_delta_interaction_minus_baseline':x[v]-b[v] for v in vals}})
 paired=pd.DataFrame(paired);paired.to_csv(OUT/'tables/pilot_paired_validation.csv',index=False,float_format='%.17g')
 rep_pattern='；'.join(f'rep{int(rep)}为{int((g.validation_combined_normalized_rmse_delta_interaction_minus_baseline<0).sum())}/{len(g)}改善' for rep,g in paired.groupby('replicate'))
 # Within each axis/target, one group means one component. Ten components per
 # state; molecules counted separately. Replicate is not an independent outer seed.
 gm=group.groupby(['seed','available_budget','arm','replicate','axis','target']).agg(
    n_components=('group_id','nunique'),n_molecules=('molecule_sha256','nunique'),pair_events=('pair_count','sum'),
    group_mean_abs_difference_error_ml=('absolute_difference_error_ml','mean'),group_mean_signed_difference_error_ml=('signed_difference_error_ml','mean'),
    group_mean_direction_match=('direction_match','mean'),group_mean_observed_abs_difference_ml=('observed_abs_difference_ml','mean'),
    group_mean_predicted_abs_difference_ml=('predicted_abs_difference_ml','mean')).reset_index()
 gm['amplitude_ratio_of_group_means']=gm.group_mean_predicted_abs_difference_ml/gm.group_mean_observed_abs_difference_ml
 gm.to_csv(OUT/'tables/pilot_condition_metrics.csv',index=False,float_format='%.17g')
 contextual=group.groupby(['seed','available_budget','arm','replicate','axis','target','molecule_present_in_training']).agg(
     n_components=('group_id','nunique'),n_molecules=('molecule_sha256','nunique'),
     group_mean_abs_difference_error_ml=('absolute_difference_error_ml','mean'),group_mean_observed_abs_difference_ml=('observed_abs_difference_ml','mean'),
     group_mean_predicted_abs_difference_ml=('predicted_abs_difference_ml','mean')).reset_index()
 contextual.to_csv(OUT/'tables/pilot_condition_by_molecule_context.csv',index=False,float_format='%.17g')
 gp=[]
 for keys,g in gm.groupby(['seed','available_budget','replicate','axis','target']):
  b=g.set_index('arm').loc['baseline'];x=g.set_index('arm').loc['interaction']
  gp.append({'seed':keys[0],'available_budget':keys[1],'replicate':keys[2],'axis':keys[3],'target':keys[4],
    'n_components':int(b.n_components),'n_molecules':int(b.n_molecules),'baseline_abs_difference_error_ml':b.group_mean_abs_difference_error_ml,
    'interaction_abs_difference_error_ml':x.group_mean_abs_difference_error_ml,
    'delta_abs_difference_error_ml':x.group_mean_abs_difference_error_ml-b.group_mean_abs_difference_error_ml,
    'relative_abs_difference_error_change':x.group_mean_abs_difference_error_ml/b.group_mean_abs_difference_error_ml-1,
    'baseline_direction_match':b.group_mean_direction_match,'interaction_direction_match':x.group_mean_direction_match,
    'baseline_amplitude_ratio':b.amplitude_ratio_of_group_means,'interaction_amplitude_ratio':x.amplitude_ratio_of_group_means})
 gp=pd.DataFrame(gp);gp.to_csv(OUT/'tables/pilot_paired_condition.csv',index=False,float_format='%.17g')
 state=paired.groupby(['seed','available_budget']).agg(actual_train_rows=('actual_train_rows','first'),
   baseline_nrmse_mean=('baseline_nrmse','mean'),baseline_nrmse_sd=('baseline_nrmse','std'),
   interaction_nrmse_mean=('interaction_nrmse','mean'),interaction_nrmse_sd=('interaction_nrmse','std'),
   paired_delta_mean=('validation_combined_normalized_rmse_delta_interaction_minus_baseline','mean'),
   paired_delta_min=('validation_combined_normalized_rmse_delta_interaction_minus_baseline','min'),
   paired_delta_max=('validation_combined_normalized_rmse_delta_interaction_minus_baseline','max')).reset_index()
 state.to_csv(OUT/'tables/pilot_state_summary.csv',index=False,float_format='%.17g')
 vol=gp[gp.axis=='V/ul'].groupby(['seed','available_budget','target']).agg(
   n_components=('n_components','first'),baseline_error_ml=('baseline_abs_difference_error_ml','mean'),interaction_error_ml=('interaction_abs_difference_error_ml','mean'),
   paired_delta_ml=('delta_abs_difference_error_ml','mean'),baseline_amplitude_ratio=('baseline_amplitude_ratio','mean'),interaction_amplitude_ratio=('interaction_amplitude_ratio','mean')).reset_index()
 vol.to_csv(OUT/'tables/pilot_volume_summary.csv',index=False,float_format='%.17g')
 interpretation=json.loads((OUT/'verification/pilot_interpretation_plan.json').read_text())
 qrows=[]
 for (seed,budget,arm,rep),g in gm[gm.axis=='V/ul'].groupby(['seed','available_budget','arm','replicate']):
     scales=json.loads((root/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json').read_text())['contract']['preprocessing']['target_scales']
     vals=g.set_index('target').group_mean_abs_difference_error_ml
     qrows.append({'seed':seed,'available_budget':budget,'arm':arm,'replicate':rep,
       'Q_volume_normalized_difference_mae':.5*(vals['V1']/scales['V1']+vals['V2']/scales['V2'])})
 qframe=pd.DataFrame(qrows)
 qs=qframe.groupby(['seed','available_budget','arm']).Q_volume_normalized_difference_mae.mean().unstack('arm').reset_index()
 qs['Q_delta_interaction_minus_baseline']=qs.interaction-qs.baseline
 qs=qs.rename(columns={'baseline':'baseline_Q','interaction':'interaction_Q'})
 joined=state.merge(qs,on=['seed','available_budget'],validate='one_to_one')
 joined['row_relative_change']=joined.interaction_nrmse_mean/joined.baseline_nrmse_mean-1
 joined.to_csv(OUT/'tables/pilot_joint_interpretation.csv',index=False,float_format='%.17g')
 outer=joined.groupby('seed').agg(row_delta=('paired_delta_mean','mean'),Q_delta=('Q_delta_interaction_minus_baseline','mean')).reset_index()
 sensitivity=[]
 for (seed,budget),g in group[group.axis=='V/ul'].groupby(['seed','available_budget']):
  scales=json.loads((root/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json').read_text())['contract']['preprocessing']['target_scales']
  z=g.copy();z['normalized_error']=z.apply(lambda r:r.absolute_difference_error_ml/scales[r.target],axis=1)
  by=z.groupby(['group_id','arm']).normalized_error.mean().unstack('arm')
  delta=(by.interaction-by.baseline).to_numpy()
  lomo=[np.delete(delta,i).mean() for i in range(len(delta))]
  sensitivity.append({'seed':seed,'available_budget':budget,'volume_molecules':len(delta),'mean_Q_delta':delta.mean(),
    'negative_molecule_deltas':int((delta<0).sum()),'positive_molecule_deltas':int((delta>0).sum()),
    'leave_one_molecule_out_min_mean_delta':min(lomo),'leave_one_molecule_out_max_mean_delta':max(lomo),
    'all_leave_one_molecule_out_means_negative':all(v<0 for v in lomo)})
 sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(OUT/'tables/pilot_volume_group_effect_sensitivity.csv',index=False,float_format='%.17g')
 row_ok=bool((outer.row_delta<0).all() and (joined.paired_delta_mean<0).sum()>=3 and (joined.row_relative_change<=.01).all())
 q_ok=bool((outer.Q_delta<0).all() and (joined.Q_delta_interaction_minus_baseline<0).sum()>=3)
 verdict={'status':'STABLE_JOINT_SUPPORT' if row_ok and q_ok else 'NO_STABLE_JOINT_SUPPORT_FOR_THIS_SINGLE_CHANGE',
   'row_criterion_pass':row_ok,'volume_Q_criterion_pass':q_ok,'row_improved_states':int((joined.paired_delta_mean<0).sum()),
   'Q_improved_states':int((joined.Q_delta_interaction_minus_baseline<0).sum()),'outer_seed_effects':outer.to_dict('records'),
   'scope':'descriptive plan set after training start and reported validation progress/first paired scores; H had not been read. Not blinded, not training preregistration or efficacy confirmation',
   'does_not_rule_out_all_H3':True,'additional_architecture_search_performed':False}
 (OUT/'verification/pilot_joint_interpretation.json').write_text(json.dumps(verdict,indent=2)+'\n')
 report=f'''# 固定标签条件交互试验结果

本次单一融合对照的四个状态平均Row NRMSE均下降，但八个配对中只有五个改善、三个恶化；H上样差分Q只在两个状态方向改善，887的outer平均Q反而恶化。**整体预测改善与loading响应修复在本试验中没有同步成立**，不能据Row均值改善宣称已解决上样学习瓶颈。结果于2026-11-13汇总，目录保留任务指定的20261112。

本试验按预先冻结的唯一结构与元数据H规则执行，共完成{len(fits)}次成功拟合；没有新AL选择、付费LLM或test真值读取。四个状态来自NoCW固定seed157/887、已有L525/L1005。训练量分别492/958/490/960，H分别33/47/35/45行。不能将结果冒充原L525/L1005同训练量曲线。

## 1. 执行与唯一改动

唯一改动为 `p+r+r*tanh(nonaffine LayerNorm(p))`。新增参数0，两臂各458952参数；每rep所有初始参数/buffer及零条件残差初函数完全配对。数据、原标准化、原L333分母、loss、Adam、max1000/patience100、validation checkpoint规则相同。两个rep改变配对初始化/训练RNG，不能算两个独立outer seed。

v1检查组与validation的两处同条件重合在任何拟合前被root X-only审查发现；v2整组排除后重新按原SHA规则选择。v1记录保留，详见协议；没有按响应或结果换H。

## 2. Row validation结果

{md(state)}

表中delta为interaction−baseline，NRMSE越低越好。SD仅描述两个训练rep，既不是统计显著性，也不是五seed泛化证据。{rep_pattern}；四状态均值改善不能读成跨初始化稳定提升，当前结果明显依赖这两个初始化/训练RNG组合。原始八个配对值和六个V1/V2指标保存在 `pilot_paired_validation.csv`；不丢弃反向rep或预算。

## 3. H中上样体积差分

{md(vol)}

实际较明显的局部改善集中在157/L1005：V1差分MAE约2.173→1.912 mL，而V2约1.822→1.911 mL反向。该状态V1/V2响应幅度比增加至约0.181/0.142，但远未接近1；其他状态仍近乎平坦或方向混合。157/L1005的Q改善在十个分子上均同方向、删除任一分子后仍为负，说明该特定结果不是单一极端分子支配；它也不应被推广为全部状态或端点。

每个component内部先平均，再对component平均；本次每状态H有10个不同分子，含V的component数为8/10/10/10。pair共享实验点，因此不按pair数推显著性。幅度比是component均值的比值，既不按响应挑样，也不对接近零的单pair除法。

L525与L1005的H由各自已获L和同一规则选择，可能是不同记录/分子组。因此预算间H原始误差不是固定检查集学习曲线；可解释的模型对照仅是同seed、同预算、同H的两臂配对差。本报告不把H点连成预算学习曲线，跨状态平均的是预定配对效果。

H两端可能在不同AL轮次被采集，原始物理实验批次/操作者未完整记录。单因素匹配只控制已记录字段，不将响应差分全部归为条件的物理因果效应。按分子删除一个组的效果敏感性另保存在 `pilot_volume_group_effect_sensitivity.csv`；不据此改H、重新拟合或删除反例。

signed error定义预测条件差分−真实条件差分。连续数值轴由低到高定向；PE/EA按EA体积分数由低到高，加载溶剂类别只按稳定字符串方向，不作连续物理导数解释。完整PE/EA和其他轴、方向一致率及分子数在 `pilot_condition_metrics.csv`。

## 4. 检查组的分子插值与外推背景

{md(context_summary)}

H同时包含训练中见过及完全未见的分子，不能将全部H误差归于条件学习。`pilot_condition_by_molecule_context.csv`保留相同固定H按此背景分层的描述性结果；小组样本尤其有限，未改变任何H或增加拟合。

## 5. 解释计划与联合支持判断

`pilot_interpretation_plan.json`在训练已开始且若干validation进度/首个L525配对score已通过执行简报报告后设定；当时未读取H结果。未打开interaction audit不能等同于完全未知干预score。因此这只是透明的描述性解释约定，不是盲法、训练前预注册或独立确认。按该规则，Row与Q两条件的联合判断为 **{verdict['status']}**。Row改善状态{verdict['row_improved_states']}/4，Q改善状态{verdict['Q_improved_states']}/4。

{md(joined[['seed','available_budget','paired_delta_mean','row_relative_change','baseline_Q','interaction_Q','Q_delta_interaction_minus_baseline']])}

Q定义为 `0.5*(group/molecule均值abs差分误差V1/s1 + V2/s2)`，并先在训练rep内计算、再在状态平均。联合支持要求Row两outer平均均改善、至少3/4状态改善且无状态相对恶化>1%；Q也要求两outer平均均改善且至少3/4状态改善。未通过只说明这一个预定义改动未获得稳定联合支持，不能否定全部H3，也不据此试别的结构。

## 6. 机制与竞争解释

模型效果必须分别看局部条件差分与整体row NRMSE。局部响应增强不自动说明标签更有学习价值；整体误差改善也不自动证明原模型唯一缺陷是交互表达。本改动作用于全部completion residual，包含eluent HBA/LogP及loading特征，不是只改纯loading通道；PE/EA轴控制结果必须一起看。本改动同时改变r路径的梯度conditioning，不能分离表达和优化机制；原始记录未提供完整复测身份，因此剩余差分误差不能直接当不可约实验噪声。

八个配对只有两个outer seed和两个事后已暴露开发预算，结果属于小型机制筛查。H来自已采集NoCW分布、优先上样V对，不能推广全候选/全化学空间。H不参与early stopping，也未用test真值。不得在看到本结果后继续试新结构、改loss或加seed找正例。

## 7. 成本与停止

成功fit={len(fits)}，audit记录fit秒数总和{fits.training_seconds.sum():.1f}，运行epoch总和{int(fits.epochs_run.sum())}。每fit计时含epoch训练/validation/checkpoint与训练后评价，排除先前加载和哈希；两个worker重叠，所以总和不是端到端墙钟或CPU秒数。这是本机实际执行元数据，不估算货币费用。图/报告读取已完成结果，没有调用训练。协议与成功fit上限见 `verification/pilot_protocol_freeze.json`、`pilot_execution_status.json`，权重和逐行H评估留在gitignored runtime。

本轮训练结束。后续是否扩充需依据总体研究裁决另行授权，不自动扩展到五outerseed或多个结构。
'''
 (OUT/'MODEL_PILOT_RESULTS.md').write_text(report)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for (seed,b),g in paired.groupby(['seed','available_budget']):
  axs[0].scatter([f'{seed}/L{b}']*len(g),g.validation_combined_normalized_rmse_delta_interaction_minus_baseline,label=f'{seed}/L{b}')
 axs[0].axhline(0,c='k',lw=.8);axs[0].set_ylabel('Interaction - baseline validation NRMSE');axs[0].set_title('8 paired comparisons, 2 outer seeds')
 v=gp[gp.axis=='V/ul']
 for target,g in v.groupby('target'):
  x=np.arange(len(g))+(0 if target=='V1' else .15);axs[1].scatter(x,g.delta_abs_difference_error_ml,label=target)
 axs[1].axhline(0,c='k',lw=.8);axs[1].set_ylabel('Difference MAE change (mL)');axs[1].set_xlabel('Fixed seed/state/training replicate order');axs[1].set_title('H sample-volume axis, group-weighted');axs[1].legend()
 fig.text(.02,.015,'Source: pilot_paired_validation.csv / pilot_paired_condition.csv. H metadata fixed before fitting; 16 fits are not 16 independent studies.\nOriginal labels already acquired, no new acquisition or test evaluation. Negative deltas favor the single prespecified interaction model.',fontsize=8)
 fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(OUT/'figures/pilot_local_and_row_tradeoff.png',dpi=170);plt.close(fig)
 print('summarized',len(fits),'fits;',len(paired),'paired comparisons')
if __name__=='__main__':main()

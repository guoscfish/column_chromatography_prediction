from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'studies/active_learning'; OUT=Path(__file__).resolve().parent
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
REV=BASE/'qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001'
SOURCES={}
def source(p):
 p=Path(p);SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(source(p).read_text())
def csv(p):return pd.read_csv(source(p))
def table(df):
 def fmt(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def area(x,y):return float(np.trapezoid(y,x)/(max(x)-min(x)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white','savefig.facecolor':'white'})
COLORS=['#167c80','#e18531','#4e64ad','#9199a0','#9956a8','#af5656','#5a934c','#846c4c','#427fc1']
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
labels={'center_width_lcmd':'CW-LCMD','random':'Random','random32':'Random','lcmd':'Gradient LCMD','hybrid':'Traditional Hybrid','gradient_maxdet':'Gradient MaxDet','kernel_ivr':'Kernel IVR','cw16_llm16_scientist_v2':'CW16 + LLM16','free_llm32_scientist_v2':'LLM free32','gradient_latent_fusion_maxdet':'Fusion MaxDet','gradient_latent_fusion_a080_maxdet':'Fusion MaxDet a=0.80','cw_ivr_portfolio':'CW / IVR portfolio'}
# Latest validation results: never open a test label store.
v=js(REV/'validation_four_strategy_comparison.json')['curves']
v['cw16_llm16_scientist_v2']=js(REV/'extension_hybrid157_to589/result.json')['validation_learning_curve']
for r in (7,8):
 d=js(BASE/f'qgeognn_v2_row_cw_lcmd_to_653/runtime/seed_157/center_width_lcmd/round_{r:02d}/model/fit_audit.json')
 v['center_width_lcmd'].append({'labels':d['train_rows'],'validation_nrmse':d['best_validation_combined_normalized_rmse']})
vr=pd.DataFrame([{'method':m,'labels':r['labels'],'NRMSE':r['validation_nrmse'],'evaluation':'validation','seed':157} for m,rs in v.items() for r in rs]);vr.to_json(OUT/'latest_validation_curves.json',orient='records',indent=2)
fig,ax=plt.subplots(figsize=(10,5.3))
for m,c in zip(['cw16_llm16_scientist_v2','center_width_lcmd','free_llm32_scientist_v2','random32'],COLORS):
 g=vr[vr.method.eq(m)];ax.plot(g.labels,g.NRMSE,'o-',label=labels[m],color=c,lw=2,ms=4)
ax.axvspan(525,589,color='#167c80',alpha=.07);ax.axvline(525,color='gray',ls=':',lw=1);ax.set(title='Latest LLM vs controls | seed 157 | VALIDATION',xlabel='Active training labels (shared validation: +416)',ylabel='Validation NRMSE (lower is better)',xticks=sorted(vr.labels.unique()));ax.legend(ncol=2);fig.text(.13,.005,'Shaded region: exploratory hybrid extension. Free32 and Random stop at 525 in this comparison.',fontsize=9);save(fig,'01_latest_validation')
vs=[]
for m,g in vr.groupby('method',sort=False):
 g=g[g.labels.le(525)].sort_values('labels');vs.append({'method':labels[m],'NRMSE@525':g.iloc[-1].NRMSE,'AULC333–525':area(g.labels,g.NRMSE)})
vs=pd.DataFrame(vs).sort_values('NRMSE@525')
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,col,title in zip(axes,['NRMSE@525','AULC333–525'],['Endpoint at 525 labels','Whole curve: 333–525 labels']):
 ax.barh(vs.method,vs[col],color=[COLORS[['CW16 + LLM16','CW-LCMD','LLM free32','Random'].index(n)] for n in vs.method]);ax.invert_yaxis();ax.set(title=title,xlabel='Validation score (lower is better)',xlim=(0,1.0));
 for i,val in enumerate(vs[col]):ax.text(val+.01,i,f'{val:.4f}',va='center')
fig.suptitle('Matched budget comparison | seed 157 | VALIDATION');fig.tight_layout();save(fig,'02_latest_endpoint_and_aulc')
# Historical traditional strategies, explicit unique authoritative method sources.
specs=[('qgeognn_v2_row_sequential_b32',['random','lcmd','hybrid']),('qgeognn_v2_row_maxdet_b32',['gradient_maxdet','u50_gradient_maxdet']),('qgeognn_v2_row_kernel_ivr_b32',['kernel_ivr']),('qgeognn_v2_row_cw_lcmd_to_1005',['center_width_lcmd']),('qgeognn_v2_row_fused_maxdet_b32',['gradient_latent_fusion_maxdet']),('qgeognn_v2_row_fused_maxdet_a080_b32_screen',['gradient_latent_fusion_a080_maxdet']),('qgeognn_v2_row_cw_ivr_portfolio_b32',['cw_ivr_portfolio'])]
parts=[]
for study,methods in specs:
 d=csv(BASE/study/'results/learning_curve_metrics.csv');d=d[d.method.isin(methods)&d.outer_seed.isin([157,6101])].copy();d['source_study']=study;parts.append(d)
td=pd.concat(parts,ignore_index=True);assert not td.duplicated(['outer_seed','method','active_label_count']).any();td.to_json(OUT/'traditional_test_curves.json',orient='records',indent=2)
main=['center_width_lcmd','hybrid','kernel_ivr','gradient_maxdet','lcmd','random']
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,seed in zip(axes,[157,6101]):
 for m,c in zip(main,COLORS):
  g=td[td.method.eq(m)&td.outer_seed.eq(seed)].sort_values('active_label_count');ax.plot(g.active_label_count,g.combined_normalized_RMSE,label=labels[m],color=c,lw=1.8)
 ax.set(title=f'Seed {seed}',xlabel='Active training labels');ax.set_xticks([333,525,653,813,1005])
axes[0].set_ylabel('Test NRMSE (lower is better)');axes[1].legend(fontsize=9);fig.suptitle('Historical traditional strategies | TEST | two shared seeds');fig.tight_layout();save(fig,'03_traditional_test_curves')
tr=[]
for (seed,m),g in td.groupby(['outer_seed','method']):
 g=g.sort_values('active_label_count');tr.append({'seed':seed,'method':m,'max_labels':int(g.active_label_count.max()),'endpoint':g.iloc[-1].combined_normalized_RMSE,'AULC':area(g.active_label_count,g.combined_normalized_RMSE)})
tr=pd.DataFrame(tr);tm=tr.groupby(['method','max_labels'])[['endpoint','AULC']].mean().reset_index();tm['display']=tm.method.map(labels).fillna(tm.method);tm.to_json(OUT/'traditional_test_summary.json',orient='records',indent=2)
mainmean=tm[tm.method.isin(main)].sort_values('AULC')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,col,title in zip(axes,['AULC','endpoint'],['AULC 333–1005','NRMSE at 1005']):
 ax.barh(mainmean.display,mainmean[col],color='#4e64ad');ax.invert_yaxis();ax.set(title=title,xlim=(0,.95),xlabel='Mean test score (lower is better)')
 for i,val in enumerate(mainmean[col]):ax.text(val+.01,i,f'{val:.4f}',va='center')
fig.suptitle('Traditional strategies | TEST mean of seeds 157 and 6101');fig.tight_layout();save(fig,'04_traditional_test_ranking')
# Older LLM protocols evaluated on test; separate panels and windows.
hist=[];fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,study,title in zip(axes,['qgeognn_v2_row_llm16_screen','qgeognn_v2_row_llm_dialog_feedback_v1'],['128-candidate LLM screen | 333–429','Dialog full-pool feedback | 333–525']):
 d=csv(BASE/study/'results/learning_curves.csv');g=d.groupby(['method','budget']).combined_normalized_RMSE.mean().reset_index()
 for i,(m,rows) in enumerate(g.groupby('method')):
  name='CW-LCMD' if m=='center_width_lcmd' else ('CW16 + random16' if 'random' in m else 'CW16 + LLM16');ax.plot(rows.budget,rows.combined_normalized_RMSE,'o-',label=name,color=COLORS[i],ms=3)
  hist.append({'study':study,'method':name,'seeds':'157,6101','end_budget':int(rows.budget.max()),'mean_test_endpoint':float(rows.iloc[-1].combined_normalized_RMSE),'mean_test_AULC':area(rows.budget,rows.combined_normalized_RMSE)})
 ax.set(title=title,xlabel='Active training labels',ylabel='Mean test NRMSE');ax.legend(fontsize=9)
fig.suptitle('Earlier LLM experiments | TEST | separate protocols, two-seed means');fig.tight_layout();save(fig,'05_historical_llm_test');hist=pd.DataFrame(hist)
# Exploratory traditional comparisons at a common 653 budget.
fig,ax=plt.subplots(figsize=(10,5))
explore=['center_width_lcmd','gradient_maxdet','gradient_latent_fusion_maxdet','gradient_latent_fusion_a080_maxdet','cw_ivr_portfolio','u50_gradient_maxdet']
for m,c in zip(explore,COLORS):
 g=td[td.method.eq(m)&td.active_label_count.le(653)].groupby('active_label_count').combined_normalized_RMSE.mean()
 if len(g):ax.plot(g.index,g.values,label=labels.get(m,m),color=c,lw=1.8)
ax.set(title='Exploratory traditional variants | TEST | mean of seeds 157 and 6101',xlabel='Active training labels',ylabel='Test NRMSE');ax.legend(fontsize=9);save(fig,'06_traditional_exploratory')
# Actual inventory, distinguish replay and non-predictive work.
inv=[]
for d in sorted(BASE.glob('qgeognn*')):
 fits=list(d.glob('runtime/**/fit_audit.json'));summaries=list(d.glob('results/*.csv'))+list(d.glob('formal_results/*.csv'))
 groups={}
 for f in fits:
  a=json.loads(f.read_text());seed=next((x for x in f.parts if x.startswith('seed_')), '?');key=seed;groups[key]=max(groups.get(key,0),a.get('train_rows',0))
 inv.append({'study':d.name,'local_fit_audits':len(fits),'local_max_train_rows_by_seed':groups,'existing_result_tables':len(summaries),'reports':[str(x.relative_to(ROOT)) for x in d.glob('*REPORT.md')]})
llminv=[]
for d in [BASE/'qgeognn_v2_row_llm_full_pool_feedback_v1',*sorted((BASE/'qgeognn_v2_row_llm_scientist_v2/revisions').iterdir())]:
 if not d.is_dir():continue
 fits=list(d.glob('runtime/**/fit_audit.json'));batches=list(d.glob('selections/**/batch_freeze.json'))
 groups={}
 for f in fits:
  a=json.loads(f.read_text());rel=f.relative_to(d).parts;key='/'.join(rel[1:3]);groups[key]=max(groups.get(key,0),a.get('train_rows',0))
 llminv.append({'revision':d.name,'frozen_batches':len(batches),'fits':len(fits),'max_labels_by_trajectory':groups,'complete_six_round_trajectories':len(list(d.glob('runtime/**/trajectory_freeze.json')))})
(OUT/'inventory.json').write_text(json.dumps({'traditional_studies':inv,'llm_revisions':llminv},ensure_ascii=False,indent=2))
# Render a detailed Chinese research note from measured data.
notes={
'qgeognn_v2_row_lcmd':'大批量单步 LCMD pilot；不能与 B32 连续曲线直接混排。',
'qgeognn_v2_row_small_batch_benchmark':'B32/B16 多策略单步 benchmark；formal_results 已有结果，README 的未执行说明过时。',
'qgeognn_v2_row_hybrid_extension':'传统 Hybrid 单步匹配扩展，非 LLM。',
'qgeognn_v2_row_sequential_b32':'Random / Gradient LCMD / 传统 Hybrid；五 seed，333→1005。',
'qgeognn_v2_row_kernel_ivr_b32':'Kernel-IVR 五 seed 连续实验。',
'qgeognn_v2_row_short_sequential_b32':'CW-LCMD / Direction-LCMD 两 seed，333→429。',
'qgeognn_v2_row_cw_lcmd_to_525':'CW 两 seed 429→525 延续。',
'qgeognn_v2_row_cw_lcmd_to_653':'CW 两 seed 525→653 延续。',
'qgeognn_v2_row_cw_lcmd_to_1005':'CW 两 seed 653→1005 延续。上述四项是一条轨迹的分段，不能当独立重复。',
'qgeognn_v2_row_maxdet_b32':'Gradient MaxDet / U50 MaxDet，两 seed，至1005。',
'qgeognn_v2_row_fused_maxdet_b32':'Gradient+latent Fusion MaxDet，两 seed，至1005。',
'qgeognn_v2_row_fused_maxdet_a080_b32_screen':'Fusion a=0.80，两 seed，至653。',
'qgeognn_v2_row_cw_ivr_portfolio_b32':'CW/IVR portfolio，两 seed，至653。',
'qgeognn_v2_batch_adaptivity':'Static / adaptive / nested random 的653标签匹配预算控制。',
'qgeognn_v2_same_state_branching':'相同状态分支短程实验；不能当从333开始的新独立轨迹。',
'qgeognn_v2_same_state_branching_cw_hybrid':'CW/Hybrid源状态分支；结果文件已存在，README的尚未训练说明不宜单独作状态依据。',
'qgeognn_v2_row_lcmd_to_ivr_b32':'development 有局部结果，尚无 results 正式汇总；不纳入完整轨迹排名。',
'qgeognn_v2_row_innovation_screen':'仅选样几何诊断，无训练效果结论。',
'qgeognn_v2_ivr_mechanism_audit':'机制审计，不是新增预测效果实验。',
'qgeognn_v2_efficiency_review':'历史结果再汇总，不是独立新增实验。',
'qgeognn_v2_row_master_report':'历史结果再汇总，不是独立新增实验。'}
report='''# LLM 与传统主动学习实验整理\n\n截至 2026-10-05；范围为当前仓库 QGeoGNN-V2、4g Row 主动学习实验。本次只读取已保存的汇总指标、训练审计及冻结记录，不训练、不重新评估测试集。旧 E2/A1a predictor 与跨柱 transfer 不并入当前V2排名。\n\n## 先看结论\n\n- 最新全量输入 LLM：seed157 的 free32 完成六轮至525；CW16+LLM16完成八轮至589。seed6101尚未完成对应当前版本轨迹。\n- 最新混合策略在525/557/589验证点优于同预算CW；589时0.4124 vs 0.4667，低11.6%。333→589全程验证AULC仍高3.7%，前期落后尚未完全抵消。\n- 最新free32在525验证误差优于Random，但不及CW；不能将化学理由合理等同于整体预测收益。\n- 历史两个LLM实验已有两seed测试评估：均优于CW+随机补充，均未通过“优于纯CW”的原定扩展门槛。\n- 传统完整轨迹在共同seed157/6101口径下，CW-LCMD的333→1005测试AULC及1005终点最好。五seed结果与两seed开发子集须分开报告。\n\n## 比较口径\n\n1. 最新LLM图使用用于checkpoint选择的固定验证集；历史传统图及旧LLM图使用已经发表在仓库的测试指标。两者不作数值直比。\n2. 所有图的NRMSE/AULC越低越好。AULC为标签轴上的梯形面积除以区间宽度；不同区间的AULC不可直接排名。\n3. 标签轴为主动训练标签，另有固定416条验证标签成本。传统测试主图只取共同seed157/6101，不把五seed均值与单seed值混用。\n4. Traditional Hybrid是传统不确定性/多样性策略，不是CW16+LLM16。CW各延续study和历史重用对照去重。\n5. 最新free第六轮有第四次模型回复及人工授权单字符ID修正；混合八轮没有人工ID替换。第7–8轮是在看过六轮验证表现后追加的探索性实验。\n\n## 最新LLM：验证集学习曲线\n\n![最新验证曲线](figures/01_latest_validation.png)\n\n'''
report+=table(vr.pivot(index='labels',columns='method',values='NRMSE').rename(columns=labels).reset_index().fillna('—'))+'\n\n### 同预算525比较\n\n'+table(vs)+'\n\n![终点与全程](figures/02_latest_endpoint_and_aulc.png)\n\n'
report+='## 较早LLM实验：已完成的测试评估\n\n'+table(hist.rename(columns={'study':'实验目录','method':'策略','seeds':'seed','end_budget':'终点标签','mean_test_endpoint':'平均测试NRMSE','mean_test_AULC':'平均测试AULC'}))+'\n\n![历史LLM](figures/05_historical_llm_test.png)\n\n128候选筛查的333→429平均AULC：LLM 0.8114，CW 0.8050；对话式全池反馈的333→525平均AULC：LLM 0.7405，CW 0.7275。它们不是最新gpt-6-astra全表直送版本，不宜用版本间差值证明模型升级有效。\n\n### 旧版本与未完成试跑清单\n\n'
report+=table(pd.DataFrame([{'revision':x['revision'],'冻结批次':x['frozen_batches'],'训练产物':x['fits'],'轨迹最高标签':json.dumps(x['max_labels_by_trajectory'],ensure_ascii=False)} for x in llminv if x['frozen_batches'] or x['fits']]))+'\n\n没有训练或冻结批次的revision也保存在inventory.json，不能记作效果结果。query_error_feedback及其replay应视为修订/重放记录，不作为独立seed或独立成功实验相加。full_pool_feedback_v1目录主要提供实现及传统补充对照，不能单凭目录名称认定其LLM轨迹已完成。\n\n'
report+='## 传统策略：共同两seed测试结果\n\n![传统测试曲线](figures/03_traditional_test_curves.png)\n\n'+table(mainmean[['display','max_labels','endpoint','AULC']].rename(columns={'display':'策略','max_labels':'终点标签','endpoint':'平均测试NRMSE','AULC':'平均测试AULC333–1005'}))+'\n\n![传统测试排名](figures/04_traditional_test_ranking.png)\n\n### 扩展方法\n\n'+table(tm[~tm.method.isin(main)][['display','max_labels','endpoint','AULC']].rename(columns={'display':'策略','max_labels':'终点标签','endpoint':'平均测试NRMSE','AULC':'平均测试AULC333至各自终点'}))+'\n\n![传统探索对照](figures/06_traditional_exploratory.png)\n\n扩展表各终点不同，图统一截到653用于观察。完整Fusion虽有1005结果，a=0.80与portfolio只到653，不能将其较短区间AULC和1005全程AULC直接比较。\n\n'
report+='## 所有当前V2 study 的产物盘点\n\n以下按本机实际文件统计；本地fit_audit数量可能含缓存、重用或分支，不代表独立重复数。README与实际产物不一致时，以已保存的结果和审计记录为准。\n\n'
report+=table(pd.DataFrame([{'study':x['study'],'本地fit审计':x['local_fit_audits'],'结果表数量':x['existing_result_tables'],'说明':notes.get(x['study'],'LLM实验，见上文；scientist revision产物单独盘点。')} for x in inv]))
report+='\n\n## 建议与证据边界\n\n当前最有价值的是补齐seed6101的相同LLM策略，优先验证混合策略后期优势；保留333→525原窗口作为固定比较，589扩展单列。不要只挑589终点宣称全程效率更高，也不要在新LLM未完成原测试屏障前重新打开测试标签。已有验证曲线无法证明LLM化学知识是收益的原因，需额外消融才可归因。\n\n## 文件与复现\n\n- 本报告：REPORT.md；图：figures/*.png及可编辑矢量SVG。\n- latest_validation_curves.json：最新LLM及匹配对照验证数据。\n- traditional_test_curves.json / traditional_test_summary.json：传统共同两seed的原始曲线与汇总。\n- inventory.json：study和LLM revision实际产物盘点。\n- source_manifest.json：直接用于本次图表的源文件SHA256，便于追溯。\n- build_report.py：使用fish环境Python运行，可重新生成汇总。\n'
(OUT/'REPORT.md').write_text(report)
(OUT/'source_manifest.json').write_text(json.dumps(SOURCES,indent=2))
print('Generated',OUT,'figures',len(list(FIG.glob('*.png'))));print(mainmean[['display','endpoint','AULC']].to_string(index=False));print('extras',tm[~tm.method.isin(main)][['display','max_labels']].to_string(index=False))
# Additional completed traditional evidence, retaining its own cohort and budget.
appendix='\n\n## 补充：传统策略其他已完成设计的数值\n\n以下各表内部可比较，不与上面的单seed验证曲线或不同预算排名混排。\n\n'
five=[]
for study,methods in [('qgeognn_v2_row_sequential_b32',['random','lcmd','hybrid']),('qgeognn_v2_row_kernel_ivr_b32',['kernel_ivr'])]:
 d=csv(BASE/study/'results/learning_curve_metrics.csv');d=d[d.method.isin(methods)]
 for (seed,m),g in d.groupby(['outer_seed','method']):
  g=g.sort_values('active_label_count');five.append({'seed':seed,'method':labels[m],'endpoint':g.iloc[-1].combined_normalized_RMSE,'AULC':area(g.active_label_count,g.combined_normalized_RMSE)})
five=pd.DataFrame(five);five_summary=five.groupby('method').agg(seed_count=('seed','nunique'),test_NRMSE_1005=('endpoint','mean'),test_AULC_333_1005=('AULC','mean')).reset_index()
appendix+='### 五seed连续轨迹（333→1005）\n\n'+table(five_summary)+'\n\n'
b32=csv(BASE/'qgeognn_v2_row_small_batch_benchmark/formal_results/seed_summary_b32.csv');b32['method']=b32.arm.where(~b32.arm.str.startswith('random_control_'),'random_mean')
per=b32.groupby(['cohort','outer_seed','method']).combined_normalized_RMSE.mean().reset_index();bs=per.groupby(['cohort','method']).agg(seeds=('outer_seed','nunique'),test_NRMSE_365=('combined_normalized_RMSE','mean')).reset_index()
appendix+='### B32单步benchmark（333→365）\n\nRandom先在每seed内部平均，再在seed间平均，避免多次随机重复改变权重。\n\n'+table(bs)+'\n\n'
hyb=csv(BASE/'qgeognn_v2_row_hybrid_extension/results/three_method_summary.csv')
appendix+='### 传统Hybrid匹配单步扩展（333→365）\n\n'+table(hyb[['method','n_outer_seeds','mean_combined_normalized_RMSE']])+'\n\n'
ad=csv(BASE/'qgeognn_v2_batch_adaptivity/results/metric_summary.csv');ad=ad[ad.metric.str.contains('normalized',case=False)]
appendix+='### 静态与自适应预算控制（终点653）\n\n'+table(ad[['method','metric','endpoint_mean','aulc_mean']])+'\n\n'
pilot=csv(BASE/'qgeognn_v2_row_lcmd/results/seed_summary.csv')
appendix+='### 早期大批量LCMD pilot\n\n'+table(pilot[['outer_seed','baseline_L0_NRMSE','LCMD_after_NRMSE','Random_after_NRMSE_median']])+'\n\n此处为大批量单步设计，不代表B32连续过程。\n'
with (OUT/'REPORT.md').open('a') as f:f.write(appendix)
(OUT/'traditional_additional_summaries.json').write_text(json.dumps({'five_seed_sequential':five_summary.to_dict('records'),'B32_one_step':bs.to_dict('records'),'traditional_hybrid_one_step':hyb.to_dict('records'),'adaptivity':ad.to_dict('records'),'large_batch_pilot':pilot.to_dict('records')},indent=2))
(OUT/'source_manifest.json').write_text(json.dumps(SOURCES,indent=2))

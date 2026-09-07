#!/usr/bin/env python3
"""Post-freeze descriptive outcomes and scientific reports, never model selection."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/physics-column-mpl')
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.studies import run_physics_column_transfer as run
from scripts.studies.evaluate_physics_column_transfer import validate_freeze
from scripts.studies.audit_column_physics import table
from src.qgeognn_al.transfer.column_physics import packing_mass
OUT=run.STUDY
LABELS={'packing_mass_physical_scale':'Mass scale','physics_scale_residual':'Physics + Ridge','raw_column_conditioned':'Context raw','mass_normalized_column_conditioned':'Context mL/g','scale_only':'Scale','local_identity_shrinkage':'Shrinkage','conditional_EA':'Conditional EA','conditional_policy':'Conditional policy','target_head_only':'Head-only','standard_shallow_finetune':'N1 shallow'}
ORDER=list(LABELS)
COLORS=dict(zip(ORDER,plt.get_cmap('tab10').colors))


def save(fig,name):
    fig.savefig(OUT/'plots'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'plots'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def full_outcomes():
    rows=[];data={};raw=[]
    for c in ['4g',*run.COLUMNS]:
        path=run.old.SOURCE_DATA if c=='4g' else run.target_path(c)
        d=pd.read_csv(path)
        d['column']=c;d['mass']=d.column_specs.map(packing_mass);d['center']=(d.V1_ml+d.V2_ml)/2;d['width']=d.V2_ml-d.V1_ml
        d['EA_bin']=d['PE/EA'].map(lambda s:float(s.split('/')[1])/sum(map(float,s.split('/')))).map(lambda x:'low' if x<=.1 else 'mid' if x<=.5 else 'high')
        data[c]=d
        for target in ['V1_ml','V2_ml','center','width']:
            for norm in [False,True]:
                v=d[target]/d.mass if norm else d[target]
                for dimension in ['overall','EA_bin','Flow mL/min']:
                    groups=[('all',d.index)] if dimension=='overall' else [(str(k),g.index) for k,g in d.groupby(dimension)]
                    for level,idx in groups:
                        z=v.loc[idx];rows.append(dict(column=c,target=target,normalization='mass_normalized' if norm else 'raw',dimension=dimension,level=level,n=len(z),mean=z.mean(),median=z.median(),std=z.std(),q10=z.quantile(.1),q90=z.quantile(.9),cv=z.std()/abs(z.mean())))
        # All raw rows with usable times; retain missing/invalid counts rather than invent labels.
        r=pd.read_csv(ROOT/'dataset'/f'dataset_{c}.csv');flow=pd.to_numeric(r['Flow mL/min'],errors='coerce')
        times=[pd.to_numeric(r['verified '+t],errors='coerce').fillna(pd.to_numeric(r[t],errors='coerce')) for t in ['t1','t2']]
        valid=times[0].notna()&times[1].notna()&(times[0]>=0)&(times[1]>=times[0])&flow.gt(0)
        for j,t in enumerate(['V1','V2']):
            z=times[j][valid]*flow[valid]/1200
            for norm in [False,True]:
                v=z/float(c[:-1]) if norm else z
                raw.append(dict(column=c,target=t,normalization='mass_normalized' if norm else 'raw',raw_rows=len(r),usable_time_rows=int(valid.sum()),excluded_rows=int((~valid).sum()),mean=v.mean(),median=v.median(),std=v.std()))
    pd.DataFrame(rows).to_csv(OUT/'postfreeze_target_distributions.csv',index=False)
    pd.DataFrame(raw).to_csv(OUT/'postfreeze_raw_time_distributions.csv',index=False)
    matched=[]
    for c in run.COLUMNS:
        for target in ['V1_ml','V2_ml','center','width']:
            for norm in [False,True]:
                a,b=data['4g'].copy(),data[c].copy()
                if norm:a[target]/=a.mass;b[target]/=b.mass
                for keys in [['canonical_smiles'],['canonical_smiles','EA_bin'],['canonical_smiles','Flow mL/min']]:
                    aa=a.groupby(keys)[target].mean();bb=b.groupby(keys)[target].mean();j=pd.concat([aa.rename('source'),bb.rename('target')],axis=1).dropna()
                    matched.append(dict(column=c,target=target,normalization='mass_normalized' if norm else 'raw',matching='+'.join(keys),groups=len(j),mean_absolute_difference=(j.target-j.source).abs().mean(),median_relative_discrepancy=((j.target-j.source).abs()/j.source.abs().clip(lower=.5/(4 if norm else 1))).median()))
    pd.DataFrame(matched).to_csv(OUT/'postfreeze_matched_discrepancy.csv',index=False)
    return data


def plots(aulc,b100,strata,data):
    (OUT/'plots').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    scales=pd.read_csv(OUT/'scale_physics_comparison.csv');scales=scales.loc[scales.budget.eq(100)]
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for ax,c in zip(axs,run.COLUMNS):
        for offset,t in zip([-.12,.12],['V1','V2']):
            for i,p in enumerate(['row','compound']):
                z=scales.loc[scales.column.eq(c)&scales.protocol.eq(p)&scales.target.eq(t),'scale']
                ax.scatter(np.arange(len(z))*0+i+offset,z,s=18,label=t if i==0 else None)
                ax.plot([i+offset-.06,i+offset+.06],[z.mean()]*2,color='black')
        ax.axhline(float(c[:-1])/4,ls='--',color='black',label='Packing-mass ratio');ax.set(title=c,xticks=[0,1],xticklabels=['row','compound'],ylabel='Frozen scale coefficient')
    axs[0].legend(fontsize=7);fig.suptitle('Budget 100: empirical coefficients versus nominal mass ratio');fig.tight_layout();save(fig,'learned_vs_physical_scale')
    fig,axs=plt.subplots(2,2,figsize=(9,6))
    for j,t in enumerate(['V1_ml','V2_ml']):
        for i,norm in enumerate([False,True]):
            ax=axs[i,j];values=[d[t]/d.mass if norm else d[t] for d in data.values()]
            ax.boxplot(values,tick_labels=list(data),showfliers=False);ax.set(title=t.replace('_ml',''),ylabel='mL/g' if norm else 'mL')
    fig.suptitle('Canonical outcome distributions; post-freeze descriptive only');fig.tight_layout();save(fig,'raw_vs_mass_normalized')
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for ax,t in zip(axs,['center','width']):
        ax.boxplot([d[t] for d in data.values()],tick_labels=list(data),showfliers=False);ax.set(title=t+' (window proxy)',ylabel='mL')
    fig.tight_layout();save(fig,'center_width_distributions')
    fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for i,p in enumerate(['row','compound']):
        for ax,c in zip(axs[i],run.COLUMNS):
            sub=aulc.loc[aulc.column.eq(c)&aulc.protocol.eq(p)]
            g=sub.groupby('method').normalized_aulc.agg(['mean','std']).reindex(ORDER)
            ax.bar(range(len(g)),g['mean'],yerr=g['std'],color=[COLORS[m] for m in ORDER],capsize=2)
            ax.set(title=f'{c} / {p}',ylabel='AULC (lower is better)',xticks=range(len(g)),xticklabels=[LABELS[m] for m in ORDER]);ax.tick_params(axis='x',rotation=70)
    fig.tight_layout();save(fig,'aulc_by_model')
    fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for i,p in enumerate(['row','compound']):
        for ax,c in zip(axs[i],run.COLUMNS):
            g=b100.loc[b100.column.eq(c)&b100.protocol.eq(p)].groupby('method')[['V1_rmse','V2_rmse']].mean().reindex(ORDER)
            x=np.arange(len(g));ax.bar(x-.18,g.V1_rmse,.36,label='V1');ax.bar(x+.18,g.V2_rmse,.36,label='V2')
            ax.set(title=f'{c} / {p}',ylabel='Budget100 RMSE (mL)',xticks=x,xticklabels=[LABELS[m] for m in ORDER]);ax.tick_params(axis='x',rotation=70)
    axs[0,0].legend();fig.tight_layout();save(fig,'budget100_rmse')
    selected=['scale_only','local_identity_shrinkage','conditional_EA',*run.METHODS]
    for dimension,levels,filename in [('source_q50',['low','mid','high','top10'],'q50_stratified_error'),('EA',['low','mid','high'],'ea_stratified_v1_error')]:
        fig,axs=plt.subplots(2,3,figsize=(13,7))
        for i,p in enumerate(['row','compound']):
            for ax,c in zip(axs[i],run.COLUMNS):
                g=strata.loc[strata.budget.eq(100)&strata.column.eq(c)&strata.protocol.eq(p)&strata.target.eq('V1')&strata.dimension.eq(dimension)]
                for m in selected:
                    v=g.loc[g.method.eq(m)].groupby('level').rmse.mean().reindex(levels)
                    ax.plot(levels,v,marker='o',ms=3,color=COLORS[m],label=LABELS[m])
                ax.set(title=f'{c} / {p}',ylabel='V1 RMSE (mL)')
        axs[0,0].legend(fontsize=6);fig.suptitle('Budget100 '+dimension+' strata; five-seed means');fig.tight_layout();save(fig,filename)


def main():
    validate_freeze()
    if not (OUT/'execution_audit.json').exists():raise RuntimeError('evaluate first')
    a=pd.read_csv(OUT/'aulc_by_seed.csv');b=pd.read_csv(OUT/'budget100_metrics.csv');s=pd.read_csv(OUT/'error_stratification.csv');cw=pd.read_csv(OUT/'center_width_metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');decision=json.loads((OUT/'decision.json').read_text())
    sanity=[]
    for f in (OUT/'contexts').glob('**/predictions_blind.csv.gz'):
        usage=json.loads((f.parent/'label_usage.json').read_text());pred=pd.read_csv(f)
        for m in [*run.BASELINES,*run.METHODS]:
            sanity.append({**{k:usage[k] for k in ['column','protocol','seed','budget']},'method':m,'rows':len(pred),'negative_V1':int((pred[m+'_V1']<0).sum()),'negative_V2':int((pred[m+'_V2']<0).sum()),'reversed_window':int((pred[m+'_V2']<pred[m+'_V1']).sum())})
    pd.DataFrame(sanity).to_csv(OUT/'prediction_physical_sanity.csv',index=False)
    data=full_outcomes();plots(a,b,s,data)
    am=a.groupby(['column','protocol','method']).normalized_aulc.agg(['mean','std']).reset_index();am.to_csv(OUT/'aulc_summary.csv',index=False)
    bm=b.groupby(['column','protocol','method'])[['V1_r2','V2_r2','V1_rmse','V2_rmse','V1_mae','V2_mae','combined_normalized_rmse','V1_macro_compound_mae','V2_macro_compound_mae','V1_macro_compound_rmse','V2_macro_compound_rmse']].mean().reset_index();bm.to_csv(OUT/'budget100_mean_metrics.csv',index=False)
    cwm=cw.loc[cw.budget.eq(100)].groupby(['column','protocol','method','target'])[['rmse','mae']].mean().reset_index();cwm.to_csv(OUT/'budget100_center_width_summary.csv',index=False)
    compact=am.pivot(index=['column','protocol'],columns='method',values='mean').reindex(columns=ORDER).reset_index().round(4)
    for m in ORDER:compact.rename(columns={m:LABELS[m]},inplace=True)
    score_rows=[]
    for (c,p),g in am.groupby(['column','protocol']):
        ref=g.loc[g.method.isin(run.BASELINES)].sort_values('mean').iloc[0]
        for m in run.METHODS:
            q=pairs.loc[pairs.column.eq(c)&pairs.protocol.eq(p)&pairs.method.eq(m)&pairs.reference.eq(ref['method'])&pairs.endpoint.eq('aulc')].iloc[0]
            score_rows.append(dict(column=c,protocol=p,method=m,strongest_frozen_aulc_reference=ref['method'],gain_percent=100*q.relative_gain,seed_wins=q.wins,median_paired_delta=q.median_delta))
    scores=pd.DataFrame(score_rows);scores.to_csv(OUT/'strongest_reference_comparisons.csv',index=False)
    gate_rows=[dict(method=m,endpoint=e,replicated=g['replicated'],passing_contexts=str(g['contexts'])) for m,items in decision['gates'].items() for e,g in items.items()]

    normalized_comparisons=[]
    for (column,protocol),g in a.groupby(['column','protocol']):
        v=g.pivot(index='seed',columns='method',values='normalized_aulc')
        raw=v['raw_column_conditioned'];norm=v['mass_normalized_column_conditioned']
        bg=b.loc[b.column.eq(column)&b.protocol.eq(protocol)].groupby('method')[['combined_normalized_rmse','V1_rmse','V2_rmse']].mean()
        normalized_comparisons.append(dict(column=column,protocol=protocol,raw_aulc=raw.mean(),normalized_aulc=norm.mean(),aulc_gain_percent=100*(1-norm.mean()/raw.mean()),wins=int((norm-raw < -1e-7).sum()),raw_seed_sd=raw.std(),normalized_seed_sd=norm.std(),budget100_combined_gain_percent=100*(1-bg.loc['mass_normalized_column_conditioned','combined_normalized_rmse']/bg.loc['raw_column_conditioned','combined_normalized_rmse'])))
    normalized_comparisons=pd.DataFrame(normalized_comparisons)
    normalized_comparisons.to_csv(OUT/'normalized_vs_raw_comparisons.csv',index=False)
    residual_detail=[]
    for (column,protocol),g in a.groupby(['column','protocol']):
        v=g.pivot(index='seed',columns='method',values='normalized_aulc');model='physics_scale_residual'
        for reference in ['scale_only','local_identity_shrinkage','conditional_EA']:
            entry=dict(column=column,protocol=protocol,reference=reference,aulc_gain_percent=100*(1-v[model].mean()/v[reference].mean()),wins=int((v[model]-v[reference]<-1e-7).sum()),residual_seed_sd=v[model].std(),reference_seed_sd=v[reference].std())
            for dimension,level in [('source_q50','low'),('source_q50','top10'),('EA','low'),('EA','high')]:
                z=s.loc[s.column.eq(column)&s.protocol.eq(protocol)&s.budget.eq(100)&s.target.eq('V1')&s.dimension.eq(dimension)&s.level.eq(level)].groupby('method').rmse.mean()
                entry[dimension+'_'+level+'_V1_rmse_gain_percent']=100*(1-z[model]/z[reference])
            residual_detail.append(entry)
    residual_detail=pd.DataFrame(residual_detail);residual_detail.to_csv(OUT/'physics_residual_detailed_comparisons.csv',index=False)

    report='''# 显式柱物理条件迁移：科研解释\n\n**'''+decision['decision']+'''**。120 contexts 全部完成，240 neural fits + 120 fixed Ridge fits，四个实验臂与六个冻结参考，共 1200 metric records。全部预测先冻结再评估；历史 test 已暴露，审计只读冻结第一 seed 的已购训练标签，因此仍是开发性证据，不能作为独立外部确认。\n\n## 一、为什么 Scale 看起来这么强？\n\n质量比 2 / 6.25 / 10 捕获跨柱体量变化的一阶方向，但不是拟合系数的精确解释。budget100 compound 的历史 V1/V2 scale 为 8g 2.267/1.759、25g 5.795/4.268、40g 11.353/7.427。同一物理质量比不能同时解释两个输出。OLS 系数是 source-q50 平方加权的 ratio 均值；历史训练上端 10% 占约 46–70% x² 权重。稳定 seed 系数不等于普适物理规律。\n\n仓库有几何硬编码线索，但无可验证单位、床体积/壳体定义、制造商型号和实测来源。REAL_COLUMN_VOLUME_SCALE_NOT_IDENTIFIABLE_FROM_CURRENT_REPOSITORY_METADATA。8g 实为 4g+4g，质量和连接结构不可混为一谈。\n\n## 二、V1 和 V2 为什么 scale 不一样？\n\n训练 exact 8g 配对 center/width 中位比例约 1.87/1.59；relaxed 25g 约 3.88/2.83，40g 约 7.05/3.74。窗口位置与宽度呈不同缩放行为，与 V1/V2 不同系数相容。25g/40g 没有相同 flow 配对，EA、loading 与分子组成也混杂，不能据此宣称 dispersion 的物理因果规律。center 不是色谱峰顶，width 不是峰方差。详见训练分层比例和本轮 center/width errors。\n\n## 三、显式 Column Context 是否有用？\n\n下表为五 seeds 平均 AULC（低优）；每项单独对比全部冻结强参考，未以测试排名创建新选择策略。\n\n'''+table(compact)+'''\n\n相对各场景最低平均 AULC 冻结参考的描述性比较（正 gain 为改善）：\n\n'''+table(scores.round(4))+'''\n\n所有方法 budget100 绝对误差（mL）及 R²/combined NRMSE：\n\n'''+table(bm.round(4))+'''\n\n预注册 material gate：\n\n'''+table(pd.DataFrame(gate_rows))+'''\n\n完整每 seed/budget 指标在 all_metrics.csv，标准差/中位数/范围在 aggregate_metrics.csv；40g 的绝对误差必须与 NRMSE 同时阅读，不能称 TRANSFER_SOLVED。\n\n## 四、Physical Scale + Residual 是否优于经验 Scale/Shrinkage？\n\n固定 alpha=100 的 Ridge 使用 frozen 128D 分子表示、EA 与四个柱工程变量；没有验证集 alpha 搜索。是否更准以以上逐场景 AULC 和 budget100 errors 为准，稳定性以 aulc_summary.csv 的跨 seed SD 与 paired_comparisons.csv 为准。q50/EA 分层图及 error_stratification.csv 保留低/中/高与尾部失败，局部改善不能替代整体 material gate。\n\n## 五、归一化目标有没有让跨柱问题变简单？\n\nPHYSICAL_NORMALIZATION_PARTIALLY_SUPPORTED（训练审计）。四柱均值 CV：V1 0.846→0.149，V2 0.751→0.265，center 0.787→0.220，width 0.606→0.427。单柱 CV 除以常量后代数上不变；不能用它证明学习更简单。mL/g 输出的实际测试增益和 seed SD 由 raw/context mL/g 两臂检验，不将分布收拢等同于预测提升。postfreeze_target_distributions.csv 和 postfreeze_matched_discrepancy.csv 是全 canonical 数据的事后描述，未反馈模型。\n\n## 六、后续 Active Learning baseline\n\n'''
    if not decision['joint_endpoint_candidates']:
        report+='没有新模型同时满足两个复制端点的提升门槛。保留 Scale/Shrinkage 为主要 baseline 家族，Conditional EA/policy 与竞争性的 8g head-only 作冻结对照；不因单个场景 test 最优而构建 oracle，也不在本轮启动 AL。\n'
    else:
        report+='同时满足两个复制端点的候选为 '+', '.join(decision['joint_endpoint_candidates'])+'。其 AL 适用性仍需独立 compound/batch 确认和既有 UQ 资格，不在本轮执行 AL。\n'
    report+='''\n## 七、数据还缺什么？\n\n优先补真实 V0/示踪方法、柱内径/床长、粒径与 packing density、准确产品型号/批次和 4g+4g 连接体积；独立重复与批次 ID；crossed mass×flow（当前 4g 有 4/5/6/8/10，target 8g/25g/40g 固定 10/15/30 mL/min）；高 retention tail 和 source-unseen compounds。质量、几何、flow 无法独立识别。flow mL/min 不是线速度；flow_per_g 只是工程 proxy。模型成功也只能说明 explicit context has predictive value，不能说明质量或 flow 导致 shift；失败也不能证明不存在物理结构。\n\n## 图表\n\n'''
    for name in ['learned_vs_physical_scale','raw_vs_mass_normalized','center_width_distributions','aulc_by_model','budget100_rmse','q50_stratified_error','ea_stratified_v1_error']:report+=f'![{name}](plots/{name}.png)\n\n'
    report=report.replace('## 五、归一化目标有没有让跨柱问题变简单？', 'Residual 对强参考的定量比较（正数表示改善；seed SD 是重叠分区下的变动，不是独立实验方差）：\n\n'+table(residual_detail.round(4))+'\n\n## 五、归一化目标有没有让跨柱问题变简单？')
    report=report.replace('## 六、后续 Active Learning baseline', 'Normalized versus raw 的直接比较：\n\n'+table(normalized_comparisons.round(4))+'\n\n归一化臂初始 source function 在 mL 上保持一致，但 target 初值已经包含 packing_mass_ratio，而 raw 臂初值仍为 source q50；故该比较同时包含物理初值结构与 loss 数值尺度的影响，不是唯一识别数值优化难度的因果实验。\n\n## 六、后续 Active Learning baseline')
    report+='\n## Budget100 center/width errors (mL)\n\n'+table(cwm.round(4))+'\n'
    (OUT/'RESULT_INTERPRETATION.md').write_text(report)
    (OUT/'NEXT_STAGE_DECISION.md').write_text('# Physics column-conditioned transfer decision\n\n**'+decision['decision']+'**\n\n'+table(pd.DataFrame(gate_rows))+'\n\n'+('Retain Scale/Shrinkage; no new arm passes both replicated endpoints.' if not decision['joint_endpoint_candidates'] else 'Joint endpoint candidates: '+', '.join(decision['joint_endpoint_candidates']))+'\n\nDo not add features, attention, LR/alpha/architecture sweeps, column IDs, readout or source anchoring after test. No AL launched. Prioritize metadata verification, crossed mass × flow, independent batches/repeats and source-unseen/tail coverage. A negative result applies to this source-initialized training recipe, not physical impossibility. A predictive gain cannot identify a causal mass or flow mechanism. No claim of TRANSFER_SOLVED; 40g absolute errors remain separately reported.\n\nSee RESULT_INTERPRETATION.md, paired_comparisons.csv and budget100_metrics.csv for quantitative evidence.\n')
    print(table(compact));print(table(pd.DataFrame(gate_rows)))
if __name__=='__main__':main()

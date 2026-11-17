#!/usr/bin/env python3
"""Read frozen validation summaries only. Never import a runner or load predictions/truth."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/chrom_history_mpl')
import json, csv, gzip, hashlib, math, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
SEEDS = [157,887,2357,6101,12203]
CURRENT = 'new_free_cw_sol61_xhigh'
STAGES = [(333,525),(525,813),(813,1005),(1005,1485)]
NAMES = {'random32':'Random32','gradient_lcmd32':'Raw gradient LCMD32',
 'numerical_hybrid32':'Numerical Hybrid32','historical_free_mixed_model':'Historical Free32*',
 CURRENT:'CW-visible Free32','nocw':'NoCW32','free16_lcmd16':'Free16 + CW-LCMD16*',
 'cw_gradient_lcmd32':'CW-gradient LCMD32'}
SOURCES={}
def read(p):
    p=Path(p); p=p if p.is_absolute() else ROOT/p
    b=p.read_bytes(); key=str(p.relative_to(ROOT))
    SOURCES[key]={'path':key,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),
                  'role':'validation summary / training metadata; no test truth'}
    return b
def js(p): return json.loads(read(p))
def frame(p):
    import io
    return pd.read_csv(io.BytesIO(read(p)))
def table(name, rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT/'tables'/f'{name}.csv',index=False,float_format='%.14g')
    return d
def area(g):
    g=g.sort_values('budget'); x=g.budget.to_numpy(); y=g.nrmse.to_numpy()
    assert len(g)>1 and np.all(np.diff(x)==32)
    return float(np.sum(np.diff(x)*(y[1:]+y[:-1])*.5)/(x[-1]-x[0]))
def savefig(name, fig, caption):
    fig.text(.02,.015,caption,fontsize=8,va='bottom')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=175,bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)

def main():
    base=frame('reports/free_cw_final_20261112/validation_curves.csv')
    base=base.rename(columns={'validation_selection_score':'nrmse'})
    hashes=frame('reports/free_cw_final_20261112/final_curves_20261112/validation_curves.csv')
    base=base.merge(hashes[['method','seed','budget','fit_audit_sha256']],on=['method','seed','budget'],validate='one_to_one')
    extra=frame('reports/five_methods_cw_gradient_20261103/validation_curves.csv')
    extra=extra[extra.method.isin(['free16_lcmd16','cw_gradient_lcmd32'])].rename(columns={'validation_nrmse':'nrmse','fit_audit':'source'})
    rows=pd.concat([base,extra],ignore_index=True).to_dict('records')
    for seed in SEEDS:
        for r in range(22):
            p=Path(f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/random/round_{r:02d}/model/fit_audit.json')
            a=js(p)
            rows.append({'method':'random32','seed':seed,'budget':333+32*r,
                         'nrmse':a['best_validation_combined_normalized_rmse'],'source':str(p),
                         'fit_audit_sha256':SOURCES[str(p)]['sha256']})
    # Two-seed numerical extensions are completed states, not five-seed evidence.
    ext_curve=js('reports/llm_eight_to1549_20261018/current/validation_curves.json')
    extensions=[(26,'sol_xhigh_all_strategies_to1165_20261014'),(30,'sol_xhigh_all_strategies_to1293_20261015'),
                (34,'sol_xhigh_all_strategies_to1421_20261017'),(38,'sol_xhigh_all_strategies_to1549_20261018')]
    for seed in [157,6101]:
        for method,original in [('random32','random32'),('cw_gradient_lcmd32','center_width_lcmd')]:
            published={v['labels']:v['nrmse'] for v in ext_curve[str(seed)][original]}
            for row in rows:
                if row['method']==method and row['seed']==seed:
                    assert math.isclose(row['nrmse'],published[row['budget']],abs_tol=1e-11)
            for r in range(22,39):
                owner=next(name for end,name in extensions if r<=end)
                fp=Path(f'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/{owner}/runtime/seed_{seed}/{original}/round_{r:02d}/prediction_freeze.json')
                fr=js(fp);p=Path(fr['checkpoint_path']).parent/'fit_audit.json';a=js(p)
                assert a['checkpoint_sha256']==fr['checkpoint_sha256'] and fr['test_truth_access_count']==0
                assert math.isclose(a['best_validation_combined_normalized_rmse'],published[333+32*r],abs_tol=1e-11)
                source=str(p.relative_to(ROOT))
                rows.append({'method':method,'seed':seed,'budget':333+32*r,'nrmse':published[333+32*r],
                             'source':source,'fit_audit_sha256':SOURCES[source]['sha256']})
    enriched=[]
    for row in rows:
        p=ROOT/row['source']; a=js(p)
        assert SOURCES[row['source']]['sha256']==row['fit_audit_sha256'],p
        assert a['train_rows']==row['budget'] and a['test_labels_used_for_fit_or_checkpoint_selection']==0
        assert math.isclose(a['best_validation_combined_normalized_rmse'],row['nrmse'],abs_tol=1e-12)
        h=frame(p.parent/'history.csv'); best=h[h.epoch==a['best_epoch']]
        assert len(best)==1
        assert math.isclose(best.iloc[0].validation_selection_score,row['nrmse'],abs_tol=1e-11)
        row.update({k:best.iloc[0][k] for k in h.columns if k.startswith('validation_')})
        row.update({k:a[k] for k in ['checkpoint_sha256','initialization_hash','initialization_seed',
             'validation_ids_hash','train_ids_hash','training_seconds','epochs_run','best_epoch','parameter_count']})
        row['fit_round']=(row['budget']-333)//32
        enriched.append(row)
    df=pd.DataFrame(enriched).sort_values(['method','seed','budget'])
    assert not df.duplicated(['method','seed','budget']).any()
    for seed,g in df.groupby('seed'):
        assert g.initialization_hash.nunique()==1 and g.validation_ids_hash.nunique()==1
    scales=[]
    for seed in SEEDS:
        p=f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json'
        s=js(p)['contract']['preprocessing']['target_scales']
        scales.append({'seed':seed,**s})
        g=df[df.seed==seed]
        score=np.sqrt(.5*((g.validation_V1_rmse/s['V1'])**2+(g.validation_V2_rmse/s['V2'])**2))
        assert np.max(np.abs(score-g.nrmse))<1e-10
    table('normalization_scales',scales)
    table('round_metrics',df)
    registry=[]; phases=[]; gains=[]; thresholds=[]
    for (method,seed),g in df.groupby(['method','seed'],sort=False):
        g=g.sort_values('budget'); assert np.all(np.diff(g.budget)==32)
        registry.append({'method':method,'seed':seed,'min_budget':int(g.budget.min()),'max_budget':int(g.budget.max()),
          'points':len(g),'completed_batches':len(g)-1,'endpoint_nrmse':g.nrmse.iloc[-1],
          'full_raw_aulc':area(g),'source':g.source.iloc[-1],'served_model':
          'gpt-6.1-sol (locked)' if method==CURRENT else ('not applicable' if method in ['random32','gradient_lcmd32','numerical_hybrid32','cw_gradient_lcmd32'] else 'see protocol audit'),
          'initialization_hash':g.initialization_hash.iloc[0],'validation_ids_hash':g.validation_ids_hash.iloc[0]})
        for lo,hi in STAGES+[(333,1005),(333,1453),(1005,1453),(333,1485),(333,1549)]:
            z=g[(g.budget>=lo)&(g.budget<=hi)]
            if len(z)>1 and z.budget.min()==lo and z.budget.max()==hi:
                phases.append({'method':method,'seed':seed,'start':lo,'end':hi,'n_points':len(z),
                  'aulc':area(z),'start_nrmse':z.nrmse.iloc[0],'endpoint_nrmse':z.nrmse.iloc[-1],
                  'total_gain':z.nrmse.iloc[0]-z.nrmse.iloc[-1]})
        for (_,before),(_,after) in zip(g.iloc[:-1].iterrows(),g.iloc[1:].iterrows()):
            lo=before.budget; hi=after.budget
            rec={'method':method,'seed':seed,'round':int((lo-333)//32),'labeled_size_before':int(lo),
              'labeled_size_after':int(hi),'nrmse_before':before.nrmse,'nrmse_after':after.nrmse,
              'gain':before.nrmse-after.nrmse,'phase':next((f'{a}-{b}' for a,b in STAGES if a<=lo<b),'1485-1549'),
              'training_seconds':after.training_seconds,'best_epoch':after.best_epoch,'epochs_run':after.epochs_run}
            for target in ['V1','V2']:
                rec[f'{target}_rmse_gain']=before[f'validation_{target}_rmse']-after[f'validation_{target}_rmse']
            gains.append(rec)
        for cut in [.6,.5,.4,.35,.3]:
            z=g[g.nrmse<=cut]; hits=g.nrmse.le(cut).to_numpy()
            sustained=[int(g.budget.iloc[i]) for i in range(len(g)-2) if all(hits[i:i+3])]
            thresholds.append({'method':method,'seed':seed,'threshold':cut,
             'first_budget':None if z.empty else int(z.budget.min()),'right_censored':z.empty,
             'three_consecutive_first_budget':sustained[0] if sustained else None,'observed_to':int(g.budget.max())})
    registry=table('run_registry',registry); phase=table('stage_metrics',phases); gain=table('batch_gains',gains)
    table('threshold_crossings',thresholds)
    summaries=phase.groupby(['method','start','end']).agg(seeds=('seed','size'),aulc_mean=('aulc','mean'),aulc_sd=('aulc','std'),
         endpoint_mean=('endpoint_nrmse','mean'),endpoint_sd=('endpoint_nrmse','std'),total_gain_mean=('total_gain','mean')).reset_index()
    table('stage_summary',summaries)
    pairs=[]
    for comp in NAMES:
        if comp==CURRENT:continue
        shared=sorted(set(df[df.method==comp].seed)&set(SEEDS))
        cap=min(registry[(registry.method.isin([CURRENT,comp]))&registry.seed.isin(shared)].max_budget)
        for lo,hi in STAGES+[(333,1005),(333,1453),(1005,1453),(333,1485)]:
            if hi>cap:continue
            a=phase[(phase.method==CURRENT)&(phase.start==lo)&(phase.end==hi)&phase.seed.isin(shared)].set_index('seed')
            b=phase[(phase.method==comp)&(phase.start==lo)&(phase.end==hi)&phase.seed.isin(shared)].set_index('seed')
            for metric in ['aulc','endpoint_nrmse','total_gain']:
                for seed in shared:pairs.append({'comparator':comp,'seed':seed,'start':lo,'end':hi,'metric':metric,
                  'cw_free':a.loc[seed,metric],'reference':b.loc[seed,metric],'delta_cw_minus_reference':a.loc[seed,metric]-b.loc[seed,metric]})
    pairs=table('paired_effects',pairs)
    table('paired_effect_summary',pairs.groupby(['comparator','start','end','metric']).agg(n=('seed','size'),mean_delta=('delta_cw_minus_reference','mean'),sample_sd=('delta_cw_minus_reference','std'),negative_deltas=('delta_cw_minus_reference',lambda x:int((x<0).sum()))).reset_index())
    stat=[]
    for (m,s,p),g in gain.groupby(['method','seed','phase']):
        for eps in [0,.001,.005]:stat.append({'method':m,'seed':s,'phase':p,'epsilon':eps,'n':len(g),
           'improved':int((g.gain>eps).sum()),'worsened':int((g.gain< -eps).sum()),'near_zero':int((g.gain.abs()<=eps).sum()),
           'gain_mean':g.gain.mean(),'gain_sd':g.gain.std(),'worst_gain':g.gain.min(),'best_gain':g.gain.max()})
    table('gain_stability',stat)
    table('training_costs',df[df.budget>333].groupby('method').agg(fits=('budget','size'),recorded_training_seconds=('training_seconds','sum'),median_seconds=('training_seconds','median'),median_best_epoch=('best_epoch','median'),median_epochs=('epochs_run','median'),max_epochs=('epochs_run','max')).reset_index())
    # Two-way descriptive variance decomposition, not random-effect variance estimates.
    variance=[]
    for cap in [1005,1485]:
        eligible=registry.groupby('method').filter(lambda x:len(x)==5 and x.max_budget.min()>=cap).method.unique()
        for metric in ['aulc','endpoint_nrmse']:
            z=phase[(phase.start==333)&(phase.end==cap)&phase.method.isin(eligible)]
            y=z.pivot(index='seed',columns='method',values=metric).to_numpy(); grand=y.mean()
            seed_ss=y.shape[1]*np.sum((y.mean(axis=1)-grand)**2); meth_ss=y.shape[0]*np.sum((y.mean(axis=0)-grand)**2)
            total=np.sum((y-grand)**2); residual=np.sum((y-y.mean(axis=1)[:,None]-y.mean(axis=0)[None,:]+grand)**2)
            for component,ss in [('seed',seed_ss),('method',meth_ss),('interaction_unexplained',residual)]:
                variance.append({'end':cap,'metric':metric,'methods':len(eligible),'seeds':5,'component':component,'ss_fraction':ss/total})
    table('descriptive_variance',variance)
    draw(df,phase,gain)
    costs()
    verification={'status':'PASS','curve_points':len(df),'runs':len(registry),'batches':len(gain),
      'all_fit_hashes_checked':True,'every_best_epoch_crosschecked':True,'fixed_scale_formula_checked':True,
      'common_initialization_and_validation_ids_per_seed':True,'uniform_32_increment':True,
      'test_truth_reads':0,'new_model_calls':0,'new_training_runs':0,
      'limitations':['test prediction files intentionally not opened','Hybrid training sum excludes extra ensemble members unless recorded here','repeated budgets are not independent trials']}
    (OUT/'verification/learning_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (OUT/'verification/learning_input_manifest.json').write_text(json.dumps(list(SOURCES.values()),indent=2)+'\n')
    print(json.dumps(verification))

def draw(df,phase,gain):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=dict(zip(NAMES,plt.get_cmap('tab10').colors)); fig,axs=plt.subplots(4,2,figsize=(13,14),sharex=True,sharey=True)
    for ax,(method,name) in zip(axs.flat,NAMES.items()):
        z=df[df.method==method]
        for seed,g in z.groupby('seed'):ax.plot(g.budget,g.nrmse,lw=1,alpha=.65,label=str(seed))
        cap=int(z.groupby('seed').budget.max().min()); means=z[z.budget<=cap].groupby('budget').nrmse.mean()
        ax.plot(means.index,means.values,'k',lw=2,label='common-seed mean');ax.set_title(name+f' (n={z.seed.nunique()})');ax.grid(alpha=.2);ax.set_ylim(.1,1.15)
        ax.set_xlabel('Active training labels L');ax.set_ylabel('Validation combined NRMSE');ax.legend(fontsize=7,ncol=3)
    fig.suptitle('Saved row-level validation: all raw seed trajectories',fontsize=16); fig.tight_layout(rect=[0,.045,1,.97])
    savefig('learning_all_methods',fig,'Source: tables/round_metrics.csv; 416 validation rows/seed; seeds 157/887/2357/6101/12203 (CW-LCMD and 16+16: 157/6101 only).\nUnitless NRMSE; frozen L333 scales; raw nonmonotone values. * Mixed LLM history. Different endpoints are not a fair ranking.')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axs,['aulc','endpoint_nrmse']):
        z=phase[(phase.method.isin([CURRENT,'gradient_lcmd32','historical_free_mixed_model']))&phase.apply(lambda r:(r.start,r.end) in STAGES,axis=1)]
        for m in [CURRENT,'gradient_lcmd32','historical_free_mixed_model']:
            v=z[z.method==m].groupby(['start','end'])[metric].agg(['mean','std']).reindex(STAGES)
            ax.errorbar(np.arange(4),v['mean'],yerr=v['std'],marker='o',capsize=3,label=NAMES[m])
        ax.set_xticks(range(4),[f'{a}-{b}' for a,b in STAGES]);ax.set_ylabel(metric+' (unitless)');ax.set_xlabel('Stage, active labels');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('Stage mean ± sample SD: endpoint and area answer different questions');fig.tight_layout(rect=[0,.11,1,.94])
    savefig('learning_stage_aulc_endpoint',fig,'Source: tables/stage_metrics.csv; each point n=5 paired seeds; stage AULC uses every raw point and divides by stage width.\nBars describe seed dispersion, not confidence intervals. No extrapolation; same budget does not imply same model state.')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    order=[CURRENT,'gradient_lcmd32','historical_free_mixed_model','nocw','numerical_hybrid32','random32']
    z=gain[gain.labeled_size_after<=1005]
    axs[0].boxplot([z[z.method==m].gain for m in order],tick_labels=[NAMES[m].replace(' ','\n') for m in order],showfliers=True)
    axs[0].axhline(0,color='k',ls='--',lw=1);axs[0].set_ylabel('NRMSE before - after (unitless)');axs[0].set_title('105 sequential gains/method, L333-1005')
    z=phase[(phase.start==333)&(phase.end==1005)&phase.method.isin(order)]
    for seed,g in z.groupby('seed'):
        vals=g.set_index('method').reindex(order).aulc;axs[1].plot(range(len(order)),vals,'o-',label=str(seed),alpha=.8)
    axs[1].set_xticks(range(len(order)),[NAMES[m].replace(' ','\n') for m in order]);axs[1].set_ylabel('Raw normalized AULC (unitless)');axs[1].legend(title='Seed');axs[1].grid(alpha=.2)
    fig.tight_layout(rect=[0,.12,1,1]);savefig('learning_gain_and_stability',fig,'Source: tables/batch_gains.csv and stage_metrics.csv; five shared seeds, 21 batches/seed.\nBatch changes include model refitting and validation variation; 105 batches are not 105 independent replicates.')

def costs():
    p=ROOT/'reports/free_cw_final_20261112/costs.json.gz'; raw=read(p); c=json.loads(gzip.decompress(raw))
    attempts=c['requests']; by=collections.defaultdict(list)
    for r in attempts:by[int(r['seed']),int(r['round'])].append(r)
    out=[]
    for (s,r),v in sorted(by.items()):
        good=[x for x in v if x.get('success')]; usage=[x.get('usage') or {} for x in good]
        out.append({'seed':s,'round':r,'transport_attempts':len(v),'successful_complete_responses':len(good),
          'distinct_answer_turns':len({x['answer_turn'] for x in v}),
          'elapsed_attempt_seconds_sum':sum(float(x.get('elapsed_seconds') or 0) for x in v),
          'successful_input_tokens':sum(int(x.get('input_tokens') or 0) for x in usage),
          'successful_output_tokens':sum(int(x.get('output_tokens') or 0) for x in usage),
          'successful_cached_tokens':sum(int((x.get('input_tokens_details') or {}).get('cached_tokens') or 0) for x in usage),
          'usage_present':sum(bool(x) for x in usage)})
    d=table('cost_by_batch',out);table('cost_by_seed',d.groupby('seed').sum(numeric_only=True).drop(columns='round').reset_index())
    summary={'physical_transport_attempts':len(attempts),'successful_complete_responses':int(d.successful_complete_responses.sum()),
      'distinct_seed_round_answer_turn_keys':int(d.distinct_answer_turns.sum()),'accepted_batches':c['accepted_batches'],
      'reported_transport_retries':c['transport_retries'],'reported_correction_requests':c['correction_requests'],
      'recorded_success_input_tokens':int(d.successful_input_tokens.sum()),'recorded_success_output_tokens':int(d.successful_output_tokens.sum()),
      'currency_cost':None,'monetary_price_verified':False,
      'note':'Physical receipts are not billable LLM completions. Logical keys may span request revisions. Token sums cover successful responses with recorded usage, not independently verified provider invoices. Elapsed sums include overlapping attempts and are not wall-clock duration.'}
    (OUT/'tables/cost_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Descriptive trajectory-level links to saved next-fit validation losses."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
OUT=Path(__file__).resolve().parent
FEATURES=['hamming_novelty_random_delta','condition_novelty_random_delta','structure_novelty',
          'rarity_random_delta','joint_coverage_gain_random_delta','single_factor_history_fraction',
          'new_molecule_slot_fraction','batch_condition_nn']

def partial(d,feature):
    cols=[feature,'next_gain','L_before','validation_nrmse_before','method']
    a=d[cols].dropna()
    if len(a)<12:return np.nan
    x=rankdata(a[feature]);y=rankdata(a.next_gain)
    if np.std(x)<1e-10:return np.nan
    controls=np.column_stack([np.ones(len(a)),rankdata(a.L_before),rankdata(a.validation_nrmse_before),
        pd.get_dummies(a.method,drop_first=True,dtype=float).to_numpy()])
    x=x-controls@np.linalg.lstsq(controls,x,rcond=None)[0]
    y=y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
    if np.std(x)<1e-10 or np.std(y)<1e-10:return np.nan
    return float(np.corrcoef(x,y)[0,1])

def main():
    d=pd.read_csv(OUT/'batch_metrics.csv');records=[]
    score_path=OUT/'candidate_score_percentiles.csv'
    features=FEATURES.copy()
    if score_path.exists():
        scores=pd.read_csv(score_path).pivot(index=['method','seed','round'],columns='feature',values='selected_percentile_mean').reset_index()
        scores=scores.rename(columns={'G_cw_distance':'G_percentile','U_quantile_width':'U_proxy_percentile'})
        d=d.merge(scores,on=['method','seed','round'],how='left',validate='one_to_one')
        features+=['G_percentile','U_proxy_percentile']
    for (m,s),g in d.groupby(['method','seed']):
        for stage in ['all','early','middle','late','extension']:
            a=g if stage=='all' else g[g.phase==stage]
            for f in features:
                valid=a[[f,'next_gain']].dropna()
                rho=spearmanr(valid[f],valid.next_gain).statistic if len(valid)>3 and valid[f].nunique()>1 else np.nan
                records.append(dict(method=m,seed=s,stage=stage,feature=f,n=len(valid),rho=rho,partial_rho=partial(a,f)))
    pd.DataFrame(records).to_csv(OUT/'gain_seed_associations.csv',index=False)
    summary=[];rng=np.random.default_rng(20261116)
    for scope,group in [('five_seed_all_methods',d),('A_common_two_seed',d[d.seed.isin([157,6101])&(d.L_after<=1453)])]:
        for f in features:
            vals=np.array([partial(g,f) for _,g in group.groupby('seed')]);vals=vals[np.isfinite(vals)]
            boot=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)])
            summary.append(dict(scope=scope,feature=f,n_campaign_seed_blocks=len(vals),mean_seed_partial_rho=vals.mean(),
                seed_min=vals.min(),seed_max=vals.max(),positive_seeds=int((vals>0).sum()),
                cluster_bootstrap_low=np.quantile(boot,.025),cluster_bootstrap_high=np.quantile(boot,.975),
                eligible_batches=int(group[[f,'next_gain']].dropna().shape[0]),
                interpretation='exploratory association; block bootstrap is descriptive with 2/5 seeds; not causal'))
    pd.DataFrame(summary).to_csv(OUT/'gain_associations.csv',index=False)
    # Lag two excludes terminal batches and never backfills a missing update.
    lag=d.merge(d[['method','seed','round','next_gain']].assign(round=lambda x:x['round']-1).rename(columns={'next_gain':'following_gain'}),
        on=['method','seed','round'],how='left')
    lag.to_csv(OUT/'gain_linked_batches.csv',index=False)
    lags=[]
    for (m,s),g in lag.groupby(['method','seed']):
        for f in features:
            a=g[[f,'following_gain']].dropna()
            lags.append(dict(method=m,seed=s,feature=f,n=len(a),lag2_rho=spearmanr(a[f],a.following_gain).statistic if a[f].nunique()>1 else np.nan))
    pd.DataFrame(lags).to_csv(OUT/'lag2_associations.csv',index=False)
    # Differences are paired within a seed before averaging; rows are not replicates.
    seed=pd.read_csv(OUT/'seed_metrics.csv');pairs=[]
    a=seed[seed.scope=='A_common'].set_index(['method','seed'])
    base='cw_gradient_lcmd32'
    for m in d.method.unique():
        if m==base:continue
        for s in [157,6101]:
            for f in FEATURES+['hamming_novelty','rarity','joint_coverage_gain']:
                pairs.append(dict(method=m,reference=base,seed=s,feature=f,delta=float(a.loc[(m,s),f]-a.loc[(base,s),f])))
    pd.DataFrame(pairs).to_csv(OUT/'paired_behavior_differences.csv',index=False)
    mc=[]
    for f in ['hamming_novelty','rarity','joint_coverage_gain']:
        diff=d[f+'_random_delta_first100']-d[f+'_random_delta']
        mc.append(dict(metric=f,mean_absolute_delta100_vs500=diff.abs().mean(),p95_absolute_delta=diff.abs().quantile(.95),
            mean_mcse500=d[f+'_random_mcse'].mean(),n_states=len(d)))
    pd.DataFrame(mc).to_csv(OUT/'random_stability.csv',index=False)
    numerical=[c for c in d.select_dtypes('number').columns if c not in ['seed','round','L_before','L_after']]
    d.groupby(['method','seed','phase'])[numerical].mean().reset_index().to_csv(OUT/'phase_metrics.csv',index=False)
    sensitivity=[]
    for (m,s),group in d.groupby(['method','seed']):
        for lhs,rhs in [('condition_novelty','condition_novelty_rms'),('rarity_bins4_random_delta','rarity_bins8_random_delta'),
                        ('rarity_bins8_random_delta','rarity_bins16_random_delta')]:
            sensitivity.append(dict(method=m,seed=s,measure1=lhs,measure2=rhs,
                spearman=spearmanr(group[lhs],group[rhs]).statistic,n=len(group)))
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_metrics.csv',index=False)
    print('Performance links',len(d),'seed coefficients',len(records),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent review of historical gain joins and rank statistics; no labels."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.linalg import null_space
D=Path(__file__).resolve().parents[1]
def read(name): return pd.read_csv(D/'tables'/name,float_precision='round_trip')
def ranks(x):return rankdata(np.round(np.asarray(x,dtype=float),12))
links=read('gain_linked_batches.csv'); fits=read('round_metrics.csv'); cor=read('gain_feature_seed_correlations.csv')
summary=read('gain_feature_summary.csv'); geo=read('behavior_geometry.csv')
assert len(links)==873 and not links.duplicated(['method','seed','round']).any()
assert (links.labeled_size_before==333+32*links['round']).all()
assert (links.labeled_size_after==links.labeled_size_before+32).all()
assert (links.L_before==links.labeled_size_before).all() and (links.L_after==links.labeled_size_after).all()
idx=fits.set_index(['method','seed','budget']); errors=[]
for row in links.itertuples():
    a=idx.loc[(row.method,row.seed,row.labeled_size_before)];b=idx.loc[(row.method,row.seed,row.labeled_size_after)]
    errors.extend([abs(row.nrmse_before-a.nrmse),abs(row.nrmse_after-b.nrmse),abs(row.gain-a.nrmse+b.nrmse)])
    assert row.checkpoint_sha256==a.checkpoint_sha256
assert max(errors)<1e-10
cw=geo[geo.method=='new_free_cw_sol61_xhigh']
assert len(cw)==360
assert cw.groupby(['seed','space']).size().eq(36).all()
geocount=0
for space,g in geo.groupby('space'):
    z=g.set_index(['method','seed','round'])
    for row in links.itertuples():
        key=(row.method,row.seed,row.round)
        if key not in z.index:continue
        for col in ['reference_coverage_relative_drop','reference_coverage_relative_drop_random_percentile',
                    'batch_nn_distance_mean_random_percentile','selected_to_l_distance_mean_random_percentile']:
            assert np.isclose(getattr(row,space+'_'+col),z.loc[key,col],rtol=0,atol=1e-14)
            geocount+=1
corr_errors=[]; counts={'spearman':0,'partial_rank_correlation':0,'partial_rank_without_before':0}
for row in cor.itertuples():
    g=links[(links.method==row.method)&(links.seed==row.seed)]
    if pd.notna(row.phase):g=g[g.phase==row.phase]
    g=g.dropna(subset=[row.feature,'gain','nrmse_before'])
    assert len(g)==row.n_batches
    for metric in counts:
        expected=getattr(row,metric)
        if not np.isfinite(expected):continue
        x,y=ranks(g[row.feature]),ranks(g.gain)
        if metric=='spearman':
            x=x-x.mean();y=y-y.mean()
        else:
            columns=[np.ones(len(g)),ranks(g.labeled_size_before)]
            if metric=='partial_rank_correlation':columns.append(ranks(g.nrmse_before))
            columns.append(pd.get_dummies(g.phase,drop_first=True,dtype=float).to_numpy())
            # Independent orthogonal complement; does not use the production lstsq residual implementation.
            N=null_space(np.column_stack(columns).T)
            x=N@(N.T@x);y=N@(N.T@y)
        observed=float(np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)))
        corr_errors.append(abs(observed-expected));counts[metric]+=1
assert max(corr_errors)<1e-10
summary_errors=[]
for row in summary.itertuples():
    g=cor[(cor.method==row.method)&(cor.feature==row.feature)&cor.phase.isna()][row.metric].dropna()
    assert len(g)==row.n_seeds
    assert int((g>0).sum())==row.positive_seeds and int((g<0).sum())==row.negative_seeds
    summary_errors.extend([abs(g.mean()-row.mean_seed_correlation),abs(g.std(ddof=1)-row.sample_sd)])
assert max(summary_errors)<1e-12
for name in ['model_batch_metrics.csv','reasoning_batches.csv']:
    d=read(name);assert len(d)==180 and not d.duplicated(['method','seed','round']).any()
    assert (d.labeled_size_before==333+32*d['round']).all()
reviewed=['scripts/analyze_gain_links.py','ACQUISITION_VS_LEARNING_GAIN.md','PROTOCOL_AND_FAIRNESS_AUDIT.md',
          'tables/gain_linked_batches.csv','tables/gain_feature_seed_correlations.csv','tables/gain_feature_summary.csv',
          'figures/gain_feature_correlations.png','figures/gain_coverage_scatter.png']
result={
 'status':'PASS_AFTER_CORRECTIONS','reviewer':'behavior subagent independent of gain-analysis author','review_date':'2026-11-12 Asia/Shanghai',
 'blocking_findings_remaining':0,
 'checks':{'joined_batches':873,'temporal_alignment_max_error':max(errors),'all_preselection_checkpoint_sha_match':True,
   'cw_raw_and_cw_geometry_rows':360,'per_seed_space_rows':36,'geometry_joined_values_checked':geocount,
   'independent_correlation_checks':counts,'independent_orthogonal_projection_max_error':max(corr_errors),
   'summary_max_error':max(summary_errors),'model_reasoning_round_alignment':'180 each, all aligned',
   'constant_features':'undefined correlations remain NA','pseudo_independence':'no batch-level p-values; bootstrap samples seed correlations',
   'information_roles':'selected residuals post-reveal; validation gain and before score offline only; raw/cw/latent separately named',
   'visual_qa':'two revised PNGs reviewed; explicit colorbars outside seed panels; no data occlusion'},
 'resolved_findings':[
   {'id':'R1','severity':'P2','problem':'Colorbars covered seed12203 data in both gain figures','resolution':'Explicit external cax; inspected corrected figures'},
   {'id':'R2','severity':'P2','problem':'Loading-contrast label ambiguous for sample-volume-only feature','resolution':'Changed to sample-volume contrast to history'},
   {'id':'R3','severity':'P2','problem':'14-digit joint export changed a near-equal HHI rank; independent partial differed 0.005443','resolution':'17-digit export, round-trip CSV parsing, documented 12-decimal ranking tie rule; all statistics independently pass'},
   {'id':'R4','severity':'P2','problem':'Gain shares before-NRMSE with adjustment; selected-CW and correction-count signs sensitive to adjustment','resolution':'Report explicitly discusses mathematical coupling/regression to mean; partial_rank_without_before added with 4/5 counterexamples'}],
 'remaining_limits':[
   'Five overlapping-split seed trajectories do not identify causal acquisition effects; reported bootstrap remains descriptive',
   'Budget+phase and before-NRMSE are partial state controls, not repeated identical-state training replicates',
   'Reference coverage percentiles use 128 references and 16 random batches; state-normalized proxies are not common information utility',
   'Behavior regressions cover six method identities; the extra two-seed arms lack full acquisition features and are disclosed',
   'Protocol label-access incident is disclosed; this review examined derived allowed files only and does not erase that incident'],
 'sources':[{'path':p,'sha256':hashlib.sha256((D/p).read_bytes()).hexdigest()} for p in reviewed]}
(D/'verification/independent_gain_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result['checks'],ensure_ascii=False,indent=2))

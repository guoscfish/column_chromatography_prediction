#!/usr/bin/env python3
"""Exploratory seed-level associations. No selection, training, or hidden labels."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/chrom_history_mpl')
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
CURRENT='new_free_cw_sol61_xhigh'
inputs=[]
def read(name):
    p=OUT/'tables'/name;inputs.append({'path':str(p.relative_to(OUT.parents[2])), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return pd.read_csv(p,float_precision='round_trip')
def write(name,df):df.to_csv(OUT/'tables'/name,index=False,float_format='%.17g')
def ranks(v):return rankdata(np.round(np.asarray(v,dtype=float),12))
def rho(x,y):
    if len(x)<6 or np.std(x)<1e-12 or np.std(y)<1e-12:return np.nan
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])
def partial(g,f,include_before=True):
    if len(g)<12 or g[f].nunique()<3:return np.nan
    # Rank-transform x, y, state; stage indicators allow stage intercept shifts.
    x=ranks(g[f]);y=ranks(g.gain)
    columns=[np.ones(len(g)),ranks(g.labeled_size_before)]
    if include_before:columns.append(ranks(g.nrmse_before))
    columns.append(pd.get_dummies(g.phase,drop_first=True,dtype=float).to_numpy())
    cov=np.column_stack(columns)
    xr=x-cov@np.linalg.lstsq(cov,x,rcond=None)[0];yr=y-cov@np.linalg.lstsq(cov,y,rcond=None)[0]
    if np.std(xr)<1e-10 or np.std(yr)<1e-10:return np.nan
    return float(np.corrcoef(xr,yr)[0,1])
def main():
    gains=read('batch_gains.csv');b=read('behavior_batches.csv');keys=['method','seed','round']
    link=gains.merge(b.drop(columns=[c for c in ['gain','phase','labeled_size_before','labeled_size_after'] if c in b]),on=keys,validate='one_to_one')
    assert np.max(abs(link.gain-(link.validation_nrmse_before-link.validation_nrmse_after)))<1e-10
    geo=read('behavior_geometry.csv');geo=geo[geo.available.astype(str).str.lower().eq('true')]
    gm=['reference_coverage_relative_drop','reference_coverage_relative_drop_random_percentile',
        'batch_nn_distance_mean_random_percentile','selected_to_l_distance_mean_random_percentile']
    for space,g in geo.groupby('space'):
        z=g[keys+gm].rename(columns={k:f'{space}_{k}' for k in gm})
        assert not z.duplicated(keys).any(); link=link.merge(z,on=keys,how='left',validate='one_to_one')
    model=read('model_batch_metrics.csv'); model=model.drop(columns=['labeled_size_before','phase'])
    link=link.merge(model,on=keys,how='left',validate='one_to_one')
    reasoning=read('reasoning_batches.csv')
    rf=['evidence_molecule_hhi','memory_top8_evidence_fraction','previous_batch_evidence_fraction',
        'high_confidence_n','weakened_rejected_n','feedback_revision_keyword','complete_response_n','first_complete_response_legal']
    link=link.merge(reasoning[keys+rf],on=keys,how='left',validate='one_to_one')
    for c in ['feedback_revision_keyword','first_complete_response_legal']:
        link[c]=link[c].map({True:1,False:0,'True':1,'False':0})
    features=['unique_molecules','new_molecule_slot_fraction','within_tanimoto_distance_mean','unique_scaffolds',
       'within_any_single_selected_fraction','cross_only_sample_volume_selected_fraction',
       'within_exact_selected_fraction','ea_fraction_mean','novel_recorded_condition_sets']
    features += [c for c in link if any(c==f'{s}_{m}' for s in geo.space.unique() for m in gm)]
    features += ['selected_error_combined_nrmse','selected_residual_asymmetry_v2_minus_v1',
       'selected_cw_percentile_mean','selected_v1_signed_width_mean_ml','selected_v2_signed_width_mean_ml']+rf
    rows=[]
    for (method,seed),g in link.groupby(['method','seed']):
        for feature in features:
            z=g.dropna(subset=[feature,'gain','nrmse_before'])
            if len(z)<6:continue
            rows.append({'method':method,'seed':seed,'feature':feature,'n_batches':len(z),
              'spearman':rho(z[feature],z.gain),'partial_rank_correlation':partial(z,feature),
              'partial_rank_without_before':partial(z,feature,False),
              'time_role':'post_reveal_diagnostic' if feature in ['selected_error_combined_nrmse','selected_residual_asymmetry_v2_minus_v1'] else 'preselection_or_proposal',
              'gain_role':'post_fit_validation'})
            for phase,h in z.groupby('phase'):
                rows.append({'method':method,'seed':seed,'feature':feature,'phase':phase,'n_batches':len(h),
                  'spearman':rho(h[feature],h.gain),'partial_rank_correlation':np.nan,
                  'time_role':'post_reveal_diagnostic' if feature in ['selected_error_combined_nrmse','selected_residual_asymmetry_v2_minus_v1'] else 'preselection_or_proposal','gain_role':'post_fit_validation'})
    cor=pd.DataFrame(rows);write('gain_feature_seed_correlations.csv',cor)
    summaries=[];rng=np.random.default_rng(20261112)
    for (method,feature),g in cor[cor.phase.isna()].groupby(['method','feature']):
        for metric in ['spearman','partial_rank_correlation','partial_rank_without_before']:
            a=g[metric].dropna().to_numpy()
            if len(a)<2:continue
            boot=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
            summaries.append({'method':method,'feature':feature,'metric':metric,'n_seeds':len(a),
                'mean_seed_correlation':a.mean(),'sample_sd':a.std(ddof=1),'positive_seeds':int((a>0).sum()),
                'negative_seeds':int((a<0).sum()),'seed_bootstrap_p025':np.quantile(boot,.025),'seed_bootstrap_p975':np.quantile(boot,.975),
                'min_leave_one_seed_out_mean':min(np.delete(a,i).mean() for i in range(len(a))),
                'max_leave_one_seed_out_mean':max(np.delete(a,i).mean() for i in range(len(a)))})
    summary=pd.DataFrame(summaries);write('gain_feature_summary.csv',summary)
    write('gain_linked_batches.csv',link)
    # Explicitly exploratory joint-profile descriptive check; never an acquisition rule.
    profiles=[]
    f='raw_gradient_reference_coverage_relative_drop_random_percentile'
    if f in link:
        z=link[link.method==CURRENT].dropna(subset=[f]).copy()
        for seed,g in z.groupby('seed'):
            for cov_hi in [False,True]:
                for local_hi in [False,True]:
                    mask=(g[f]>=g[f].median()).eq(cov_hi)&(g.within_any_single_selected_fraction>=g.within_any_single_selected_fraction.median()).eq(local_hi)
                    h=g[mask];profiles.append({'seed':seed,'coverage_above_within_seed_median':cov_hi,'local_contrast_above_within_seed_median':local_hi,
                         'n_batches':len(h),'gain_mean':h.gain.mean(),'budget_before_mean':h.labeled_size_before.mean(),'nrmse_before_mean':h.nrmse_before.mean()})
        write('gain_exploratory_joint_profiles.csv',pd.DataFrame(profiles))
    chosen=['new_molecule_slot_fraction','within_tanimoto_distance_mean','within_any_single_selected_fraction',
        'cross_only_sample_volume_selected_fraction','raw_gradient_reference_coverage_relative_drop_random_percentile',
        'cw_gradient_reference_coverage_relative_drop_random_percentile','selected_error_combined_nrmse','selected_cw_percentile_mean',
        'selected_v2_signed_width_mean_ml','memory_top8_evidence_fraction']
    z=cor[(cor.method==CURRENT)&cor.phase.isna()&cor.feature.isin(chosen)]
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    labels=['new molecule slots','Tanimoto diversity','within-batch single-factor','sample-volume contrast to history',
       'raw coverage vs random','CW coverage vs random','selected actual residual*','selected CW percentile','selected V2 width','top8-error evidence citations']
    for ax,metric in zip(axs,['spearman','partial_rank_correlation']):
        piv=z.pivot(index='feature',columns='seed',values=metric).reindex(chosen)
        im=ax.imshow(piv.to_numpy(),vmin=-.7,vmax=.7,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(len(piv.columns)),piv.columns);ax.set_yticks(range(len(chosen)),labels);ax.set_title(metric.replace('_',' '))
        for i in range(len(chosen)):
            for j in range(len(piv.columns)):
                v=piv.iloc[i,j]
                if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
    axs[1].set_yticklabels([])
    fig.suptitle('CW-Free: batch features versus the next observed learning gain',fontsize=14)
    fig.text(.03,.015,'Source: tables/gain_feature_seed_correlations.csv; five seeds, up to 36 sequential batches/seed. *Post-reveal feature, unavailable for selection.\nRight panel controls ranked budget, pre-fit validation NRMSE and phase. Exploratory correlations, no causal claim or independent-batch p-values.',fontsize=8)
    fig.subplots_adjust(left=.28,right=.89,bottom=.14,top=.89,wspace=.12)
    cax=fig.add_axes([.915,.25,.015,.52]);fig.colorbar(im,cax=cax)
    fig.savefig(OUT/'figures/gain_feature_correlations.png',dpi=175,bbox_inches='tight');fig.savefig(OUT/'figures/gain_feature_correlations.pdf',bbox_inches='tight');plt.close(fig)
    # Requested direct numerical-coverage/gain view preserves all five seeds.
    f='raw_gradient_reference_coverage_relative_drop_random_percentile'
    if f in link:
        fig,axs=plt.subplots(1,5,figsize=(15,3.7),sharex=True,sharey=True)
        for ax,(seed,g) in zip(axs,link[link.method==CURRENT].groupby('seed')):
            im=ax.scatter(g[f],g.gain,c=g.labeled_size_before,cmap='viridis',vmin=333,vmax=1453,s=17)
            ax.axhline(0,c='grey',lw=.7);ax.set_title(f'Seed {seed}, n={g[f].notna().sum()}');ax.set_xlabel('Coverage percentile')
        axs[0].set_ylabel('NRMSE before - after')
        fig.text(.02,.01,'Frozen raw geometry; 128 fixed reference points, 16 same-state random batches; percentile is discrete. Source: gain_linked_batches.csv.\nA coverage proxy is not true information gain. Points within a seed are dependent. Validation outcomes never enter historical selectors.',fontsize=8)
        fig.subplots_adjust(bottom=.25,top=.87,right=.87,wspace=.15)
        cax=fig.add_axes([.9,.3,.012,.5]);fig.colorbar(im,cax=cax,label='L before')
        fig.savefig(OUT/'figures/gain_coverage_scatter.png',dpi=175,bbox_inches='tight');plt.close(fig)
    (OUT/'verification/gain_input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (OUT/'verification/gain_verification.json').write_text(json.dumps({'status':'PASS','joined_batches':len(link),'cw_batches':int((link.method==CURRENT).sum()),
     'keys_unique':not link.duplicated(keys).any(),'historical_gain_alignment':'all matched <1e-10','bootstrap_unit':'seed; entire within-seed correlation retained',
     'inference':'exploratory, no p-values, many correlated features','controls_include_validation':'offline diagnostic only'},indent=2)+'\n')
    print('PASS',len(link),'linked batches;',len(summary),'seed-level summaries')
if __name__=='__main__':main()

"""Deterministic reporting over every finite and failed preregistered fit."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gollum_gp_matplotlib')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from data_boundary import HERE, dump

KEYS=['data_seed','budget_state','endpoint','representation','method','initialization']


def main():
    rows=[];fits=[];stability=[];raw={};failures=[];templates=[]
    for p in sorted((HERE/'phase1').glob('*.json')):
        d=json.loads(p.read_text())
        if 'method' not in d:
            continue
        identity={k:d[k] for k in KEYS}
        if d['status']!='finite_fit':
            failures.append({**identity,'error':d.get('error')});continue
        raw[tuple(d[k] for k in KEYS)]=d
        audit=d['fit_audit'];opt=audit['optimizer']
        f={**identity,'optimizer_success':audit['optimizer_success'],'optimizer_message':audit['optimizer_message'],
           'gradient_inf':opt.get('final_gradient_inf'),'projected_gradient_inf':opt.get('final_projected_gradient_inf'),
           'noise_variance':audit['noise_variance'],'signal_variance':audit['signal_variance'],
           'noise_signal_ratio':audit['noise_signal_ratio'],'condition_number':audit['covariance_condition_number'],
           'training_mll':audit['marginal_log_likelihood'],'iterations':opt['iterations'],
           'first_stage_iterations':opt.get('first_stage',{}).get('iterations'),
           'training_seconds':audit['training_seconds'],'total_seconds':d['elapsed_seconds'],
           'boundary_hit':opt['boundary_hit'],'temperature':d['calibration_temperature'],
           'compressed_pair_fraction':audit.get('compressed_pair_fraction'),'lengthscale':audit['lengthscale'],
           'training_rows':d['training_rows'],'calibration_rows':d['calibration_rows'],'check_rows':d['check_rows'],
           'gradient_qualified':bool(audit['optimizer_success'] and opt.get('final_projected_gradient_inf',float('inf'))<1e-3)}
        fits.append(f)
        for role,variants in d['metrics'].items():
            for calibration,m in variants.items():
                values={k:v for k,v in m.items() if np.isscalar(v) or v is None}
                rows.append({**identity,'role':role,'calibration':calibration,**values,'temperature':d['calibration_temperature']})
        rob=d.get('template_robustness')
        if rob is not None:
            templates.append({**identity,'variance_spearman':rob['variance_spearman'],
                              'mean_absolute_change_ml':rob['mean_absolute_change_ml'],
                              **{f'{m}_overlap':v for m,v in rob['top32_overlap'].items()}})
    if not rows:
        raise ValueError('no actual results')
    output=HERE/'summary';output.mkdir(exist_ok=True)
    metrics=pd.DataFrame(rows);fitdf=pd.DataFrame(fits)
    metrics.to_csv(output/'phase1_metrics.csv',index=False)
    fitdf.to_csv(output/'phase1_fits.csv',index=False)
    pd.DataFrame(templates).to_csv(output/'phase1_template_stability.csv',index=False)
    dump(output/'phase1_failures.json',failures)
    for key,a in raw.items():
        seed,budget,endpoint,rep,method,init=key
        counterparts=[('initialization', (seed,budget,endpoint,rep,method,29))] if init==17 else []
        if method not in ('original','frozen'):
            counterparts.append(('method_vs_original',(seed,budget,endpoint,rep,'original',init)))
        for kind,other in counterparts:
            if other not in raw or other==key:
                continue
            b=raw[other]
            av=np.asarray(a['acquisitions']['latent_variance']);bv=np.asarray(b['acquisitions']['latent_variance'])
            rho=float(spearmanr(av,bv).statistic) if np.ptp(av)>0 and np.ptp(bv)>0 else None
            common={k:a[k] for k in KEYS}
            r={**common,'comparison_kind':kind,'other_method':b['method'],'other_initialization':b['initialization'],
               'variance_rank_spearman':rho}
            for name,m in a['acquisitions']['methods'].items():
                r[name+'_top32_overlap']=len(set(m['selected_candidate_positions'])&set(b['acquisitions']['methods'][name]['selected_candidate_positions']))/32
            stability.append(r)
    pd.DataFrame(stability).to_csv(output/'phase1_acquisition_stability.csv',index=False)
    acquisition_rows=[]
    for d in raw.values():
        for method,a in d['acquisitions']['methods'].items():
            acquisition_rows.append({**{k:d[k] for k in KEYS},'acquisition':method,'unique_molecules':a['unique_molecules'],
                                     'batch_information':a['batch_information'],
                                     'mean_absolute_batch_latent_correlation':a['mean_absolute_batch_latent_correlation'],
                                     'seconds_all_three':d['acquisitions']['seconds'],
                                     **{k+'_distinct':v for k,v in a['condition_unique_counts'].items()}})
    pd.DataFrame(acquisition_rows).to_csv(output/'phase1_acquisition_coverage.csv',index=False)
    means=metrics.groupby(['data_seed','budget_state','representation','method','role','calibration'],as_index=False).mean(numeric_only=True)
    means.to_csv(output/'phase1_seed_budget_means.csv',index=False)
    group=['data_seed','representation','method','role','calibration']
    seedmeans=means.groupby(group,as_index=False).mean(numeric_only=True)
    seedmeans.to_csv(output/'phase1_seed_means.csv',index=False)
    aggregate=seedmeans.groupby(['representation','method','role','calibration'],as_index=False).mean(numeric_only=True)
    aggregate.to_csv(output/'phase1_aggregate.csv',index=False)
    pairs=[]
    for (seed,rep,role,cal),g in seedmeans.groupby(['data_seed','representation','role','calibration']):
        lookup=g.set_index('method')
        if 'original' not in lookup.index:
            continue
        for method in lookup.index:
            if method=='original':
                continue
            for metric in ['normalized_rmse_L0','nlpd','coverage_90','coverage_95','mean_interval_width_90','standardized_error_rms']:
                pairs.append({'data_seed':seed,'representation':rep,'role':role,'calibration':cal,'method':method,
                              'reference':'original','metric':metric,'paired_difference':float(lookup.loc[method,metric]-lookup.loc['original',metric])})
    pd.DataFrame(pairs).to_csv(output/'phase1_paired_seed_differences.csv',index=False)
    gate=[]
    for (rep,method),g in fitdf.loc[fitdf.method!='frozen'].groupby(['representation','method']):
        scores=seedmeans.loc[(seedmeans.representation==rep)&(seedmeans.method==method)&(seedmeans.role=='check')&(seedmeans.calibration=='raw')]
        frozen=seedmeans.loc[(seedmeans.representation=='qwen')&(seedmeans.method=='frozen')&(seedmeans.role=='check')&(seedmeans.calibration=='raw')].set_index('data_seed')
        expected={(seed,budget,endpoint,init) for seed in (157,887) for budget in (333,813) for endpoint in ('V1','V2') for init in (17,29)}
        observed={(int(r.data_seed),int(r.budget_state),r.endpoint,int(r.initialization)) for r in g.itertuples()}
        matrix_complete=observed==expected and len(g)==len(expected)
        seed_complete=set(scores.data_seed)=={157,887}
        success_rate=float(g.optimizer_success.sum()/len(expected))
        gradient_rate=float(g.gradient_qualified.sum()/len(expected))
        qualified=gradient_rate>=.9
        coverage=bool((scores.coverage_90>=.8).all())
        rms=bool((scores.standardized_error_rms<=1.5).all())
        nlpd=bool(all(r.nlpd<=frozen.loc[r.data_seed,'nlpd'] for r in scores.itertuples()))
        gate.append({'representation':rep,'method':method,'finite_fits':len(g),'planned_fits':len(expected),'matrix_complete':matrix_complete,'seed_complete':seed_complete,
                     'optimizer_success_rate':success_rate,'gradient_qualified_rate':gradient_rate,'coverage_gate':coverage,'standardized_rms_gate':rms,
                     'nlpd_vs_frozen_qwen_gate':nlpd,'pass':bool(matrix_complete and seed_complete and qualified and coverage and rms and nlpd),
                     'evidence':'development screening; fusion has representation leakage on internal check'})
    dump(output/'phase1_gates.json',gate)
    dump(output/'phase1_cost.json',{'finite_fits':len(fitdf),'failed_fits':len(failures),'fit_seconds':float(fitdf.training_seconds.sum()),
                                  'total_fit_plus_diagnostics_seconds':float(fitdf.total_seconds.sum()),'all_failed_preserved':True})
    # Plot data are the complete equally weighted seed/endpoint/budget/init means.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    methods=['frozen','original','converged','regularized','noise_floor','staged']
    for ri,rep in enumerate(['qwen','fusion']):
        for ci,(metric,title) in enumerate([('normalized_rmse_L0','Normalized RMSE'),('nlpd','NLPD (mL)'),('coverage_90','90% coverage')]):
            for mode,marker in [('raw','o'),('temperature','s')]:
                g=aggregate.loc[(aggregate.representation==rep)&(aggregate.role=='check')&(aggregate.calibration==mode)].set_index('method')
                axes[ri,ci].plot(range(len(methods)),[g.loc[m,metric] if m in g.index else np.nan for m in methods],marker=marker,label=mode)
            axes[ri,ci].set_xticks(range(len(methods)),['Frozen','A40','B200','Ridge','Noise floor','Staged'],rotation=35,ha='right');axes[ri,ci].set_title(rep+': '+title)
            axes[ri,ci].grid(alpha=.2)
            if metric=='nlpd':axes[ri,ci].set_yscale('log')
            if metric=='coverage_90':axes[ri,ci].axhline(.9,color='gray',ls='--');axes[ri,ci].set_ylim(0,1.03)
    axes[0,0].legend();fig.suptitle('Molecular-group GP holdout (GNN/Fusion: conditional on historically supervised features)')
    fig.savefig(output/'phase1_diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps({'finite':len(fits),'failures':failures,'gates':gate},indent=2))

if __name__=='__main__':
    main()

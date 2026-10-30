"""Analysis-only diagnostics from frozen validation metrics; never imports a runner.

Default: check 370 fit_audit records and five matched full-reference records,
then write only this study directory. --snapshot-only reproduces the numerical
curve analysis from versioned snapshots and explicitly skips runtime provenance.
Thresholds below were fixed before computing threshold results and are
exploratory descriptive levels, not success criteria or tuned strategy targets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CURVES = Path('reports/free_llm_five_seeds_to1485_20261024/figures/validation_curves.json')
SNAPSHOT_SUMMARY = CURVES.parent / 'current_summary.json'
AUDIT = Path('reports/free_lcmd_five_seeds_to1485_20261026/final_audit.json')
REFERENCE = Path('reports/free_lcmd_five_seeds_to1485_20261026/analysis/full_reference_validation_comparison_20261028.json')
BASE = Path('studies/active_learning/qgeognn_v2_row_sequential_b32')
EXT = Path('studies/active_learning/qgeognn_v2_row_lcmd_to1485_20261026')
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = 'free_llm32_scientist_v2'
METHODS = (FREE, 'lcmd')
NAMES = {FREE: 'Free LLM32', 'lcmd': 'LCMD32'}
STAGES = ((333,525),(525,653),(653,813),(813,1005),(1005,1197),(1197,1485),
          (333,1485),(653,1005),(333,653),(1005,1485))
PRIMARY_STAGES = STAGES[:7]
THRESHOLDS = (0.50,0.45,0.40,0.35,0.30)
EXPECTED_LABELS = list(range(333,1486,32))
EPS = 1e-12


def read_json(path: Path, sources: dict[str,str]) -> dict:
    path = path if path.is_absolute() else ROOT / path
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def stat(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {'n':len(arr), 'mean':float(arr.mean()) if len(arr) else None,
            'sample_sd':float(arr.std(ddof=1)) if len(arr)>1 else None,
            'sample_variance':float(arr.var(ddof=1)) if len(arr)>1 else None,
            'min':float(arr.min()) if len(arr) else None,
            'max':float(arr.max()) if len(arr) else None}


def normalized_aulc(labels: Sequence[int], errors: Sequence[float], start: int, stop: int) -> float:
    x, y = np.asarray(labels), np.asarray(errors,dtype=float)
    if len(x)!=len(y) or len(set(x))!=len(x) or np.any(np.diff(x)<=0):
        raise ValueError('budgets must be unique, ordered and align with errors')
    if start>=stop or start not in x or stop not in x or not np.isfinite(y).all():
        raise ValueError('finite values and exact observed interval endpoints are required')
    mask=(x>=start)&(x<=stop)
    return float(np.trapezoid(y[mask],x[mask])/(stop-start))


def stage_metrics(labels, errors, start, stop, full) -> dict:
    x,y=np.asarray(labels),np.asarray(errors,dtype=float)
    keep=(x>=start)&(x<=stop)
    xs,ys=x[keep],y[keep]
    if len(xs)<2 or xs[0]!=start or xs[-1]!=stop or np.any(np.diff(xs)!=32):
        raise ValueError('stage must contain every frozen batch32 fit')
    steps=ys[:-1]-ys[1:]
    gain=float(ys[0]-ys[-1]); gap=float(ys[0]-full)
    return {'start_labels':start,'stop_labels':stop,'n_transitions':len(steps),
            'start_nrmse':float(ys[0]),'end_nrmse':float(ys[-1]),
            'normalized_aulc':normalized_aulc(labels,errors,start,stop),
            'endpoint_gain':gain,'gain_per_32_labels':gain/len(steps),
            'relative_gain_from_stage_start':gain/float(ys[0]),
            'worsening_steps':int((steps < -EPS).sum()),
            'worsening_fraction':float((steps < -EPS).mean()),
            'total_worsening_magnitude':float(-np.minimum(steps,0).sum()),
            'total_variation':float(np.abs(steps).sum()),
            'step_gain_sample_sd':float(steps.std(ddof=1)) if len(steps)>1 else None,
            'worst_one_step_worsening':float(max(0,-steps.min())),
            'largest_one_step_gain':float(steps.max()),
            'full_reference_nrmse':full,'start_gap_to_full_reference':gap,
            'end_gap_to_full_reference':float(ys[-1]-full),
            'fraction_of_start_gap_to_full_closed':gain/gap if gap>EPS else None}


def threshold_hit(labels, errors, threshold) -> dict:
    hits=[int(x) for x,y in zip(labels,errors) if y<=threshold]
    sustained=next((int(labels[i]) for i in range(len(labels))
                    if all(y<=threshold for y in errors[i:])),None)
    return {'threshold_nrmse':threshold,'first_observed_labels':hits[0] if hits else None,
            'censored':not bool(hits),'censor_labels':int(labels[-1]) if not hits else None,
            'sustained_from_labels':sustained,
            'sustained_censored':sustained is None,
            'sustained_definition':'first observed fit from which every remaining frozen fit through L1485 is at or below threshold; retrospective, final point alone can qualify',
            'definition':'first frozen fit at or below threshold; no interpolation; may be transient'}


def crossings(labels, free, lcmd) -> dict:
    delta=np.asarray(free)-np.asarray(lcmd)
    signs=np.sign(np.where(np.abs(delta)<=EPS,0,delta))
    nonzero=[i for i,s in enumerate(signs) if s]
    events=[]
    for left,right in zip(nonzero[:-1],nonzero[1:]):
        if signs[left]!=signs[right]:
            interp=float(labels[left]+(labels[right]-labels[left])*(-delta[left])/(delta[right]-delta[left]))
            events.append({'left_observed_labels':int(labels[left]),'right_observed_labels':int(labels[right]),
                           'free_minus_lcmd_left':float(delta[left]),'free_minus_lcmd_right':float(delta[right]),
                           'new_better_method':FREE if delta[right]<0 else 'lcmd',
                           'linear_interpolated_equality_labels_descriptive_only':interp})
    first=next((int(labels[i]) for i,d in enumerate(delta) if d < -EPS),None)
    persistent=next((int(labels[i]) for i in range(len(labels)) if np.all(delta[i:] < -EPS)),None)
    return {'crossing_count_excluding_equal_initialization':len(events),'events':events,
            'first_observed_free_better_labels':first,
            'free_better_at_every_remaining_observed_fit_from':persistent,
            'free_better_fits':int((delta < -EPS).sum()),'lcmd_better_fits':int((delta > EPS).sum()),
            'equal_fits':int((np.abs(delta)<=EPS).sum()),
            'final_winner':FREE if delta[-1]<-EPS else 'lcmd' if delta[-1]>EPS else 'tie'}


def stratum(seed):
    return 'historical_Astra_high_prefix' if seed in (157,6101) else 'Sol_xhigh_throughout'


def local_path(recorded):
    path=Path(recorded)
    if path.exists(): return path
    parts=path.parts
    return ROOT / Path(*parts[parts.index('studies'):])


def audit_inputs(curves, frozen, sources, snapshot_only=False):
    old=read_json(REFERENCE,sources)
    references={int(row['seed']):float(row['full3330']) for row in old['rows']}
    audit={'mode':'snapshot_only' if snapshot_only else 'frozen_runtime_metadata',
           'verified_curve_fit_records':0,'verified_full_reference_fit_records':0,
           'normalization':'sqrt(0.5*((RMSE_V1/sV1)^2+(RMSE_V2/sV2)^2)); '
                           'sV1/sV2 = seed-specific original L0 population std, ddof=0, held fixed',
           'aulc_normalization':'trapezoidal integral over observed NRMSE / (stop-start); no error or gap rescaling',
           'target_scales':{},'validation_rows':416,'full_reference_training_rows':3330,
           'opened_truth_or_prediction_tables':False,'notes':[]}
    for seed in SEEDS:
        if len(curves[str(seed)][FREE])!=37 or len(curves[str(seed)]['lcmd'])!=37:
            raise ValueError('all ten complete 37-fit trajectories required')
        for method in METHODS:
            rows=curves[str(seed)][method]
            if [p['labels'] for p in rows]!=EXPECTED_LABELS or not all(math.isfinite(p['nrmse']) for p in rows):
                raise ValueError('snapshot must have exact 333..1485 batch32 budgets')
        if abs(curves[str(seed)][FREE][0]['nrmse']-curves[str(seed)]['lcmd'][0]['nrmse'])>EPS:
            raise ValueError('paired initial errors differ')
        if snapshot_only: continue
        context=read_json(BASE/f'runtime/seed_{seed}/context.json',sources)['contract']
        prep=context['preprocessing']
        if prep['target_scale_fit_role']!='l0_labels_only' or prep['target_scale_fit_rows']!=333:
            raise ValueError('unexpected frozen target-scale source')
        audit['target_scales'][str(seed)]=prep['target_scales']
        ownership=frozen['verification'][str(seed)]['fit_ownership']
        initial=read_json(BASE/f'runtime/seed_{seed}/lcmd/round_00/model/fit_audit.json',sources)
        for method in METHODS:
            for r,row in enumerate(curves[str(seed)][method]):
                if method==FREE:
                    fr=read_json(Path(ownership[r]['freeze']),sources)
                    if fr['test_truth_access_count']!=0: raise ValueError('unexpected test access receipt')
                    fitpath=local_path(fr['checkpoint_path']).parent/'fit_audit.json'
                else:
                    fitpath=(BASE if r<=21 else EXT)/f'runtime/seed_{seed}/lcmd/round_{r:02d}/model/fit_audit.json'
                fit=read_json(fitpath,sources)
                if fit['train_rows']!=row['labels'] or fit['validation_rows']!=416:
                    raise ValueError('fit budget/validation mismatch')
                for key in ('initialization_hash','validation_ids_hash'):
                    if fit[key]!=initial[key]: raise ValueError('unmatched paired fit state')
                if fit['test_labels_used_for_fit_or_checkpoint_selection']!=0:
                    raise ValueError('unexpected test use')
                if abs(row['nrmse']-fit['best_validation_combined_normalized_rmse'])>EPS:
                    raise ValueError(f'frozen curve metric mismatch: {fitpath}')
                audit['verified_curve_fit_records']+=1
        full_dir=BASE/f'runtime/seed_{seed}/full_data_reference'
        full=read_json(full_dir/'model/fit_audit.json',sources)
        freeze=read_json(full_dir/'pre_test_freeze.json',sources)
        if full['train_rows']!=3330 or full['validation_rows']!=416 or freeze['test_truth_access_count']!=0:
            raise ValueError('full-reference fit contract mismatch')
        for key in ('initialization_hash','validation_ids_hash'):
            if full[key]!=initial[key]: raise ValueError('full reference not paired')
        path=str(full_dir/'model/fit_audit.json')
        if sources[path]!=freeze['files']['model/fit_audit.json']:
            raise ValueError('full-reference audit hash mismatch')
        if abs(full['best_validation_combined_normalized_rmse']-references[seed])>EPS:
            raise ValueError('full-reference snapshot mismatch')
        audit['verified_full_reference_fit_records']+=1
    if snapshot_only: audit['notes'].append('Runtime fit consistency, fixed scales and matched reference provenance not rechecked in snapshot-only mode.')
    audit['notes'].append('Checkpoint/batch content hashes are independently checked by root provenance audit; this script checks metrics and metadata only.')
    return references,audit


def csv_write(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def paired_stage_summary(rows,seeds):
    output={}
    keys=('start_nrmse','end_nrmse','normalized_aulc','endpoint_gain','gain_per_32_labels',
          'relative_gain_from_stage_start','worsening_fraction','total_worsening_magnitude',
          'total_variation','fraction_of_start_gap_to_full_closed')
    for start,stop in STAGES:
        stage=[r for r in rows if r['seed'] in seeds and r['start_labels']==start and r['stop_labels']==stop]
        pair={m:{r['seed']:r for r in stage if r['method']==m} for m in METHODS}
        output[f'{start}-{stop}']={m:{k:stat([pair[m][s][k] for s in seeds if pair[m][s][k] is not None]) for k in keys} for m in METHODS}
        output[f'{start}-{stop}']['paired_free_minus_lcmd']={k:stat([pair[FREE][s][k]-pair['lcmd'][s][k] for s in seeds if pair[FREE][s][k] is not None and pair['lcmd'][s][k] is not None]) for k in keys}
        output[f'{start}-{stop}']['free_lower_aulc_seed_count']=sum(pair[FREE][s]['normalized_aulc']<pair['lcmd'][s]['normalized_aulc'] for s in seeds)
        output[f'{start}-{stop}']['free_larger_gain_seed_count']=sum(pair[FREE][s]['endpoint_gain']>pair['lcmd'][s]['endpoint_gain'] for s in seeds)
    return output


def compute(curves,references):
    stages=[];rounds=[];thresholds=[];best=[];cross={};closure=[]
    for seed in SEEDS:
        ys={m:np.array([r['nrmse'] for r in curves[str(seed)][m]]) for m in METHODS}
        labels=EXPECTED_LABELS
        for method in METHODS:
            y=ys[method]; initial=float(y[0]);full=references[seed]
            for start,stop in STAGES:
                result={'seed':seed,'method':method,'model_provenance_stratum':stratum(seed),
                        'stage':f'{start}-{stop}',**stage_metrics(labels,y,start,stop,full)}
                result['fraction_of_L0_to_full_gap_closed_at_end']=(initial-result['end_nrmse'])/(initial-full)
                stages.append(result)
            for r,value in enumerate(y):
                rounds.append({'seed':seed,'method':method,'round':r,'labels':labels[r],
                    'model_provenance_stratum':stratum(seed),'validation_nrmse':float(value),
                    'previous_fit_labels':labels[r-1] if r else None,
                    'adjacent_gain_positive_is_improvement':float(y[r-1]-value) if r else None,
                    'free_minus_lcmd_at_same_labels':float(ys[FREE][r]-ys['lcmd'][r]),
                    'full_reference_nrmse':full,'remaining_gap_to_full_reference':float(value-full),
                    'gain_from_L0':initial-float(value),
                    'fraction_of_L0_to_full_gap_closed':(initial-float(value))/(initial-full)})
            for threshold in THRESHOLDS:
                thresholds.append({'seed':seed,'method':method,**threshold_hit(labels,y,threshold)})
            i=int(y.argmin());best.append({'seed':seed,'method':method,'best_validation_nrmse':float(y[i]),
                'first_labels_at_best':labels[i],'round_at_best':i,'final_validation_nrmse':float(y[-1]),
                'final_minus_best':float(y[-1]-y[i]),'full_reference_nrmse':full,
                'best_minus_full_reference':float(y[i]-full)})
        cross[str(seed)]=crossings(labels,ys[FREE],ys['lcmd'])
        i,j=labels.index(653),labels.index(1005)
        fg=float(ys[FREE][i]-ys[FREE][j]);lg=float(ys['lcmd'][i]-ys['lcmd'][j])
        start=float(ys[FREE][i]-ys['lcmd'][i]);end=float(ys[FREE][j]-ys['lcmd'][j])
        closure.append({'seed':seed,'model_provenance_stratum':stratum(seed),
            'free_start':float(ys[FREE][i]),'lcmd_start':float(ys['lcmd'][i]),
            'free_end':float(ys[FREE][j]),'lcmd_end':float(ys['lcmd'][j]),
            'start_free_minus_lcmd':start,'end_free_minus_lcmd':end,
            'free_gain':fg,'lcmd_gain':lg,'extra_free_gain':fg-lg,
            'gap_closure_start_minus_end':start-end,'identity_residual':(fg-lg)-(start-end),
            'start_disadvantage_fraction_of_extra_gain':start/(fg-lg) if abs(fg-lg)>EPS else None,
            'free_relative_gain':fg/ys[FREE][i],'lcmd_relative_gain':lg/ys['lcmd'][i],
            'free_start_gap_to_full':float(ys[FREE][i]-references[seed]),
            'lcmd_start_gap_to_full':float(ys['lcmd'][i]-references[seed]),
            'free_fraction_start_full_gap_closed':fg/(ys[FREE][i]-references[seed]),
            'lcmd_fraction_start_full_gap_closed':lg/(ys['lcmd'][i]-references[seed])})
    lookup={(r['seed'],r['method'],r['stage']):r for r in stages}
    for row in stages:
        for key in ('start_nrmse','end_nrmse','normalized_aulc','endpoint_gain'):
            row['paired_free_minus_lcmd_'+key]=lookup[(row['seed'],FREE,row['stage'])][key]-lookup[(row['seed'],'lcmd',row['stage'])][key]
    summary={'validation_only':True,'paired_unit':'seed; five paired development trajectories; no independent-round inference',
        'thresholds_fixed_before_threshold_computation':list(THRESHOLDS),
        'threshold_interpretation':'descriptive arbitrary levels, first observed attainment, right-censored at L1485; no interpolation or optimization',
        'stages':paired_stage_summary(stages,SEEDS),
        'model_provenance_sensitivity':{name:{'seeds':list(seeds),'interpretation':'nonrandom seed strata; model-version effect is not identified',
           'stages':paired_stage_summary(stages,seeds)} for name,seeds in (
               ('historical_Astra_high_prefix',(157,6101)),('Sol_xhigh_throughout',(887,2357,12203)))},
        'crossings_by_seed':cross,'threshold_attainment':thresholds,'best_validation_points':best,
        '653_to1005_gap_closure_by_seed':closure,
        '653_to1005_gap_closure_means':{key:stat([r[key] for r in closure]) for key in closure[0] if key not in ('seed','model_provenance_stratum')},
        'full_reference_by_seed':references,'full_reference_summary':stat(list(references.values())),
        'full_reference_caveat':'Matched full-training fit is a descriptive reference, not a noise-free lower bound; negative gaps are possible.',
        'inference_limit':'No causal mechanism, significance, future generalization or strategy complementarity claim follows from curve shape.'}
    return rounds,stages,summary


def plot(curves,stages,summary,out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={FREE:'#A63B83','lcmd':'#148B88'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8.5))
    for ax,seed in zip(axes.flat,SEEDS):
        for m in METHODS:
            ax.plot(EXPECTED_LABELS,[r['nrmse'] for r in curves[str(seed)][m]],color=colors[m],label=NAMES[m],lw=1.8)
        ax.axhline(summary['full_reference_by_seed'][seed],color='#34445b',ls='--',lw=1,label='Full3330 reference')
        ax.set_title(f'Seed {seed}'+(' | historical prefix' if seed in (157,6101) else ' | Sol/xhigh'))
        ax.set(xlabel='Active training labels',ylabel='Validation NRMSE');ax.grid(alpha=.15)
    ax=axes.flat[-1]
    for m in METHODS:
        y=np.array([[r['nrmse'] for r in curves[str(seed)][m]] for seed in SEEDS])
        ax.plot(EXPECTED_LABELS,y.mean(0),color=colors[m],label=NAMES[m],lw=2)
        ax.fill_between(EXPECTED_LABELS,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=colors[m],alpha=.10)
    ax.axhline(summary['full_reference_summary']['mean'],color='#34445b',ls='--',lw=1,label='Mean full3330 reference')
    ax.set(title='Five seeds: mean and sample SD',xlabel='Active training labels',ylabel='Validation NRMSE');ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Frozen validation learning curves | 5 paired seeds, 37 fits per trajectory')
    fig.tight_layout();fig.savefig(out/'validation_learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,key,title in zip(axes,('normalized_aulc','endpoint_gain'),('Paired stage AULC: Free minus LCMD','Paired stage gain: Free minus LCMD')):
        for j,(start,stop) in enumerate(PRIMARY_STAGES):
            rows=[r for r in stages if r['method']==FREE and r['start_labels']==start and r['stop_labels']==stop]
            vals=[r['paired_free_minus_lcmd_'+key] for r in rows]
            for i,v in enumerate(vals):
                ax.scatter(j+(i-2)*.06,v,s=28,color=plt.get_cmap('tab10')(i),
                           label=str(SEEDS[i]) if j==0 else None)
            ax.plot([j-.22,j+.22],[np.mean(vals)]*2,color='black',lw=2)
        ax.axhline(0,color='#555',lw=.8);ax.grid(axis='y',alpha=.15)
        ax.set_xticks(range(len(PRIMARY_STAGES)),[f'{a}\n{b}' for a,b in PRIMARY_STAGES]);ax.set(title=title,xlabel='Stage labels',ylabel='Validation NRMSE units')
    axes[0].legend(title='Paired seed',frameon=False,fontsize=8,ncol=2)
    fig.suptitle('Five seed units; points = paired seeds, black bars = seed mean; descriptive')
    fig.tight_layout();fig.savefig(out/'paired_stage_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    c=summary['653_to1005_gap_closure_by_seed']
    for row in c:
        axes[0].plot([0,1],[row['start_free_minus_lcmd'],row['end_free_minus_lcmd']],marker='o',label=str(row['seed']))
        axes[1].plot([0,1],[row['free_fraction_start_full_gap_closed'],row['lcmd_fraction_start_full_gap_closed']],marker='o',label=str(row['seed']))
    axes[0].axhline(0,color='#555',lw=.8);axes[1].axhline(1,color='#555',lw=.8,ls='--')
    axes[0].set_xticks([0,1],['L653','L1005']);axes[1].set_xticks([0,1],['Free LLM32','LCMD32'])
    axes[0].set(title='Same-budget error gap closes',ylabel='Validation NRMSE: Free minus LCMD')
    axes[1].set(title='Fraction of L653 gap to full3330 closed',ylabel='(error653 - error1005) / (error653 - full)')
    axes[0].legend(title='Paired seed',frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('L653–1005 | 5 paired validation seeds | full fit is a reference, not a lower bound')
    fig.tight_layout();fig.savefig(out/'midstage_headroom_diagnostics.png',dpi=180);plt.close(fig)


def findings(summary,out):
    st=summary['stages'];cl=summary['653_to1005_gap_closure_means'];rows=[]
    for a,b in STAGES:
        v=st[f'{a}-{b}'];f=v[FREE];l=v['lcmd'];d=v['paired_free_minus_lcmd']
        rows.append(f"| {a}–{b} | {f['start_nrmse']['mean']:.6f}→{f['end_nrmse']['mean']:.6f} | {l['start_nrmse']['mean']:.6f}→{l['end_nrmse']['mean']:.6f} | {f['normalized_aulc']['mean']:.6f} | {l['normalized_aulc']['mean']:.6f} | {d['normalized_aulc']['mean']:+.6f} ± {d['normalized_aulc']['sample_sd']:.6f} | {f['endpoint_gain']['mean']:.6f} / {l['endpoint_gain']['mean']:.6f} |")
    fluct=[]
    for a,b in PRIMARY_STAGES:
        v=st[f'{a}-{b}'];f=v[FREE];l=v['lcmd']
        fluct.append(f"| {a}–{b} | {f['worsening_fraction']['mean']:.1%} / {l['worsening_fraction']['mean']:.1%} | {f['total_worsening_magnitude']['mean']:.6f} / {l['total_worsening_magnitude']['mean']:.6f} | {f['gain_per_32_labels']['mean']:.6f} / {l['gain_per_32_labels']['mean']:.6f} |")
    cr=[]
    for seed,c in summary['crossings_by_seed'].items():
        brackets=', '.join(f"{e['left_observed_labels']}–{e['right_observed_labels']}" for e in c['events'])
        cr.append(f"| {seed} | {c['crossing_count_excluding_equal_initialization']} | {c['first_observed_free_better_labels']} | {c['free_better_at_every_remaining_observed_fit_from']} | {brackets} |")
    closure=[]
    for r in summary['653_to1005_gap_closure_by_seed']:
        closure.append(f"| {r['seed']} | {r['start_free_minus_lcmd']:+.6f} | {r['end_free_minus_lcmd']:+.6f} | {r['free_gain']:.6f} / {r['lcmd_gain']:.6f} | {r['free_relative_gain']:.1%} / {r['lcmd_relative_gain']:.1%} | {r['free_fraction_start_full_gap_closed']:.1%} / {r['lcmd_fraction_start_full_gap_closed']:.1%} |")
    bp=[]
    for r in summary['best_validation_points']:
        bp.append(f"| {r['seed']} | {NAMES[r['method']]} | {r['best_validation_nrmse']:.6f} | {r['first_labels_at_best']} | {r['final_minus_best']:.6f} |")
    sens=[]
    for name,s in summary['model_provenance_sensitivity'].items():
        for key in ('333-1485','653-1005','1005-1485'):
            v=s['stages'][key]
            sens.append(f"| {name} ({len(s['seeds'])} seeds) | {key} | {v['paired_free_minus_lcmd']['normalized_aulc']['mean']:+.6f} | {v['paired_free_minus_lcmd']['endpoint_gain']['mean']:+.6f} | {v['paired_free_minus_lcmd']['end_nrmse']['mean']:+.6f} |")
    late=st['1005-1485'];f0=st['333-1485'][FREE];l0=st['333-1485']['lcmd']
    txt=f'''# 学习曲线独立诊断（Task A）

本分析仅读取冻结验证指标和 fit 元数据，未训练、未调用 LLM、未读取预测或真值表。默认运行复核了 {summary['input_audit']['verified_curve_fit_records']} 个曲线 fit_audit 和 {summary['input_audit']['verified_full_reference_fit_records']} 个全训练参考 fit_audit；checkpoint/选批哈希由总审计独立核对。固定 seed 为 157、887、2357、6101、12203。配对统计单位为 seed；不把 180 轮当独立样本，不报告显著性。

## 定义与证据边界

NRMSE 沿用 `evaluation/point.py`：`sqrt(0.5*((RMSE_V1/sV1)^2+(RMSE_V2/sV2)^2))`。`runner.py::fit_al_preprocessing` 在各 seed 的原始 L0=333 标签上以 float32 计算目标 population std（ddof=0），固定用于整条轨迹。绝不以 validation、后续标签或全量标签重新定标。AULC=实际冻结 NRMSE 对标签数的梯形积分除以预算跨度，只是该段平均误差。不同 seed 的尺度本来不同；配对比较与原报告完全一致。

阈值在计算命中结果前固定为 `{list(THRESHOLDS)}`，仅为描述性水平，不是预注册成功标准。仅报告首个已观测拟合预算 `error<=threshold`；未达为 L1485 右删失。不插值阈值预算；首次命中可能短暂，不能当作持续达到。交叉点输出相邻观测预算区间；JSON 另附线性插值交点，仅作描述，不声称真实预算精度。初始相等不算交叉。非单调性以相邻 fit 的误差上升次数、累计幅度、总变差与相邻改善 SD 表示，包含训练随机性和选样效应，不能单独归因于选样。

补充持续阈值字段 `sustained_from_labels`：从该观测预算起到 L1485 所有冻结点均不超过阈值；不存在则为 NA 且 `sustained_censored=true`。这是事后描述，若只有末点达到，该末点也可作为持续起点，不代表已证实后续稳定性。

全训练参考是同 seed/初始化/416 验证划分的既有 3330 标签拟合（平均 {summary['full_reference_summary']['mean']:.6f}），并非无噪声下界。`fraction_of_start_gap_to_full_closed=(error_start-error_stop)/(error_start-full_reference)`；分母非正时记缺失。该描述和原始 NRMSE/AULC 的定义相互独立，不替换其归一化。

## 全程与阶段

Free 终点均值 {f0['end_nrmse']['mean']:.6f}，LCMD {l0['end_nrmse']['mean']:.6f}；全程 AULC 分别 {f0['normalized_aulc']['mean']:.6f}、{l0['normalized_aulc']['mean']:.6f}。终点低误差与全程累计标签效率给出不同排序。各阶段绝对误差、阶段改善和 AULC 必须分别解释。

| 标签段 | Free 首尾 NRMSE | LCMD 首尾 NRMSE | Free AULC | LCMD AULC | 配对 AULC 差 F−L（均值±样本SD） | 首尾改善 Free / LCMD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

`stage_comparison.csv` 每 seed、方法、阶段给出原始全部指标和配对差；`curve_summary.json` 给出各方法 seed 间样本方差及配对差样本方差。六个主阶段是互不重叠的增量区间（共享端点但无重复增量）；其加权 AULC 与 333–1485 全程一致。333–653、653–1005、1005–1485 为补充聚合窗。

| 标签段 | 非单调上升轮比例 F / L（seed均值） | 上升累计幅度 F / L（seed均值） | 每32标签净改善 F / L（seed均值） |
| --- | ---: | ---: | ---: |
{chr(10).join(fluct)}

## 653–1005 的改善空间复核

所有五 seed 的 Free 在 L653 均落后，且该段首尾改善均比 LCMD 大。均值：起点差 F−L={cl['start_free_minus_lcmd']['mean']:+.6f}，终点差={cl['end_free_minus_lcmd']['mean']:+.6f}；额外 Free 改善={cl['extra_free_gain']['mean']:.6f}。恒等式 `gain_Free-gain_LCMD = (Free-LCMD)_653-(Free-LCMD)_1005` 在各 seed 残差近零。因此均值层面的额外改善中，{cl['start_free_minus_lcmd']['mean']/cl['extra_free_gain']['mean']:.1%} 数值上用于抹平起点劣势；这是算术分解，不能解释为因果份额。

但整个653–1005阶段的平均 AULC 仍为 LCMD 较低（Free {st['653-1005'][FREE]['normalized_aulc']['mean']:.6f}、LCMD {st['653-1005']['lcmd']['normalized_aulc']['mean']:.6f}），只有 {st['653-1005']['free_lower_aulc_seed_count']}/5 seed 的 Free 阶段 AULC 更低。813–1005 平均 AULC 也近乎持平。因此“中期反转”须限定为该段净改善和段末误差，不能等同于整个中期累计标签效率更优。

| seed | L653 起点差 F−L | L1005 终点差 F−L | 绝对改善 F / L | 相对各自起点改善 F / L | 各自 L653 至全量参考余量关闭 F / L |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(closure)}

相对改善和全量参考余量关闭仍显示 Free 该段取得更多改善；这排除了“只比较绝对下降量”这一表述问题，却不能控制不同训练集、模型状态、轨迹历史、版本及随机性的混杂。不能据此证明局部补线、反馈修复或覆盖饱和的机制。尤其 LCMD 接近参考时该比例对小分母敏感，full 也不是下界。

1005–1485 的首尾平均改善为 Free {late[FREE]['endpoint_gain']['mean']:.6f}、LCMD {late['lcmd']['endpoint_gain']['mean']:.6f}，数值上小于前段但未宣称平台成立。终点至 full 参考平均余量：Free {f0['end_nrmse']['mean']-summary['full_reference_summary']['mean']:.6f}、LCMD {l0['end_nrmse']['mean']-summary['full_reference_summary']['mean']:.6f}。部分 seed/拟合点低于 full 参考，证明不能把 full 当严格最优界。

## 交叉与最佳点

| seed | 交叉次数 | 首个 Free 更低观测预算 | 此后全部拟合 Free 更低的起始预算 | 交叉观测区间 |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(cr)}

交叉重复且 seed 不同步；固定 L653 或 L1005 并非普适切换依据。只凭独立轨迹的交叉也不能证明同状态切换会获益。

| seed | 方法 | 最佳 validation NRMSE | 首个最佳预算 | 最终减最佳误差 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(bp)}

最佳点只是已看见的验证轨迹最小值，有反复评估选择偏差，不视为最终泛化误差。阈值完整逐 seed 结果见 `threshold_attainment.csv`；删失预算不当作实际达到预算均值。

## 模型版本敏感性

157/6101 为保留历史 Astra/high 前缀的混合轨迹；887/2357/12203 为 Sol/xhigh 轨迹。以下按 seed 来源分层，不能把组差当作版本因果效应；前缀、seed/划分与轨迹历史不能分离。没有把历史组后半段重新标作“纯 Sol”，因训练集带有历史选择影响。

| 来源分层 | 标签段 | 配对 AULC差 F−L | 配对改善差 F−L | 段末误差差 F−L |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(sens)}

两个分层都保留 LCMD 全程 AULC 较低、Free 在653–1005改善较多的方向；小样本分层不适合进一步显著性推断。需要同 seed、同模型版本、同冻结状态分支或预注册消融才能区分机制与版本影响。

## 复现

`python studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/learning_curve_analysis.py`

只写本新目录。缺少 gitignored runtime 时可加 `--snapshot-only` 从版本化冻结曲线和历史全训练参考快照复现数值，届时输出明确跳过 runtime 来源复核。脚本没有训练/selector/label store 导入，也不读取 test、候选标签或预测表。输入文件 SHA256 全列于 `curve_summary.json::source_sha256`。轻量测试覆盖不齐预算拒绝、非单调/删失/交叉规则、阶段加权恒等式和已冻结报告数值一致性。
'''
    (out/'CURVE_FINDINGS.md').write_text(txt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot-only',action='store_true');args=p.parse_args()
    sources={};curves=read_json(CURVES,sources);frozen=read_json(SNAPSHOT_SUMMARY,sources)
    for source in ('src/qgeognn_al/active_learning_v2/runner.py',
                   'src/qgeognn_al/evaluation/point.py',
                   'src/qgeognn_al/active_learning_v2/sequential_protocol.py'):
        sources[source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    published=read_json(AUDIT,sources)
    refs,audit=audit_inputs(curves,frozen,sources,args.snapshot_only)
    rounds,stages,summary=compute(curves,refs)
    summary['input_audit']=audit
    for m in METHODS:
        check=published['common_budget_comparison']['normalized_aulc_333_to1485'][m]['mean']
        if abs(summary['stages']['333-1485'][m]['normalized_aulc']['mean']-check)>EPS:
            raise ValueError('independent AULC disagrees with frozen final audit')
    summary['source_sha256']=sources
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/'round_metrics.csv',rounds);csv_write(OUT/'stage_comparison.csv',stages)
    csv_write(OUT/'threshold_attainment.csv',summary['threshold_attainment'])
    csv_write(OUT/'best_validation_points.csv',summary['best_validation_points'])
    csv_write(OUT/'midstage_headroom.csv',summary['653_to1005_gap_closure_by_seed'])
    (OUT/'curve_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    findings(summary,OUT);plot(curves,stages,summary,OUT)
    print(json.dumps({'curve_fit_metadata_checked':audit['verified_curve_fit_records'],
                      'full_reference_metadata_checked':audit['verified_full_reference_fit_records'],
                      'stage_rows':len(stages),'round_rows':len(rounds),
                      '653_to1005_gap_closure':summary['653_to1005_gap_closure_means']['extra_free_gain']},indent=2))


if __name__=='__main__':main()

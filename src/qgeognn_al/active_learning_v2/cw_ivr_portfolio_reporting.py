"""Post-global-freeze development metrics, component regret and diagnostics."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA
from ..training.predictor import atomic_json
from .benchmark_reporting import metric_row
from .protocol import RestrictedLabelStore
from .cw_ivr_portfolio_study import (STUDY, METHOD, COMPARATORS, SEEDS, ACTIVE_LABEL_BUDGETS,
    TOLERANCE, comparator_sources, read_json, validate_prepared)

METRIC='combined_normalized_RMSE'
WINDOWS=((333,653),(333,525),(429,653),(525,653),(333,429),(429,525))


def summarize(curves):
    rows=[]
    for seed in SEEDS:
        for method in (METHOD,*COMPARATORS):
            arm=curves.loc[curves.outer_seed.eq(seed)&curves.method.eq(method)].sort_values('active_label_count')
            if arm.active_label_count.tolist()!=list(ACTIVE_LABEL_BUDGETS):
                raise ValueError('complete unique eleven-point matched grid required')
            row={'outer_seed':seed,'method':method}
            for start,stop in WINDOWS:
                subset=arm.loc[arm.active_label_count.between(start,stop)]
                x=subset.active_label_count.to_numpy(float)
                y=subset[METRIC].to_numpy(float)
                row[f'AULC_{start}_{stop}']=float(np.sum((y[1:]+y[:-1])*np.diff(x)/2)/(stop-start))
            for key in (METRIC,'V1_RMSE','V2_RMSE','V1_R2','V2_R2'):
                row[key+'_653']=float(arm.iloc[-1][key])
            rows.append(row)
    summary=pd.DataFrame(rows)
    regret=[]
    for seed in SEEDS:
        s=summary.loc[summary.outer_seed.eq(seed)].set_index('method')
        oracle=min(s.loc['center_width_lcmd','AULC_333_653'],s.loc['kernel_ivr','AULC_333_653'])
        for method in (METHOD,'center_width_lcmd','kernel_ivr'):
            regret.append({'outer_seed':seed,'method':method,'oracle_AULC':oracle,
                           'regret':float(s.loc[method,'AULC_333_653']-oracle)})
    regret=pd.DataFrame(regret)
    regret_summary=regret.groupby('method').regret.agg(mean_regret='mean',max_seed_regret='max').reset_index()
    means=summary.groupby('method').mean(numeric_only=True).drop(columns='outer_seed').reset_index()
    return summary,means,regret,regret_summary


def decide(summary,means,regrets):
    means=means.set_index('method')
    regrets=regrets.set_index('method')
    components=['center_width_lcmd','kernel_ivr']
    deltas=[]
    for seed in SEEDS:
        s=summary.loc[summary.outer_seed.eq(seed)].set_index('method')
        deltas.append({'outer_seed':seed,
            'AULC_delta_to_oracle':float(s.loc[METHOD,'AULC_333_653']-s.loc[components,'AULC_333_653'].min()),
            'AULC_delta_to_worse':float(s.loc[METHOD,'AULC_333_653']-s.loc[components,'AULC_333_653'].max()),
            'endpoint_delta_to_oracle':float(s.loc[METHOD,METRIC+'_653']-s.loc[components,METRIC+'_653'].min())})
    delta=float(means.loc[METHOD,'AULC_333_653']-means.loc[components,'AULC_333_653'].min())
    stable=all(d['AULC_delta_to_oracle']<=TOLERANCE and d['endpoint_delta_to_oracle']<=TOLERANCE for d in deltas)
    gains={m:float(regrets.loc[m,'max_seed_regret']-regrets.loc[METHOD,'max_seed_regret']) for m in components}
    if all(d['AULC_delta_to_worse']>TOLERANCE for d in deltas):
        decision='STOP'
    elif delta<=0 and stable and min(gains.values())>0:
        decision='STRONG_PORTFOLIO_SIGNAL'
    elif delta<=TOLERANCE and stable and min(gains.values())>=TOLERANCE:
        decision='PROMISING_PORTFOLIO'
    else:
        decision='NO_CLEAR_PORTFOLIO_GAIN'
    return {'decision':decision,'evidence':'development / exposed cohort; not independent confirmation',
            'mean_AULC_delta_to_best_component':delta,'paired_deltas':deltas,
            'stable_within_frozen_tolerance':stable,'worst_regret_improvements':gains,
            'tolerance':TOLERANCE,'automatic_followup':False}


def report(study=STUDY):
    from .cw_ivr_portfolio_runner import finalize_pre_test, PortfolioContext
    study=Path(study)
    decision_path = study/'decision.json'
    completed=read_json(decision_path) if decision_path.exists() else {}
    result_paths = {
        name: study/'results'/f'{name}.csv'
        for name in (
            'learning_curve_metrics', 'summary_per_seed', 'summary_mean',
            'component_regret', 'regret_summary', 'acquisition_diagnostics',
            'selection_trace', 'fit_audit', 'test_label_access_audit',
        )
    }
    if completed.get('status')=='COMPLETE' and (study/'artifact_manifest.json').exists():
        for name,digest in read_json(study/'artifact_manifest.json')['files'].items():
            if sha256_file(ROOT/name)!=digest:
                raise RuntimeError('completed report artifact changed')
        return completed
    if completed.get('status') == 'COMPLETE' and all(path.exists() for path in result_paths.values()):
        # Reporting may be resumed after the metrics and decision were written.
        # Reuse those frozen post-test tables so recovery never opens the label
        # store or increments the post-freeze access count.
        tables = {name: pd.read_csv(path) for name, path in result_paths.items()}
        _write_report(
            study,
            tables['summary_per_seed'],
            tables['summary_mean'],
            tables['regret_summary'],
            tables['acquisition_diagnostics'],
            completed,
        )
        files=[p for p in study.rglob('*') if p.is_file() and 'runtime' not in p.parts and p.name!='artifact_manifest.json']
        atomic_json(study/'artifact_manifest.json',{'files':{str(p.relative_to(ROOT)):sha256_file(p) for p in files},
                                                  'runtime_global_freeze':str(study/'global_pre_test_freeze.json')})
        return completed
    # Verify every nested artifact and both trajectories BEFORE constructing a label store.
    freeze=finalize_pre_test(study)
    if len(freeze['entries'])!=22:
        raise RuntimeError('global test barrier incomplete')
    rows=[]
    accesses=[]
    for seed in SEEDS:
        context=PortfolioContext(seed,study)
        base=context.runtime/METHOD
        final=pd.read_csv(base/'round_10/state.csv')
        selected=final.loc[final.role.eq('labeled'),'sample_id'].astype(str).tolist()[333:]
        store=RestrictedLabelStore(SOURCE_DATA,context.partition)
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        test_ids=context.ids(context.roles['test'])
        truth=store.reveal(test_ids,'final_test_evaluation')
        accesses.extend({'outer_seed':seed,**r} for r in store.audit)
        for r,n in enumerate(ACTIVE_LABEL_BUDGETS):
            record=read_json(base/f'round_{r:02d}/round_freeze.json')
            pred=pd.read_csv(record['prediction_path'])
            if pred.sample_id.astype(str).tolist()!=test_ids:
                raise RuntimeError('test-X prediction identity drift')
            rows.append({'outer_seed':seed,'method':METHOD,'round':r,'active_label_count':n,
                **metric_row(truth,pred.drop(columns='sample_id').to_numpy(float),context.preprocessing['target_scales'])})
    curves=pd.DataFrame(rows)
    for method,path in comparator_sources().items():
        table=pd.read_csv(path)
        curves=pd.concat([curves,table.loc[table.method.eq(method)&table.outer_seed.isin(SEEDS)&table.active_label_count.isin(ACTIVE_LABEL_BUDGETS)]],ignore_index=True)
    summary,means,regret,regret_summary=summarize(curves)
    decision=decide(summary,means,regret_summary)
    results=study/'results'
    results.mkdir(exist_ok=True)
    diagnostics=[]
    traces=[]
    fits=[]
    elapsed=0.
    for seed in SEEDS:
        base=study/f'runtime/seed_{seed}'
        elapsed+=(datetime.fromisoformat(read_json(base/'execution_completed.json')['completed_at'])-datetime.fromisoformat(read_json(base/'execution_started.json')['started_at'])).total_seconds()
        fits.append(pd.read_csv(base/METHOD/'fit_audit.csv'))
        for r in range(10):
            a=base/METHOD/f'round_{r:02d}/acquisition_artifacts'
            d=read_json(a/'diagnostics.json')
            c=read_json(a/'contract.json')
            trace=pd.read_csv(a/'selection_trace.csv')
            traces.append(trace)
            diagnostics.append({'outer_seed':seed,'round':r+1,'active_labels_before':333+32*r,
                **{k:v for k,v in d.items() if not isinstance(v,list)},
                'gradient_seconds':c['gradient_extraction_seconds'],'selector_seconds':c['selector_seconds'],
                **{f'{e}_marginal_variance_reduction':float(trace.loc[trace.expert.eq(e),'selected_marginal_variance_reduction'].sum()) for e in ('cw','ivr')},
                **{f'{e}_selected_nearest_distance_mean':float(np.sqrt(trace.loc[trace.expert.eq(e),'nearest_center_sq_distance_before_selection']).mean()) for e in ('cw','ivr')}})
    diagnostics=pd.DataFrame(diagnostics)
    fits=pd.concat(fits,ignore_index=True)
    tables={'learning_curve_metrics':curves,'summary_per_seed':summary,'summary_mean':means,
            'component_regret':regret,'regret_summary':regret_summary,'acquisition_diagnostics':diagnostics,
            'selection_trace':pd.concat(traces,ignore_index=True),'fit_audit':fits,
            'test_label_access_audit':pd.DataFrame(accesses)}
    for name,table in tables.items():
        table.to_csv(results/f'{name}.csv',index=False)
    decision.update(status='COMPLETE',new_fits=int((fits.reuse_status=='new_fit').sum()),
                    reused_anchor_fits=2,new_anchor_fits=0,execution_wall_seconds=elapsed,
                    training_seconds=float(fits.loc[fits.reuse_status.eq('new_fit'),'training_seconds'].sum()),
                    gradient_seconds=float(diagnostics.gradient_seconds.sum()),
                    selector_seconds=float(diagnostics.selector_seconds.sum()),
                    test_truth_access_count_before_global_freeze=0,post_freeze_test_accesses=len(accesses))
    atomic_json(study/'decision.json',decision)
    _plots(study,curves,regret)
    _write_report(study,summary,means,regret_summary,diagnostics,decision)
    files=[p for p in study.rglob('*') if p.is_file() and 'runtime' not in p.parts and p.name!='artifact_manifest.json']
    atomic_json(study/'artifact_manifest.json',{'files':{str(p.relative_to(ROOT)):sha256_file(p) for p in files},
                                              'runtime_global_freeze':str(study/'global_pre_test_freeze.json')})
    return decision


def _plots(study,curves,regret):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=study/'figures'
    figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
    for ax,seed in zip(axes,(*SEEDS,'mean')):
        for m in (METHOD,*COMPARATORS):
            s=curves.loc[curves.method.eq(m)]
            if seed!='mean':
                s=s.loc[s.outer_seed.eq(seed)]
            s=s.groupby('active_label_count')[METRIC].mean()
            ax.plot(s.index,s.values,marker='o',markersize=3,label=m,linewidth=2.5 if m==METHOD else 1.2,
                    alpha=1 if m in (METHOD,'center_width_lcmd','kernel_ivr') else .55)
        ax.set(title=str(seed),xlabel='Active labels')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Combined NRMSE (lower is better)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Balanced shared-context CW + IVR | development / exposed cohort')
    fig.tight_layout()
    fig.savefig(figures/'learning_curves.png',dpi=180)
    plt.close(fig)
    pivot=regret.pivot(index='outer_seed',columns='method',values='regret')
    ax=pivot.plot.bar(figsize=(8,5))
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_ylabel('AULC regret to per-seed component oracle')
    ax.figure.tight_layout()
    ax.figure.savefig(figures/'component_regret.png',dpi=180)
    plt.close(ax.figure)


def _write_report(study,summary,means,regret,diagnostics,decision):
    def table(frame):
        def value(item):
            if isinstance(item, (float, np.floating)):
                return f'{float(item):.6f}'
            return str(item)
        header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
        separator = '| ' + ' | '.join('---' for _ in frame.columns) + ' |'
        rows = [
            '| ' + ' | '.join(value(item) for item in row) + ' |'
            for row in frame.itertuples(index=False, name=None)
        ]
        return '\n'.join([header, separator, *rows])
    component=means.set_index('method').loc[['center_width_lcmd','kernel_ivr'],'AULC_333_653']
    best=component.idxmin()
    portfolio=means.set_index('method').loc[METHOD]
    delta=portfolio-means.set_index('method').loc[best]
    stages={f'{a}-{b}':float(delta[f'AULC_{a}_{b}']) for a,b in ((333,429),(429,525),(525,653))}
    overlap=float(diagnostics.independent_proposal_overlap_count.mean())
    text=f'''# Balanced shared-context CW + IVR portfolio

Decision: **{decision['decision']}**. Development / exposed cohort; not independent confirmation.
No p-values or generalization/significance claims are made from two exposed seeds.

## Definition and provenance

At every current checkpoint, one full-network q50 derivative pass generates both
historical CountSketch-512 geometries. CW uses the frozen L333 Center/Width scales;
IVR uses endpoint scales then all-reference RMS, unit prior and unit noise.
Exactly 32 alternating steps begin with CW. Every pick becomes a CW center and
an IVR observation location through a rank-one covariance update. No labels enter
selection. The whole batch is frozen before reveal; the next predictor is trained
from the identical historical scratch initialization, never warm-started.
L333 checkpoints and test-X predictions are reused for both seeds. L333 gradients
are re-extracted once in shared multi-transform form, because endpoint-only
compressed sketches cannot reconstruct CW. `engineering_smoke.json` records
numerical agreement and unchanged historical pure-selector L333 batch IDs.

New fits: {decision['new_fits']}; new L333 fits: 0; reused anchor checkpoints: 2.
Execution wall time (summed sequential seed windows): {decision['execution_wall_seconds']:.1f}s;
training {decision['training_seconds']:.1f}s; all gradient extraction including smoke
{decision['gradient_seconds']:.1f}s; selectors/diagnostics {decision['selector_seconds']:.1f}s.
Smoke precedes the execution wall window, so these timing quantities differ.
Both complete trajectories and 22 checkpoint/test-X prediction points were
recursively frozen before {decision['post_freeze_test_accesses']} test accesses
(one per seed); pre-global-freeze access count was zero. Source hashes revalidated.

## Per-seed metrics

{table(summary)}

## Two-seed development mean

{table(means)}

## Component oracle regret

Regret is AULC(method) minus min(AULC(CW),AULC(IVR)) for the same seed.
Negative values are permitted; no clipping. Historical CW wins both seeds on
333-653, so its oracle regret is zero. This is a strict hedging benchmark,
not evidence that pure CW wins all future states.

{table(regret)}

## Answers to the scientific questions

1. Mean AULC 333-653 is {portfolio['AULC_333_653']:.6f}; delta to best component mean
({best}) is {decision['mean_AULC_delta_to_best_component']:+.6f}. The frozen closeness
tolerance is 0.01, approximately 1.5% of the historical scale.
2. Stability within the frozen per-seed AULC and endpoint oracle tolerance:
{decision['stable_within_frozen_tolerance']}. Exact per-seed deltas:
{decision['paired_deltas']}. No seed was dropped based on performance.
3. Worst-regret reductions versus fixed components (positive is improvement):
{decision['worst_regret_improvements']}. The decision uses both comparisons.
4. AULC 333-525={portfolio['AULC_333_525']:.6f}, 429-653={portfolio['AULC_429_653']:.6f},
525-653={portfolio['AULC_525_653']:.6f}. The tables include every seed and comparator.
5. Disjoint early/middle/late-middle AULC deltas to the best full-trajectory
component are {stages}. Negative identifies the stage with predictive gain;
overlapping requested windows must not be summed as independent contributions.
6. Shared conditioning prevents duplicates by construction and updates both
objectives after every pick. Independent CW16/IVR16 proposals overlap by
{overlap:.2f} rows on average. Before/after geometry and union diagnostics appear
below. Duplicate prevention does not by itself establish optimal redundancy
reduction: independent unions have fewer locations when overlap is nonzero,
and no equal-budget no-conditioning ablation was trained.
7. Expert-specific nearest-center distances, marginal variance contributions and
same-state proposal agreement quantify differing selection profiles below.
Both experts supply 16 rows per batch, but this alone does not establish two
disjoint chemical regions; geometric complementarity is only descriptive.
8. Predictive gain must be assessed by the frozen metrics/regrets above, not
selection diversity. Acquisition differences do not automatically imply gains.

## Acquisition mechanism diagnostics, all 20 rounds

{table(diagnostics)}

Full 640-step expert trace, IDs, canonical indices, checkpoint/bank hashes,
cluster scores and variance updates are in `results/selection_trace.csv`.
Per-round label-free banks and immutable contracts are under `runtime/`.

## Interpretation and next step

'''
    if decision['decision'] in ('STRONG_PORTFOLIO_SIGNAL','PROMISING_PORTFOLIO'):
        text+='''The balanced portfolio supports further investigation. The smallest next step
is a separately preregistered replication of fixed 16/16 versus both components
on an unexposed cohort. Adaptive quota q_t in [0,32] remains a proposed later
study, using gradient norm std/p95, coverage p90, relative reducible variance and
effective rank as candidate inputs. This two-seed study supplies no validated
quota policy; no adaptive fits or quota sweep were performed.
'''
    else:
        text+='''The frozen decision does not justify proceeding directly to adaptive quota.
(1) Independent overlap and contribution profiles test geometric complementarity,
but cannot establish universal predictive complementarity. (2) Regression and
explicit covariance checks preserve both mathematical objectives; conditioning
changes their greedy path intentionally, not their definition. (3) A 16/16
bottleneck cannot be identified from a single fixed quota; no quota sweep is
warranted by these data alone. (4) CW path dependence remains plausible because
the portfolio changes the trajectory from the first batch. (5) Matched scratch
initialization controls one source of variation but does not estimate optimization
variance; repeated predictor fits would be needed. (6) Use the paired metrics to
distinguish observed selection diversity from actual predictive improvement.
There is no controlled evidence here that allocation is the principal bottleneck.
A future experiment, if pursued, should first isolate shared-conditioning and
path effects with an equal-budget preregistered comparison; it is not run here.
'''
    text+='''
## Artifact index

- `PROTOCOL.md`, `protocol.json`, `seal.json`, `reuse_audit.json`: frozen design and provenance.
- `preflight_tests.xml`, `engineering_smoke.json`: tests and selection-only regression.
- `global_pre_test_freeze.json`: both complete trajectories bound by content hashes.
- `results/learning_curve_metrics.csv`: 11-point learning curves for all six methods.
- `results/summary_per_seed.csv`, `results/summary_mean.csv`: all AULC windows and endpoint metrics.
- `results/component_regret.csv`, `results/regret_summary.csv`: per-seed/mean/worst regret.
- `results/acquisition_diagnostics.csv`, `results/selection_trace.csv`: round and step audits.
- `results/fit_audit.csv`, `results/test_label_access_audit.csv`: cost and firewall audits.
- `figures/learning_curves.png`, `figures/component_regret.png`: scientific figures.
- `runtime/seed_{157,6101}/cw_ivr_portfolio/round_{00..10}/`: states, selections,
  gradient banks, model checkpoints, predictions and nested freezes (local, git-ignored).
- `decision.json`, `artifact_manifest.json`: decision and committed artifact hashes.
'''
    (study/'FINAL_REPORT.md').write_text(text)

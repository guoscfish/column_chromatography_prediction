#!/usr/bin/env python
"""Summarize existing bounded-study artifacts; never fit or call a provider."""
from pathlib import Path
import json
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_factorial'
def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(df):
    names=list(df.columns)
    def value(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(names)+' |\n| '+' | '.join(['---']*len(names))+' |\n'+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
state=read(S/'RUN_STATE.json'); ledger=read(S/'fit_ledger.json') if (S/'fit_ledger.json').exists() else {}
curves=pd.read_csv(S/'validation_curves.csv'); summary=pd.read_csv(S/'validation_summary.csv')
decisions=[]; effects=[]
for selector in ('lcmd','maxdet','ivr'):
    block=summary[summary.arm.isin(['raw_'+selector,'cw_'+selector])]
    pivot=block.pivot(index='seed',columns='arm',values='AULC333_525')
    if set(pivot.columns)!={'raw_'+selector,'cw_'+selector} or pivot.isna().any().any() or len(pivot)!=2:
        decisions.append({'selector':selector,'status':'incomplete','eligible':False});continue
    raw=pivot['raw_'+selector];cw=pivot['cw_'+selector]
    improvement=float(1-cw.mean()/raw.mean()); both=bool((cw<raw).all())
    decisions.append({'selector':selector,'status':'completed','raw_mean_AULC':float(raw.mean()),'cw_mean_AULC':float(cw.mean()),'relative_mean_improvement':improvement,'both_seeds_improve':both,'eligible':bool(improvement>=.02 and both)})
    for seed in pivot.index:
        effects.append({'seed':seed,'selector':selector,'delta_AULC':float(cw[seed]-raw[seed]),'relative_improvement':float(1-cw[seed]/raw[seed])})
complete=len(summary)==16
new_gates=[d for d in decisions if d['selector'] in ('maxdet','ivr')]
eligibility='pending_incomplete_matrix' if not complete else ('cross_selector_candidate' if all(d['eligible'] for d in new_gates) else 'selector_specific_candidate' if any(d['eligible'] for d in new_gates) else 'no_new_selector_candidate')
white=[]
for seed in (157,6101):
    g=summary[summary.seed.eq(seed)].set_index('arm')
    if {'whitening_lcmd','cw_lcmd'}<=set(g.index):white.append({'seed':seed,'whitening_minus_CW_AULC':float(g.loc['whitening_lcmd','AULC333_525']-g.loc['cw_lcmd','AULC333_525'])})
white_mean=None if len(white)!=2 else float(np.mean([x['whitening_minus_CW_AULC'] for x in white]))
geometry_claim='pending whitening trajectories' if white_mean is None else 'output geometry correction; no CW-semantic exclusivity' if white_mean<=0 else 'CW exceeds whitening mean here; semantic exclusivity still not established'
scientific={'status':'completed' if complete else 'running','primary_metric':'validation normalized AULC333-525','eligibility':eligibility,'gates':decisions,'whitening_vs_CW':white,'interpretation':geometry_claim,'new_test_evaluations':0,'historical_test_is_blind':False}
(S/'decision.json').write_text(json.dumps(scientific,indent=2)+'\n')
resources=[]
for k,v in ledger.items():
    seed,arm,r=k.split('/')
    row={'seed':int(seed),'arm':arm,'round':int(r),'status':v['status'],'training_seconds':v.get('training_seconds',0)}
    p=S/f'runtime/seed_{seed}/{arm}/round_{int(r)-1:02d}/selection_diagnostics.json'
    if p.exists():row['acquisition_seconds']=read(p)['seconds']
    resources.append(row)
res=pd.DataFrame(resources);res.to_csv(S/'resource_costs.csv',index=False)
finished=sum(v['status']=='completed' for v in ledger.values())
endpoints=curves[curves.budget.eq(525)][['seed','arm','combined_normalized_RMSE','V1_RMSE','V2_RMSE','V1_MAE','V2_MAE','V1_R2','V2_R2']]
endpoints.to_csv(S/'endpoint_metrics.csv',index=False)
if effects:pd.DataFrame(effects).to_csv(S/'paired_effects.csv',index=False)
text=f'''# CW geometry and LLM feedback results

Status: **{'completed' if complete else 'running'}**. P0 numerical/compatibility audit and P2 offline boundary implementation completed. Formal fits completed: {finished}/36; reserved attempts: {len(ledger)}/36. Paid LLM calls: 0. New test evaluations: 0.

## Frozen primary result

Development seeds157/6101; budgets333,365,397,429,461,493,525. Normalized trapezoidal validation AULC333–525 is the primary metric. Only complete seven-point curves enter the table. No endpoint-based or test-based method selection.

{table(summary)}

![Validation curves](validation_curves.png)

## Paired effects and continuation gate

Negative delta means CW improves on Raw within the same selector. Mean relative improvement is computed as 1−mean(CW AULC)/mean(Raw AULC), with an additional requirement that both seed deltas are negative. The 2% threshold is an engineering resource allocation rule, not significance.

{table(pd.DataFrame(effects)) if effects else 'No complete paired comparisons yet.'}

Decision: **{eligibility}**. Full machine-readable gates: decision.json. Whitening interpretation: **{geometry_claim}**.

Cross-selector comparisons remain developmental interactions with two seeds. MaxDet and IVR retain different RMS policies. A favorable CW result under one selector does not establish universality. Finite CountSketch orientation contributes to the realized representation; see MECHANISM_AUDIT.md.

## Secondary endpoint metrics

{table(endpoints)}

## Evidence and limitations

70 historical control points were reused after byte hashes, numerical training configuration, normalization, preprocessing, initialization, L/U ordering and trajectory checks. Source paths and hashes are retained in REUSE_MANIFEST.json and reuse_compatibility.json. Historical test exposure is disclosed; none of these data constitute a new blind test. This report evaluates validation from saved checkpoints only.

P0: real-model uncompressed isometry, projection grid and exact historical Raw/CW first-batch replay; existing MaxDet logdet and IVR covariance oracle tests. Supplemental eight-row selection checks are not multi-round experiments. P2: synthetic adversarial fixtures plus sixteen real historical input adaptations; the live prompt adapter and positive paid budget remain future work. Historical Full prompt/source and all ongoing trajectory artifacts are untouched. LLM_HISTORY_AUDIT.json records completed budgets and P2_ABLATION_PLAN.md records quota/manual-correction limitations.

## Cost and recovery

Completed training wall-seconds (sum over fits): {sum(v.get('training_seconds',0) for v in ledger.values()):.1f}. Acquisition wall-seconds: {sum(r.get('acquisition_seconds',0) for r in resources):.1f}. Training uses two CPU threads per fit; this is wall time, not metered CPU/GPU cost. See resource_costs.csv. Large checkpoints remain local under ignored runtime. No dollar charge was incurred by LLM calls in this study.

```sh
# Verify/continue the exact bounded P1 trajectory only if it is incomplete:
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py run
# Recompute validation from frozen checkpoint artifacts, then this report:
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py report
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/report_cw_geometry_feedback.py
# Offline boundary checks (no provider):
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/active_learning_v2/test_cw_factorial.py tests/active_learning_v2/test_feedback_ablation_adapter.py
```

An interrupted fit reservation is not silently retried or removed: account for its attempt within the36 cap before any recovery. No automatic expansion beyond525, no test reveal command, and no live LLM command are enabled here. Future confirmation requires one frozen candidate/strong controls and five audited unused split seeds; extensions to813/1005 require a later preregistered budget. New splits are robustness checks, not external validation. The current round stops at the registered boundary.
'''
(S/'RESULTS.md').write_text(text)
# Keep stage table synchronized without claiming incomplete training is complete.
plan=ROOT/'docs/research/CW_GEOMETRY_AND_LLM_FEEDBACK_PLAN.md';lines=plan.read_text().splitlines()
for i,line in enumerate(lines):
    for stage in ('P1a','P1b'):
        if line.startswith('| '+stage+' |'):
            actual=state.get(stage,{}).get('status','planned')
            lines[i]=f'| {stage} | {actual} | {finished}/36 total completed fits; see RUN_STATE.json and RESULTS.md |'
plan.write_text('\n'.join(lines)+'\n')
print(json.dumps({'complete_curves':len(summary),'completed_fits':finished,'eligibility':eligibility}))

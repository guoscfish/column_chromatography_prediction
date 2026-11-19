"""Report only verified completed fits; missing trajectories stay incomplete."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/epig_v2_matplotlib')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (157,887,2357,6101,12203)
ARMS = ('posterior_epig_conditional_greedy32','posterior_ivr_conditional_greedy32')
BUDGETS = np.arange(333,1006,32)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)


def aulc(frame,lo,hi):
    part=frame.loc[frame.budget.between(lo,hi)].sort_values('budget')
    wanted=np.arange(lo,hi+1,32)
    if not np.array_equal(part.budget.to_numpy(),wanted):
        return None
    return float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(part.combined_nrmse.to_numpy(),part.budget.to_numpy())/(hi-lo))


def report(study=HERE,verify_files=True):
    study=Path(study);rows=[];input_hashes={};archive_hashes={}
    def archive(source,relative):
        """Export immutable small receipts outside ignored training runtime."""
        target=study/'receipts'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if target.exists() and target.read_bytes()!=data:
            raise RuntimeError(f'archived immutable scientific receipt changed: {target}')
        if not target.exists():target.write_bytes(data)
        archive_hashes[str(target.relative_to(study))]=hashlib.sha256(data).hexdigest()
    for pattern in ('seed_*/*/round_*/selection*.json','seed_*/*/round_*/state.json',
                    'seed_*/*/round_*/label_access_attempt_*.json'):
        for source in sorted((study/'runtime').glob(pattern)):
            archive(source,source.relative_to(study/'runtime'))
    for p in sorted((study/'runtime').glob('seed_*/*/round_*/fit_receipt.json')):
        r=read(p);seed=int(r['seed']);arm=r['arm'];round_index=int(r['round'])
        if seed not in SEEDS or arm not in ARMS or r['budget'] != 333+32*round_index:
            raise RuntimeError(f'fit receipt identity drift: {p}')
        if r['validation']['evaluation_role'] != 'validation_checkpoint_selection':
            raise RuntimeError('invalid evaluation role')
        model=ROOT/r['model_directory']
        if verify_files:
            files=r.get('files',{}) or {'best.pt':r['checkpoint_sha256']}
            for name,expected in files.items():
                if sha(model/name) != expected:raise RuntimeError(f'fit artifact drift: {model/name}')
        input_hashes[str(p.relative_to(study))]=sha(p)
        archive(p,p.relative_to(study/'runtime'))
        archive(model/'fit_audit.json',p.parent.relative_to(study/'runtime')/'model_fit_audit.json')
        rows.append({'seed':seed,'arm':arm,'round':round_index,'budget':r['budget'],
            **r['validation'],'training_seconds':float(r.get('training_seconds',0)),
            'reused_L333':bool(r['reused']),'checkpoint_sha256':r['checkpoint_sha256'],
            'fit_receipt':str(p.relative_to(study))})
    cols=['seed','arm','round','budget','combined_nrmse','V1_RMSE','V2_RMSE','V1_MAE','V2_MAE',
          'best_epoch','epochs_run','evaluation_role','training_seconds','reused_L333','checkpoint_sha256','fit_receipt']
    curves=pd.DataFrame(rows,columns=cols).sort_values(['seed','arm','round'])
    if curves.duplicated(['seed','arm','round']).any():raise RuntimeError('duplicate seed/arm/budget')
    curves.to_csv(study/'validation_curves.csv',index=False)
    (study/'per_seed').mkdir(exist_ok=True)
    for seed in SEEDS:
        curves.loc[curves.seed==seed].to_csv(study/f'per_seed/seed_{seed}_curves.csv',index=False)
    metrics=[];pairs=[]
    for seed in SEEDS:
        for arm in ARMS:
            f=curves.loc[(curves.seed==seed)&(curves.arm==arm)].sort_values('budget')
            complete=np.array_equal(f.budget.to_numpy(),BUDGETS)
            rec={'seed':seed,'arm':arm,'complete_L1005':complete,'completed_downstream_fits':max(0,len(f)-1),
                 'last_completed_budget':int(f.budget.max()) if len(f) else None,
                 'training_seconds':float(f.training_seconds.sum())}
            for lo,hi,name in ((333,1005,'full'),(333,525,'early'),(525,813,'middle'),(813,1005,'late')):
                rec['AULC_'+name]=aulc(f,lo,hi)
            for budget in (525,813,1005):
                at=f.loc[f.budget==budget]
                rec[f'NRMSE_L{budget}']=float(at.combined_nrmse.iloc[0]) if len(at) else None
            metrics.append(rec)
        e,v=metrics[-2:]
        pair={'seed':seed,'complete_pair_L1005':bool(e['complete_L1005'] and v['complete_L1005'])}
        for name in ('AULC_full','AULC_early','AULC_middle','AULC_late','NRMSE_L525','NRMSE_L813','NRMSE_L1005'):
            pair['EPIG_minus_IVR_'+name]=(e[name]-v[name]) if e[name] is not None and v[name] is not None else None
        pairs.append(pair)
    pd.DataFrame(metrics).to_csv(study/'seed_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(study/'paired_differences.csv',index=False)
    complete=all(p['complete_pair_L1005'] for p in pairs)
    success=int((~curves.reused_L333).sum()) if len(curves) else 0
    ledger_path=study/'runtime/compute_budget.json'
    ledger=read(ledger_path) if ledger_path.exists() else {}
    dump(study/'compute_ledger_snapshot.json',ledger)
    attempts=ledger.get('attempts',[])
    costs={'successful_downstream_fits_verified':success,'target_successful_fits':210,
           'qgeognn_attempts':sum(a['kind']=='qgeognn' for a in attempts),
           'failures':[a for a in attempts if a['status'] not in ('RUNNING','COMPLETED')],
           'active_seconds_completed_and_failed':ledger.get('active_seconds',{}),
           'running_operations':[a for a in attempts if a['status']=='RUNNING'],
           'offline_reservations':ledger.get('external_reservations',{}),
           'actual_offline_costs':{}}
    for name,path in [('surrogate','surrogate_evidence/complete.json'),('surrogate_followup','surrogate_evidence/followup/complete.json'),('reference','reference/status.json')]:
        p=study/path
        if p.exists():costs['actual_offline_costs'][name]=read(p)
    dump(study/'compute_cost.json',costs)
    summary={'status':'COMPLETE_L1005' if complete else 'INCOMPLETE_RUNNING_OR_PRESERVED',
        'successful_fits':success,'target_successful_fits':210,'complete_seed_pairs':sum(p['complete_pair_L1005'] for p in pairs),
        'primary_efficacy_evaluated':complete,'original_five_seed_gate_satisfied':False,
        'test_labels_evaluated':0,'study_type':'post-stage1/stage2 exploratory development study',
        'receipt_hashes':input_hashes,'generated_unix':time.time()}
    summary['archived_scientific_receipt_hashes']=archive_hashes
    if complete:
        diff=np.asarray([p['EPIG_minus_IVR_AULC_full'] for p in pairs],float)
        summary['paired_AULC_mean']=float(diff.mean());summary['paired_AULC_sample_SD']=float(diff.std(ddof=1))
        summary['paired_AULC_all_seeds']=diff.tolist()
        summary['five_overlapping_outer_seeds_not_independent_confirmations']=True
    dump(study/'reporting_manifest.json',summary)
    lines=['# Actual Row-level learning results','',
      'This is a post-stage1/stage2 exploratory development study. The original five-seed probability gate remains unsatisfied. All numbers below come from completed fit receipts, with checkpoint hashes verified. No test targets are evaluated.','',
      f'Completed downstream QGeoGNN fits: **{success}/210**. Complete L333–L1005 seed pairs: **{summary["complete_seed_pairs"]}/5**.','']
    if not complete:
        lines += ['**Primary five-seed efficacy is not yet evaluated.** Missing budgets are not interpolated and partial curves are not substitutes for the required learning window. No EPIG failure or success verdict follows from this snapshot.','']
    else:
        lines += [f'Paired EPIG−IVR full-window AULC mean: {summary["paired_AULC_mean"]:.8f}; seed sample SD: {summary["paired_AULC_sample_SD"]:.8f}. Lower is better. Splits overlap; these are descriptive paired effects.','']
    lines += ['| Seed | Arm | Completed fits | Last budget | Full AULC | L1005 NRMSE |',
              '|---|---|---:|---:|---:|---:|']
    for m in metrics:
        fmt=lambda x:'pending' if x is None else f'{x:.7f}'
        lines.append(f'| {m["seed"]} | {"EPIG" if m["arm"]==ARMS[0] else "IVR"} | {m["completed_downstream_fits"]}/21 | {m["last_completed_budget"] or "pending"} | {fmt(m["AULC_full"])} | {fmt(m["NRMSE_L1005"])} |')
    lines += ['', 'Files: `validation_curves.csv` (including physical V1/V2 MAE/RMSE in ml), `seed_metrics.csv` (early/middle/late and milestone metrics), `paired_differences.csv`, `compute_cost.json`, `reporting_manifest.json`. Curves are original nonmonotone checkpoint metrics. Full AULC is only defined for all22 expected budgets. Initial checkpoints are reused, never counted as new fits.','',
      'The two arms use identical predefined GP modeling rules. Once selections diverge their training sets and posteriors differ. Offline common-state comparisons isolate the objective; trajectory comparisons measure the entire acquisition-and-retraining procedure. Historical methods remain in the original baseline inventory and are not silently pooled as protocol-identical controls.','',
      'Resume/run: `/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/run_exploratory.py run`. Refresh this report: same Python with `report_study.py`. Do not change sealed decision files or delete partial attempts.']
    (study/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    decision=['# Next-stage decision','',
        'The old Stage1/2 gates and conclusions remain unchanged. This v2 study is exploratory. Numerical validity, probability calibration, proxy MI and real Row AULC are distinct evidence levels.','']
    if not complete:
        decision += [f'**Continue the frozen complete primary pair.** The verified snapshot contains {success}/210 downstream fits and {summary["complete_seed_pairs"]}/5 complete seed pairs. No primary efficacy verdict, failure-mechanism verdict or Free-LLM deployment recommendation is supported yet. Early loss, calibration failure or overlap cannot stop this study. Resource/protocol failures must retain artifacts and exact reasons.','']
    else:
        early=np.mean([p['EPIG_minus_IVR_AULC_early'] for p in pairs])
        late=np.mean([p['EPIG_minus_IVR_AULC_late'] for p in pairs])
        terminal=np.mean([p['EPIG_minus_IVR_NRMSE_L1005'] for p in pairs])
        signal=summary['paired_AULC_mean']<0 or terminal<0 or late<0
        decision += [f'Complete five-seed paired effects (EPIG−IVR; lower better): full AULC {summary["paired_AULC_mean"]:.8f}, early AULC {early:.8f}, late AULC {late:.8f}, terminal NRMSE {terminal:.8f}. The overlapping five splits support a bounded descriptive comparison, not independent confirmation.','',
            ('At least one predefined aggregate signal favors EPIG. Freeze a separate finite-budget protocol and extend both methods together to L1485; this runner does not silently begin an unsealed extension.' if signal else 'No predefined aggregate advantage is present in this model/distribution/L1005 window. Inspect the full seed/budget heterogeneity before a new paired extension decision. This does not establish long-run EPIG failure.'),'',
            'These curves isolate the acquisition objective under one fixed surrogate rule and the original QGeoGNN protocol. They do not identify probability quality, batch redundancy or predictor limitations as a sole cause. Free LLM numeric assistance needs separately verified marginal value and cost; proxy MI alone does not authorize a deployment conclusion.','']
    decision += ['See `RESULTS.md`, `MECHANISM_ANALYSIS.md`, `LIMITATIONS.md` and the immutable protocol for scope. No paid LLM API was called.']
    (study/'NEXT_STAGE_DECISION.md').write_text('\n'.join(decision)+'\n')
    if len(curves):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
        for ax,seed in zip(axes.flat,SEEDS):
            for arm,color,label in zip(ARMS,('#c94e34','#277da1'),('EPIG','IVR')):
                f=curves.loc[(curves.seed==seed)&(curves.arm==arm)]
                if len(f):ax.plot(f.budget,f.combined_nrmse,'o-',ms=3,color=color,label=label)
            ax.set_title(f'Seed {seed}');ax.set_xlim(333,1005);ax.grid(alpha=.2)
            ax.set_xlabel('Active row labels');ax.set_ylabel('Validation combined NRMSE')
        axes.flat[-1].axis('off');axes.flat[0].legend()
        fig.suptitle(f'Actual nonmonotone QGeoGNN curves — {success}/210 fits; '+('complete' if complete else 'incomplete, no efficacy verdict'))
        fig.tight_layout();(study/'figures').mkdir(exist_ok=True)
        fig.savefig(study/'figures/validation_learning_curves.png',dpi=160);plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=HERE);args=p.parse_args()
    (args.study/'runtime').mkdir(exist_ok=True)
    with (args.study/'runtime/report.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'ANOTHER_REPORT_WRITER_ACTIVE'}));raise SystemExit(0)
        result=report(args.study);print(json.dumps({k:v for k,v in result.items() if k not in {'receipt_hashes','archived_scientific_receipt_hashes'}},indent=2))

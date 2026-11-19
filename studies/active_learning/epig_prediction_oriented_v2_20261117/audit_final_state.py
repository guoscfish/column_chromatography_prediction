"""Artifact-only final audit; never trains, reveals targets, or edits old evidence.

Import the byte-locked existing auditor normally and redirect only its OUT
directory. It checks every fit; this wrapper independently integrates all ten
history-derived curves, compares seventy metrics and verifies all ten trajectory
seals. Original source and old evidence remain unchanged. Completion is mandatory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'audit_completed_fits.py'
SOURCE_SHA256 = '2dd299b829450a81a033bf37015cbba2f854a8ad49153a9903f8ea32fdb135bf'
SEEDS = (157,887,2357,6101,12203)
ARMS = ('posterior_epig_conditional_greedy32','posterior_ivr_conditional_greedy32')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def readiness(study=HERE):
    """Only metadata/receipts; a running or paused study produces no output files."""
    study=Path(study);reasons=[]
    required={'report':study/'reporting_manifest.json',
              'worker':study/'exploratory_status.json',
              'ledger':study/'runtime/compute_budget.json'}
    missing=[str(p) for p in required.values() if not p.exists()]
    if missing:return {'ready':False,'reasons':['missing completion metadata'],'missing':missing}
    report,worker,ledger=(read(required[k]) for k in ('report','worker','ledger'))
    if report.get('status')!='COMPLETE_L1005':reasons.append('report is not COMPLETE_L1005')
    if (report.get('successful_fits'),report.get('complete_seed_pairs'),report.get('primary_efficacy_evaluated'))!=(210,5,True):
        reasons.append('report lacks all 210 fits/five complete seed pairs')
    if worker.get('status')!='COMPLETED_L1005' or worker.get('successful_new_fits')!=210:
        reasons.append('worker is not completed at 210 successful fits')
    attempts=ledger.get('attempts',[])
    neural=[a for a in attempts if a.get('kind')=='qgeognn']
    successes=[a for a in neural if a.get('status')=='COMPLETED']
    success_keys=[a.get('key') for a in successes]
    if len(successes)!=210 or len(set(success_keys))!=210:
        reasons.append('ledger does not contain exactly 210 unique successful neural fits')
    if any(a.get('status')=='RUNNING' for a in attempts):reasons.append('ledger still has a running operation')
    if len(neural)>231:reasons.append('neural attempt cap exceeded')
    if max(Counter(a.get('key') for a in neural).values(),default=0)>3:reasons.append('per-fit attempt cap exceeded')
    receipts=list((study/'runtime').glob('seed_*/*/round_*/fit_receipt.json'))
    if len(receipts)!=220:reasons.append('expected 220 completed fit receipts including ten reused initial fits')
    seals=[]
    for seed in SEEDS:
        for arm in ARMS:
            path=study/f'runtime/seed_{seed}/{arm}/trajectory_freeze.json'
            if not path.exists():reasons.append(f'missing trajectory seal: {seed}/{arm}');continue
            seal=read(path)
            if (seal.get('status'),seal.get('rounds'),seal.get('budget'))!=('COMPLETED_L1005',21,1005):
                reasons.append(f'incomplete trajectory seal: {seed}/{arm}')
            seals.append(str(path.relative_to(study)))
    return {'ready':not reasons,'reasons':reasons,'report_status':report.get('status'),
            'worker_status':worker.get('status'),'report_successful_fits':report.get('successful_fits'),
            'ledger_unique_successes':len(set(success_keys)),'neural_attempts':len(neural),
            'completed_fit_receipts':len(receipts),'trajectory_seals':len(seals),
            'metadata_sha256':{k:sha(p) for k,p in required.items()}}


def stable(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def full_metrics_and_seals(study,fit_csv,output):
    """Seventy explicit trapezoid/milestone comparisons; no raw-label interface."""
    with fit_csv.open(newline='') as handle:rows=list(csv.DictReader(handle))
    with (study/'seed_metrics.csv').open(newline='') as handle:published=list(csv.DictReader(handle))
    comparisons=[];seals={}
    for seed in SEEDS:
        for arm in ARMS:
            curve=sorted((r for r in rows if int(r['seed'])==seed and r['arm']==arm),key=lambda r:int(r['round']))
            if [int(r['budget']) for r in curve]!=[333+32*r for r in range(22)]:
                raise RuntimeError(f'incomplete history-derived curve: {seed}/{arm}')
            values={int(r['budget']):float(r['combined_nrmse']) for r in curve}
            metrics={}
            for lo,hi,name in ((333,1005,'full'),(333,525,'early'),(525,813,'middle'),(813,1005,'late')):
                budgets=list(range(lo,hi+1,32))
                metrics['AULC_'+name]=sum((b-a)*(values[a]+values[b])/2 for a,b in zip(budgets,budgets[1:]))/(hi-lo)
            for budget in (525,813,1005):metrics[f'NRMSE_L{budget}']=values[budget]
            match=[r for r in published if int(r['seed'])==seed and r['arm']==arm]
            if len(match)!=1:raise RuntimeError('missing or duplicate published seed metrics')
            for name,value in metrics.items():
                reported=float(match[0][name])
                if not math.isclose(value,reported,rel_tol=2e-12,abs_tol=2e-12):
                    raise RuntimeError(f'history-derived versus published {name} differs: {seed}/{arm}')
                comparisons.append({'seed':seed,'arm':arm,'metric':name,'independent_history_value':value,
                                    'published_value':reported,'absolute_difference':abs(value-reported)})
            trajectory=study/f'runtime/seed_{seed}/{arm}'
            seal_path=trajectory/'trajectory_freeze.json';seal=read(seal_path)
            expected={str((trajectory/f'round_{r:02d}/fit_receipt.json').relative_to(study)) for r in range(22)}
            if set(seal['fit_receipts'])!=expected:raise RuntimeError('trajectory seal does not cover exactly twenty-two fits')
            for relative,digest in seal['fit_receipts'].items():
                if sha(study/relative)!=digest:raise RuntimeError(f'trajectory fit receipt changed: {relative}')
            ordered_selected=[]
            for r in range(21):ordered_selected.extend(read(trajectory/f'round_{r:02d}/selection.json')['selected_ids'])
            if len(ordered_selected)!=672 or len(set(ordered_selected))!=672 or stable(ordered_selected)!=seal['selected_ids_hash']:
                raise RuntimeError('trajectory ordered selection history differs from final seal')
            seals[str(seal_path.relative_to(study))]=sha(seal_path)
    if len(comparisons)!=70 or len(seals)!=10:raise RuntimeError('final metric/seal count differs')
    path=output/'all_five_seed_metric_comparisons.csv'
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    return {'metric_comparisons':70,'trajectory_seals_verified':10,'trajectory_seal_hashes':seals,
            'max_metric_absolute_difference':max(r['absolute_difference'] for r in comparisons)}


def run_final_audit():
    before=readiness()
    if not before['ready']:
        print(json.dumps({'status':'NOT_READY_NO_AUDIT_RUN',**before},ensure_ascii=False,indent=2));return None
    if sha(SOURCE)!=SOURCE_SHA256:raise RuntimeError('existing audit source bytes differ from the reviewed version')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=HERE/'evidence/runner'/f'final_learning_QA_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':'epig-v2-final-artifact-QA','started_unix':time.time(),'status':'RUNNING',
             'reused_audit_sha256':SOURCE_SHA256,'final_audit_sha256':sha(__file__),
             'reuse_mode':'normal Python import; only auditor.OUT changes to a fresh subdirectory',
             'output_directory':str(output),'completion_gate_before':before,
             'source_targets_read':0,'new_fits':0,'old_evidence_overwritten':False}
    receipt_path=output/'final_manifest.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        spec=importlib.util.spec_from_file_location('epig_existing_completed_fit_auditor',SOURCE)
        auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
        auditor.OUT=output/'reused_all_fit_audit'
        auditor.main()
        # Filenames are intentionally those of the unchanged existing auditor,
        # isolated under this new final-QA directory; no prior evidence is touched.
        reused=read(auditor.OUT/'in_progress_learning_QA_20261118.json')
        expected={'verified_downstream_fits':210,'verified_fit_receipts':220,'verified_selections':210,'status':'PASS'}
        if any(reused.get(k)!=v for k,v in expected.items()) or reused['errors']:
            raise RuntimeError('existing all-fit auditor did not verify the complete required snapshot')
        metrics=full_metrics_and_seals(HERE,auditor.OUT/'in_progress_learning_QA_20261118_fits.csv',output)
        after=readiness()
        if not after['ready']:raise RuntimeError('completion gate changed during the audit')
        if sha(SOURCE)!=SOURCE_SHA256:raise RuntimeError('existing audit source changed during validation')
        receipt.update(status='PASS',completed_unix=time.time(),completion_gate_after=after,
                       verified_downstream_fits=210,verified_fit_receipts=220,verified_selections=210,
                       complete_seed_pairs=5,source_bytes_unchanged=True,**metrics,
                       output_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p!=receipt_path},
                       interpretation='Artifact consistency and five-seed reported metrics verified; overlapping seeds are not independent confirmations.')
    except BaseException as exc:
        receipt.update(status='FAIL_PRESERVED',completed_unix=time.time(),error_type=type(exc).__name__,error=str(exc))
        receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');raise
    receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'PASS','final_evidence':str(output),'metric_comparisons':70,
                      'verified_downstream_fits':210,'new_fits':0},indent=2))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-ready',action='store_true',help='read completion metadata only; never write outputs')
    args=parser.parse_args()
    if args.check_ready:print(json.dumps(readiness(),ensure_ascii=False,indent=2))
    else:run_final_audit()

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Export a fixed saved-validation snapshot without API calls, labels or training."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
SEEDS=(157,887,2357,6101,12203)
ARM='free_cw_sol61_xhigh_rerun32'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def export(out):
    if out.exists():raise RuntimeError('Snapshot directory already exists; choose a new output')
    out.mkdir(parents=True)
    at=datetime.now(timezone.utc);markers={s:sorted((STUDY/'progress').glob(f'completed_seed_{s}_round_*.json')) for s in SEEDS}
    captured={s:sorted((STUDY/f'selections/seed_{s}/{ARM}').glob('round_*/batch_freeze.json')) for s in SEEDS}
    spec=importlib.util.spec_from_file_location('saved_free_report_export',ROOT/'scripts/studies/report_free_cw_sol61_five_20261101.py');report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    reference_bytes=(STUDY/'reports/validation_curves.csv').read_bytes()
    reference_sha256=hashlib.sha256(reference_bytes).hexdigest()
    reference=list(csv.DictReader(io.StringIO(reference_bytes.decode())))
    curves=[{k:(v=='True') if v in ('True','False') else float(v) if k.startswith('validation_') else int(v) if k in ('seed','budget') else v for k,v in r.items()} for r in reference if r['method']!=report.METHODS[0]]
    fits=[];completion=[];completed_receipts=[];batches=[]
    for seed in SEEDS:
        records=[read(f) for f in markers[seed]]
        if [r['round'] for r in records]!=list(range(len(records))):raise RuntimeError('Noncontiguous completed rounds')
        for path,record in zip(markers[seed],records):
            for name,h in record['files'].items():
                if sha(STUDY/name)!=h:raise RuntimeError('Completed artifact binding drift')
            completed_receipts.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'receipt':record})
        for rnd in range(len(records)+1):
            point=report.saved_point(report.METHODS[0],seed,rnd)
            if point is None:raise RuntimeError('Missing saved validation point')
            curves.append(point);model,frozen=report.model_directory(report.METHODS[0],seed,rnd)
            fits.append({'seed':seed,'budget':333+32*rnd,'fit_audit':read(model/'fit_audit.json'),'fit_audit_sha256':sha(model/'fit_audit.json'),'prediction_freeze':frozen})
        for path in captured[seed]:
            batch=read(path)
            if len(batch['batch_ids'])!=len(set(batch['batch_ids'])) or len(batch['batch_ids'])!=32:raise RuntimeError('Invalid frozen batch cardinality')
            batches.append({'seed':seed,'round':int(path.parent.name.split('_')[1]),'path':str(path.relative_to(ROOT)),'sha256':sha(path),'completed_fit':int(path.parent.name.split('_')[1])<len(records),'freeze':batch})
        completion.append({'seed':seed,'completed_budget':333+32*len(records),'completed_batches':len(records),'accepted_batches':len(captured[seed])})
    by={(m,s):sorted([r for r in curves if r['method']==m and int(r['seed'])==s],key=lambda r:int(r['budget'])) for m in report.METHODS for s in SEEDS}
    paired=[];summaries=[]
    for other in report.METHODS[1:]:
        common=min(min(len(by[report.METHODS[0],s]),len(by[other,s])) for s in SEEDS)
        for endpoint in sorted({common,*[(b-333)//32+1 for b in (525,813,1005,1389,1485) if (b-333)//32+1<=common]}):
            if endpoint<2:continue
            metrics=['validation_selection_score','validation_V1_r2','validation_V2_r2']
            for seed in SEEDS:
                new,old=by[report.METHODS[0],seed][:endpoint],by[other,seed][:endpoint]
                values={'raw_validation_AULC':(report.aulc(new),report.aulc(old)),**{k:(new[-1][k],old[-1][k]) for k in metrics}}
                for metric,(a,b) in values.items():paired.append({'comparator':other,'budget':333+32*(endpoint-1),'seed':seed,'metric':metric,'new_free':a,'reference':b,'paired_delta_new_minus_reference':a-b})
            for metric in ['raw_validation_AULC',*metrics]:
                group=[r for r in paired if r['comparator']==other and r['budget']==333+32*(endpoint-1) and r['metric']==metric]
                summaries.append({'comparator':other,'budget':333+32*(endpoint-1),'metric':metric,'new_free':report.summarize([r['new_free'] for r in group]),'reference':report.summarize([r['reference'] for r in group]),'paired_delta_new_minus_reference':report.summarize([r['paired_delta_new_minus_reference'] for r in group])})
    attempts=[]
    for seed in SEEDS:
        for path in sorted((STUDY/f'selections/seed_{seed}/{ARM}').glob('round_*/attempt_*.json')):
            if path.name.endswith('.started.json'):continue
            record=read(path)
            if not isinstance(record.get('success'),bool):raise RuntimeError('Invalid physical attempt receipt')
            attempts.append({'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'request_sha256':record['request_sha256'],**{k:record.get(k) for k in ('success','error_type','http_status','reason','ended_utc','next_backoff_seconds','elapsed_seconds')}})
    active=read(STUDY/'active_runtime.json');policies=[]
    for name in ['protocol.json','parallel_execution_amendment.json','persistent_20261102_v2/authorization.json','transport_recovery_20261102/authorization.json','transport_recovery_20261102/controller_authorization.json','http403_recovery_20261102/authorization.json','immediate_retry_20261102/authorization.json']:
        policies.append({'path':str((STUDY/name).relative_to(ROOT)),'sha256':sha(STUDY/name)})
    snapshot={'snapshot_utc':at.isoformat(),'snapshot_shanghai':at.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),'target_budget':1485,'requested_model':'gpt-6.1-sol','reasoning_effort':'xhigh','transport_backoff_seconds':active.get('transport_backoff_seconds'),'seeds':completion,'accepted_batches':len(batches),'completed_batches':len(completed_receipts),'pending_accepted_batches':len(batches)-len(completed_receipts),'test_truth_reads':0,'new_model_calls':0,'new_fits':0,'selector_feedback':False,'source_curves_sha256':reference_sha256,'runtime_entry':str(Path(active['entry']).relative_to(ROOT))}
    table(out/'validation_curves.csv',curves);table(out/'paired_differences.csv',paired)
    for name,data in [('snapshot.json',snapshot),('fit_audits.json',fits),('accepted_batch_index.json',batches),('completion_receipts.json',completed_receipts),('physical_attempt_index.json',attempts),('comparison_summary.json',summaries),('amendment_index.json',policies)]:write(out/name,data)
    common=min(r['completed_budget'] for r in completion)
    lines=['# CW-visible Free LLM32: fixed progress snapshot','',f"Snapshot: {snapshot['snapshot_shanghai']} (Asia/Shanghai).",'',f"Accepted batches: {len(batches)}/180; completed fits: {len(completed_receipts)}/180; pending accepted batches: {snapshot['pending_accepted_batches']}.",'','| Seed | Completed L | Completed batches | Accepted batches |','|---|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['completed_budget']} | {r['completed_batches']} | {r['accepted_batches']} |" for r in completion]
    lines += ['',f'Five-seed comparisons are restricted to each comparator\'s common completed budget (current new-Free minimum: L{common}). Lower validation RMSE and AULC are better. AULC uses every raw nonmonotone point, normalized by budget width; paired mean and sample standard deviation use ddof=1.','', 'Historical Free retains its mixed-model label: seeds 157 and 6101 include Astra/high prefixes; the other seeds use Sol/xhigh. These descriptive comparisons do not isolate a causal CW effect.','', 'The experiment remains running; this is a fixed snapshot, not a final result. Accepted batches without a completion receipt are excluded from fitted curves. No test truth, selector input or scientific memory is read or modified by publication.','', 'Local runtime models, full requests/catalogs, raw SSE mirrors, console logs and live process state are excluded from Git. Compact attempts, frozen selections, protocol/amendments, code and this validation snapshot are retained. Full local lineage is retained for the active run.','', 'Reproduce with the existing local frozen run data: `python scripts/studies/export_free_cw_snapshot_20261102.py --output reports/<new-snapshot-directory>`. A fresh clone needs the original local dataset/initial checkpoints and separately supplied credentials to execute or re-export fits; credentials are never published.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'snapshot_manifest.json',{'snapshot_utc':at.isoformat(),'files':{f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file()}})
    print(json.dumps(snapshot))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);export(parser.parse_args().output.resolve())

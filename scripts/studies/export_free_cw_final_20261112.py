#!/usr/bin/env python3
"""Publish a complete, lossless result snapshot without network, fits or truth reads."""
from __future__ import annotations
import argparse
import collections
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
SEEDS=(157,887,2357,6101,12203)
ARM='free_cw_sol61_xhigh_rerun32'
SECRET=re.compile(rb'(?:sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"(?:api_key|access_token|refresh_token)"\s*:\s*"(?!\s*"|null)[^"\s]{16,}")')
LOCAL_ONLY=('selection3_compute5_20261109','selection3_compute5_offline_audit_fix_20261110')

def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(p): return json.loads(p.read_text())
def dump(p,value): p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def checked_bytes(p):
    raw=p.read_bytes()
    if SECRET.search(raw): raise RuntimeError('Credential-like value in '+str(p.relative_to(ROOT)))
    return raw
def archive(destination, paths):
    counts=collections.Counter(); originals=0
    with destination.open('xb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0,compresslevel=6) as zipped:
            for p in paths:
                raw=checked_bytes(p)
                if p.suffix=='.json':json.loads(raw)
                record={'path':str(p.relative_to(ROOT)),'sha256':sha(raw),'raw_utf8':raw.decode('utf-8')}
                zipped.write((json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode())
                kind='started' if p.name.endswith('.started.json') else p.name.split('_')[0].split('.')[0]
                counts[kind]+=1;originals+=len(raw)
    return {'files':sum(counts.values()),'file_counts':dict(counts),'original_bytes':originals,'archive_bytes':destination.stat().st_size,'sha256':sha(destination.read_bytes())}

def export(out):
    if out.exists():raise RuntimeError('Choose a fresh publication directory')
    verification=read(STUDY/'reports/final_verification_20261112.json')
    terminal=read(STUDY/'reports/terminal_completion_20261112.json')
    assert verification['status']=='FINAL_OFFLINE_AUDIT_PASSED'
    assert terminal['accepted_batches']==terminal['completed_fits']==180
    assert verification['transmission_attempts']==173511
    out.mkdir(parents=True)
    reports=STUDY/'reports'; provenance={};archives={}
    names=['FINAL_REPORT_20261112.html','REPORT.md','comparison_summary.json','completion.csv',
           'validation_curves.csv','paired_differences.csv','final_verification_20261112.json',
           'final_audit_20261112.py','render_final_report_20261112.py']
    names += [f'milestone_L{budget}.json' for budget in (525,813,1005,1389,1485)]
    for name in names:
        p=reports/name;raw=checked_bytes(p);(out/name).write_bytes(raw)
        provenance[name]={'source':str(p.relative_to(ROOT)),'source_sha256':sha(raw),'byte_identical':True}
    for p in sorted((reports/'final_curves_20261112').iterdir()):
        if not p.is_file():continue
        target=out/'final_curves_20261112'/p.name;target.parent.mkdir(exist_ok=True)
        raw=checked_bytes(p);target.write_bytes(raw)
        provenance[str(target.relative_to(out))]={'source':str(p.relative_to(ROOT)),'source_sha256':sha(raw),'byte_identical':True}
    # Do not publish machine-specific authorization/runtime evidence previously kept local.
    p=out/'FINAL_REPORT_20261112.html';text=p.read_text()
    text=re.sub(r'<li><a href="final_runtime_validation_20261112.json">.*?</a></li>','',text)
    text=text.replace('href="costs.json"','href="costs.json.gz"')
    text=text.replace('聊天、冻结输入、成功/失败回执、模型、报告和本地切换证据全部保留；本次工作未提交或推送Git远程。',
        '本目录为用户授权的结果Git归档。原始模型、全量输入与候选池、凭据、运行日志及此前约定仅本地保留的控制器代码/授权/切换证据仍保留本地；归档范围和哈希见publication_manifest.json。')
    p.write_text(text)
    provenance[p.name]['byte_identical']=False
    provenance[p.name]['publication_changes']='Remove link to local-only runtime authorization evidence; point cost link to lossless gzip; describe publication scope.'
    raw=checked_bytes(reports/'costs.json')
    with (out/'costs.json.gz').open('xb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0,compresslevel=6) as f:f.write(raw)
    provenance['costs.json.gz']={'source':str((reports/'costs.json').relative_to(ROOT)),'source_sha256':sha(raw),'decompressed_byte_identical':True}

    (out/'receipts').mkdir();(out/'scientific_artifacts').mkdir()
    for seed in SEEDS:
        base=STUDY/f'selections/seed_{seed}/{ARM}'
        paths=sorted(p for p in base.glob('round_*/*.json') if p.name.startswith(('attempt_','stream_','turn_','rejected_')))
        target=out/f'receipts/seed_{seed}_receipts.jsonl.gz'
        info=archive(target,paths);archives[str(target.relative_to(out))]=info
        assert info['file_counts']['started']==info['file_counts']['attempt']==next(r['transmission_attempts'] for r in verification['terminal_seed_results'] if r['seed']==seed)
        runtime=STUDY/f'runtime/seed_{seed}/{ARM}'
        paths=[]
        for r in range(36):
            d=base/f'round_{r:02d}'
            paths += [d/name for name in ('batch_freeze.json','contract.json','selection.json','feedback.json','label_access_receipt.json')]
            paths += [STUDY/f'progress/completed_seed_{seed}_round_{r:02d}.json']
        initial=STUDY/f'offline_checks/initial_state_seed_{seed}.json';paths.append(initial)
        for r in range(37):
            d=runtime/f'round_{r:02d}'
            model=Path(read(initial)['checkpoint_path']).parent if r==0 else d/'model'
            paths += [d/'prediction_freeze.json',model/'fit_audit.json',model/'history.csv']
        paths.append(runtime/'trajectory_freeze.json')
        target=out/f'scientific_artifacts/seed_{seed}_results.jsonl.gz'
        archives[str(target.relative_to(out))]=archive(target,sorted(set(paths)))
        print(json.dumps({'seed':seed,'attempts':info['file_counts']['attempt'],'receipt_archive_MB':round(info['archive_bytes']/1e6,2),'result_files':archives[str(target.relative_to(out))]['files']}),flush=True)

    snapshot={'captured_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status':'FINAL_180_BATCHES_COMPLETE','accepted_batches':180,'completed_fits':180,'transmission_attempts':173511,
        'seeds':terminal['seeds'],'new_provider_requests':0,'new_fits':0,'test_truth_reads':0,'selector_feedback':False,
        'automation_deleted':terminal['automation']['delete_status']=='deleted','source_files_deleted':False,
        'fair_common_budgets':{c['comparator']:c['budget'] for c in verification['fair_comparisons']},
        'historical_free_model_mixing':verification['historical_free_model_mixing']}
    dump(out/'snapshot.json',snapshot)
    (out/'README.md').write_text('''# CW-visible Free32: completed five-seed results

All five seeds (157, 887, 2357, 6101, 12203) completed 36 rounds / L1485.
180 batches were accepted and trained. All 173511 physical transport attempts
and their started/terminal receipts are retained in the lossless archives.

Open [the final report](FINAL_REPORT_20261112.html) for the Chinese summary,
milestones, per-seed comparisons, paired means/sample standard deviations,
and complete learning curves. [PDF curves](final_curves_20261112/overall_curves.pdf).

Scientific archives retain the original selected batches, contracts, selections,
feedback for selected training rows, label-access receipts, 180 completion markers,
five full trajectory freezes, 185 fit audits/prediction freezes and training histories.
The reused L333 baseline records are included; their existing local paths identify provenance.
Each gzip JSONL row contains path, SHA256 and original UTF-8 bytes in raw_utf8.
Re-encoding raw_utf8 as UTF-8 recovers the exact original file bytes. No source files
were removed or altered. costs.json.gz decompresses byte-for-byte to the automatic costs report.

Per the repository's existing artifact policy, model weights, prediction/feature
arrays, full requests/input packets/catalogs, raw SSE body mirrors, machine logs,
credentials and live locks remain local. Their saved scientific hashes are retained.
The two selection3_compute5 controller directories, tests, authorizations and switch
evidence previously designated local-only are not published. No other experiment's
unrelated working-tree changes are included.

Validation RMSE and raw normalized trapezoidal AULC are lower-is-better.
Every raw nonmonotone point is used; no smoothing or best-so-far curve is substituted.
Five-seed fair common budgets: historical mixed-model Free and Gradient-LCMD L1485;
NoCW and numerical Hybrid L1005. Endpoint improvements do not imply better whole-curve
AULC or isolate a causal CW effect. Historical Free retains its mixed-model tag.

The report's final_verification JSON preserves the original local audit snapshot.
Its report hashes refer to original uncompressed local files, with publication
transformations recorded in publication_manifest.json. Live runtime authorization
evidence remains local. The local terminal audit predates this Git publication.

Publication performs no model API call, fitting or test-truth read and never feeds
scores into acquisition inputs or scientific memory. NoCW was not modified or signaled.
No monetary price is invented; usage and receipts are preserved.

Re-export from the existing local frozen run with:
`python scripts/studies/export_free_cw_final_20261112.py --output reports/<fresh-directory>`.
A fresh clone needs separately retained original runtime models/data and credentials
to rerun the experiment; this Git snapshot is a scientific result/evidence archive.
''')
    files={str(p.relative_to(out)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
    assert all(x['bytes']<50_000_000 for x in files.values()),'Archive should remain comfortably below GitHub file limits'
    manifest={'captured_shanghai':snapshot['captured_shanghai'],'study':str(STUDY.relative_to(ROOT)),'snapshot':snapshot,
        'source_report_provenance':provenance,'lossless_archives':archives,'files':files,
        'local_only_controller_directories':list(LOCAL_ONLY),'new_provider_requests':0,'new_fits':0,'test_truth_reads':0}
    dump(out/'publication_manifest.json',manifest)
    print(json.dumps({'status':'EXPORTED','files':len(files)+1,'archive_MB':round(sum(x['bytes'] for x in files.values())/1e6,2),'snapshot':str(out.relative_to(ROOT))}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.output.resolve())

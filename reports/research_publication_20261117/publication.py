#!/usr/bin/env python3
"""Publish compact evidence without rewriting sealed study files."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,lzma,shutil,subprocess

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
STUDIES=['studies/active_learning/decision_mechanism_followup_20261117','studies/active_learning/volume_response_retry_audit_20261117']
VOLUME=STUDIES[1]
ARCHIVED=[f'{VOLUME}/{name}' for name in ['retry_audit_inputs.json','retry_transport_attempts.csv','VERIFICATION.json']]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def compressed_sha(path):
    h=hashlib.sha256();size=0
    with lzma.open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b);size+=len(b)
    return h.hexdigest(),size

def pack(relative):
    original=ROOT/relative;before=sha(original)
    archive=OUT/'archives'/(original.name+'.xz');archive.parent.mkdir(exist_ok=True)
    temp=archive.with_suffix(archive.suffix+'.tmp')
    with original.open('rb') as src,lzma.open(temp,'wb',preset=6) as dst:shutil.copyfileobj(src,dst)
    digest,size=compressed_sha(temp)
    assert digest==before==sha(original) and size==original.stat().st_size
    temp.replace(archive)
    return dict(original_path=relative,original_sha256=before,original_bytes=size,
        archive_path=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)

def build():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:archives=list(executor.map(pack,ARCHIVED))
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z','--',*STUDIES],cwd=ROOT).decode().split('\0')
    published=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(set(names)-{''})]
    local=[]
    for p in sorted((ROOT/VOLUME).rglob('*')):
        if not p.is_file():continue
        if p.name in ['best.pt','history.csv','started.json','progress.json','metadata.csv','authorized_labels.npz','training_status.json'] or 'worker_logs' in p.parts:
            local.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    payload=dict(schema_version=1,studies=STUDIES,archives=archives,published_study_files=published,local_runtime_files=local,
        original_study_files_rewritten=False,archive_roundtrip_verified=True,
        excluded_preexisting_modifications=['reports/llm_eight_to1549_20261018/live_status.json','studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018/recovery_monitor_status.json'],
        policy='docs/README.md artifact retention rules: compact scientific evidence in Git; new checkpoints/training runtime remain local.',
        scope='Restores only three losslessly archived evidence files. Full study verification also requires the original local inputs and runtime artifacts recorded in each study seal.')
    (OUT/'MANIFEST.json').write_text(json.dumps(payload,indent=2)+'\n')

def verify(restore=False):
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for entry in manifest['archives']:
        assert entry['original_path'] in ARCHIVED
        archive=ROOT/entry['archive_path'];assert sha(archive)==entry['archive_sha256']
        digest,size=compressed_sha(archive);assert digest==entry['original_sha256'] and size==entry['original_bytes']
        target=ROOT/entry['original_path']
        if target.exists():assert sha(target)==digest
        elif restore:
            target.parent.mkdir(parents=True,exist_ok=True)
            with lzma.open(archive,'rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            assert sha(target)==digest
    for entry in manifest['published_study_files']:assert sha(ROOT/entry['path'])==entry['sha256']
    print(json.dumps(dict(status='PASS',published_study_files=len(manifest['published_study_files']),archives=len(manifest['archives']),
        archive_bytes=sum(v['archive_bytes'] for v in manifest['archives']),local_runtime_files=len(manifest['local_runtime_files']),original_study_files_rewritten=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');parser.add_argument('--restore',action='store_true');args=parser.parse_args()
    if args.build:build()
    verify(args.restore)

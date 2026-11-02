#!/usr/bin/env python
"""User-authorized recovery of interrupted CW fits with preserved attempt evidence."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005'
PYTHON='/Users/fish/miniforge3/envs/fish/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def main():
    authorization=read(S/'CONTINUE_UNTIL_COMPLETE_AUTHORIZATION.json')
    assert authorization['automatic_interrupted_fit_retries_authorized'] is True
    with (S/'recovery.lock').open('a+') as recovery:
        fcntl.flock(recovery,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks=[]
        try:
            for name in ('pipeline.lock','execution.lock'):
                f=(S/name).open('a+');locks.append(f)
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    print(json.dumps({'status':'already_running','lock':name}));return
            ps=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
            if any('python' in row.lower() and 'scripts/studies/continue_cw_without_whitening.py' in row and ' -c ' not in row for row in ps):
                print(json.dumps({'status':'already_running','reason':'pipeline process found'}));return
            for name,digest in read(S/'AMENDED_EXECUTION_FREEZE.json').items():
                assert sha(ROOT/name)==digest,name
            receipts=S/'recovery_attempts';receipts.mkdir(exist_ok=True)
            restarts=[]
            for seed in (157,6101):
                for arm in ('cw_maxdet','cw_ivr'):
                    root=S/f'runtime/seed_{seed}/{arm}';ledger_path=root/'fit_ledger.json'
                    if not ledger_path.exists():continue
                    ledger=read(ledger_path)
                    for key,entry in list(ledger.items()):
                        d=root/f'round_{int(key):02d}'
                        audit=d/'model/fit_audit.json'
                        if entry['status']=='completed':
                            assert audit.exists() and sha(audit)==entry['audit_sha256'],'completed evidence missing'
                            continue
                        if audit.exists():continue
                        assert entry['status'] in ('running','retry_authorized_after_interruption')
                        assert not (d/'state.json').exists()
                        before=root/f'round_{int(key)-1:02d}'
                        frozen=read(before/'batch_freeze.json')
                        assert sha(before/'premeasurement.csv.gz')==frozen['premeasurement_sha256']
                        tag=f'{seed}_{arm}_{key}_{time.time_ns()}'
                        archive=S/f'runtime/interrupted_attempts/{tag}/model'
                        source=d/'model'
                        evidence={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()} if source.exists() else {}
                        receipt={'status':'authorized_scratch_restart','authorization_sha256':sha(S/'CONTINUE_UNTIL_COMPLETE_AUTHORIZATION.json'),
                            'seed':seed,'arm':arm,'round':int(key),'budget':333+32*int(key),'previous_reservation':entry,
                            'ledger_sha256_before':sha(ledger_path),'partial_artifacts':evidence,
                            'archive':str(archive.relative_to(ROOT)),'batch_freeze_sha256':sha(before/'batch_freeze.json'),
                            'recorded_unix':time.time(),'policy':'same batch and initialization; partial fit not counted as completed; no new selection'}
                        save(receipts/f'{tag}.json',receipt)
                        if source.exists():
                            archive.parent.mkdir(parents=True,exist_ok=False);source.rename(archive)
                        ledger[key]={'status':'retry_authorized_after_interruption','started_unix':time.time(),
                            'recovery_receipt':str((receipts/f'{tag}.json').relative_to(ROOT))}
                        save(ledger_path,ledger)
                        restarts.append(tag)
        finally:
            for f in locks:f.close()
        # Launch detached from tool/terminal pipes; computer sleep is inhibited while it runs.
        log=S/f'recovery_{time.strftime("%Y%m%d_%H%M%S")}.log'
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONUNBUFFERED='1')
        with log.open('ab') as stream:
            child=subprocess.Popen([PYTHON,'-u',str(ROOT/'scripts/studies/continue_cw_without_whitening.py')],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        save(S/'RECOVERY_LAUNCH.json',{'status':'launched','pid':child.pid,'log':str(log.relative_to(ROOT)),
            'started_unix':time.time(),'restarts':restarts,'detached_session':True})
        save(S/'pipeline_state.json',{'status':'starting','pid':child.pid,'recovery_launch':'RECOVERY_LAUNCH.json'})
        print(json.dumps({'pid':child.pid,'log':str(log),'restarted_fits':len(restarts)}))

if __name__=='__main__':main()

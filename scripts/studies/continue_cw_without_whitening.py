#!/usr/bin/env python
"""Explicit user amendment: preserve frozen science, stop Whitening at525."""
import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2 import cw_factorial_extension as first
from src.qgeognn_al.active_learning_v2 import cw_factorial_to1005 as last
from src.qgeognn_al.active_learning_v2.protocol import stable_hash
from src.qgeognn_al.artifacts import sha256_file as sha
from src.qgeognn_al.training.predictor import atomic_json
import torch

ROOT = first.base.ROOT
ACTIVE = ('cw_maxdet','cw_ivr')


def worker(seed):
    # Only orchestration membership changes. Frozen transforms/selectors/training stay unchanged.
    last.ARMS = {name:last.ARMS[name] for name in ACTIVE}
    last.run_seed(seed)
    atomic_json(last.STUDY/f'progress_seed_{seed}.json',{'status':'completed','fits':22,'budget':1005,'whitening':'stopped_at525_by_user'})


def command(name,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts/studies'/name),*args],cwd=ROOT,check=True)


def main():
    torch.set_num_threads(2)
    amendment = first.base.read(last.STUDY/'WHITENING_STOP_AMENDMENT.json')
    assert amendment['allowed_continuing_arms'] == list(ACTIVE)
    freeze = first.base.read(last.STUDY/'AMENDED_EXECUTION_FREEZE.json')
    for name,digest in freeze.items():
        assert sha(ROOT/name) == digest
    p1,p2 = first.verify(),last.verify()
    with (last.STUDY/'pipeline.lock').open('a+') as pipeline:
        fcntl.flock(pipeline,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(last.STUDY/'pipeline_state.json',{'status':'running','step':'finish653_without_whitening','amendment_sha256':sha(last.STUDY/'WHITENING_STOP_AMENDMENT.json')})
        # Wait for the prior training process to save a complete fit and relinquish its lock.
        with (first.STUDY/'execution.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=first.base.read(first.STUDY/'fit_ledger.json')
            assert all('/whitening_lcmd/' not in key for key in ledger)
            for arm in ACTIVE:
                for seed in first.SEEDS:
                    first.run_trajectory(seed,arm,p1)
            ledger=first.base.read(first.STUDY/'fit_ledger.json')
            assert len(ledger)==16 and all(x['status']=='completed' for x in ledger.values())
            first.save_once(first.STUDY/'GLOBAL_FREEZE.json',{'status':'complete_user_amended_scope','test_reveals':0,
                'whitening':'stopped_at525_by_user','amendment_sha256':sha(last.STUDY/'WHITENING_STOP_AMENDMENT.json'),
                'trajectories':{f'{s}/{a}':sha(first.STUDY/f'runtime/seed_{s}/{a}/trajectory_freeze.json') for s in first.SEEDS for a in ACTIVE}})
            atomic_json(first.STUDY/'RUN_STATE.json',{'status':'completed','completed_fits':16,'new_fit_limit':16,'whitening':'stopped_at525_by_user','paid_llm_calls':0,'test_reveals':0})
        command('run_cw_factorial_extension.py','report')
        command('audit_cw_factorial_extension.py')
        anchors=[]
        for seed in first.SEEDS:
            for arm in ACTIVE:
                d=first.STUDY/f'runtime/seed_{seed}/{arm}/round_10'
                state=first.base.read(d/'state.json');a=first.base.read(d/'model/fit_audit.json')
                assert len(state['labeled'])==653 and sha(d/'model/best.pt')==state['checkpoint_sha256']==a['checkpoint_sha256']
                anchors.append({'seed':seed,'arm':arm,'state_sha256':sha(d/'state.json'),'checkpoint_sha256':a['checkpoint_sha256'],'audit_sha256':sha(d/'model/fit_audit.json')})
        first.save_once(last.STUDY/'ANCHORS.json',{'protocol_hash':stable_hash(p2),'source_global_freeze_sha256':sha(first.STUDY/'GLOBAL_FREEZE.json'),
            'amendment_sha256':sha(last.STUDY/'WHITENING_STOP_AMENDMENT.json'),'anchors':anchors})
        with (last.STUDY/'execution.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            atomic_json(last.STUDY/'RUN_STATE.json',{'status':'running','new_fit_limit':44,'whitening':'stopped_at525_by_user','max_concurrent_fits':2})
            atomic_json(last.STUDY/'pipeline_state.json',{'status':'running','step':'continue_two_arms_to1005'})
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
                list(pool.map(worker,first.SEEDS))
            first.save_once(last.STUDY/'GLOBAL_FREEZE.json',{'status':'complete_user_amended_scope','test_reveals':0,
                'whitening':'stopped_at525_by_user','amendment_sha256':sha(last.STUDY/'WHITENING_STOP_AMENDMENT.json'),
                'trajectories':{f'{s}/{a}':sha(last.STUDY/f'runtime/seed_{s}/{a}/trajectory_freeze.json') for s in first.SEEDS for a in ACTIVE}})
            atomic_json(last.STUDY/'RUN_STATE.json',{'status':'completed','completed_fits':44,'new_fit_limit':44,'whitening':'stopped_at525_by_user','paid_llm_calls':0,'test_reveals':0})
        last.ARMS={name:last.ARMS[name] for name in ACTIVE}
        last.report()
        command('audit_cw_factorial_to1005.py')
        command('finalize_cw_factorial_to1005.py')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        atomic_json(last.STUDY/'pipeline_state.json',{'status':'blocked','error':repr(error),'whitening':'stopped_at525_by_user'})
        raise

"""Authorized 15-batch validation-only extension; original scientific code unchanged."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import scientist_study as s
from src.qgeognn_al.active_learning_v2.dialog_bridge import read, once
s.configure_revision('astra_high_full_pool_20261001')
EXT = s.STUDY / 'extension_authorized_to813'
TARGETS = ((157, 'cw16_llm16_scientist_v2', 8),
           (6101, 'cw16_llm16_scientist_v2', 0),
           (6101, 'free_llm32_scientist_v2', 0))
FROZEN = read(s.STUDY / 'protocol.json')
# Preserve exact hash checks on every original file; unrelated newly merged
# modules are recorded in the extension manifest, not silently added to protocol.
def frozen_code_hashes():
    return {name: s.sha256_file(ROOT / name) for name in FROZEN['code_hashes']}
s.code_hashes = frozen_code_hashes
s.BUDGETS = tuple(range(333, 814, 32))
def selection_directory(seed, method, r):
    if (seed, method) not in {(a,b) for a,b,_ in TARGETS} or r not in range(15):
        raise ValueError('outside authorized extension')
    return s.STUDY / f'selections/seed_{seed}/{method}/round_{r:02d}'
s.selection_directory = selection_directory

def curve(seed, method):
    rows=[]
    for r,b in enumerate(s.BUDGETS):
        p=s.STUDY/f'runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json'
        if not p.exists(): continue
        f=read(p)
        for k in ('checkpoint','prediction'):
            if s.sha256_file(Path(f[k+'_path'])) != f[k+'_sha256']: raise RuntimeError('frozen prediction drift')
        a=read(Path(f['checkpoint_path']).parent/'fit_audit.json')
        if a['train_rows']!=b: raise RuntimeError('fit budget mismatch')
        rows.append({'labels':b,'validation_nrmse':a['best_validation_combined_normalized_rmse']})
    return rows

def random_reuse():
    rows=[]; hashes={}
    base=ROOT/'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_157/random'
    for r,b in enumerate(s.BUDGETS):
        d=base/f'round_{r:02d}';f=read(d/'contract.json')
        assert f['status']=='FROZEN_BEFORE_TEST_TRUTH' and f['input']['active_label_count']==b
        assert f['input']['outer_seed']==157 and f['test_truth_access_count']==0
        assert all(f['input']['training_config'][k]==v for k,v in FROZEN['training'].items())
        s.verify_files(d,f['files'])
        a=read(d/'model/fit_audit.json')
        assert a['train_rows']==b
        hashes[str((d/'contract.json').relative_to(ROOT))]=s.sha256_file(d/'contract.json')
        rows.append({'labels':b,'validation_nrmse':a['best_validation_combined_normalized_rmse']})
    return {'seed':157,'method':'random32','status':'REUSED_EXISTING_FROZEN_TRAJECTORY',
            'validation_learning_curve':rows,'contracts':hashes,'test_evaluation_performed':False}

def preflight():
    s.validate();s.check_transport(FROZEN['transport'])
    prior={}
    for p in (s.STUDY/'runtime/seed_157/cw16_llm16_scientist_v2/trajectory_freeze.json',
              s.STUDY/'extension_hybrid157_to589/trajectory_freeze.json'):
        f=read(p);s.verify_files(s.STUDY,f['files']);prior[str(p.relative_to(s.STUDY))]=s.sha256_file(p)
    assert len(s.state(s.Context(157),TARGETS[0][1],8)[0])==589
    for seed,method,_ in TARGETS[1:]:
        assert len(s.state(s.Context(seed),method,0)[0])==333
    manifest={'authorization':'User explicitly approved 813 labels, unchanged initial settings, additional rounds and seed6101.',
        'classification':'exploratory extension after inspection of seed157 validation results',
        'targets':[{'seed':a,'method':b,'existing_rounds':c} for a,b,c in TARGETS],
        'budgets':list(s.BUDGETS),'batch_size':32,'hybrid_CW_count':16,'hybrid_LLM_count':16,
        'transport':FROZEN['transport'],'test_evaluation':False,'correction_budget_changed':False,
        'protocol_sha256':s.sha256_file(s.STUDY/'protocol.json'),'prior_freezes':prior,
        'runner_sha256':s.sha256_file(Path(__file__)),
        'unrelated_added_modules':sorted(set(str(p.relative_to(ROOT)) for p in (ROOT/'src/qgeognn_al').rglob('*.py'))-set(FROZEN['code_hashes']))}
    return manifest,random_reuse()

def finish(seed,method):
    labeled,_,_,_=s.state(s.Context(seed),method,15)
    assert len(labeled)==813
    ids=[];hashes={}
    for r in range(15):
        d=selection_directory(seed,method,r);f=s.audit_batch(d)
        assert len(f['batch_ids'])==32 and len(f['pending_ids'])==(16 if method==s.METHODS[0] else 0)
        ids+=f['batch_ids']
        for n in ('batch_freeze.json','feedback.json','label_access_receipt.json'):
            hashes[str((d/n).relative_to(s.STUDY))]=s.sha256_file(d/n)
    assert len(ids)==len(set(ids))==480
    for r in range(16):
        p=s.STUDY/f'runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json'
        hashes[str(p.relative_to(s.STUDY))]=s.sha256_file(p)
    rows=curve(seed,method);assert len(rows)==16
    once(EXT/f'{seed}_{method}_trajectory_freeze.json',{'seed':seed,'method':method,'selected_ids':ids,'final_active_labels':813,'files':hashes})
    once(EXT/f'{seed}_{method}_result.json',{'seed':seed,'method':method,'final_active_labels':813,'validation_learning_curve':rows,'test_evaluation_performed':False})

if __name__=='__main__':
    with s.exclusive_lock(s.STUDY/'execution.lock'):
        manifest,random=preflight()
        if '--check' in sys.argv:
            print(json.dumps({'status':'PREFLIGHT_PASS','targets':manifest['targets'],'budgets':manifest['budgets'],'random':random['status']}),flush=True)
            sys.exit(0)
        EXT.mkdir(exist_ok=True);once(EXT/'manifest.json',manifest);once(EXT/'random157_reuse.json',random)
        s._progress({'action':'extension813_started','pid':os.getpid()})
        try:
            # Sequential round-robin scheduling avoids competing training jobs.
            for step in range(15):
                for seed,method,start in TARGETS:
                    r=start+step
                    if r>=15: continue
                    d=selection_directory(seed,method,r)
                    pred=s.STUDY/f'runtime/seed_{seed}/{method}/round_{r+1:02d}/prediction_freeze.json'
                    details={'seed':seed,'method':method,'round':r,'extension813':True}
                    if pred.exists() and (d/'feedback.json').exists():
                        s.state(s.Context(seed),method,r+1);continue
                    s._progress({'action':'extension813_cycle_started',**details})
                    with s.status_phase(f'813 extension seed={seed} {method} {s.BUDGETS[r]}->{s.BUDGETS[r+1]} select',30,**details):
                        s.select(seed,method,r)
                    with s.status_phase(f'813 extension seed={seed} {method} train {s.BUDGETS[r+1]}',30,**details):
                        s.advance(seed,method,r)
                    s._progress({'action':'extension813_cycle_completed',**details})
            for seed,method,_ in TARGETS: finish(seed,method)
            for name,h in manifest['prior_freezes'].items():
                assert s.sha256_file(s.STUDY/name)==h
                s.verify_files(s.STUDY,read(s.STUDY/name)['files'])
            once(EXT/'result.json',{'status':'ALL_AUTHORIZED_TRAJECTORIES_COMPLETE','final_active_labels':813,'test_evaluation_performed':False})
            s._progress({'action':'extension813_completed'})
        except Exception as e:
            s._progress({'action':'extension813_stopped','error_type':type(e).__name__,'message':str(e)[:1000]})
            raise

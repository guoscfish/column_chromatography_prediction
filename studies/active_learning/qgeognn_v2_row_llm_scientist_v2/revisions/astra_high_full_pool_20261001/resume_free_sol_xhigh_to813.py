"""User-authorized model transition; immutable inherited Astra prefix, new Sol suffix."""
import os,sys,shutil,json
from pathlib import Path
import resume_free_two_seeds_to589 as inherited
from src.qgeognn_al.active_learning_v2 import scientist_transport as transport
s=inherited.s
original=inherited.original
OLD=s.STUDY
NEW=OLD.parent/'free_sol_xhigh_continuation_20261012'
STARTS={157:6,6101:10}
METHOD=inherited.METHOD
OLD_CONFIG=original.read(OLD/'protocol.json')['transport']
NEW_CONFIG={**OLD_CONFIG,'model':'gpt-6.1-sol','reasoning_effort':'xhigh'}
old_predict=s.prediction_freeze
prefix_selector=s.run_selector

def directory(seed,method,r):
    if seed not in STARTS or method!=METHOD or r not in range(15):
        raise ValueError('Only authorized Free two-seed continuation')
    return NEW/f'selections/seed_{seed}/{method}/round_{r:02d}'

def choose(packet,catalog,path,config,call=None):
    seed,r=packet['seed'],packet['round']
    if r<STARTS[seed]:
        return prefix_selector(packet,catalog,path,OLD_CONFIG,call=call)
    original.once(Path(path)/'model_transition.json',{'model':NEW_CONFIG['model'],'reasoning_effort':'xhigh',
        'historical_prefix_model':OLD_CONFIG['model'],'first_new_selection_round':STARTS[seed],
        'classification':'model-switched exploratory continuation'})
    return transport.run_selector(packet,catalog,path,NEW_CONFIG,call=call)

def prediction(context,method,r,labeled):
    if r<=STARTS[context.seed]:
        path=NEW/f'runtime/seed_{context.seed}/{method}/round_{r:02d}'
        f=original.read(path/'prediction_freeze.json')
        for kind in ('checkpoint','prediction'):
            assert s.sha256_file(Path(f[kind+'_path']))==f[kind+'_sha256']
        assert f['L_t_ids_hash']==s.ids_hash(context.ids(labeled))
        return Path(f['checkpoint_path']),path
    return old_predict(context,method,r,labeled)

def prepare():
    current,_=original.preflight()
    frozen=original.read(original.EXT/'manifest.json')
    comparable=dict(current);comparable['unrelated_added_modules']=frozen['unrelated_added_modules']
    assert comparable==frozen,'Original manifest changed beyond independent additions'
    added={name:s.sha256_file(original.ROOT/name) for name in current['unrelated_added_modules']}
    for seed,r in STARTS.items():
        assert len(s.state(s.Context(seed),METHOD,r)[0])==s.BUDGETS[r]
        for j in range(r+1):
            f=original.read(OLD/f'runtime/seed_{seed}/{METHOD}/round_{j:02d}/prediction_freeze.json')
            for kind in ('checkpoint','prediction'):
                assert s.sha256_file(Path(f[kind+'_path']))==f[kind+'_sha256']
    NEW.mkdir(exist_ok=True)
    for name in ('protocol.json','selector_prompt.txt','control_reuse.json','historical_artifacts.json'):
        if not (NEW/name).exists():shutil.copy2(OLD/name,NEW/name)
        assert s.sha256_file(NEW/name)==s.sha256_file(OLD/name)
    hashes={}
    for seed,r in STARTS.items():
        for subtree in ('selections','runtime'):
            a=OLD/f'{subtree}/seed_{seed}/{METHOD}';b=NEW/f'{subtree}/seed_{seed}/{METHOD}'
            b.mkdir(parents=True,exist_ok=True)
            for j in range(r+(subtree=='runtime')):
                src=a/f'round_{j:02d}';dst=b/src.name
                if not dst.exists():shutil.copytree(src,dst)
                # Prefix is verified byte-for-byte and retained with original absolute checkpoint references.
                for f in src.rglob('*'):
                    if f.is_file():
                        q=dst/f.relative_to(src);h=s.sha256_file(f);assert s.sha256_file(q)==h
                        hashes[str(q.relative_to(NEW))]=h
        for name in ('context.json',):
            a=OLD/f'runtime/seed_{seed}/{name}';b=NEW/f'runtime/seed_{seed}/{name}'
            if not b.exists():shutil.copy2(a,b)
    record={'authorization':'User requested both Free seeds switch to gpt-6.1-sol/xhigh then complete813 on2026-10-12',
        'source_revision':str(OLD),'start_labels':{str(k):s.BUDGETS[v] for k,v in STARTS.items()},
        'target_labels':813,'historical_transport':OLD_CONFIG,'new_transport':NEW_CONFIG,
        'original_protocol_sha256':s.sha256_file(OLD/'protocol.json'),'original_manifest_sha256':s.sha256_file(original.EXT/'manifest.json'),
        'independent_modules_sha256':added,'inherited_files':hashes,'runner_sha256':s.sha256_file(Path(__file__)),
        'classification':'exploratory model transition; not a pure Sol trajectory',
        'historical_deviations':'157 round6 extra correction and character repair;6101 round0 extra correction',
        'training_prompt_selection_budget_changed':False,'test_evaluation_performed':False}
    original.once(NEW/'transition_manifest.json',record)
    s.configure_revision(NEW.name)
    s.selection_directory=directory;s.run_selector=choose;s.prediction_freeze=prediction
    inherited.repair.DIRECTORY=directory(157,METHOD,5)
    original.authorized.DIRECTORY=directory(6101,METHOD,0)
    s.validate()
    return record

def finish(seed):
    assert len(s.state(s.Context(seed),METHOD,15)[0])==813
    ids=[];files={}
    for r in range(15):
        d=directory(seed,METHOD,r);b=s.audit_batch(d)
        assert len(b['batch_ids'])==32 and not b['pending_ids'];ids+=b['batch_ids']
        for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):
            files[str((d/name).relative_to(NEW))]=s.sha256_file(d/name)
        if r>=STARTS[seed]:
            for f in d.glob('turn_*.json'):
                receipt=original.read(f);assert receipt['provenance']['served_model'].startswith('gpt-6.1-sol')
    assert len(ids)==len(set(ids))==480
    rows=original.curve(seed,METHOD);assert len(rows)==16
    original.once(NEW/f'{seed}_result.json',{'seed':seed,'final_labels':813,'selected_ids':ids,
        'validation_learning_curve':rows,'files':files,'transition_labels':s.BUDGETS[STARTS[seed]],
        'suffix_model':'gpt-6.1-sol','suffix_effort':'xhigh','test_evaluation_performed':False})

def main():
    # Shared source-study lock prevents another scheduler continuing the inherited trajectory.
    with s.exclusive_lock(OLD/'execution.lock'):
        record=prepare()
        if '--check' in sys.argv:
            print('SOL_XHIGH_TRANSITION_PREFLIGHT_PASS',flush=True);return
        s._progress({'action':'sol_free813_started','pid':os.getpid()})
        try:
            for seed,start in STARTS.items():
                for r in range(start,15):
                    p=NEW/f'runtime/seed_{seed}/{METHOD}/round_{r+1:02d}/prediction_freeze.json'
                    if p.exists():s.state(s.Context(seed),METHOD,r+1);continue
                    details={'seed':seed,'method':METHOD,'round':r,'model':'gpt-6.1-sol','effort':'xhigh'}
                    with s.status_phase(f'Sol Free seed{seed} {s.BUDGETS[r]}->{s.BUDGETS[r+1]} select',30,**details):s.select(seed,METHOD,r)
                    with s.status_phase(f'Sol Free seed{seed} train {s.BUDGETS[r+1]}',30,**details):s.advance(seed,METHOD,r)
                finish(seed)
            assert s.sha256_file(OLD/'protocol.json')==record['original_protocol_sha256']
            assert s.sha256_file(original.EXT/'manifest.json')==record['original_manifest_sha256']
            original.once(NEW/'result.json',{'status':'BOTH_FREE_COMPLETE813','test_evaluation_performed':False})
            s._progress({'action':'sol_free813_completed'})
        except Exception as exc:
            s._progress({'action':'sol_free813_stopped','error_type':type(exc).__name__,'message':str(exc)[:1000]});raise

if __name__=='__main__':main()

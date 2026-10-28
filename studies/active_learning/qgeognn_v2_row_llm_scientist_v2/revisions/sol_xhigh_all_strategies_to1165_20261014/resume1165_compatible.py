"""Resume the authorized eight trajectories; extend Random reporting metadata only."""
import continue_eight_to1165 as original
import continue_cw_store_compatible as cw_adapter
from continue_eight_to1165 import *
from src.qgeognn_al.active_learning_v2 import sequential_runner, sequential_protocol

old_label_budget = sequential_runner.label_budget

def extended_label_budget(active):
    if active <= 1005:
        return old_label_budget(active)
    assert int(active) in (1037,1069,1101,1133,1165)
    return {'active_label_count':int(active),
        'active_label_fraction_outer_train':int(active)/sequential_protocol.OUTER_TRAIN_ROWS,
        'shared_validation_label_count':sequential_protocol.VALIDATION_ROWS,
        'total_observed_non_test_labels':int(active)+sequential_protocol.VALIDATION_ROWS,
        'total_observed_fraction_full_dataset':(int(active)+sequential_protocol.VALIDATION_ROWS)/sequential_protocol.FULL_DATASET_ROWS}

def guard_adapter():
    record=read(HERE/'resume1165_compatibility.json')
    assert sha(__file__)==record['adapter_sha256']
    assert sha(HERE/'continue_eight_to1165.py')==record['original_runner_sha256']
    assert sha(HERE/'continue_cw_store_compatible.py')==record['cw_adapter_sha256']
    original.guard()
    sequential_runner.label_budget=extended_label_budget
    original.control_worker=cw_adapter.control_worker

def main():
    guard_adapter()
    if '--check' in sys.argv:
        assert extended_label_budget(1165)['active_label_count']==1165
        assert extended_label_budget(1005)==old_label_budget(1005)
        for seed in (157,6101):
            f=HERE/f'runtime/seed_{seed}/random32/round_22/model/fit_audit.json';a=read(f)
            assert a['train_rows']==1037
            assert sha(f.parent/'best.pt')==a['checkpoint_sha256']
            assert sha(f.parent/'predictions.csv.gz')==a['prediction_sha256']
        print('RESUME1165_PREFLIGHT_PASS: source guards, Random metadata extension, cached1037 fit hashes.',flush=True)
        return 0
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');return original.worker(int(sys.argv[i+1]),sys.argv[i+2])
    original.activate()
    children={}
    with s.exclusive_lock(HERE/'coordinator.lock'):
        for seed in (157,6101):
            for method in METHODS:
                result=HERE/f'{seed}_{method}_result.json'
                if result.exists():
                    assert read(result)['final_labels']==1165
                    continue
                log=HERE/f'resumed_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
                with log.open('ab') as out:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(seed),method],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=os.environ.copy())
                children[seed,method]=child
        once(HERE/f'resume_launch_{time.strftime("%Y%m%d_%H%M%S")}.json',{'coordinator_pid':os.getpid(),'worker_pids':{f'{seed}:{method}':child.pid for (seed,method),child in children.items()}})
        while any(c.poll() is None for c in children.values()):
            for (seed,method),child in children.items():
                if method in (HYBRID,FREE) and (HERE/f'quota_seed_{seed}.json').exists() and child.poll() is None:child.terminate()
            time.sleep(1)
        exits={f'{seed}:{method}':c.returncode for (seed,method),c in children.items()}
        if any(exits.values()):s._progress({'action':'incomplete_workers','exit_codes':exits});return 79
        guard_adapter()
        for seed in (157,6101):
            for method in METHODS:
                r=read(HERE/f'{seed}_{method}_result.json')
                assert r['final_labels']==1165 and len(r['additional_ids'])==len(set(r['additional_ids']))==160
        once(HERE/'result.json',{'status':'ALL_EIGHT_COMPLETE1165','test_evaluation_performed':False});return 0

if __name__=='__main__':sys.exit(main())

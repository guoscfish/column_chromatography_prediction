"""User-authorized bounded continuation of all factorial curves to1005."""
from __future__ import annotations
import fcntl
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import cw_factorial_extension as previous
from .cw_factorial import ARMS, SEEDS
from .protocol import stable_hash, ids_hash
from .runner import fit_from_same_initialization, predict_outputs
from .benchmark_protocol import initialization_seed
from .benchmark_reporting import metric_row
from ..artifacts import sha256_file as sha
from ..models import load_predictor_checkpoint
from ..training.predictor import atomic_json

base = previous.base
STUDY = base.PARENT / 'qgeognn_v2_row_cw_selector_extension_to_1005'
ROUNDS = range(11, 22)
CONTROLS = ('raw_lcmd', 'raw_maxdet', 'raw_ivr', 'cw_lcmd', 'random')


def control_dir(seed, arm, r):
    if arm == 'cw_lcmd' and r >= 10:
        return base.PARENT / f'qgeognn_v2_row_cw_lcmd_to_1005/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
    return previous.source_dir(seed, arm, r)


def control_state(seed, arm, r):
    d = control_dir(seed, arm, r)
    if arm in ('raw_lcmd', 'random'):
        return np.load(d/'labeled_indices.npy'), np.load(d/'unlabeled_indices.npy')
    if arm == 'raw_ivr':
        x = base.read(d/'input.json')
        return np.array(x['ordered_labeled_indices']), np.array(x['ordered_unlabeled_indices'])
    x = pd.read_csv(d/'state.csv')
    return x.loc[x.role.eq('labeled'),'canonical_index'].to_numpy(int), x.loc[x.role.eq('unlabeled'),'canonical_index'].to_numpy(int)


def prepare():
    previous.verify()
    STUDY.mkdir(parents=True, exist_ok=True)
    controls = []
    for seed in SEEDS:
        c = base.Context(seed)
        for arm in CONTROLS:
            old_l, old_u = previous.historic_state(seed, arm, 10)
            for r in ROUNDS:
                d = control_dir(seed, arm, r)
                a = base.read(d/'model/fit_audit.json')
                cp = torch.load(d/'model/best.pt', weights_only=False)
                assert cp['training_config'] == c.config and cp['preprocessing'] == c.preprocessing
                assert cp['normalization'] == c.old['normalization']
                assert a['initialization_hash'] == c.anchor_audit['initialization_hash']
                assert a['validation_ids_hash'] == c.anchor_audit['validation_ids_hash']
                assert sha(d/'model/best.pt') == a['checkpoint_sha256']
                assert sha(d/'model/predictions.csv.gz') == a['prediction_sha256']
                l, u = control_state(seed, arm, r)
                assert l[:len(old_l)].tolist() == old_l.tolist() and len(l) == 333+32*r == a['train_rows']
                assert len(set(l)) == len(l) and not set(l) & set(u)
                assert u.tolist() == [v for v in old_u if v not in set(l[-32:])]
                assert a['train_ids_hash'] == ids_hash(c.ids(l))
                prev = control_dir(seed, arm, r-1)
                if arm == 'raw_ivr':
                    ids = base.read(prev/'selection.json')['selected_ids']
                else:
                    folder = 'acquisition' if arm == 'raw_maxdet' else 'acquisition_artifacts'
                    ids = pd.read_csv(prev/folder/'selected_next_batch.csv').sample_id.astype(str).tolist()
                assert ids == c.ids(l[-32:])
                controls.append({'seed':seed, 'arm':arm, 'round':r, 'budget':333+32*r,
                    'checkpoint':str((d/'model/best.pt').relative_to(base.ROOT)),
                    'checkpoint_sha256':a['checkpoint_sha256'], 'prediction_sha256':a['prediction_sha256'],
                    'audit_sha256':sha(d/'model/fit_audit.json'), 'ordered_trajectory_verified':True})
                old_l, old_u = l, u
    protocol = {'status':'frozen_before_653_to_1005_fits', 'authorization':'User: 都跑到1005把, 2026-10-09',
        'evidence_class':'post-development exploratory extension', 'seeds':SEEDS, 'arms':ARMS,
        'source_budget':653, 'final_budget':1005, 'new_budgets':list(range(685,1006,32)),
        'new_fit_limit':66, 'per_trajectory_fit_limit':11, 'total_extension_fit_limit_from525':90,
        'primary_metric':'normalized validation AULC525-1005',
        'secondary_metrics':['AULC653-1005','AULC333-1005','NRMSE1005','V1/V2 RMSE MAE R2'],
        'original_primary_unchanged':'validation AULC333-525; original gate unchanged',
        'intermediate_window_retained':'AULC525-653 remains a separately frozen descriptive result',
        'stop_rule':'all eight arms and both seeds through1005; no effect-based early stopping',
        'max_concurrent_fits':2, 'threads_per_fit':2, 'paid_llm_call_limit':0, 'new_test_reveals':0,
        'anchor_rule':'bind all six completed653 own states and checkpoints after prior global freeze, before new fits',
        'previous_protocol_sha256':sha(previous.STUDY/'protocol.json'),
        'code_hashes':{str(Path(__file__).relative_to(base.ROOT)):sha(Path(__file__)),
                       'scripts/studies/run_cw_factorial_to1005.py':sha(base.ROOT/'scripts/studies/run_cw_factorial_to1005.py')}}
    previous.save_once(STUDY/'protocol.json', protocol)
    previous.save_once(STUDY/'REUSE_MANIFEST.json', {'entries':controls})
    previous.save_once(STUDY/'RUN_STATE.json', {'status':'planned', 'waiting_for':'six653 trajectories', 'new_fit_limit':66})
    print(json.dumps({'prepared':True, 'reused_control_points':len(controls), 'new_fit_limit':66}), flush=True)


def verify():
    previous.verify()
    p = base.read(STUDY/'protocol.json')
    assert p['previous_protocol_sha256'] == sha(previous.STUDY/'protocol.json')
    for name, h in p['code_hashes'].items():
        assert sha(base.ROOT/name) == h, name
    return p


def bind_anchors():
    p = verify()
    assert base.read(previous.STUDY/'RUN_STATE.json')['status'] == 'completed'
    assert base.read(previous.STUDY/'DELIVERY_AUDIT.json')['verified_fits'] == 24
    records = []
    for seed in SEEDS:
        for arm in ARMS:
            root = previous.STUDY/f'runtime/seed_{seed}/{arm}'
            d = root/'round_10'
            state, a = base.read(d/'state.json'), base.read(d/'model/fit_audit.json')
            assert len(state['labeled']) == 653 and len(state['selected_all']) == 320
            assert sha(d/'model/best.pt') == a['checkpoint_sha256'] == state['checkpoint_sha256']
            assert sha(root/'trajectory_freeze.json') == base.read(previous.STUDY/'GLOBAL_FREEZE.json')['trajectories'][f'{seed}/{arm}']
            records.append({'seed':seed, 'arm':arm, 'state_sha256':sha(d/'state.json'),
                'checkpoint_sha256':a['checkpoint_sha256'], 'audit_sha256':sha(d/'model/fit_audit.json')})
    previous.save_once(STUDY/'ANCHORS.json', {'protocol_hash':stable_hash(p),
        'source_global_freeze_sha256':sha(previous.STUDY/'GLOBAL_FREEZE.json'), 'anchors':records})


def reserve(ledger, key, audit_exists):
    """Interrupted attempts keep their reservation; no silent extra fit."""
    if key not in ledger:
        if len(ledger) >= 11:
            raise RuntimeError('per-trajectory11-fit allowance exhausted')
        ledger[key] = {'status':'running', 'started_unix':time.time()}
    elif ledger[key]['status'] == 'running' and not audit_exists:
        raise RuntimeError('interrupted fit: reservation retained; explicit attempt accounting required')
    elif ledger[key]['status'] == 'completed' and not audit_exists:
        raise RuntimeError('completed fit audit missing; restore evidence rather than refit')
    return ledger


def run_seed(seed):
    torch.set_num_threads(2)
    p = verify()
    anchors = base.read(STUDY/'ANCHORS.json')
    assert anchors['protocol_hash'] == stable_hash(p)
    for arm, (rep, selector) in ARMS.items():
        c = base.Context(seed)
        anchor = previous.STUDY/f'runtime/seed_{seed}/{arm}/round_10'
        saved = next(x for x in anchors['anchors'] if x['seed'] == seed and x['arm'] == arm)
        assert sha(anchor/'state.json') == saved['state_sha256']
        state = base.read(anchor/'state.json')
        l, u, selected = np.array(state['labeled']), np.array(state['unlabeled']), list(state['selected_all'])
        assert c.ids(l) == c.ids(c.roles['l0']) + selected
        assert c.ids(u) == [v for v in c.ids(c.roles['u0']) if v not in set(selected)]
        c.store.freeze_acquisitions(selected)
        truth = np.vstack([c.truth, c.store.reveal(selected,'after_acquisition_fit')])
        checkpoint = anchor/'model/best.pt'
        assert sha(checkpoint) == saved['checkpoint_sha256']
        root = STUDY/f'runtime/seed_{seed}/{arm}'
        root.mkdir(parents=True,exist_ok=True)
        for r in ROUNDS:
            positions = previous.acquisition(c,rep,selector,l,u,checkpoint,root/f'round_{r-1:02d}',p)
            ids = c.ids(u[positions])
            assert len(ids) == len(set(ids)) == 32 and not set(ids)&set(selected)
            selected += ids
            c.store.freeze_acquisitions(selected)
            truth = np.vstack([truth,c.store.reveal(ids,'after_acquisition_fit')])
            l, u = np.r_[l,u[positions]], np.delete(u,positions)
            d = root/f'round_{r:02d}'
            fitdir = d/'model'
            ledger_path = root/'fit_ledger.json'
            ledger = base.read(ledger_path) if ledger_path.exists() else {}
            key = str(r)
            ledger = reserve(ledger,key,(fitdir/'fit_audit.json').exists())
            atomic_json(ledger_path,ledger)
            progress = {'status':'running','seed':seed,'arm':arm,'budget':len(l),'reserved_fits':len(ledger),'limit':11}
            atomic_json(STUDY/f'progress_seed_{seed}.json',progress)
            print(json.dumps(progress),flush=True)
            contract = {'study':STUDY.name,'protocol_hash':stable_hash(p),'outer_seed':seed,'method':arm,'round':r,
                'train_sample_ids':c.ids(l),'validation_sample_ids':c.ids(c.roles['validation']),'test_truth_access_count':0}
            a = fit_from_same_initialization(atom_base=c.atom,angle=c.angle,normalization=c.normalization,preprocessing=c.preprocessing,
                train_indices=l,train_truth=truth,validation_indices=c.roles['validation'],validation_truth=c.validation,
                prediction_indices=c.roles['test'],prediction_sample_ids=c.ids(c.roles['test']),initialization_seed=initialization_seed(seed),
                training_config=c.config,contract=contract,runtime=fitdir)
            assert a['initialization_hash'] == c.anchor_audit['initialization_hash']
            ledger[key] = {'status':'completed','audit_sha256':sha(fitdir/'fit_audit.json'),'training_seconds':a['training_seconds']}
            atomic_json(ledger_path,ledger)
            checkpoint = fitdir/'best.pt'
            previous.save_once(d/'state.json',{'labeled':l.tolist(),'unlabeled':u.tolist(),'selected_all':selected,
                'checkpoint_sha256':a['checkpoint_sha256'],'prediction_sha256':a['prediction_sha256']})
        previous.save_once(root/'trajectory_freeze.json',{'budget':1005,'all_selected_ids':selected,'test_truth_access_count':0})
        atomic_json(root/'label_access_audit.json',{'events':c.store.audit})
    atomic_json(STUDY/f'progress_seed_{seed}.json',{'status':'completed','fits':33,'budget':1005})


def run():
    verify()
    with (STUDY/'execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bind_anchors()
        atomic_json(STUDY/'RUN_STATE.json',{'status':'running','new_fit_limit':66,'max_concurrent_fits':2})
        # Spawn avoids inheriting initialized PyTorch worker threads on macOS.
        import multiprocessing
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(run_seed,SEEDS):
                pass
        previous.save_once(STUDY/'GLOBAL_FREEZE.json',{'status':'complete','test_reveals':0,
            'trajectories':{f'{s}/{a}':sha(STUDY/f'runtime/seed_{s}/{a}/trajectory_freeze.json') for s in SEEDS for a in ARMS}})
        atomic_json(STUDY/'RUN_STATE.json',{'status':'completed','completed_fits':66,'new_fit_limit':66,'paid_llm_calls':0,'test_reveals':0})


def report():
    verify()
    frames = [pd.read_csv(previous.STUDY/'validation_curves.csv')]
    rows = []
    controls = base.read(STUDY/'REUSE_MANIFEST.json')['entries']
    for seed in SEEDS:
        c = base.Context(seed)
        for arm in (*ARMS,*CONTROLS):
            for r in ROUNDS:
                d = STUDY/f'runtime/seed_{seed}/{arm}/round_{r:02d}' if arm in ARMS else control_dir(seed,arm,r)
                path = d/'model/best.pt'
                if not path.with_name('fit_audit.json').exists():
                    continue
                a = base.read(path.with_name('fit_audit.json'))
                assert sha(path) == a['checkpoint_sha256']
                if arm in CONTROLS:
                    assert a['checkpoint_sha256'] == next(x['checkpoint_sha256'] for x in controls if (x['seed'],x['arm'],x['round']) == (seed,arm,r))
                pred, order = predict_outputs(load_predictor_checkpoint(path),c.atom,c.angle,c.roles['validation'])
                assert np.array_equal(order,c.roles['validation'])
                rows.append({'seed':seed,'arm':arm,'budget':333+32*r,'checkpoint_sha256':a['checkpoint_sha256'],
                    **metric_row(c.validation,pred,c.preprocessing['target_scales'])})
    df = pd.concat(frames+[pd.DataFrame(rows)],ignore_index=True)
    df.to_csv(STUDY/'validation_curves.csv',index=False)
    summary = []
    for (seed,arm),g in df.groupby(['seed','arm']):
        g = g.sort_values('budget')
        if g.budget.tolist() != list(range(333,1006,32)):
            continue
        item = {'seed':seed,'arm':arm,'NRMSE1005':float(g.combined_normalized_RMSE.iloc[-1])}
        for start in (333,525,653):
            w = g[g.budget>=start]
            item[f'AULC{start}_1005'] = float(np.trapezoid(w.combined_normalized_RMSE,w.budget)/(1005-start))
        summary.append(item)
    pd.DataFrame(summary).to_csv(STUDY/'validation_summary.csv',index=False)
    df[df.budget == 1005].to_csv(STUDY/'endpoint_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(14,4.8))
    for ax,seed in zip(axes,SEEDS):
        for arm,g in df[df.seed == seed].groupby('arm'):
            ax.plot(g.budget,g.combined_normalized_RMSE,'o-',label=arm,markersize=3)
        for x in (525,653):
            ax.axvline(x,color='black',linestyle=':',alpha=.4)
        ax.set(title=f'Seed {seed} — exploratory extension',xlabel='Active labels',ylabel='Validation NRMSE')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(STUDY/'validation_curves.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'complete_curves':len(summary),'new_test_reveals':0}),flush=True)


def main(action):
    torch.set_num_threads(2)
    try:
        {'prepare':prepare,'run':run,'report':report}[action]()
    except Exception as e:
        atomic_json(STUDY/'BLOCKER.json',{'status':'blocked','error':repr(e),'action':action})
        raise

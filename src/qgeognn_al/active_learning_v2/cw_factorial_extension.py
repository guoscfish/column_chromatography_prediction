"""Bounded post-development continuation; original 525 study stays immutable."""
from __future__ import annotations
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import cw_factorial_study as base
from .cw_factorial import ARMS, SEEDS, select
from .gradient_features import extract_linear_output_gradient_sketches
from .protocol import stable_hash, ids_hash
from .runner import fit_from_same_initialization, predict_outputs, write_predictions
from .benchmark_protocol import initialization_seed, sketch_seed
from .benchmark_reporting import metric_row
from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..training.predictor import atomic_json

STUDY=base.PARENT/'qgeognn_v2_row_cw_selector_extension_to_653'
FINAL_ROUND=10
NEW_ROUNDS=range(7,11)


def save_once(path,value):
    base.once(path,json.loads(json.dumps(value)))


def source_dir(seed,arm,r):
    if arm in ARMS:
        return base.STUDY/f'runtime/seed_{seed}/{arm}/round_{r:02d}'
    if arm=='cw_lcmd' and r>6:
        return base.PARENT/f'qgeognn_v2_row_cw_lcmd_to_653/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
    return base.control_dir(seed,arm,r)


def historic_state(seed,arm,r):
    d=source_dir(seed,arm,r)
    if arm in ('raw_lcmd','random'):
        return np.load(d/'labeled_indices.npy'),np.load(d/'unlabeled_indices.npy')
    if arm=='raw_ivr':
        x=base.read(d/'input.json');return np.array(x['ordered_labeled_indices']),np.array(x['ordered_unlabeled_indices'])
    x=pd.read_csv(d/'state.csv')
    return x.loc[x.role.eq('labeled'),'canonical_index'].to_numpy(int),x.loc[x.role.eq('unlabeled'),'canonical_index'].to_numpy(int)


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    assert base.read(base.STUDY/'DELIVERY_AUDIT.json')['new_fit_count']==36
    anchors=[];controls=[]
    for seed in SEEDS:
        c=base.Context(seed)
        for arm in ARMS:
            d=source_dir(seed,arm,6);state=base.read(d/'state.json');a=base.read(d/'model/fit_audit.json')
            assert len(state['labeled'])==525 and len(state['selected_all'])==192
            assert c.ids(state['labeled'])==c.ids(c.roles['l0'])+state['selected_all']
            assert c.ids(state['unlabeled'])==[x for x in c.ids(c.roles['u0']) if x not in set(state['selected_all'])]
            assert sha256_file(d/'model/best.pt')==a['checkpoint_sha256']==state['checkpoint_sha256']
            anchors.append({'seed':seed,'arm':arm,'state_sha256':sha256_file(d/'state.json'),
                'checkpoint_sha256':a['checkpoint_sha256'],'audit_sha256':sha256_file(d/'model/fit_audit.json')})
        for arm in ('raw_lcmd','raw_maxdet','raw_ivr','cw_lcmd','random'):
            old_l,_=historic_state(seed,arm,6)
            for r in NEW_ROUNDS:
                d=source_dir(seed,arm,r);a=base.read(d/'model/fit_audit.json');cp=torch.load(d/'model/best.pt',weights_only=False)
                assert cp['training_config']==c.config and cp['preprocessing']==c.preprocessing
                assert cp['normalization']==c.old['normalization']
                assert a['initialization_hash']==c.anchor_audit['initialization_hash']
                assert a['train_rows']==333+32*r and a['validation_ids_hash']==c.anchor_audit['validation_ids_hash']
                assert sha256_file(d/'model/best.pt')==a['checkpoint_sha256']
                assert sha256_file(d/'model/predictions.csv.gz')==a['prediction_sha256']
                l,u=historic_state(seed,arm,r)
                assert np.array_equal(l[:len(old_l)],old_l) and len(l)==333+32*r
                assert len(set(l))==len(l) and not set(l)&set(u) and u.tolist()==sorted(u)
                assert set(l)|set(u)==set(c.roles['l0'])|set(c.roles['u0'])
                assert a['train_ids_hash']==ids_hash(c.ids(l))
                prev=source_dir(seed,arm,r-1)
                if arm=='cw_lcmd' and r==7: prev=source_dir(seed,arm,7).parent/'round_06'
                if arm=='raw_ivr': ids=base.read(prev/'selection.json')['selected_ids']
                else:
                    folder='acquisition' if arm=='raw_maxdet' else 'acquisition_artifacts'
                    ids=pd.read_csv(prev/folder/'selected_next_batch.csv').sample_id.astype(str).tolist()
                assert ids==c.ids(l[-32:])
                controls.append({'seed':seed,'arm':arm,'round':r,'budget':333+32*r,'reused':True,
                    'checkpoint':str((d/'model/best.pt').relative_to(base.ROOT)),
                    'checkpoint_sha256':a['checkpoint_sha256'],'audit_sha256':sha256_file(d/'model/fit_audit.json'),
                    'prediction_sha256':a['prediction_sha256'],'ordered_trajectory_verified':True})
                old_l=l
    code_paths=[*base.read(base.STUDY/'execution_code_freeze.json'),str(Path(__file__).relative_to(base.ROOT)),
                'scripts/studies/run_cw_factorial_extension.py']
    protocol={'status':'frozen_before_extension_fits','evidence_class':'post-development exploratory extension authorized 2026-10-09',
        'seeds':SEEDS,'arms':ARMS,'source_budget':525,'final_budget':653,'new_budgets':[557,589,621,653],
        'new_fit_limit':24,'original_primary_metric':'validation AULC333-525 remains unchanged',
        'extension_primary':'validation normalized AULC525-653','extension_secondary':['AULC333-653','NRMSE653','V1/V2 RMSE MAE R2'],
        'stop_rule':'all six trajectories complete four additional rounds then stop regardless of effect',
        'paid_llm_call_limit':0,'new_test_reveals':0,'no_new_confirmatory_claims':True,
        'source_protocol_sha256':sha256_file(base.STUDY/'protocol.json'),
        'scrubbed_graph_hashes':{str(seed):sha256_file(base.BASE/f'runtime/seed_{seed}/scrubbed_graphs.pt') for seed in SEEDS},
        'source_global_freeze_sha256':sha256_file(base.STUDY/'GLOBAL_MATRIX_FREEZE.json'),
        'code_hashes':{name:sha256_file(base.ROOT/name) for name in code_paths},'anchors':anchors}
    save_once(STUDY/'protocol.json',protocol)
    save_once(STUDY/'REUSE_MANIFEST.json',{'entries':controls})
    save_once(STUDY/'RUN_STATE.json',{'status':'planned','new_fit_limit':24,'paid_llm_calls':0,'test_reveals':0})
    print(json.dumps({'prepared':True,'anchors':len(anchors),'reused_control_points':len(controls)}),flush=True)


def verify():
    p=base.read(STUDY/'protocol.json')
    assert p['source_protocol_sha256']==sha256_file(base.STUDY/'protocol.json')
    assert p['source_global_freeze_sha256']==sha256_file(base.STUDY/'GLOBAL_MATRIX_FREEZE.json')
    for name,h in p['code_hashes'].items():assert sha256_file(base.ROOT/name)==h,name
    for seed,h in p['scrubbed_graph_hashes'].items():assert sha256_file(base.BASE/f'runtime/seed_{seed}/scrubbed_graphs.pt')==h
    return p


def acquisition(c,rep,selector,l,u,checkpoint,d,protocol):
    """Freeze inputs, selected IDs and predictions; no label-store access."""
    d.mkdir(parents=True,exist_ok=True)
    contract={'checkpoint_sha256':sha256_file(checkpoint),'ordered_l':c.ids(l),'ordered_u':c.ids(u),
              'T':c.transforms[rep].tolist(),'dimension':512,'sketch_seed':sketch_seed(c.seed),'protocol_hash':stable_hash(protocol)}
    path=d/'batch_freeze.json'
    if path.exists():
        f=base.read(path);assert f['contract']==contract
        assert sha256_file(d/'premeasurement.csv.gz')==f['premeasurement_sha256']
        positions=np.array(f['positions'],dtype=int)
        assert c.ids(u[positions])==f['ids']
        return positions
    started=time.perf_counter();model=load_predictor_checkpoint(checkpoint)
    bank=extract_linear_output_gradient_sketches(model,c.atom,c.angle,np.r_[l,u],c.transforms[rep],dimension=512,sketch_seed=sketch_seed(c.seed))
    positions,trace=select(bank.features,len(l),selector)
    pred,order=predict_outputs(model,c.atom,c.angle,u[positions]);assert np.array_equal(order,u[positions])
    digest=write_predictions(d/'premeasurement.csv.gz',c.ids(u[positions]),pred)
    atomic_json(d/'selection_audit.json',{'gradient':bank.audit,'trace':trace,'seconds':time.perf_counter()-started})
    save_once(path,{'contract':contract,'positions':positions.tolist(),'ids':c.ids(u[positions]),'premeasurement_sha256':digest})
    return positions


def run_trajectory(seed,arm,p):
    c=base.Context(seed);rep,selector=ARMS[arm];anchor=source_dir(seed,arm,6)
    saved=next(x for x in p['anchors'] if x['seed']==seed and x['arm']==arm)
    assert sha256_file(anchor/'state.json')==saved['state_sha256']
    state=base.read(anchor/'state.json');l=np.array(state['labeled']);u=np.array(state['unlabeled']);selected=list(state['selected_all'])
    c.store.freeze_acquisitions(selected)
    truth=np.vstack([c.truth,c.store.reveal(selected,'after_acquisition_fit')])
    checkpoint=anchor/'model/best.pt';assert sha256_file(checkpoint)==saved['checkpoint_sha256']
    root=STUDY/f'runtime/seed_{seed}/{arm}'
    for r in NEW_ROUNDS:
        positions=acquisition(c,rep,selector,l,u,checkpoint,root/f'round_{r-1:02d}',p)
        ids=c.ids(u[positions]);assert len(ids)==32 and len(set(ids))==32 and not set(ids)&set(selected)
        selected+=ids;c.store.freeze_acquisitions(selected)
        truth=np.vstack([truth,c.store.reveal(ids,'after_acquisition_fit')])
        l=np.r_[l,u[positions]];u=np.delete(u,positions)
        assert len(l)==333+32*r
        d=root/f'round_{r:02d}';fitdir=d/'model'
        ledger_path=STUDY/'fit_ledger.json';ledger=base.read(ledger_path) if ledger_path.exists() else {};key=f'{seed}/{arm}/{r}'
        if key not in ledger:
            if len(ledger)>=p['new_fit_limit']:raise RuntimeError('extension fit quota exhausted')
            ledger[key]={'status':'running','started_unix':time.time()};atomic_json(ledger_path,ledger)
        elif ledger[key]['status']=='running' and not (fitdir/'fit_audit.json').exists():
            raise RuntimeError('interrupted fit retains reservation; explicit attempt accounting required')
        progress={'status':'running','seed':seed,'arm':arm,'budget':len(l),'reserved_fits':len(ledger),'new_fit_limit':24,'paid_llm_calls':0,'test_reveals':0}
        atomic_json(STUDY/'RUN_STATE.json',progress);print(json.dumps(progress),flush=True)
        contract={'study':STUDY.name,'protocol_hash':stable_hash(p),'outer_seed':seed,'method':arm,'round':r,
                  'train_sample_ids':c.ids(l),'validation_sample_ids':c.ids(c.roles['validation']),'test_truth_access_count':0}
        a=fit_from_same_initialization(atom_base=c.atom,angle=c.angle,normalization=c.normalization,preprocessing=c.preprocessing,
            train_indices=l,train_truth=truth,validation_indices=c.roles['validation'],validation_truth=c.validation,
            prediction_indices=c.roles['test'],prediction_sample_ids=c.ids(c.roles['test']),initialization_seed=initialization_seed(seed),
            training_config=c.config,contract=contract,runtime=fitdir)
        assert a['initialization_hash']==c.anchor_audit['initialization_hash']
        ledger[key]={'status':'completed','audit_sha256':sha256_file(fitdir/'fit_audit.json'),'training_seconds':a['training_seconds']}
        atomic_json(ledger_path,ledger);checkpoint=fitdir/'best.pt'
        save_once(d/'state.json',{'labeled':l.tolist(),'unlabeled':u.tolist(),'selected_all':selected,
            'checkpoint_sha256':a['checkpoint_sha256'],'prediction_sha256':a['prediction_sha256']})
    save_once(root/'trajectory_freeze.json',{'budget':653,'all_selected_ids':selected,'test_truth_access_count':0})
    atomic_json(root/'label_access_audit.json',{'events':c.store.audit})


def run():
    p=verify()
    with (STUDY/'execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in ARMS:
            for seed in SEEDS:run_trajectory(seed,arm,p)
        save_once(STUDY/'GLOBAL_FREEZE.json',{'status':'complete','test_reveals':0,
            'trajectories':{f'{s}/{a}':sha256_file(STUDY/f'runtime/seed_{s}/{a}/trajectory_freeze.json') for s in SEEDS for a in ARMS}})
        atomic_json(STUDY/'RUN_STATE.json',{'status':'completed','completed_fits':24,'new_fit_limit':24,'paid_llm_calls':0,'test_reveals':0})


def report():
    verify();frames=[pd.read_csv(base.STUDY/'validation_curves.csv')];rows=[]
    controls=base.read(STUDY/'REUSE_MANIFEST.json')['entries']
    for seed in SEEDS:
        c=base.Context(seed)
        for arm in (*ARMS,'raw_lcmd','raw_maxdet','raw_ivr','cw_lcmd','random'):
            for r in NEW_ROUNDS:
                path=STUDY/f'runtime/seed_{seed}/{arm}/round_{r:02d}/model/best.pt' if arm in ARMS else source_dir(seed,arm,r)/'model/best.pt'
                if not path.with_name('fit_audit.json').exists():continue
                a=base.read(path.with_name('fit_audit.json'));assert sha256_file(path)==a['checkpoint_sha256']
                if arm not in ARMS:assert a['checkpoint_sha256']==next(x['checkpoint_sha256'] for x in controls if (x['seed'],x['arm'],x['round'])==(seed,arm,r))
                pred,order=predict_outputs(load_predictor_checkpoint(path),c.atom,c.angle,c.roles['validation'])
                assert np.array_equal(order,c.roles['validation'])
                rows.append({'seed':seed,'arm':arm,'budget':333+32*r,'checkpoint_sha256':a['checkpoint_sha256'],**metric_row(c.validation,pred,c.preprocessing['target_scales'])})
    df=pd.concat(frames+[pd.DataFrame(rows)],ignore_index=True);df.to_csv(STUDY/'validation_curves.csv',index=False)
    summary=[]
    for (seed,arm),g in df.groupby(['seed','arm']):
        g=g.sort_values('budget')
        if g.budget.tolist()!=list(range(333,654,32)):continue
        late=g[g.budget>=525]
        summary.append({'seed':seed,'arm':arm,'AULC525_653':float(np.trapezoid(late.combined_normalized_RMSE,late.budget)/128),
            'AULC333_653':float(np.trapezoid(g.combined_normalized_RMSE,g.budget)/320),'NRMSE653':float(g.combined_normalized_RMSE.iloc[-1])})
    pd.DataFrame(summary).to_csv(STUDY/'validation_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,seed in zip(axes,SEEDS):
        for arm,g in df[df.seed==seed].groupby('arm'):ax.plot(g.budget,g.combined_normalized_RMSE,'o-',label=arm,markersize=3)
        ax.axvline(525,color='black',linestyle=':',alpha=.5);ax.set(title=f'Seed {seed} — exploratory extension',xlabel='Active labels',ylabel='Validation NRMSE');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(STUDY/'validation_curves.png',dpi=160);plt.close(fig)
    print(json.dumps({'complete_curves':len(summary),'new_test_reveals':0}),flush=True)


def main(action):
    torch.set_num_threads(2)
    try: {'prepare':prepare,'run':run,'report':report}[action]()
    except Exception as e:
        atomic_json(STUDY/'BLOCKER.json',{'status':'blocked','error':repr(e),'action':action});raise

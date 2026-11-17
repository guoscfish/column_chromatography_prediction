#!/usr/bin/env python3
"""Project only historical L1005 and original validation targets into a new study."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/volume_controls_mpl')
from pathlib import Path
import hashlib,json,lzma,sys,subprocess
import numpy as np
import pandas as pd
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT/'src'))
from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore,stable_hash,ids_hash
from qgeognn_al.active_learning_v2.benchmark_protocol import FEATURE_COLUMNS,seed_config,initialization_seed
from qgeognn_al.active_learning_v2.gradient_features import state_dict_hash
from qgeognn_al.training.predictor import seed_everything,loader_pair
from qgeognn_al.schemas.conditions import ConditionNormalization
from isolated_models import build_control
SEEDS=[157,887,2357,6101,12203]
FILES={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):
    p=Path(p);b=p.read_bytes();key=str(p.relative_to(ROOT));entry=dict(path=key,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    assert key not in FILES or FILES[key]==entry;FILES[key]=entry;return b
def load(p):return json.loads(read(p))
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def load_seed(seed):
    d=load(OUT/f'data/seed_{seed}/identity.json')
    g=ROOT/d['graphs_path'];assert sha(g)==d['graphs_sha256']
    atom,angle=torch.load(g,map_location='cpu',weights_only=False)
    if any(torch.count_nonzero(a.y) for a in atom):raise ValueError('Cached base graphs contain targets')
    meta=pd.read_csv(OUT/f'data/seed_{seed}/metadata.csv',dtype={'sample_id':str})
    assert sha(OUT/f'data/seed_{seed}/metadata.csv')==d['metadata_sha256']
    for i,a in enumerate(atom):
        assert int(a.canonical_position)==i
        a.sample_id=str(meta.iloc[i].sample_id)
        a.sample_volume_ul=torch.tensor([float(meta.iloc[i]['V/ul'])],dtype=torch.float32)
    assert sha(OUT/f'data/seed_{seed}/authorized_labels.npz')==d['authorized_labels_sha256']
    values=np.load(OUT/f'data/seed_{seed}/authorized_labels.npz',allow_pickle=False)
    return d,atom,angle,values

def assert_identity(batch,ids,volume):
    pos=batch.canonical_position.detach().cpu().numpy().astype(int).reshape(-1)
    if list(batch.sample_id)!=[ids[i] for i in pos]:raise ValueError('Stable ID and graph batch disagree')
    raw=batch.sample_volume_ul.detach().cpu().numpy().reshape(-1)
    if not np.array_equal(raw,np.asarray(volume,dtype=np.float32)[pos]):raise ValueError('Raw volume and graph batch disagree')

def main():
    torch.set_num_threads(2)
    previous=ROOT/'studies/active_learning/decision_mechanism_followup_20261117'
    r=subprocess.run([sys.executable,'-B',str(previous/'run.py'),'--check'],cwd=ROOT,capture_output=True,text=True,check=True)
    snapshot=json.loads(lzma.decompress(read(ROOT/'studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz')))
    hold=pd.read_csv(previous/'model_control_holdouts.csv');read(previous/'model_control_holdouts.csv')
    source=ROOT/'experiments/e0_4g_baseline/canonical_4g.csv'
    # The parser returns covariates only; numeric targets are never loaded here.
    metadata=pd.read_csv(source,usecols=list(FEATURE_COLUMNS)).reset_index(drop=True)
    source_sha=sha(source)
    checks=[];plans=[]
    for seed in SEEDS:
        dest=OUT/f'data/seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        root=ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}'
        context=load(root/'context.json')['contract'];assert context['source_sha256']==source_sha
        splitpath=ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv'
        part=pd.read_csv(splitpath);read(splitpath)
        assert part.sample_id.tolist()==metadata.sample_id.tolist()
        assert part.canonical_smiles.tolist()==metadata.canonical_smiles.tolist()
        state=next(b for b in snapshot['states'] if b['method']=='new_free_cw_sol61_xhigh' and b['seed']==seed and b['L_before']==1005)
        base=(ROOT/state['source']).parent;f=load(base/'batch_freeze.json');assert sha(base/'batch_freeze.json')==state['source_sha256']
        contract=load(base/'contract.json');assert sha(base/'contract.json')==f['artifact_hashes']['contract.json']
        catbytes=read(base/'catalog.json');assert hashlib.sha256(catbytes).hexdigest()==contract['catalog_sha256']
        catalog=json.loads(catbytes);observed={v['id']:v for v in catalog['observed']}
        assert set(observed)==set(contract['L_t_ids'])==set(state['labeled_ids']) and len(observed)==1005
        planned=hold[(hold.seed==seed)&(hold.budget==1005)];assert len(planned)==10
        check_ids=sorted({v for text in planned.ids for v in text.split('|')})
        history=state['labeled_ids'];assert set(check_ids)<=set(history)
        l0=set(part.loc[part.role=='l0','sample_id']);assert not l0&set(check_ids)
        # Full response-series isolation was fixed in the prior metadata-only plan.
        train_ids=[v for v in history if v not in set(check_ids)]
        validation_ids=part.loc[part.role=='validation','sample_id'].tolist()
        positions={sid:i for i,sid in enumerate(metadata.sample_id)}
        history_y=np.array([[observed[sid]['V1_ml'],observed[sid]['V2_ml']] for sid in history],dtype=np.float32)
        store=RestrictedLabelStore(source,part)
        validation_y=store.reveal(validation_ids,'initial_fit')
        assert all(a['roles']=='validation' for a in store.audit)
        ymap=dict(zip(history,history_y));train_y=np.array([ymap[v] for v in train_ids],dtype=np.float32);check_y=np.array([ymap[v] for v in check_ids],dtype=np.float32)
        outer=part.loc[part.role.isin(['l0','u0']),'canonical_index'].to_numpy(int)
        volumes=metadata.iloc[outer]['V/ul'].to_numpy(np.float32);vscale=dict(min=float(volumes.min()),max=float(volumes.max()),role='original outer-train covariates only',unit='uL')
        graph=root/'scrubbed_graphs.pt';gc=load(root/'scrubbed_graphs.pt.contract.json')
        assert gc['contract']==context and sha(graph)==gc['sha256']
        config=seed_config(seed);config['split_sha256']=stable_hash(part.to_dict('list'))
        d=dict(seed=seed,history_budget=1005,history_ids=history,train_ids=train_ids,check_ids=check_ids,validation_ids=validation_ids,
            train_indices=[positions[v] for v in train_ids],check_indices=[positions[v] for v in check_ids],validation_indices=[positions[v] for v in validation_ids],
            ordered_ids=metadata.sample_id.tolist(),normalization=context['normalization'],preprocessing=context['preprocessing'],volume_scale=vscale,
            training_config=config,initialization_seed=initialization_seed(seed),graphs_path=str(graph.relative_to(ROOT)),graphs_sha256=gc['sha256'],
            label_access=store.audit,validation_was_original_shared_observed_role=True,unobserved_or_test_target_values_projected=0,
            train_ids_hash=ids_hash(train_ids),check_ids_hash=ids_hash(check_ids),validation_ids_hash=ids_hash(validation_ids),
            historical_checkpoint_used_for_training=False)
        metadata.to_csv(dest/'metadata.csv',index=False)
        np.savez_compressed(dest/'authorized_labels.npz',train_y=train_y,check_y=check_y,validation_y=validation_y)
        d['metadata_sha256']=sha(dest/'metadata.csv');d['authorized_labels_sha256']=sha(dest/'authorized_labels.npz')
        dump(dest/'identity.json',d)
        dd,atoms,angles,values=load_seed(seed)
        norm=ConditionNormalization(**d['normalization']);initials={};rngs={};outputs={}
        indices=(d['train_indices'][:8]+d['validation_indices'][:8]+d['check_indices'][:4])
        for kind in ['A_original','B_amount_skip','C_volume_skip']:
            seed_everything(d['initialization_seed']);model=build_control(kind,norm,vscale['min'],vscale['max']);model.eval()
            common={k:v for k,v in model.state_dict().items() if k not in ['skip_weight','sample_volume_min','sample_volume_range']}
            h=hashlib.sha256()
            for k,v in common.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            initials[kind]=h.hexdigest();rngs[kind]=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            with torch.no_grad():
                a,b=next(iter(zip(*loader_pair(atoms,angles,indices,2048))));assert_identity(a,d['ordered_ids'],metadata['V/ul'])
                outputs[kind]=model(a,b).detach().cpu().numpy()
            del model
        assert len(set(initials.values()))==len(set(rngs.values()))==1
        assert np.array_equal(outputs['A_original'],outputs['B_amount_skip']) and np.array_equal(outputs['A_original'],outputs['C_volume_skip'])
        checks.append(dict(seed=seed,train_n=len(train_ids),check_n=len(check_ids),validation_n=len(validation_ids),common_initial_hash=initials,
            post_init_torch_rng_hash=rngs,zero_skip_max_output_difference=0,blind_graphs_all_y_zero=True,stable_batch_identity=True,
            actual_volume_range_ul=vscale,validation_label_access=store.audit,data_identity_sha256=sha(dest/'identity.json')))
        for kind in ['A_original','B_amount_skip','C_volume_skip']:plans.append(dict(seed=seed,kind=kind,initialization_seed=d['initialization_seed'],training_seed=d['training_config']['seed']))
        print('DATA_GATE',seed,len(train_ids),len(check_ids),'zero skip exact',flush=True)
        del atoms,angles
    gate=dict(status='PASS',pretraining_checks=checks,planned_fits=plans,maximum_new_fits=15,hyperparameter_search=False,
        previous_seal_check=r.stdout.strip(),inputs=list(FILES.values()),canonical_covariate_source_sha256=source_sha,
        training_code_sha256={n:sha(OUT/n) for n in ['prepare_data.py','isolated_models.py','train_controls.py']},
        label_policy='Historical H1005 from its own observed catalog; original validation only via role-gated streaming numeric projection; no test/unrevealed target values converted')
    dump(OUT/'DATA_FEASIBILITY.json',gate)
    print('ALL_PRETRAINING_GATES_PASS',len(plans),flush=True)

if __name__=='__main__':main()

"""Frozen, bounded factorial study. No LLM transport or test-label evaluation."""
from __future__ import annotations
import json
import fcntl
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json, loader_pair
from .benchmark_protocol import load_features, seed_config, initialization_seed, sketch_seed, TRAINING_CONFIG, package_versions
from .benchmark_reporting import metric_row
from .protocol import RestrictedLabelStore, stable_hash, ids_hash, validate_row_protocol
from .runner import fit_from_same_initialization, predict_outputs, write_predictions
from .gradient_features import extract_linear_output_gradient_sketches, countsketch_mapping, state_dict_hash
from .cw_factorial import SEEDS, BUDGETS, ARMS, PROJECTION_SEEDS, DIMENSIONS, transforms, select

STUDY = ROOT / 'studies/active_learning/qgeognn_v2_row_cw_selector_factorial'
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
PARENT = BASE.parent

def read(path):
    return json.loads(Path(path).read_text())

def once(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'frozen artifact drift: {path}')
    else:
        atomic_json(path, value)

def status(stage, value, **extra):
    path = STUDY / 'RUN_STATE.json'
    state = read(path) if path.exists() else {'new_fits_limit': 36, 'paid_llm_calls': 0, 'test_truth_access_count': 0}
    state[stage] = {'status': value, **extra}
    atomic_json(path, state)
    print(json.dumps({'stage': stage, 'status': value, **extra}), flush=True)

class Context:
    def __init__(self, seed):
        self.seed = seed
        base = BASE / f'runtime/seed_{seed}'
        self.partition = pd.read_csv(BASE / f'splits/row_seed_{seed}.csv')
        validate_row_protocol(self.partition)
        self.data = load_features()
        assert self.partition.sample_id.tolist() == self.data.sample_id.tolist()
        self.roles = {r: self.partition.loc[self.partition.role.eq(r), 'canonical_index'].to_numpy(int)
                      for r in ('l0', 'u0', 'validation', 'test')}
        self.old = read(base / 'context.json')['contract']
        assert self.old['source_sha256'] == sha256_file(SOURCE_DATA)
        assert self.old['source_graph_cache_sha256'] == sha256_file(SOURCE_GRAPH_CACHE)
        for role in self.roles:
            assert self.old[f'{role}_ids_hash'] == ids_hash(self.ids(self.roles[role]))
        self.preprocessing = self.old['preprocessing']
        self.normalization = ConditionNormalization(**self.old['normalization'])
        self.atom, self.angle = torch.load(base / 'scrubbed_graphs.pt', weights_only=False)
        assert all(not torch.count_nonzero(g.y) for g in self.atom)
        self.store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.truth = self.store.reveal(self.ids(self.roles['l0']), 'initial_fit')
        self.validation = self.store.reveal(self.ids(self.roles['validation']), 'initial_fit')
        s = self.preprocessing['target_scales']
        self.transforms, self.transform_audit = transforms(self.truth, (s['V1'], s['V2']))
        self.anchor = base / 'lcmd/round_00/model/best.pt'
        self.anchor_audit = read(self.anchor.with_name('fit_audit.json'))
        assert sha256_file(self.anchor) == self.anchor_audit['checkpoint_sha256']
        self.config = seed_config(seed)
        self.config['split_sha256'] = stable_hash(self.partition.to_dict('list'))

    def ids(self, idx):
        return self.data.iloc[list(idx)].sample_id.astype(str).tolist()


def control_dir(seed, arm, r):
    if r == 0:
        return BASE / f'runtime/seed_{seed}/lcmd/round_00'
    if arm in ('raw_lcmd', 'random'):
        return BASE / f'runtime/seed_{seed}/{"lcmd" if arm == "raw_lcmd" else "random"}/round_{r:02d}'
    if arm == 'raw_maxdet':
        return PARENT / f'qgeognn_v2_row_maxdet_b32/runtime/seed_{seed}/gradient_maxdet/round_{r:02d}'
    if arm == 'raw_ivr':
        return PARENT / f'qgeognn_v2_row_kernel_ivr_b32/runtime/seed_{seed}/round_{r:02d}'
    if arm == 'cw_lcmd':
        study = 'short_sequential_b32' if r <= 3 else 'cw_lcmd_extension_to_525'
        # Extension name is resolved from its retained source constant.
        if r > 3:
            from .cw_lcmd_extension_study import STUDY as extension
            return extension / f'runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
        return PARENT / f'qgeognn_v2_row_{study}/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
    raise ValueError(arm)


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    controls, contexts = [], {}
    for seed in SEEDS:
        c = Context(seed)
        contexts[str(seed)] = {'split_sha256': sha256_file(BASE / f'splits/row_seed_{seed}.csv'),
            'context_sha256': sha256_file(BASE / f'runtime/seed_{seed}/context.json'),
            'anchor_sha256': sha256_file(c.anchor), 'initialization_hash': c.anchor_audit['initialization_hash'],
            'target_scales': c.preprocessing['target_scales'],
            'transforms': {k: v.tolist() for k, v in c.transforms.items()},
            'transform_audit': c.transform_audit, 'training_config': c.config,
            'sketch_seed': sketch_seed(seed)}
        for arm in ('raw_lcmd', 'raw_maxdet', 'raw_ivr', 'cw_lcmd', 'random'):
            for r in range(7):
                d = control_dir(seed, arm, r)
                model = d / 'model/best.pt'
                if arm == 'cw_lcmd' and r <= 3 and r > 0:
                    model = Path(read(d / 'round_freeze.json')['checkpoint_path'])
                a = read(model.with_name('fit_audit.json'))
                assert sha256_file(model) == a['checkpoint_sha256']
                assert sha256_file(model.with_name('predictions.csv.gz')) == a['prediction_sha256']
                assert a['initialization_hash'] == c.anchor_audit['initialization_hash']
                assert a['initialization_seed'] == initialization_seed(seed)
                assert a['train_rows'] == BUDGETS[r]
                assert a['validation_ids_hash'] == c.anchor_audit['validation_ids_hash']
                assert a['test_labels_used_for_fit_or_checkpoint_selection'] == 0
                controls.append({'seed': seed, 'arm': arm, 'round': r, 'budget': BUDGETS[r],
                    'reused': True, 'checkpoint': str(model.relative_to(ROOT)),
                    'checkpoint_sha256': a['checkpoint_sha256'], 'audit_sha256': sha256_file(model.with_name('fit_audit.json')),
                    'prediction_sha256': a['prediction_sha256'], 'directory': str(d.relative_to(ROOT))})
    record = {'seeds': SEEDS, 'budgets': BUDGETS, 'new_arms': ARMS, 'maximum_new_fits': 36,
        'source_sha256': sha256_file(SOURCE_DATA), 'graph_sha256': sha256_file(SOURCE_GRAPH_CACHE),
        'training': TRAINING_CONFIG, 'contexts': contexts, 'production_sketch_dimension': 512,
        'projection_seeds': PROJECTION_SEEDS, 'diagnostic_dimensions': DIMENSIONS,
        'normalization': {'lcmd': 'none', 'maxdet': 'current_L_RMS', 'ivr': 'fixed_L_plus_U_RMS'},
        'primary': 'validation normalized trapezoidal AULC333-525', 'test_barrier': 'entire matrix frozen before any new test evaluation',
        'threshold': {'mean_relative_AULC_improvement': .02, 'both_seed_directions': 'negative'},
        'packages': package_versions(), 'paid_llm_calls_limit': 0}
    # JSON canonicalization makes tuple/list semantics identical on resume.
    once(STUDY / 'protocol.json', json.loads(json.dumps(record)))
    once(STUDY / 'REUSE_MANIFEST.json', {'entries': controls, 'test_exposure': 'historical; not blind'})
    status('inventory', 'completed', reused_points=len(controls))


def audit_reuse():
    """Check numerical training configuration, ordered lineage and feature provenance."""
    records=[]
    for seed in SEEDS:
        c=Context(seed)
        for arm in ('raw_lcmd','raw_maxdet','raw_ivr','cw_lcmd','random'):
            previous=c.roles['l0'].tolist()
            for r in range(7):
                d=control_dir(seed,arm,r)
                item=next(x for x in read(STUDY/'REUSE_MANIFEST.json')['entries'] if x['seed']==seed and x['arm']==arm and x['round']==r)
                model=ROOT/item['checkpoint']
                cp=torch.load(model,weights_only=False)
                assert cp['training_config']==c.config, (arm,r,'training drift')
                assert cp['preprocessing']==c.preprocessing
                assert cp['normalization']==c.old['normalization']
                if r==0: l,u=c.roles['l0'],c.roles['u0']
                elif arm in ('raw_lcmd','random'):
                    l=np.load(d/'labeled_indices.npy');u=np.load(d/'unlabeled_indices.npy')
                elif arm=='raw_ivr':
                    inp=read(d/'input.json');l=np.array(inp['ordered_labeled_indices']);u=np.array(inp['ordered_unlabeled_indices'])
                else:
                    state=pd.read_csv(d/'state.csv')
                    l=state.loc[state.role.eq('labeled'),'canonical_index'].to_numpy(int)
                    u=state.loc[state.role.eq('unlabeled'),'canonical_index'].to_numpy(int)
                assert l.tolist()[:len(previous)]==previous
                assert len(set(l))==len(l) and not set(l)&set(u)
                assert sorted(u)==u.tolist()
                assert set(l)|set(u)==set(c.roles['l0'])|set(c.roles['u0'])
                a=read(model.with_name('fit_audit.json'))
                assert a['train_ids_hash']==ids_hash(c.ids(l))
                if r>0:
                    if arm=='raw_ivr':
                        prev=PARENT/f'qgeognn_v2_row_kernel_ivr_b32/runtime/seed_{seed}/round_{r-1:02d}'
                        selected=read(prev/'selection.json')['selected_ids']
                    else:
                        prev=control_dir(seed,arm,r-1)
                        if arm=='cw_lcmd' and r==1:
                            prev=PARENT/f'qgeognn_v2_row_short_sequential_b32/runtime/seed_{seed}/center_width_lcmd/round_00'
                        if arm=='raw_maxdet' and r==1:
                            prev=PARENT/f'qgeognn_v2_row_maxdet_b32/runtime/seed_{seed}/gradient_maxdet/round_00'
                        if arm=='random' and r==1:
                            prev=BASE/f'runtime/seed_{seed}/random/round_00'
                        if arm=='cw_lcmd' and r==4:
                            from .cw_lcmd_extension_study import STUDY as extension
                            prev=extension/f'runtime/seed_{seed}/center_width_lcmd/round_03'
                        folder='acquisition' if arm=='raw_maxdet' else 'acquisition_artifacts'
                        selected=pd.read_csv(prev/folder/'selected_next_batch.csv').sample_id.astype(str).tolist()
                    assert c.ids(l[-32:])==selected, (arm,r,'trajectory mismatch')
                previous=l.tolist()
                records.append({'seed':seed,'arm':arm,'round':r,'training_config_match':True,
                    'preprocessing_match':True,'normalization_match':True,'ordered_trajectory_match':True,
                    'checkpoint_sha256':sha256_file(model),'train_ids_hash':a['train_ids_hash']})
    once(STUDY/'reuse_compatibility.json',{'entries':records})


def geometry_audit():
    audit_reuse()
    rows, regressions = [], []
    for seed in SEEDS:
        c = Context(seed)
        model = load_predictor_checkpoint(c.anchor)
        before = state_dict_hash(model)
        # 8 rows (4 L, 4 U), ~59 MB float64 task Jacobians. Never saved.
        indices = np.r_[c.roles['l0'][:4], c.roles['u0'][:4]]
        pars = tuple(p for p in model.parameters() if p.requires_grad)
        js = []
        model.eval()
        for a, b in zip(*loader_pair(c.atom, c.angle, indices, 1)):
            out = model(a, b)
            tasks = []
            for col in (1, 4):
                grad = torch.autograd.grad(out[0, col], pars, retain_graph=col == 1, allow_unused=True)
                tasks.append(torch.cat([torch.zeros(p.numel()) if g is None else g.detach().reshape(-1).cpu()
                                        for p, g in zip(pars, grad)]).numpy().astype(np.float64))
            js.append(tasks)
        j = np.asarray(js)
        f = {k: np.einsum('ab,nbp->nap', t, j).reshape(len(j), -1) for k, t in c.transforms.items()}
        grams = {k: x @ x.T for k, x in f.items()}
        for a, b in [('raw', 'rotation'), ('cw', 'matched')]:
            error = float(np.linalg.norm(grams[a]-grams[b])/np.linalg.norm(grams[a]))
            assert error < 1e-10
            rows.append({'seed': seed, 'kind': 'uncompressed_isometry', 'representation': a+'_'+b, 'dimension': j.shape[-1]*2, 'projection_seed': -1, 'relative_error': error})
        for dim in DIMENSIONS:
            for ps in PROJECTION_SEEDS:
                bucket, sign = countsketch_mapping(j.shape[-1], dim, ps)
                sk = {k: np.vstack([np.bincount(bucket.numpy(), weights=x*sign.numpy(), minlength=dim) for x in v]) for k,v in f.items()}
                for k, x in sk.items():
                    rows.append({'seed': seed, 'kind': 'production_Gram_error', 'representation': k, 'dimension': dim,
                        'projection_seed': ps, 'relative_error': float(np.linalg.norm(x@x.T-grams[k])/np.linalg.norm(grams[k]))})
                for a,b in [('raw','rotation'),('cw','matched')]:
                    rows.append({'seed': seed, 'kind': 'production_pair_discrepancy', 'representation': a+'_'+b, 'dimension': dim,
                        'projection_seed': ps, 'relative_error': float(np.linalg.norm(sk[a]@sk[a].T-sk[b]@sk[b].T)/np.linalg.norm(grams[a]))})
                # Same S for each task; half dimension per block. Diagnostic only.
                bucket, sign = countsketch_mapping(j.shape[-1], dim//2, ps)
                bucket, sign = bucket.numpy()[:j.shape[-1]], sign.numpy()[:j.shape[-1]]
                projected = np.array([[np.bincount(bucket, weights=g*sign, minlength=dim//2) for g in pair] for pair in j])
                blocks = {k: np.einsum('ab,nbp->nap', t, projected).reshape(len(j),-1) for k,t in c.transforms.items()}
                for a,b in [('raw','rotation'),('cw','matched')]:
                    error = float(np.linalg.norm(blocks[a]@blocks[a].T-blocks[b]@blocks[b].T)/np.linalg.norm(blocks[a]@blocks[a].T))
                    assert error < 1e-10
                    rows.append({'seed':seed,'kind':'shared_block_isometry','representation':a+'_'+b,'dimension':dim,'projection_seed':ps,'relative_error':error})
        assert before == state_dict_hash(model)
        del j, js, f, sk, blocks
        # Replay historical full L333 feature banks with unchanged selectors.
        for rep, path, key in [
            ('raw', BASE / f'runtime/seed_{seed}/lcmd/round_00/acquisition_artifacts', 'current_gradient_features.npz'),
            ('cw', PARENT / f'qgeognn_v2_row_short_sequential_b32/runtime/seed_{seed}/center_width_lcmd/round_00/acquisition_artifacts', 'current_transformed_gradient_features.npz')]:
            with np.load(path / key) as z:
                x = z['features']
            picks, _ = select(x, 333, 'lcmd')
            expected = pd.read_csv(path / 'selected_next_batch.csv').sample_id.astype(str).tolist()
            assert c.ids(c.roles['u0'][picks]) == expected
            # Independently extract the 8 rows to catch model/order/T/sketch incompatibility.
            fresh = extract_linear_output_gradient_sketches(model,c.atom,c.angle,indices,c.transforms[rep],dimension=512,sketch_seed=sketch_seed(seed)).features
            np.testing.assert_allclose(fresh, x[np.r_[np.arange(4), np.arange(333,337)]], rtol=2e-4, atol=2e-5)
            regressions.append({'seed':seed,'representation':rep,'historical_batch_equal':True,'fresh_subset_equal':True,'feature_sha256':sha256_file(path/key)})
    pd.DataFrame(rows).to_csv(STUDY/'geometry_audit.csv',index=False)
    atomic_json(STUDY/'compatibility_audit.json', {'rows':regressions,'model_state_unchanged':True,'sample_rows_per_seed':8,'full_Jacobian_saved':False})
    status('P0','completed', geometry_rows=len(rows), historical_replays=len(regressions))
    status('audit','completed', reuse_points=70)


def verify_protocol():
    p=read(STUDY/'protocol.json')
    assert p['source_sha256']==sha256_file(SOURCE_DATA)
    assert p['graph_sha256']==sha256_file(SOURCE_GRAPH_CACHE)
    for s in SEEDS:
        assert p['contexts'][str(s)]['split_sha256']==sha256_file(BASE/f'splits/row_seed_{s}.csv')
    assert read(STUDY/'RUN_STATE.json')['P0']['status']=='completed'
    assert len(read(STUDY/'reuse_compatibility.json')['entries'])==70
    for path,digest in read(STUDY/'execution_code_freeze.json').items():
        assert sha256_file(ROOT/path)==digest, path
    return p


def run_arm(seed, arm):
    p=verify_protocol()
    c=Context(seed)
    assert {k:v.tolist() for k,v in c.transforms.items()}==p['contexts'][str(seed)]['transforms']
    rep, selector=ARMS[arm]
    l,u=c.roles['l0'].copy(),c.roles['u0'].copy()
    truth=c.truth.copy()
    model_path=c.anchor
    selected_all=[]
    root=STUDY/f'runtime/seed_{seed}/{arm}'
    for r in range(6):
        d=root/f'round_{r:02d}'
        d.mkdir(parents=True,exist_ok=True)
        contract={'seed':seed,'arm':arm,'round':r,'protocol_hash':stable_hash(p),'checkpoint_sha256':sha256_file(model_path),
            'ordered_l':c.ids(l),'ordered_u':c.ids(u),'transform':c.transforms[rep].tolist(),'sketch_seed':sketch_seed(seed),'dimension':512}
        freeze=d/'batch_freeze.json'
        model=load_predictor_checkpoint(model_path)
        if freeze.exists():
            record=read(freeze)
            assert record['contract']==contract
            assert sha256_file(d/'premeasurement.csv.gz')==record['premeasurement_sha256']
            picks=np.array(record['positions'],dtype=int)
            assert c.ids(u[picks])==record['ids']
        else:
            started=time.perf_counter()
            bank=extract_linear_output_gradient_sketches(model,c.atom,c.angle,np.r_[l,u],c.transforms[rep],dimension=512,sketch_seed=sketch_seed(seed))
            picks,trace=select(bank.features,len(l),selector)
            prediction,order=predict_outputs(model,c.atom,c.angle,u[picks])
            assert np.array_equal(order,u[picks])
            digest=write_predictions(d/'premeasurement.csv.gz',c.ids(u[picks]),prediction)
            eigen=np.maximum(np.linalg.eigvalsh(bank.features.astype(float).T@bank.features.astype(float)),0)
            prob=eigen[eigen>0]/eigen.sum()
            rank=float(np.exp(-np.sum(prob*np.log(prob))))
            chosen_inputs=c.data.iloc[u[picks]]
            geometry_rows=bank.features[np.linspace(0,len(bank.features)-1,128,dtype=int)].astype(float)
            distances=np.sqrt(np.maximum(np.sum((geometry_rows[:,None]-geometry_rows[None,:])**2,axis=2),0))
            atomic_json(d/'selection_diagnostics.json',{'gradient':bank.audit,'effective_rank':rank,
                'distance_quantiles_128_evenly_spaced_rows':np.quantile(distances[np.triu_indices(128,1)],[0,.25,.5,.75,1]).tolist(),
                'selected_unique_structures':int(chosen_inputs.canonical_smiles.nunique()),
                'selected_condition_counts':{col:{str(k):int(v) for k,v in chosen_inputs[col].value_counts().items()} for col in ('PE/EA','loading solvent')},'trace':trace,'seconds':time.perf_counter()-started,
                'norm_quantiles':np.quantile(np.linalg.norm(bank.features,axis=1),[0,.25,.5,.75,1]).tolist()})
            once(freeze,{'contract':contract,'positions':picks.tolist(),'ids':c.ids(u[picks]),'premeasurement_sha256':digest})
            del bank
        assert len(set(picks))==32 and len(picks)==32
        chosen=u[picks]
        incoming=c.ids(chosen)
        assert not set(incoming)&set(selected_all)
        selected_all+=incoming
        c.store.freeze_acquisitions(selected_all)
        truth=np.vstack([truth,c.store.reveal(incoming,'after_acquisition_fit')])
        l=np.r_[l,chosen]
        u=np.delete(u,picks)
        fitdir=root/f'round_{r+1:02d}/model'
        # A fit reservation counts interrupted attempts too; never silently expands cap.
        ledger=STUDY/'fit_ledger.json'
        entries=read(ledger) if ledger.exists() else {}
        key=f'{seed}/{arm}/{r+1}'
        if key not in entries:
            if len(entries)>=36: raise RuntimeError('36-fit quota exhausted')
            entries[key]={'status':'running','started_unix':time.time()}
            atomic_json(ledger,entries)
        elif entries[key]['status']=='running' and not (fitdir/'fit_audit.json').exists():
            raise RuntimeError(f'interrupted fit {key}; explicit attempt accounting required before restart')
        status('P1b' if rep=='whitening' else 'P1a','running',seed=seed,arm=arm,budget=len(l),reserved_fits=len(entries))
        fitcontract={'study':STUDY.name,'protocol_hash':stable_hash(p),'outer_seed':seed,'method':arm,'round':r+1,
            'train_sample_ids':c.ids(l),'validation_sample_ids':c.ids(c.roles['validation']), 'test_truth_access_count':0}
        audit=fit_from_same_initialization(atom_base=c.atom,angle=c.angle,normalization=c.normalization,preprocessing=c.preprocessing,
            train_indices=l,train_truth=truth,validation_indices=c.roles['validation'],validation_truth=c.validation,
            prediction_indices=c.roles['test'],prediction_sample_ids=c.ids(c.roles['test']),initialization_seed=initialization_seed(seed),
            training_config=c.config,contract=fitcontract,runtime=fitdir)
        entries[key]={'status':'completed','audit_sha256':sha256_file(fitdir/'fit_audit.json'),'training_seconds':audit['training_seconds']}
        atomic_json(ledger,entries)
        model_path=fitdir/'best.pt'
        atomic_json(root/f'round_{r+1:02d}/state.json',{'labeled':l.tolist(),'unlabeled':u.tolist(),'selected_all':selected_all,
            'checkpoint_sha256':sha256_file(model_path),'test_prediction_sha256':audit['prediction_sha256']})
    atomic_json(root/'label_access_audit.json',{'events':c.store.audit})
    atomic_json(root/'trajectory_freeze.json',{'budget':525,'all_selected_ids':selected_all,'test_truth_access_count':0})


def run():
    with (STUDY/'execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in ARMS:
            for seed in SEEDS:
                run_arm(seed,arm)
            if arm=='cw_ivr': status('P1a','completed',new_fits=24)
        status('P1b','completed',new_fits=12)
        atomic_json(STUDY/'global_pre_test_freeze.json',{'status':'all_new_trajectories_frozen','test_truth_access_count':0,
            'trajectories':{f'{s}/{a}':sha256_file(STUDY/f'runtime/seed_{s}/{a}/trajectory_freeze.json') for s in SEEDS for a in ARMS}})


def report():
    rows=[]
    controls=read(STUDY/'REUSE_MANIFEST.json')['entries']
    for seed in SEEDS:
        c=Context(seed)
        for arm in ('raw_lcmd','raw_maxdet','raw_ivr','cw_lcmd','random',*ARMS):
            for r,budget in enumerate(BUDGETS):
                if arm in ARMS:
                    model=c.anchor if r==0 else STUDY/f'runtime/seed_{seed}/{arm}/round_{r:02d}/model/best.pt'
                else:
                    item=next(x for x in controls if x['seed']==seed and x['arm']==arm and x['round']==r)
                    model=ROOT/item['checkpoint']
                    assert sha256_file(model)==item['checkpoint_sha256']
                if not model.with_name('fit_audit.json').exists(): continue
                dest=STUDY/f'runtime/validation/seed_{seed}/{arm}/{r}.json'
                identity=sha256_file(model)
                if dest.exists():
                    record=read(dest)
                    assert record['checkpoint_sha256']==identity
                else:
                    prediction,order=predict_outputs(load_predictor_checkpoint(model),c.atom,c.angle,c.roles['validation'])
                    assert np.array_equal(order,c.roles['validation'])
                    record={'checkpoint_sha256':identity,**metric_row(c.validation,prediction,c.preprocessing['target_scales'])}
                    atomic_json(dest,record)
                rows.append({'seed':seed,'arm':arm,'budget':budget,**record})
    df=pd.DataFrame(rows)
    df.to_csv(STUDY/'validation_curves.csv',index=False)
    summaries=[]
    for (seed,arm),g in df.groupby(['seed','arm']):
        g=g.sort_values('budget')
        if g.budget.tolist()!=list(BUDGETS): continue
        summaries.append({'seed':seed,'arm':arm,'AULC333_525':float(np.trapezoid(g.combined_normalized_RMSE,g.budget)/192),
                          'NRMSE525':float(g.combined_normalized_RMSE.iloc[-1])})
    summary=pd.DataFrame(summaries)
    summary.to_csv(STUDY/'validation_summary.csv',index=False)
    effects=[]
    for sel in ('lcmd','maxdet','ivr'):
        for seed in SEEDS:
            a=summary.loc[(summary.seed==seed)&(summary.arm=='cw_'+sel),'AULC333_525']
            b=summary.loc[(summary.seed==seed)&(summary.arm=='raw_'+sel),'AULC333_525']
            if len(a) and len(b): effects.append({'seed':seed,'selector':sel,'delta':float(a.iloc[0]-b.iloc[0]),'relative_improvement':float(1-a.iloc[0]/b.iloc[0])})
    pd.DataFrame(effects).to_csv(STUDY/'paired_effects.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,seed in zip(axes,SEEDS):
        for arm,g in df.loc[df.seed==seed].groupby('arm'):
            ax.plot(g.budget,g.combined_normalized_RMSE,'o-',label=arm,markersize=3)
        ax.set(title=f'Seed {seed} — validation',xlabel='Active labels',ylabel='NRMSE')
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(STUDY/'validation_curves.png',dpi=160); plt.close(fig)
    status('report','completed' if len(summary)==16 else 'running',complete_curves=len(summary))


def main(action):
    torch.set_num_threads(2)
    try:
        if action=='prepare': prepare()
        elif action=='audit': geometry_audit()
        elif action=='run': run()
        elif action=='report': report()
        elif action=='offline':
            from .feedback_ablation import offline_audit
            offline_audit(STUDY)
    except Exception as error:
        status(action,'blocked',error=f'{type(error).__name__}: {error}',resume=f'KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py {action}')
        raise

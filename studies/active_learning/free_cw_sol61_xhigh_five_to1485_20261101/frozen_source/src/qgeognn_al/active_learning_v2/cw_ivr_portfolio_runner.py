"""Resumable shared-batch portfolio; all selection is frozen before label reveal."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Sequence
import json
import time
import numpy as np
import pandas as pd
import torch
from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, load_features, seed_config, sketch_seed
from .cache import array_hash
from .gradient_features import extract_linear_output_gradient_sketches_many, state_dict_hash
from .gradient_transforms import center_width_transform
from .maxdet_study import historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import fit_from_same_initialization
from .sequential_acquisition import validate_trajectory_transition
from .short_sequential_runner import _write_json_once, _round0_evaluation
from .cw_ivr_portfolio_study import (
    STUDY, METHOD, METHODS, SEEDS, ACTIVE_LABEL_BUDGETS, ACQUISITION_ROUNDS,
    BATCH_SIZE, SKETCH_DIMENSION, FINAL_ACTIVE_LABELS, protocol_record,
    split_path, validate_prepared, read_json as _json)
from .cw_ivr_portfolio import select_portfolio
from .ivr_study import exclusive_lock, now


def _write_csv_once(path, frame):
    payload = frame.to_csv(index=False).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing mismatched CSV artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

class PortfolioContext:
    def __init__(self, seed: int, study: Path = STUDY):
        self.seed = int(seed)
        self.study = Path(study)
        self.runtime = self.study / "runtime" / f"seed_{self.seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(split_path(self.seed, self.study))
        validate_row_protocol(self.partition)
        self.data = load_features(SOURCE_DATA)
        if self.partition.sample_id.astype(str).tolist() != self.data.sample_id.astype(str).tolist():
            raise RuntimeError("short-sequential split/source order drift")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        old = _json(historical_round0_paths(self.seed)["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(historical_round0_paths(self.seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("historical scrubbed graph cache contains labels")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = stable_hash(protocol_record())
        _write_json_once(self.runtime / "context.json", {
            "study": self.study.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "split_hash": stable_hash(self.partition.to_dict("list")),
            "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])),
            "u0_ids_hash": ids_hash(self.ids(self.roles["u0"])),
            "validation_ids_hash": ids_hash(self.ids(self.roles["validation"])),
            "test_ids_hash": ids_hash(self.ids(self.roles["test"])),
            "normalization": asdict(self.normalization),
            "preprocessing": self.preprocessing,
            "center_width_transform": self.cw_audit,
            "test_truth_access_count": 0,
        })

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def new_store(self) -> RestrictedLabelStore:
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        if not np.array_equal(store.reveal(self.ids(self.roles["l0"]), "initial_fit"), self.l0_truth):
            raise RuntimeError("L0 truth drift")
        if not np.array_equal(store.reveal(self.ids(self.roles["validation"]), "initial_fit"), self.validation_truth):
            raise RuntimeError("validation truth drift")
        return store

    def training_config(self) -> dict:
        config = seed_config(self.seed, 0, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def fit(self, method: str, round_index: int, labeled: np.ndarray, truth: np.ndarray, runtime: Path) -> dict:
        train_ids = self.ids(labeled)
        contract = {
            "study": self.study.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "method": method,
            "round": round_index,
            "member": 0,
            "train_sample_ids": train_ids,
            "validation_sample_ids": self.ids(self.roles["validation"]),
            "L_t_ids_hash": ids_hash(train_ids),
            "prediction_role": "test_X_without_test_truth",
            "test_truth_access_count": 0,
        }
        audit = fit_from_same_initialization(
            atom_base=self.atom,
            angle=self.angle,
            normalization=self.normalization,
            preprocessing=self.preprocessing,
            train_indices=labeled,
            train_truth=truth,
            validation_indices=self.roles["validation"],
            validation_truth=self.validation_truth,
            prediction_indices=self.roles["test"],
            prediction_sample_ids=self.ids(self.roles["test"]),
            initialization_seed=initialization_seed(self.seed, 0),
            training_config=self.training_config(),
            contract=contract,
            runtime=runtime,
        )
        historical = _json(historical_round0_paths(self.seed)["member0_audit"])
        if audit["initialization_hash"] != historical["initialization_hash"]:
            raise RuntimeError("scratch initialization changed")
        return {"reuse_status": "new_fit", **audit}


def run_trajectory(context: PortfolioContext, method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    store = context.new_store()
    labeled = context.roles["l0"].copy()
    unlabeled = context.roles["u0"].copy()
    truth = context.l0_truth.copy()
    incoming: list[str] = []
    selected_all: list[str] = []
    round_hashes: list[str] = []
    fit_rows: list[dict] = []
    for round_index in range(ACQUISITION_ROUNDS + 1):
        if len(labeled) != ACTIVE_LABEL_BUDGETS[round_index]:
            raise RuntimeError("active-label budget drift")
        directory = context.runtime / method / f"round_{round_index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        input_contract = {
            "study": context.study.name,
            "protocol_hash": context.protocol_hash,
            "outer_seed": context.seed,
            "method": method,
            "round": round_index,
            "active_label_count": len(labeled),
            "unlabeled_count": len(unlabeled),
            "L_t_ids_hash": ids_hash(context.ids(labeled)),
            "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
            "incoming_selected_ids_hash": stable_hash(incoming),
            "trajectory_specific_model": True,
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "input_contract.json", input_contract)
        _write_csv_once(directory / "state.csv", pd.DataFrame({
            "role": ["labeled"] * len(labeled) + ["unlabeled"] * len(unlabeled),
            "position": list(range(len(labeled))) + list(range(len(unlabeled))),
            "canonical_index": np.r_[labeled, unlabeled],
            "sample_id": context.ids(np.r_[labeled, unlabeled]),
        }))
        _write_csv_once(directory / "selected_batch.csv", pd.DataFrame({
            "outer_seed": [context.seed] * len(incoming),
            "method": [method] * len(incoming),
            "round": [round_index] * len(incoming),
            "selection_order": np.arange(len(incoming)),
            "sample_id": incoming,
        }))
        if round_index == 0:
            model_path, prediction_path, evaluation = _round0_evaluation(context)
        else:
            evaluation = context.fit(method, round_index, labeled, truth, directory / "model")
            model_path = directory / "model/best.pt"
            prediction_path = directory / "model/predictions.csv.gz"
        fit_rows.append({"outer_seed": context.seed, "method": method, "round": round_index, **evaluation})
        outgoing: list[str] = []
        acquisition_contract = None
        if round_index < ACQUISITION_ROUNDS:
            outgoing, acquisition_contract = _acquire(
                context, method, round_index, labeled, unlabeled, model_path, directory
            )
        freeze = {
            "input": input_contract,
            "status": "FROZEN_BEFORE_TEST_TRUTH",
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": sha256_file(model_path),
            "checkpoint_state_hash": evaluation["checkpoint_state_hash"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "fit_contract_hash": evaluation["fit_contract_hash"],
            "outgoing_selected_ids_hash": stable_hash(outgoing),
            "acquisition_contract_hash": None if acquisition_contract is None else stable_hash(acquisition_contract),
            "test_truth_access_count": 0,
            "files": {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*"))
                      if p.is_file() and p.name != "round_freeze.json"},
        }
        _write_json_once(directory / "round_freeze.json", freeze)
        print(json.dumps({"seed":context.seed,"round_frozen":round_index,"active_labels":len(labeled),"test_truth_access_count":0}),flush=True)
        round_hashes.append(sha256_file(directory / "round_freeze.json"))
        if round_index == ACQUISITION_ROUNDS:
            break
        old_l, old_u = context.ids(labeled), context.ids(unlabeled)
        position_by_id = {value: index for index, value in enumerate(old_u)}
        selected_indices = unlabeled[np.asarray([position_by_id[value] for value in outgoing], dtype=int)]
        selected_set = set(outgoing)
        next_unlabeled = np.asarray([index for index, value in zip(unlabeled, old_u) if value not in selected_set])
        next_labeled = np.r_[labeled, selected_indices]
        validate_trajectory_transition(old_l, old_u, outgoing, context.ids(next_labeled), context.ids(next_unlabeled))
        selected_all.extend(outgoing)
        if len(selected_all) != len(set(selected_all)):
            raise RuntimeError("trajectory selected an ID more than once")
        store.freeze_acquisitions(selected_all)
        truth = np.vstack([truth, store.reveal(outgoing, "after_acquisition_fit")])
        labeled, unlabeled, incoming = next_labeled, next_unlabeled, outgoing
    if len(labeled) != FINAL_ACTIVE_LABELS or len(selected_all) != ACQUISITION_ROUNDS * BATCH_SIZE:
        raise RuntimeError("trajectory did not stop exactly at 653")
    _write_csv_once(context.runtime / method / "fit_audit.csv", pd.DataFrame(fit_rows))
    _write_csv_once(context.runtime / method / "label_access_audit.csv", pd.DataFrame(store.audit))
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "outer_seed": context.seed,
        "method": method,
        "round_points": ACQUISITION_ROUNDS + 1,
        "final_active_labels": len(labeled),
        "selected_ids_hash": stable_hash(selected_all),
        "round_freeze_sha256": round_hashes,
        "summary_files": {name: sha256_file(context.runtime / method / name) for name in ("fit_audit.csv", "label_access_audit.csv")},
        "test_truth_access_count": 0,
    }
    _write_json_once(context.runtime / method / "trajectory_freeze.json", freeze)
    return freeze



def _features(context, round_index, model_path, current, directory):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'shared_gradient_bank.npz'
    receipt=directory/'gradient_contract.json'
    expected={'checkpoint_sha256':sha256_file(model_path),'current_model_round':round_index,
              'ordered_indices':current.tolist(),'sketch_seed':sketch_seed(context.seed),
              'protocol_hash':context.protocol_hash,'center_width_transform':context.cw_audit,
              'endpoint_scales':context.preprocessing['target_scales'],'test_truth_access_count':0}
    if receipt.exists():
        record=_json(receipt)
        if record['input']!=expected or sha256_file(path)!=record['gradient_bank_sha256']:
            raise RuntimeError('gradient bank cache drift')
        with np.load(path) as values:
            return values['cw'],values['ivr'],record
    started=time.perf_counter()
    model=load_predictor_checkpoint(model_path)
    if state_dict_hash(model)!=_json(model_path.parent/'fit_audit.json')['checkpoint_state_hash']:
        raise RuntimeError('checkpoint state drift')
    scales=context.preprocessing['target_scales']
    result=extract_linear_output_gradient_sketches_many(model,context.atom,context.angle,current,
        {'cw':context.cw_transform,'ivr':np.diag([1/scales['V1'],1/scales['V2']])},
        dimension=512,sketch_seed=sketch_seed(context.seed))
    np.savez_compressed(path,cw=result['cw'].features,ivr=result['ivr'].features,canonical_indices=current)
    record={'input':expected,'gradient_bank_sha256':sha256_file(path),
            'cw_feature_hash':array_hash(result['cw'].features),'ivr_feature_hash':array_hash(result['ivr'].features),
            'gradient_extraction_seconds':time.perf_counter()-started,
            'cw_audit':result['cw'].audit,'ivr_audit':result['ivr'].audit,'extraction_passes':1}
    _write_json_once(receipt,record)
    return result['cw'].features,result['ivr'].features,record


def _acquire(context, method, round_index, labeled, unlabeled, model_path, directory):
    acquisition=directory/'acquisition_artifacts'
    path=acquisition/'contract.json'
    selected_path=acquisition/'selected_next_batch.csv'
    expected={'outer_seed':context.seed,'method':method,'source_round':round_index,
              'active_labels_before':len(labeled),'active_labels_after':len(labeled)+32,
              'checkpoint_sha256':sha256_file(model_path),'L_t_ids_hash':ids_hash(context.ids(labeled)),
              'U_t_ids_hash':ids_hash(context.ids(unlabeled)),'protocol_hash':context.protocol_hash,
              'test_truth_access_count':0}
    if path.exists():
        saved=_json(path)
        if saved['input']!=expected:
            raise RuntimeError('acquisition input drift')
        for name,digest in saved['files'].items():
            if sha256_file(acquisition/name)!=digest:
                raise RuntimeError('acquisition artifact drift')
        return pd.read_csv(selected_path).sample_id.astype(str).tolist(),saved
    cw,ivr,bank=_features(context,round_index,model_path,np.r_[labeled,unlabeled],acquisition)
    started=time.perf_counter()
    result=select_portfolio(cw,ivr,len(labeled))
    selector_seconds=time.perf_counter()-started
    positions=np.array(result['selected_pool_positions'],dtype=int)
    selected_ids=np.array(context.ids(unlabeled))[positions].tolist()
    if len(selected_ids)!=32 or len(set(selected_ids))!=32:
        raise RuntimeError('batch must contain 32 distinct U rows')
    trace=pd.DataFrame(result['trace'])
    trace['selected_sample_id']=selected_ids
    trace['selected_canonical_index']=unlabeled[positions]
    trace['outer_seed']=context.seed
    trace['round']=round_index+1
    trace['active_labels_before']=len(labeled)
    trace['active_labels_after']=len(labeled)+32
    trace['checkpoint_sha256']=sha256_file(model_path)
    trace['gradient_bank_sha256']=bank['gradient_bank_sha256']
    _write_csv_once(selected_path,pd.DataFrame({'sample_id':selected_ids,'canonical_index':unlabeled[positions],
        'selection_order':np.arange(32),'expert':trace.expert}))
    _write_csv_once(acquisition/'selection_trace.csv',trace)
    _write_json_once(acquisition/'diagnostics.json',result['diagnostics'])
    saved={'input':expected,'gradient_bank_sha256':bank['gradient_bank_sha256'],
           'gradient_extraction_seconds':bank['gradient_extraction_seconds'],'selector_seconds':selector_seconds,
           'selected_ids_hash':stable_hash(selected_ids),'cw_turns':16,'ivr_turns':16,'batch_size':32,'unique_ids':32,
           'test_truth_access_count':0,
           'files':{str(p.relative_to(acquisition)):sha256_file(p) for p in sorted(acquisition.iterdir()) if p.is_file()}}
    _write_json_once(path,saved)
    return selected_ids,saved


def verify_seed(seed,study=STUDY):
    base=Path(study)/f'runtime/seed_{seed}'/METHOD
    trajectory=_json(base/'trajectory_freeze.json')
    if trajectory['final_active_labels']!=653 or trajectory['test_truth_access_count']!=0 or len(trajectory['round_freeze_sha256'])!=11:
        raise RuntimeError('incomplete blind trajectory')
    for name,digest in trajectory['summary_files'].items():
        if sha256_file(base/name)!=digest:
            raise RuntimeError('trajectory summary artifact changed')
    entries={}
    selected_all=[]
    for r in range(11):
        directory=base/f'round_{r:02d}'
        path=directory/'round_freeze.json'
        record=_json(path)
        if sha256_file(path)!=trajectory['round_freeze_sha256'][r] or record['test_truth_access_count']!=0:
            raise RuntimeError('round freeze mismatch')
        for name,digest in record['files'].items():
            if sha256_file(directory/name)!=digest:
                raise RuntimeError(f'nested artifact changed: {directory/name}')
        for kind in ('checkpoint','prediction'):
            if sha256_file(Path(record[f'{kind}_path']))!=record[f'{kind}_sha256']:
                raise RuntimeError('frozen model/predictions changed')
        state=pd.read_csv(directory/'state.csv')
        if len(state.loc[state.role.eq('labeled')])!=ACTIVE_LABEL_BUDGETS[r]:
            raise RuntimeError('state budget mismatch')
        if r<10:
            batch=pd.read_csv(directory/'acquisition_artifacts/selected_next_batch.csv')
            if len(batch)!=32 or batch.sample_id.nunique()!=32 or batch.expert.tolist()!=['cw','ivr']*16:
                raise RuntimeError('invalid portfolio batch')
            if not set(batch.sample_id)<=set(state.loc[state.role.eq('unlabeled'),'sample_id']):
                raise RuntimeError('selection outside U')
            next_dir=base/f'round_{r+1:02d}'
            next_record=_json(next_dir/'round_freeze.json')
            if sha256_file(next_dir/'state.csv')!=next_record['files']['state.csv']:
                raise RuntimeError('nested artifact changed: next state')
            next_state=pd.read_csv(next_dir/'state.csv')
            validate_trajectory_transition(state.loc[state.role.eq('labeled'),'sample_id'].tolist(),state.loc[state.role.eq('unlabeled'),'sample_id'].tolist(),batch.sample_id.tolist(),next_state.loc[next_state.role.eq('labeled'),'sample_id'].tolist(),next_state.loc[next_state.role.eq('unlabeled'),'sample_id'].tolist())
            selected_all.extend(batch.sample_id.tolist())
        entries[f'seed_{seed}/round_{r:02d}']={'round_freeze_sha256':sha256_file(path),
             'checkpoint_sha256':record['checkpoint_sha256'],'prediction_sha256':record['prediction_sha256']}
    if len(selected_all)!=320 or len(set(selected_all))!=320:
        raise RuntimeError('trajectory duplicate/missing acquisitions')
    fits=pd.read_csv(base/'fit_audit.csv')
    if len(fits)!=11 or (fits.reuse_status=='new_fit').sum()!=10:
        raise RuntimeError('fit count mismatch')
    return entries


def execute_seed(seed,study=STUDY):
    if seed not in SEEDS:
        raise ValueError('only development seeds 157 and 6101')
    study=Path(study)
    validate_prepared(study)
    if _json(study/'global_pre_test_freeze.json')['status']!='PENDING_TRAJECTORIES':
        raise RuntimeError('execution closed after global freeze')
    smoke=_json(study/'engineering_smoke.json')
    if smoke['status']!='PASS' or smoke['new_fits']!=0:
        raise RuntimeError('passing selection-only smoke required')
    if seed==6101:
        verify_seed(157,study)
    with exclusive_lock(study/'runtime/worker.lock'):
        torch.set_num_threads(2)
        started=study/f'runtime/seed_{seed}/execution_started.json'
        if not started.exists():
            atomic_json(started,{'started_at':now(),'test_truth_access_count':0})
        context=PortfolioContext(seed,study)
        result=run_trajectory(context,METHOD)
        verify_seed(seed,study)
        completed=study/f'runtime/seed_{seed}/execution_completed.json'
        if not completed.exists():
            atomic_json(completed,{'completed_at':now(),'new_fits':10,'test_truth_access_count':0})
        return result


def finalize_pre_test(study=STUDY):
    study=Path(study)
    validate_prepared(study)
    entries={}
    for seed in SEEDS:
        entries.update(verify_seed(seed,study))
    frozen={'status':'FROZEN_BEFORE_TEST_TRUTH','entries':entries,'new_fits':20,
            'frozen_prediction_points':22,'test_truth_access_count':0}
    path=study/'global_pre_test_freeze.json'
    previous=_json(path)
    if previous['status']!='PENDING_TRAJECTORIES' and previous!=frozen:
        raise RuntimeError('global freeze changed')
    atomic_json(path,frozen)
    return frozen


def selection_smoke(study=STUDY):
    validate_prepared(study)
    torch.set_num_threads(2)
    checks=[]
    for seed in SEEDS:
        context=PortfolioContext(seed,Path(study))
        l,u=context.roles['l0'],context.roles['u0']
        model=historical_round0_paths(seed)['member0_model']
        directory=context.runtime/METHOD/'round_00'
        ids,record=_acquire(context,METHOD,0,l,u,model,directory)
        again,cached=_acquire(context,METHOD,0,l,u,model,directory)
        if ids!=again or record!=cached:
            raise RuntimeError('selection resume not deterministic')
        # Same real historical L333, compare new shared-pass geometries to frozen banks.
        from .lcmd import lcmd_tp_select
        from .ivr import conditional_batch_ivr
        with np.load(directory/'acquisition_artifacts/shared_gradient_bank.npz') as bank:
            cw,ivr=bank['cw'],bank['ivr']
        source=STUDY.parent/'qgeognn_v2_row_short_sequential_b32'/f'runtime/seed_{seed}/center_width_lcmd/round_00/acquisition_artifacts/current_transformed_gradient_features.npz'
        receipt=_json(source.with_suffix('.npz.contract.json'))
        if receipt['sha256']!=sha256_file(source) or receipt['contract']['checkpoint_sha256']!=sha256_file(model):
            raise RuntimeError('historical CW bank provenance mismatch')
        with np.load(source) as bank:
            if not np.array_equal(bank['canonical_indices'],np.r_[l,u]):
                raise RuntimeError('historical CW bank row order mismatch')
            old_cw=bank['features']
        with np.load(historical_round0_paths(seed)['gradient']) as bank:
            old_ivr=bank['features']
        errors={}
        for name,a,b in [('cw',cw,old_cw),('ivr',ivr,old_ivr)]:
            relative=float(np.linalg.norm(a.astype(float)-b)/np.linalg.norm(b))
            if relative>1e-5:
                raise RuntimeError('shared-pass geometric regression failed')
            errors[name+'_relative_feature_error']=relative
        if not np.array_equal(lcmd_tp_select(cw[333:],cw[:333],32).selected_pool_positions,lcmd_tp_select(old_cw[333:],old_cw[:333],32).selected_pool_positions):
            raise RuntimeError('historical CW selected IDs changed under shared extraction')
        args=(np.arange(333),np.arange(333,3330),32)
        if conditional_batch_ivr(ivr,*args)['selected_candidate_positions']!=conditional_batch_ivr(old_ivr,*args)['selected_candidate_positions']:
            raise RuntimeError('historical IVR selected IDs changed under shared extraction')
        checks.append({'seed':seed,'unique_ids':len(set(ids)),'cw_turns':16,'ivr_turns':16,
                       'historical_cw_ivr_batch_ids_unchanged':True,**errors})
    result={'status':'PASS','checks':checks,'new_fits':0,'test_truth_access_count':0}
    atomic_json(Path(study)/'engineering_smoke.json',result)
    return result

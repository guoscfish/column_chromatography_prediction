"""Resumable original-QGeoGNN formal pair; no test-target evaluation path.

The acquisition callback receives only authorized outer-training X and L targets.
A passing preregistered, seed-local posterior qualification is mandatory. This module never changes
historical artifacts and never calls an LLM or a smoke training configuration.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import hashlib
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd

import study_support as support
from qgeognn_al.active_learning_v2.benchmark_protocol import initialization_seed
from qgeognn_al.active_learning_v2.protocol import ids_hash, stable_hash
from qgeognn_al.active_learning_v2.sequential_acquisition import validate_trajectory_transition
from qgeognn_al.active_learning_v2.sequential_runner import SequentialSeedContext

HERE = Path(__file__).resolve().parent
ARMS = ('posterior_epig_conditional_greedy32', 'posterior_ivr_conditional_greedy32')
SEEDS = (157, 887, 2357, 6101, 12203)
REQUIRED_FIT_FILES = ('best.pt', 'predictions.csv.gz', 'fit_audit.json', 'history.csv')


class BudgetExhausted(RuntimeError):
    pass


@contextmanager
def exclusive_worker(path):
    """An OS-held nonblocking lock, not a stale-file existence check."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another EPIG formal worker holds the study lock') from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'pid': os.getpid(), 'acquired_unix': time.time()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def qualification_gate(study):
    """Reject unqualified, stale, or unbound decisions before reading new labels."""
    study = Path(study)
    support.verify_freeze()
    decision = support.read(study / 'qualification/decision.json')
    families = decision.get('selected_family_by_seed', {})
    allowed = {'gp_rbf', 'gp_matern32', 'blr_independent', 'blr_coregionalized'}
    if decision.get('status') != 'QUALIFIED' or decision.get('run_allowed') is not True or set(families) != set(map(str, SEEDS)) or any(f not in allowed for f in families.values()):
        raise RuntimeError('formal AL requires a qualified preregistered posterior for every seed; efficacy remains unevaluated')
    if decision.get('freeze_sha256') != support.digest(study / 'freeze.json'):
        raise RuntimeError('qualification decision is not bound to this frozen implementation')
    config = support.read(study / 'config.json')
    if tuple(config['seeds']) != SEEDS or tuple(config['primary_arms']) != ARMS:
        raise RuntimeError('formal seed/arm drift')
    if (config['initial_labels'], config['batch_size'], config['primary_rounds']) != (333, 32, 21):
        raise RuntimeError('formal budget schedule drift')
    budget = config['compute_budget']
    if budget['new_primary_qgeognn_fits_max'] != 210 or budget['formal_wall_budget_hours'] != 40:
        raise RuntimeError('formal fit/wall budget differs from registered 210 fits / 40 hours')
    if budget['max_concurrent_qgeognn'] != 1 or budget['cpu_threads_per_fit'] != 2:
        raise RuntimeError('formal resource limits differ from one worker / two threads')
    return decision, config


class BudgetLedger:
    """Cumulative launched-fit cap and an absolute 40h deadline across resumes.

    Time while the process is absent is conservatively charged. Interrupted
    attempts also consume a fit slot, so automatic retries cannot exceed budget.
    Completed cached fits never reserve another slot.
    """
    def __init__(self, path, *, fit_limit=210, wall_hours=40, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        if self.path.exists():
            self.value = support.read(self.path)
            if self.value['fit_limit'] != fit_limit or self.value['wall_hours'] != wall_hours:
                raise RuntimeError('resume budget drift')
        else:
            self.value = {'started_unix': clock(), 'fit_limit': fit_limit,
                          'wall_hours': wall_hours, 'attempts': []}
            support.dump(self.path, self.value)

    def remaining_seconds(self):
        return self.value['started_unix'] + self.value['wall_hours'] * 3600 - self.clock()

    def check(self):
        if self.remaining_seconds() <= 0:
            raise BudgetExhausted('40h registered formal wall budget exhausted')

    def reserve(self, key):
        self.check()
        if len(self.value['attempts']) >= self.value['fit_limit']:
            raise BudgetExhausted('210 launched QGeoGNN fit attempts exhausted')
        rec = {'key': key, 'started_unix': self.clock(), 'status': 'RUNNING'}
        self.value['attempts'].append(rec)
        support.dump(self.path, self.value)
        return len(self.value['attempts']) - 1

    def finish(self, position, status, **extra):
        self.value['attempts'][position].update(status=status, ended_unix=self.clock(), **extra)
        support.dump(self.path, self.value)


@contextmanager
def fit_deadline(seconds):
    """Stop a CPU fit at its global deadline, preserving its partial artifacts."""
    if seconds <= 0:
        raise BudgetExhausted('registered formal wall budget exhausted')
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise BudgetExhausted('registered formal wall deadline reached during fit')
    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def quarantine_partial(model, attempts):
    """Never overwrite a partial scratch attempt; keep it for cost/provenance."""
    model = Path(model)
    if not model.exists() or not any(model.iterdir()):
        return None
    if all((model / name).exists() for name in REQUIRED_FIT_FILES):
        return None
    attempts = Path(attempts)
    attempts.mkdir(parents=True, exist_ok=True)
    destination = attempts / f'{model.parent.name}_{time.time_ns()}'
    model.rename(destination)
    return destination


def validate_selected(selected_ids, labeled_ids, unlabeled_ids, *, batch_size=32):
    values = [str(i) for i in selected_ids]
    if len(values) != batch_size or len(set(values)) != batch_size:
        raise ValueError('selection must contain exactly 32 distinct IDs')
    if not set(values) <= set(unlabeled_ids) or set(values) & set(labeled_ids):
        raise PermissionError('selected IDs must be unlabelled members of current U')
    return values


def freeze_selection_then_reveal(store, path, record, labeled_ids, unlabeled_ids, previous_selected):
    """The persisted immutable selection is the authorization prerequisite."""
    path = Path(path)
    values = validate_selected(record['selected_ids'], labeled_ids, unlabeled_ids)
    record = {**record, 'selected_ids': values, 'selected_ids_hash': stable_hash(values)}
    support.write_once(path, record)
    if support.read(path) != record:
        raise RuntimeError('selection failed persistence verification')
    # Bind the exact bytes that precede label access, independently of JSON hashing.
    support.write_once(path.with_suffix('.sha256.json'), {'sha256': support.digest(path)})
    if support.read(path.with_suffix('.sha256.json'))['sha256'] != support.digest(path):
        raise RuntimeError('selection byte hash drift')
    store.freeze_acquisitions(list(previous_selected) + values)
    truth = store.reveal(values, 'after_acquisition_fit')
    return truth


def reuse_initial(context, study, arm):
    """Verify the seed-matched historical L333 model without retraining it."""
    base = support.BASE / f'runtime/seed_{context.seed}'
    model = base / 'lcmd/round_00/model'
    old = support.read(base / 'lcmd/round_00/input_contract.json')
    source_context = support.read(base / 'context.json')['contract']
    audit = support.read(model / 'fit_audit.json')
    expected_config = context.training_config(0)
    if old['training_config'] != expected_config:
        raise RuntimeError('historical L333 training configuration mismatch')
    if old['L_t_ids_hash'] != ids_hash(context.ids(context.roles['l0'])) or old['U_t_ids_hash'] != ids_hash(context.ids(context.roles['u0'])):
        raise RuntimeError('historical L333 L/U identity mismatch')
    for role in ('l0', 'u0', 'validation', 'test'):
        if source_context[f'{role}_ids_hash'] != ids_hash(context.ids(context.roles[role])):
            raise RuntimeError(f'historical {role} identity mismatch')
    for key in ('partition_hash', 'preprocessing', 'normalization'):
        if source_context[key] != context.context_contract[key]:
            raise RuntimeError(f'historical {key} mismatch')
    if audit['initialization_seed'] != initialization_seed(context.seed, 0):
        raise RuntimeError('historical initialization seed mismatch')
    if audit['train_rows'] != 333 or audit['validation_rows'] != 416 or audit['parameter_count'] != 458952 or audit['model_variant'] != 'qgeognn_v2':
        raise RuntimeError('historical model/row identity mismatch')
    if audit['train_ids_hash'] != ids_hash(context.ids(context.roles['l0'])) or audit['validation_ids_hash'] != ids_hash(context.ids(context.roles['validation'])):
        raise RuntimeError('historical fit identity mismatch')
    protocol = support.read(support.ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/protocol.json')
    initial = next(s for s in protocol['initial_states'] if s['seed'] == context.seed)
    if initial['initialization_hash'] != audit['initialization_hash'] or initial['checkpoint_sha256'] != audit['checkpoint_sha256']:
        raise RuntimeError('historical model differs from published L333 initial state')
    if audit['checkpoint_sha256'] != support.digest(model / 'best.pt') or audit['prediction_sha256'] != support.digest(model / 'predictions.csv.gz'):
        raise RuntimeError('historical L333 model/prediction hash drift')
    metric = validation_metrics(model, context.preprocessing['target_scales'])
    result = {'seed': context.seed, 'arm': arm, 'round': 0, 'budget': 333,
              'reused': True, 'model_directory': str(model.relative_to(support.ROOT)),
              'fit_audit_sha256': support.digest(model/'fit_audit.json'),
              'checkpoint_sha256': audit['checkpoint_sha256'], 'initialization_hash': audit['initialization_hash'],
              'validation': metric, 'new_fit_attempts': 0}
    support.write_once(Path(study) / f'runtime/seed_{context.seed}/{arm}/round_00/fit_receipt.json', result)
    return result


def validation_metrics(model, scales):
    model = Path(model)
    audit = support.read(model / 'fit_audit.json')
    history = pd.read_csv(model / 'history.csv')
    row = history.loc[history.epoch.eq(audit['best_epoch'])]
    if len(row) != 1:
        raise RuntimeError('missing/nonunique validation best epoch')
    row = row.iloc[0]
    score = float(audit['best_validation_combined_normalized_rmse'])
    direct = np.sqrt(((row.validation_V1_rmse/scales['V1'])**2 + (row.validation_V2_rmse/scales['V2'])**2)/2)
    if not np.isclose(score, row.validation_combined_normalized_rmse, atol=1e-12, rtol=0) or not np.isclose(score, direct, atol=2e-7, rtol=0):
        raise RuntimeError('validation metric/history/fixed scale mismatch')
    return {'combined_nrmse': score, 'V1_RMSE': float(row.validation_V1_rmse),
            'V2_RMSE': float(row.validation_V2_rmse), 'V1_MAE': float(row.validation_V1_mae),
            'V2_MAE': float(row.validation_V2_mae), 'best_epoch': int(audit['best_epoch']),
            'epochs_run': int(audit['epochs_run']), 'evaluation_role': 'validation_checkpoint_selection'}


def fit_round(context, study, arm, r, labeled, unlabeled, truth, ledger, initial_hash):
    directory = Path(study) / f'runtime/seed_{context.seed}/{arm}/round_{r:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    state = {'seed': context.seed, 'arm': arm, 'round': r, 'budget': len(labeled),
             'ordered_labeled_indices': np.asarray(labeled).tolist(), 'ordered_unlabeled_indices': np.asarray(unlabeled).tolist(),
             'L_ids_hash': ids_hash(context.ids(labeled)), 'U_ids_hash': ids_hash(context.ids(unlabeled)),
             'train_truth_float32_sha256': hashlib.sha256(np.asarray(truth, dtype=np.float32).tobytes()).hexdigest(),
             'freeze_sha256': support.digest(Path(study)/'freeze.json'),
             'qualification_sha256': support.digest(Path(study)/'qualification/decision.json'),
             'training_config': context.training_config(0), 'initialization_seed': initialization_seed(context.seed),
             'target_scales': context.preprocessing['target_scales'], 'test_labels_used': 0}
    support.write_once(directory/'state.json', state)
    receipt_path = directory / 'fit_receipt.json'
    model = directory/'model'
    if receipt_path.exists():
        receipt = support.read(receipt_path)
        if receipt['state_hash'] != stable_hash(state):
            raise RuntimeError('resumed fit state differs')
        for name, expected in receipt['files'].items():
            if support.digest(model/name) != expected:
                raise RuntimeError(f'completed fit artifact drift: {model/name}')
        return receipt
    quarantine_partial(model, directory.parent/'incomplete_attempts')
    cached = all((model/name).exists() for name in REQUIRED_FIT_FILES)
    position = None if cached else ledger.reserve(f'{context.seed}:{arm}:{r}')
    try:
        # Reuses the original fit_from_same_initialization implementation inside
        # context.fit. Explicit U prediction inputs keep test X/targets out here.
        with fit_deadline(ledger.remaining_seconds()):
            audit = context.fit(method=arm, round_index=r, labeled_indices=labeled, labeled_truth=truth,
                runtime=model, prediction_indices=unlabeled)
        if audit['initialization_hash'] != initial_hash or audit['parameter_count'] != 458952 or audit['model_variant'] != 'qgeognn_v2':
            raise RuntimeError('formal model differs from original initialization/architecture')
        if audit['train_ids_hash'] != ids_hash(context.ids(labeled)) or audit['validation_ids_hash'] != ids_hash(context.ids(context.roles['validation'])):
            raise RuntimeError('formal fit identities drifted')
        receipt = {'seed': context.seed, 'arm': arm, 'round': r, 'budget': len(labeled), 'reused': False,
            'recovered_complete_fit': cached, 'state_hash': stable_hash(state),
            'checkpoint_sha256': audit['checkpoint_sha256'], 'initialization_hash': audit['initialization_hash'],
            'validation': validation_metrics(model, context.preprocessing['target_scales']),
            'training_seconds': audit['training_seconds'], 'model_directory': str(model.relative_to(support.ROOT)),
            'files': {name: support.digest(model/name) for name in REQUIRED_FIT_FILES}}
        support.write_once(receipt_path, receipt)
        if position is not None:
            ledger.finish(position, 'COMPLETED', checkpoint_sha256=audit['checkpoint_sha256'])
        return receipt
    except BaseException as exc:
        if position is not None:
            ledger.finish(position, 'INTERRUPTED_OR_FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


def acquisition_inputs(context, raw_x, arm, r, labeled, unlabeled, truth, family):
    """NaN-mask validation/test covariates as an additional capability boundary."""
    x = np.full(np.asarray(raw_x).shape, np.nan, dtype=np.float64)
    x[context.outer] = np.asarray(raw_x)[context.outer]
    x.setflags(write=False)
    y = np.asarray(truth, dtype=np.float64).copy()
    y.setflags(write=False)
    frame = context.data.copy()
    forbidden = np.setdiff1d(np.arange(len(frame)), context.outer)
    frame.loc[frame.index[forbidden], :] = None
    return dict(seed=context.seed, arm=arm, round_index=r,
                labeled_indices=np.asarray(labeled,dtype=int).copy(),
                unlabeled_indices=np.asarray(unlabeled,dtype=int).copy(), labeled_truth=y,
                feature_matrix_X=x, feature_frame=frame,
                target_scales=dict(context.preprocessing['target_scales']), previous_checkpoint_path=None)


def run_seed_arm(context, study, arm, acquire_callback, raw_x, family, ledger):
    initial = reuse_initial(context, study, arm)
    store = context.new_method_store()
    labeled, unlabeled = context.roles['l0'].copy(), context.roles['u0'].copy()
    truth = context.l0_truth.copy()
    selected_history, fits = [], [initial]
    root = Path(study) / f'runtime/seed_{context.seed}/{arm}'
    for r in range(21):
        ledger.check()
        support.verify_freeze()
        selection_path = root/f'round_{r:02d}/selection.json'
        expected = {'seed': context.seed, 'arm': arm, 'source_round': r, 'batch_size': 32,
                    'L_ids_hash': ids_hash(context.ids(labeled)), 'U_ids_hash': ids_hash(context.ids(unlabeled)),
                    'source_checkpoint_sha256': fits[-1]['checkpoint_sha256'],
                    'freeze_sha256': support.digest(Path(study)/'freeze.json'),
                    'qualification_sha256': support.digest(Path(study)/'qualification/decision.json'),
                    'selected_family': family}
        if selection_path.exists():
            selection = support.read(selection_path)
            if selection['input'] != expected:
                raise RuntimeError('resumed acquisition input drift')
        else:
            with fit_deadline(ledger.remaining_seconds()):
                result = acquire_callback(**acquisition_inputs(context, raw_x, arm, r, labeled, unlabeled, truth, family))
            selection = {'input': expected, 'selected_ids': list(result['selected_ids']),
                         'diagnostics': result.get('diagnostics', {}),
                         'posterior': result.get('posterior', result.get('probabilistic_model_audit', {})),
                         'units': result.get('units', 'declared by acquisition callback'),
                         'target_labels_read': 0, 'test_labels_read': 0}
        new_truth = freeze_selection_then_reveal(store, selection_path, selection, context.ids(labeled), context.ids(unlabeled), selected_history)
        chosen = support.read(selection_path)['selected_ids']
        lookup = dict(zip(context.ids(unlabeled), unlabeled.tolist()))
        new_l = np.r_[labeled, [lookup[i] for i in chosen]]
        chosen_set = set(chosen)
        new_u = np.asarray([i for i in unlabeled if context.ids([i])[0] not in chosen_set],dtype=int)
        validate_trajectory_transition(context.ids(labeled), context.ids(unlabeled), chosen, context.ids(new_l), context.ids(new_u))
        selected_history.extend(chosen)
        truth = np.vstack([truth,new_truth])
        labeled, unlabeled = new_l,new_u
        support.dump(root/'label_access_audit.json', {'seed':context.seed,'arm':arm,'calls':store.audit,
            'test_truth_access_count':0,'acquisition_supervised_validation_rows':0})
        fits.append(fit_round(context,study,arm,r+1,labeled,unlabeled,truth,ledger,initial['initialization_hash']))
        support.dump(root/'progress.json', {'status':'RUNNING','completed_rounds':r+1,'budget':len(labeled),
            'selected_ids_hash':stable_hash(selected_history),'test_labels_read':0})
    result = {'status':'COMPLETED_L1005','seed':context.seed,'arm':arm,'rounds':21,'budget':len(labeled),
              'selected_ids_hash':stable_hash(selected_history),'test_labels_read':0,
              'fit_receipts':{str((root/f'round_{r:02d}/fit_receipt.json').relative_to(Path(study))):support.digest(root/f'round_{r:02d}/fit_receipt.json') for r in range(22)}}
    support.write_once(root/'trajectory_freeze.json',result)
    return result


def run_formal(acquire_callback, study=HERE):
    """Run the registered 5×2×21 pair, or resume without duplicate fits.

    Callable signature is **acquisition_inputs(...); result must contain the
    exact `selected_ids` (32), with optional JSON-serializable diagnostics and
    posterior audit. No performance-based stopping or seed cancellation exists.
    """
    study=Path(study)
    decision, config=qualification_gate(study)
    import torch
    torch.set_num_threads(2)
    with exclusive_worker(study/'runtime/formal.lock'):
        ledger=BudgetLedger(study/'runtime/compute_budget.json')
        frame, raw_x=support.feature_matrix()
        results=[]
        try:
            for seed in SEEDS:
                ledger.check()
                context=SequentialSeedContext(seed, support.partition(seed), study/f'runtime/seed_{seed}/context', support.read(study/'freeze.json')['git_commit'])
                if not np.array_equal(context.data.sample_id.to_numpy(),frame.sample_id.to_numpy()):
                    raise RuntimeError('formal feature/context identities differ')
                for arm in ARMS:
                    results.append(run_seed_arm(context,study,arm,acquire_callback,raw_x,decision['selected_family_by_seed'][str(seed)],ledger))
                del context
            output={'status':'COMPLETED_L1005','trajectories':results,'new_fit_attempts':len(ledger.value['attempts']),
                    'test_labels_read':0,'freeze_sha256':support.digest(study/'freeze.json')}
            support.dump(study/'formal_status.json',output)
            return output
        except BudgetExhausted as exc:
            output={'status':'BUDGET_EXHAUSTED_PARTIAL','reason':str(exc),'completed_trajectories':len(results),
                'new_fit_attempts':len(ledger.value['attempts']),'test_labels_read':0,'efficacy':'INCOMPLETE_NOT_A_NO_GO'}
            support.dump(study/'formal_status.json',output)
            return output
        except BaseException as exc:
            support.dump(study/'formal_status.json',{'status':'FAILED_PRESERVED_FOR_RESUME','reason':str(exc),
                'error_type':type(exc).__name__,'new_fit_attempts':len(ledger.value['attempts']),
                'test_labels_read':0,'efficacy':'NOT_EVALUATED'})
            raise

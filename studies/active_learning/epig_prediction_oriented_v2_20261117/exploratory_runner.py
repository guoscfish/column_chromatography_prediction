"""Exploratory, finite-budget EPIG/IVR continuation using the original predictor.

This runner does not relax or invoke the original formal study's qualification
gate. The separate v2 freeze explicitly records that the old gate remains shut.
It never calls a test-label evaluation path or changes historical artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = HERE.with_name('epig_prediction_oriented_20261117')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ORIGINAL))
import study_support as support
import epig_formal_runner as original
import numpy as np
from qgeognn_al.active_learning_v2.protocol import ids_hash, stable_hash
from qgeognn_al.active_learning_v2.sequential_acquisition import validate_trajectory_transition
from qgeognn_al.active_learning_v2.sequential_runner import SequentialSeedContext

SEEDS = (157, 887, 2357, 6101, 12203)
ARMS = ('posterior_epig_conditional_greedy32', 'posterior_ivr_conditional_greedy32')
REQUIRED = original.REQUIRED_FIT_FILES
BudgetExhausted = original.BudgetExhausted


def config_contract(config):
    """Accept no drift in the scientific design or maximum resource envelope."""
    if config.get('study_type') != 'post-stage1/stage2 exploratory development study':
        raise ValueError('explicit exploratory study nature required')
    if tuple(config['seeds']) != SEEDS or tuple(config['primary_arms']) != ARMS:
        raise ValueError('five-seed paired design required')
    if (config['initial_labels'], config['batch_size'], config['primary_rounds']) != (333, 32, 21):
        raise ValueError('L333 to L1005 schedule required')
    b = config['compute_budget']
    expected = {'target_successful_qgeognn_fits': 210, 'qgeognn_attempts_max': 231,
                'qgeognn_active_hours': 72, 'auxiliary_active_hours': 12,
                'total_active_hours': 84, 'max_concurrent_qgeognn': 1,
                'cpu_threads_per_fit': 2, 'max_fit_attempts_per_key': 3,
                'auxiliary_model_fits_max': 300, 'auxiliary_attempts_max': 330}
    if any(b.get(k) != v for k, v in expected.items()):
        raise ValueError(f'resource envelope must be {expected}')
    model = config['surrogate']
    if (model['family'], model['features'], model['max_components'],
        model['maxiter'], model['restarts']) != ('gp_matern32', 'F0_original_29', 16, 60, 1):
        raise ValueError('primary surrogate must be fixed before efficacy observations')
    if config['targets']['primary'] != 'P1_uniform_all_3330_outer_training_X' or config['targets']['formal_reference_size'] != 'all':
        raise ValueError('primary reference must use full finite P1 population')
    return config


def freeze_study(study=HERE):
    """Seal implementation/config/data dependencies before any new selection."""
    study = Path(study)
    support.verify_freeze()
    config_contract(support.read(study / 'config.json'))
    if (study / 'freeze.json').exists():
        return verify_freeze(study)
    if list((study / 'runtime').glob('seed_*/*/round_*/selection.json')):
        raise RuntimeError('cannot freeze after exploratory acquisitions')
    # A concrete passing test receipt must precede the decision freeze.
    test_path=study/'evidence/preflight_tests.json'
    tests=support.read(test_path)
    if tests.get('exit_code') != 0:
        raise RuntimeError('passing combined preflight tests required before freeze')
    junit=(ROOT/tests['junit_path']).resolve()
    if not junit.is_relative_to(study.resolve()) or support.digest(junit) != tests['junit_sha256']:
        raise RuntimeError('preflight JUnit receipt hash/path mismatch')
    # Reports can evolve; executable decision and training paths cannot.
    files = [study/name for name in ('exploratory_runner.py','acquisition_v2.py','run_exploratory.py','supervisor.py','config.json','EXPERIMENT_PROTOCOL.md')]
    files += [test_path,junit]
    if not all(p.exists() for p in files):
        raise FileNotFoundError('protocol and decision implementation must exist before freeze')
    record = {'schema': 'epig-exploratory-v2-freeze',
              'study_nature': 'post-stage1/stage2 exploratory development study',
              'created_unix': time.time(),
              'git_commit': __import__('subprocess').check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
              'files': {str(p.relative_to(ROOT)): support.digest(p) for p in files},
              'original_freeze_sha256': support.digest(ORIGINAL/'freeze.json'),
              'original_qualification_sha256': support.digest(ORIGINAL/'qualification/decision.json'),
              'original_stage2_freeze_sha256':support.digest(ORIGINAL/'stage2_nested_calibration/freeze.json'),
              'original_stage2_qualification_sha256':support.digest(ORIGINAL/'stage2_nested_calibration/qualification/decision.json'),
              'preflight_test_receipt_sha256':support.digest(test_path),
              'preflight_junit_sha256':support.digest(junit),
              'original_qualification_status': support.read(ORIGINAL/'qualification/decision.json')['status'],
              'old_gate_relaxed': False,
              'dependencies': {str(p.relative_to(ROOT)): support.digest(p) for p in support.dependency_paths()},
              'initial_new_gnn_fits': 0, 'initial_new_U_label_reveals': 0,
              'packages': support.read(ORIGINAL/'freeze.json')['packages']}
    support.write_once(study/'freeze.json', record)
    return record


def verify_freeze(study=HERE):
    study = Path(study)
    support.verify_freeze()
    record = support.read(study/'freeze.json')
    config_contract(support.read(study/'config.json'))
    for rel, digest in {**record['files'], **record['dependencies']}.items():
        if support.digest(ROOT/rel) != digest:
            raise RuntimeError(f'exploratory frozen dependency drift: {rel}')
    if record['original_freeze_sha256'] != support.digest(ORIGINAL/'freeze.json') or record['original_qualification_sha256'] != support.digest(ORIGINAL/'qualification/decision.json'):
        raise RuntimeError('original freeze or qualification changed')
    if record['original_stage2_freeze_sha256'] != support.digest(ORIGINAL/'stage2_nested_calibration/freeze.json') or record['original_stage2_qualification_sha256'] != support.digest(ORIGINAL/'stage2_nested_calibration/qualification/decision.json'):
        raise RuntimeError('original Stage2 freeze or qualification changed')
    tests=support.read(study/'evidence/preflight_tests.json')
    if tests.get('exit_code')!=0 or support.digest(study/'evidence/preflight_tests.json')!=record['preflight_test_receipt_sha256'] or support.digest(ROOT/tests['junit_path'])!=record['preflight_junit_sha256']:
        raise RuntimeError('sealed passing preflight receipt drift')
    return record


class ActiveBudgetLedger:
    """Durable finite attempts and active-compute limits, excluding downtime.

    A 2s heartbeat bounds the unobserved tail after SIGKILL. On resume that tail
    is conservatively charged by at most one interval, never the idle gap.
    A single OS lock must wrap every use of this ledger.
    """
    HEARTBEAT_SECONDS = 2.0

    def __init__(self, path, *, config, clock=time.monotonic, wall=time.time):
        self.path, self.clock, self.wall = Path(path), clock, wall
        self.limits = {k: config[k] for k in ('target_successful_qgeognn_fits','qgeognn_attempts_max',
            'qgeognn_active_hours','auxiliary_active_hours','total_active_hours',
            'max_fit_attempts_per_key','auxiliary_model_fits_max','auxiliary_attempts_max')}
        self.mutex = threading.RLock()
        if self.path.exists():
            self.value = support.read(self.path)
            if self.value['limits'] != self.limits:
                raise RuntimeError('resume budget drift')
        else:
            self.value = {'limits': self.limits, 'created_unix': wall(), 'attempts': [],
                          'active_seconds': {'qgeognn': 0., 'auxiliary': 0.},
                          'external_reservations': {'auxiliary_successful_fits':65,
                            'auxiliary_attempts':65,'auxiliary_active_seconds':9000,
                            'nature':'upper-bound reservation for offline C/D and reference; not actual consumption'}}
            self.save()
        # A held lock makes any leftover RUNNING entry an interrupted process.
        for rec in self.value['attempts']:
            if rec['status'] == 'RUNNING':
                charged = rec.get('last_heartbeat_seconds', 0.) + self.HEARTBEAT_SECONDS
                self.value['active_seconds'][rec['kind']] += charged
                rec.update(status='INTERRUPTED', active_seconds=charged,
                           recovered_unix=wall(), reason='process ended without final receipt; heartbeat tail charged')
        self.save()

    def save(self):
        support.dump(self.path, self.value)

    def seconds_remaining(self, kind):
        totals = self.value['active_seconds']
        reserved=self.value['external_reservations']['auxiliary_active_seconds']
        return min(self.limits[kind+'_active_hours']*3600-totals[kind]-(reserved if kind=='auxiliary' else 0),
                   self.limits['total_active_hours']*3600-sum(totals.values())-reserved)

    def check(self, kind='qgeognn'):
        if self.seconds_remaining(kind) <= 0:
            raise BudgetExhausted(f'cumulative {kind}/total active-compute budget exhausted')

    def completed(self):
        return {r['key'] for r in self.value['attempts'] if r['kind']=='qgeognn' and r['status']=='COMPLETED'}

    def reserve(self, key, kind):
        self.check(kind)
        records = [r for r in self.value['attempts'] if r['kind']==kind]
        if kind == 'qgeognn':
            if key in self.completed():
                raise RuntimeError('successful fit cannot reserve another attempt')
            if len(records) >= self.limits['qgeognn_attempts_max']:
                raise BudgetExhausted('231 QGeoGNN attempts exhausted; target remains 210 successful fits')
            if len(self.completed()) >= self.limits['target_successful_qgeognn_fits']:
                raise BudgetExhausted('210 successful fits already completed')
        elif len(records)+self.value['external_reservations']['auxiliary_attempts'] >= self.limits['auxiliary_attempts_max']:
            raise BudgetExhausted('330 auxiliary acquisition attempts exhausted')
        if kind == 'auxiliary' and sum(r['status']=='COMPLETED' for r in records)+self.value['external_reservations']['auxiliary_successful_fits'] >= self.limits['auxiliary_model_fits_max']:
            raise BudgetExhausted('300 successful auxiliary model fits exhausted')
        previous=[r for r in records if r['key']==str(key)]
        if previous and previous[-1]['status']=='FAILED_OR_INTERRUPTED' and previous[-1].get('error_type') not in {
                'OSError','IOError','TimeoutError','InterruptedError','KeyboardInterrupt','SystemExit','BudgetExhausted'}:
            raise RuntimeError('unresolved protocol/numerical failure cannot be automatically retried')
        if sum(r['key']==str(key) for r in records) >= self.limits['max_fit_attempts_per_key']:
            raise BudgetExhausted('three attempts for this fit/acquisition key exhausted')
        rec = {'key':str(key),'kind':kind,'status':'RUNNING','started_unix':self.wall(),
               'last_heartbeat_seconds':0.}
        self.value['attempts'].append(rec)
        self.save()
        return len(self.value['attempts'])-1

    @contextmanager
    def operation(self, key, kind):
        pos = self.reserve(key, kind)
        start = self.clock()
        stopped = threading.Event()
        def heartbeat():
            while not stopped.wait(self.HEARTBEAT_SECONDS):
                with self.mutex:
                    self.value['attempts'][pos]['last_heartbeat_seconds'] = max(0.,self.clock()-start)
                    self.save()
        worker = threading.Thread(target=heartbeat,daemon=True)
        worker.start()
        try:
            with original.fit_deadline(min(self.seconds_remaining(kind),4*3600)):
                yield pos
        except BaseException as exc:
            status, extra = 'FAILED_OR_INTERRUPTED', {'error_type':type(exc).__name__,'error':str(exc)}
            raise
        else:
            status, extra = 'COMPLETED', {}
        finally:
            stopped.set(); worker.join()
            with self.mutex:
                elapsed = max(0.,self.clock()-start)
                self.value['active_seconds'][kind] += elapsed
                self.value['attempts'][pos].update(status=status,active_seconds=elapsed,
                    ended_unix=self.wall(),**extra)
                self.save()

    def recover_completed(self, key, checkpoint_sha256):
        """Bind a completed scratch cache left between fit completion and receipt."""
        matching = [r for r in self.value['attempts'] if r['kind']=='qgeognn' and r['key']==key]
        if not matching:
            raise RuntimeError('completed model has no pre-reserved training attempt')
        matching[-1].update(status='COMPLETED',checkpoint_sha256=checkpoint_sha256,
                            recovered_complete_fit=True)
        self.save()


def fit_round(context, study, arm, r, labeled, unlabeled, truth, ledger, initial_hash):
    directory = Path(study)/f'runtime/seed_{context.seed}/{arm}/round_{r:02d}'
    directory.mkdir(parents=True,exist_ok=True)
    key = f'{context.seed}:{arm}:{r}'
    state = {'seed':context.seed,'arm':arm,'round':r,'budget':len(labeled),
             'ordered_labeled_indices':np.asarray(labeled).tolist(),
             'ordered_unlabeled_indices':np.asarray(unlabeled).tolist(),
             'L_ids_hash':ids_hash(context.ids(labeled)),'U_ids_hash':ids_hash(context.ids(unlabeled)),
             'train_truth_float32_sha256':hashlib.sha256(np.asarray(truth,dtype=np.float32).tobytes()).hexdigest(),
             'freeze_sha256':support.digest(Path(study)/'freeze.json'),
             'training_config':context.training_config(0),
             'target_scales':context.preprocessing['target_scales'],'test_labels_used':0}
    support.write_once(directory/'state.json',state)
    model, receipt_path = directory/'model', directory/'fit_receipt.json'
    if receipt_path.exists():
        receipt = support.read(receipt_path)
        if receipt['state_hash'] != stable_hash(state):
            raise RuntimeError('resumed fit state differs')
        for name,digest in receipt['files'].items():
            if support.digest(model/name) != digest:
                raise RuntimeError(f'completed fit artifact drift: {model/name}')
        if key not in ledger.completed():
            ledger.recover_completed(key,receipt['checkpoint_sha256'])
        return receipt
    original.quarantine_partial(model,directory.parent/'incomplete_attempts')
    cached = all((model/name).exists() for name in REQUIRED)
    def execute():
        return context.fit(method=arm,round_index=r,labeled_indices=labeled,labeled_truth=truth,
                           runtime=model,prediction_indices=unlabeled)
    def validate_and_write(audit):
        if (audit['initialization_hash'] != initial_hash or audit['parameter_count'] != 458952 or
            audit['model_variant'] != 'qgeognn_v2' or audit['train_rows'] != len(labeled) or
            audit['train_ids_hash'] != ids_hash(context.ids(labeled)) or
            audit['validation_ids_hash'] != ids_hash(context.ids(context.roles['validation']))):
            raise RuntimeError('predictor architecture, scratch initialization, or row identities drifted')
        if audit['checkpoint_sha256'] != support.digest(model/'best.pt') or audit['prediction_sha256'] != support.digest(model/'predictions.csv.gz'):
            raise RuntimeError('fit checkpoint or prediction hash differs from original audit')
        receipt = {'seed':context.seed,'arm':arm,'round':r,'budget':len(labeled),'reused':False,
                   'recovered_complete_fit':cached,'state_hash':stable_hash(state),
                   'checkpoint_sha256':audit['checkpoint_sha256'],'initialization_hash':audit['initialization_hash'],
                   'validation':original.validation_metrics(model,context.preprocessing['target_scales']),
                   'training_seconds':audit['training_seconds'],'model_directory':str(model.relative_to(ROOT)),
                   'files':{name:support.digest(model/name) for name in REQUIRED}}
        support.write_once(receipt_path,receipt)
        return receipt
    if cached:
        receipt=validate_and_write(execute())  # original helper checks exact cache contract
        ledger.recover_completed(key,receipt['checkpoint_sha256'])
    else:
        # Success counts only after original fit, architecture/identity checks,
        # metric validation, all hashes, and immutable fit receipt have succeeded.
        with ledger.operation(key,'qgeognn'):
            receipt=validate_and_write(execute())
    return receipt


def verify_auxiliary_receipt(selection, study):
    """Verify the exact saved probability model and acquisition behind a batch."""
    posterior=selection.get('posterior',{})
    model_path=posterior.get('model_path')
    files=posterior.get('files')
    if not model_path or not isinstance(files,dict) or set(files)!={'model.pkl','fit_audit.json','acquisition.json'}:
        raise RuntimeError('selection lacks complete posterior/acquisition artifact receipt')
    model=(ROOT/model_path).resolve()
    if not model.is_relative_to(Path(study).resolve()):
        raise RuntimeError('posterior cache lies outside this exploratory study')
    for name,digest in files.items():
        if support.digest(model.parent/name)!=digest:
            raise RuntimeError(f'posterior/acquisition artifact drift: {model.parent/name}')
    complete=support.read(model.parent/'complete.json')
    if complete!=posterior:
        raise RuntimeError('posterior completion receipt differs from selection')


class Trajectory:
    def __init__(self, context, study, arm):
        self.context,self.study,self.arm = context,Path(study),arm
        self.root = self.study/f'runtime/seed_{context.seed}/{arm}'
        self.initial = original.reuse_initial(context,self.study,arm)
        self.fits = [self.initial]
        self.store = context.new_method_store()
        self.labeled,self.unlabeled = context.roles['l0'].copy(),context.roles['u0'].copy()
        self.truth = context.l0_truth.copy()
        self.selected_history = []

    def step(self,r,callback,raw_x,ledger):
        verify_freeze(self.study)
        ledger.check('qgeognn')
        ctx = self.context
        support.dump(self.study/'exploratory_status.json',{'status':'RUNNING','current_seed':ctx.seed,
                     'current_arm':self.arm,'source_round':r,'target_budget':333+32*(r+1),
                     'successful_new_fits':len(ledger.completed()),
                     'new_fit_attempts':sum(a['kind']=='qgeognn' for a in ledger.value['attempts']),
                     'active_seconds':ledger.value['active_seconds'],'updated_unix':time.time(),
                     'efficacy':'INCOMPLETE_NOT_A_NO_GO','test_labels_read':0})
        path = self.root/f'round_{r:02d}/selection.json'
        expected = {'seed':ctx.seed,'arm':self.arm,'source_round':r,'batch_size':32,
                    'L_ids_hash':ids_hash(ctx.ids(self.labeled)),'U_ids_hash':ids_hash(ctx.ids(self.unlabeled)),
                    'source_checkpoint_sha256':self.fits[-1]['checkpoint_sha256'],
                    'freeze_sha256':support.digest(self.study/'freeze.json'),
                    'posterior_rule':'gp_matern32_F0_PCA16_maxiter60_restart1_P1full'}
        if path.exists():
            selection = support.read(path)
            if selection['input'] != expected:
                raise RuntimeError('resumed acquisition input drift')
        else:
            with ledger.operation(f'{ctx.seed}:{self.arm}:{r}','auxiliary'):
                result = callback(**original.acquisition_inputs(ctx,raw_x,self.arm,r,self.labeled,
                                                               self.unlabeled,self.truth,'gp_matern32'))
            selection = {'input':expected,'selected_ids':list(result['selected_ids']),
                         'diagnostics':result.get('diagnostics',{}),
                         'posterior':result.get('posterior',result.get('probabilistic_model_audit',{})),
                         'target_labels_read':0,'test_labels_read':0,
                         'original_gate_satisfied':False,'study_nature':'exploratory development'}
        verify_auxiliary_receipt(selection,self.study)
        access_record = {'seed':ctx.seed,'arm':self.arm,'source_round':r,
                         'selection_ids':selection['selected_ids'],
                         'before_reveal_unix':time.time(),'status':'REQUESTED',
                         'test_labels_requested':0}
        # Persist the access attempt itself, including failures before store.audit.
        access_path=self.root/f'round_{r:02d}/label_access_attempt_{time.time_ns()}.json'
        support.write_once(access_path,access_record)
        outcome_path=access_path.with_suffix('.outcome.json')
        try:
            truth = original.freeze_selection_then_reveal(self.store,path,selection,ctx.ids(self.labeled),
                        ctx.ids(self.unlabeled),self.selected_history)
        except BaseException as exc:
            access_record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),
                                 ended_unix=time.time(),calls=self.store.audit)
            support.write_once(outcome_path,access_record)
            support.dump(self.root/'label_access_audit.json',{'seed':ctx.seed,'arm':self.arm,
                         'calls':self.store.audit,'failed_access_attempt':access_record,
                         'test_truth_access_count':0,'acquisition_supervised_validation_rows':0})
            raise
        else:
            access_record.update(status='REVEALED',ended_unix=time.time(),calls=self.store.audit,
                                 selection_sha256=support.digest(path))
            support.write_once(outcome_path,access_record)
        chosen = support.read(path)['selected_ids']
        lookup = dict(zip(ctx.ids(self.unlabeled),self.unlabeled.tolist()))
        new_l = np.r_[self.labeled,[lookup[i] for i in chosen]]
        chosen_set=set(chosen)
        new_u = np.asarray([i for i in self.unlabeled if ctx.ids([i])[0] not in chosen_set],dtype=int)
        validate_trajectory_transition(ctx.ids(self.labeled),ctx.ids(self.unlabeled),chosen,
                                       ctx.ids(new_l),ctx.ids(new_u))
        self.selected_history.extend(chosen)
        self.truth = np.vstack([self.truth,truth])
        self.labeled,self.unlabeled = new_l,new_u
        support.dump(self.root/'label_access_audit.json',{'seed':ctx.seed,'arm':self.arm,'calls':self.store.audit,
                    'test_truth_access_count':0,'acquisition_supervised_validation_rows':0})
        self.fits.append(fit_round(ctx,self.study,self.arm,r+1,self.labeled,self.unlabeled,self.truth,
                                  ledger,self.initial['initialization_hash']))
        support.dump(self.root/'progress.json',{'status':'RUNNING','completed_rounds':r+1,
                    'budget':len(self.labeled),'selected_ids_hash':stable_hash(self.selected_history),'test_labels_read':0})

    def finalize(self):
        result={'status':'COMPLETED_L1005','seed':self.context.seed,'arm':self.arm,'rounds':21,
                'budget':len(self.labeled),'selected_ids_hash':stable_hash(self.selected_history),'test_labels_read':0,
                'fit_receipts':{str((self.root/f'round_{r:02d}/fit_receipt.json').relative_to(self.study)):
                 support.digest(self.root/f'round_{r:02d}/fit_receipt.json') for r in range(22)}}
        support.write_once(self.root/'trajectory_freeze.json',result)
        return result


def run_exploratory(acquire_callback,study=HERE):
    """Run/resume the full paired five-seed design; no efficacy stop exists."""
    study=Path(study)
    freeze=verify_freeze(study)
    config=config_contract(support.read(study/'config.json'))
    import torch
    torch.set_num_threads(2)
    with original.exclusive_worker(study/'runtime/exploratory.lock'):
        ledger=ActiveBudgetLedger(study/'runtime/compute_budget.json',config=config['compute_budget'])
        support.dump(study/'exploratory_status.json',{'status':'RUNNING','phase':'initializing',
                     'successful_new_fits':len(ledger.completed()),'updated_unix':time.time(),
                     'test_labels_read':0,'efficacy':'INCOMPLETE_NOT_A_NO_GO'})
        frame,raw_x=support.feature_matrix()
        results=[]
        try:
            for seed in SEEDS:
                context=SequentialSeedContext(seed,support.partition(seed),study/f'runtime/seed_{seed}/context',freeze['git_commit'])
                if not np.array_equal(context.data.sample_id.to_numpy(),frame.sample_id.to_numpy()):
                    raise RuntimeError('feature/context identity drift')
                trajectories=[Trajectory(context,study,arm) for arm in ARMS]
                # Each seed's two arms are advanced together; their L sets may diverge.
                for r in range(21):
                    for trajectory in trajectories:
                        trajectory.step(r,acquire_callback,raw_x,ledger)
                results.extend(t.finalize() for t in trajectories)
                del trajectories,context
            output={'status':'COMPLETED_L1005','study_nature':config['study_type'],
                    'trajectories':results,'successful_new_fits':len(ledger.completed()),
                    'new_fit_attempts':sum(r['kind']=='qgeognn' for r in ledger.value['attempts']),
                    'active_seconds':ledger.value['active_seconds'],'test_labels_read':0,
                    'freeze_sha256':support.digest(study/'freeze.json')}
            if output['successful_new_fits'] != 210:
                raise RuntimeError('completed trajectories lack exactly 210 successful downstream fits')
            support.dump(study/'exploratory_status.json',output)
            return output
        except BaseException as exc:
            output={'status':'BUDGET_EXHAUSTED_PARTIAL' if isinstance(exc,BudgetExhausted) else 'FAILED_PRESERVED_FOR_RESUME',
                    'reason':str(exc),'error_type':type(exc).__name__,
                    'completed_trajectories':len(results),'successful_new_fits':len(ledger.completed()),
                    'new_fit_attempts':sum(r['kind']=='qgeognn' for r in ledger.value['attempts']),
                    'active_seconds':ledger.value['active_seconds'],'test_labels_read':0,
                    'efficacy':'INCOMPLETE_NOT_A_NO_GO'}
            support.dump(study/'exploratory_status.json',output)
            raise


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--callback',help='module:function; must be part of the sealed implementation')
    args=parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_study(),ensure_ascii=False,indent=2))
    elif args.verify:
        print(json.dumps(verify_freeze(),ensure_ascii=False,indent=2))
    elif args.callback:
        name,function=args.callback.split(':')
        callback=getattr(importlib.import_module(name),function)
        module_path=Path(sys.modules[name].__file__).resolve()
        frozen=verify_freeze()
        if str(module_path.relative_to(ROOT)) not in frozen['files']:
            raise RuntimeError('acquisition callback implementation is not frozen')
        print(json.dumps(run_exploratory(callback),ensure_ascii=False,indent=2))
    else:
        parser.error('--freeze, --verify or --callback required')

if __name__=='__main__':
    main()

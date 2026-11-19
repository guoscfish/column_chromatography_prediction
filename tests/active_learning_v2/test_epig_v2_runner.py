"""Focused safety/recovery tests for the exploratory runner, without training."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'studies/active_learning/epig_prediction_oriented_v2_20261117'
sys.path.insert(0,str(HERE))
import exploratory_runner as runner


@pytest.fixture
def budget():
    return {'target_successful_qgeognn_fits':210,'qgeognn_attempts_max':231,
            'qgeognn_active_hours':72,'auxiliary_active_hours':12,'total_active_hours':84,
            'max_fit_attempts_per_key':3,'auxiliary_model_fits_max':300,'auxiliary_attempts_max':330}


def test_failures_do_not_consume_success_target_but_attempts_are_finite(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    for n in range(231):
        with pytest.raises(OSError):
            with ledger.operation(str(n),'qgeognn'):
                raise OSError('finite retry test')
    assert ledger.completed()==set()
    assert len(ledger.value['attempts'])==231
    with pytest.raises(runner.BudgetExhausted,match='231'):
        ledger.reserve('232','qgeognn')


def test_successes_do_not_reserve_twice(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    with ledger.operation('fit_1','qgeognn'):
        pass
    assert ledger.completed()=={'fit_1'}
    with pytest.raises(RuntimeError,match='another attempt'):
        ledger.reserve('fit_1','qgeognn')


def test_resume_does_not_charge_idle_wall_time(tmp_path,budget):
    path=tmp_path/'ledger.json'
    ledger=runner.ActiveBudgetLedger(path,config=budget,clock=lambda:10.,wall=lambda:100.)
    pos=ledger.reserve('interrupted','qgeognn')
    ledger.value['attempts'][pos]['last_heartbeat_seconds']=11.
    ledger.save()
    resumed=runner.ActiveBudgetLedger(path,config=budget,clock=lambda:100000.,wall=lambda:100000.)
    assert resumed.value['active_seconds']['qgeognn']==13.
    assert resumed.value['attempts'][0]['status']=='INTERRUPTED'
    again=runner.ActiveBudgetLedger(path,config=budget,clock=lambda:200000.,wall=lambda:200000.)
    assert again.value['active_seconds']['qgeognn']==13.


def test_recovered_complete_fit_requires_reserved_attempt(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    with pytest.raises(RuntimeError,match='pre-reserved'):
        ledger.recover_completed('foreign','hash')
    ledger.reserve('valid','qgeognn')
    ledger.recover_completed('valid','hash')
    assert ledger.completed()=={'valid'}


def test_active_total_and_per_kind_limits(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    ledger.value['active_seconds']['auxiliary']=12*3600
    with pytest.raises(runner.BudgetExhausted,match='auxiliary'):
        ledger.check('auxiliary')
    ledger.check('qgeognn')
    ledger.value['active_seconds']['qgeognn']=72*3600
    with pytest.raises(runner.BudgetExhausted,match='qgeognn'):
        ledger.check('qgeognn')


def test_selection_is_immutable_before_authorized_label_reveal(tmp_path):
    class Store:
        def __init__(self): self.called=False
        def freeze_acquisitions(self,ids): self.called=True
        def reveal(self,ids,purpose):
            assert (tmp_path/'selection.json').exists()
            assert (tmp_path/'selection.sha256.json').exists()
            assert self.called
            assert purpose=='after_acquisition_fit'
            return ids
    chosen=[f'u{i}' for i in range(32)]
    store=Store()
    result=runner.original.freeze_selection_then_reveal(store,tmp_path/'selection.json',
            {'selected_ids':chosen},['l0'],chosen,[])
    assert result==chosen
    altered={'selected_ids':chosen[::-1]}
    with pytest.raises(RuntimeError,match='immutable'):
        runner.original.freeze_selection_then_reveal(store,tmp_path/'selection.json',altered,['l0'],chosen,[])


def test_invalid_selection_does_not_reveal(tmp_path):
    class Store:
        def freeze_acquisitions(self,ids): raise AssertionError('must not reach label store')
    with pytest.raises(ValueError,match='exactly 32'):
        runner.original.freeze_selection_then_reveal(Store(),tmp_path/'selection.json',
                {'selected_ids':['u1']},['l0'],['u1'],[])


def test_per_key_retry_cap_is_three(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    for _ in range(3):
        with pytest.raises(OSError):
            with ledger.operation('same_fit','qgeognn'):
                raise OSError('transient I/O')
    with pytest.raises(runner.BudgetExhausted,match='three attempts'):
        ledger.reserve('same_fit','qgeognn')


def test_offline_reservations_leave_callback_budget_without_claiming_consumption(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    assert ledger.value['active_seconds']=={'qgeognn':0.,'auxiliary':0.}
    assert ledger.seconds_remaining('auxiliary')==9.5*3600
    assert ledger.value['external_reservations']['auxiliary_successful_fits']==65
    assert ledger.seconds_remaining('qgeognn')==72*3600


class FakeContext:
    seed=157
    roles={'validation':[3]}
    preprocessing={'target_scales':{'V1':1.,'V2':2.}}
    parameter_count=458952
    def training_config(self,member): return {'scratch':True,'maximum_epochs':1000}
    def ids(self,indices): return [f'id{int(i)}' for i in indices]
    def fit(self,**kwargs):
        model=kwargs['runtime'];model.mkdir(parents=True,exist_ok=True)
        for name in runner.REQUIRED: (model/name).write_text(name)
        return {'initialization_hash':'initial','parameter_count':self.parameter_count,
                'model_variant':'qgeognn_v2','train_rows':len(kwargs['labeled_indices']),
                'train_ids_hash':runner.ids_hash(self.ids(kwargs['labeled_indices'])),
                'validation_ids_hash':runner.ids_hash(self.ids([3])),
                'checkpoint_sha256':runner.support.digest(model/'best.pt'),
                'prediction_sha256':runner.support.digest(model/'predictions.csv.gz'),
                'training_seconds':1.}


def test_completed_fit_resume_checks_hash_and_state_without_new_attempt(tmp_path,budget,monkeypatch):
    import numpy as np
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.original,'validation_metrics',lambda model,scales:{'combined_nrmse':0.5})
    (tmp_path/'freeze.json').write_text('{}')
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    args=(FakeContext(),tmp_path,runner.ARMS[0],1,np.array([0,1]),np.array([2]),np.ones((2,2)),ledger,'initial')
    first=runner.fit_round(*args)
    assert len(ledger.completed())==1 and len(ledger.value['attempts'])==1
    assert runner.fit_round(*args)==first
    assert len(ledger.value['attempts'])==1
    with pytest.raises(RuntimeError,match='immutable'):
        runner.fit_round(*args[:6],np.zeros((2,2)),*args[7:])
    model=tmp_path/f'runtime/seed_157/{runner.ARMS[0]}/round_01/model'
    (model/'best.pt').write_text('corrupted')
    with pytest.raises(RuntimeError,match='artifact drift'):
        runner.fit_round(*args)
    assert len(ledger.value['attempts'])==1


def test_invalid_fit_is_failed_not_successful(tmp_path,budget,monkeypatch):
    import numpy as np
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'freeze.json').write_text('{}')
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    context=FakeContext();context.parameter_count=1
    with pytest.raises(RuntimeError,match='architecture'):
        runner.fit_round(context,tmp_path,runner.ARMS[0],1,np.array([0,1]),np.array([2]),
                         np.ones((2,2)),ledger,'initial')
    assert ledger.completed()==set()
    assert ledger.value['attempts'][0]['status']=='FAILED_OR_INTERRUPTED'
    assert not (tmp_path/f'runtime/seed_157/{runner.ARMS[0]}/round_01/fit_receipt.json').exists()


def test_partial_scratch_fit_is_preserved(tmp_path):
    model=tmp_path/'round_01/model';model.mkdir(parents=True)
    (model/'best.pt').write_text('partial checkpoint')
    destination=runner.original.quarantine_partial(model,tmp_path/'incomplete_attempts')
    assert not model.exists()
    assert (destination/'best.pt').read_text()=='partial checkpoint'


def test_unresolved_protocol_failure_is_not_retried(tmp_path,budget):
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    with pytest.raises(RuntimeError,match='numerical'):
        with ledger.operation('bad_fit','qgeognn'):
            raise RuntimeError('numerical invalidity')
    with pytest.raises(RuntimeError,match='cannot be automatically retried'):
        ledger.reserve('bad_fit','qgeognn')
    assert len(ledger.value['attempts'])==1


def test_auxiliary_artifacts_are_hash_verified(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    directory=tmp_path/'study/aux';directory.mkdir(parents=True)
    files={}
    for name in ('model.pkl','fit_audit.json','acquisition.json'):
        (directory/name).write_text(name)
        files[name]=runner.support.digest(directory/name)
    posterior={'model_path':'study/aux/model.pkl','files':files}
    runner.support.write_once(directory/'complete.json',posterior)
    selection={'posterior':posterior}
    runner.verify_auxiliary_receipt(selection,tmp_path/'study')
    (directory/'model.pkl').write_text('drift')
    with pytest.raises(RuntimeError,match='artifact drift'):
        runner.verify_auxiliary_receipt(selection,tmp_path/'study')


def test_failed_label_access_has_durable_request_and_outcome(tmp_path,budget,monkeypatch):
    import numpy as np
    monkeypatch.setattr(runner,'verify_freeze',lambda study:None)
    monkeypatch.setattr(runner,'verify_auxiliary_receipt',lambda selection,study:None)
    (tmp_path/'freeze.json').write_text('{}')
    ledger=runner.ActiveBudgetLedger(tmp_path/'ledger.json',config=budget)
    context=FakeContext()
    trajectory=runner.Trajectory.__new__(runner.Trajectory)
    trajectory.context=context;trajectory.study=tmp_path;trajectory.arm=runner.ARMS[0]
    trajectory.root=tmp_path/f'runtime/seed_157/{runner.ARMS[0]}'
    trajectory.labeled=np.array([0,1]);trajectory.unlabeled=np.arange(2,34)
    trajectory.fits=[{'checkpoint_sha256':'checkpoint'}];trajectory.selected_history=[]
    class Store:
        audit=[]
        def freeze_acquisitions(self,ids): raise AssertionError('bad IDs must never reach store')
    trajectory.store=Store()
    path=trajectory.root/'round_00/selection.json'
    expected={'seed':157,'arm':runner.ARMS[0],'source_round':0,'batch_size':32,
              'L_ids_hash':runner.ids_hash(context.ids(trajectory.labeled)),
              'U_ids_hash':runner.ids_hash(context.ids(trajectory.unlabeled)),
              'source_checkpoint_sha256':'checkpoint','freeze_sha256':runner.support.digest(tmp_path/'freeze.json'),
              'posterior_rule':'gp_matern32_F0_PCA16_maxiter60_restart1_P1full'}
    runner.support.write_once(path,{'input':expected,'selected_ids':[f'outside{i}' for i in range(32)]})
    with pytest.raises(PermissionError):
        trajectory.step(0,None,None,ledger)
    outcomes=list(path.parent.glob('label_access_attempt_*.outcome.json'))
    requests=[p for p in path.parent.glob('label_access_attempt_*.json') if '.outcome.' not in p.name]
    assert len(outcomes)==len(requests)==1
    assert runner.support.read(requests[0])['status']=='REQUESTED'
    assert runner.support.read(outcomes[0])['status']=='FAILED'
    assert runner.support.read(trajectory.root/'label_access_audit.json')['failed_access_attempt']['error_type']=='PermissionError'


def test_freeze_requires_passing_hashed_preflight_and_binds_stage2(tmp_path,monkeypatch):
    import subprocess
    study=tmp_path/'new';study.mkdir()
    old=tmp_path/'old';(old/'qualification').mkdir(parents=True)
    (old/'stage2_nested_calibration/qualification').mkdir(parents=True)
    config=json.loads((HERE/'config.json').read_text())
    runner.support.dump(study/'config.json',config)
    for name in ('exploratory_runner.py','acquisition_v2.py','run_exploratory.py','supervisor.py','EXPERIMENT_PROTOCOL.md'):
        (study/name).write_text('sealed content')
    runner.support.dump(old/'freeze.json',{'packages':{}})
    runner.support.dump(old/'qualification/decision.json',{'status':'INCOMPLETE'})
    runner.support.dump(old/'stage2_nested_calibration/freeze.json',{'stage2':True})
    runner.support.dump(old/'stage2_nested_calibration/qualification/decision.json',{'status':'INCOMPLETE'})
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'ORIGINAL',old)
    monkeypatch.setattr(runner.support,'verify_freeze',lambda:None)
    monkeypatch.setattr(runner.support,'dependency_paths',lambda:[])
    monkeypatch.setattr(subprocess,'check_output',lambda *args,**kwargs:'commit\n')
    junit=study/'evidence/tests.xml';junit.parent.mkdir();junit.write_text('<testsuite failures="0"/>')
    receipt=study/'evidence/preflight_tests.json'
    runner.support.dump(receipt,{'exit_code':1,'junit_path':'new/evidence/tests.xml','junit_sha256':'invalid'})
    with pytest.raises(RuntimeError,match='passing combined'):
        runner.freeze_study(study)
    runner.support.dump(receipt,{'exit_code':0,'junit_path':'new/evidence/tests.xml','junit_sha256':'invalid'})
    with pytest.raises(RuntimeError,match='JUnit'):
        runner.freeze_study(study)
    runner.support.dump(receipt,{'exit_code':0,'junit_path':'new/evidence/tests.xml','junit_sha256':runner.support.digest(junit)})
    freeze=runner.freeze_study(study)
    assert freeze['old_gate_relaxed'] is False
    assert runner.verify_freeze(study)==freeze
    (old/'stage2_nested_calibration/qualification/decision.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Stage2'):
        runner.verify_freeze(study)

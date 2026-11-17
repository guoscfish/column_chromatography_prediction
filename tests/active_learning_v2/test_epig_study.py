"""Study boundary/resume tests use stand-ins; no neural training or raw labels."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

STUDY = Path(__file__).resolve().parents[2] / 'studies/active_learning/epig_prediction_oriented_20261117'
sys.path.insert(0, str(STUDY))
spec = importlib.util.spec_from_file_location('epig_formal_runner_for_test', STUDY/'epig_formal_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_acquisition_capability_contains_no_validation_targets_or_covariates():
    context = SimpleNamespace(seed=157, outer=np.array([0,2,4]),
        data=pd.DataFrame({'sample_id':['a','v','b','t','c'], 'value':[1.,9.,2.,9.,3.]}),
        preprocessing={'target_scales':{'V1':2.,'V2':4.}})
    result=runner.acquisition_inputs(context,np.arange(15).reshape(5,3),'gp_epig_conditional_greedy32',0,
                                     [0],[2,4],np.array([[2.,3.]]),'gp_rbf')
    assert set(result)=={'seed','arm','round_index','labeled_indices','unlabeled_indices','labeled_truth',
                        'feature_matrix_X','feature_frame','target_scales','previous_checkpoint_path'}
    assert not any('validation' in k or 'context' in k for k in result)
    assert np.isnan(result['feature_matrix_X'][[1,3]]).all()
    assert result['feature_frame'].iloc[[1,3]].isna().all().all()
    assert result['feature_frame'].iloc[[0,2,4]].sample_id.tolist()==['a','b','c']
    assert result['feature_matrix_X'].flags.writeable is False
    assert result['labeled_truth'].flags.writeable is False


def test_selection_freeze_bytes_exist_before_label_authorization(tmp_path):
    path=tmp_path/'selection.json'
    ids=[f'u{i}' for i in range(32)]
    events=[]
    class Store:
        def freeze_acquisitions(self, selected):
            assert path.exists() and path.with_suffix('.sha256.json').exists()
            assert runner.support.read(path)['selected_ids']==ids
            assert runner.support.read(path.with_suffix('.sha256.json'))['sha256']==runner.support.digest(path)
            events.append('freeze')
        def reveal(self, selected, purpose):
            assert events==['freeze'] and purpose=='after_acquisition_fit'
            events.append('reveal')
            return np.ones((32,2))
    truth=runner.freeze_selection_then_reveal(Store(),path,{'selected_ids':ids},['l0'],ids,[])
    assert truth.shape==(32,2) and events==['freeze','reveal']


@pytest.mark.parametrize('bad', [list(range(31)), list(range(31))+[0], list(range(31))+['validation'],list(range(31))+['l0']])
def test_illegal_or_duplicate_ids_never_trigger_label_read(tmp_path,bad):
    class Store:
        def freeze_acquisitions(self,*args):
            pytest.fail('invalid IDs reached authorization')
        def reveal(self,*args):
            pytest.fail('invalid IDs reached raw label source')
    with pytest.raises((ValueError,PermissionError)):
        runner.freeze_selection_then_reveal(Store(),tmp_path/'selection.json',{'selected_ids':bad},['l0'],list(map(str,range(40))),[])
    assert not (tmp_path/'selection.json').exists()


def test_changed_frozen_selection_cannot_reveal_again(tmp_path):
    path=tmp_path/'selection.json'
    ids=list(map(str,range(32)))
    runner.support.write_once(path,{'selected_ids':ids,'selected_ids_hash':'bad'})
    class Store:
        def freeze_acquisitions(self,*args):
            pytest.fail('changed receipt reached label source')
    with pytest.raises(RuntimeError,match='immutable'):
        runner.freeze_selection_then_reveal(Store(),path,{'selected_ids':ids},[],ids,[])


def test_partial_fit_moves_without_overwriting_and_complete_fit_stays(tmp_path):
    model=tmp_path/'round_01/model'
    model.mkdir(parents=True)
    (model/'best.pt').write_bytes(b'partial-weights')
    saved=runner.quarantine_partial(model,tmp_path/'incomplete_attempts')
    assert saved.exists() and (saved/'best.pt').read_bytes()==b'partial-weights'
    assert not model.exists()
    model.mkdir()
    for name in runner.REQUIRED_FIT_FILES:
        (model/name).write_bytes(b'complete')
    assert runner.quarantine_partial(model,tmp_path/'incomplete_attempts') is None
    assert model.exists()


def test_cumulative_budget_survives_resume_and_counts_interrupted_attempts(tmp_path):
    now=[100.]
    clock=lambda:now[0]
    path=tmp_path/'compute.json'
    first=runner.BudgetLedger(path,fit_limit=2,wall_hours=1,clock=clock)
    first.finish(first.reserve('fit1'),'FAILED')
    resumed=runner.BudgetLedger(path,fit_limit=2,wall_hours=1,clock=clock)
    resumed.finish(resumed.reserve('fit1-retry'),'COMPLETED')
    with pytest.raises(runner.BudgetExhausted,match='210 launched'):
        resumed.reserve('fit3')
    now[0]=3701.
    with pytest.raises(runner.BudgetExhausted,match='wall budget'):
        resumed.check()


def test_unqualified_decision_stops_before_context_or_label_read(tmp_path,monkeypatch):
    (tmp_path/'qualification').mkdir()
    runner.support.dump(tmp_path/'qualification/decision.json',{'status':'NO_QUALIFIED_GP','selected_gp_family':None})
    monkeypatch.setattr(runner.support,'verify_freeze',lambda:None)
    with pytest.raises(RuntimeError,match='qualified preregistered posterior'):
        runner.qualification_gate(tmp_path)


def fake_context(tmp_path):
    calls=[]
    class Context:
        seed=157
        preprocessing={'target_scales':{'V1':2.,'V2':4.}}
        roles={'validation':np.array([4])}
        def ids(self,indices):
            return [f'id{i}' for i in indices]
        def training_config(self,member):
            return {'maximum_epochs':1000,'patience':100,'model_variant':'qgeognn_v2','seed':3000174}
        def fit(self,**kw):
            calls.append(kw)
            assert kw['prediction_indices'].tolist()==[2,3]
            model=kw['runtime'];model.mkdir(exist_ok=True)
            (model/'best.pt').write_bytes(b'real-fixture-weights')
            (model/'predictions.csv.gz').write_bytes(b'fixture-U-predictions')
            pd.DataFrame([dict(epoch=1,validation_V1_rmse=1.,validation_V2_rmse=2.,validation_V1_mae=.8,
                validation_V2_mae=1.5,validation_combined_normalized_rmse=.5)]).to_csv(model/'history.csv',index=False)
            result=dict(initialization_hash='init',parameter_count=458952,model_variant='qgeognn_v2',
                train_ids_hash=runner.ids_hash(self.ids(kw['labeled_indices'])),
                validation_ids_hash=runner.ids_hash(self.ids(self.roles['validation'])),
                best_epoch=1,epochs_run=1,best_validation_combined_normalized_rmse=.5,
                checkpoint_sha256=runner.support.digest(model/'best.pt'),training_seconds=1.)
            runner.support.dump(model/'fit_audit.json',result)
            return result
    return Context(),calls


def test_completed_round_resume_never_refits_and_checks_hashes(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.support,'ROOT',tmp_path)
    study=tmp_path/'study'
    (study/'qualification').mkdir(parents=True)
    runner.support.dump(study/'freeze.json',{'fixture':True})
    runner.support.dump(study/'qualification/decision.json',{'fixture':True})
    context,calls=fake_context(tmp_path)
    ledger=runner.BudgetLedger(study/'budget.json')
    args=(context,study,runner.ARMS[0],1,np.array([0,1]),np.array([2,3]),np.array([[1.,2.],[3.,4.]]),ledger,'init')
    first=runner.fit_round(*args)
    assert len(calls)==1 and len(ledger.value['attempts'])==1
    second=runner.fit_round(*args)
    assert second==first and len(calls)==1 and len(ledger.value['attempts'])==1
    model=tmp_path/first['model_directory']
    (model/'best.pt').write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='artifact drift'):
        runner.fit_round(*args)
    assert len(calls)==1


def test_same_ids_with_changed_revealed_values_cannot_reuse_completed_fit(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.support,'ROOT',tmp_path)
    study=tmp_path/'study';(study/'qualification').mkdir(parents=True)
    runner.support.dump(study/'freeze.json',{})
    runner.support.dump(study/'qualification/decision.json',{})
    context,calls=fake_context(tmp_path)
    ledger=runner.BudgetLedger(study/'budget.json')
    common=(context,study,runner.ARMS[0],1,np.array([0,1]),np.array([2,3]))
    runner.fit_round(*common,np.ones((2,2)),ledger,'init')
    with pytest.raises(RuntimeError,match='immutable receipt'):
        runner.fit_round(*common,np.zeros((2,2)),ledger,'init')
    assert len(calls)==1


def qualified_fixture(tmp_path, monkeypatch):
    (tmp_path/'qualification').mkdir()
    runner.support.dump(tmp_path/'freeze.json',{'registered':True})
    config={'seeds':list(runner.SEEDS),'primary_arms':list(runner.ARMS),'initial_labels':333,'batch_size':32,'primary_rounds':21,
        'compute_budget':{'new_primary_qgeognn_fits_max':210,'formal_wall_budget_hours':40,'max_concurrent_qgeognn':1,'cpu_threads_per_fit':2}}
    runner.support.dump(tmp_path/'config.json',config)
    decision={'status':'QUALIFIED','run_allowed':True,'freeze_sha256':runner.support.digest(tmp_path/'freeze.json'),
              'selected_family_by_seed':{str(seed):('gp_rbf' if i<3 else 'blr_coregionalized') for i,seed in enumerate(runner.SEEDS)}}
    runner.support.dump(tmp_path/'qualification/decision.json',decision)
    monkeypatch.setattr(runner.support,'verify_freeze',lambda:None)
    return decision


def test_seed_local_gp_or_blr_fallback_can_qualify_without_relabelling_as_gp(tmp_path,monkeypatch):
    decision=qualified_fixture(tmp_path,monkeypatch)
    got,_=runner.qualification_gate(tmp_path)
    assert got==decision
    assert 'blr_coregionalized' in got['selected_family_by_seed'].values()
    assert runner.ARMS==('posterior_epig_conditional_greedy32','posterior_ivr_conditional_greedy32')


@pytest.mark.parametrize('change',['missing_seed','false_run_allowed','stale_freeze'])
def test_qualification_requires_all_five_seed_local_models_and_current_freeze(tmp_path,monkeypatch,change):
    decision=qualified_fixture(tmp_path,monkeypatch)
    if change=='missing_seed':
        decision['selected_family_by_seed'].pop('157')
    elif change=='false_run_allowed':
        decision['run_allowed']=False
    else:
        decision['freeze_sha256']='a-different-preregistration'
    runner.support.dump(tmp_path/'qualification/decision.json',decision)
    with pytest.raises(RuntimeError):
        runner.qualification_gate(tmp_path)

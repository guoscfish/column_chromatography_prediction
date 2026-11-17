"""Isolation and unchanged formal-gate tests; no neural training or raw labels."""
import importlib.util
from pathlib import Path
import sys

import pytest

STUDY=Path(__file__).resolve().parents[2]/'studies/active_learning/epig_prediction_oriented_20261117'
sys.path.insert(0,str(STUDY))
spec=importlib.util.spec_from_file_location('stage2_adapter_test_module',STUDY/'stage2_formal_adapter.py')
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture(tmp_path,monkeypatch,qualified=True):
    first=tmp_path/'stage1';second=first/'stage2_nested_calibration'
    second.mkdir(parents=True)
    original=first/'epig_formal_runner.py'
    original.write_bytes((STUDY/'epig_formal_runner.py').read_bytes())
    monkeypatch.setattr(adapter,'STAGE1',first)
    monkeypatch.setattr(adapter.stage1_support,'ROOT',tmp_path)
    calls=[]
    monkeypatch.setattr(adapter.stage1_support,'verify_freeze',lambda:calls.append('stage1'))
    adapter.stage1_support.dump(first/'freeze.json',{'files':{str(original.relative_to(tmp_path)):adapter.stage1_support.digest(original)}})
    adapter.stage1_support.dump(first/'features.json',{'shape':[4163,29],'labels_read':0})
    record={'stage1_freeze_sha256':adapter.stage1_support.digest(first/'freeze.json'),
            'git_commit':'fixture','files':{'stage2_driver.py':'fixture'},'data_hashes':{},'packages':{}}
    adapter.stage1_support.dump(second/'freeze.json',record)
    decision={'status':'QUALIFIED' if qualified else 'INCOMPLETE_POSTERIOR_QUALIFICATION','run_allowed':qualified,
        'selected_family_by_seed':{str(v):'gp_rbf' if qualified else None for v in (157,887,2357,6101,12203)},
        'freeze_sha256':adapter.stage1_support.digest(second/'freeze.json')}
    adapter.stage1_support.dump(second/'qualification/decision.json',decision)
    adapter.stage1_support.dump(second/'config.json',{'seeds':[157,887,2357,6101,12203],
        'primary_arms':['posterior_epig_conditional_greedy32','posterior_ivr_conditional_greedy32'],
        'initial_labels':333,'batch_size':32,'primary_rounds':21,
        'compute_budget':{'new_primary_qgeognn_fits_max':210,'formal_wall_budget_hours':40,
            'max_concurrent_qgeognn':1,'cpu_threads_per_fit':2}})
    def verify_second():
        calls.append('stage2')
        return adapter.stage1_support.read(second/'freeze.json')
    return first,second,verify_second,calls


def test_private_facade_verifies_both_without_changing_original_module(tmp_path,monkeypatch):
    first,second,verify,calls=fixture(tmp_path,monkeypatch)
    original_verify=adapter.stage1_support.verify_freeze
    loaded=adapter.load_isolated_runner(second,verify)
    assert loaded.support is not adapter.stage1_support
    assert adapter.stage1_support.verify_freeze is original_verify
    loaded.support.verify_freeze()
    assert calls==['stage1','stage2','stage1','stage2']
    assert loaded.support.HERE==second
    assert loaded.support.BASE==adapter.stage1_support.BASE


def test_runner_writes_are_confined_to_stage_two(tmp_path,monkeypatch):
    first,second,verify,_=fixture(tmp_path,monkeypatch)
    loaded=adapter.load_isolated_runner(second,verify)
    loaded.support.write_once(second/'runtime/state.json',{'legal':True})
    with pytest.raises(PermissionError,match='escapes'):
        loaded.support.dump(first/'overwrite.json',{})
    with pytest.raises(PermissionError,match='escapes'):
        loaded.support.write_once(tmp_path/'outside.json',{})
    assert not (first/'overwrite.json').exists()
    assert not (tmp_path/'outside.json').exists()


def test_stage_two_cannot_be_stage_one_outside_or_symlink_escape(tmp_path,monkeypatch):
    first,second,verify,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(PermissionError):
        adapter.load_isolated_runner(first,verify)
    with pytest.raises(PermissionError):
        adapter.load_isolated_runner(tmp_path/'elsewhere',verify)
    (first/'escape').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(PermissionError):
        adapter.load_isolated_runner(first/'escape',verify)


def test_missing_parent_binding_and_changed_runner_bytes_reject(tmp_path,monkeypatch):
    first,second,verify,_=fixture(tmp_path,monkeypatch)
    record=adapter.stage1_support.read(second/'freeze.json')
    record['stage1_freeze_sha256']='wrong'
    adapter.stage1_support.dump(second/'freeze.json',record)
    with pytest.raises(RuntimeError,match='does not bind'):
        adapter.load_isolated_runner(second,verify)
    record['stage1_freeze_sha256']=adapter.stage1_support.digest(first/'freeze.json')
    adapter.stage1_support.dump(second/'freeze.json',record)
    (first/'epig_formal_runner.py').write_text('# drifted runner')
    with pytest.raises(RuntimeError,match='runner bytes differ'):
        adapter.load_isolated_runner(second,verify)


def test_unqualified_stage_two_stops_before_any_training_context(tmp_path,monkeypatch):
    first,second,verify,_=fixture(tmp_path,monkeypatch,qualified=False)
    with pytest.raises(RuntimeError,match='qualified preregistered posterior'):
        adapter.run_stage2_formal(lambda **kw:pytest.fail('acquisition must not run'),second,verify)
    assert not (second/'runtime').exists()
    assert not (first/'runtime').exists()


def test_original_gate_accepts_plain_seed_families_and_unchanged_limits(tmp_path,monkeypatch):
    first,second,verify,_=fixture(tmp_path,monkeypatch)
    loaded=adapter.load_isolated_runner(second,verify)
    decision,config=loaded.qualification_gate(second)
    assert decision['run_allowed']
    assert config['compute_budget']['new_primary_qgeognn_fits_max']==210
    config['compute_budget']['max_concurrent_qgeognn']=2
    adapter.stage1_support.dump(second/'config.json',config)
    with pytest.raises(RuntimeError,match='one worker'):
        loaded.qualification_gate(second)

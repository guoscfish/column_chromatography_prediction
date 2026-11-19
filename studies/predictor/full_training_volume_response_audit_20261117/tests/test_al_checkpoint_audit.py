"""Checkpoint-budget disambiguation and fail-closed pair permission checks."""
from pathlib import Path
import importlib.util
import json
import pytest

spec = importlib.util.spec_from_file_location('al_checkpoint_audit_unit', Path(__file__).parents[1] / 'al_checkpoint_audit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_budget_is_not_formal_seed():
    assert 'round_06' in str(module.checkpoint_path(157, 525))
    assert 'round_21' in str(module.checkpoint_path(157, 1005))
    assert 'round_36' in str(module.checkpoint_path(157, 1485))
    with pytest.raises(ValueError):
        module.checkpoint_path(525, 525)


def test_exposure_class_test_never_legalized_by_train_other_endpoint():
    assert module.exposure_class('train', 'train') == 'both_source_train'
    assert module.exposure_class('train', 'validation') == 'mixed_source_train_validation'
    assert module.exposure_class('validation', 'validation') == 'both_source_validation'
    assert module.exposure_class('train', 'test') == 'source_test_involved'
    assert module.exposure_class('test', 'validation') == 'source_test_involved'
    assert module.exposure_class('unmapped', 'train') == 'unmapped_or_other'


def test_pair_key_orientation_does_not_forge_new_pair():
    assert module.pair_key('V/ul', 'a', 'b') == module.pair_key('V/ul', 'b', 'a')
    assert module.pair_key('V/ul', 'a', 'b') != module.pair_key('PE/EA', 'a', 'b')


def test_quarantine_requires_explicit_id_array(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path / 'incident.json'
    path.write_text(json.dumps({'quarantined_ids': ['a']}))
    assert module.load_quarantine(path) == {'a'}
    path.write_text(json.dumps({'status': 'no IDs supplied'}))
    with pytest.raises(ValueError):
        module.load_quarantine(path)


def test_permission_blocks_test_and_global_incident_even_for_train_pairs():
    assert module.scoring_permission('a', 'b', 'train', 'validation', set())[2]
    assert not module.scoring_permission('a', 'b', 'train', 'test', set())[2]
    kind, quarantined, allowed, reason = module.scoring_permission('a', 'b', 'train', 'train', {'b'})
    assert kind == 'both_source_train' and quarantined and not allowed
    assert reason == 'global_incident_quarantine'
    assert not module.scoring_permission('a', 'b', 'train', 'unmapped', set())[2]

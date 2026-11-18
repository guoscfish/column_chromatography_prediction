"""Read-only model-identity and rejection tests; no forward pass or labels."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoint_audit as audit


def test_three_formal_models_use_official_native_schema():
    for seed in (42, 525, 1101):
        model, identity = audit.load_formal(seed)
        assert not model.training
        assert identity['native_v2'] is True
        assert identity['conversion_performed'] is False
        assert identity['loader_state_bitwise_equal'] is True
        assert identity['checkpoint_sha256'] == audit.EXPECTED_SHA256[seed]
        assert identity['role_counts'] == {'train': 3330, 'validation': 416, 'test': 417}
        assert identity['training_rows'] == identity['preprocessing']['fit_rows'] == 3330
        assert identity['fit_ids_hash'] == identity['normalization']['fit_ids_hash']
        assert sum(p.numel() for p in model.parameters()) == 458952
        del model
        gc.collect()


@pytest.mark.parametrize('seed', [157, '42', True, 525.0])
def test_nonformal_seed_is_rejected(seed):
    with pytest.raises(ValueError, match='Only formal Row'):
        audit.load_formal(seed)


def test_wrong_checkpoint_sha_is_rejected(monkeypatch):
    monkeypatch.setitem(audit.EXPECTED_SHA256, 42, '0'*64)
    with pytest.raises(ValueError, match='Formal checkpoint SHA mismatch'):
        audit.load_formal(42)


@pytest.mark.parametrize('change', ['failed', 'nonzero', 'keys', 'bitwise'])
def test_invalid_official_equivalence_evidence_is_rejected(change):
    evidence = json.loads((audit.ENGINEERING/'equivalence_audit.json').read_text())
    bad = deepcopy(evidence)
    if change == 'failed':
        bad['status'] = 'FAIL'
    elif change == 'nonzero':
        bad['max_abs_difference'] = 1e-7
    elif change == 'keys':
        bad['conversion']['missing_keys'] = ['example']
    else:
        bad['conversion']['retained_values_bitwise_equal'] = False
    with pytest.raises(ValueError):
        audit.check_engineering_evidence(bad)


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('model_variant', 'legacy'),
                                        ('input_schema_hash', '0'*64), ('output_order', [])])
def test_official_schema_validation_remains_strict(field, value):
    payload = audit.torch.load(audit.FORMAL/'runtime/row/seed_42/best.pt',
                               map_location='cpu', weights_only=False)
    payload[field] = value
    with pytest.raises(ValueError):
        audit.validate_predictor_checkpoint(payload)

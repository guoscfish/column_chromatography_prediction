import inspect
import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from scripts.studies import run_r2_pruned_requalification as old
from scripts.studies import run_final_4g_qualification as qualification
from src.qgeognn_al.historical.conversion import convert_pruned_to_standalone
from src.qgeognn_al.models import build_predictor, load_predictor_checkpoint, predictor_checkpoint
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.training import predictor as training


@pytest.fixture(scope="module")
def inputs():
    return old.inputs()


def test_direct_builder_has_no_historical_or_clean_model_dependency():
    code = """
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.schemas.conditions import ConditionNormalization
import sys
assert not any(('clean_fusion' in k or 'clean_schema' in k or 'condition_complete_v2' in k) for k in sys.modules)
normalization = ConditionNormalization(0, 1, 0, 1, '4g', 'source_train', 1, '0' * 64, '0' * 64)
model = build_predictor(normalization)
assert sum(p.numel() for p in model.parameters()) == 458952
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    source = inspect.getsource(build_predictor)
    assert "legacy" not in source.lower() and "prun" not in source.lower()


def test_standalone_mapping_and_multibatch_reachability(inputs):
    *_, cn, norm, scales, batches = inputs
    legacy = old.mapped_model(old.REFERENCE, cn, norm, torch.device("cpu"))
    model = build_predictor(ConditionNormalization(**asdict(norm)))
    converted, audit = convert_pruned_to_standalone(legacy.state_dict(), model)
    assert audit["retained_values_bitwise_equal"]
    model.load_state_dict(converted)
    model.eval(); legacy.eval()
    with torch.no_grad():
        for a, b in batches:
            assert torch.max(torch.abs(model(a, b)-legacy(a, b)[0])).item() <= 1e-6
    model.train()
    for a, b in batches:
        model.zero_grad(set_to_none=True)
        model(a, b).sum().backward()
        assert sum(p.numel() for p in model.parameters()) == 458952
        assert all(p.requires_grad and p.grad is not None for p in model.parameters())
    assert len(model.backbone.convs) == 5
    assert len(model.backbone.convs_bond_angle) == 4
    assert not any(isinstance(m, (torch.nn.LayerNorm, torch.nn.Dropout)) for m in model.modules())


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape"])
def test_migration_rejects_bad_states(inputs, problem):
    *_, cn, norm, scales, batches = inputs
    pruned = old.mapped_model(old.REFERENCE, cn, norm, torch.device("cpu"))
    model = build_predictor(ConditionNormalization(**asdict(norm)))
    state = dict(pruned.state_dict())
    key = next(iter(state))
    if problem == "missing":
        del state[key]
    elif problem == "unexpected":
        state["unknown"] = torch.ones(1)
    else:
        state[key] = torch.ones(1)
    with pytest.raises(ValueError):
        convert_pruned_to_standalone(state, model)


def test_checkpoint_roundtrip_and_contract_rejection(inputs, tmp_path):
    *_, cn, norm, scales, batches = inputs
    model = build_predictor(ConditionNormalization(**asdict(norm)))
    path = tmp_path / "model.pt"
    payload = predictor_checkpoint(model, preprocessing={"fit_role": "source_train"}, training_config={}, provenance={})
    torch.save(payload, path)
    loaded = load_predictor_checkpoint(path)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    payload["model_variant"] = "legacy"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_predictor_checkpoint(path)


def test_frozen_qualification_manifests_and_training_isolation():
    manifest = json.loads((qualification.STUDY / "splits/split_manifest.json").read_text())
    assert manifest["regenerated"] is False
    assert len(manifest["splits"]) == 6
    for item in manifest["splits"]:
        path = qualification.STUDY / "splits" / f"{item['split_mode']}_seed_{item['seed']}.csv"
        assert qualification.sha256_file(path) == item["sha256"]
    assert "test" not in inspect.signature(training.train_source).parameters
    source = inspect.getsource(training.train_source)
    assert "test_indices" not in source
    norm_source = inspect.getsource(training.fit_source_preprocessing)
    assert 'eq("train")' in norm_source
    assert qualification.CONFIG["test_during_training"] is False
    assert qualification.CONFIG["8g_rows_used"] == 0


def test_full_domain_engineering_evidence():
    gate = json.loads((old.ROOT / "studies/predictor/final_v2_engineering/equivalence_audit.json").read_text())
    assert gate["status"] == "PASS" and gate["fixture_rows"] == 4163
    assert gate["max_abs_difference"] == 0
    assert gate["metric_max_abs_difference"] == 0

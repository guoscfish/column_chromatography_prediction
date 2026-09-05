from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch
from torch import nn

from scripts.studies import run_r2_pruned_requalification as study
from src.qgeognn_al.condition_complete_v2_pruned import (
    DEAD_MODULE_PREFIXES, PrunedConditionCompleteQGeoGNNV2,
    convert_r2_state_dict_to_pruned, is_dead_key,
)


@pytest.fixture(scope="module")
def fixture():
    return study.inputs()


@pytest.fixture(scope="module")
def models(fixture):
    *_, cn, norm, scales, batches = fixture
    original = study.r2.build_variant(study.REFERENCE, cn, norm, torch.device("cpu"))
    return original, PrunedConditionCompleteQGeoGNNV2(original)


def test_dead_inventory_and_all_parameter_reachability(fixture, models):
    original, pruned = models
    before = study.reachability(original, fixture[-1])
    after = study.reachability(pruned, fixture[-1])
    dead = {r["name"] for r in before["parameters"] if r["deletion_eligibility"]}
    assert dead == {n for n, _ in original.named_parameters() if is_dead_key(n)}
    assert before["nominal_parameters"] == 777808
    assert before["forward_unreachable_trainable_parameters"] == 318856
    assert after["nominal_parameters"] == after["requires_grad_parameters"] == after["gradient_bearing_parameters"] == 458952
    assert after["forward_unreachable_trainable_parameters"] == 0
    assert all(b["gradient_parameter_count"] == 458952 for b in after["batches"])
    assert not any(is_dead_key(n) for n, _ in pruned.named_parameters())


def test_conversion_complete_and_values_exact(models):
    original, pruned = models
    state, report = convert_r2_state_dict_to_pruned(original.state_dict(), original, pruned)
    assert set(state) == set(pruned.state_dict())
    assert report["conversion_status"] == "PASS"
    assert not report["missing_keys"] and not report["unexpected_keys"]
    assert report["retained_values_bitwise_equal"]
    assert all(torch.equal(v, original.state_dict()[k]) for k, v in state.items())
    assert set(report["removed_keys"]) == {k for k in original.state_dict() if is_dead_key(k)}
    pruned.load_state_dict(state, strict=True)


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype", "unknown_dead_prefix"])
def test_conversion_rejects_corrupt_state(models, corruption):
    original, pruned = models
    state = dict(original.state_dict())
    key = next(iter(pruned.state_dict()))
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["new.weight"] = torch.ones(1)
    elif corruption == "unknown_dead_prefix":
        state[DEAD_MODULE_PREFIXES[0] + ".unknown"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = torch.ones(1)
    else:
        state[key] = state[key].double()
    with pytest.raises(ValueError):
        convert_r2_state_dict_to_pruned(state, original, pruned)


@pytest.mark.parametrize("trained", [False, True])
def test_six_output_full_split_equivalence(fixture, trained):
    data, split, indices, atom, angle, cn, norm, scales, batches = fixture
    original = study.r2.build_variant(study.REFERENCE, cn, norm, torch.device("cpu"))
    pruned = PrunedConditionCompleteQGeoGNNV2(original)
    if trained:
        checkpoint = study.r2.RUNTIME / study.REFERENCE / "best.pt"
        if not checkpoint.exists():
            pytest.skip("Historical runtime checkpoint unavailable; committed full-domain audit tested separately")
        original.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
    state, _ = convert_r2_state_dict_to_pruned(original.state_dict(), original, pruned)
    pruned.load_state_dict(state)
    results = study.compare(original, pruned, data, atom, angle, indices, cn, scales)
    assert sum(row["rows"] for row in results.values()) == 4163
    for row in results.values():
        assert row["status"] == "PASS"
        assert len(row["six_output_max_abs_difference"]) == 6
        assert row["max_abs_difference"] <= 1e-6


def test_mapped_initialization_preserves_rng_order_and_values(fixture):
    *_, cn, norm, scales, batches = fixture
    original = study.r2.build_variant(study.REFERENCE, cn, norm, torch.device("cpu"))
    rng = torch.get_rng_state().clone()
    pruned = PrunedConditionCompleteQGeoGNNV2(original)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, original.state_dict()[k]) for k, v in pruned.state_dict().items())
    assert [n for n, _ in pruned.named_parameters()] == [n for n, _ in original.named_parameters() if not is_dead_key(n)]
    assert all(p.data_ptr() != dict(original.named_parameters())[n].data_ptr() for n, p in pruned.named_parameters())


def test_frozen_protocol_and_training_boundaries():
    protocol = json.loads((study.STUDY / "protocol.json").read_text())
    reference = study.r2.variant_config(study.REFERENCE)
    for key in ("seed", "split_path", "split_sha256", "source_data_sha256", "graph_cache_sha256",
                "optimizer", "learning_rate", "weight_decay", "batch_size", "maximum_epochs", "patience",
                "shuffle", "loss_weights", "checkpoint_selection", "normalization_fit_role"):
        assert protocol[key] == reference[key]
    assert study.r2.sha256_file(study.r2.FROZEN_SPLIT) == study.r2.EXPECTED_SPLIT_SHA
    for key in ("active_learning", "transfer", "uq", "test_during_training", "split_generation_called"):
        assert protocol[key] is False
    for key in ("8g_rows_used", "validation_rows_used_for_normalization", "test_rows_used_for_normalization"):
        assert protocol[key] == 0
    loop = inspect.getsource(study.r2.run_variant).split("for epoch in range", 1)[1].split("pd.DataFrame(history)", 1)[0]
    assert 'indices["test"]' not in loop
    assert 'indices["validation"]' in loop


def test_preserves_architecture_head_and_effective_geometry(models):
    original, pruned = models
    node = pruned.legacy_model.gnn_node
    assert len(node.convs) == 5
    assert len(node.convs_bond_angle) == 4
    assert len(node.convs_angle_float) == 4
    assert pruned.legacy_model.pool is original.legacy_model.pool
    assert isinstance(pruned.legacy_model.graph_pred_linear[1], nn.ReLU)
    assert pruned.legacy_model.graph_pred_linear[0].in_features == 128
    assert not any(isinstance(m, nn.LayerNorm) for m in pruned.modules())
    assert not any(isinstance(m, nn.Linear) and (m.in_features, m.out_features) == (128, 64) for m in pruned.modules())
    assert pruned.condition_branch.output.out_features == 128


def test_committed_gate_evidence():
    gate = json.loads((study.STUDY / "function_equivalence_audit.json").read_text())
    assert gate["status"] == "PASS"
    assert gate["fixture_rows"] == 4163
    for phase in ("P0_random_init", "P1_trained_checkpoint"):
        for role in ("train", "validation", "test"):
            assert gate[phase][role]["max_abs_difference"] <= 1e-6
    assert max(gate["two_adam_step_retained_state_max_differences"]) <= 1e-6


@pytest.mark.parametrize("attribute,value", [("drop_ratio", .1), ("JK", "sum"), ("residual", True)])
def test_rejects_unqualified_forward_configurations(models, attribute, value):
    original, _ = models
    node = original.legacy_model.gnn_node
    previous = getattr(node, attribute)
    try:
        setattr(node, attribute, value)
        with pytest.raises(ValueError, match="frozen R2"):
            PrunedConditionCompleteQGeoGNNV2(original)
    finally:
        setattr(node, attribute, previous)

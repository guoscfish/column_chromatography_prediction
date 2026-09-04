from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.condition_complete_v2 import (
    HIDDEN_DIM,
    INTENDED_CONDITION_FEATURES,
    MISSING_CONDITION_FEATURES,
    MODEL_VARIANT,
    ConditionCompleteQGeoGNNV2,
    assert_no_formal_run,
    build_condition_complete_v2,
    condition_complete_v2_schema,
    fit_condition_normalization,
    load_legacy_checkpoint,
    load_v2_checkpoint,
    v2_checkpoint_payload,
    v2_input_schema_hash,
)
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.model import build_model, configure_trainable
from src.qgeognn_al.resources import (
    SOURCE_CHECKPOINTS,
    SOURCE_DATA,
    SOURCE_GRAPH_CACHE,
    SOURCE_SCALER,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
PREFLIGHT = ROOT / "studies/track_b_transfer/predictor_v2_preflight"


def inputs(rows: int = 12):
    frame = pd.read_csv(SOURCE_DATA)
    chosen = frame.drop_duplicates("canonical_smiles").head(rows).reset_index(drop=True)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = json.loads(SOURCE_SCALER.read_text())
    atom, angle = build_model_data(chosen, cache, pd.DataFrame(), scaler)
    atom_batch = next(iter(DataLoader(atom, batch_size=len(atom), shuffle=False)))
    angle_batch = next(iter(DataLoader(angle, batch_size=len(angle), shuffle=False)))
    normalization = fit_condition_normalization(frame, pd.read_csv(SPLIT), SOURCE_SCALER)
    return frame, atom_batch, angle_batch, normalization


def test_legacy_builder_and_parameter_counts_are_unchanged() -> None:
    model = build_model(torch.device("cpu"))
    assert not isinstance(model, ConditionCompleteQGeoGNNV2)
    assert sum(p.numel() for p in model.parameters()) == 775476
    assert configure_trainable(model, "head_only") == (774, 775476)
    assert configure_trainable(model, "last1_head") == (93454, 775476)
    assert configure_trainable(model, "last2_head") == (186134, 775476)


def test_all_three_legacy_checkpoints_load_without_v2_metadata() -> None:
    for checkpoint in SOURCE_CHECKPOINTS.values():
        legacy = load_legacy_checkpoint(checkpoint, torch.device("cpu"))
        assert sum(p.numel() for p in legacy.parameters()) == 775476


def test_v2_variant_schema_and_hash_are_explicit_and_deterministic() -> None:
    schema = condition_complete_v2_schema()
    assert MODEL_VARIANT == "qgeognn_condition_complete_v2"
    assert schema["model_variant"] == MODEL_VARIANT
    assert schema["all_intended_condition_features"] == list(INTENDED_CONDITION_FEATURES)
    assert len(schema["all_intended_condition_features"]) == 9
    assert v2_input_schema_hash() == v2_input_schema_hash(schema)


def test_v2_uses_typed_solvent_and_separate_loading_normalization() -> None:
    schema = condition_complete_v2_schema()
    assert schema["loading_solvent"]["type"] == "categorical_embedding"
    assert schema["continuous_completion_features"]["dtype"] == "float32"
    assert schema["continuous_completion_features"]["loading_normalization"] == "source_train_only_minmax"
    assert "loading_amount_density_x_volume" in schema["completion_condition_features"]
    assert schema["condition_branch"]["hidden_dim"] == HIDDEN_DIM == 16


def test_normalization_is_fit_only_on_frozen_source_train_ids() -> None:
    frame = pd.read_csv(SOURCE_DATA)
    split = pd.read_csv(SPLIT)
    normalization = fit_condition_normalization(frame, split, SOURCE_SCALER)
    assert normalization.fit_dataset == "4g"
    assert normalization.fit_role == "source_train"
    assert normalization.fit_row_count == int(split["split"].eq("train").sum())
    assert len(normalization.fit_ids_hash) == 64
    assert normalization.eluent_scaler_sha256 == sha256_file(SOURCE_SCALER)


def test_zero_initialization_preserves_source_predictions_exactly() -> None:
    _, atom, angle, normalization = inputs(10)
    for checkpoint in SOURCE_CHECKPOINTS.values():
        legacy = load_legacy_checkpoint(checkpoint, torch.device("cpu")); legacy.eval()
        v2 = build_condition_complete_v2(checkpoint, normalization, torch.device("cpu")); v2.eval()
        with torch.no_grad():
            legacy_prediction = legacy(atom, angle)[0]
            v2_prediction = v2(atom, angle)[0]
        torch.testing.assert_close(v2_prediction, legacy_prediction, rtol=0, atol=0)


def test_missing_feature_perturbations_reach_branch_and_diagnostic_output() -> None:
    _, atom, angle, normalization = inputs(3)
    model = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        model.condition_branch.output.weight.fill_(0.01)
        base_prediction = model(atom, angle)[0]
        base_typed = model.condition_branch.typed_inputs(atom)
    positions = {
        "eluent_h_acceptors": 3 + 5,
        "eluent_logp": 3 + 6,
        "loading_solvent": 3 + 7,
        "loading_amount_density_x_volume": 3 + 8,
        "loading_solvent_volume_ul": 3 + 9,
    }
    for feature in MISSING_CONDITION_FEATURES:
        changed = atom.clone()
        if feature == "loading_solvent":
            changed.edge_attr[:, positions[feature]] = (changed.edge_attr[:, positions[feature]] + 1) % 3
        else:
            changed.edge_attr[:, positions[feature]] += 0.25
        with torch.no_grad():
            typed = model.condition_branch.typed_inputs(changed)
            prediction = model(changed, angle)[0]
        assert not torch.equal(typed, base_typed), feature
        assert not torch.equal(prediction, base_prediction), feature


def test_branch_parameters_receive_gradients_after_two_engineering_steps() -> None:
    frame = pd.read_csv(SOURCE_DATA)
    chosen = pd.concat([
        frame.loc[frame["loading solvent"].eq(value)].drop_duplicates("canonical_smiles").head(2)
        for value in ("PE", "EA", "DCM")
    ]).reset_index(drop=True)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    atom_data, angle_data = build_model_data(chosen, cache, pd.DataFrame(), json.loads(SOURCE_SCALER.read_text()))
    atom = next(iter(DataLoader(atom_data, batch_size=len(atom_data), shuffle=False)))
    angle = next(iter(DataLoader(angle_data, batch_size=len(angle_data), shuffle=False)))
    normalization = fit_condition_normalization(frame, pd.read_csv(SPLIT), SOURCE_SCALER)
    model = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.condition_branch.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.SGD(model.condition_branch.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(atom, angle)[0]
        prediction.sum().backward()
        optimizer.step()
    for name, parameter in model.condition_branch.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_v2_checkpoint_requires_schema_hash_while_legacy_does_not() -> None:
    _, _, _, normalization = inputs(2)
    model = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    payload = v2_checkpoint_payload(model, SOURCE_CHECKPOINTS[42], 0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v2.pt"
        torch.save(payload, path)
        loaded = load_v2_checkpoint(path, SOURCE_CHECKPOINTS[42], torch.device("cpu"))
        assert loaded.model_variant == MODEL_VARIANT
        payload.pop("input_schema_hash")
        torch.save(payload, path)
        with pytest.raises(ValueError, match="input_schema_hash"):
            load_v2_checkpoint(path, SOURCE_CHECKPOINTS[42], torch.device("cpu"))
    assert "input_schema_hash" not in torch.load(SOURCE_CHECKPOINTS[42], weights_only=False)


def test_v2_checkpoint_rejects_the_wrong_legacy_anchor() -> None:
    _, _, _, normalization = inputs(2)
    model = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    payload = v2_checkpoint_payload(model, SOURCE_CHECKPOINTS[42], 0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "v2.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="legacy_anchor_sha256"):
            load_v2_checkpoint(path, SOURCE_CHECKPOINTS[525], torch.device("cpu"))


def test_formal_entry_is_refused_without_authorization() -> None:
    with pytest.raises(RuntimeError, match="formal_authorized=false"):
        assert_no_formal_run(False)


def test_historical_formal_results_are_unchanged() -> None:
    expected = {
        "studies/track_b_transfer/t1_low_label_adaptation/FORMAL_RESULTS.md": "76e0255b07c690b12fd8317923ed1e2d0ccdb4513977c7d326959f0eedf63833",
        "studies/track_b_transfer/t1b1_adapter_capacity/FORMAL_RESULTS.md": "dfe57bb95158c2006443ae98f8b1e1449349129a3dd73f0866c92633ff384934",
    }
    for relative, digest in expected.items():
        assert sha256_file(ROOT / relative) == digest


def test_preflight_artifacts_record_a_pass_without_performance_training() -> None:
    smoke = json.loads((PREFLIGHT / "engineering_smoke_audit.json").read_text())
    decision = json.loads((PREFLIGHT / "decision.json").read_text())
    assert smoke["all_checks_pass"] is True
    assert smoke["source_fixture_count"] == 10
    assert smoke["source_identity_max_abs_by_member"] == {"42": 0.0, "525": 0.0, "1101": 0.0}
    assert decision["implementation_preflight_pass"] is True
    assert decision["new_predictor_performance_result"] is False
    assert decision["formal_source_qualification_authorized"] is False
    assert decision["8g_transfer_started"] is False


def test_preflight_artifacts_cover_reachability_parameters_and_no_leakage() -> None:
    reachability = pd.read_csv(PREFLIGHT / "v2_feature_reachability_audit.csv")
    parameters = json.loads((PREFLIGHT / "condition_branch_parameter_audit.json").read_text())
    normalization = json.loads((PREFLIGHT / "normalization_audit.json").read_text())
    assert set(reachability.feature) == set(INTENDED_CONDITION_FEATURES)
    assert reachability.v2_reachable.all()
    assert parameters["v2_added_parameters"] == 2332
    assert parameters["v2_total_nominal_parameters"] == 777808
    assert parameters["v2_diagnostic_activation_classification"]["gradient_bearing"] == 458952
    assert normalization["fit_row_count"] == 3330
    assert normalization["fit_role"] == "source_train"
    assert normalization["target_rows_used"] == normalization["validation_rows_used"] == normalization["test_rows_used"] == 0
    assert normalization["leakage_detected"] is False

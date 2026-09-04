from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models.clean_fusion import (
    REQUIRED_CLEAN_CHECKPOINT_FIELDS,
    build_clean_model,
    clean_checkpoint_payload,
    latent_l2_norms,
    parameter_reachability,
    per_target_gradient_contribution,
    permute_conditions,
    validate_clean_checkpoint,
)
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.schemas.clean import (
    CLEAN_MODEL_VARIANT,
    CONTINUOUS_CONDITION_NAMES,
    SOLVENT_VOCABULARY,
    CleanConditionBatch,
    clean_condition_schema_hash,
    clean_input_schema,
    clean_input_schema_hash,
    fit_clean_condition_normalization,
    loading_mass_mg,
    parse_clean_conditions,
    parse_ea_fraction,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
PREFLIGHT = ROOT / "studies/predictor/clean_qgeognn/preflight"


def clean_inputs(rows_per_solvent: int = 1):
    frame = pd.read_csv(SOURCE_DATA)
    chosen = pd.concat([
        frame.loc[frame["loading solvent"].eq(solvent)].drop_duplicates("canonical_smiles").head(rows_per_solvent)
        for solvent in SOLVENT_VOCABULARY
    ]).reset_index(drop=True)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = json.loads((ROOT / "experiments/e0_4g_baseline/scaler.json").read_text())
    atoms, angles = build_model_data(chosen, cache, pd.DataFrame(), scaler)
    atom_batch = next(iter(DataLoader(atoms, batch_size=len(atoms), shuffle=False)))
    angle_batch = next(iter(DataLoader(angles, batch_size=len(angles), shuffle=False)))
    normalization = fit_clean_condition_normalization(frame, pd.read_csv(SPLIT))
    conditions = parse_clean_conditions(chosen, normalization)
    return frame, chosen, atom_batch, angle_batch, conditions, normalization


def test_clean_schema_hashes_are_deterministic_and_typed() -> None:
    schema = clean_input_schema()
    assert schema["model_variant"] == CLEAN_MODEL_VARIANT == "qgeognn_clean_fusion_v1"
    assert clean_input_schema_hash() == clean_input_schema_hash(schema)
    assert clean_condition_schema_hash() == clean_condition_schema_hash()
    assert len(clean_input_schema_hash()) == len(clean_condition_schema_hash()) == 64
    assert schema["experimental_conditions"]["scope"] == "sample_or_graph_level"
    assert schema["column_context"]["included_in_clean_4g"] is False
    assert schema["geometry_contract"] == "official_code_first_embedded_for_controlled_comparison"
    assert schema["molecular_descriptor_contract"] == "official_code_molecular_descriptor_16"
    assert schema["condition_contract"] == "clean_typed_sample_level_v1"
    assert schema["paper_method_equivalent"] is False


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [("40/1", 1 / 41), ("1/1", 0.5), ("0/1", 1.0), ("2.5/7.5", 0.75)],
)
def test_pe_ea_parsing_uses_one_fraction(ratio: str, expected: float) -> None:
    assert parse_ea_fraction(ratio) == pytest.approx(expected)


@pytest.mark.parametrize("invalid", ["1", "1/2/3", "0/0", "-1/2", "x/2"])
def test_pe_ea_parsing_rejects_invalid_ratios(invalid: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_ea_fraction(invalid)


def test_loading_mass_has_explicit_mg_unit() -> None:
    assert loading_mass_mg(1.0, 1.0) == pytest.approx(1.0)
    assert loading_mass_mg(0.99, 100.0) == pytest.approx(99.0)


def test_normalization_uses_only_source_train_rows() -> None:
    frame = pd.read_csv(SOURCE_DATA)
    split = pd.read_csv(SPLIT)
    normalization = fit_clean_condition_normalization(frame, split)
    assert normalization.continuous_names == CONTINUOUS_CONDITION_NAMES
    assert normalization.fit_dataset == "4g"
    assert normalization.fit_role == "source_train"
    assert normalization.fit_row_count == int(split["split"].eq("train").sum()) == 3330
    assert normalization.validation_rows_used == 0
    assert normalization.test_rows_used == 0
    assert normalization.target_8g_rows_used == 0
    assert len(normalization.fit_ids_hash) == 64


def test_normalization_rejects_target_domain_and_uncovered_rows() -> None:
    frame = pd.read_csv(SOURCE_DATA).head(3)
    split = pd.DataFrame({"sample_id": frame.sample_id, "split": "train"})
    with pytest.raises(ValueError, match="only the 4g source"):
        fit_clean_condition_normalization(frame, split, dataset="8g")
    with pytest.raises(ValueError, match="does not cover"):
        fit_clean_condition_normalization(frame, split.head(2))


def test_condition_categories_are_int64_and_not_ordinal_continuous() -> None:
    _, _, _, _, conditions, _ = clean_inputs()
    assert conditions.loading_solvent.dtype == torch.int64
    assert conditions.continuous.dtype == torch.float32
    assert set(conditions.loading_solvent.tolist()) == {0, 1, 2}
    assert conditions.continuous.shape[1] == 3


def test_clean_forward_is_finite_ordered_and_deterministic() -> None:
    _, _, atom, angle, conditions, normalization = clean_inputs()
    torch.manual_seed(20260904)
    model = build_clean_model(normalization, torch.device("cpu")).eval()
    with torch.no_grad():
        first = model(atom, angle, conditions)
        second = model(atom, angle, conditions)
    assert first.shape == (3, 6)
    assert torch.isfinite(first).all()
    assert torch.all(first >= 0)
    assert torch.equal(first, second)
    assert torch.all(first[:, 0] <= first[:, 1])
    assert torch.all(first[:, 1] <= first[:, 2])
    assert torch.all(first[:, 3] <= first[:, 4])
    assert torch.all(first[:, 4] <= first[:, 5])


def test_clean_quantile_train_and_eval_output_contract() -> None:
    _, _, atom, angle, conditions, normalization = clean_inputs()
    torch.manual_seed(101)
    model = build_clean_model(normalization, torch.device("cpu"))
    with torch.no_grad():
        model.quantile_head.linear.weight.zero_()
        model.quantile_head.linear.bias.copy_(torch.tensor([-20.0, 5.0, 1.0, -20.0, 5.0, 1.0]))
        train_output = model.train()(atom, angle, conditions)
        eval_output = model.eval()(atom, angle, conditions)
    assert torch.isfinite(train_output).all() and torch.isfinite(eval_output).all()
    assert torch.all(train_output[:, 0] <= train_output[:, 1])
    assert torch.all(train_output[:, 1] <= train_output[:, 2])
    assert torch.all(train_output[:, 3] <= train_output[:, 4])
    assert torch.all(train_output[:, 4] <= train_output[:, 5])
    assert torch.any(train_output < 0), "training keeps the frozen offset parameterization"
    assert torch.all(eval_output >= 0)
    assert torch.all(eval_output[:, 0] <= eval_output[:, 1])
    assert torch.all(eval_output[:, 1] <= eval_output[:, 2])
    assert torch.all(eval_output[:, 3] <= eval_output[:, 4])
    assert torch.all(eval_output[:, 4] <= eval_output[:, 5])


def test_every_intended_feature_reaches_its_internal_representation() -> None:
    _, _, atom, angle, conditions, normalization = clean_inputs()
    torch.manual_seed(13)
    model = build_clean_model(normalization, torch.device("cpu")).eval()
    with torch.no_grad():
        baseline = model.representations(atom, angle, conditions)

    changed_atom = atom.clone()
    changed_atom.edge_attr[:, 3] += 0.2
    with torch.no_grad():
        assert not torch.equal(model.representations(changed_atom, angle, conditions)["molecular"], baseline["molecular"])

    changed_angle = angle.clone()
    changed_angle.edge_attr[:, 0] += 0.1
    changed_angle.edge_attr[:, 1:] += 0.05
    with torch.no_grad():
        assert not torch.equal(model.representations(atom, changed_angle, conditions)["molecular"], baseline["molecular"])

    for position, name in enumerate(CONTINUOUS_CONDITION_NAMES):
        values = conditions.continuous.clone()
        values[:, position] += 0.25
        changed = replace(conditions, continuous=values)
        with torch.no_grad():
            encoded = model.representations(atom, angle, changed)["condition"]
        assert not torch.equal(encoded, baseline["condition"]), name

    solvent = (conditions.loading_solvent + 1) % len(SOLVENT_VOCABULARY)
    changed = replace(conditions, loading_solvent=solvent)
    with torch.no_grad():
        encoded = model.representations(atom, angle, changed)["condition"]
    assert not torch.equal(encoded, baseline["condition"])


def test_real_backward_has_zero_forward_unreachable_trainable_parameters() -> None:
    _, _, atom, angle, conditions, normalization = clean_inputs(rows_per_solvent=2)
    torch.manual_seed(71)
    model = build_clean_model(normalization, torch.device("cpu")).train()
    prediction = model(atom, angle, conditions)
    coefficients = torch.linspace(0.5, 1.5, prediction.numel()).reshape_as(prediction)
    (prediction.square().mean() + (prediction * coefficients).mean()).backward()
    audit = parameter_reachability(model)
    assert audit["nominal_parameters"] == audit["requires_grad_parameters"]
    assert audit["gradient_bearing_parameters"] == audit["requires_grad_parameters"]
    assert audit["forward_unreachable_trainable_parameters"] == 0
    assert all(row["gradient_reachable"] for row in audit["parameters"] if row["requires_grad"])


def test_collision_case_is_separable_in_condition_latent() -> None:
    frame, _, atom, angle, _, normalization = clean_inputs()
    base = frame.head(1).copy()
    collision = pd.concat([base, base], ignore_index=True)
    collision.loc[1, "loading solvent"] = "EA" if base.iloc[0]["loading solvent"] != "EA" else "DCM"
    collision.loc[1, "V/ul"] = float(base.iloc[0]["V/ul"]) + 10.0
    collision.loc[1, "Volume of loading solvent/ul"] = float(base.iloc[0]["Volume of loading solvent/ul"]) + 25.0
    conditions = parse_clean_conditions(collision, normalization, sample_ids=("a", "b"))
    torch.manual_seed(5)
    model = build_clean_model(normalization, torch.device("cpu")).eval()
    with torch.no_grad():
        latent = model.condition_encoder(conditions)
    assert collision.loc[0, "canonical_smiles"] == collision.loc[1, "canonical_smiles"]
    assert collision.loc[0, "PE/EA"] == collision.loc[1, "PE/EA"]
    assert not torch.equal(latent[0], latent[1])
    assert atom.num_graphs == angle.num_graphs == 3


def test_modality_and_target_gradient_diagnostics_are_available() -> None:
    _, chosen, atom, angle, conditions, normalization = clean_inputs()
    torch.manual_seed(19)
    model = build_clean_model(normalization, torch.device("cpu")).eval()
    with torch.no_grad():
        full = model.representations(atom, angle, conditions)
        norms = latent_l2_norms(full)
        molecule_only = model(atom, angle, conditions, mode="molecule_only")
        condition_only = model(atom, angle, conditions, mode="condition_only")
        permuted = model(atom, angle, permute_conditions(conditions, [1, 2, 0]))
    assert set(norms) == {"molecular_latent_l2_mean", "condition_latent_l2_mean"}
    assert not torch.equal(molecule_only, condition_only)
    assert not torch.equal(permuted, model(atom, angle, conditions))
    targets = torch.tensor(chosen[["V1_ml", "V2_ml"]].to_numpy(), dtype=torch.float32)
    contributions = per_target_gradient_contribution(model, atom, angle, conditions, targets)
    assert set(contributions) == {"V1", "V2"}
    for row in contributions.values():
        assert row["molecular_projection_gradient_l2"] > 0
        assert row["condition_encoder_gradient_l2"] > 0


def test_clean_checkpoint_contract_is_complete_and_validated(tmp_path: Path) -> None:
    _, _, atom, angle, conditions, normalization = clean_inputs()
    model = build_clean_model(normalization, torch.device("cpu"))
    model(atom, angle, conditions).sum().backward()
    reachable = parameter_reachability(model)["gradient_bearing_parameters"]
    payload = clean_checkpoint_payload(
        model,
        gradient_bearing_parameter_count=reachable,
        git_commit_sha="a" * 40,
        source_split_hash=sha256_file(SPLIT),
        training_config={"formal_training": False, "purpose": "engineering_preflight"},
    )
    assert REQUIRED_CLEAN_CHECKPOINT_FIELDS.issubset(payload)
    validate_clean_checkpoint(payload)
    payload["input_schema"]["schema_version"] = 99
    with pytest.raises(ValueError, match="input_schema_hash"):
        validate_clean_checkpoint(payload)


def test_navigation_links_resolve_and_benchmark_remains_preregistration_only() -> None:
    markdown_files = [ROOT / "README.md", ROOT / "docs/README.md", ROOT / "studies/README.md"]
    markdown_files.extend((ROOT / "docs/model").glob("*.md"))
    markdown_files.extend((ROOT / "docs/protocols").glob("*.md"))
    markdown_files.extend((ROOT / "docs/roadmap").glob("*.md"))
    markdown_files.extend((ROOT / "docs/repository").glob("*.md"))
    markdown_files.extend((ROOT / "studies/predictor").rglob("*.md"))
    markdown_files.extend((ROOT / "studies/transfer").rglob("*.md"))
    markdown_files.extend((ROOT / "studies/active_learning").rglob("*.md"))
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    missing = []
    for markdown in markdown_files:
        for target in pattern.findall(markdown.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = (markdown.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                missing.append(f"{markdown.relative_to(ROOT)} -> {target}")
    assert not missing
    benchmark = ROOT / "studies/predictor/4g_source_benchmark"
    assert {path.name for path in benchmark.iterdir()} == {
        "README.md", "PREREGISTRATION.md", "protocol.json", "data_usage.json",
        "decision.json", "data_audit",
    }
    forbidden = {"metrics.json", "predictions.csv", "checkpoints", "training_history.csv"}
    assert not forbidden.intersection(path.name for path in benchmark.rglob("*"))
    protocol = json.loads((benchmark / "protocol.json").read_text())
    assert protocol["formal_authorized"] is False
    assert protocol["models_trained_this_round"] == 0


def test_legacy_history_was_not_overwritten() -> None:
    expected = {
        "studies/track_b_transfer/t1_low_label_adaptation/FORMAL_RESULTS.md": "76e0255b07c690b12fd8317923ed1e2d0ccdb4513977c7d326959f0eedf63833",
        "studies/track_b_transfer/t1b1_adapter_capacity/FORMAL_RESULTS.md": "dfe57bb95158c2006443ae98f8b1e1449349129a3dd73f0866c92633ff384934",
    }
    for relative, digest in expected.items():
        assert sha256_file(ROOT / relative) == digest


def test_preflight_record_matches_implementation_and_scientific_boundary() -> None:
    parameters = json.loads((PREFLIGHT / "parameter_reachability_audit.json").read_text())
    normalization = json.loads((PREFLIGHT / "normalization_audit.json").read_text())
    checkpoint = json.loads((PREFLIGHT / "checkpoint_contract.json").read_text())
    decision = json.loads((PREFLIGHT / "decision.json").read_text())
    features = pd.read_csv(PREFLIGHT / "feature_reachability_audit.csv")
    assert parameters == {
        "audit_mode": "real forward/backward on multi-molecule fixtures spanning PE, EA, and DCM loading solvents; structural objective; no performance evaluation",
        "preflight_revision": 2,
        "nominal_parameters": 413732,
        "requires_grad_parameters": 413732,
        "gradient_bearing_parameters": 413732,
        "forward_unreachable_trainable_parameters": 0,
        "all_intended_trainable_modules_gradient_reachable": True,
        "known_legacy_dead_modules_registered": False,
    }
    assert normalization["fit_row_count"] == 3330
    assert normalization["validation_rows_used"] == 0
    assert normalization["test_rows_used"] == 0
    assert normalization["target_8g_rows_used"] == 0
    assert normalization["leakage_detected"] is False
    assert checkpoint["input_schema_hash"] == clean_input_schema_hash()
    assert checkpoint["condition_schema_hash"] == clean_condition_schema_hash()
    assert set(checkpoint["required_fields"]) == REQUIRED_CLEAN_CHECKPOINT_FIELDS
    assert checkpoint["preflight_revision"] == 2
    assert decision["preflight_revision"] == 2
    assert features["preflight_revision"].eq(2).all()
    assert features["perturbation_reaches_internal_representation"].all()
    assert decision["implementation_preflight_pass"] is True
    assert decision["formal_performance_qualified"] is False
    assert decision["formal_4g_benchmark_run"] is False
    assert decision["8g_transfer_started"] is False
    assert decision["active_learning_started"] is False

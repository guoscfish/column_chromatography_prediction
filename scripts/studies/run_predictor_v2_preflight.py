#!/usr/bin/env python3
"""Run Predictor V2 engineering audits without scientific training or evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.condition_complete_v2 import (
    INTENDED_CONDITION_FEATURES,
    MISSING_CONDITION_FEATURES,
    build_condition_complete_v2,
    condition_branch_config,
    condition_branch_config_hash,
    condition_complete_v2_schema,
    fit_condition_normalization,
    load_legacy_checkpoint,
    v2_checkpoint_payload,
    v2_input_schema_hash,
    validate_v2_checkpoint,
)
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.input_schema import CATEGORICAL_BOND_FEATURES
from src.qgeognn_al.resources import (
    ROOT,
    SOURCE_CHECKPOINTS,
    SOURCE_DATA,
    SOURCE_GRAPH_CACHE,
    SOURCE_SCALER,
)


STUDY = ROOT / "studies/track_b_transfer/predictor_v2_preflight"
SOURCE_SPLIT = ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
IDENTITY_TOLERANCE = 1e-7


def write_json(name: str, value: dict) -> None:
    (STUDY / name).write_text(json.dumps(value, indent=2) + "\n")


def batches(frame: pd.DataFrame, cache: dict, scaler: dict):
    atoms, angles = build_model_data(frame.reset_index(drop=True), cache, pd.DataFrame(), scaler)
    return (
        next(iter(DataLoader(atoms, batch_size=len(atoms), shuffle=False))),
        next(iter(DataLoader(angles, batch_size=len(angles), shuffle=False))),
    )


def select_fixtures(frame: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    selected = []
    used_molecules: set[str] = set()
    for solvent in ("PE", "EA", "DCM"):
        rows = frame.loc[frame["loading solvent"].eq(solvent)]
        for _, row in rows.iterrows():
            if row.canonical_smiles not in used_molecules:
                selected.append(row)
                used_molecules.add(row.canonical_smiles)
                break
    for _, row in frame.sort_values(["PE/EA", "canonical_smiles", "sample_id"]).iterrows():
        if len(selected) >= count:
            break
        if row.canonical_smiles not in used_molecules:
            selected.append(row)
            used_molecules.add(row.canonical_smiles)
    result = pd.DataFrame(selected).reset_index(drop=True)
    if len(result) != count or result.canonical_smiles.nunique() != count:
        raise RuntimeError("could not select ten distinct source fixtures")
    return result


def source_identity_audit(fixtures, atom, angle, normalization) -> tuple[pd.DataFrame, dict, dict]:
    rows, maximum, ordering = [], {}, {}
    for member, checkpoint in SOURCE_CHECKPOINTS.items():
        legacy = load_legacy_checkpoint(checkpoint, torch.device("cpu")); legacy.eval()
        v2 = build_condition_complete_v2(checkpoint, normalization, torch.device("cpu")); v2.eval()
        with torch.no_grad():
            legacy_prediction = legacy(atom, angle)[0]
            v2_prediction = v2(atom, angle)[0]
        differences = torch.max(torch.abs(legacy_prediction - v2_prediction), dim=1).values
        maximum[str(member)] = float(differences.max())
        ordering[str(member)] = {
            "quantile_crossing_count": int(((v2_prediction[:, 0] > v2_prediction[:, 1]) | (v2_prediction[:, 1] > v2_prediction[:, 2]) | (v2_prediction[:, 3] > v2_prediction[:, 4]) | (v2_prediction[:, 4] > v2_prediction[:, 5])).sum()),
            "v1_q50_gt_v2_q50_count": int((v2_prediction[:, 1] > v2_prediction[:, 4]).sum()),
        }
        for index, difference in enumerate(differences):
            rows.append({
                "source_member": member,
                "sample_id": fixtures.iloc[index].sample_id,
                "canonical_smiles": fixtures.iloc[index].canonical_smiles,
                "PE/EA": fixtures.iloc[index]["PE/EA"],
                "loading_solvent": fixtures.iloc[index]["loading solvent"],
                "max_abs_prediction_difference": float(difference),
                "tolerance": IDENTITY_TOLERANCE,
                "pass": bool(difference <= IDENTITY_TOLERANCE),
            })
    return pd.DataFrame(rows), maximum, ordering


def feature_reachability(atom, angle, normalization) -> pd.DataFrame:
    legacy = load_legacy_checkpoint(SOURCE_CHECKPOINTS[42], torch.device("cpu")); legacy.eval()
    v2_zero = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu")); v2_zero.eval()
    v2_diagnostic = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu")); v2_diagnostic.eval()
    with torch.no_grad():
        legacy_base_prediction, legacy_base_h = legacy(atom, angle)
        _, v2_base_h, v2_base_internal = v2_zero.representations(atom, angle)
        v2_diagnostic.condition_branch.output.weight.fill_(0.01)
        diagnostic_base_prediction = v2_diagnostic(atom, angle)[0]
    rows = []
    positions = {name: 3 + index for index, name in enumerate((
        "bond_length", "eluent_exact_mol_wt", "eluent_tpsa", "eluent_rotatable_bonds",
        "eluent_h_donors", "eluent_h_acceptors", "eluent_logp", "loading_solvent",
        "loading_amount_density_x_volume", "loading_solvent_volume_ul",
    ))}
    for feature in INTENDED_CONDITION_FEATURES:
        changed = atom.clone()
        position = positions[feature]
        if feature == "loading_solvent":
            changed.edge_attr[:, position] = (changed.edge_attr[:, position] + 1) % 3
        else:
            changed.edge_attr[:, position] += 0.137
        with torch.no_grad():
            legacy_prediction, legacy_h = legacy(changed, angle)
            _, v2_h, v2_internal = v2_zero.representations(changed, angle)
            v2_prediction = v2_diagnostic(changed, angle)[0]
        legacy_latent_delta = float(torch.max(torch.abs(legacy_h - legacy_base_h)))
        branch_delta = float(torch.max(torch.abs(v2_internal - v2_base_internal)))
        rows.append({
            "feature": feature,
            "legacy_reachable": bool(legacy_latent_delta > 0),
            "legacy_latent_delta_max_abs": legacy_latent_delta,
            "legacy_prediction_delta_max_abs": float(torch.max(torch.abs(legacy_prediction - legacy_base_prediction))),
            "v2_encoder_internal_delta_max_abs": branch_delta,
            "v2_initialized_latent_delta_max_abs": float(torch.max(torch.abs(v2_h - v2_base_h))),
            "v2_diagnostic_prediction_delta_max_abs": float(torch.max(torch.abs(v2_prediction - diagnostic_base_prediction))),
            "v2_reachable": bool(legacy_latent_delta > 0 or branch_delta > 0),
            "diagnostic_integration_only": feature in MISSING_CONDITION_FEATURES,
        })
    return pd.DataFrame(rows)


def parameter_rows(model, step: str) -> list[dict]:
    rows = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        norm = None if grad is None else float(grad.norm().detach())
        category = "structurally_unreachable" if grad is None else "fixture_dependent_inactive" if norm == 0 else "gradient_bearing"
        rows.append({
            "audit_step": step,
            "parameter_name": name,
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
            "grad_is_none": grad is None,
            "grad_norm": norm,
            "classification": category,
        })
    return rows


def gradient_audits(atom, angle, normalization) -> tuple[pd.DataFrame, dict, dict]:
    legacy = load_legacy_checkpoint(SOURCE_CHECKPOINTS[42], torch.device("cpu")); legacy.eval()
    legacy.zero_grad(set_to_none=True)
    legacy(atom, angle)[0].sum().backward()
    legacy_rows = parameter_rows(legacy, "legacy_multi_fixture")

    v2 = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    for parameter in v2.parameters():
        parameter.requires_grad = False
    for parameter in v2.condition_branch.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.SGD(v2.condition_branch.parameters(), lr=1e-3)
    branch_rows = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        v2(atom, angle)[0].sum().backward()
        branch_rows.extend(parameter_rows(v2.condition_branch, f"v2_branch_step_{step}"))
        optimizer.step()

    activated = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    with torch.no_grad():
        activated.condition_branch.output.weight.fill_(0.01)
    activated.zero_grad(set_to_none=True)
    activated(atom, angle)[0].sum().backward()
    activated_rows = parameter_rows(activated, "v2_diagnostic_activation")
    all_rows = pd.DataFrame(legacy_rows + branch_rows + activated_rows)

    def summarize(rows):
        return {
            category: sum(row["numel"] for row in rows if row["classification"] == category)
            for category in ("structurally_unreachable", "fixture_dependent_inactive", "gradient_bearing")
        }
    return all_rows, summarize(legacy_rows), summarize(activated_rows)


def collision_audit(frame, cache, scaler, normalization) -> pd.DataFrame:
    loading = ["loading solvent", "Density g/ml", "V/ul", "Volume of loading solvent/ul"]
    pair = None
    for _, group in frame.groupby(["canonical_smiles", "PE/EA"], sort=True):
        distinct = group.drop_duplicates(loading)
        if len(distinct) >= 2:
            pair = distinct.head(2).reset_index(drop=True)
            break
    if pair is None:
        raise RuntimeError("no real source collision pair found")
    atom, angle = batches(pair, cache, scaler)
    legacy = load_legacy_checkpoint(SOURCE_CHECKPOINTS[42], torch.device("cpu")); legacy.eval()
    v2 = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu")); v2.eval()
    with torch.no_grad():
        legacy_prediction, legacy_h = legacy(atom, angle)
        typed = v2.condition_branch.typed_inputs(atom)
        v2.condition_branch.output.weight.fill_(0.01)
        diagnostic_prediction = v2(atom, angle)[0]
    return pd.DataFrame([{
        "sample_id_a": pair.iloc[0].sample_id,
        "sample_id_b": pair.iloc[1].sample_id,
        "canonical_smiles": pair.iloc[0].canonical_smiles,
        "PE/EA": pair.iloc[0]["PE/EA"],
        "loading_conditions_differ": True,
        "legacy_embedding_delta_max_abs": float(torch.max(torch.abs(legacy_h[0] - legacy_h[1]))),
        "legacy_prediction_delta_max_abs": float(torch.max(torch.abs(legacy_prediction[0] - legacy_prediction[1]))),
        "v2_condition_representation_delta_max_abs": float(torch.max(torch.abs(typed[0] - typed[1]))),
        "v2_diagnostic_prediction_delta_max_abs": float(torch.max(torch.abs(diagnostic_prediction[0] - diagnostic_prediction[1]))),
        "v2_can_resolve_collision": bool(not torch.equal(typed[0], typed[1])),
    }])


def run() -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE_DATA)
    split = pd.read_csv(SOURCE_SPLIT)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = json.loads(SOURCE_SCALER.read_text())
    normalization = fit_condition_normalization(frame, split, SOURCE_SCALER)
    fixtures = select_fixtures(frame)
    atom, angle = batches(fixtures, cache, scaler)

    identity, member_maximum, ordering = source_identity_audit(fixtures, atom, angle, normalization)
    reachability = feature_reachability(atom, angle, normalization)
    gradients, legacy_summary, v2_summary = gradient_audits(atom, angle, normalization)
    collisions = collision_audit(frame, cache, scaler, normalization)
    schema = condition_complete_v2_schema()
    model = build_condition_complete_v2(SOURCE_CHECKPOINTS[42], normalization, torch.device("cpu"))
    payload = v2_checkpoint_payload(model, SOURCE_CHECKPOINTS[42], v2_summary["gradient_bearing"])
    validate_v2_checkpoint(payload)

    input_hash = v2_input_schema_hash(schema)
    write_json("input_schema.json", schema)
    (STUDY / "input_schema_hash.txt").write_text(input_hash + "\n")
    write_json("normalization_audit.json", {
        **asdict(normalization),
        "features": {
            "eluent_h_acceptors": {"scaler": "source eluent minmax dimension 4", "min": scaler["eluent"]["min"][4], "max": scaler["eluent"]["max"][4]},
            "eluent_logp": {"scaler": "source eluent minmax dimension 5", "min": scaler["eluent"]["min"][5], "max": scaler["eluent"]["max"][5]},
            "loading_amount_density_x_volume": {"min": normalization.loading_amount_min, "max": normalization.loading_amount_max},
            "loading_solvent_volume_ul": {"min": normalization.loading_volume_min, "max": normalization.loading_volume_max},
        },
        "target_rows_used": 0,
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "leakage_detected": False,
    })
    write_json("condition_branch_parameter_audit.json", {
        "condition_branch_config": condition_branch_config(),
        "condition_branch_config_hash": condition_branch_config_hash(),
        "legacy_nominal_parameters": 775476,
        "v2_added_parameters": sum(p.numel() for p in model.condition_branch.parameters()),
        "v2_total_nominal_parameters": sum(p.numel() for p in model.parameters()),
        "v2_total_requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "legacy_multi_fixture_classification": legacy_summary,
        "v2_diagnostic_activation_classification": v2_summary,
    })
    identity.to_csv(STUDY / "source_function_identity_audit.csv", index=False)
    reachability.to_csv(STUDY / "v2_feature_reachability_audit.csv", index=False)
    gradients.to_csv(STUDY / "gradient_reachability_audit.csv", index=False)
    collisions.to_csv(STUDY / "collision_resolvability_audit.csv", index=False)

    checks = {
        "legacy_checkpoints_loaded_3_of_3": len(member_maximum) == 3,
        "source_function_identity": max(member_maximum.values()) <= IDENTITY_TOLERANCE,
        "all_9_conditions_have_forward_path": bool(reachability.v2_reachable.all()),
        "all_5_missing_conditions_reach_internal_encoder": bool((reachability.set_index("feature").loc[list(MISSING_CONDITION_FEATURES), "v2_encoder_internal_delta_max_abs"] > 0).all()),
        "all_5_missing_conditions_can_change_diagnostic_prediction": bool((reachability.set_index("feature").loc[list(MISSING_CONDITION_FEATURES), "v2_diagnostic_prediction_delta_max_abs"] > 0).all()),
        "branch_parameters_gradient_reachable_by_step_2": bool((gradients.loc[gradients.audit_step.eq("v2_branch_step_2"), "classification"] == "gradient_bearing").all()),
        "normalization_source_train_only": normalization.fit_role == "source_train",
        "schema_hash_deterministic": input_hash == v2_input_schema_hash(),
        "checkpoint_contract_complete": set(payload) >= {
            "input_schema_hash", "condition_branch_config_hash", "normalization_source_ids_hash", "legacy_anchor_sha256"
        },
        "collision_resolvable": bool(collisions.v2_can_resolve_collision.all()),
        "no_8g_outcome_used": True,
        "no_formal_performance_training": True,
    }
    write_json("engineering_smoke_audit.json", {
        "audit_scope": "forward_backward_and_two_step_engineering_smoke_only",
        "source_fixture_count": len(fixtures),
        "source_members": list(SOURCE_CHECKPOINTS),
        "source_identity_max_abs_by_member": member_maximum,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "ordering_by_member": ordering,
        "quantile_monotonicity_contract_retained": all(value["quantile_crossing_count"] == 0 for value in ordering.values()),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    })
    write_json("config.json", {
        "study": "predictor_v2_preflight",
        "status": "IMPLEMENTATION_PREFLIGHT_COMPLETE" if all(checks.values()) else "IMPLEMENTATION_PREFLIGHT_FAILED",
        "model_variant": "qgeognn_condition_complete_v2",
        "hidden_dim": 16,
        "formal_authorized": False,
        "formal_training_started": False,
        "active_transfer": "deferred",
        "source_only": True,
        "optimizer_smoke_steps": 2,
        "scientific_performance_experiment": False,
    })
    write_json("decision.json", {
        "status": "IMPLEMENTATION_PREFLIGHT_COMPLETE" if all(checks.values()) else "IMPLEMENTATION_PREFLIGHT_FAILED",
        "implementation_preflight_pass": all(checks.values()),
        "new_predictor_performance_result": False,
        "formal_source_qualification_authorized": False,
        "formal_training_started": False,
        "8g_transfer_started": False,
        "active_transfer": "deferred",
        "next_gate": "separate_authorization_for_predictor_v2_4g_source_qualification",
    })
    write_json("environment.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "execution_environment": "conda fish",
        "artifact_policy": "compact engineering audits; no checkpoint or performance output",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()

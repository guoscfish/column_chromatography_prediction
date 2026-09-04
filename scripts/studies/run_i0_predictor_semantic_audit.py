#!/usr/bin/env python3
"""Generate the non-training I0 legacy QGeoGNN semantic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data, condition_matrix, load_combined_graph_cache, qg
from src.qgeognn_al.engine import QGeoGNNActiveLearningEngine
from src.qgeognn_al.input_schema import (
    CATEGORICAL_BOND_FEATURES, CONTINUOUS_EDGE_FEATURES,
    LEGACY_CONSUMED_CONTINUOUS_POSITIONS, input_schema_hash,
    legacy_input_schema,
)
from src.qgeognn_al.resources import (
    ROOT, SOURCE_CHECKPOINTS, SOURCE_DATA, SOURCE_GRAPH_CACHE, SOURCE_SCALER,
    TARGET_DATA, TARGET_GRAPH_CACHE,
)


STUDY = ROOT / "studies/i0_predictor_semantic_audit"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _load_engine(data_path: Path, target: bool = False) -> QGeoGNNActiveLearningEngine:
    data = pd.read_csv(data_path)
    cache = (
        load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
        if target else torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    )
    scaler = json.loads(SOURCE_SCALER.read_text())
    return QGeoGNNActiveLearningEngine(
        data, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu")
    )


def input_schema_audit() -> dict:
    schema = legacy_input_schema()
    schema.update(
        {
            "schema_sha256": input_schema_hash(schema),
            "runtime_categorical_bond_features": list(qg.bond_id_names),
            "runtime_legacy_bond_float_names": list(qg.bond_float_names),
            "predictor_consumed_condition_dimensions": 4,
            "acquisition_condition_dimensions": len(condition_matrix(pd.DataFrame({
                "PE/EA": ["1/1"], "loading solvent": ["PE"],
                "Density g/ml": [1.0], "V/ul": [1.0],
                "Volume of loading solvent/ul": [1.0],
            }))[0]),
            "acquisition_joint_representation": "128D graph embedding + 9D conditions",
            "semantic_mismatch": True,
            "potential_block_dimension_dominance": True,
        }
    )
    return schema


def _forward(model, atom, angle) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        prediction, embedding = model(
            next(iter(DataLoader([atom], batch_size=1))),
            next(iter(DataLoader([angle], batch_size=1))),
        )
    return prediction.numpy(), embedding.numpy()


def feature_perturbation_audit(engine: QGeoGNNActiveLearningEngine) -> pd.DataFrame:
    model = engine._load_model(SOURCE_CHECKPOINTS[42])
    atom = engine.atom_data[0]
    angle = engine.angle_data[0]
    base_prediction, base_embedding = _forward(model, atom, angle)
    rows = []
    for position, feature in enumerate(CONTINUOUS_EDGE_FEATURES):
        changed = atom.clone()
        edge_position = len(CATEGORICAL_BOND_FEATURES) + position
        changed.edge_attr[:, edge_position] += 0.137
        prediction, embedding = _forward(model, changed, angle)
        delta_embedding = float(np.max(np.abs(embedding - base_embedding)))
        delta_prediction = float(np.max(np.abs(prediction - base_prediction)))
        expected = position in LEGACY_CONSUMED_CONTINUOUS_POSITIONS
        rows.append(
            {
                "audit_kind": "direct_edge_feature_perturbation",
                "feature": feature,
                "continuous_position": position,
                "delta_embedding_max_abs": delta_embedding,
                "delta_prediction_max_abs": delta_prediction,
                "feature_reaches_encoder": bool(delta_embedding > 0.0),
                "expected_or_unexpected": "expected" if bool(delta_embedding > 0.0) == expected else "unexpected",
            }
        )

    for feature, position, value in (
        ("loading_solvent_code_realistic", 7, 1.0),
        ("loading_amount_density_x_volume_realistic", 8, 25.0),
        ("loading_solvent_volume_ul_realistic", 9, 100.0),
    ):
        changed = atom.clone()
        edge_position = len(CATEGORICAL_BOND_FEATURES) + position
        changed.edge_attr[:, edge_position] += value
        prediction, embedding = _forward(model, changed, angle)
        rows.append(
            {
                "audit_kind": "fixed_smiles_and_eluent_loading_perturbation",
                "feature": feature,
                "continuous_position": position,
                "delta_embedding_max_abs": float(np.max(np.abs(embedding - base_embedding))),
                "delta_prediction_max_abs": float(np.max(np.abs(prediction - base_prediction))),
                "feature_reaches_encoder": bool(np.max(np.abs(embedding - base_embedding)) > 0.0),
                "expected_or_unexpected": "expected",
            }
        )
    return pd.DataFrame(rows)


def _module_group(name: str) -> str:
    if name.startswith("NN_descriptor"):
        return "NN_descriptor"
    if ".atom_encoder." in name:
        return "atom_encoder"
    if "bond_angle_encoder" in name or "convs_angle_float" in name:
        return "bond_angle_encoder"
    if "bond_float" in name:
        return "bond_float_encoder"
    if "bond_encoder" in name or "convs_bond_embeding" in name:
        return "bond_encoder"
    if "batch_norms_ba" in name:
        return "outer_batch_norms_ba"
    if "batch_norms" in name:
        return "outer_batch_norms"
    if "gnn_node.convs" in name:
        return "gnn_layers"
    if name.startswith("pool"):
        return "graph_pooling"
    if name.startswith("graph_pred_linear"):
        return "prediction_head"
    return "other"


def effective_parameter_audit(engine: QGeoGNNActiveLearningEngine) -> dict:
    model = engine._load_model(SOURCE_CHECKPOINTS[42])
    model.eval()
    atom = next(iter(DataLoader([engine.atom_data[0]], batch_size=1)))
    angle = next(iter(DataLoader([engine.angle_data[0]], batch_size=1)))
    model.zero_grad(set_to_none=True)
    prediction, _ = model(atom, angle)
    prediction.sum().backward()
    details = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        details.append(
            {
                "parameter_name": name,
                "numel": parameter.numel(),
                "requires_grad": parameter.requires_grad,
                "grad_is_none": grad is None,
                "grad_norm": None if grad is None else float(grad.norm().detach()),
                "module_group": _module_group(name),
            }
        )
    nominal = sum(row["numel"] for row in details)
    requires_grad = sum(row["numel"] for row in details if row["requires_grad"])
    gradient_bearing = sum(row["numel"] for row in details if not row["grad_is_none"])
    unreachable = sum(row["numel"] for row in details if row["grad_is_none"])
    by_group = {}
    for group in sorted({row["module_group"] for row in details}):
        selected = [row for row in details if row["module_group"] == group]
        by_group[group] = {
            "nominal_parameters": sum(row["numel"] for row in selected),
            "gradient_bearing_parameters": sum(row["numel"] for row in selected if not row["grad_is_none"]),
            "forward_unreachable_parameters": sum(row["numel"] for row in selected if row["grad_is_none"]),
        }
    return {
        "audit_mode": "loaded source checkpoint; eval forward plus backward on one canonical fixture",
        "nominal_parameters": nominal,
        "requires_grad_parameters": requires_grad,
        "gradient_bearing_parameters": gradient_bearing,
        "forward_unreachable_parameters": unreachable,
        "nn_descriptor_forward_unreachable": by_group["NN_descriptor"]["gradient_bearing_parameters"] == 0,
        "outer_geometry_batch_norms_forward_unreachable": all(
            by_group[group]["gradient_bearing_parameters"] == 0
            for group in ("outer_batch_norms", "outer_batch_norms_ba")
        ),
        "by_module_group": by_group,
        "parameters": details,
    }


def collision_audit() -> tuple[dict, pd.DataFrame]:
    summaries, records = {}, []
    ignored_columns = [
        "loading solvent", "Density g/ml", "V/ul",
        "Volume of loading solvent/ul",
    ]
    for dataset, path in (("4g", SOURCE_DATA), ("8g_no_threshold", TARGET_DATA)):
        data = pd.read_csv(path)
        grouped = data.groupby(["canonical_smiles", "PE/EA"], sort=True, dropna=False)
        duplicated = [group for _, group in grouped if len(group) > 1]
        different = [
            group for group in duplicated
            if len(group[ignored_columns].drop_duplicates()) > 1
        ]
        rows_in_duplicates = sum(len(group) for group in duplicated)
        summaries[dataset] = {
            "rows": len(data),
            "effective_input_groups": grouped.ngroups,
            "duplicated_effective_input_groups": len(duplicated),
            "rows_in_duplicated_groups": rows_in_duplicates,
            "rows_in_duplicated_groups_fraction": rows_in_duplicates / len(data),
            "groups_with_different_ignored_loading_conditions": len(different),
            "rows_with_different_ignored_loading_conditions": sum(len(group) for group in different),
        }
        for group in duplicated:
            first = group.iloc[0]
            payload = f"{dataset}|{first.canonical_smiles}|{first['PE/EA']}"
            records.append(
                {
                    "dataset": dataset,
                    "effective_group_id": hashlib.sha256(payload.encode()).hexdigest()[:16],
                    "canonical_smiles": first.canonical_smiles,
                    "PE/EA": first["PE/EA"],
                    "rows": len(group),
                    "distinct_loading_conditions": len(group[ignored_columns].drop_duplicates()),
                    "ignored_loading_conditions_differ": len(group[ignored_columns].drop_duplicates()) > 1,
                    "V1_range": float(group.V1_ml.max() - group.V1_ml.min()),
                    "V2_range": float(group.V2_ml.max() - group.V2_ml.min()),
                }
            )
    return summaries, pd.DataFrame(records)


def ordering_audit() -> dict:
    result = {"cross_target_constraint_in_model": False, "quantile_ordering_only": True}
    for dataset, path, target in (
        ("4g", SOURCE_DATA, False), ("8g_no_threshold", TARGET_DATA, True)
    ):
        engine = _load_engine(path, target=target)
        prediction = engine.predict(
            engine.data.sample_id.astype(str).tolist(), SOURCE_CHECKPOINTS[42],
            return_quantiles=True, return_embedding=False,
        ).table
        data = engine.data.set_index("sample_id").loc[prediction.sample_id]
        truth_violation = data.V1_ml.to_numpy() > data.V2_ml.to_numpy()
        median_violation = prediction.V1_q50.to_numpy() > prediction.V2_q50.to_numpy()
        interval_overlap_diagnostic = prediction.V1_q90.to_numpy() > prediction.V2_q10.to_numpy()
        error = (
            np.abs(prediction.V1_q50.to_numpy() - data.V1_ml.to_numpy())
            + np.abs(prediction.V2_q50.to_numpy() - data.V2_ml.to_numpy())
        ) / 2
        uncertainty = (
            prediction.V1_q90.to_numpy() - prediction.V1_q10.to_numpy()
            + prediction.V2_q90.to_numpy() - prediction.V2_q10.to_numpy()
        ) / 2
        high_error = error >= np.quantile(error, 0.75)
        high_uncertainty = uncertainty >= np.quantile(uncertainty, 0.75)
        result[dataset] = {
            "rows": len(data),
            "truth_V1_gt_V2_count": int(truth_violation.sum()),
            "truth_V1_gt_V2_fraction": float(truth_violation.mean()),
            "prediction_V1_q50_gt_V2_q50_count": int(median_violation.sum()),
            "prediction_V1_q50_gt_V2_q50_fraction": float(median_violation.mean()),
            "prediction_V1_q90_gt_V2_q10_count": int(interval_overlap_diagnostic.sum()),
            "prediction_V1_q90_gt_V2_q10_fraction": float(interval_overlap_diagnostic.mean()),
            "median_violations_in_high_error_quartile": int((median_violation & high_error).sum()),
            "median_violations_in_high_uncertainty_quartile": int((median_violation & high_uncertainty).sum()),
            "checkpoint_sha256": sha256_file(SOURCE_CHECKPOINTS[42]),
        }
    source_compounds = set(pd.read_csv(SOURCE_DATA).canonical_smiles)
    target_compounds = set(pd.read_csv(TARGET_DATA).canonical_smiles)
    result["source_target_compound_overlap"] = {
        "target_unique_compounds": len(target_compounds),
        "target_compounds_seen_in_4g": len(source_compounds & target_compounds),
        "target_compounds_unseen_in_4g": len(target_compounds - source_compounds),
    }
    return result


def run() -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    source_engine = _load_engine(SOURCE_DATA)
    schema = input_schema_audit()
    perturbations = feature_perturbation_audit(source_engine)
    parameters = effective_parameter_audit(source_engine)
    collision_summary, collisions = collision_audit()
    ordering = ordering_audit()
    _write_json(STUDY / "input_schema_audit.json", schema)
    _write_json(STUDY / "effective_parameter_audit.json", parameters)
    perturbations.to_csv(STUDY / "feature_perturbation_audit.csv", index=False)
    collisions.to_csv(STUDY / "input_collision_audit.csv", index=False)
    _write_json(STUDY / "input_collision_summary.json", collision_summary)
    _write_json(STUDY / "cross_target_ordering_audit.json", ordering)
    decision = {
        "study": "I0_predictor_semantic_audit",
        "status": "audit_complete",
        "scientific_role": "engineering_and_scientific_mechanism_audit",
        "new_predictor_performance_experiment": False,
        "legacy_forward_changed": False,
        "confirmed_constructed_continuous_features": schema["constructed_continuous_feature_count"],
        "confirmed_consumed_continuous_features": schema["consumed_continuous_feature_count"],
        "predictor_v2_formal_authorized": False,
        "active_transfer": "deferred",
    }
    _write_json(STUDY / "decision.json", decision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()

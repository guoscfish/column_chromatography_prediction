"""Label-safe, selection-only acquisition-geometry innovation screen."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
import platform
from pathlib import Path
import time
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..data import condition_matrix
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import DEVELOPMENT_SEEDS, package_versions, sketch_seed
from .cache import seal_cache, verify_cache
from .coverage import coreset_tp_select, extract_representations
from .diagnostics import input_identities
from .gradient_features import extract_linear_output_gradient_sketches_many, state_dict_hash
from .gradient_transforms import (
    center_width_transform,
    output_whitening_transform,
    power_normalize_gradients,
    raw_output_transform,
)
from .lcmd import _squared_distances, lcmd_tp_select
from .protocol import RestrictedLabelStore, stable_hash, validate_row_protocol
from .runner import predict_outputs


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_innovation_screen"
SOURCE_STUDY_RELATIVE = Path("studies/active_learning/qgeognn_v2_row_small_batch_benchmark")
BATCH_SIZE = 32
CONFIRMATION_SEEDS = (157, 887, 2357, 6101, 12203)
METHODS = (
    "latent_farthest_first",
    "latent_lcmd",
    "gradient_farthest_first",
    "raw_gradient_lcmd",
    "gradient_norm_topB",
    "direction_lcmd",
    "tempered_lcmd_alpha_0p5",
    "output_whitened_lcmd",
    "center_width_lcmd",
)
METHOD_LABELS = {
    "latent_farthest_first": "Latent farthest-first",
    "latent_lcmd": "Latent LCMD",
    "gradient_farthest_first": "Gradient farthest-first",
    "raw_gradient_lcmd": "Raw Gradient-LCMD",
    "gradient_norm_topB": "Gradient norm Top-B",
    "direction_lcmd": "Direction LCMD",
    "tempered_lcmd_alpha_0p5": "Tempered LCMD (alpha=0.5)",
    "output_whitened_lcmd": "Output-whitened LCMD",
    "center_width_lcmd": "Center/width LCMD",
}
METHOD_SPACES = {
    "latent_farthest_first": "latent",
    "latent_lcmd": "latent",
    "gradient_farthest_first": "raw_gradient",
    "raw_gradient_lcmd": "raw_gradient",
    "gradient_norm_topB": "raw_gradient",
    "direction_lcmd": "direction_gradient",
    "tempered_lcmd_alpha_0p5": "tempered_gradient_alpha_0p5",
    "output_whitened_lcmd": "output_whitened_gradient",
    "center_width_lcmd": "center_width_gradient",
}


@dataclass(frozen=True)
class InnovationSelections:
    positions: dict[str, np.ndarray]
    spaces: dict[str, np.ndarray]
    transform_audits: dict[str, dict[str, object]]


def validate_screen_seed(seed: int) -> int:
    value = int(seed)
    if value in CONFIRMATION_SEEDS:
        raise PermissionError("confirmation seeds are forbidden in innovation screening")
    if value not in DEVELOPMENT_SEEDS:
        raise ValueError("innovation screening is fixed to registered development seeds")
    return value


def reveal_l0_targets_only(source: Path, partition: pd.DataFrame) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    validate_row_protocol(partition)
    seed = validate_screen_seed(int(partition.outer_seed.iloc[0]))
    if set(partition.outer_seed.astype(int)) != {seed}:
        raise ValueError("partition contains more than one outer seed")
    store = RestrictedLabelStore(source, partition)
    l0_ids = partition.loc[partition.role.eq("l0"), "sample_id"].astype(str).tolist()
    truth = store.reveal(l0_ids, "initial_fit")
    if len(store.audit) != 1 or store.audit[0]["roles"] != "l0":
        raise RuntimeError("innovation screen label firewall admitted non-L0 truth")
    return truth, tuple(store.audit)


def gradient_norm_topb(pool_features: np.ndarray, batch_size: int) -> np.ndarray:
    values = np.asarray(pool_features)
    if values.ndim != 2 or not np.isfinite(values).all() or not 0 < int(batch_size) <= len(values):
        raise ValueError("finite pool feature matrix and valid batch size required")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    return np.argsort(-norms, kind="stable")[: int(batch_size)].astype(np.int64)


def select_innovation_methods(
    *,
    raw_gradients: np.ndarray,
    latent_features: np.ndarray,
    output_whitened_gradients: np.ndarray,
    center_width_gradients: np.ndarray,
    l0_count: int,
    batch_size: int = BATCH_SIZE,
) -> InnovationSelections:
    raw = _validated_feature_matrix(raw_gradients, "raw_gradients")
    latent = _validated_feature_matrix(latent_features, "latent_features")
    whitened = _validated_feature_matrix(output_whitened_gradients, "output_whitened_gradients")
    center_width = _validated_feature_matrix(center_width_gradients, "center_width_gradients")
    if len({len(raw), len(latent), len(whitened), len(center_width)}) != 1:
        raise ValueError("all acquisition spaces must share ordered L0/U0 rows")
    if not 0 < int(l0_count) < len(raw):
        raise ValueError("l0_count must split nonempty L0 and U0")
    pool_size = len(raw) - int(l0_count)
    if not 0 < int(batch_size) <= pool_size:
        raise ValueError("invalid innovation-screen batch size")

    direction = power_normalize_gradients(raw, 0.0)
    tempered = power_normalize_gradients(raw, 0.5)
    spaces = {
        "raw_gradient": raw,
        "latent": latent,
        "direction_gradient": direction.features,
        "tempered_gradient_alpha_0p5": tempered.features,
        "output_whitened_gradient": whitened,
        "center_width_gradient": center_width,
    }
    n = int(l0_count)
    positions = {
        "latent_farthest_first": coreset_tp_select(latent[n:], latent[:n], batch_size),
        "latent_lcmd": lcmd_tp_select(latent[n:], latent[:n], batch_size).selected_pool_positions,
        "gradient_farthest_first": coreset_tp_select(raw[n:], raw[:n], batch_size),
        "raw_gradient_lcmd": lcmd_tp_select(raw[n:], raw[:n], batch_size).selected_pool_positions,
        "gradient_norm_topB": gradient_norm_topb(raw[n:], batch_size),
        "direction_lcmd": lcmd_tp_select(direction.features[n:], direction.features[:n], batch_size).selected_pool_positions,
        "tempered_lcmd_alpha_0p5": lcmd_tp_select(
            tempered.features[n:], tempered.features[:n], batch_size
        ).selected_pool_positions,
        "output_whitened_lcmd": lcmd_tp_select(whitened[n:], whitened[:n], batch_size).selected_pool_positions,
        "center_width_lcmd": lcmd_tp_select(center_width[n:], center_width[:n], batch_size).selected_pool_positions,
    }
    if tuple(positions) != METHODS:
        raise RuntimeError("innovation method ordering drift")
    for method, selected in positions.items():
        if len(selected) != batch_size or len(np.unique(selected)) != batch_size:
            raise RuntimeError(f"{method} returned an invalid batch")
        if np.any(selected < 0) or np.any(selected >= pool_size):
            raise RuntimeError(f"{method} selected outside U0")
    return InnovationSelections(
        positions,
        spaces,
        {"direction": direction.audit, "tempered_alpha_0p5": tempered.audit},
    )


def execute_innovation_screen(artifact_repository: Path, study: Path = STUDY) -> dict[str, object]:
    """Run the fixed five-seed screen without training or hidden-label access."""

    artifact_repository = Path(artifact_repository).resolve()
    study = Path(study)
    torch.set_num_threads(2)
    for directory in (study, study / "results", study / "figures", study / "runtime"):
        directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(
        SOURCE_DATA,
        usecols=[
            "sample_id", "canonical_smiles", "Density g/ml", "V/ul", "loading solvent",
            "Volume of loading solvent/ul", "PE/EA",
        ],
    ).reset_index(drop=True)
    graph_cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    selected_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    transform_audits: dict[str, object] = {}
    label_access: list[dict[str, object]] = []
    source_artifacts: dict[str, str] = {}
    raw_regressions: list[dict[str, object]] = []

    for seed in DEVELOPMENT_SEEDS:
        print(json.dumps({"innovation_seed_started": seed}), flush=True)
        seed_result = _execute_seed(
            seed=seed,
            artifact_repository=artifact_repository,
            study=study,
            data=data,
            graph_cache=graph_cache,
        )
        selected_frames.append(seed_result["selected"])
        profile_frames.append(seed_result["profiles"])
        overlap_frames.append(seed_result["overlap"])
        transform_audits[str(seed)] = seed_result["transform_audit"]
        label_access.extend(seed_result["label_access"])
        source_artifacts.update(seed_result["source_artifacts"])
        raw_regressions.append(seed_result["raw_regression"])
        print(json.dumps({"innovation_seed_complete": seed}), flush=True)

    selected = pd.concat(selected_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    selected.to_csv(study / "results/selected_batches.csv", index=False)
    overlap.to_csv(study / "results/selection_overlap.csv", index=False)
    profiles.to_csv(study / "results/method_profiles.csv", index=False)
    atomic_json(study / "results/transform_audit.json", transform_audits)
    pd.DataFrame(label_access).to_csv(study / "results/label_access_audit.csv", index=False)
    pd.DataFrame(raw_regressions).to_csv(study / "results/raw_compatibility_regression.csv", index=False)
    _make_figures(overlap, profiles, study / "figures")
    _write_mechanism_reports(overlap, profiles, study)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "device": "cpu",
        "torch_threads": 2,
        "base_commit": "a419ea75e7894004aef393cb3734a7ae5e1356a1",
        "branch": "exp/qgeognn-v2-4g-al-innovation-screen",
        "artifact_repository": str(artifact_repository),
        "source_data_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
    }
    atomic_json(study / "environment.json", environment)
    manifest = {
        "study": study.name,
        "status": "SELECTION_ONLY_INNOVATION_SCREEN_COMPLETE",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds_accessed": [],
        "batch_size": BATCH_SIZE,
        "methods": list(METHODS),
        "training_performed": False,
        "selected_label_reveal_performed": False,
        "test_truth_access_count": 0,
        "read_only_source_artifacts": source_artifacts,
    }
    artifact_files = [
        path for path in study.rglob("*")
        if path.is_file() and "runtime" not in path.parts and path.name != "artifact_manifest.json"
    ]
    manifest["files"] = {str(path.relative_to(study)): sha256_file(path) for path in sorted(artifact_files)}
    atomic_json(study / "artifact_manifest.json", manifest)
    return {
        "status": manifest["status"],
        "seeds": list(DEVELOPMENT_SEEDS),
        "methods": len(METHODS),
        "selected_rows": len(selected),
        "test_truth_access_count": 0,
    }


def _execute_seed(*, seed, artifact_repository, study, data, graph_cache):
    seed = validate_screen_seed(seed)
    source_study = artifact_repository / SOURCE_STUDY_RELATIVE
    source_runtime = source_study / "runtime" / f"seed_{seed}"
    partition_path = ROOT / SOURCE_STUDY_RELATIVE / "splits" / f"row_seed_{seed}.csv"
    partition = pd.read_csv(partition_path)
    validate_row_protocol(partition)
    if data.sample_id.astype(str).tolist() != partition.sample_id.astype(str).tolist():
        raise RuntimeError("canonical data and frozen split identity order differ")
    roles = {
        role: partition.loc[partition.role.eq(role), "canonical_index"].to_numpy(dtype=int)
        for role in ("l0", "u0", "validation", "test")
    }
    outer = np.r_[roles["l0"], roles["u0"]]
    l0_count = len(roles["l0"])
    if l0_count != 333 or len(roles["u0"]) != 2997:
        raise RuntimeError("frozen innovation split size drift")

    gradient_path = source_runtime / "gradient_features.npz"
    gradient_receipt = _verified_receipt(gradient_path)
    checkpoint = source_runtime / "baseline_l0/best.pt"
    scrubbed_path = source_runtime / "scrubbed_graphs.pt"
    scrubbed_receipt = _verified_receipt(scrubbed_path)
    context_path = source_runtime / "context.json"
    raw_selection_path = source_runtime / "b32/selected_batches.csv"
    required_sources = (gradient_path, gradient_path.with_suffix(".npz.contract.json"), checkpoint,
                        scrubbed_path, scrubbed_path.with_suffix(".pt.contract.json"), context_path,
                        raw_selection_path)
    for path in required_sources:
        if not path.is_file():
            raise FileNotFoundError(f"required frozen development artifact missing: {path}")
        if any(f"seed_{blocked}" in str(path) for blocked in CONFIRMATION_SEEDS):
            raise PermissionError("confirmation artifact path entered innovation screen")

    contract = gradient_receipt["contract"]
    if contract["checkpoint_sha256"] != sha256_file(checkpoint):
        raise RuntimeError("gradient cache/checkpoint contract mismatch")
    expected_ids = data.iloc[outer].sample_id.astype(str).tolist()
    if contract["sample_ids"] != expected_ids or int(contract["sketch_seed"]) != sketch_seed(seed):
        raise RuntimeError("gradient cache identity/seed contract mismatch")
    if scrubbed_receipt["contract_hash"] != contract["context_hash"]:
        raise RuntimeError("scrubbed graphs and gradient context mismatch")
    with np.load(gradient_path) as stored:
        raw_gradients = stored["features"]
        stored_indices = stored["canonical_indices"]
    if raw_gradients.shape != (3330, 512) or not np.array_equal(stored_indices, outer):
        raise RuntimeError("raw gradient cache shape/order drift")

    context = json.loads(context_path.read_text())
    preprocessing = context["preprocessing"]
    scales = tuple(float(preprocessing["target_scales"][target]) for target in ("V1", "V2"))
    if list(scales) != list(contract["scales"]):
        raise RuntimeError("L0 target-scale contract drift")
    l0_truth, access = reveal_l0_targets_only(SOURCE_DATA, partition)
    raw_transform, raw_audit = raw_output_transform(scales)
    whitening_transform, whitening_audit = output_whitening_transform(l0_truth, scales)
    cw_transform, cw_audit = center_width_transform(l0_truth)

    atom, angle = torch.load(scrubbed_path, weights_only=False)
    if len(atom) != len(data) or len(angle) != len(data) or any(bool(torch.count_nonzero(graph.y)) for graph in atom):
        raise RuntimeError("source graph artifact is not aligned and label-scrubbed")
    model = load_predictor_checkpoint(checkpoint)
    latent = extract_representations(model, atom, angle, outer)
    predictions, prediction_order = predict_outputs(model, atom, angle, outer)
    if not np.array_equal(prediction_order, outer):
        raise RuntimeError("prediction order drift")

    runtime = study / "runtime" / f"seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=True)
    output_transforms = {"output_whitened": whitening_transform, "center_width": cw_transform}
    cache_records = {}
    all_cached = True
    for name, transform in output_transforms.items():
        feature_path = runtime / f"{name}_gradient_features.npz"
        feature_contract = {
            "outer_seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "ordered_sample_ids_hash": stable_hash(expected_ids),
            "sketch_seed": sketch_seed(seed),
            "sketch_dimension": 512,
            "output_transform": transform.tolist(),
        }
        cached = verify_cache(feature_path, feature_contract)
        all_cached &= cached
        cache_records[name] = (feature_path, feature_contract)
    transformed = {}
    extraction_audits = {}
    if all_cached:
        for name, (feature_path, _) in cache_records.items():
            with np.load(feature_path) as stored:
                values = stored["features"]
            audit = json.loads((runtime / f"{name}_gradient_audit.json").read_text())
            transformed[name] = values
            extraction_audits[name] = audit
    else:
        started = time.perf_counter()
        results = extract_linear_output_gradient_sketches_many(
            model, atom, angle, outer, output_transforms, dimension=512, sketch_seed=sketch_seed(seed),
            progress=lambda row: print(json.dumps({"outer_seed": seed, "transforms": list(output_transforms), **row}), flush=True)
            if int(row["completed_rows"]) % 500 == 0 or int(row["completed_rows"]) == len(outer) else None,
        )
        for name, result in results.items():
            feature_path, feature_contract = cache_records[name]
            values = result.features
            np.savez_compressed(feature_path, features=values, canonical_indices=outer)
            seal_cache(feature_path, feature_contract)
            audit = {**result.audit, "wall_seconds": time.perf_counter() - started}
            atomic_json(runtime / f"{name}_gradient_audit.json", audit)
            transformed[name] = values
            extraction_audits[name] = audit
    for name, values in transformed.items():
        if values.shape != raw_gradients.shape or not np.isfinite(values).all():
            raise RuntimeError(f"invalid transformed gradient cache: {name}")

    selections = select_innovation_methods(
        raw_gradients=raw_gradients,
        latent_features=latent,
        output_whitened_gradients=transformed["output_whitened"],
        center_width_gradients=transformed["center_width"],
        l0_count=l0_count,
    )
    raw_regression = _raw_selection_regression(
        seed, data, roles["u0"], selections.positions["raw_gradient_lcmd"], raw_selection_path
    )
    identities = input_identities(data, graph_cache, atom, angle, outer)
    uncertainty = _safe_development_uncertainty(
        source_runtime, data.iloc[roles["u0"]].sample_id.astype(str).tolist(), predictions[l0_count:], scales
    )
    selected = _selected_table(
        seed=seed,
        data=data,
        outer=outer,
        l0_count=l0_count,
        selections=selections,
        identities=identities,
        raw_gradients=raw_gradients,
        latent=latent,
        predictions=predictions,
        uncertainty=uncertainty,
    )
    profiles = _method_profiles(selected)
    overlap = _selection_overlap(seed, selections.positions)
    source_hashes = {str(path): sha256_file(path) for path in required_sources}
    source_hashes.update(uncertainty["source_artifacts"])
    return {
        "selected": selected,
        "profiles": profiles,
        "overlap": overlap,
        "label_access": [{"outer_seed": seed, **row, "u0_truth_accessed": False,
                          "test_truth_accessed": False} for row in access],
        "source_artifacts": source_hashes,
        "raw_regression": raw_regression,
        "transform_audit": {
            "raw": raw_audit,
            "whitened": {**whitening_audit, "extraction": extraction_audits["output_whitened"]},
            "center_width": {**cw_audit, "extraction": extraction_audits["center_width"]},
            **selections.transform_audits,
            "checkpoint_state_hash": state_dict_hash(model),
            "raw_gradient_cache_sha256": sha256_file(gradient_path),
            "raw_historical_b32_selection_identical": raw_regression["ordered_selected_ids_identical"],
        },
    }


def _verified_receipt(path: Path) -> dict[str, object]:
    receipt_path = path.with_suffix(path.suffix + ".contract.json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"incomplete frozen artifact: {path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt["sha256"] != sha256_file(path) or receipt["contract_hash"] != stable_hash(receipt["contract"]):
        raise RuntimeError(f"frozen artifact receipt mismatch: {path}")
    return receipt


def _raw_selection_regression(seed, data, pool_indices, selected_positions, path):
    historical = pd.read_csv(path)
    old_ids = historical.loc[historical.arm.eq("lcmd")].sort_values("selection_order").sample_id.astype(str).tolist()
    current_ids = data.iloc[pool_indices[selected_positions]].sample_id.astype(str).tolist()
    if current_ids != old_ids:
        raise RuntimeError(f"raw Gradient-LCMD historical B32 selection drift for seed {seed}")
    return {
        "outer_seed": seed,
        "batch_size": BATCH_SIZE,
        "ordered_selected_ids_identical": True,
        "ordered_selected_ids_hash": stable_hash(current_ids),
        "historical_selection_sha256": sha256_file(path),
    }


def _safe_development_uncertainty(runtime, pool_ids, member0, scales):
    predictions = [np.asarray(member0)]
    sources = {}
    for member in (1, 2):
        path = runtime / f"ensemble_member_{member}/predictions.csv.gz"
        table = pd.read_csv(path)
        if table.sample_id.astype(str).tolist() != pool_ids:
            raise RuntimeError("frozen development ensemble prediction order drift")
        predictions.append(table.drop(columns="sample_id").to_numpy(dtype=float))
        sources[str(path)] = sha256_file(path)
    values = np.stack(predictions)
    disagreement = values[:, :, [1, 4]].std(axis=0, ddof=0)
    scores = np.linalg.norm(disagreement / np.asarray(scales)[None, :], axis=1)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return {"scores": scores, "ranks": ranks, "source_artifacts": sources}


def _selected_table(*, seed, data, outer, l0_count, selections, identities, raw_gradients,
                    latent, predictions, uncertainty):
    condition = condition_matrix(data, outer).astype(np.float64)
    condition_scale = condition.std(axis=0, ddof=0)
    condition_scale[condition_scale < 1e-8] = 1.0
    condition_z = (condition - condition.mean(axis=0)) / condition_scale
    distance_vectors = {
        name: _nearest_l0_distance(values, l0_count) for name, values in selections.spaces.items()
    }
    distance_vectors["condition"] = _nearest_l0_distance(condition_z, l0_count)
    raw_norm = np.linalg.norm(raw_gradients.astype(np.float64), axis=1)
    feature_norms = {
        name: np.linalg.norm(values.astype(np.float64), axis=1) for name, values in selections.spaces.items()
    }
    input_hashes = identities.input_hash.astype(str).to_numpy()
    rows = []
    for method in METHODS:
        space_name = METHOD_SPACES[method]
        seen_input_hashes = set(input_hashes[:l0_count])
        for selection_order, pool_position in enumerate(selections.positions[method]):
            outer_position = l0_count + int(pool_position)
            canonical_index = int(outer[outer_position])
            record = data.iloc[canonical_index]
            input_hash = input_hashes[outer_position]
            rows.append({
                "outer_seed": seed,
                "method": method,
                "selection_order": int(selection_order),
                "canonical_row_id": str(record.sample_id),
                "sample_id": str(record.sample_id),
                "canonical_index": canonical_index,
                "compound_id": str(record.canonical_smiles),
                "canonical_smiles": str(record.canonical_smiles),
                "acquisition_space": space_name,
                "raw_gradient_norm": float(raw_norm[outer_position]),
                "transformed_feature_norm": float(feature_norms[space_name][outer_position]),
                "nearest_l0_distance_own_space": float(distance_vectors[space_name][pool_position]),
                "nearest_l0_raw_gradient_distance": float(distance_vectors["raw_gradient"][pool_position]),
                "nearest_l0_latent_distance": float(distance_vectors["latent"][pool_position]),
                "nearest_l0_condition_distance": float(distance_vectors["condition"][pool_position]),
                "exact_model_input_hash": input_hash,
                "exact_X_duplicate_to_L0_or_earlier_batch": input_hash in seen_input_hashes,
                "loading_solvent": str(record["loading solvent"]),
                "eluent_EA_ratio": str(record["PE/EA"]),
                "loading_amount_density_x_volume": float(record["Density g/ml"] * record["V/ul"]),
                "loading_volume_ul": float(record["Volume of loading solvent/ul"]),
                "predicted_V1_q50": float(predictions[outer_position, 1]),
                "predicted_V2_q50": float(predictions[outer_position, 4]),
                "ensemble_uncertainty": float(uncertainty["scores"][pool_position]),
                "ensemble_uncertainty_rank": int(uncertainty["ranks"][pool_position]),
                "ensemble_uncertainty_top_25pct": bool(
                    uncertainty["ranks"][pool_position] <= math.ceil(len(uncertainty["ranks"]) * 0.25)
                ),
            })
            seen_input_hashes.add(input_hash)
    result = pd.DataFrame(rows)
    if not result.groupby(["outer_seed", "method"]).size().eq(BATCH_SIZE).all():
        raise RuntimeError("selected batch table size mismatch")
    return result


def _method_profiles(selected):
    rows = []
    for (seed, method), group in selected.groupby(["outer_seed", "method"], sort=False):
        ordered = group.sort_values("selection_order")
        compound_counts = ordered.canonical_smiles.value_counts()
        amount = ordered.loading_amount_density_x_volume
        volume = ordered.loading_volume_ul
        solvent_counts = ordered.loading_solvent.value_counts().sort_index().to_dict()
        rows.append({
            "outer_seed": int(seed),
            "method": method,
            "acquisition_space": METHOD_SPACES[method],
            "batch_size": len(ordered),
            "mean_raw_gradient_norm": ordered.raw_gradient_norm.mean(),
            "median_raw_gradient_norm": ordered.raw_gradient_norm.median(),
            "mean_transformed_feature_norm": ordered.transformed_feature_norm.mean(),
            "mean_nearest_l0_distance_own_space": ordered.nearest_l0_distance_own_space.mean(),
            "mean_nearest_l0_raw_gradient_distance": ordered.nearest_l0_raw_gradient_distance.mean(),
            "mean_nearest_l0_latent_distance": ordered.nearest_l0_latent_distance.mean(),
            "mean_nearest_l0_condition_distance": ordered.nearest_l0_condition_distance.mean(),
            "unique_compounds": ordered.canonical_smiles.nunique(),
            "repeated_compound_fraction": float(
                compound_counts[compound_counts > 1].sum() / len(ordered) if (compound_counts > 1).any() else 0.0
            ),
            "exact_X_duplicate_rate": ordered.exact_X_duplicate_to_L0_or_earlier_batch.mean(),
            "loading_solvent_counts_json": json.dumps(solvent_counts, sort_keys=True),
            "loading_solvent_PE_count": int(solvent_counts.get("PE", 0)),
            "loading_solvent_EA_count": int(solvent_counts.get("EA", 0)),
            "loading_solvent_DCM_count": int(solvent_counts.get("DCM", 0)),
            "eluent_EA_ratio_unique_count": ordered.eluent_EA_ratio.nunique(),
            "eluent_EA_ratio_values_json": json.dumps(sorted(ordered.eluent_EA_ratio.unique().tolist())),
            **_quantile_columns("loading_amount", amount),
            **_quantile_columns("loading_volume", volume),
            "predicted_V1_q50_mean": ordered.predicted_V1_q50.mean(),
            "predicted_V1_q50_std": ordered.predicted_V1_q50.std(ddof=0),
            "predicted_V2_q50_mean": ordered.predicted_V2_q50.mean(),
            "predicted_V2_q50_std": ordered.predicted_V2_q50.std(ddof=0),
            "mean_ensemble_uncertainty": ordered.ensemble_uncertainty.mean(),
            "mean_ensemble_uncertainty_rank": ordered.ensemble_uncertainty_rank.mean(),
            "fraction_in_uncertainty_top_25pct": ordered.ensemble_uncertainty_top_25pct.mean(),
        })
    return pd.DataFrame(rows)


def _quantile_columns(prefix, values):
    quantiles = np.quantile(np.asarray(values, dtype=float), (0.0, 0.25, 0.5, 0.75, 1.0))
    return {f"{prefix}_{name}": float(value) for name, value in zip(("min", "q25", "median", "q75", "max"), quantiles)}


def _selection_overlap(seed, positions):
    rows = []
    sets = {method: set(map(int, values)) for method, values in positions.items()}
    for method_a, method_b in combinations(METHODS, 2):
        count = len(sets[method_a] & sets[method_b])
        rows.append({
            "outer_seed": seed,
            "method_a": method_a,
            "method_b": method_b,
            "overlap_count": count,
            "overlap_fraction": count / BATCH_SIZE,
            "jaccard": count / len(sets[method_a] | sets[method_b]),
            "comparison_to_raw_gradient_lcmd": "raw_gradient_lcmd" in (method_a, method_b),
        })
    return pd.DataFrame(rows)


def _nearest_l0_distance(features, l0_count):
    values = np.asarray(features)
    return np.sqrt(_squared_distances(values[l0_count:], values[:l0_count]).min(axis=1))


def _validated_feature_matrix(features, name):
    result = np.asarray(features)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return result


def _pair_mean(overlap, left, right, column="overlap_fraction"):
    match = overlap.loc[
        ((overlap.method_a == left) & (overlap.method_b == right))
        | ((overlap.method_a == right) & (overlap.method_b == left)), column
    ]
    if len(match) != len(DEVELOPMENT_SEEDS):
        raise RuntimeError("pairwise overlap table is incomplete")
    return float(match.mean())


def _make_figures(overlap, profiles, directory):
    directory.mkdir(parents=True, exist_ok=True)
    n = len(METHODS)
    matrix = np.eye(n)
    for i, left in enumerate(METHODS):
        for j, right in enumerate(METHODS):
            if i < j:
                matrix[i, j] = matrix[j, i] = _pair_mean(overlap, left, right)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.55 else "black", fontsize=7)
    labels = [METHOD_LABELS[method] for method in METHODS]
    ax.set(xticks=range(n), yticks=range(n), xticklabels=labels, yticklabels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(image, ax=ax, label="Mean overlap fraction across five seeds")
    fig.tight_layout()
    fig.savefig(directory / "selection_overlap_heatmap.png", dpi=180)
    plt.close(fig)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(METHODS)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for color, method in zip(colors, METHODS):
        rows = profiles.loc[profiles.method.eq(method)]
        ax.scatter(rows.mean_raw_gradient_norm, rows.mean_nearest_l0_raw_gradient_distance,
                   label=METHOD_LABELS[method], color=color, s=42, alpha=0.85)
    ax.set(xlabel="Batch mean raw gradient norm", ylabel="Batch mean nearest-L0 raw-gradient distance")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "gradient_norm_vs_distance.png", dpi=180)
    plt.close(fig)

    means = profiles.groupby("method", sort=False).mean(numeric_only=True).loc[list(METHODS)]
    raw = means.loc["raw_gradient_lcmd"]
    metrics = (
        ("unique_compounds", "Unique compounds"),
        ("mean_nearest_l0_raw_gradient_distance", "Gradient distance"),
        ("mean_nearest_l0_condition_distance", "Condition distance"),
    )
    x = np.arange(len(METHODS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, (column, label) in enumerate(metrics):
        ax.bar(x + (index - 1) * width, means[column] / raw[column], width=width, label=label)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set(xticks=x, xticklabels=[METHOD_LABELS[m] for m in METHODS], ylabel="Ratio to raw Gradient-LCMD")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "method_profile_comparison_to_raw.png", dpi=180)
    plt.close(fig)


def _write_mechanism_reports(overlap, profiles, study):
    raw_overlap = {}
    for method in METHODS:
        raw_overlap[method] = 1.0 if method == "raw_gradient_lcmd" else _pair_mean(
            overlap, method, "raw_gradient_lcmd"
        )
    pair_rows = "\n".join(
        f"| {label} | {value * BATCH_SIZE:.1f}/32 | {value:.3f} |"
        for label, value in ((METHOD_LABELS[m], raw_overlap[m]) for m in METHODS)
    )
    factorial = {
        "A_latent_to_gradient_fixed_farthest_first": _pair_mean(
            overlap, "latent_farthest_first", "gradient_farthest_first"
        ),
        "B_farthest_first_to_lcmd_fixed_gradient": _pair_mean(
            overlap, "gradient_farthest_first", "raw_gradient_lcmd"
        ),
        "C_farthest_first_to_lcmd_fixed_latent": _pair_mean(
            overlap, "latent_farthest_first", "latent_lcmd"
        ),
        "D_gradient_norm_to_raw_lcmd": _pair_mean(overlap, "gradient_norm_topB", "raw_gradient_lcmd"),
    }
    identical = []
    for left, right in combinations(METHODS, 2):
        rows = overlap.loc[
            ((overlap.method_a == left) & (overlap.method_b == right))
            | ((overlap.method_a == right) & (overlap.method_b == left))
        ]
        if rows.overlap_count.eq(BATCH_SIZE).all():
            identical.append(f"{left} = {right}")
    candidates = [
        "direction_lcmd", "tempered_lcmd_alpha_0p5", "output_whitened_lcmd",
        "center_width_lcmd", "gradient_farthest_first", "latent_lcmd",
    ]
    shortlist = sorted(candidates, key=lambda method: (raw_overlap[method], METHODS.index(method)))[:3]
    mechanism = f"""# Selection mechanism summary

Status: **selection-only innovation screening complete**.

This report describes label-free selection geometry only. It does not compare
prediction accuracy, reveal selected labels, or promote a method.

## Mean overlap with raw Gradient-LCMD

| Method | Mean overlap count | Mean overlap fraction |
| --- | ---: | ---: |
{pair_rows}

## Representation x selector controls

- Latent -> gradient at fixed farthest-first retained {factorial['A_latent_to_gradient_fixed_farthest_first']:.3f} of each batch on average.
- Farthest-first -> LCMD at fixed gradient retained {factorial['B_farthest_first_to_lcmd_fixed_gradient']:.3f}.
- Farthest-first -> LCMD at fixed latent retained {factorial['C_farthest_first_to_lcmd_fixed_latent']:.3f}.
- Gradient-norm Top-B retained {factorial['D_gradient_norm_to_raw_lcmd']:.3f} of raw Gradient-LCMD.

These contrasts show how strongly selections change under one controlled
representation or selector substitution. They do not identify a causal
performance mechanism.

## Identical batches

{('; '.join(identical)) if identical else 'No method pair selected the same ordered membership set in all five seeds.'}

## Mechanism-only next-stage shortlist

The three most selection-distinct prespecified candidates relative to raw
Gradient-LCMD are: {', '.join(f'`{method}`' for method in shortlist)}. This is a
diversity-of-geometry shortlist for a separately authorized one-step training
study, not an accuracy ranking.
"""
    (study / "mechanism_summary.md").write_text(mechanism)
    results = f"""# Results - QGeoGNN-V2 row AL innovation screen

Status: **selection-only innovation screening complete**.

The screen used only seeds {list(DEVELOPMENT_SEEDS)}, B={BATCH_SIZE}, L0 labels,
frozen L0 checkpoints, and label-free U0 inputs/model outputs. No selected label,
test truth, confirmation result, retraining, or test metric was accessed.

## Mathematical definitions

- `raw_gradient_lcmd`: LCMD-TP on the existing 512D CountSketch of
  `grad(V1_q50/s_V1)` and `grad(V2_q50/s_V2)`.
- `latent_lcmd`: the same LCMD-TP selector on the authoritative 128D pre-head
  QGeoGNN-V2 representation.
- `gradient_farthest_first`: TP greedy k-center on the raw gradient sketch.
- `gradient_norm_topB`: stable descending `||g||_2` ranking.
- `direction_lcmd`: LCMD-TP on `g/||g||` with audited zero vectors retained.
- `tempered_lcmd_alpha_0p5`: LCMD-TP on `g/sqrt(||g||)`.
- `output_whitened_lcmd`: LCMD-TP on gradients after the fixed L0 population
  covariance transform `Sigma_z^(-1/2) diag(1/s_V1,1/s_V2)`.
- `center_width_lcmd`: LCMD-TP on gradients of `(V1+V2)/(2s_C)` and
  `(V2-V1)/s_W`, with L0 population scales.
- `latent_farthest_first`: existing current-V2 CoreSet control.

## Raw compatibility

All five ordered raw Gradient-LCMD B=32 selections exactly match the frozen
historical development selection artifacts. The legacy raw extractor was not
rewritten.

## Selection interpretation

{mechanism.split('## Representation x selector controls', 1)[1].split('## Identical batches', 1)[0].strip()}

Full seed-by-method overlaps are in `results/selection_overlap.csv`; profiles
and transform numerics are in `results/method_profiles.csv` and
`results/transform_audit.json`.

## Next-stage candidates

For a separately authorized one-step training comparison, the mechanism-only
shortlist is {', '.join(f'`{method}`' for method in shortlist)} because these
prespecified methods produced the lowest mean membership overlap with the raw
control. Hidden-label performance played no role in this ranking.

## Infrastructure boundary

The current code now supports arbitrary fixed 2x2 linear output-gradient
coordinates and reusable LCMD/farthest-first controls. Target-IVR still needs a
predictive covariance/influence operator and a candidate-to-target utility
contract. MaxDet still needs a numerically stable incremental log-determinant
selector (with explicit regularization and tie rules). Neither method is
implemented or executed here.
"""
    (study / "RESULTS.md").write_text(results)

"""Independent, resumable runner for Gradient+Latent Fusion-MaxDet."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import SOURCE_DATA
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, load_features, seed_config, sketch_seed
from .coverage import extract_representations
from .fused_features import (
    FUSED_DIMENSION,
    fuse_gradient_latent_features,
    latent_common_mode_diagnostics,
    representation_diagnostics,
)
from .fused_maxdet_study import (
    DEFAULT_SPEC,
    FusedMaxDetStudySpec,
    frozen_a050_round0_paths,
    historical_round0,
    validate_pre_test_manifest,
)
from .gradient_features import extract_q50_gradient_sketches, state_dict_hash
from .maxdet import conditional_gradient_maxdet
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .cache import array_hash
from .runner import fit_from_same_initialization, predict_outputs
from .sequential_acquisition import validate_trajectory_transition
from .sequential_protocol import BATCH_SIZE, INITIAL_ACTIVE_LABELS


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json_once(path: Path, value: dict) -> None:
    if Path(path).exists():
        if _json(path) != value:
            raise RuntimeError(f"refusing to replace mismatched JSON: {path}")
        return
    atomic_json(path, value)


def _write_csv_once(path: Path, frame: pd.DataFrame, *, rtol: float = 0.0, atol: float = 1e-12) -> None:
    path = Path(path)
    if path.exists():
        current = pd.read_csv(path, keep_default_na=False)
        if list(current.columns) != list(frame.columns) or len(current) != len(frame):
            raise RuntimeError(f"refusing to replace mismatched CSV: {path}")
        for column in frame:
            left, right = current[column], frame[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                if not np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=rtol, atol=atol, equal_nan=True):
                    raise RuntimeError(f"refusing to replace mismatched CSV: {path}")
            elif left.astype(str).tolist() != right.astype(str).tolist():
                raise RuntimeError(f"refusing to replace mismatched CSV: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _phase_budget(active: int) -> dict[str, int | float]:
    return {"active_label_count": int(active), "active_label_fraction_outer_train": int(active) / 3330,
            "shared_validation_label_count": 416, "total_observed_non_test_labels": int(active) + 416,
            "total_observed_fraction_full_dataset": (int(active) + 416) / 4163}


@dataclass
class FusedSeedContext:
    seed: int
    partition: pd.DataFrame
    runtime: Path
    protocol_hash: str
    spec: FusedMaxDetStudySpec = DEFAULT_SPEC

    def __post_init__(self) -> None:
        torch.set_num_threads(2)
        self.seed = int(self.seed)
        self.runtime = Path(self.runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        validate_row_protocol(self.partition)
        self.data = load_features()
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError("fusion split/source identity mismatch")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        old = _json(historical_round0(self.seed)["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(historical_round0(self.seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("scrubbed graph cache contains labels")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        context = {"study": self.spec.study.name, "method": self.spec.method, "alpha": float(self.spec.alpha),
                   "protocol_hash": self.protocol_hash, "outer_seed": self.seed,
                   "split_hash": stable_hash(self.partition.to_dict("list")),
                   "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])), "u0_ids_hash": ids_hash(self.ids(self.roles["u0"])),
                   "validation_ids_hash": ids_hash(self.ids(self.roles["validation"])), "test_ids_hash": ids_hash(self.ids(self.roles["test"])),
                   "normalization": old["normalization"], "preprocessing": self.preprocessing,
                   "historical_context_hash": stable_hash(old), "test_truth_access_count": 0}
        _write_json_once(self.runtime / "context.json", context)
        self.context = context

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def new_store(self) -> RestrictedLabelStore:
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        if not np.array_equal(store.reveal(self.ids(self.roles["l0"]), "initial_fit"), self.l0_truth):
            raise RuntimeError("fusion L0 truth drift")
        if not np.array_equal(store.reveal(self.ids(self.roles["validation"]), "initial_fit"), self.validation_truth):
            raise RuntimeError("fusion validation truth drift")
        return store

    def training_config(self) -> dict:
        config = seed_config(self.seed, 0, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def fit(self, round_index: int, labeled: np.ndarray, truth: np.ndarray, prediction_indices: np.ndarray, runtime: Path) -> dict[str, object]:
        train_ids = self.ids(labeled)
        contract = {"study": self.spec.study.name, "protocol_hash": self.protocol_hash, "outer_seed": self.seed,
                    "method": self.spec.method, "alpha": float(self.spec.alpha), "round": int(round_index), "member": 0,
                    "train_sample_ids": train_ids, "validation_sample_ids": self.ids(self.roles["validation"]),
                    "L_t_ids_hash": ids_hash(train_ids), "prediction_role": "test_X_without_test_truth", "test_truth_access_count": 0}
        audit = fit_from_same_initialization(atom_base=self.atom, angle=self.angle, normalization=self.normalization,
            preprocessing=self.preprocessing, train_indices=labeled, train_truth=truth,
            validation_indices=self.roles["validation"], validation_truth=self.validation_truth,
            prediction_indices=prediction_indices, prediction_sample_ids=self.ids(prediction_indices),
            initialization_seed=initialization_seed(self.seed, 0), training_config=self.training_config(),
            contract=contract, runtime=runtime)
        return {"outer_seed": self.seed, "method": self.spec.method, "alpha": float(self.spec.alpha),
                "round": int(round_index), "member": 0,
                "reuse_status": "new_fit", **audit, **_phase_budget(len(labeled))}


def _round0_evaluation(context: FusedSeedContext) -> tuple[Path, dict[str, object]]:
    paths = historical_round0(context.seed)
    prediction = pd.read_csv(paths["member0_predictions"])
    if prediction.sample_id.astype(str).tolist() != context.ids(context.roles["test"]):
        raise RuntimeError("round-zero prediction identity drift")
    return paths["member0_model"], {"outer_seed": context.seed, "method": context.spec.method,
        "alpha": float(context.spec.alpha), "round": 0, "member": 0,
        "reuse_status": "reused_historical_round0", "source_checkpoint": str(paths["member0_model"]),
        **_json(paths["member0_audit"]), **_phase_budget(INITIAL_ACTIVE_LABELS)}


def _cache_features(path: Path, contract: dict[str, object], features: np.ndarray, indices: np.ndarray) -> None:
    receipt = path.with_suffix(path.suffix + ".contract.json")
    if path.exists() or receipt.exists():
        if not path.exists() or not receipt.exists() or _json(receipt).get("contract") != contract:
            raise RuntimeError(f"incomplete or incompatible fusion feature cache: {path}")
        with np.load(path) as cached:
            if not np.array_equal(cached["canonical_indices"], indices) or not np.array_equal(cached["features"], features):
                raise RuntimeError(f"fusion feature cache content drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.asarray(features), canonical_indices=np.asarray(indices, dtype=int))
    atomic_json(receipt, {"contract": contract, "sha256": sha256_file(path)})


def _load_feature_cache(
    path: Path,
    expected_contract: dict[str, object],
    indices: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Load a feature cache while allowing its content hash to be self-describing.

    ``feature_sha256`` cannot be part of the pre-load expected contract because it
    is derived from the cached array.  All provenance inputs are checked first,
    then both the npz file and decoded array are checked against the receipt.
    """
    receipt_path = path.with_suffix(path.suffix + ".contract.json")
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise RuntimeError(f"incomplete fusion feature cache: {path}")
    receipt = _json(receipt_path)
    saved = receipt.get("contract")
    if not isinstance(saved, dict):
        raise RuntimeError(f"invalid fusion feature cache contract: {path}")
    without_content_hash = {key: value for key, value in saved.items() if key != "feature_sha256"}
    if without_content_hash != expected_contract:
        raise RuntimeError(f"incompatible fusion feature cache: {path}")
    if receipt.get("sha256") != sha256_file(path):
        raise RuntimeError(f"fusion feature cache file hash drift: {path}")
    with np.load(path) as cached:
        features = np.asarray(cached["features"], dtype=np.float32)
        stored = np.asarray(cached["canonical_indices"], dtype=int)
    if not np.array_equal(stored, indices) or features.shape != (len(indices), int(dimension)):
        raise RuntimeError(f"cached fusion feature order/shape drift: {path}")
    if saved.get("feature_sha256") != array_hash(features):
        raise RuntimeError(f"fusion feature cache array hash drift: {path}")
    return features, saved


def _load_or_extract(
    context: FusedSeedContext, round_index: int, model_path: Path, current: np.ndarray, labeled_count: int, directory: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    artifacts = directory / "acquisition_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    model = load_predictor_checkpoint(model_path)
    checkpoint_sha = sha256_file(model_path)
    checkpoint_state = state_dict_hash(model)
    ordered_ids = context.ids(current)
    base = {"study": context.spec.study.name, "method": context.spec.method,
            "fusion_alpha": float(context.spec.alpha), "outer_seed": context.seed, "round": round_index,
            "checkpoint_sha256": checkpoint_sha, "checkpoint_state_hash": checkpoint_state,
            "ordered_current_canonical_indices": current.tolist(), "ordered_current_sample_ids": ordered_ids,
            "ordered_current_ids_hash": stable_hash(ordered_ids), "labeled_ids_hash": ids_hash(context.ids(current[:labeled_count])),
            "unlabeled_ids_hash": ids_hash(context.ids(current[labeled_count:])), "sketch_seed": sketch_seed(context.seed),
            "gradient_dimension": 512, "latent_dimension": 128, "test_truth_access_count": 0}
    scales = tuple(float(context.preprocessing["target_scales"][key]) for key in ("V1", "V2"))
    gradient_path = artifacts / "current_gradient_features.npz"
    gradient_contract = {
        **base,
        "kind": "authoritative_q50_CountSketch",
        "endpoint_scales": context.preprocessing["target_scales"],
        "reuse_status": "reused_historical_round0" if round_index == 0 else "new_extraction",
    }
    cached_gradient = _load_feature_cache(gradient_path, gradient_contract, current, 512)
    if cached_gradient is not None:
        gradient, gradient_contract = cached_gradient
        gradient_audit = _json(artifacts / "gradient_audit.json")
    elif round_index == 0:
        old = historical_round0(context.seed)
        with np.load(old["gradient"]) as cached:
            gradient = np.asarray(cached["features"], dtype=np.float32)
            stored = np.asarray(cached["canonical_indices"], dtype=int)
        if not np.array_equal(stored, current) or gradient.shape != (len(current), 512):
            raise RuntimeError("historical round-zero gradient order/shape drift")
        gradient_audit = _json(old["gradient_audit"])
    else:
        result = extract_q50_gradient_sketches(model, context.atom, context.angle, current, scales, dimension=512, sketch_seed=sketch_seed(context.seed))
        gradient, gradient_audit = result.features, result.audit
    if cached_gradient is None:
        gradient_contract["feature_sha256"] = array_hash(gradient)
        _cache_features(gradient_path, gradient_contract, gradient, current)
        _write_json_once(artifacts / "gradient_audit.json", gradient_audit)

    latent_path = artifacts / "current_latent_features.npz"
    latent_contract = {**base, "kind": "current_QGeoGNN_V2_prehead_latent", "batch_size": 2048,
                       "reuse_status": "reused_frozen_a050_round0" if round_index == 0 else "new_extraction"}
    cached_latent = _load_feature_cache(latent_path, latent_contract, current, 128)
    if cached_latent is None:
        if round_index == 0:
            source = frozen_a050_round0_paths(context.seed)
            with np.load(source["latent"]) as cached:
                latent = np.asarray(cached["features"], dtype=np.float32)
                stored = np.asarray(cached["canonical_indices"], dtype=int)
            if not np.array_equal(stored, current) or latent.shape != (len(current), 128):
                raise RuntimeError("frozen alpha=.5 round-zero latent order/shape drift")
        else:
            latent = extract_representations(model, context.atom, context.angle, current).astype(np.float32)
        latent_contract["feature_sha256"] = array_hash(latent)
        _cache_features(latent_path, latent_contract, latent, current)
    else:
        latent, latent_contract = cached_latent

    fused = fuse_gradient_latent_features(gradient, latent, labeled_count, alpha=context.spec.alpha)
    fused_path = artifacts / "current_fused_features.npz"
    fused_contract = {**base, "kind": "gradient_latent_fusion", "alpha": float(context.spec.alpha),
                      "gradient_feature_sha256": gradient_contract["feature_sha256"], "latent_feature_sha256": latent_contract["feature_sha256"],
                      "gradient_block_scale": fused.gradient_scale, "latent_block_scale": fused.latent_scale,
                      "fused_dimension": FUSED_DIMENSION, "feature_sha256": fused.audit["fused_feature_sha256"]}
    _cache_features(fused_path, fused_contract, fused.features, current)
    audit = {**fused.audit, "gradient_status": gradient_contract["reuse_status"], "gradient_audit": gradient_audit,
             "checkpoint_sha256": checkpoint_sha, "checkpoint_state_hash": checkpoint_state,
             "current_ids_hash": stable_hash(ordered_ids), "round": round_index, "seed": context.seed}
    _write_json_once(artifacts / "fusion_audit.json", audit)
    return gradient, latent, fused.features, audit


def _historical_batch(seed: int, round_index: int) -> set[str]:
    path = DEFAULT_SPEC.study.parents[2] / "studies/active_learning/qgeognn_v2_row_maxdet_b32" / "runtime" / f"seed_{seed}" / "gradient_maxdet" / f"round_{round_index:02d}/acquisition/selected_next_batch.csv"
    return set(pd.read_csv(path).sample_id.astype(str)) if path.exists() else set()


def _historical_a050_batch(seed: int, round_index: int) -> set[str]:
    path = DEFAULT_SPEC.study / "runtime" / f"seed_{seed}" / DEFAULT_SPEC.method / f"round_{round_index:02d}/acquisition_artifacts/selected_batch.csv"
    return set(pd.read_csv(path).sample_id.astype(str)) if path.exists() else set()


def _acquire(context: FusedSeedContext, round_index: int, labeled: np.ndarray, unlabeled: np.ndarray, model_path: Path, directory: Path) -> tuple[list[str], dict[str, object]]:
    current = np.r_[labeled, unlabeled]
    input_contract = {"study": context.spec.study.name, "method": context.spec.method,
                      "alpha": float(context.spec.alpha), "outer_seed": context.seed, "source_round": round_index,
                      "L_t_ids_hash": ids_hash(context.ids(labeled)), "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
                      "checkpoint_sha256": sha256_file(model_path), "batch_size": BATCH_SIZE, "test_truth_access_count": 0}
    acquisition = directory / "acquisition_artifacts"
    selection_path = acquisition / "selected_batch.csv"
    contract_path = acquisition / "contract.json"
    if contract_path.exists():
        if not selection_path.exists():
            raise RuntimeError(f"partial fusion acquisition cache: {directory}")
        saved = _json(contract_path)
        if saved.get("input") != input_contract or sha256_file(selection_path) != saved.get("selected_table_sha256"):
            raise RuntimeError(f"fusion acquisition contract mismatch: {directory}")
        # Re-validate all three feature contracts before accepting a resumed
        # selection.  A changed checkpoint, ordering, alpha, or block scale
        # therefore fails loudly instead of silently trusting the batch.
        _load_or_extract(context, round_index, model_path, current, len(labeled), directory)
        return pd.read_csv(selection_path).sample_id.astype(str).tolist(), saved
    # A process may stop after writing one or more deterministic acquisition
    # artifacts but before the atomic contract write.  Recompute from validated
    # feature caches and let the write-once helpers prove those partial files are
    # identical before completing the contract.
    gradient, latent, fused, audit = _load_or_extract(context, round_index, model_path, current, len(labeled), directory)
    selection = conditional_gradient_maxdet(fused, np.arange(len(labeled)), np.arange(len(labeled), len(current)), BATCH_SIZE)
    selected_pool = selection.selected_positions - len(labeled)
    if np.any(selected_pool < 0) or np.any(selected_pool >= len(unlabeled)):
        raise RuntimeError("fusion MaxDet selected outside current U_t")
    selected_ids = np.asarray(context.ids(unlabeled))[selected_pool].astype(str).tolist()
    if len(set(selected_ids)) != BATCH_SIZE:
        raise RuntimeError("fusion MaxDet did not return 32 unique IDs")
    g_norm = np.linalg.norm(gradient[selected_pool + len(labeled)], axis=1)
    h_norm = np.linalg.norm(latent[selected_pool + len(labeled)], axis=1)
    f_norm = np.linalg.norm(fused[selected_pool + len(labeled)], axis=1)
    def nearest(block: np.ndarray) -> np.ndarray:
        q, l = block[len(labeled):][selected_pool].astype(float), block[:len(labeled)].astype(float)
        return np.sqrt(np.maximum(np.sum(q*q, axis=1)[:, None] + np.sum(l*l, axis=1)[None, :] - 2*q@l.T, 0)).min(axis=1)
    rows = pd.DataFrame({"outer_seed": context.seed, "method": context.spec.method,
        "alpha": float(context.spec.alpha), "source_round": round_index,
        "selection_order": np.arange(BATCH_SIZE), "pool_position": selected_pool, "canonical_index": unlabeled[selected_pool],
        "sample_id": selected_ids, "gradient_norm": g_norm, "latent_norm": h_norm, "fused_norm": f_norm,
        "nearest_L_gradient_distance": nearest(gradient), "nearest_L_latent_distance": nearest(latent), "nearest_L_fused_distance": nearest(fused),
        "marginal_logdet_gain": [item["marginal_logdet_gain"] for item in selection.trace]})
    # Norm diagnostics were originally computed from the in-memory extraction
    # arrays.  Reloading the frozen float32 cache can differ at float32 rounding
    # scale, while sample IDs, positions, and selection order remain exact.
    _write_csv_once(selection_path, rows, rtol=1e-7)
    _write_csv_once(acquisition / "selected_next_batch.csv", rows, rtol=1e-7)
    _write_csv_once(acquisition / "maxdet_trace.csv", pd.DataFrame(selection.trace))
    old = _historical_batch(context.seed, round_index)
    old_a050 = _historical_a050_batch(context.seed, round_index)
    gradient_diagnostics = representation_diagnostics(gradient)
    latent_diagnostics = representation_diagnostics(latent)
    fused_diagnostics = representation_diagnostics(fused)
    fused_a050 = fuse_gradient_latent_features(gradient, latent, len(labeled), alpha=0.5)
    fused_a050_diagnostics = representation_diagnostics(fused_a050.features)
    common_mode = latent_common_mode_diagnostics(latent, len(labeled))
    profile = {"outer_seed": context.seed, "method": context.spec.method,
        "alpha": float(context.spec.alpha), "source_round": round_index, "active_label_count": len(labeled),
        "gradient_block_scale": audit["gradient_block_scale"], "latent_block_scale": audit["latent_block_scale"],
        "gradient_feature_mean_norm": float(np.linalg.norm(gradient, axis=1).mean()), "latent_feature_mean_norm": float(np.linalg.norm(latent, axis=1).mean()),
        "fused_feature_mean_norm": float(np.linalg.norm(fused, axis=1).mean()),
        "gradient_effective_rank": gradient_diagnostics["effective_rank"], "latent_effective_rank": latent_diagnostics["effective_rank"],
        "fused_effective_rank": fused_diagnostics["effective_rank"], "fused_a050_counterfactual_effective_rank": fused_a050_diagnostics["effective_rank"],
        "gradient_pairwise_kernel_abs_corr": gradient_diagnostics["pairwise_kernel_abs_corr"],
        "latent_pairwise_kernel_abs_corr": latent_diagnostics["pairwise_kernel_abs_corr"],
        "fused_pairwise_kernel_abs_corr": fused_diagnostics["pairwise_kernel_abs_corr"],
        "fused_a050_counterfactual_pairwise_kernel_abs_corr": fused_a050_diagnostics["pairwise_kernel_abs_corr"],
        **common_mode,
        "selected_gradient_norm_mean": float(g_norm.mean()), "selected_latent_norm_mean": float(h_norm.mean()), "selected_fused_norm_mean": float(f_norm.mean()),
        "nearest_L_gradient_distance": float(nearest(gradient).mean()), "nearest_L_latent_distance": float(nearest(latent).mean()), "nearest_L_fused_distance": float(nearest(fused).mean()),
        "gradient_maxdet_overlap_count": len(set(selected_ids) & old), "gradient_maxdet_overlap_fraction": len(set(selected_ids) & old) / BATCH_SIZE,
        "a050_fusion_overlap_count": len(set(selected_ids) & old_a050), "a050_fusion_overlap_fraction": len(set(selected_ids) & old_a050) / BATCH_SIZE,
        "normalization_scale": selection.normalization_scale, "selector_seconds": selection.audit["elapsed_seconds"],
        "gradient_seconds": audit["gradient_audit"].get("elapsed_seconds", 0.0), "latent_seconds": 0.0, "fusion_seconds": 0.0}
    atomic_json(acquisition / "fusion_profile.json", profile)
    contract = {"input": input_contract, "status": "FROZEN_BEFORE_TEST_TRUTH", "selected_ids_ordered_hash": stable_hash(selected_ids),
        "selected_ids_set_hash": ids_hash(selected_ids), "selected_table_sha256": sha256_file(selection_path), "fusion_audit": audit,
        "selection_profile": profile, "test_truth_access_count": 0}
    atomic_json(contract_path, contract)
    return selected_ids, contract


def run_trajectory(context: FusedSeedContext) -> dict[str, object]:
    store = context.new_store()
    labeled, unlabeled, truth = context.roles["l0"].copy(), context.roles["u0"].copy(), context.l0_truth.copy()
    incoming: list[str] = []
    selected_all: list[str] = []
    fits: list[dict[str, object]] = []
    round_hashes: list[str] = []
    for round_index in range(context.spec.acquisition_rounds + 1):
        directory = context.runtime / context.spec.method / f"round_{round_index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        input_contract = {"study": context.spec.study.name, "protocol_hash": context.protocol_hash, "outer_seed": context.seed,
            "method": context.spec.method, "alpha": float(context.spec.alpha), "round": round_index,
            "active_label_count": len(labeled), "L_t_ids_hash": ids_hash(context.ids(labeled)),
            "U_t_ids_hash": ids_hash(context.ids(unlabeled)), "incoming_selected_ids_hash": stable_hash(incoming), "test_truth_access_count": 0}
        _write_json_once(directory / "input_contract.json", input_contract)
        _write_csv_once(directory / "state.csv", pd.DataFrame({"role": ["labeled"] * len(labeled) + ["unlabeled"] * len(unlabeled), "position": list(range(len(labeled))) + list(range(len(unlabeled))), "canonical_index": np.r_[labeled, unlabeled], "sample_id": context.ids(np.r_[labeled, unlabeled])}))
        if round_index == 0:
            model_path, evaluation = _round0_evaluation(context)
            prediction_path = historical_round0(context.seed)["member0_predictions"]
        else:
            evaluation = context.fit(round_index, labeled, truth, context.roles["test"], directory / "model")
            model_path, prediction_path = directory / "model/best.pt", directory / "model/predictions.csv.gz"
        fits.append(evaluation)
        outgoing: list[str] = []
        acquisition_contract = None
        if round_index < context.spec.acquisition_rounds:
            outgoing, acquisition_contract = _acquire(context, round_index, labeled, unlabeled, model_path, directory)
        round_contract = {"input": input_contract, "status": "FROZEN_BEFORE_TEST_TRUTH", "checkpoint_path": str(model_path),
            "checkpoint_sha256": sha256_file(model_path), "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
            "outgoing_selected_ids_hash": stable_hash(outgoing), "acquisition_contract_hash": None if acquisition_contract is None else stable_hash(acquisition_contract), "test_truth_access_count": 0}
        _write_json_once(directory / "round_freeze.json", round_contract)
        round_hashes.append(sha256_file(directory / "round_freeze.json"))
        if round_index == context.spec.acquisition_rounds:
            break
        old_l, old_u = context.ids(labeled), context.ids(unlabeled)
        by_id = {value: i for i, value in enumerate(old_u)}
        selected_indices = unlabeled[np.asarray([by_id[value] for value in outgoing], dtype=int)]
        next_unlabeled = np.asarray([index for index, value in zip(unlabeled, old_u) if value not in set(outgoing)], dtype=int)
        next_labeled = np.r_[labeled, selected_indices]
        validate_trajectory_transition(old_l, old_u, outgoing, context.ids(next_labeled), context.ids(next_unlabeled))
        store.freeze_acquisitions(selected_all + outgoing)
        truth = np.vstack([truth, store.reveal(outgoing, "after_acquisition_fit")])
        selected_all.extend(outgoing)
        labeled, unlabeled, incoming = next_labeled, next_unlabeled, outgoing
    if len(labeled) != context.spec.final_active_labels or len(selected_all) != context.spec.acquisition_rounds * BATCH_SIZE:
        raise RuntimeError(f"fusion trajectory did not reach {context.spec.final_active_labels} labels")
    _write_csv_once(context.runtime / context.spec.method / "fit_audit.csv", pd.DataFrame(fits))
    _write_csv_once(context.runtime / context.spec.method / "label_access_audit.csv", pd.DataFrame(store.audit))
    trajectory = {"status": "FROZEN_BEFORE_TEST_TRUTH", "outer_seed": context.seed, "method": context.spec.method,
        "alpha": float(context.spec.alpha), "round_points": context.spec.acquisition_rounds + 1,
        "acquisition_rounds": context.spec.acquisition_rounds, "final_active_labels": len(labeled),
        "selected_ids_ordered_hash": stable_hash(selected_all), "final_L_t_ids_hash": ids_hash(context.ids(labeled)),
        "final_U_t_ids_hash": ids_hash(context.ids(unlabeled)), "round_freeze_sha256": round_hashes,
        "fit_audit_sha256": sha256_file(context.runtime / context.spec.method / "fit_audit.csv"), "test_truth_access_count": 0}
    _write_json_once(context.runtime / context.spec.method / "trajectory_freeze.json", trajectory)
    return trajectory


def execute_seed(seed: int, study: Path | None = None, *, spec: FusedMaxDetStudySpec = DEFAULT_SPEC) -> dict[str, object]:
    study = spec.study if study is None else Path(study)
    validation = validate_pre_test_manifest(study, spec=spec)
    if int(seed) not in spec.seeds:
        raise ValueError("seed is outside the fusion confirmation cohort")
    freeze = _json(study / "global_pre_test_freeze.json")
    if freeze.get("status") != "PENDING_PRE_TEST_EXECUTION":
        raise RuntimeError("fusion execution is closed after the global barrier")
    context = FusedSeedContext(int(seed), pd.read_csv(study / "splits" / f"row_seed_{seed}.csv"), study / "runtime" / f"seed_{seed}", validation["protocol_hash"], spec)
    trajectory_path = context.runtime / spec.method / "trajectory_freeze.json"
    frozen = _json(trajectory_path) if trajectory_path.exists() else run_trajectory(context)
    print(json.dumps({"seed": int(seed), "method_frozen": spec.method, "alpha": float(spec.alpha), "test_truth_access_count": 0}), flush=True)
    return {"seed": int(seed), "status": "SEED_FROZEN_BEFORE_TEST_TRUTH", "trajectory": frozen}


def build_global_pre_test_freeze(study: Path | None = None, *, spec: FusedMaxDetStudySpec = DEFAULT_SPEC) -> dict[str, object]:
    study = spec.study if study is None else Path(study)
    validate_pre_test_manifest(study, spec=spec)
    entries = {}
    for seed in spec.seeds:
        root = study / "runtime" / f"seed_{seed}" / spec.method
        trajectory = _json(root / "trajectory_freeze.json")
        if trajectory.get("status") != "FROZEN_BEFORE_TEST_TRUTH":
            raise RuntimeError("fusion trajectory is not frozen")
        for round_index in range(spec.acquisition_rounds + 1):
            path = root / f"round_{round_index:02d}/round_freeze.json"
            record = _json(path)
            if record.get("test_truth_access_count") != 0 or sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
                raise RuntimeError("fusion round escaped or changed before test barrier")
            entries[f"seed_{seed}/round_{round_index:02d}"] = {"round_freeze_sha256": sha256_file(path), "checkpoint_sha256": record["checkpoint_sha256"], "prediction_sha256": record["prediction_sha256"]}
    freeze = {"status": "FROZEN_BEFORE_TEST_TRUTH", "alpha": float(spec.alpha), "entries": entries,
              "frozen_prediction_points": len(entries), "test_truth_access_count": 0}
    atomic_json(study / "global_pre_test_freeze.json", freeze)
    return freeze


__all__ = ["execute_seed", "build_global_pre_test_freeze", "run_trajectory", "FusedSeedContext"]

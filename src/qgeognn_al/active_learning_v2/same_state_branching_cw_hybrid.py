"""Same-state, two-batch acquisition branching with a global test barrier."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, load_features, seed_config, sketch_seed
from .benchmark_reporting import metric_row
from .cache import array_hash, seal_cache, verify_cache
from .gradient_features import extract_q50_gradient_sketches, extract_linear_output_gradient_sketches
from .gradient_transforms import center_width_transform
from .ivr import conditional_batch_ivr
from .lcmd import lcmd_tp_select
from .maxdet import conditional_gradient_maxdet
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import fit_from_same_initialization
from .sequential_acquisition import validate_trajectory_transition
from .sequential_protocol import ACTIVE_LABEL_BUDGETS, STUDY as LCMD_STUDY
from .sequential_runner import _assert_protected, _partition, _verify_global_freeze, _verify_round_artifacts
from . import ivr_study
from . import lcmd_to_ivr_study


STUDY = ROOT / "studies/active_learning/qgeognn_v2_same_state_branching_cw_hybrid"
ENTRY = ROOT / "scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py"
IVR_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
CW_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_short_sequential_b32"
CW_EXTENSION_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_cw_lcmd_to_653"
HYBRID_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_sequential_b32"
SEEDS = (157, 6101)
SOURCES = ("center_width_lcmd", "hybrid")
ANCHOR_BUDGETS = (429, 653)
STRATEGIES = ("center_width_lcmd", "kernel_ivr", "gradient_maxdet")
BATCH_SIZE = 32
SKETCH_DIMENSION = 512


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json_once(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"refusing to replace mismatched JSON: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(Path(path).relative_to(ROOT)): sha256_file(path) for path in sorted(set(map(Path, paths)))}


def assert_hashes(values: dict[str, str]) -> None:
    _assert_protected(ROOT, values)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another worker owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _round_for_budget(budget: int) -> int:
    if int(budget) not in ACTIVE_LABEL_BUDGETS:
        raise ValueError(f"budget absent from frozen schedule: {budget}")
    return list(ACTIVE_LABEL_BUDGETS).index(int(budget))


def _id_lists(partition: pd.DataFrame, labeled: Sequence[int], unlabeled: Sequence[int]) -> tuple[list[str], list[str]]:
    ids = partition.sample_id.astype(str).tolist()
    return [ids[int(i)] for i in labeled], [ids[int(i)] for i in unlabeled]


def _source_paths(source: str, seed: int, budget: int) -> tuple[Path, Path]:
    """Resolve the frozen historical state for the requested trajectory/budget."""
    if source == "center_width_lcmd":
        if budget == 429:
            return CW_STUDY / f"runtime/seed_{seed}/center_width_lcmd/round_03", CW_STUDY
        return CW_EXTENSION_STUDY / f"runtime/seed_{seed}/center_width_lcmd/round_10", CW_EXTENSION_STUDY
    if source == "hybrid":
        return HYBRID_STUDY / f"runtime/seed_{seed}/hybrid/round_{_round_for_budget(budget):02d}", HYBRID_STUDY
    raise ValueError(f"unknown source trajectory: {source}")


def _source_validation_history(source: str, seed: int, budget: int) -> list[dict[str, float | int]]:
    if source == "center_width_lcmd":
        study = CW_STUDY if budget == 429 else CW_EXTENSION_STUDY
        method = "center_width_lcmd"
        frame = pd.read_csv(study / f"runtime/seed_{seed}/{method}/fit_audit.csv")
        if budget == 429:
            frame = frame.loc[frame["round"].between(0, 3)]
        else:
            frame = frame.loc[frame["round"].between(7, 10)]
    else:
        frame = pd.read_csv(HYBRID_STUDY / f"runtime/seed_{seed}/hybrid/fit_audit.csv")
        frame = frame.loc[(frame["member"] == 0) & (frame["round"].between(_round_for_budget(budget)-3, _round_for_budget(budget)))]
    frame = frame.sort_values("round")
    return [{"round": int(row["round"]), "active_label_count": int(row["train_rows"]),
             "validation_combined_nrmse": float(row["best_validation_combined_normalized_rmse"])}
            for _, row in frame.iterrows()]



def _materialize_anchor_gradient(seed: int, source: str, budget: int, model_dir: Path,
                                 outer: list[int], preprocessing: dict, *, center_width: bool = False) -> tuple[Path, str, str]:
    path = STUDY / "runtime" / "anchor_features" / f"seed_{seed}_{source}_{budget}.npz"
    if center_width:
        path = path.with_name(path.stem + "_cw.npz")
    audit_path = path.with_name(path.stem + "_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if center_width:
        partition = _partition(seed)
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        l0 = partition.loc[partition.role.eq("l0"), "canonical_index"].astype(int).tolist()
        l0_truth = store.reveal(_id_lists(partition, l0, [])[0], "initial_fit")
        transform, transform_audit = center_width_transform(l0_truth)
        cw_context = read_json(CW_STUDY / f"runtime/seed_{seed}/context.json")
        if transform_audit != cw_context["center_width_transform"]:
            raise RuntimeError("anchor CW transform differs from the frozen L0-only historical transform")
    cache_contract = {"checkpoint_sha256": sha256_file(model_dir / "best.pt"),
                      "ordered_indices": outer, "dimension": SKETCH_DIMENSION,
                      "sketch_seed": sketch_seed(seed), "representation": "L0_center_width" if center_width else "ordinary_q50",
                      "endpoint_scales": preprocessing["target_scales"], "test_truth_access_count": 0}
    if center_width:
        cache_contract["transform_audit"] = transform_audit
    if verify_cache(path, cache_contract) and audit_path.exists():
        with np.load(path) as values:
            if not np.array_equal(values["canonical_indices"], outer):
                raise RuntimeError("anchor cache order mismatch")
            return path, str(path.relative_to(ROOT)), array_hash(values["features"])
    atom, angle = torch.load(HYBRID_STUDY / f"runtime/seed_{seed}/scrubbed_graphs.pt", weights_only=False)
    if any(torch.count_nonzero(item.y) for item in atom):
        raise RuntimeError("anchor graph labels were not scrubbed")
    model = load_predictor_checkpoint(model_dir / "best.pt")
    if center_width:
        result = extract_linear_output_gradient_sketches(
            model, atom, angle, outer, transform, dimension=SKETCH_DIMENSION, sketch_seed=sketch_seed(seed))
    else:
        result = extract_q50_gradient_sketches(
            model, atom, angle, outer,
            tuple(preprocessing["target_scales"][key] for key in ("V1", "V2")),
            dimension=SKETCH_DIMENSION, sketch_seed=sketch_seed(seed))
    np.savez_compressed(path, features=result.features, canonical_indices=np.asarray(outer, dtype=int))
    atomic_json(audit_path, {**result.audit, "label_access": store.audit if center_width else []})
    seal_cache(path, cache_contract)
    return path, str(path.relative_to(ROOT)), array_hash(result.features)


def inspect_anchor(seed: int, source: str, budget: int) -> dict:
    """Verify and describe a historical anchor without fitting or reading test truth."""
    if seed not in SEEDS or source not in SOURCES or budget not in ANCHOR_BUDGETS:
        raise ValueError("anchor outside the frozen factorial design")
    directory, source_study = _source_paths(source, seed, budget)
    if not directory.exists():
        raise RuntimeError(f"missing historical source state: {directory}")
    freeze = read_json(directory / ("round_freeze.json" if (directory / "round_freeze.json").exists() else "contract.json"))
    if freeze.get("test_truth_access_count", 0) != 0:
        raise RuntimeError("historical source has nonzero test access")
    state = pd.read_csv(directory / "state.csv") if (directory / "state.csv").exists() else None
    if state is None:
        labeled = np.load(directory / "labeled_indices.npy").astype(int).tolist()
        unlabeled = np.load(directory / "unlabeled_indices.npy").astype(int).tolist()
        partition = _partition(seed)
        labeled_ids, unlabeled_ids = _id_lists(partition, labeled, unlabeled)
    else:
        labeled = state.loc[state.role.eq("labeled"), "canonical_index"].astype(int).tolist()
        unlabeled = state.loc[state.role.eq("unlabeled"), "canonical_index"].astype(int).tolist()
        labeled_ids = state.loc[state.role.eq("labeled"), "sample_id"].astype(str).tolist()
        unlabeled_ids = state.loc[state.role.eq("unlabeled"), "sample_id"].astype(str).tolist()
    partition = _partition(seed)
    validate_row_protocol(partition)
    l0 = partition.loc[partition.role.eq("l0"), "canonical_index"].astype(int).tolist()
    u0 = partition.loc[partition.role.eq("u0"), "canonical_index"].astype(int).tolist()
    outer = l0 + u0
    if len(labeled) != budget or len(unlabeled) != len(outer) - budget:
        raise RuntimeError("historical source state budget mismatch")
    model_dir = directory / "model"
    fit = read_json(model_dir / "fit_audit.json")
    checkpoint = model_dir / "best.pt"
    prediction = model_dir / "predictions.csv.gz"
    if (freeze.get("checkpoint_sha256") != sha256_file(checkpoint)
            or fit["checkpoint_sha256"] != sha256_file(checkpoint)
            or freeze.get("checkpoint_state_hash") != fit["checkpoint_state_hash"]):
        raise RuntimeError("historical anchor checkpoint differs from its frozen receipt")
    source_input = freeze["input"]
    if (source_input["L_t_ids_hash"] != ids_hash(labeled_ids)
            or source_input["U_t_ids_hash"] != ids_hash(unlabeled_ids)):
        raise RuntimeError("historical anchor label state differs from its frozen receipt")
    if (len(set(labeled + unlabeled)) != len(outer) or set(labeled + unlabeled) != set(outer)
            or labeled_ids != _id_lists(partition, labeled, unlabeled)[0]
            or unlabeled_ids != _id_lists(partition, labeled, unlabeled)[1]):
        raise RuntimeError("anchor must be an exact ordered partition of outer training rows")
    if fit.get("train_rows") != budget or fit.get("test_labels_used_for_fit_or_checkpoint_selection", 0) != 0:
        raise RuntimeError("historical source checkpoint contract mismatch")
    context_path = (CW_STUDY if source == "center_width_lcmd" else HYBRID_STUDY) / f"runtime/seed_{seed}/context.json"
    historical_record = read_json(context_path)
    historical = historical_record.get("contract", historical_record)
    gradient_path, gradient_rel, gradient_hash = _materialize_anchor_gradient(
        seed, source, budget, model_dir, outer, historical["preprocessing"]
    )
    cw_gradient_path, cw_gradient_rel, cw_gradient_hash = _materialize_anchor_gradient(
        seed, source, budget, model_dir, outer, historical["preprocessing"], center_width=True
    )
    files = [directory / "input_contract.json", directory / "model/best.pt", directory / "model/fit_audit.json",
             directory / "model/predictions.csv.gz", directory / "state.csv", context_path, gradient_path,
             gradient_path.with_suffix(".npz.contract.json"), cw_gradient_path,
             cw_gradient_path.with_suffix(".npz.contract.json")]
    files += [directory / name for name in ("labeled_indices.npy", "unlabeled_indices.npy", "contract.json", "round_freeze.json")]
    files += [gradient_path.with_name(gradient_path.stem + "_audit.json"),
              cw_gradient_path.with_name(cw_gradient_path.stem + "_audit.json"),
              CW_STUDY / f"runtime/seed_{seed}/context.json",
              HYBRID_STUDY / f"runtime/seed_{seed}/scrubbed_graphs.pt"]
    files = [item for item in files if item.exists()]
    history = _source_validation_history(source, seed, budget)
    return {
        "seed": seed, "source_trajectory": source, "anchor_budget": budget,
        "source_round": _round_for_budget(budget), "source_study": source_study.name,
        "labeled_indices": labeled, "unlabeled_indices": unlabeled, "outer_indices": outer,
        "labeled_ids": labeled_ids, "unlabeled_ids": unlabeled_ids,
        "ordered_L_hash": stable_hash(labeled_ids), "ordered_U_hash": stable_hash(unlabeled_ids),
        "L_set_hash": ids_hash(labeled_ids), "U_set_hash": ids_hash(unlabeled_ids),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_state_hash": fit["checkpoint_state_hash"],
        "anchor_prediction_path": str(prediction.relative_to(ROOT)),
        "gradient_path": gradient_rel,
        "cw_gradient_path": cw_gradient_rel,
        "gradient_contract_path": str(gradient_path.with_suffix(".npz.contract.json").relative_to(ROOT)),
        "gradient_audit_path": str(gradient_path.with_name(gradient_path.stem + "_audit.json").relative_to(ROOT)),
        "gradient_file_sha256": sha256_file(gradient_path), "gradient_bank_hash": gradient_hash,
        "cw_gradient_file_sha256": sha256_file(cw_gradient_path), "cw_gradient_bank_hash": cw_gradient_hash,
        "gradient_order_kind": "outer_L0_then_U0", "gradient_order_hash": stable_hash(outer),
        "recent_validation": history, "source_files": hashes(files), "test_truth_access_count": 0,
    }


@dataclass
class BranchContext:
    seed: int

    def __post_init__(self) -> None:
        torch.set_num_threads(2)
        self.partition = _partition(self.seed)
        validate_row_protocol(self.partition)
        self.data = load_features(SOURCE_DATA)
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError("branch split/source identity mismatch")
        baseline = HYBRID_STUDY / f"runtime/seed_{self.seed}"
        historical = read_json(baseline / "context.json")["contract"]
        self.preprocessing = historical["preprocessing"]
        self.normalization = ConditionNormalization(**historical["normalization"])
        self.atom, self.angle = torch.load(baseline / "scrubbed_graphs.pt", weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("source graph cache is not label-scrubbed")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        self.outer = np.r_[self.roles["l0"], self.roles["u0"]]
        initial = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = initial.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.cw_transform, self.cw_transform_audit = center_width_transform(self.l0_truth)
        if self.cw_transform_audit != read_json(CW_STUDY / f"runtime/seed_{self.seed}/context.json")["center_width_transform"]:
            raise RuntimeError("branch Center/Width transform does not match the fixed L0 transform")
        self.validation_truth = initial.reveal(self.ids(self.roles["validation"]), "initial_fit")
        self.historical_context = historical

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(map(int, indices))].sample_id.astype(str).tolist()

    def training_config(self) -> dict:
        config = seed_config(self.seed, 0, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def restore_anchor(self, lineage: dict) -> tuple[RestrictedLabelStore, np.ndarray, list[str]]:
        if lineage["outer_indices"] != self.outer.tolist():
            raise RuntimeError("anchor/context outer order mismatch")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        l0 = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        validation = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        if not np.array_equal(l0, self.l0_truth) or not np.array_equal(validation, self.validation_truth):
            raise RuntimeError("branch stores do not share the same initial labels")
        acquired = lineage["labeled_ids"][len(self.roles["l0"]):]
        store.freeze_acquisitions(acquired)
        truth = np.vstack([self.l0_truth, store.reveal(acquired, "after_acquisition_fit")])
        return store, truth, list(acquired)


def load_anchor_bank(lineage: dict) -> np.ndarray:
    path = ROOT / lineage["gradient_path"]
    if sha256_file(path) != lineage["gradient_file_sha256"]:
        raise RuntimeError("anchor gradient file changed after lineage freeze")
    receipt = read_json(ROOT / lineage["gradient_contract_path"])
    if not verify_cache(path, receipt["contract"]):
        raise RuntimeError("anchor gradient cache no longer verifies")
    with np.load(path) as values:
        features = values["features"].copy()
        indices = values["canonical_indices"].astype(int)
    if array_hash(features) != lineage["gradient_bank_hash"]:
        raise RuntimeError("anchor gradient array hash changed")
    positions = {int(value): i for i, value in enumerate(indices)}
    if set(positions) != set(lineage["outer_indices"]):
        raise RuntimeError("anchor gradient universe changed")
    return features[[positions[int(i)] for i in lineage["outer_indices"]]]


def load_anchor_banks(lineage: dict) -> dict[str, np.ndarray]:
    ordinary = load_anchor_bank(lineage)
    path = ROOT / lineage["cw_gradient_path"]
    if sha256_file(path) != lineage["cw_gradient_file_sha256"]:
        raise RuntimeError("anchor CW gradient file changed after lineage freeze")
    receipt = read_json(path.with_suffix(".npz.contract.json"))
    if not verify_cache(path, receipt["contract"]):
        raise RuntimeError("anchor CW gradient cache no longer verifies")
    with np.load(path) as values:
        features = values["features"].copy()
        indices = values["canonical_indices"].astype(int)
    if array_hash(features) != lineage["cw_gradient_bank_hash"]:
        raise RuntimeError("anchor CW gradient hash changed")
    positions = {int(value): i for i, value in enumerate(indices)}
    if set(positions) != set(lineage["outer_indices"]):
        raise RuntimeError("anchor CW gradient universe changed")
    return {"ordinary_q50": ordinary, "center_width": features[[positions[int(i)] for i in lineage["outer_indices"]]]}


def _positions(lineage_or_state: dict) -> tuple[np.ndarray, np.ndarray]:
    position = {int(value): i for i, value in enumerate(lineage_or_state["outer_indices"])}
    return (np.asarray([position[int(i)] for i in lineage_or_state["labeled_indices"]], dtype=int),
            np.asarray([position[int(i)] for i in lineage_or_state["unlabeled_indices"]], dtype=int))


def feature_diagnostics(features: np.ndarray, labeled_positions: np.ndarray, candidate_positions: np.ndarray) -> dict:
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    gram = x.T @ x
    trace = float(np.trace(gram))
    denominator = float(np.square(gram).sum())
    effective_rank = trace * trace / denominator if denominator > 0 else 0.0
    nearest = np.full(len(candidate_positions), np.inf)
    centers = x[labeled_positions]
    center_sq = np.einsum("ij,ij->i", centers, centers)
    for start in range(0, len(candidate_positions), 256):
        rows = x[candidate_positions[start:start + 256]]
        distance_sq = np.maximum(
            np.einsum("ij,ij->i", rows, rows)[:, None] + center_sq[None, :] - 2.0 * rows @ centers.T,
            0.0,
        )
        nearest[start:start + len(rows)] = np.sqrt(distance_sq.min(axis=1))
    return {
        "gradient_norm_mean": float(norms.mean()), "gradient_norm_std": float(norms.std(ddof=0)),
        "gradient_norm_p50": float(np.quantile(norms, .50)), "gradient_norm_p90": float(np.quantile(norms, .90)),
        "gradient_norm_p95": float(np.quantile(norms, .95)), "gradient_norm_max": float(norms.max()),
        "gradient_effective_rank_participation_ratio": effective_rank,
        "coverage_nearest_distance_mean": float(nearest.mean()),
        "coverage_nearest_distance_median": float(np.median(nearest)),
        "coverage_nearest_distance_p90": float(np.quantile(nearest, .90)),
        "coverage_nearest_distance_max": float(nearest.max()),
        "active_label_count": int(len(labeled_positions)), "candidate_pool_size": int(len(candidate_positions)),
    }


def select_batch(strategy: str, features: dict[str, np.ndarray], state: dict) -> dict:
    """Label-free selectors on a shared state with method-specific representations."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if not isinstance(features, dict):
        raise TypeError("explicit method-specific gradient banks are required")
    banks = features
    method_features = banks["center_width"] if strategy == "center_width_lcmd" else banks["ordinary_q50"]
    labeled_positions, candidate_positions = _positions(state)
    pool = np.asarray(method_features[candidate_positions], dtype=np.float64)
    centers = np.asarray(method_features[labeled_positions], dtype=np.float64)
    if strategy == "center_width_lcmd":
        result = lcmd_tp_select(pool, centers, BATCH_SIZE)
        selected_local = result.selected_pool_positions
        trace = list(result.trace)
        objective = {
            "definition": "L0_fixed_CW_transform_full_network_linear_output_gradient_CountSketch_LCMD_TP",
            "largest_cluster_score_first": float(trace[0]["largest_cluster_score"]),
        }
    elif strategy == "kernel_ivr":
        result = conditional_batch_ivr(method_features, labeled_positions, candidate_positions, BATCH_SIZE)
        selected_local = np.asarray(result["selected_candidate_positions"], dtype=int)
        trace = result["trace"]
        before, after = float(trace[0]["integrated_variance_before"]), float(trace[-1]["integrated_variance_after"])
        objective = {"definition": "global_RMS_unit_prior_unit_noise_integrated_variance_reduction",
                     "raw_mean_squared_norm": result["audit"]["raw_mean_squared_norm"],
                     "integrated_variance_before": before, "integrated_variance_after": after,
                     "predicted_total_reduction": before - after,
                     "predicted_relative_reduction": (before - after) / before}
    else:
        result = conditional_gradient_maxdet(method_features, labeled_positions, candidate_positions, BATCH_SIZE)
        selected_local = result.selected_candidate_positions
        trace = list(result.trace)
        objective = {"definition": "historical_q50_gradient_conditional_D_optimal_current_L_conditioning",
                     "normalization_scale": result.normalization_scale,
                     "total_marginal_logdet_gain": float(sum(row["marginal_logdet_gain"] for row in result.trace))}
    selected_indices = np.asarray(state["unlabeled_indices"], dtype=int)[selected_local]
    selected_ids = np.asarray(state["unlabeled_ids"], dtype=str)[selected_local].tolist()
    if len(selected_ids) != BATCH_SIZE or len(set(selected_ids)) != BATCH_SIZE:
        raise RuntimeError("selector did not return exactly 32 unique current-U rows")
    return {"strategy": strategy, "selected_indices": selected_indices.tolist(), "selected_ids": selected_ids,
            "selected_positions_in_current_U": np.asarray(selected_local, dtype=int).tolist(),
            "objective": objective, "trace": trace, "test_truth_access_count": 0}


def _validation_slopes(history: list[dict]) -> dict[str, float]:
    y = np.asarray([row["validation_combined_nrmse"] for row in history], dtype=float)
    x = np.asarray([row["active_label_count"] for row in history], dtype=float)
    if len(y) < 4:
        raise RuntimeError("anchor lacks the preregistered validation history window")
    def slope(intervals: int) -> float:
        return float(np.polyfit(x[-(intervals + 1):], y[-(intervals + 1):], 1)[0])
    return {"recent_validation_nrmse": float(y[-1]), "recent_validation_last_step_delta": float(y[-1] - y[-2]),
            "recent_validation_2step_linear_slope_per_label": slope(2),
            "recent_validation_3step_linear_slope_per_label": slope(3)}


def _anchor_key(seed: int, source: str, budget: int) -> str:
    return f"seed_{seed}/{source}/budget_{budget}"


def _lineage_path(seed: int, source: str, budget: int) -> Path:
    return STUDY / f"lineage/seed_{seed}_{source}_budget_{budget}.json"


def _anchor_runtime(seed: int, source: str, budget: int) -> Path:
    return STUDY / "runtime" / f"seed_{seed}" / source / f"budget_{budget}"


def bank_diagnostics(banks: dict[str, np.ndarray], state: dict) -> dict:
    lpos, upos = _positions(state)
    result = {"active_label_count": len(lpos), "candidate_pool_size": len(upos)}
    for bank, prefix in (("ordinary_q50", "raw_gradient"), ("center_width", "cw_gradient")):
        for name, value in feature_diagnostics(banks[bank], lpos, upos).items():
            if name in {"active_label_count", "candidate_pool_size"}:
                continue
            result[f"{prefix}_{name.removeprefix('gradient_')}"] = value
    return result


def _anchor_selections(lineage: dict, *, persist: bool) -> tuple[dict[str, dict], dict]:
    banks = load_anchor_banks(lineage)
    diagnostics = {**bank_diagnostics(banks, lineage),
                   **_validation_slopes(lineage["recent_validation"]),
                   "seed": lineage["seed"], "source_trajectory": lineage["source_trajectory"],
                   "anchor_budget": lineage["anchor_budget"], "state_kind": "anchor",
                   "raw_gradient_bank_hash": lineage["gradient_bank_hash"],
                   "cw_gradient_bank_hash": lineage["cw_gradient_bank_hash"]}
    selections = {strategy: select_batch(strategy, banks, lineage) for strategy in STRATEGIES}
    # Store all method objectives on the anchor state, before any branch label is revealed.
    diagnostics.update({
        "ivr_integrated_variance_before": selections["kernel_ivr"]["objective"]["integrated_variance_before"],
        "ivr_predicted_total_reduction_B32": selections["kernel_ivr"]["objective"]["predicted_total_reduction"],
        "ivr_predicted_relative_reduction_B32": selections["kernel_ivr"]["objective"]["predicted_relative_reduction"],
        "maxdet_total_marginal_logdet_gain_B32": selections["gradient_maxdet"]["objective"]["total_marginal_logdet_gain"],
        "cw_lcmd_largest_cluster_score_first": selections["center_width_lcmd"]["objective"]["largest_cluster_score_first"],
    })
    if persist:
        runtime = _anchor_runtime(lineage["seed"], lineage["source_trajectory"], lineage["anchor_budget"])
        write_json_once(runtime / "anchor_state_diagnostics.json", diagnostics)
        input_contract = {"lineage_sha256": sha256_file(_lineage_path(lineage["seed"], lineage["source_trajectory"], lineage["anchor_budget"])),
                          "ordered_L_hash": lineage["ordered_L_hash"], "ordered_U_hash": lineage["ordered_U_hash"],
                          "checkpoint_sha256": lineage["checkpoint_sha256"], "checkpoint_state_hash": lineage["checkpoint_state_hash"],
                          "gradient_bank_hash": lineage["gradient_bank_hash"],
                          "cw_gradient_bank_hash": lineage["cw_gradient_bank_hash"], "test_truth_access_count": 0}
        for strategy, selection in selections.items():
            write_json_once(runtime / "anchor_selections" / f"{strategy}.json", {"input": input_contract, **selection})
    return selections, diagnostics


def _overlap_rows(lineage: dict, selections: dict[str, dict]) -> list[dict]:
    rows = []
    for left, right in itertools.combinations(STRATEGIES, 2):
        a, b = set(selections[left]["selected_ids"]), set(selections[right]["selected_ids"])
        intersection = len(a & b)
        rows.append({"seed": lineage["seed"], "source_trajectory": lineage["source_trajectory"],
                     "anchor_budget": lineage["anchor_budget"], "left_strategy": left, "right_strategy": right,
                     "intersection_count": intersection, "intersection_fraction_of_32": intersection / BATCH_SIZE,
                     "jaccard": intersection / len(a | b)})
    return rows


def prepare(test_report: Path) -> dict:
    STUDY.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(STUDY / "runtime/prepare.lock"):
        if (STUDY / "seal.json").exists():
            return validate_seal()
        if list((STUDY / "runtime").glob("seed_*")):
            raise RuntimeError("cannot seal after branching execution has started")
        root = ET.parse(test_report).getroot()
        cases = list(root.iter("testcase"))
        if sum("test_same_state_branching_cw_hybrid" in case.get("classname", "") for case in cases) < 12 or any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")):
            raise RuntimeError("at least 12 passing non-skipped preflight tests are required")
        audit = read_json(STUDY / "selector_audit.json")
        if (audit.get("status") != "PASSED_EXACT_ORDERED_SELECTOR_AUDIT"
                or audit.get("strategies") != list(STRATEGIES)
                or audit.get("test_truth_access_count") != 0 or audit.get("new_fits") != 0
                or audit.get("cw_exact_states") != 4 or audit.get("hybrid_exact_source_states") != 4
                or audit.get("ivr_exact_states", 0) < 1 or audit.get("maxdet_exact_states", 0) < 1):
            raise RuntimeError("complete exact selector regression evidence is required before sealing")
        assert_hashes(audit["files"])
        assert_hashes(audit["source_files"])
        lineages, source_files = [], dict(audit["source_files"])
        for seed in SEEDS:
            for source in SOURCES:
                for budget in ANCHOR_BUDGETS:
                    lineage = inspect_anchor(seed, source, budget)
                    write_json_once(_lineage_path(seed, source, budget), lineage)
                    lineages.append(lineage)
                    source_files.update(lineage["source_files"])
        table = pd.DataFrame([{k: v for k, v in item.items() if k not in {
            "labeled_indices", "unlabeled_indices", "outer_indices", "labeled_ids", "unlabeled_ids",
            "recent_validation", "source_files"}} for item in lineages])
        table.to_csv(STUDY / "anchor_lineage_audit.csv", index=False)
        test_copy = STUDY / "preflight_tests.xml"
        test_copy.write_bytes(Path(test_report).read_bytes())
        code_paths = [ENTRY, Path(__file__), ROOT / "src/qgeognn_al/active_learning_v2/lcmd.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/ivr.py", ROOT / "src/qgeognn_al/active_learning_v2/maxdet.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/gradient_features.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/runner.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/protocol.py",
                      ROOT / "tests/active_learning_v2/test_same_state_branching_cw_hybrid.py", STUDY / "PROTOCOL.md",
                      STUDY / "config.json", STUDY / "selector_audit.json", STUDY / "IMPLEMENTATION_AUDIT.md",
                      STUDY / "superseded_seal.json", STUDY / "results/planned_compute_cost.json",
                      STUDY / "README.md",
                      ROOT / "src/qgeognn_al/active_learning_v2/same_state_selector_audit.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/gradient_transforms.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/coverage.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/sequential_acquisition.py",
                      ROOT / "src/qgeognn_al/active_learning_v2/uncertainty.py",
                      ROOT / "scripts/studies/finalize_qgeognn_v2_same_state_branching_cw_hybrid_report.py",
                      STUDY / "anchor_lineage_audit.csv", test_copy, *sorted((STUDY / "lineage").glob("*.json"))]
        seal = {"status": "READY_FOR_EXECUTION_AFTER_EXACT_SELECTOR_AUDIT", "created_at": now(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "seeds": list(SEEDS), "sources": list(SOURCES), "anchor_budgets": list(ANCHOR_BUDGETS),
                "strategies": list(STRATEGIES), "batch_size": BATCH_SIZE, "rollout_rounds": 2,
                "planned_new_fits": len(SEEDS) * len(SOURCES) * len(ANCHOR_BUDGETS) * len(STRATEGIES) * 2, "test_truth_access_count": 0,
                "evidence": "development_mechanism_not_independent_confirmation",
                "files": {**audit["files"], **hashes(code_paths)}, "source_files": source_files}
        write_json_once(STUDY / "seal.json", seal)
        selection_smoke()
        atomic_json(STUDY / "decision.json", {
            "status": seal["status"], "evidence": seal["evidence"],
            "planned_new_fits": seal["planned_new_fits"], "actual_new_fits": 0,
            "branch_count": 24, "strategies": list(STRATEGIES),
            "test_truth_access_count": 0, "controller_trained": False,
            "mean_oracle_gain_over_CW": None, "recommendation": "Selector audit complete; execution has not started"})
        return seal


def validate_seal() -> dict:
    seal = read_json(STUDY / "seal.json")
    expected_fits = len(SEEDS) * len(SOURCES) * len(ANCHOR_BUDGETS) * len(STRATEGIES) * 2
    if seal.get("status") != "READY_FOR_EXECUTION_AFTER_EXACT_SELECTOR_AUDIT" or seal.get("planned_new_fits") != expected_fits:
        raise RuntimeError("invalid same-state study seal")
    if (seal.get("strategies") != list(STRATEGIES) or seal.get("seeds") != list(SEEDS)
            or seal.get("sources") != list(SOURCES) or seal.get("anchor_budgets") != list(ANCHOR_BUDGETS)):
        raise RuntimeError("sealed factorial design differs from the implementation")
    assert_hashes(seal["files"])
    assert_hashes(seal["source_files"])
    if sha256_file(SOURCE_DATA) != read_json(CW_STUDY / "protocol.json")["source_sha256"]:
        raise RuntimeError("canonical source data changed")
    return seal


def selection_smoke() -> dict:
    validate_seal()
    checks, overlaps = [], []
    for seed in SEEDS:
        for source in SOURCES:
            for budget in ANCHOR_BUDGETS:
                lineage = read_json(_lineage_path(seed, source, budget))
                first, diagnostics = _anchor_selections(lineage, persist=False)
                second, _ = _anchor_selections(lineage, persist=False)
                if first != second:
                    raise RuntimeError("selection-only repeat is not deterministic")
                overlaps.extend(_overlap_rows(lineage, first))
                checks.append({"seed": seed, "source_trajectory": source, "anchor_budget": budget,
                               "source_round": lineage["source_round"], "status": "PASSED",
                               "ordered_L_hash": lineage["ordered_L_hash"], "ordered_U_hash": lineage["ordered_U_hash"],
                               "checkpoint_sha256": lineage["checkpoint_sha256"],
                               "gradient_bank_hash": lineage["gradient_bank_hash"],
                               "effective_rank": diagnostics["raw_gradient_effective_rank_participation_ratio"],
                               "new_fits": 0, "new_label_reveals": 0, "test_truth_access_count": 0})
    result = {"status": "PASSED", "checks": checks, "overlap": overlaps, "new_fits": 0,
              "test_truth_access_count": 0, "seal_sha256": sha256_file(STUDY / "seal.json")}
    atomic_json(STUDY / "engineering_smoke.json", result)
    return result


def cw_selection_regression_audit() -> pd.DataFrame:
    from .same_state_selector_audit import cw_regression
    lineages = [read_json(_lineage_path(seed, "center_width_lcmd", budget))
                for seed in SEEDS for budget in ANCHOR_BUDGETS]
    return cw_regression(lineages, [])


def _extract_branch_bank(context: BranchContext, checkpoint: Path, directory: Path, state: dict) -> tuple[dict[str, np.ndarray], dict]:
    path = directory / "postfit_gradient_features.npz"
    audit_path = directory / "postfit_gradient_audit.json"
    contract = {"checkpoint_sha256": sha256_file(checkpoint), "checkpoint_state_hash": read_json(checkpoint.parent / "fit_audit.json")["checkpoint_state_hash"],
                "ordered_indices": context.outer.tolist(), "dimension": SKETCH_DIMENSION,
                "sketch_seed": sketch_seed(context.seed), "scales": context.preprocessing["target_scales"],
                "ordered_L_hash": stable_hash(state["labeled_ids"]), "ordered_U_hash": stable_hash(state["unlabeled_ids"]),
                "test_truth_access_count": 0}
    if not verify_cache(path, contract):
        result = extract_q50_gradient_sketches(
            load_predictor_checkpoint(checkpoint), context.atom, context.angle, context.outer,
            tuple(context.preprocessing["target_scales"][key] for key in ("V1", "V2")),
            dimension=SKETCH_DIMENSION, sketch_seed=sketch_seed(context.seed),
        )
        np.savez_compressed(path, features=result.features, canonical_indices=context.outer)
        atomic_json(audit_path, result.audit)
        seal_cache(path, contract)
    with np.load(path) as values:
        if not np.array_equal(values["canonical_indices"], context.outer):
            raise RuntimeError("branch gradient canonical order changed")
        features = values["features"].copy()
    cw_path = directory / "postfit_cw_gradient_features.npz"
    cw_audit_path = directory / "postfit_cw_gradient_audit.json"
    cw_contract = {**contract, "representation": "L0_fixed_center_width",
                   "transform_audit": context.cw_transform_audit}
    if not verify_cache(cw_path, cw_contract):
        cw_result = extract_linear_output_gradient_sketches(
            load_predictor_checkpoint(checkpoint), context.atom, context.angle, context.outer,
            context.cw_transform, dimension=SKETCH_DIMENSION, sketch_seed=sketch_seed(context.seed),
        )
        np.savez_compressed(cw_path, features=cw_result.features, canonical_indices=context.outer)
        atomic_json(cw_audit_path, {**cw_result.audit, "transform_audit": context.cw_transform_audit})
        seal_cache(cw_path, cw_contract)
    with np.load(cw_path) as values:
        if not np.array_equal(values["canonical_indices"], context.outer):
            raise RuntimeError("branch CW gradient canonical order changed")
        cw_features = values["features"].copy()
    return {"ordinary_q50": features, "center_width": cw_features}, {"ordinary_q50": read_json(audit_path), "center_width": read_json(cw_audit_path)}


def _fit(context: BranchContext, lineage: dict, strategy: str, step: int, labeled: list[int], truth: np.ndarray,
         directory: Path) -> dict:
    model_dir = directory / "model"
    expected = [model_dir / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
    if any(path.exists() for path in expected) and not all(path.exists() for path in expected):
        destination = directory.parent / "incomplete_attempts" / f"step_{step:02d}_{time.time_ns()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        model_dir.rename(destination)
    train_ids = context.ids(labeled)
    contract = {"study": STUDY.name, "seal_sha256": sha256_file(STUDY / "seal.json"),
                "seed": context.seed, "source_trajectory": lineage["source_trajectory"],
                "anchor_budget": lineage["anchor_budget"], "strategy": strategy, "rollout_step": step,
                "train_sample_ids": train_ids, "validation_sample_ids": context.ids(context.roles["validation"]),
                "ordered_L_hash": stable_hash(train_ids), "prediction_role": "test_X_without_test_truth",
                "test_truth_access_count": 0}
    return fit_from_same_initialization(
        atom_base=context.atom, angle=context.angle, normalization=context.normalization,
        preprocessing=context.preprocessing, train_indices=labeled, train_truth=truth,
        validation_indices=context.roles["validation"], validation_truth=context.validation_truth,
        prediction_indices=context.roles["test"], prediction_sample_ids=context.ids(context.roles["test"]),
        initialization_seed=initialization_seed(context.seed, 0), training_config=context.training_config(),
        contract=contract, runtime=model_dir,
    )


def _best_validation_row(model_dir: Path, audit: dict) -> dict:
    history = pd.read_csv(model_dir / "history.csv")
    row = history.loc[history.epoch.eq(audit["best_epoch"])]
    if len(row) != 1:
        raise RuntimeError("best validation epoch is absent or duplicated")
    row = row.iloc[0]
    return {"train_loss_at_best_epoch": float(row.train_loss),
            "validation_combined_nrmse": float(row.validation_combined_normalized_rmse),
            "validation_V1_RMSE": float(row.validation_V1_rmse), "validation_V1_R2": float(row.validation_V1_r2),
            "validation_V2_RMSE": float(row.validation_V2_rmse), "validation_V2_R2": float(row.validation_V2_r2),
            "best_epoch": int(audit["best_epoch"]), "epochs_run": int(audit["epochs_run"]),
            "training_seconds": float(audit["training_seconds"])}


def _advance(context: BranchContext, state: dict, selection: dict) -> dict:
    selected = list(map(int, selection["selected_indices"]))
    selected_set = set(selected)
    next_l = state["labeled_indices"] + selected
    next_u = [value for value in state["unlabeled_indices"] if value not in selected_set]
    validate_trajectory_transition(state["labeled_ids"], state["unlabeled_ids"], selection["selected_ids"],
                                   context.ids(next_l), context.ids(next_u))
    return {"outer_indices": state["outer_indices"], "labeled_indices": next_l, "unlabeled_indices": next_u,
            "labeled_ids": context.ids(next_l), "unlabeled_ids": context.ids(next_u)}


def _verify_branch(lineage: dict, strategy: str) -> dict:
    runtime = _anchor_runtime(lineage["seed"], lineage["source_trajectory"], lineage["anchor_budget"]) / strategy
    trajectory = read_json(runtime / "trajectory_freeze.json")
    if (trajectory.get("status") != "BRANCH_FROZEN_BEFORE_TEST_TRUTH" or trajectory.get("test_truth_access_count") != 0
            or trajectory.get("new_fits") != 2 or trajectory.get("final_active_labels") != lineage["anchor_budget"] + 64):
        raise RuntimeError("invalid branch trajectory freeze")
    assert_hashes(trajectory["files"])
    previous_checkpoint = lineage["checkpoint_sha256"]
    expected_budget = lineage["anchor_budget"]
    for step in (1, 2):
        directory = runtime / f"step_{step:02d}"
        frozen = read_json(directory / "step_freeze.json")
        assert_hashes(frozen["files"])
        if (frozen["input_checkpoint_sha256"] != previous_checkpoint or frozen["from_active_labels"] != expected_budget
                or frozen["to_active_labels"] != expected_budget + 32):
            raise RuntimeError("branch step did not use the preceding branch-specific state")
        previous_checkpoint = frozen["output_checkpoint_sha256"]
        expected_budget += 32
    return trajectory


def _run_branch(context: BranchContext, lineage: dict, strategy: str, anchor_selection: dict) -> dict:
    runtime = _anchor_runtime(context.seed, lineage["source_trajectory"], lineage["anchor_budget"]) / strategy
    if (runtime / "trajectory_freeze.json").exists():
        return _verify_branch(lineage, strategy)
    store, truth, acquired = context.restore_anchor(lineage)
    state = {key: list(lineage[key]) for key in ("outer_indices", "labeled_indices", "unlabeled_indices", "labeled_ids", "unlabeled_ids")}
    current_features = load_anchor_banks(lineage)
    input_checkpoint = ROOT / lineage["checkpoint_path"]
    protected: list[Path] = []
    validations = []
    started = time.perf_counter()
    for step in (1, 2):
        directory = runtime / f"step_{step:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        input_contract = {"seed": context.seed, "source_trajectory": lineage["source_trajectory"],
                          "anchor_budget": lineage["anchor_budget"], "strategy": strategy, "rollout_step": step,
                          "active_label_count": len(state["labeled_indices"]), "candidate_pool_size": len(state["unlabeled_indices"]),
                          "ordered_L_hash": stable_hash(state["labeled_ids"]), "ordered_U_hash": stable_hash(state["unlabeled_ids"]),
                          "checkpoint_sha256": sha256_file(input_checkpoint),
                          "checkpoint_state_hash": read_json(input_checkpoint.parent / "fit_audit.json")["checkpoint_state_hash"],
                          "raw_gradient_bank_hash": array_hash(current_features["ordinary_q50"]),
                          "cw_gradient_bank_hash": array_hash(current_features["center_width"]), "test_truth_access_count": 0}
        write_json_once(directory / "input.json", input_contract)
        selection = anchor_selection if step == 1 else select_batch(strategy, current_features, state)
        selection_record = {"input": input_contract, **selection}
        write_json_once(directory / "selection.json", selection_record)
        # The selected batch is immutable before this branch-local store can reveal it.
        selected_ids = selection["selected_ids"]
        acquired += selected_ids
        store.freeze_acquisitions(acquired)
        new_truth = store.reveal(selected_ids, "after_acquisition_fit")
        previous_l_ids, previous_u_ids = state["labeled_ids"], state["unlabeled_ids"]
        state = _advance(context, state, selection)
        validate_trajectory_transition(previous_l_ids, previous_u_ids, selected_ids, state["labeled_ids"], state["unlabeled_ids"])
        truth = np.vstack([truth, new_truth])
        audit = _fit(context, lineage, strategy, step, state["labeled_indices"], truth, directory)
        if (audit["train_rows"] != lineage["anchor_budget"] + step * BATCH_SIZE
                or audit["test_labels_used_for_fit_or_checkpoint_selection"] != 0):
            raise RuntimeError("branch fit budget or label-access violation")
        model_dir = directory / "model"
        validation = {"seed": context.seed, "source_trajectory": lineage["source_trajectory"],
                      "anchor_budget": lineage["anchor_budget"], "strategy": strategy, "rollout_step": step,
                      "active_label_count": len(state["labeled_indices"]), **_best_validation_row(model_dir, audit)}
        validations.append(validation)
        write_json_once(directory / "validation_metrics.json", validation)
        paths = [directory / "input.json", directory / "selection.json", directory / "validation_metrics.json",
                 model_dir / "best.pt", model_dir / "predictions.csv.gz", model_dir / "fit_audit.json", model_dir / "history.csv"]
        next_features = None
        if step == 1:
            next_features, gradient_audit = _extract_branch_bank(context, model_dir / "best.pt", directory, state)
            diagnostics = {"seed": context.seed, "source_trajectory": lineage["source_trajectory"],
                           "anchor_budget": lineage["anchor_budget"], "strategy": strategy,
                           "state_kind": "branch_after_1", "rollout_step": 1,
                           "raw_gradient_bank_hash": array_hash(next_features["ordinary_q50"]),
                           "cw_gradient_bank_hash": array_hash(next_features["center_width"]),
                           **bank_diagnostics(next_features, state),
                           "gradient_extraction_seconds": float(sum(x["elapsed_seconds"] for x in gradient_audit.values())),
                           "current_validation_nrmse": validation["validation_combined_nrmse"],
                           **_validation_slopes(lineage["recent_validation"][-3:] + [validation])}
            ivr = select_batch("kernel_ivr", next_features, state)["objective"]
            maxdet = select_batch("gradient_maxdet", next_features, state)["objective"]
            diagnostics.update({"ivr_integrated_variance_before": ivr["integrated_variance_before"],
                                "ivr_predicted_relative_reduction_B32": ivr["predicted_relative_reduction"],
                                "maxdet_total_marginal_logdet_gain_B32": maxdet["total_marginal_logdet_gain"]})
            write_json_once(directory / "state_diagnostics.json", diagnostics)
            paths += [directory / "postfit_gradient_features.npz",
                      directory / "postfit_gradient_features.npz.contract.json",
                      directory / "postfit_gradient_audit.json", directory / "postfit_cw_gradient_features.npz",
                      directory / "postfit_cw_gradient_features.npz.contract.json",
                      directory / "postfit_cw_gradient_audit.json", directory / "state_diagnostics.json"]
        frozen = {"status": "STEP_FROZEN_BEFORE_TEST_TRUTH", "input_checkpoint_sha256": sha256_file(input_checkpoint),
                  "output_checkpoint_sha256": audit["checkpoint_sha256"],
                  "from_active_labels": lineage["anchor_budget"] + (step - 1) * BATCH_SIZE,
                  "to_active_labels": lineage["anchor_budget"] + step * BATCH_SIZE,
                  "ordered_L_hash": stable_hash(state["labeled_ids"]), "ordered_U_hash": stable_hash(state["unlabeled_ids"]),
                  "test_truth_access_count": 0, "files": hashes(paths)}
        write_json_once(directory / "step_freeze.json", frozen)
        protected.append(directory / "step_freeze.json")
        input_checkpoint = model_dir / "best.pt"
        if next_features is not None:
            current_features = next_features
    pd.DataFrame(store.audit).to_csv(runtime / "label_access_audit.csv", index=False)
    pd.DataFrame(validations).to_csv(runtime / "validation_metrics.csv", index=False)
    protected += [runtime / "label_access_audit.csv", runtime / "validation_metrics.csv"]
    trajectory = {"status": "BRANCH_FROZEN_BEFORE_TEST_TRUTH", "seed": context.seed,
                  "source_trajectory": lineage["source_trajectory"], "anchor_budget": lineage["anchor_budget"],
                  "strategy": strategy, "new_fits": 2, "final_active_labels": len(state["labeled_indices"]),
                  "elapsed_seconds": time.perf_counter() - started, "test_truth_access_count": 0,
                  "files": hashes(protected)}
    write_json_once(runtime / "trajectory_freeze.json", trajectory)
    return _verify_branch(lineage, strategy)


def execute_seed(seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("seed outside the frozen development cohort")
    validate_seal()
    smoke = read_json(STUDY / "engineering_smoke.json")
    if (smoke.get("status") != "PASSED" or smoke.get("test_truth_access_count") != 0
            or smoke.get("seal_sha256") != sha256_file(STUDY / "seal.json")):
        raise RuntimeError("passing selection-only engineering smoke is required")
    if seed == 6101:
        stage = read_json(STUDY / "runtime/stage_157_check.json")
        if stage.get("status") != "PASSED" or stage.get("test_truth_access_count") != 0:
            raise RuntimeError("seed 157 test-blind stage check must pass before seed 6101")
    seed_runtime = STUDY / "runtime" / f"seed_{seed}"
    with exclusive_lock(seed_runtime / "worker.lock"):
        if (seed_runtime / "seed_freeze.json").exists():
            record = read_json(seed_runtime / "seed_freeze.json")
            assert_hashes(record["files"])
            return record
        context = BranchContext(seed)
        context_record = {"seed": seed, "split_hash": stable_hash(context.partition.to_dict("list")),
                          "preprocessing": context.preprocessing, "normalization": asdict(context.normalization),
                          "training_config": context.training_config(), "initialization_seed": initialization_seed(seed, 0),
                          "source_data_sha256": sha256_file(SOURCE_DATA), "source_graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
                          "test_truth_access_count": 0}
        write_json_once(seed_runtime / "context.json", context_record)
        trajectories, overlaps, anchor_diagnostics = [], [], []
        started = time.perf_counter()
        for source in SOURCES:
            for budget in ANCHOR_BUDGETS:
                lineage = read_json(_lineage_path(seed, source, budget))
                selections, diagnostics = _anchor_selections(lineage, persist=True)
                anchor_diagnostics.append(diagnostics)
                overlaps.extend(_overlap_rows(lineage, selections))
                for strategy in STRATEGIES:
                    print(json.dumps({"seed": seed, "source": source, "anchor_budget": budget,
                                      "strategy": strategy, "stage": "two_round_rollout", "at": now()}), flush=True)
                    _run_branch(context, lineage, strategy, selections[strategy])
                    trajectories.append(_anchor_runtime(seed, source, budget) / strategy / "trajectory_freeze.json")
        pd.DataFrame(overlaps).to_csv(seed_runtime / "anchor_selection_overlap.csv", index=False)
        pd.DataFrame(anchor_diagnostics).to_csv(seed_runtime / "anchor_state_diagnostics.csv", index=False)
        paths = [seed_runtime / "context.json", seed_runtime / "anchor_selection_overlap.csv",
                 seed_runtime / "anchor_state_diagnostics.csv", *trajectories]
        record = {"status": "SEED_FROZEN_BEFORE_TEST_TRUTH", "seed": seed, "branch_count": 12,
                  "new_fits": 24, "elapsed_seconds": time.perf_counter() - started,
                  "test_truth_access_count": 0, "files": hashes(paths)}
        write_json_once(seed_runtime / "seed_freeze.json", record)
        return record


def stage_check(seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("unknown staged seed")
    validate_seal()
    frozen = read_json(STUDY / f"runtime/seed_{seed}/seed_freeze.json")
    assert_hashes(frozen["files"])
    if frozen.get("branch_count") != 12 or frozen.get("new_fits") != 24 or frozen.get("test_truth_access_count") != 0:
        raise RuntimeError("seed stage is incomplete")
    warnings, endpoints, initialization = [], [], set()
    overlap = pd.read_csv(STUDY / f"runtime/seed_{seed}/anchor_selection_overlap.csv")
    for source in SOURCES:
        for budget in ANCHOR_BUDGETS:
            group = overlap.loc[(overlap.source_trajectory == source) & (overlap.anchor_budget == budget)]
            if len(group) != 3:
                raise RuntimeError("incomplete pairwise anchor overlap")
            if bool((group.intersection_count >= 31).all()):
                warnings.append(f"near-identical first batches: {source}@{budget}")
            lineage = read_json(_lineage_path(seed, source, budget))
            for strategy in STRATEGIES:
                _verify_branch(lineage, strategy)
                runtime = _anchor_runtime(seed, source, budget) / strategy
                audit = pd.read_csv(runtime / "label_access_audit.csv")
                if "final_test_evaluation" in set(audit.purpose) or any(audit.roles.astype(str).str.contains("test")):
                    raise RuntimeError("test label access occurred before the global freeze")
                final = read_json(runtime / "step_02/model/fit_audit.json")
                initialization.add(final["initialization_hash"])
                endpoints.append({"source": source, "budget": budget, "strategy": strategy,
                                  "final_active_labels": final["train_rows"]})
    if len(initialization) != 1:
        raise RuntimeError("scratch initialization differs across staged branches")
    expected = {budget + 64 for budget in ANCHOR_BUDGETS}
    if {row["final_active_labels"] for row in endpoints} != expected:
        raise RuntimeError("rollout endpoint budget mismatch")
    result = {"status": "PASSED", "seed": seed, "checks": {"branches_diverged": not warnings,
              "all_artifacts_complete": True, "zero_test_access": True, "endpoints_493_and_717": True,
              "shared_initialization": True}, "warnings": warnings, "validation_ranking_used_to_drop_methods": False,
              "test_truth_access_count": 0}
    atomic_json(STUDY / f"runtime/stage_{seed}_check.json", result)
    return result


def global_freeze() -> dict:
    validate_seal()
    stage157 = read_json(STUDY / "runtime/stage_157_check.json")
    if stage157.get("status") != "PASSED":
        raise RuntimeError("seed 157 stage check did not pass")
    paths = [STUDY / "runtime/stage_157_check.json"]
    for seed in SEEDS:
        frozen = read_json(STUDY / f"runtime/seed_{seed}/seed_freeze.json")
        assert_hashes(frozen["files"])
        if frozen.get("branch_count") != 12 or frozen.get("new_fits") != 24:
            raise RuntimeError("global freeze requires all 24 branches and 48 fits")
        paths.append(STUDY / f"runtime/seed_{seed}/seed_freeze.json")
        for source in SOURCES:
            for budget in ANCHOR_BUDGETS:
                lineage = read_json(_lineage_path(seed, source, budget))
                for strategy in STRATEGIES:
                    _verify_branch(lineage, strategy)
    record = {"status": "ALL_24_BRANCHES_48_FITS_FROZEN_BEFORE_TEST_TRUTH", "branch_count": 24,
              "new_fit_count": 48, "test_truth_access_count": 0, "frozen_at": now(), "files": hashes(paths)}
    write_json_once(STUDY / "global_pre_test_freeze.json", record)
    return record


def _verify_global_branch_freeze() -> dict:
    record = read_json(STUDY / "global_pre_test_freeze.json")
    if (record.get("status") != "ALL_24_BRANCHES_48_FITS_FROZEN_BEFORE_TEST_TRUTH"
            or record.get("branch_count") != 24 or record.get("new_fit_count") != 48
            or record.get("test_truth_access_count") != 0):
        raise RuntimeError("test reveal refused: incomplete global branch freeze")
    assert_hashes(record["files"])
    # Recursive branch verification protects the predictions, not just top-level manifests.
    for seed in SEEDS:
        for source in SOURCES:
            for budget in ANCHOR_BUDGETS:
                lineage = read_json(_lineage_path(seed, source, budget))
                for strategy in STRATEGIES:
                    _verify_branch(lineage, strategy)
    return record


def _rank(values: pd.Series) -> str:
    return " > ".join(values.sort_values().index.astype(str).tolist())


def reveal_test_and_report() -> dict:
    """Read test truth only after the complete recursive prediction freeze."""

    validate_seal()
    _verify_global_branch_freeze()
    metric_rows, branch_rows, validation_rows, diagnostic_rows, overlap_rows, access_rows = [], [], [], [], [], []
    for seed in SEEDS:
        context = BranchContext(seed)
        test_ids = context.ids(context.roles["test"])
        evaluation_store = RestrictedLabelStore(SOURCE_DATA, context.partition)
        evaluation_store.freeze_acquisitions([])
        evaluation_store.freeze_predictions()
        truth = evaluation_store.reveal(test_ids, "final_test_evaluation")
        access_rows.extend({"seed": seed, **row} for row in evaluation_store.audit)
        overlap_rows.extend(pd.read_csv(STUDY / f"runtime/seed_{seed}/anchor_selection_overlap.csv").to_dict("records"))
        diagnostic_rows.extend(pd.read_csv(STUDY / f"runtime/seed_{seed}/anchor_state_diagnostics.csv").to_dict("records"))
        for source in SOURCES:
            for budget in ANCHOR_BUDGETS:
                lineage = read_json(_lineage_path(seed, source, budget))
                anchor_prediction = pd.read_csv(ROOT / lineage["anchor_prediction_path"])
                if anchor_prediction.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("anchor test-X prediction identity drift")
                anchor_metric = metric_row(truth, anchor_prediction.drop(columns="sample_id").to_numpy(float),
                                           context.preprocessing["target_scales"])
                for strategy in STRATEGIES:
                    values = [(0, budget, anchor_metric)]
                    runtime = _anchor_runtime(seed, source, budget) / strategy
                    for step in (1, 2):
                        prediction = pd.read_csv(runtime / f"step_{step:02d}/model/predictions.csv.gz")
                        if prediction.sample_id.astype(str).tolist() != test_ids:
                            raise RuntimeError("branch test-X prediction identity drift")
                        metrics = metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float),
                                             context.preprocessing["target_scales"])
                        values.append((step, budget + step * BATCH_SIZE, metrics))
                        validation_rows.extend(pd.read_csv(runtime / "validation_metrics.csv").to_dict("records"))
                        if step == 1:
                            diagnostic_rows.append(read_json(runtime / "step_01/state_diagnostics.json"))
                    for step, active, metrics in values:
                        metric_rows.append({"seed": seed, "source_trajectory": source, "anchor_budget": budget,
                                            "strategy": strategy, "rollout_step": step, "active_label_count": active, **metrics})
                    errors = [row[2]["combined_normalized_RMSE"] for row in values]
                    branch_rows.append({"seed": seed, "source_trajectory": source, "anchor_budget": budget,
                                        "strategy": strategy, "NRMSE_anchor": errors[0], "NRMSE_plus32": errors[1],
                                        "NRMSE_plus64": errors[2], "short_AULC": (errors[0] + 2 * errors[1] + errors[2]) / 4,
                                        "delta_NRMSE_32": errors[0] - errors[1], "delta_NRMSE_64": errors[0] - errors[2],
                                        "V1_RMSE_plus64": values[2][2]["V1_RMSE"], "V1_R2_plus64": values[2][2]["V1_R2"],
                                        "V2_RMSE_plus64": values[2][2]["V2_RMSE"], "V2_R2_plus64": values[2][2]["V2_R2"]})
    # The loop above reads each branch's two-row validation file twice; canonicalize exact keys.
    validation = pd.DataFrame(validation_rows).drop_duplicates(
        ["seed", "source_trajectory", "anchor_budget", "strategy", "rollout_step"]
    ).sort_values(["seed", "source_trajectory", "anchor_budget", "strategy", "rollout_step"])
    metrics = pd.DataFrame(metric_rows)
    branches = pd.DataFrame(branch_rows)
    diagnostics = pd.DataFrame(diagnostic_rows).drop_duplicates(
        [column for column in ("seed", "source_trajectory", "anchor_budget", "strategy", "state_kind") if column in pd.DataFrame(diagnostic_rows)]
    )
    overlap = pd.DataFrame(overlap_rows)
    paired = []
    for key, rows in branches.groupby(["seed", "source_trajectory", "anchor_budget"]):
        rows = rows.set_index("strategy")
        for left, right in itertools.combinations(STRATEGIES, 2):
            paired.append({"seed": key[0], "source_trajectory": key[1], "anchor_budget": key[2],
                           "left_strategy": left, "right_strategy": right,
                           "plus32_left_minus_right": rows.loc[left, "NRMSE_plus32"] - rows.loc[right, "NRMSE_plus32"],
                           "plus64_left_minus_right": rows.loc[left, "NRMSE_plus64"] - rows.loc[right, "NRMSE_plus64"],
                           "short_AULC_left_minus_right": rows.loc[left, "short_AULC"] - rows.loc[right, "short_AULC"]})
    winners = []
    for key, rows in branches.groupby(["seed", "source_trajectory", "anchor_budget"]):
        indexed = rows.set_index("strategy")
        val = validation.loc[(validation.seed == key[0]) & (validation.source_trajectory == key[1])
                             & (validation.anchor_budget == key[2]) & (validation.rollout_step == 2)].set_index("strategy")
        winners.append({"seed": key[0], "source_trajectory": key[1], "anchor_budget": key[2],
                        "winner_plus32": indexed.NRMSE_plus32.idxmin(), "winner_plus64": indexed.NRMSE_plus64.idxmin(),
                        "winner_short_AULC": indexed.short_AULC.idxmin(),
                        "test_AULC_ranking": _rank(indexed.short_AULC),
                        "validation_plus64_ranking": _rank(val.validation_combined_nrmse),
                        "validation_test_ranking_exact_match": _rank(indexed.NRMSE_plus64) == _rank(val.validation_combined_nrmse),
                        "within_anchor_AULC_range": float(indexed.short_AULC.max() - indexed.short_AULC.min())})
    winners = pd.DataFrame(winners)
    # Eight-anchor descriptive correlations only; no p-values or fitted controller.
    anchor_diag = diagnostics.loc[diagnostics.state_kind.eq("anchor")].copy()
    wide = branches.pivot(index=["seed", "source_trajectory", "anchor_budget"], columns="strategy", values="short_AULC").reset_index()
    merged = anchor_diag.merge(wide, on=["seed", "source_trajectory", "anchor_budget"], validate="one_to_one")
    feature_names = [name for name in (
        "recent_validation_nrmse", "recent_validation_last_step_delta",
        "recent_validation_2step_linear_slope_per_label", "recent_validation_3step_linear_slope_per_label",
        "raw_gradient_norm_mean", "raw_gradient_norm_std", "raw_gradient_norm_p90", "raw_gradient_norm_p95",
        "raw_gradient_effective_rank_participation_ratio", "cw_gradient_coverage_nearest_distance_mean",
        "cw_gradient_coverage_nearest_distance_p90", "ivr_predicted_relative_reduction_B32",
        "maxdet_total_marginal_logdet_gain_B32") if name in merged]
    correlations = []
    for feature in feature_names:
        for strategy in STRATEGIES:
            correlations.append({"feature": feature, "outcome": f"short_AULC_{strategy}", "n_anchors": len(merged),
                                 "spearman_rho": float(merged[feature].corr(merged[strategy], method="spearman")),
                                 "p_value": None, "interpretation": "exploratory_descriptive_only"})
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    for name, frame in {"test_metrics": metrics, "branch_results": branches, "paired_comparison": pd.DataFrame(paired),
                        "winner_matrix": winners, "selection_overlap": overlap, "state_diagnostics": diagnostics,
                        "validation_metrics": validation, "exploratory_state_correlations": pd.DataFrame(correlations),
                        "test_label_access_audit": pd.DataFrame(access_rows)}.items():
        frame.to_csv(results / f"{name}.csv", index=False)
    total_elapsed = sum(read_json(STUDY / f"runtime/seed_{seed}/seed_freeze.json")["elapsed_seconds"] for seed in SEEDS)
    source_dependence = int(sum(group.winner_short_AULC.nunique() > 1 for _, group in winners.groupby(["seed", "anchor_budget"])))
    stage_dependence = int(sum(group.winner_short_AULC.nunique() > 1 for _, group in winners.groupby(["seed", "source_trajectory"])))
    seed_dependence = int(sum(group.winner_short_AULC.nunique() > 1 for _, group in winners.groupby(["source_trajectory", "anchor_budget"])))
    decision = {"status": "COMPLETED", "evidence": "development_mechanism_not_independent_confirmation",
                "actual_new_fits": 48, "branch_count": 24, "total_execution_seconds": total_elapsed,
                "source_trajectory_winner_changes_out_of_4_seed_budget_pairs": source_dependence,
                "budget_winner_changes_out_of_4_seed_source_pairs": stage_dependence,
                "seed_winner_changes_out_of_4_source_budget_pairs": seed_dependence,
                "validation_test_exact_ranking_matches_out_of_8": int(winners.validation_test_ranking_exact_match.sum()),
                "test_truth_revealed_only_after_global_freeze": True, "controller_trained": False,
                "completed_at": now()}
    atomic_json(STUDY / "decision.json", decision)
    _write_report(branches, winners, overlap, anchor_diag, decision)
    return decision


def _write_report(branches: pd.DataFrame, winners: pd.DataFrame, overlap: pd.DataFrame,
                  diagnostics: pd.DataFrame, decision: dict) -> None:
    summary = branches.groupby(["source_trajectory", "anchor_budget", "strategy"])[
        ["NRMSE_anchor", "NRMSE_plus32", "NRMSE_plus64", "short_AULC", "delta_NRMSE_32", "delta_NRMSE_64"]
    ].mean().reset_index()
    def markdown(frame: pd.DataFrame) -> str:
        return frame.to_markdown(index=False, floatfmt=".6f")
    rank_changes = decision["source_trajectory_winner_changes_out_of_4_seed_budget_pairs"]
    stage_changes = decision["budget_winner_changes_out_of_4_seed_source_pairs"]
    seed_changes = decision["seed_winner_changes_out_of_4_source_budget_pairs"]
    if rank_changes or stage_changes:
        adaptive = ("State-dependent ranking is present descriptively, but the two development seeds are too few for a controller. "
                    "The next minimal experiment is a preregistered same-state replication on untouched seeds or, preferably, a held-out time/experimental cohort, using only the three frozen selectors and two most discriminating test-blind diagnostics.")
    else:
        adaptive = ("No reproducible state-dependent ranking signal appears in these eight anchors. Do not build an adaptive selector; first replicate the short rollout only if a new independent cohort becomes available.")
    text = ["# Same-state short-rollout branching: final development report", "",
            "All 24 branches (48 new scratch fits) were prediction-frozen before one unified test-reveal action. "
            "This is development/mechanism evidence on two previously used seeds, not independent confirmation.", "",
            "## Mean paired outcomes", "", markdown(summary), "", "## Strategy winner matrix", "", markdown(winners), "",
            "## Direct answers", "",
            f"- Source-trajectory winner changed in {rank_changes}/4 matched seed-budget comparisons.",
            f"- Budget-stage winner changed in {stage_changes}/4 matched seed-source comparisons.",
            f"- Seed changed the winner in {seed_changes}/4 matched source-budget comparisons.",
            f"- Validation and test +64 rankings matched exactly in {decision['validation_test_exact_ranking_matches_out_of_8']}/8 anchors; validation remains a reused development observable.",
            "- A fixed label-count switch is supported only if budget effects are consistent across both source trajectories and seeds; otherwise the better hypothesis is a_t = pi(s_t), not a_t = pi(n_t).",
            f"- {adaptive}", "", "## State diagnostics", "", markdown(diagnostics), "",
            "## First-batch overlap", "", markdown(overlap), "",
            "Full per-seed test metrics, paired differences, validation fits, correlations and label-access audit are in `results/`. "
            "Correlations use n=8 anchors, report no p-values, and are descriptive only. Historical source artifacts remained read-only."]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(text) + "\n")

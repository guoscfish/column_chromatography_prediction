"""Parameterized protocols for Gradient+Latent Fusion-MaxDet studies."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import (
    CONFIRMATION_SEEDS as HISTORICAL_SEEDS,
    FEATURE_COLUMNS,
    TRAINING_CONFIG,
    load_features,
)
from .cache import array_hash
from .fused_features import FUSED_DIMENSION, GRADIENT_DIMENSION, LATENT_DIMENSION
from .maxdet_study import BASELINE, historical_round0_paths
from .protocol import ids_hash, stable_hash, validate_row_protocol
from .sequential_protocol import (
    ACTIVE_LABEL_BUDGETS as FULL_ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    INITIAL_ACTIVE_LABELS,
    INITIAL_POOL_ROWS,
    VALIDATION_ROWS,
)


FROZEN_A050_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_fused_maxdet_b32"
SCREEN_ACTIVE_LABEL_BUDGETS = (333, 365, 397, 429, 461, 493, 525, 557, 589, 621, 653)


@dataclass(frozen=True)
class FusedMaxDetStudySpec:
    """Every value that can distinguish a weighted-fusion trajectory."""

    study: Path
    method: str
    alpha: float
    active_label_budgets: tuple[int, ...]
    code_files: tuple[str, ...]
    seeds: tuple[int, ...] = (157, 6101)
    evidence_class: str = "developmental_screen"

    def __post_init__(self) -> None:
        budgets = tuple(int(value) for value in self.active_label_budgets)
        if not np.isfinite(self.alpha) or not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("fusion alpha must be finite and in [0, 1]")
        if not budgets or budgets[0] != INITIAL_ACTIVE_LABELS:
            raise ValueError("fusion budgets must start at the fixed initial label count")
        if any(right - left != BATCH_SIZE for left, right in zip(budgets, budgets[1:])):
            raise ValueError("fusion budgets must advance by the fixed batch size")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("fusion study seeds must be unique")

    @property
    def acquisition_rounds(self) -> int:
        return len(self.active_label_budgets) - 1

    @property
    def final_active_labels(self) -> int:
        return int(self.active_label_budgets[-1])

    @property
    def kernel_definition(self) -> str:
        return (
            f"{float(self.alpha):.12g}*K_gradient_normalized + "
            f"{1.0 - float(self.alpha):.12g}*K_latent_normalized"
        )


SHARED_CODE_FILES = (
    "src/qgeognn_al/active_learning_v2/fused_features.py",
    "src/qgeognn_al/active_learning_v2/fused_maxdet_study.py",
    "src/qgeognn_al/active_learning_v2/fused_maxdet_runner.py",
    "src/qgeognn_al/active_learning_v2/fused_maxdet_reporting.py",
    "src/qgeognn_al/active_learning_v2/gradient_features.py",
    "src/qgeognn_al/active_learning_v2/coverage.py",
    "src/qgeognn_al/active_learning_v2/maxdet.py",
    "src/qgeognn_al/active_learning_v2/runner.py",
)

DEFAULT_SPEC = FusedMaxDetStudySpec(
    study=FROZEN_A050_STUDY,
    method="gradient_latent_fusion_maxdet",
    alpha=0.5,
    active_label_budgets=tuple(FULL_ACTIVE_LABEL_BUDGETS),
    code_files=("scripts/studies/run_qgeognn_v2_row_fused_maxdet_b32.py", *SHARED_CODE_FILES),
    evidence_class="developmental_evidence",
)

A080_SCREEN_SPEC = FusedMaxDetStudySpec(
    study=ROOT / "studies/active_learning/qgeognn_v2_row_fused_maxdet_a080_b32_screen",
    method="gradient_latent_fusion_a080_maxdet",
    alpha=0.8,
    active_label_budgets=SCREEN_ACTIVE_LABEL_BUDGETS,
    code_files=("scripts/studies/run_qgeognn_v2_row_fused_maxdet_a080_b32_screen.py", *SHARED_CODE_FILES),
    evidence_class="truncated_developmental_screen",
)

# Compatibility aliases retained for the completed alpha=.5 entry point.
STUDY = DEFAULT_SPEC.study
METHOD = DEFAULT_SPEC.method
CONFIRMATION_SEEDS = DEFAULT_SPEC.seeds


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def code_paths(spec: FusedMaxDetStudySpec = DEFAULT_SPEC) -> tuple[Path, ...]:
    return tuple(ROOT / value for value in spec.code_files)


def code_hashes(spec: FusedMaxDetStudySpec = DEFAULT_SPEC) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in code_paths(spec)
        if path.exists()
    }


def protocol_record(spec: FusedMaxDetStudySpec = DEFAULT_SPEC) -> dict[str, object]:
    return {
        "study": spec.study.name,
        "method": spec.method,
        "evidence_class": spec.evidence_class,
        "historical_confirmation_seeds": list(HISTORICAL_SEEDS),
        "confirmation_seeds": list(spec.seeds),
        "row_split": True,
        "initial_active_labels": INITIAL_ACTIVE_LABELS,
        "initial_pool_rows": INITIAL_POOL_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "batch_size": BATCH_SIZE,
        "acquisition_rounds": spec.acquisition_rounds,
        "active_label_budgets": list(spec.active_label_budgets),
        "final_active_labels": spec.final_active_labels,
        "gradient_dimension": GRADIENT_DIMENSION,
        "latent_dimension": LATENT_DIMENSION,
        "fused_dimension": FUSED_DIMENSION,
        "alpha": float(spec.alpha),
        "normalization": "s_g/h=sqrt(mean_L_t(||block||^2)); phi=[sqrt(alpha)g/s_g,sqrt(1-alpha)h/s_h]",
        "kernel": spec.kernel_definition,
        "latent_acquisition_centering": False,
        "latent_common_mode_diagnostic": (
            "mu_h=mean_L_t(h); r_mean=||mu_h||^2/mean_L_t(||h_i||^2); "
            "centered diagnostics use h-mu_h only and never affect acquisition"
        ),
        "selector": "conditional_gradient_maxdet(features=phi), unchanged tested selector",
        "training": TRAINING_CONFIG,
        "target_scales": "fixed original L0 population std ddof=0",
        "test_reveal_barrier": (
            f"all {len(spec.seeds)} seeds x {len(spec.active_label_budgets)} prediction points "
            f"x {spec.acquisition_rounds} batches frozen"
        ),
        "comparators": ["gradient_maxdet", "gradient_latent_fusion_maxdet_alpha_0.5"],
        "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS),
        "code_hashes": code_hashes(spec),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn")
        },
        "python": platform.python_version(),
    }


def frozen_a050_round0_paths(seed: int) -> dict[str, Path]:
    base = (
        FROZEN_A050_STUDY
        / "runtime"
        / f"seed_{int(seed)}"
        / DEFAULT_SPEC.method
        / "round_00"
    )
    artifacts = base / "acquisition_artifacts"
    return {
        "round_freeze": base / "round_freeze.json",
        "latent": artifacts / "current_latent_features.npz",
        "latent_contract": artifacts / "current_latent_features.npz.contract.json",
    }


def _validate_round0_reuse(seed: int, partition: pd.DataFrame) -> dict[str, object]:
    """Prove model, data, ordering, preprocessing, gradient, and latent identity."""

    paths = historical_round0_paths(seed)
    latent_paths = frozen_a050_round0_paths(seed)
    required = (
        paths["member0_model"], paths["member0_predictions"], paths["member0_audit"],
        paths["gradient"], paths["gradient_contract"], paths["gradient_audit"],
        paths["scrubbed_graphs"], paths["context"],
        latent_paths["round_freeze"], latent_paths["latent"], latent_paths["latent_contract"],
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"round-zero reuse is incomplete for seed {seed}: {missing}")

    current = np.r_[
        partition.loc[partition.role.eq("l0"), "canonical_index"].to_numpy(int),
        partition.loc[partition.role.eq("u0"), "canonical_index"].to_numpy(int),
    ]
    current_ids = partition.iloc[current].sample_id.astype(str).tolist()
    l0_ids = partition.loc[partition.role.eq("l0"), "sample_id"].astype(str).tolist()
    u0_ids = partition.loc[partition.role.eq("u0"), "sample_id"].astype(str).tolist()
    checkpoint_sha = sha256_file(paths["member0_model"])
    fit_audit = _json(paths["member0_audit"])
    context = _json(paths["context"])["contract"]
    if context.get("source_sha256") != sha256_file(SOURCE_DATA):
        raise RuntimeError(f"round-zero source drift for seed {seed}")
    if context.get("source_graph_cache_sha256") != sha256_file(SOURCE_GRAPH_CACHE):
        raise RuntimeError(f"round-zero graph cache drift for seed {seed}")
    expected_hashes = {
        "l0_ids_hash": ids_hash(l0_ids),
        "u0_ids_hash": ids_hash(u0_ids),
        "validation_ids_hash": ids_hash(partition.loc[partition.role.eq("validation"), "sample_id"]),
        "test_ids_hash": ids_hash(partition.loc[partition.role.eq("test"), "sample_id"]),
    }
    if any(context.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError(f"round-zero split/sample identity drift for seed {seed}")

    with np.load(paths["gradient"]) as cached:
        gradient = np.asarray(cached["features"], dtype=np.float32)
        gradient_indices = np.asarray(cached["canonical_indices"], dtype=int)
    gradient_receipt = _json(paths["gradient_contract"])
    gradient_contract = gradient_receipt["contract"]
    if (
        not np.array_equal(gradient_indices, current)
        or gradient.shape != (len(current), GRADIENT_DIMENSION)
        or gradient_contract.get("evaluation_checkpoint_sha256") != checkpoint_sha
        or gradient_contract.get("evaluation_checkpoint_state_hash") != fit_audit.get("checkpoint_state_hash")
        or gradient_contract.get("L_t_ids_hash") != expected_hashes["l0_ids_hash"]
        or gradient_contract.get("U_t_ids_hash") != expected_hashes["u0_ids_hash"]
        or gradient_receipt.get("sha256") != sha256_file(paths["gradient"])
    ):
        raise RuntimeError(f"round-zero gradient cache contract drift for seed {seed}")

    with np.load(latent_paths["latent"]) as cached:
        latent = np.asarray(cached["features"], dtype=np.float32)
        latent_indices = np.asarray(cached["canonical_indices"], dtype=int)
    latent_receipt = _json(latent_paths["latent_contract"])
    latent_contract = latent_receipt["contract"]
    frozen_round = _json(latent_paths["round_freeze"])
    if (
        not np.array_equal(latent_indices, current)
        or latent.shape != (len(current), LATENT_DIMENSION)
        or latent_contract.get("checkpoint_sha256") != checkpoint_sha
        or latent_contract.get("checkpoint_state_hash") != fit_audit.get("checkpoint_state_hash")
        or latent_contract.get("ordered_current_ids_hash") != stable_hash(current_ids)
        or latent_contract.get("labeled_ids_hash") != expected_hashes["l0_ids_hash"]
        or latent_contract.get("unlabeled_ids_hash") != expected_hashes["u0_ids_hash"]
        or latent_contract.get("feature_sha256") != array_hash(latent)
        or latent_receipt.get("sha256") != sha256_file(latent_paths["latent"])
        or frozen_round.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise RuntimeError(f"round-zero latent/checkpoint contract drift for seed {seed}")

    return {
        "outer_seed": int(seed),
        "split_sha256": sha256_file(BASELINE / "splits" / f"row_seed_{seed}.csv"),
        **expected_hashes,
        "ordered_current_ids_hash": stable_hash(current_ids),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_hash": fit_audit["checkpoint_state_hash"],
        "preprocessing_hash": stable_hash(context["preprocessing"]),
        "normalization_hash": stable_hash(context["normalization"]),
        "target_scales": context["preprocessing"]["target_scales"],
        "scrubbed_graphs_sha256": sha256_file(paths["scrubbed_graphs"]),
        "gradient_cache_sha256": sha256_file(paths["gradient"]),
        "gradient_feature_sha256": array_hash(gradient),
        "latent_cache_sha256": sha256_file(latent_paths["latent"]),
        "latent_feature_sha256": array_hash(latent),
        "round_zero_reuse_legal": True,
    }


def prepare(
    study: Path | None = None,
    *,
    spec: FusedMaxDetStudySpec = DEFAULT_SPEC,
) -> dict[str, object]:
    study = spec.study if study is None else Path(study)
    if study != spec.study:
        raise ValueError("study path must match the explicit fusion study specification")
    if (study / "runtime").exists():
        raise RuntimeError("refusing to prepare over an existing fusion runtime")
    data = load_features()
    splits = []
    reuse = []
    for seed in spec.seeds:
        source = BASELINE / "splits" / f"row_seed_{seed}.csv"
        if not source.exists():
            raise RuntimeError(f"missing matched row split for seed {seed}")
        frame = pd.read_csv(source)
        validate_row_protocol(frame)
        if frame.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
            raise RuntimeError(f"split/source identity drift for seed {seed}")
        destination = study / "splits" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing to replace changed split: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        record = _validate_round0_reuse(seed, frame)
        splits.append({key: record[key] for key in ("outer_seed", "split_sha256", "l0_ids_hash", "u0_ids_hash", "validation_ids_hash", "test_ids_hash")})
        reuse.append(record)

    record = protocol_record(spec)
    study.mkdir(parents=True, exist_ok=True)
    atomic_json(study / "protocol.json", record)
    atomic_json(study / "splits/split_manifest.json", {"protocol_hash": stable_hash(record), "splits": splits})
    atomic_json(study / "reuse_audit.json", {
        "status": "PASS",
        "alpha": float(spec.alpha),
        "round_zero_reused": ["model", "predictions", "gradient_features", "latent_features", "scrubbed_graphs"],
        "later_rounds_retrained_and_reextracted": True,
        "source_trajectory_reuse_after_round_zero": False,
        "seeds": reuse,
    })
    atomic_json(study / "global_pre_test_freeze.json", {"status": "PENDING_PRE_TEST_EXECUTION", "test_truth_access_count": 0})
    atomic_json(study / "decision.json", {"status": "PENDING_COMPLETE_TRAJECTORIES", "evidence_class": spec.evidence_class})
    atomic_json(study / "pre_test_manifest.json", {
        "phase": "PRE_TEST_PROTOCOL_FROZEN",
        "protocol_hash": stable_hash(record),
        "alpha": float(spec.alpha),
        "test_truth_access_count": 0,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (*code_paths(spec), study / "protocol.json", study / "reuse_audit.json")
        },
    })
    return {"status": "PREPARED", "protocol_hash": stable_hash(record), "seeds": list(spec.seeds), "alpha": float(spec.alpha)}


def validate_pre_test_manifest(
    study: Path | None = None,
    *,
    spec: FusedMaxDetStudySpec = DEFAULT_SPEC,
) -> dict[str, object]:
    study = spec.study if study is None else Path(study)
    if study != spec.study:
        raise ValueError("study path must match the explicit fusion study specification")
    manifest = _json(study / "pre_test_manifest.json")
    if manifest.get("phase") != "PRE_TEST_PROTOCOL_FROZEN" or manifest.get("alpha") != float(spec.alpha):
        raise RuntimeError("fusion pre-test manifest is not frozen for the requested alpha")
    if _json(study / "protocol.json") != protocol_record(spec):
        raise RuntimeError("fusion protocol differs from implementation/specification")
    for relative, digest in manifest["files"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"fusion frozen input drift: {relative}")
    reuse = _json(study / "reuse_audit.json")
    if reuse.get("status") != "PASS" or reuse.get("alpha") != float(spec.alpha):
        raise RuntimeError("fusion reuse audit is not passing for the requested alpha")
    return {"status": "PASS", "protocol_hash": manifest["protocol_hash"], "alpha": float(spec.alpha)}


def historical_round0(seed: int) -> dict[str, Path]:
    return historical_round0_paths(int(seed))


__all__ = [
    "FusedMaxDetStudySpec", "DEFAULT_SPEC", "A080_SCREEN_SPEC",
    "STUDY", "METHOD", "CONFIRMATION_SEEDS", "SCREEN_ACTIVE_LABEL_BUDGETS",
    "prepare", "validate_pre_test_manifest", "historical_round0",
    "frozen_a050_round0_paths", "protocol_record",
]

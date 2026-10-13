"""Frozen protocol and Commit-A gate for sequential B=32 active learning."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import (
    CONFIRMATION_SEEDS,
    FEATURE_COLUMNS,
    STUDY as B32_STUDY,
    TRAINING_CONFIG,
    load_features,
)
from .protocol import ids_hash, stable_hash, validate_row_protocol


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_sequential_b32"
BASE_COMMIT = "4ec81c9fadb6617fd57c6729d81645752c8d7668"
COMMIT_MESSAGE = "study: preregister sequential B32 active learning efficiency"
METHODS = ("random", "hybrid", "lcmd")
BATCH_SIZE = 32
ACQUISITION_ROUNDS = 21
INITIAL_ACTIVE_LABELS = 333
FINAL_ACTIVE_LABELS = INITIAL_ACTIVE_LABELS + BATCH_SIZE * ACQUISITION_ROUNDS
ACTIVE_LABEL_BUDGETS = tuple(
    INITIAL_ACTIVE_LABELS + BATCH_SIZE * round_index
    for round_index in range(ACQUISITION_ROUNDS + 1)
)
OUTER_TRAIN_ROWS = 3330
VALIDATION_ROWS = 416
TEST_ROWS = 417
FULL_DATASET_ROWS = 4163
INITIAL_POOL_ROWS = 2997
ENSEMBLE_K = 3
SHORTLIST_FRACTION = 0.25
SKETCH_DIMENSION = 512
RANDOM_TRAJECTORY_OFFSET = 900_001
COMPARABLE_AULC_RELATIVE_MARGIN = 0.02
COMPARABLE_LABEL_MARGIN = BATCH_SIZE
MUTABLE_FORMAL_OUTPUTS = (
    "decision.json",
    "global_pre_test_freeze.json",
    "FINAL_REPORT.md",
)


def random_trajectory_seed(outer_seed: int) -> int:
    """The independent, preregistered seed for one nested Random trajectory."""

    return int(outer_seed) * 1_000_003 + RANDOM_TRAJECTORY_OFFSET


def label_budget(active: int) -> dict[str, float | int]:
    if int(active) not in ACTIVE_LABEL_BUDGETS and int(active) != OUTER_TRAIN_ROWS:
        raise ValueError("active label count is outside the frozen budget schedule")
    return {
        "active_label_count": int(active),
        "active_label_fraction_outer_train": int(active) / OUTER_TRAIN_ROWS,
        "shared_validation_label_count": VALIDATION_ROWS,
        "total_observed_non_test_labels": int(active) + VALIDATION_ROWS,
        "total_observed_fraction_full_dataset": (int(active) + VALIDATION_ROWS) / FULL_DATASET_ROWS,
    }


def sequential_code_paths() -> tuple[Path, ...]:
    """Code whose content is frozen by Commit A and checked before formal work."""

    relative = (
        "scripts/studies/run_qgeognn_v2_row_sequential_b32.py",
        "src/qgeognn_al/active_learning_v2/cache.py",
        "src/qgeognn_al/active_learning_v2/coverage.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py",
        "src/qgeognn_al/active_learning_v2/lcmd.py",
        "src/qgeognn_al/active_learning_v2/protocol.py",
        "src/qgeognn_al/active_learning_v2/runner.py",
        "src/qgeognn_al/active_learning_v2/sequential_acquisition.py",
        "src/qgeognn_al/active_learning_v2/sequential_protocol.py",
        "src/qgeognn_al/active_learning_v2/sequential_reporting.py",
        "src/qgeognn_al/active_learning_v2/sequential_runner.py",
        "src/qgeognn_al/active_learning_v2/uncertainty.py",
    )
    return tuple(ROOT / name for name in relative)


def code_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sequential_code_paths()
    }


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn")
    }


def protocol_record() -> dict[str, object]:
    return {
        "study": STUDY.name,
        "base_commit": BASE_COMMIT,
        "branch": "exp/qgeognn-v2-4g-row-al",
        "cohort_description": "sequential extension on the established confirmation cohort",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "methods": list(METHODS),
        "batch_size": BATCH_SIZE,
        "acquisition_rounds": ACQUISITION_ROUNDS,
        "active_label_budgets": list(ACTIVE_LABEL_BUDGETS),
        "initial_active_labels": INITIAL_ACTIVE_LABELS,
        "initial_pool_rows": INITIAL_POOL_ROWS,
        "outer_train_rows": OUTER_TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "test_rows": TEST_ROWS,
        "full_dataset_rows": FULL_DATASET_ROWS,
        "ensemble_K": ENSEMBLE_K,
        "hybrid_shortlist_fraction": SHORTLIST_FRACTION,
        "gradient_sketch_dimension": SKETCH_DIMENSION,
        "gradient_geometry": "squared_euclidean_corrected_LCMD_TP",
        "random_trajectory_seed_formula": "outer_seed * 1_000_003 + 900_001",
        "random_trajectory_offset": RANDOM_TRAJECTORY_OFFSET,
        "target_scales": "original_L0_population_std_ddof_0_fixed_for_trajectory",
        "training": TRAINING_CONFIG,
        "primary_metrics": ["normalized_AULC", "labels_to_T_R30", "incremental_label_saving"],
        "secondary_target": "T_80 = E_full + 0.20 * (E_0 - E_full)",
        "target_interpolation": "linear_between_adjacent_registered_budget_points_no_extrapolation",
        "test_reveal_barrier": "all_5_seeds_x_3_methods_x_rounds_0_through_21_plus_full_reference_frozen",
        "decision_thresholds": {
            "lcmd_directional_wins_vs_random": 4,
            "comparable_relative_aulc": COMPARABLE_AULC_RELATIVE_MARGIN,
            "comparable_labels": COMPARABLE_LABEL_MARGIN,
        },
        "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS),
        "code_hashes": code_hashes(),
        "packages": package_versions(),
        "python": platform.python_version(),
    }


def _validate_source_and_splits(study: Path) -> list[dict[str, object]]:
    data = load_features()
    if len(data) != FULL_DATASET_ROWS:
        raise RuntimeError("canonical 4g row count drift")
    destination = study / "splits"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in CONFIRMATION_SEEDS:
        old_path = B32_STUDY / "splits" / f"row_seed_{seed}.csv"
        if not old_path.exists():
            raise RuntimeError(f"missing established confirmation split for seed {seed}")
        partition = pd.read_csv(old_path)
        validate_row_protocol(partition)
        if partition.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
            raise RuntimeError(f"established split/source identity drift for seed {seed}")
        counts = partition.role.value_counts().to_dict()
        expected = {"u0": INITIAL_POOL_ROWS, "test": TEST_ROWS, "validation": VALIDATION_ROWS,
                    "l0": INITIAL_ACTIVE_LABELS}
        if counts != expected or set(partition.outer_seed) != {seed}:
            raise RuntimeError(f"established split contract drift for seed {seed}")
        path = destination / old_path.name
        if path.exists() and sha256_file(path) != sha256_file(old_path):
            raise RuntimeError(f"refusing to replace changed sequential split for seed {seed}")
        if not path.exists():
            shutil.copy2(old_path, path)
        records.append({
            "outer_seed": seed,
            "sha256": sha256_file(path),
            "role_counts": counts,
            **{
                f"{role}_ids_hash": ids_hash(partition.loc[partition.role.eq(role), "sample_id"])
                for role in ("l0", "u0", "validation", "test")
            },
        })
    return records


def prepare(study: Path = STUDY) -> dict[str, object]:
    """Create deterministic machine-readable Phase-1 protocol artifacts."""

    if (study / "runtime").exists():
        raise RuntimeError("Commit-A preparation cannot run over formal runtime artifacts")
    record = protocol_record()
    split_records = _validate_source_and_splits(study)
    atomic_json(study / "protocol.json", record)
    atomic_json(study / "splits/split_manifest.json", {
        "protocol_hash": stable_hash(record),
        "source_sha256": record["source_sha256"],
        "splits": split_records,
    })
    pd.DataFrame([{"round": round_index, **label_budget(active)}
                  for round_index, active in enumerate(ACTIVE_LABEL_BUDGETS)]).to_csv(
        study / "label_budget_schedule.csv", index=False
    )
    return {"status": "PREPARED", "protocol_hash": stable_hash(record), "splits": len(split_records)}


def validate_preflight(study: Path = STUDY) -> dict[str, object]:
    """Validate frozen constants, split identity, placeholders, and no formal run."""

    record = json.loads((study / "protocol.json").read_text())
    if record != protocol_record():
        raise RuntimeError("protocol.json differs from the current frozen implementation")
    manifest = json.loads((study / "splits/split_manifest.json").read_text())
    if manifest["protocol_hash"] != stable_hash(record):
        raise RuntimeError("split manifest protocol binding mismatch")
    by_seed = {int(item["outer_seed"]): item for item in manifest["splits"]}
    if set(by_seed) != set(CONFIRMATION_SEEDS):
        raise RuntimeError("split manifest does not contain the exact confirmation cohort")
    for seed, item in by_seed.items():
        path = study / "splits" / f"row_seed_{seed}.csv"
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"split hash mismatch for seed {seed}")
    if (study / "runtime").exists():
        raise RuntimeError("formal sequential runtime exists before Commit A")
    decision = json.loads((study / "decision.json").read_text())
    freeze = json.loads((study / "global_pre_test_freeze.json").read_text())
    if decision.get("status") != "PENDING_FORMAL_RUN" or freeze.get("status") != "PENDING_FORMAL_RUN":
        raise RuntimeError("Commit-A result placeholders contain a premature decision or freeze")
    return {
        "status": "PASS",
        "protocol_hash": stable_hash(record),
        "formal_training_started": False,
        "formal_test_truth_access_count": 0,
    }


def _junit_totals(path: Path) -> dict[str, int]:
    suites = ET.parse(path).getroot()
    leaves = [suites] if suites.tag == "testsuite" else list(suites.iter("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in leaves)
        for name in ("tests", "failures", "errors", "skipped")
    }


def seal_commit_a(test_report: Path, study: Path = STUDY) -> dict[str, object]:
    """Seal passing Phase-1 evidence; the manifest self-hash is intentionally excluded."""

    validation = validate_preflight(study)
    totals = _junit_totals(Path(test_report))
    if totals["tests"] == 0 or totals["failures"] or totals["errors"]:
        raise RuntimeError("a nonempty passing pytest JUnit report is required")
    atomic_json(study / "phase1_validation.json", {
        **validation,
        "tests": totals,
        "pytest_junit_sha256": sha256_file(Path(test_report)),
        "pytest_command": "python -m pytest -q --junitxml=PATH",
        "formal_run_command_executed": False,
    })
    files = {**code_hashes()}
    for path in (ROOT / "tests/active_learning_v2").glob("*.py"):
        files[str(path.relative_to(ROOT))] = sha256_file(path)
    mutable = {study / name for name in MUTABLE_FORMAL_OUTPUTS}
    for path in study.rglob("*"):
        if path.is_file() and path.name != "artifact_manifest.json" and path not in mutable:
            files[str(path.relative_to(ROOT))] = sha256_file(path)
    atomic_json(study / "artifact_manifest.json", {
        "phase": "COMMIT_A_PREREGISTERED",
        "base_commit": BASE_COMMIT,
        "formal_training_started": False,
        "formal_test_truth_access_count": 0,
        "files": dict(sorted(files.items())),
        "mutable_after_formal_start": list(MUTABLE_FORMAL_OUTPUTS),
        "self_hash_excluded": True,
        "commit_binding": "Git Commit A containing this manifest",
    })
    return {"status": "SEALED_COMMIT_A", "tests": totals, "manifest_files": len(files)}


def assert_formal_authorized(commit: str, study: Path = STUDY) -> str:
    """Require the exact committed Phase-1 state before any formal fit."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("show", "-s", "--format=%s", commit) != COMMIT_MESSAGE:
        raise RuntimeError("formal execution requires the exact sequential Commit A")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    if git("branch", "--show-current") != "exp/qgeognn-v2-4g-row-al":
        raise RuntimeError("formal execution requires the preregistered branch")
    manifest_path = study / "artifact_manifest.json"
    committed = json.loads(git("show", f"{commit}:{manifest_path.relative_to(ROOT)}"))
    current = json.loads(manifest_path.read_text())
    if committed != current or current.get("phase") != "COMMIT_A_PREREGISTERED":
        raise RuntimeError("artifact manifest differs from sequential Commit A")
    for relative, digest in current["files"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"Commit-A artifact drift: {relative}")
    if json.loads((study / "protocol.json").read_text()) != protocol_record():
        raise RuntimeError("code/data/protocol drift after Commit A")
    return git("rev-parse", commit)

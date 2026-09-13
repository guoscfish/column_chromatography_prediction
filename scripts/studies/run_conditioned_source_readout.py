#!/usr/bin/env python3
"""ROW-first conditioned source-readout transfer study.

Every fitting action is deliberately separated from outer-test scoring.  The
script is resumable by outer context so the preregistered inner screens can be
run on independent workers without changing the split or selection boundary.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.studies.run_filtered_full_data_benchmark import _read_authorized_truth
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.transfer import adaptation as a
from src.qgeognn_al.transfer.adaptive_readout import build_conditioned_readout_model
from src.qgeognn_al.transfer.source_augmentation import (
    SourcePredictionAndRepresentationFusion,
    SourcePredictionFusion,
    attach_source_features,
    build_or_load_source_cache,
)


STUDY = ROOT / "studies/transfer/conditioned_source_readout"
FROZEN = ROOT / "studies/transfer/filtered_full_data_benchmark"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
COLUMNS = ("25g", "40g")
PROTOCOLS = ("row", "compound")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
FEATURES = ("sample_id", "canonical_smiles", "PE/EA", "loading solvent", "Density g/ml", "V/ul", "Volume of loading solvent/ul")
TRAINING = {
    "optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 1e-5,
    "batch_size": 2048, "maximum_epochs": 500, "patience": 80,
    "bn_policy": "current", "weak_quantile_weight": 0.1,
}
LOSS_ARMS = {
    "L0": "raw_quantile",
    "L1": "endpoint_normalized_quantile",
    "L2": "q50_weak_quantile",
}
READOUT_ARMS = ("R0", "R1", "R2", "R3", "R4")

# The readout screen is deliberately a small state machine rather than an
# open-ended collection of arm names.  R3/R4 reuse the completed R0--R2 rows
# by staged append, but their launches and aggregations require a persisted,
# hash-verified predecessor decision.
INITIAL_READOUT_ARMS = ("R0", "R1", "R2")
R3_READOUT_ARMS = (*INITIAL_READOUT_ARMS, "R3")
R4_READOUT_ARMS = (*R3_READOUT_ARMS, "R4")
READOUT_STAGE_SPECS = {
    "initial": {
        "arms": INITIAL_READOUT_ARMS,
        "results": "INNER_SCREEN_INITIAL_RESULTS.csv",
        "decision": "INNER_SCREEN_INITIAL_DECISION.json",
    },
    "r3": {
        "arms": R3_READOUT_ARMS,
        "results": "INNER_SCREEN_R3_RESULTS.csv",
        "decision": "INNER_SCREEN_R3_DECISION.json",
    },
    "r4": {
        "arms": R4_READOUT_ARMS,
        "results": "INNER_SCREEN_R4_RESULTS.csv",
        "decision": "INNER_SCREEN_R4_DECISION.json",
    },
}
FINAL_READOUT_DECISION_NAME = "inner_screen_decision.json"
READOUT_SELECTION_BOUNDARY = "ROW inner GroupKFold only; no outer validation/test truth"

# This gate is applied only after the frozen ROW predictions have been scored.
# It is deliberately separate from the inner-CV architecture-selection gate:
# the latter chooses a candidate without outer endpoint truth, while this one
# decides whether the already-frozen candidate may proceed to COMPOUND.
ROW_CONTINUATION_DECISION_NAME = "ROW_CONTINUATION_DECISION.json"
ROW_CONTINUATION_RULE = {
    "mean_combined_gain_pct_min": 3.0,
    "seed_wins_min": 4,
    "seed_count": len(SEEDS),
    "endpoint_mean_regression_pct_max": 2.0,
    "comparison": "matched candidate versus P0 over the frozen ROW outer-test contexts",
}

# Promotion is deliberately a post-confirmation interpretation, not another
# selector.  These fixed thresholds implement the preregistered research-plan
# language while keeping the mandatory COMPOUND confirmation distinct from the
# ROW architecture/loss selection boundary.
PROMOTION_DECISION_NAME = "PROMOTION_DECISION.json"
PROMOTION_RULE = {
    "strong_row_mean_combined_gain_pct_min": 5.0,
    "strong_row_seed_wins_min": 4,
    "endpoint_mean_regression_pct_max": ROW_CONTINUATION_RULE["endpoint_mean_regression_pct_max"],
    "strong_compound_mean_combined_gain_pct_min": 0.0,
    "compound_target_mean_combined_gain_pct": 3.0,
    "promising_compound_noninferiority_loss_pct_max": 2.0,
    "comparison": "matched candidate versus P0 over the serialized frozen ROW and COMPOUND outer-test score tables",
}

LOSS_SCREEN_EVIDENCE_NAME = "LOSS_SCREEN.csv"
LOSS_SELECTION_DECISION_NAME = "loss_selection.json"
LOSS_SCREEN_REPORT_NAME = "LOSS_SCREEN_REPORT.md"
LOSS_SELECTION_TIE_TOLERANCE = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def deterministic_seed(*parts: object) -> int:
    return int(stable_hash([str(value) for value in parts])[:8], 16) & 0x7FFFFFFF


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_temporary_path(path: Path) -> Path:
    """Reserve a unique same-directory temporary path for an atomic replace.

    Independent outer-context workers all call ``prepare()`` and may write a
    shared manifest at the same time.  A fixed ``*.tmp`` name lets one worker
    replace or remove another worker's temporary file before its write finishes.
    ``mkstemp`` reserves a unique file in the target directory, retaining the
    same-filesystem guarantee needed by ``Path.replace``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(raw_path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    # The temporary name deliberately ends in `.tmp` for atomic replacement,
    # so pandas cannot infer compression from a final `.csv.gz` suffix.  Make
    # the compression contract explicit or the renamed artifact would be a
    # plain CSV masquerading as gzip and fail only at score time.
    compression = "gzip" if path.suffix == ".gz" else None
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_appended_frame(additions: pd.DataFrame, *, existing_columns: Sequence[str]) -> pd.DataFrame:
    """Require a staged arm's frame to have precisely the completed schema."""

    expected = list(existing_columns)
    if set(additions.columns) != set(expected):
        raise RuntimeError("new inner-screen rows do not match the completed CSV schema")
    return additions.loc[:, expected]


def _append_frame_without_rewriting_existing_rows(path: Path, additions: pd.DataFrame) -> None:
    """Atomically append rows while retaining the completed CSV bytes verbatim.

    A staged R3/R4 screen may extend a completed R0--R2 context.  Re-serializing
    the old frame could subtly change float formatting despite not retraining it,
    so copy the completed file and append only the newly fitted arm rows.
    """

    if additions.empty:
        return
    temporary = _atomic_temporary_path(path)
    try:
        shutil.copyfile(path, temporary)
        with path.open("rb") as original:
            original.seek(-1, os.SEEK_END)
            needs_newline = original.read(1) not in (b"\n", b"\r")
        with temporary.open("a", encoding="utf-8", newline="") as handle:
            if needs_newline:
                handle.write("\n")
            additions.to_csv(handle, index=False, header=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


INNER_APPEND_JOURNAL_NAME = ".inner_append_journal.json"
INNER_APPEND_SNAPSHOT_DIRECTORY = ".inner_append"


def _write_appended_snapshot(source: Path, additions: pd.DataFrame, destination: Path) -> None:
    """Create a full CSV postimage while retaining prior rows' exact bytes."""

    temporary = _atomic_temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        if not additions.empty:
            with source.open("rb") as original:
                original.seek(-1, os.SEEK_END)
                needs_newline = original.read(1) not in (b"\n", b"\r")
            with temporary.open("a", encoding="utf-8", newline="") as handle:
                if needs_newline:
                    handle.write("\n")
                additions.to_csv(handle, index=False, header=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _install_snapshot(snapshot: Path, target: Path) -> None:
    """Atomically install a journaled postimage without trusting live files."""

    temporary = _atomic_temporary_path(target)
    try:
        shutil.copyfile(snapshot, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _append_journal_path(run: Path) -> Path:
    return run / INNER_APPEND_JOURNAL_NAME


def _journal_snapshot_path(run: Path, record: Mapping[str, Any], *, label: str) -> tuple[Path, str]:
    raw_path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise RuntimeError(f"inner append journal has an invalid {label} snapshot record")
    snapshot_root = (run / INNER_APPEND_SNAPSHOT_DIRECTORY).resolve()
    path = (run / raw_path).resolve()
    if not path.is_relative_to(snapshot_root) or not path.exists() or sha(path) != digest:
        raise RuntimeError(f"inner append journal {label} snapshot is missing or changed")
    return path, digest


def _recover_pending_inner_append_unlocked(run: Path) -> dict[str, Path] | None:
    """Finish an interrupted staged append before validating its completion file."""

    journal_path = _append_journal_path(run)
    if not journal_path.exists():
        return None
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if not isinstance(journal, Mapping) or journal.get("schema_version") != 1 or journal.get("kind") != "INNER_APPEND":
        raise RuntimeError("inner append journal has an invalid schema")
    base = journal.get("base")
    snapshots = journal.get("snapshots")
    if not isinstance(base, Mapping) or not isinstance(snapshots, Mapping):
        raise RuntimeError("inner append journal is missing base/snapshot records")
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    complete_path = run / "context_complete.json"
    audit_path = run / "context_audit.json"
    base_hashes = {
        "summary": base.get("summary_sha256"),
        "history": base.get("history_sha256"),
        "completion": base.get("completion_sha256"),
        "audit": base.get("audit_sha256"),
    }
    if not all(isinstance(value, str) for value in base_hashes.values()):
        raise RuntimeError("inner append journal has invalid base hashes")
    if not audit_path.exists() or sha(audit_path) != base_hashes["audit"]:
        raise RuntimeError("inner append journal audit evidence changed")
    summary_snapshot, summary_digest = _journal_snapshot_path(run, snapshots.get("summary", {}), label="summary")
    history_snapshot, history_digest = _journal_snapshot_path(run, snapshots.get("history", {}), label="history")
    completion_snapshot, completion_digest = _journal_snapshot_path(run, snapshots.get("completion", {}), label="completion")
    completion_payload = json.loads(completion_snapshot.read_text(encoding="utf-8"))
    if (not isinstance(completion_payload, Mapping)
            or completion_payload.get("summary_sha256") != summary_digest
            or completion_payload.get("history_sha256") != history_digest):
        raise RuntimeError("inner append journal completion snapshot does not bind its postimages")

    def current_digest(path: Path) -> str | None:
        return sha(path) if path.exists() else None

    state = (
        current_digest(summary_path),
        current_digest(history_path),
        current_digest(complete_path),
    )
    allowed_states = {
        (base_hashes["summary"], base_hashes["history"], base_hashes["completion"]),
        (summary_digest, base_hashes["history"], base_hashes["completion"]),
        (summary_digest, history_digest, base_hashes["completion"]),
        (summary_digest, history_digest, completion_digest),
    }
    if state not in allowed_states:
        raise RuntimeError("inner append journal/live artifact state is inconsistent; refusing recovery")
    if state != (summary_digest, history_digest, completion_digest):
        _install_snapshot(summary_snapshot, summary_path)
        _install_snapshot(history_snapshot, history_path)
        _install_snapshot(completion_snapshot, complete_path)
    return {"journal": journal_path, "snapshot_directory": summary_snapshot.parent}


def _cleanup_recovered_inner_append(recovery: Mapping[str, Path]) -> None:
    """Remove only a fully validated transaction's private postimages."""

    recovery["journal"].unlink(missing_ok=True)
    shutil.rmtree(recovery["snapshot_directory"], ignore_errors=False)


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "), "torch": str(torch.__version__),
        "device": "cpu", "platform": platform.platform(),
    }


def git_sha() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                               capture_output=True, check=True)
    return completed.stdout.strip()


def _protocol() -> dict[str, Any]:
    parent = json.loads((FROZEN / "protocol.json").read_text(encoding="utf-8"))
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_scales = source_payload["preprocessing"].get("target_scales")
    if not isinstance(source_scales, dict):
        raise RuntimeError("qualified source checkpoint lacks source-train target scales")
    return {
        "study": "ROW_FIRST_CONDITIONED_SOURCE_READOUT",
        "git_sha_at_prepare": git_sha(),
        "classification": "DEVELOPMENTAL_CONFIRMATION__HISTORIC_OUTER_TEST_EXPOSED",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)), "source_checkpoint_sha256": sha(SOURCE),
        "source_model_variant": source_payload.get("model_variant"),
        "source_target_scales": {key: float(value) for key, value in source_scales.items()},
        "parent_protocol": str((FROZEN / "protocol.json").relative_to(ROOT)),
        "parent_protocol_sha256": sha(FROZEN / "protocol.json"),
        "split_manifest": str((FROZEN / "split_manifest.csv").relative_to(ROOT)),
        "split_manifest_sha256": sha(FROZEN / "split_manifest.csv"),
        "population_sha256": {column: sha(FROZEN / f"filtered_canonical_{column}.csv") for column in COLUMNS},
        "columns": list(COLUMNS), "outer_seeds": list(SEEDS), "outer_protocols": list(PROTOCOLS),
        "row_primary": True, "compound_secondary": True,
        "test_truth_used_for_fit_or_selection": False,
        "outer_validation_used_for_selection": False,
        "inner_cv": {"folds": 5, "splitter": "GroupKFold", "group": "canonical_smiles",
                     "scope": "outer gradient_train only"},
        "training": TRAINING,
        "loss_arms": LOSS_ARMS,
        "readout_arms": {
            "R0": "converged P0 architecture, no readout change",
            "R1": "P0 + learnable-global-query residual multi-head attention readout",
            "R2": "P0 + full audited-condition-query residual multi-head attention readout",
            "R3": "R2 + frozen source q50 residual fusion",
            "R4": "R3 + frozen source 128D representation residual fusion",
        },
        "readout": {"heads": 4, "hidden_dim": 128, "gate_initialization": 0.0,
                    "base": "global_add_pool(node embeddings) + existing condition residual"},
        "source_augmentation": {
            "source_prediction": "source eval q50 columns [1,4], divided only by source-train scales before fusion",
            "source_representation": "frozen QGeoGNNV2.extract_representation, pre-head, 128D",
            "cache": "sample_id + source checkpoint SHA256 + feature-table SHA256",
        },
        "trainable_scope": {
            "P0": ["backbone.convs.4.", "condition_branch.", "head."],
            "R1_R2": ["P0", "adaptive_readout."],
            "R3_R4": ["P0", "adaptive_readout.", "source_augmentation."],
        },
        "loss_selection_rule": (
            "A non-L0 recipe is selected only if it improves mean inner validation combined NRMSE "
            "by >=1% in both columns, wins >=3/5 outer-seed means and >=13/25 inner folds in both columns; "
            "otherwise retain L0. Ties retain the simpler L0."
        ),
        "architecture_continuation_rule": (
            "An arm has reasonable signal only if it improves mean inner validation combined NRMSE by >=3% "
            "versus R0 in both columns, wins >=3/5 outer-seed means and >=13/25 inner folds in both columns. "
            "R3 requires R2 signal; R4 requires R3 signal."
        ),
        "prediction_freeze": "all final blind validation/test predictions and checkpoints hash-locked before score action reads outer truth",
        "environment_at_prepare": _environment(),
    }


def prepare() -> dict[str, Any]:
    if not SOURCE.exists():
        raise RuntimeError(f"missing qualified source checkpoint: {SOURCE}")
    protocol = _protocol()
    path = STUDY / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # A study record intentionally preserves the code SHA at the moment it
        # was prepared. Later commits that add immutable artifacts must not
        # mutate that provenance field or make every resumable context fail.
        expected_static = dict(protocol); expected_static.pop("git_sha_at_prepare", None)
        existing_static = dict(existing); existing_static.pop("git_sha_at_prepare", None)
        if existing_static != expected_static:
            raise RuntimeError("conditioned-source-readout protocol drift; refusing to overwrite")
        protocol = existing
    else:
        write_json(path, protocol)
    schedule = pd.read_csv(FROZEN / "split_manifest.csv")
    rows = []
    for column in COLUMNS:
        for outer_protocol in PROTOCOLS:
            for seed in SEEDS:
                subset = schedule.loc[(schedule.column.eq(column)) & (schedule.protocol.eq(outer_protocol)) & (schedule.outer_seed.eq(seed))]
                roles = {role: subset.loc[subset.role.eq(role), "sample_id"].astype(str).tolist()
                         for role in ("gradient_train", "validation", "test")}
                if any(not values for values in roles.values()) or any(set(roles[left]) & set(roles[right]) for left in roles for right in roles if left != right):
                    raise RuntimeError(f"incomplete or overlapping frozen roles for {column}/{outer_protocol}/{seed}")
                rows.append({"column": column, "protocol": outer_protocol, "outer_seed": int(seed),
                             **{f"{role}_rows": len(values) for role, values in roles.items()},
                             **{f"{role}_ids_sha256": stable_hash(sorted(values)) for role, values in roles.items()}})
    write_frame(STUDY / "run_manifest.csv", pd.DataFrame(rows))
    return protocol


def _roles(column: str, outer_protocol: str, seed: int) -> dict[str, list[str]]:
    if column not in COLUMNS or outer_protocol not in PROTOCOLS or seed not in SEEDS:
        raise ValueError("context is outside the preregistered schedule")
    schedule = pd.read_csv(FROZEN / "split_manifest.csv")
    subset = schedule.loc[(schedule.column.eq(column)) & (schedule.protocol.eq(outer_protocol)) & (schedule.outer_seed.eq(seed)), ["sample_id", "role"]]
    result = {role: subset.loc[subset.role.eq(role), "sample_id"].astype(str).tolist()
              for role in ("gradient_train", "validation", "test")}
    if any(not result[role] for role in result) or any(set(result[x]) & set(result[y]) for x in result for y in result if x != y):
        raise RuntimeError("frozen role schedule is incomplete or overlapping")
    return result


def prepare_context(column: str, outer_protocol: str, seed: int) -> dict[str, Any]:
    """Create a label-scrubbed target context exposing gradient-train truth only."""

    protocol = prepare()
    roles = _roles(column, outer_protocol, seed)
    feature_path = FROZEN / f"filtered_features_{column}.csv"
    canonical_path = FROZEN / f"filtered_canonical_{column}.csv"
    if sha(canonical_path) != protocol["population_sha256"][column]:
        raise RuntimeError("filtered target population digest mismatch")
    feature = pd.read_csv(feature_path, usecols=FEATURES)
    feature.sample_id = feature.sample_id.astype(str)
    expected_ids = set().union(*(set(values) for values in roles.values()))
    if not feature.sample_id.is_unique or set(feature.sample_id) != expected_ids:
        raise RuntimeError("feature table and frozen split identities differ")
    # This is the sole endpoint read in fitting: the helper selects only the
    # gradient-train IDs. Validation and test labels remain zero in graph data.
    gradient_truth = _read_authorized_truth(canonical_path, roles["gradient_train"])
    frame = feature.copy()
    frame["V1_ml"] = 0.0
    frame["V2_ml"] = 0.0
    lookup = {sample_id: index for index, sample_id in enumerate(frame.sample_id)}
    for sample_id, endpoint in zip(roles["gradient_train"], gradient_truth):
        frame.loc[lookup[sample_id], ["V1_ml", "V2_ml"]] = endpoint
    cache = dict(torch.load(SOURCE_GRAPH_CACHE, map_location="cpu", weights_only=False))
    cache.update(torch.load(ROOT / f"studies/transfer/cross_column/data_audit/graph_cache_{column}_only.pt", map_location="cpu", weights_only=False))
    if set(frame.canonical_smiles.astype(str)) - set(cache):
        raise RuntimeError("molecular graph cache is incomplete for frozen target population")
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_preprocessing = source_payload["preprocessing"]
    source_scales = source_preprocessing.get("target_scales")
    if not isinstance(source_scales, Mapping):
        raise RuntimeError("qualified source preprocessing lacks target scales")
    atoms, angles = build_model_data(frame, cache, None, source_preprocessing["scaler"])
    positions = {role: [lookup[sample_id] for sample_id in ids] for role, ids in roles.items()}
    outer_scales = a.fit_target_scales(atoms, positions["gradient_train"])
    audit = {
        "column": column, "protocol": outer_protocol, "outer_seed": int(seed),
        "source_checkpoint_sha256": sha(SOURCE), "split_manifest_sha256": sha(FROZEN / "split_manifest.csv"),
        "population_sha256": sha(canonical_path), "feature_table_sha256": sha(feature_path),
        "roles": {role: {"count": len(ids), "ids_sha256": stable_hash(sorted(ids))} for role, ids in roles.items()},
        "outer_gradient_train_scales": outer_scales,
        "source_target_scales": {name: float(source_scales[name]) for name in ("V1", "V2")},
        "outer_validation_or_test_labels_loaded": False,
        "outer_test_truth_used_for_fit_or_selection": False,
    }
    return {"column": column, "protocol": outer_protocol, "seed": int(seed), "frame": frame, "roles": roles,
            "positions": positions, "atoms": atoms, "angles": angles, "outer_scales": outer_scales,
            "source_scales": audit["source_target_scales"], "audit": audit}


def _ensure_source_features(context: dict[str, Any]) -> dict[str, Any]:
    if context.get("source_cache") is not None:
        return context["source_cache"]
    source_model = load_predictor_checkpoint(SOURCE)
    cache = build_or_load_source_cache(
        STUDY / "source_cache", column=context["column"], sample_ids=context["frame"].sample_id.astype(str).tolist(),
        feature_table_sha256=context["audit"]["feature_table_sha256"], source_checkpoint_path=SOURCE,
        source_model=source_model, atom_data=context["atoms"], angle_data=context["angles"],
        source_target_scales=context["source_scales"], batch_size=TRAINING["batch_size"],
    )
    attach_source_features(context["atoms"], cache)
    context["source_cache"] = cache
    return cache


def prepare_source_cache(column: str) -> Path:
    """Materialize the immutable source cache before parallel R3/R4 workers."""

    context = prepare_context(column, "row", SEEDS[0])
    cache = _ensure_source_features(context)
    path = STUDY / "source_cache" / cache["source_checkpoint_sha256"] / f"{column}_source_features.pt"
    write_json(STUDY / "source_cache" / f"{column}_cache_manifest.json", {
        "path": str(path.relative_to(ROOT)), "sha256": sha(path), "column": column,
        "sample_ids_sha256": cache["sample_ids_sha256"], "feature_table_sha256": cache["feature_table_sha256"],
        "source_checkpoint_sha256": cache["source_checkpoint_sha256"],
        "representation_contract": cache["representation_contract"], "target_labels_read": False,
    })
    return path


def _arm_requires_source_features(arm: str) -> bool:
    return arm in ("R3", "R4")


def build_arm_model(arm: str, context: Mapping[str, Any], *, initialization_seed: int):
    """Build one source-initialized arm and declare its additional trainables."""

    if arm not in READOUT_ARMS:
        raise ValueError(f"unknown readout arm: {arm}")
    torch.manual_seed(int(initialization_seed))
    base = load_predictor_checkpoint(SOURCE)
    if arm == "R0":
        return base, ()
    source_augmentation = None
    if arm == "R3":
        source_augmentation = SourcePredictionFusion(dict(context["source_scales"]))
    elif arm == "R4":
        source_augmentation = SourcePredictionAndRepresentationFusion(dict(context["source_scales"]))
    kind = "adaptive" if arm == "R1" else "condition_query"
    model = build_conditioned_readout_model(base, kind=kind, heads=4, source_augmentation=source_augmentation)
    prefixes = ["adaptive_readout."]
    if source_augmentation is not None:
        prefixes.append("source_augmentation.")
    return model, tuple(prefixes)


def _fit_config(*, recipe: str, additional_prefixes: Sequence[str]) -> dict[str, Any]:
    if recipe not in a.LOSS_RECIPES:
        raise ValueError(f"unsupported loss recipe: {recipe}")
    return {**TRAINING, "loss_recipe": recipe, "additional_trainable_prefixes": tuple(additional_prefixes)}


def _blind_predict(model, atoms, angles, indices: Sequence[int]) -> np.ndarray:
    """Predict without ever touching graph labels for validation/test rows."""

    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*a.loader_pair(atoms, angles, indices, TRAINING["batch_size"])):
            output = model(atom_batch, angle_batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 2 or output.shape[1] != 6:
                raise RuntimeError("readout arm did not return six quantile outputs")
            outputs.append(output.detach().cpu().numpy())
    return np.vstack(outputs) if outputs else np.empty((0, 6), dtype=np.float32)


def _prediction_frame(sample_ids: Sequence[str], prediction: np.ndarray) -> pd.DataFrame:
    if prediction.shape != (len(sample_ids), 6):
        raise ValueError("blind prediction shape does not match sample IDs")
    return pd.DataFrame({"sample_id": list(sample_ids), "V1_q10": prediction[:, 0], "V1_q50": prediction[:, 1],
                         "V1_q90": prediction[:, 2], "V2_q10": prediction[:, 3], "V2_q50": prediction[:, 4],
                         "V2_q90": prediction[:, 5]})


def _context_dir(phase: str, outer_protocol: str, column: str, seed: int) -> Path:
    return STUDY / "runtime" / phase / outer_protocol / column / f"seed_{seed}"


@contextmanager
def _inner_context_lock(run: Path) -> Iterable[None]:
    """Serialize a context's inspect-fit-append transaction across workers."""

    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / ".inner_context.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _readout_state_lock() -> Iterable[None]:
    """Serialize immutable readout-stage evidence and decision commits."""

    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / ".readout_state.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _loss_selection_lock() -> Iterable[None]:
    """Serialize the immutable loss-screen evidence and decision commit."""

    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / ".loss_selection.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _promotion_decision_lock() -> Iterable[None]:
    """Serialize the terminal, immutable post-COMPOUND decision commit."""

    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / ".promotion_decision.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _score_artifact_lock() -> Iterable[None]:
    """Serialize the check-and-commit transaction for scored evidence."""

    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / ".score_artifact.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_completed_inner_context_unlocked(*, run: Path, phase: str, column: str, outer_protocol: str,
                                           seed: int, requested_arms: Sequence[str],
                                           requested_recipes: Mapping[str, str],
                                           protocol_sha256: str) -> dict[str, Any] | None:
    """Validate a completed context and identify arms that can safely be added.

    Completion artifacts are treated as immutable inputs.  A later staged call
    can add only arms absent from the completed contract; it cannot alter an
    arm's loss recipe, rerun an existing arm, or proceed from a damaged summary.
    """

    recovery = _recover_pending_inner_append_unlocked(run)
    complete = run / "context_complete.json"
    if not complete.exists():
        return None
    payload = json.loads(complete.read_text(encoding="utf-8"))
    expected_identity = {
        "protocol_sha256": protocol_sha256,
        "phase": phase,
        "column": column,
        "outer_protocol": outer_protocol,
        "outer_seed": int(seed),
        "outer_validation_or_test_labels_used": False,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"completed context identity mismatch for {key}: {run}")
    existing_arms = payload.get("arms")
    existing_recipes = payload.get("recipes")
    if (not isinstance(existing_arms, list) or not existing_arms or
            len(existing_arms) != len(set(existing_arms)) or
            not isinstance(existing_recipes, dict) or set(existing_recipes) != set(existing_arms)):
        raise RuntimeError(f"completed context has an invalid arm contract: {run}")
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    audit_path = run / "context_audit.json"
    if not (summary_path.exists() and history_path.exists() and audit_path.exists()):
        raise RuntimeError(f"completed context is missing a required immutable artifact: {run}")
    if payload.get("summary_sha256") != sha(summary_path) or payload.get("history_sha256") != sha(history_path):
        raise RuntimeError(f"completed context summary or history digest mismatch: {run}")
    summary = pd.read_csv(summary_path)
    history = pd.read_csv(history_path)
    required_summary = {"phase", "column", "protocol", "outer_seed", "inner_fold", "arm", "loss_recipe"}
    if not required_summary.issubset(summary.columns):
        raise RuntimeError(f"completed context summary schema is incomplete: {run}")
    if len(summary) != 5 * len(existing_arms) or set(summary.arm.astype(str)) != set(existing_arms):
        raise RuntimeError(f"completed context summary does not contain exactly five folds per arm: {run}")
    summary_identity = {
        "phase": phase,
        "column": column,
        "protocol": outer_protocol,
        "outer_seed": int(seed),
    }
    if any(set(summary[key].astype(type(expected))) != {expected} for key, expected in summary_identity.items()):
        raise RuntimeError(f"completed context summary identity does not match its directory: {run}")
    duplicate_key = summary.duplicated(["phase", "column", "protocol", "outer_seed", "inner_fold", "arm"])
    if bool(duplicate_key.any()):
        raise RuntimeError(f"completed context summary has duplicate arm/fold rows: {run}")
    for arm in existing_arms:
        rows = summary.loc[summary.arm.astype(str).eq(arm)]
        if (len(rows) != 5 or set(rows.inner_fold.astype(int)) != set(range(5)) or
                set(rows.loss_recipe.astype(str)) != {str(existing_recipes[arm])}):
            raise RuntimeError(f"completed context summary disagrees with its arm contract: {run}")
    if not history.empty and ("arm" not in history.columns or not set(history.arm.astype(str)).issubset(set(existing_arms))):
        raise RuntimeError(f"completed context history contains an unknown arm: {run}")
    overlap = set(existing_arms).intersection(requested_arms)
    mismatched = sorted(arm for arm in overlap if str(existing_recipes[arm]) != str(requested_recipes[arm]))
    if mismatched:
        raise RuntimeError(f"existing completed arms have different loss recipes: {mismatched}")
    missing_arms = tuple(arm for arm in requested_arms if arm not in set(existing_arms))
    result = {
        "payload": payload,
        "summary": summary,
        "history": history,
        "audit": json.loads(audit_path.read_text(encoding="utf-8")),
        "missing_arms": missing_arms,
    }
    if recovery is not None:
        _cleanup_recovered_inner_append(recovery)
    return result


def _load_completed_inner_context(*, run: Path, phase: str, column: str, outer_protocol: str,
                                  seed: int, requested_arms: Sequence[str],
                                  requested_recipes: Mapping[str, str],
                                  protocol_sha256: str) -> dict[str, Any] | None:
    """Read a completed context without observing a concurrent append."""

    if not run.exists():
        return None
    with _inner_context_lock(run):
        return _load_completed_inner_context_unlocked(
            run=run, phase=phase, column=column, outer_protocol=outer_protocol, seed=seed,
            requested_arms=requested_arms, requested_recipes=requested_recipes,
            protocol_sha256=protocol_sha256,
        )


def _commit_staged_inner_append_unlocked(*, run: Path, completed: Mapping[str, Any],
                                         additions_summary: pd.DataFrame, additions_history: pd.DataFrame,
                                         completion_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Write-ahead commit for the two CSV postimages and their completion pointer."""

    journal_path = _append_journal_path(run)
    if journal_path.exists():
        raise RuntimeError("pending inner append journal was not recovered before a new append")
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    complete_path = run / "context_complete.json"
    audit_path = run / "context_audit.json"
    snapshot_root = run / INNER_APPEND_SNAPSHOT_DIRECTORY
    snapshot_root.mkdir(parents=True, exist_ok=True)
    transaction = Path(tempfile.mkdtemp(prefix="transaction-", dir=snapshot_root))
    summary_snapshot = transaction / "summary.next.csv"
    history_snapshot = transaction / "history.next.csv"
    completion_snapshot = transaction / "context_complete.next.json"
    try:
        _write_appended_snapshot(summary_path, additions_summary, summary_snapshot)
        _write_appended_snapshot(history_path, additions_history, history_snapshot)
        expected_completion = {
            **dict(completion_identity),
            "summary_sha256": sha(summary_snapshot),
            "history_sha256": sha(history_snapshot),
        }
        write_json(completion_snapshot, expected_completion)
        journal = {
            "schema_version": 1,
            "kind": "INNER_APPEND",
            "base": {
                "summary_sha256": sha(summary_path),
                "history_sha256": sha(history_path),
                "completion_sha256": sha(complete_path),
                "audit_sha256": sha(audit_path),
            },
            "snapshots": {
                "summary": {"path": str(summary_snapshot.relative_to(run)), "sha256": sha(summary_snapshot)},
                "history": {"path": str(history_snapshot.relative_to(run)), "sha256": sha(history_snapshot)},
                "completion": {"path": str(completion_snapshot.relative_to(run)), "sha256": sha(completion_snapshot)},
            },
        }
        write_json(journal_path, journal)
        _install_snapshot(summary_snapshot, summary_path)
        _install_snapshot(history_snapshot, history_path)
        _install_snapshot(completion_snapshot, complete_path)
        return expected_completion
    except Exception:
        if not journal_path.exists():
            shutil.rmtree(transaction, ignore_errors=True)
        raise


def _run_inner_fit_context(*, phase: str, column: str, outer_protocol: str, seed: int,
                           arms: Sequence[str], recipes: Mapping[str, str]) -> Path:
    """Run all requested arms' five inner folds inside one outer context."""

    run = _context_dir(phase, outer_protocol, column, seed)
    requested_arms = tuple(arms)
    if not requested_arms or len(requested_arms) != len(set(requested_arms)):
        raise ValueError("inner-screen arms must be a non-empty, unique sequence")
    if set(recipes) != set(requested_arms):
        raise ValueError("inner-screen recipes must specify exactly one recipe per requested arm")
    requested_recipes = {arm: str(recipes[arm]) for arm in requested_arms}
    if any(recipe not in a.LOSS_RECIPES for recipe in requested_recipes.values()):
        raise ValueError("inner-screen recipes include an unsupported loss recipe")
    prepare()
    completed = _load_completed_inner_context_unlocked(
        run=run, phase=phase, column=column, outer_protocol=outer_protocol, seed=seed,
        requested_arms=requested_arms, requested_recipes=requested_recipes,
        protocol_sha256=sha(STUDY / "protocol.json"),
    )
    if completed is not None and not completed["missing_arms"]:
        return run
    arms_to_fit = requested_arms if completed is None else completed["missing_arms"]
    context = prepare_context(column, outer_protocol, seed)
    if completed is not None and completed["audit"] != context["audit"]:
        raise RuntimeError(f"completed context audit does not match the frozen context: {run}")
    if any(_arm_requires_source_features(arm) for arm in arms_to_fit):
        _ensure_source_features(context)
    positions = np.asarray(context["positions"]["gradient_train"], dtype=int)
    groups = context["frame"].canonical_smiles.astype(str).to_numpy()[positions]
    if len(set(groups)) < 5:
        raise RuntimeError("fewer than five canonical-smiles groups in outer gradient train")
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (local_train, local_valid) in enumerate(splitter.split(positions, groups=groups)):
        inner_train = positions[local_train].tolist()
        inner_valid = positions[local_valid].tolist()
        train_groups, valid_groups = groups[local_train].tolist(), groups[local_valid].tolist()
        if set(train_groups) & set(valid_groups):
            raise RuntimeError("canonical smiles leaked across inner GroupKFold roles")
        scales = a.fit_target_scales(context["atoms"], inner_train)
        for arm in arms_to_fit:
            recipe = requested_recipes[arm]
            initialized = deterministic_seed("conditioned-source-readout", phase, column, outer_protocol, seed, fold, arm)
            # L0/L1/L2 are loss-only P0 architecture controls; their artifact
            # labels remain L* while their model contract is R0/P0.
            architecture_arm = "R0" if arm in LOSS_ARMS else arm
            model, extras = build_arm_model(architecture_arm, context, initialization_seed=initialized)
            started = time.time()
            fit = a.train_target_adaptation(
                model, context["atoms"], context["angles"], inner_train, inner_valid,
                {"target_scales": scales}, mode="historical_shallow",
                seed=deterministic_seed("fit", phase, column, outer_protocol, seed, fold, arm),
                config=_fit_config(recipe=recipe, additional_prefixes=extras),
            )
            truth, prediction, _ = a.predict_point(model, context["atoms"], context["angles"], inner_valid,
                                                    batch_size=TRAINING["batch_size"])
            metrics = point_metrics(truth, prediction, scales)
            rows.append({
                "phase": phase, "column": column, "protocol": outer_protocol, "outer_seed": int(seed),
                "inner_fold": int(fold), "arm": arm, "loss_recipe": recipe,
                "n_inner_train": len(inner_train), "n_inner_validation": len(inner_valid),
                "inner_train_group_sha256": stable_hash(sorted(set(train_groups))),
                "inner_validation_group_sha256": stable_hash(sorted(set(valid_groups))),
                "inner_train_scales": json.dumps(scales, sort_keys=True),
                "best_epoch": int(fit.best_epoch), "epochs_run": int(fit.epochs_run),
                "validation_score": float(fit.validation_score), "runtime_seconds": float(time.time() - started),
                "trainable_parameter_count": int(fit.trainable_parameters), "total_parameter_count": int(fit.total_parameters),
                "all_outputs_finite": bool(metrics["all_outputs_finite"]), **metrics,
            })
            histories.extend({"phase": phase, "column": column, "protocol": outer_protocol, "outer_seed": int(seed),
                              "inner_fold": int(fold), "arm": arm, "loss_recipe": recipe, **item}
                             for item in fit.history)
    run.mkdir(parents=True, exist_ok=True)
    new_summary = pd.DataFrame(rows)
    new_history = pd.DataFrame(histories)
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    complete = run / "context_complete.json"
    if completed is None:
        write_frame(summary_path, new_summary)
        write_frame(history_path, new_history)
        write_json(run / "context_audit.json", context["audit"])
        completed_arms = list(requested_arms)
        completed_recipes = dict(requested_recipes)
        write_json(complete, {
            "protocol_sha256": sha(STUDY / "protocol.json"), "phase": phase, "column": column,
            "outer_protocol": outer_protocol, "outer_seed": int(seed), "arms": completed_arms, "recipes": completed_recipes,
            "summary_sha256": sha(summary_path), "history_sha256": sha(history_path),
            "outer_validation_or_test_labels_used": False,
        })
    else:
        # Build and journal both postimages before replacing either live CSV.
        # Recovery can therefore finish a process interrupted between files
        # without refitting or accepting a stale completion digest.
        appended_summary = _normalize_appended_frame(new_summary, existing_columns=completed["summary"].columns)
        appended_history = _normalize_appended_frame(new_history, existing_columns=completed["history"].columns)
        completed_arms = [*completed["payload"]["arms"], *arms_to_fit]
        completed_recipes = {**completed["payload"]["recipes"],
                             **{arm: requested_recipes[arm] for arm in arms_to_fit}}
        completion_identity = {
            "protocol_sha256": sha(STUDY / "protocol.json"), "phase": phase, "column": column,
            "outer_protocol": outer_protocol, "outer_seed": int(seed), "arms": completed_arms, "recipes": completed_recipes,
            "outer_validation_or_test_labels_used": False,
        }
        _commit_staged_inner_append_unlocked(
            run=run, completed=completed, additions_summary=appended_summary, additions_history=appended_history,
            completion_identity=completion_identity,
        )
        verified = _load_completed_inner_context_unlocked(
            run=run, phase=phase, column=column, outer_protocol=outer_protocol, seed=seed,
            requested_arms=tuple(completed_arms), requested_recipes=completed_recipes,
            protocol_sha256=sha(STUDY / "protocol.json"),
        )
        if verified is None or verified["missing_arms"]:
            raise RuntimeError("staged inner append did not validate its committed postimages")
    return run


def _inner_fit_context(*, phase: str, column: str, outer_protocol: str, seed: int,
                       arms: Sequence[str], recipes: Mapping[str, str]) -> Path:
    """Run one context under a cross-worker transaction lock."""

    requested_arms = tuple(arms)
    if not requested_arms or len(requested_arms) != len(set(requested_arms)):
        raise ValueError("inner-screen arms must be a non-empty, unique sequence")
    if set(recipes) != set(requested_arms):
        raise ValueError("inner-screen recipes must specify exactly one recipe per requested arm")
    requested_recipes = {arm: str(recipes[arm]) for arm in requested_arms}
    if any(recipe not in a.LOSS_RECIPES for recipe in requested_recipes.values()):
        raise ValueError("inner-screen recipes include an unsupported loss recipe")
    run = _context_dir(phase, outer_protocol, column, seed)
    with _inner_context_lock(run):
        return _run_inner_fit_context(
            phase=phase, column=column, outer_protocol=outer_protocol, seed=seed,
            arms=requested_arms, recipes=requested_recipes,
        )


def run_loss_context(column: str, seed: int) -> Path:
    # Once a global loss decision exists, all ten source contexts are immutable
    # selection evidence.  Recheck the complete frozen decision before allowing
    # an idempotent resume, rather than silently fitting a newly missing arm.
    if (STUDY / LOSS_SELECTION_DECISION_NAME).exists():
        _selected_loss()
    return _inner_fit_context(phase="loss_screen", column=column, outer_protocol="row", seed=seed,
                              arms=tuple(LOSS_ARMS), recipes=LOSS_ARMS)


def _selected_loss() -> dict[str, Any]:
    return _freeze_loss_selection()


def _readout_stage_for_context(arms: Sequence[str]) -> str:
    requested = tuple(arms)
    if requested == INITIAL_READOUT_ARMS:
        return "initial"
    if requested == ("R3",):
        return "r3"
    if requested == ("R4",):
        return "r4"
    raise ValueError(
        "readout context arms must be exactly R0 R1 R2, then staged R3, then staged R4"
    )


def _readout_stage_for_aggregate(arms: Sequence[str]) -> str:
    requested = tuple(arms)
    for stage, specification in READOUT_STAGE_SPECS.items():
        if requested == tuple(specification["arms"]):
            return stage
    raise ValueError(
        "readout aggregation arms must be exactly R0 R1 R2, R0 R1 R2 R3, or R0 R1 R2 R3 R4"
    )


def _readout_stage_path(stage: str, artifact: str) -> Path:
    try:
        name = str(READOUT_STAGE_SPECS[stage][artifact])
    except KeyError as error:
        raise ValueError(f"unknown readout stage/artifact: {stage}/{artifact}") from error
    return STUDY / name


def _readout_stage_reference(stage: str) -> dict[str, str]:
    path = _readout_stage_path(stage, "decision")
    if not path.exists():
        raise RuntimeError(f"missing persisted {stage} readout decision")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def _sorted_screen_values(values: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    expected_arms = tuple(arms)
    expected_count = len(COLUMNS) * len(SEEDS) * 5 * len(expected_arms)
    subset = values.loc[values.arm.astype(str).isin(expected_arms)].copy()
    if len(subset) != expected_count or set(subset.arm.astype(str)) != set(expected_arms):
        raise RuntimeError("readout stage evidence does not contain the exact preregistered arm schedule")
    key = ["column", "outer_seed", "inner_fold", "arm"]
    if subset.duplicated(key).any():
        raise RuntimeError("readout stage evidence has duplicate context/arm rows")
    return subset.sort_values(key).reset_index(drop=True)


def _assert_same_screen_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, path: Path) -> None:
    if list(actual.columns) != list(expected.columns):
        raise RuntimeError(f"readout evidence schema drift: {path}")
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(f"readout evidence no longer matches hash-verified completed contexts: {path}") from error


def _persist_screen_evidence(path: Path, values: pd.DataFrame) -> str:
    if path.exists():
        _assert_same_screen_frame(pd.read_csv(path), values, path=path)
    else:
        write_frame(path, values)
    return sha(path)


def _readout_summary(values: pd.DataFrame, arms: Sequence[str], stage: str) -> tuple[list[dict[str, Any]], dict[str, bool], str | None, str | None]:
    base = values.loc[values.arm.astype(str).eq("R0"), ["column", "outer_seed", "inner_fold", "validation_score"]]
    base = base.rename(columns={"validation_score": "R0_score"})
    rows: list[dict[str, Any]] = []
    arm_summary: dict[str, list[dict[str, Any]]] = {}
    for arm in tuple(arms)[1:]:
        current = values.loc[values.arm.astype(str).eq(arm), ["column", "outer_seed", "inner_fold", "validation_score"]]
        merged = base.merge(current, on=["column", "outer_seed", "inner_fold"], validate="one_to_one")
        if len(merged) != len(COLUMNS) * len(SEEDS) * 5:
            raise RuntimeError(f"readout screen cannot pair R0 with {arm}")
        merged["relative_improvement_pct"] = 100 * (merged.R0_score - merged.validation_score) / merged.R0_score
        per_arm: list[dict[str, Any]] = []
        for column, group in merged.groupby("column", sort=True):
            seed_mean = group.groupby("outer_seed", as_index=False).relative_improvement_pct.mean()
            record = {
                "arm": arm,
                "column": str(column),
                "mean_relative_improvement_pct": float(group.relative_improvement_pct.mean()),
                "std_relative_improvement_pct": float(group.relative_improvement_pct.std(ddof=1)),
                "fold_wins": int((group.relative_improvement_pct > 0).sum()),
                "seed_wins": int((seed_mean.relative_improvement_pct > 0).sum()),
                "n_folds": int(len(group)),
                "n_seeds": int(len(seed_mean)),
            }
            rows.append(record)
            per_arm.append(record)
        arm_summary[arm] = per_arm
    gates = {
        arm: bool(
            len(records) == len(COLUMNS)
            and (pd.DataFrame(records).mean_relative_improvement_pct >= 3.0).all()
            and (pd.DataFrame(records).seed_wins >= 3).all()
            and (pd.DataFrame(records).fold_wins >= 13).all()
        )
        for arm, records in arm_summary.items()
    }
    if stage == "initial":
        next_arm = "R3" if gates.get("R2", False) else None
    elif stage == "r3":
        next_arm = "R4" if gates.get("R3", False) else None
    elif stage == "r4":
        next_arm = None
    else:
        raise ValueError(f"unknown readout stage: {stage}")
    selected = None
    if next_arm is None:
        eligible = [arm for arm in tuple(arms)[1:] if gates.get(arm, False)]
        if eligible:
            # Ties are resolved toward the simpler arm, independent of CLI order.
            complexity = {arm: index for index, arm in enumerate(READOUT_ARMS)}
            selected = max(
                eligible,
                key=lambda arm: (
                    float(pd.DataFrame(arm_summary[arm]).mean_relative_improvement_pct.mean()),
                    -complexity[arm],
                ),
            )
    return rows, gates, next_arm, selected


def _stage_parent_references(stage: str) -> dict[str, dict[str, str]]:
    if stage == "initial":
        return {}
    initial = _verify_readout_stage("initial")
    if not initial["gates"].get("R2", False):
        raise RuntimeError("R3 is blocked: the persisted initial R2 gate did not pass")
    references = {"initial": _readout_stage_reference("initial")}
    if stage == "r3":
        return references
    r3 = _verify_readout_stage("r3")
    if not r3["gates"].get("R3", False):
        raise RuntimeError("R4 is blocked: the persisted R3 gate did not pass")
    references["r3"] = _readout_stage_reference("r3")
    if stage == "r4":
        return references
    raise ValueError(f"unknown readout stage: {stage}")


def _readout_stage_payload(stage: str, values: pd.DataFrame, selected_loss: Mapping[str, Any],
                           evidence_sha256: str, parents: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    specification = READOUT_STAGE_SPECS[stage]
    arms = tuple(specification["arms"])
    summary, gates, next_arm, candidate = _readout_summary(values, arms, stage)
    loss_path = STUDY / "loss_selection.json"
    protocol = json.loads((STUDY / "protocol.json").read_text(encoding="utf-8"))
    evidence_path = _readout_stage_path(stage, "results")
    return {
        "schema_version": 1,
        "decision_kind": "READOUT_STAGE",
        "stage": stage,
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "loss_selection": {
            "path": str(loss_path.relative_to(ROOT)),
            "sha256": sha(loss_path),
            "selected_loss_arm": str(selected_loss["selected_loss_arm"]),
            "selected_loss_recipe": str(selected_loss["selected_loss_recipe"]),
        },
        "parents": dict(parents),
        "screened_arms": list(arms),
        "summary": summary,
        "gates": gates,
        "next_arm": next_arm,
        "selected_candidate_if_screen_stops": candidate,
        "rule": protocol["architecture_continuation_rule"],
        "selection_boundary": READOUT_SELECTION_BOUNDARY,
        "evidence": {
            "path": str(evidence_path.relative_to(ROOT)),
            "sha256": evidence_sha256,
        },
    }


def _verify_readout_stage(stage: str) -> dict[str, Any]:
    if stage not in READOUT_STAGE_SPECS:
        raise ValueError(f"unknown readout stage: {stage}")
    decision_path = _readout_stage_path(stage, "decision")
    evidence_path = _readout_stage_path(stage, "results")
    if not decision_path.exists() or not evidence_path.exists():
        raise RuntimeError(f"missing persisted {stage} readout decision/evidence")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    parents = _stage_parent_references(stage)
    selected_loss = _selected_loss()
    arms = tuple(READOUT_STAGE_SPECS[stage]["arms"])
    values = _collect_inner(
        "inner_screen", "row", arms,
        {arm: selected_loss["selected_loss_recipe"] for arm in arms},
    )
    stage_values = _sorted_screen_values(values, arms)
    _assert_same_screen_frame(pd.read_csv(evidence_path), stage_values, path=evidence_path)
    evidence_sha256 = sha(evidence_path)
    expected = _readout_stage_payload(stage, stage_values, selected_loss, evidence_sha256, parents)
    if decision != expected:
        raise RuntimeError(f"persisted {stage} readout decision is mutable, unhashed, or inconsistent with its evidence")
    return decision


def _final_readout_payload(stage: str, stage_decision: Mapping[str, Any]) -> dict[str, Any]:
    stage_reference = _readout_stage_reference(stage)
    return {
        "schema_version": 1,
        "decision_kind": "FINAL_READOUT_CANDIDATE",
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "stage": stage,
        "stage_decision": stage_reference,
        "loss_selection": dict(stage_decision["loss_selection"]),
        "screened_arms": list(stage_decision["screened_arms"]),
        "summary": list(stage_decision["summary"]),
        "gates": dict(stage_decision["gates"]),
        "next_arm": stage_decision["next_arm"],
        "selected_candidate_if_screen_stops": stage_decision["selected_candidate_if_screen_stops"],
        "selection_boundary": stage_decision["selection_boundary"],
    }


def _persist_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"{label} drift; refusing to overwrite immutable evidence")
        return existing
    write_json(path, dict(payload))
    return dict(payload)


def _persist_immutable_text(path: Path, content: str, *, label: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"{label} drift; refusing to overwrite immutable evidence")
        return
    temporary = _atomic_temporary_path(path)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_final_readout_decision(stage: str, stage_decision: Mapping[str, Any]) -> dict[str, Any]:
    path = STUDY / FINAL_READOUT_DECISION_NAME
    return _persist_immutable_json(path, _final_readout_payload(stage, stage_decision), label="final readout decision")


def _verified_final_readout_decision() -> dict[str, Any]:
    path = STUDY / FINAL_READOUT_DECISION_NAME
    if not path.exists():
        raise RuntimeError("candidate provenance is missing the terminal ROW readout decision")
    decision = json.loads(path.read_text(encoding="utf-8"))
    stage = decision.get("stage")
    if stage not in READOUT_STAGE_SPECS:
        raise RuntimeError("candidate provenance has an invalid terminal readout stage")
    stage_decision = _verify_readout_stage(str(stage))
    expected = _final_readout_payload(str(stage), stage_decision)
    if decision != expected:
        raise RuntimeError("terminal ROW readout decision is mutable, unhashed, or inconsistent with its stage evidence")
    return decision


def run_readout_context(column: str, seed: int, arms: Sequence[str]) -> Path:
    stage = _readout_stage_for_context(arms)
    if stage == "initial":
        # Once persisted, an initial decision can only be rechecked/idempotently
        # resumed; it cannot be replaced by a fresh arm subset.
        if _readout_stage_path("initial", "decision").exists():
            _verify_readout_stage("initial")
    else:
        if (STUDY / FINAL_READOUT_DECISION_NAME).exists():
            _verified_final_readout_decision()
            raise RuntimeError("readout screen is terminal; no later arm may be appended")
        _stage_parent_references(stage)
    selected = _selected_loss()
    recipe = selected["selected_loss_recipe"]
    return _inner_fit_context(phase="inner_screen", column=column, outer_protocol="row", seed=seed,
                              arms=tuple(arms), recipes={arm: recipe for arm in arms})


def _collect_inner(phase: str, outer_protocol: str, required_arms: Sequence[str],
                   recipes: Mapping[str, str]) -> pd.DataFrame:
    """Collect only hash-verified completed inner contexts.

    The summary CSV is a selection input, so it is never trusted merely because
    it exists.  `context_complete.json` commits its identity and CSV digests;
    the same validator used for staged R3/R4 append protects loss/readout
    aggregation from a later edit or mismatched directory.
    """

    expected_arms = tuple(required_arms)
    expected_recipes = {arm: str(recipes[arm]) for arm in expected_arms}
    if set(recipes) != set(expected_arms):
        raise ValueError("inner collector recipes must match required arms exactly")
    frames: list[pd.DataFrame] = []
    missing = []
    for column in COLUMNS:
        for seed in SEEDS:
            run = _context_dir(phase, outer_protocol, column, seed)
            completed = _load_completed_inner_context(
                run=run, phase=phase, column=column, outer_protocol=outer_protocol, seed=seed,
                requested_arms=expected_arms, requested_recipes=expected_recipes,
                protocol_sha256=sha(STUDY / "protocol.json"),
            )
            if completed is None:
                missing.append(str((run / "context_complete.json").relative_to(ROOT)))
                continue
            frame = completed["summary"]
            if completed["missing_arms"]:
                missing.append(f"{(run / 'inner_cv_summary.csv').relative_to(ROOT)} missing arms {sorted(completed['missing_arms'])}")
            frames.append(frame)
    if missing:
        raise RuntimeError("incomplete preregistered inner screen:\n" + "\n".join(missing))
    result = pd.concat(frames, ignore_index=True)
    expected = len(COLUMNS) * len(SEEDS) * 5 * len(expected_arms)
    if len(result.loc[result.arm.isin(expected_arms)]) != expected:
        raise RuntimeError("inner screen row count does not equal frozen 2x5x5 schedule")
    return result


def _sorted_loss_screen_values(values: pd.DataFrame) -> pd.DataFrame:
    """Return the exact L0/L1/L2 selection schedule in a canonical order."""

    expected_arms = tuple(LOSS_ARMS)
    expected_count = len(COLUMNS) * len(SEEDS) * 5 * len(expected_arms)
    subset = values.loc[values.arm.astype(str).isin(expected_arms)].copy()
    key = ["column", "outer_seed", "inner_fold", "arm"]
    if (len(subset) != expected_count or set(subset.arm.astype(str)) != set(expected_arms)
            or bool(subset.duplicated(key).any())):
        raise RuntimeError("loss screen evidence does not contain the exact preregistered arm schedule")
    return subset.sort_values(key).reset_index(drop=True)


def _assert_same_loss_screen_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, path: Path) -> None:
    if list(actual.columns) != list(expected.columns):
        raise RuntimeError(f"loss screen evidence schema drift: {path}")
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(
            f"loss screen evidence no longer matches hash-verified completed contexts: {path}"
        ) from error


def _persist_loss_screen_evidence(path: Path, values: pd.DataFrame, *, selection_exists: bool) -> str:
    """Create the aggregate once, or prove that its existing bytes still agree."""

    if path.exists():
        _assert_same_loss_screen_frame(pd.read_csv(path), values, path=path)
    elif selection_exists:
        raise RuntimeError("frozen loss selection is missing its immutable LOSS_SCREEN.csv evidence")
    else:
        write_frame(path, values)
    return sha(path)


def _loss_screen_summary(values: pd.DataFrame) -> tuple[list[dict[str, Any]], list[tuple[float, str]], str]:
    """Apply the preregistered global L0/L1/L2 choice without outer truth."""

    summary_rows: list[dict[str, Any]] = []
    base = values.loc[
        values.arm.astype(str).eq("L0"),
        ["column", "outer_seed", "inner_fold", "validation_score"],
    ].rename(columns={"validation_score": "L0_score"})
    for arm in ("L1", "L2"):
        current = values.loc[
            values.arm.astype(str).eq(arm),
            ["column", "outer_seed", "inner_fold", "validation_score"],
        ]
        merged = base.merge(current, on=["column", "outer_seed", "inner_fold"], validate="one_to_one")
        if len(merged) != len(COLUMNS) * len(SEEDS) * 5:
            raise RuntimeError(f"loss screen cannot pair L0 with {arm}")
        if (merged.L0_score == 0).any():
            raise RuntimeError("loss screen cannot compute relative improvement against a zero L0 score")
        merged["relative_improvement_pct"] = 100 * (merged.L0_score - merged.validation_score) / merged.L0_score
        for column, group in merged.groupby("column", sort=True):
            seed_mean = group.groupby("outer_seed", as_index=False).relative_improvement_pct.mean()
            summary_rows.append({
                "arm": arm,
                "column": str(column),
                "mean_relative_improvement_pct": float(group.relative_improvement_pct.mean()),
                "std_relative_improvement_pct": float(group.relative_improvement_pct.std(ddof=1)),
                "fold_wins": int((group.relative_improvement_pct > 0).sum()),
                "seed_wins": int((seed_mean.relative_improvement_pct > 0).sum()),
                "n_folds": int(len(group)),
                "n_seeds": int(len(seed_mean)),
            })
    summary = pd.DataFrame(summary_rows)
    eligible: list[tuple[float, str]] = []
    for arm in ("L1", "L2"):
        subset = summary.loc[summary.arm.astype(str).eq(arm)]
        stable = bool(
            len(subset) == len(COLUMNS)
            and (subset.mean_relative_improvement_pct >= 1.0).all()
            and (subset.seed_wins >= 3).all()
            and (subset.fold_wins >= 13).all()
        )
        if stable:
            eligible.append((float(subset.mean_relative_improvement_pct.mean()), arm))
    if not eligible:
        # Equal-to-baseline (or otherwise insufficient) alternatives are not
        # eligible, so the protocol's "ties retain the simpler L0" rule is
        # explicit rather than an incidental consequence of ``max``.
        selected_arm = "L0"
    else:
        best_gain = max(gain for gain, _ in eligible)
        best_arms = [
            arm for gain, arm in eligible
            if math.isclose(gain, best_gain, rel_tol=0.0, abs_tol=LOSS_SELECTION_TIE_TOLERANCE)
        ]
        # The frozen protocol says ties retain the simpler baseline L0.  This
        # also avoids treating the lexical order of arm labels as evidence.
        selected_arm = best_arms[0] if len(best_arms) == 1 else "L0"
    return summary_rows, eligible, selected_arm


def _loss_screen_source_contexts() -> list[dict[str, Any]]:
    """Bind a frozen selection to every validated context and its audit bytes."""

    protocol_sha256 = sha(STUDY / "protocol.json")
    sources: list[dict[str, Any]] = []
    for column in COLUMNS:
        for seed in SEEDS:
            run = _context_dir("loss_screen", "row", column, seed)
            completed = _load_completed_inner_context(
                run=run, phase="loss_screen", column=column, outer_protocol="row", seed=seed,
                requested_arms=tuple(LOSS_ARMS), requested_recipes=LOSS_ARMS,
                protocol_sha256=protocol_sha256,
            )
            if completed is None or completed["missing_arms"]:
                raise RuntimeError(f"loss selection source context is incomplete: {run}")
            complete_path = run / "context_complete.json"
            summary_path = run / "inner_cv_summary.csv"
            history_path = run / "inner_cv_history.csv"
            audit_path = run / "context_audit.json"
            sources.append({
                "column": column,
                "outer_seed": int(seed),
                "context_complete": {"path": str(complete_path.relative_to(ROOT)), "sha256": sha(complete_path)},
                "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": sha(summary_path)},
                "history": {"path": str(history_path.relative_to(ROOT)), "sha256": sha(history_path)},
                "audit": {"path": str(audit_path.relative_to(ROOT)), "sha256": sha(audit_path)},
            })
    return sources


def _loss_selection_payload(values: pd.DataFrame, *, evidence_sha256: str,
                            source_contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the one global loss decision from persisted protocol evidence only."""

    protocol_path = STUDY / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rule = protocol.get("loss_selection_rule")
    if not isinstance(rule, str) or not rule:
        raise RuntimeError("persisted protocol lacks the loss-selection rule")
    summary, eligible, selected_arm = _loss_screen_summary(values)
    evidence_path = STUDY / LOSS_SCREEN_EVIDENCE_NAME
    return {
        "schema_version": 1,
        "decision_kind": "LOSS_SELECTION",
        "protocol_sha256": sha(protocol_path),
        "selection_boundary": READOUT_SELECTION_BOUNDARY,
        "screened_arms": list(LOSS_ARMS),
        "recipes": dict(LOSS_ARMS),
        "source_contexts": [dict(record) for record in source_contexts],
        "evidence": {
            "path": str(evidence_path.relative_to(ROOT)),
            "sha256": evidence_sha256,
        },
        "summary": summary,
        "rule": rule,
        "eligible_nonbaseline": [arm for _, arm in eligible],
        "selected_loss_arm": selected_arm,
        "selected_loss_recipe": LOSS_ARMS[selected_arm],
    }


def _loss_screen_report_text(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Loss screen report",
        "",
        "The three fixed loss recipes were evaluated only in 5-fold canonical-SMILES GroupKFold inside each ROW outer gradient-train context. No outer validation/test endpoint was available to the screen.",
        "",
        f"Selected global recipe: **{decision['selected_loss_arm']} / `{decision['selected_loss_recipe']}`**.",
        "",
        "| arm | column | mean relative improvement (%) | fold wins | seed wins |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in decision["summary"]:
        lines.append(
            f"| {row['arm']} | {row['column']} | {row['mean_relative_improvement_pct']:.3f} | "
            f"{row['fold_wins']}/25 | {row['seed_wins']}/5 |"
        )
    lines.extend([
        "",
        "A non-L0 recipe required >=1% mean improvement in both columns, >=13/25 fold wins and >=3/5 seed wins in both columns. Otherwise the simpler raw P0 recipe remains selected.",
        "",
    ])
    return "\n".join(lines)


def _freeze_loss_selection_unlocked() -> dict[str, Any]:
    """Recompute and verify the immutable loss decision under its commit lock."""

    values = _sorted_loss_screen_values(
        _collect_inner("loss_screen", "row", tuple(LOSS_ARMS), LOSS_ARMS)
    )
    decision_path = STUDY / LOSS_SELECTION_DECISION_NAME
    evidence_sha256 = _persist_loss_screen_evidence(
        STUDY / LOSS_SCREEN_EVIDENCE_NAME, values, selection_exists=decision_path.exists(),
    )
    expected = _loss_selection_payload(
        values, evidence_sha256=evidence_sha256, source_contexts=_loss_screen_source_contexts(),
    )
    decision = _persist_immutable_json(decision_path, expected, label="loss selection")
    _persist_immutable_text(
        STUDY / LOSS_SCREEN_REPORT_NAME, _loss_screen_report_text(decision), label="loss screen report",
    )
    return decision


def _freeze_loss_selection() -> dict[str, Any]:
    with _loss_selection_lock():
        return _freeze_loss_selection_unlocked()


def aggregate_loss_screen() -> dict[str, Any]:
    return _freeze_loss_selection()


def _aggregate_readout_screen_unlocked(arms: Sequence[str]) -> dict[str, Any]:
    stage = _readout_stage_for_aggregate(arms)
    decision_path = _readout_stage_path(stage, "decision")
    if decision_path.exists():
        decision = _verify_readout_stage(stage)
        # A worker can be interrupted after the stage evidence/decision commit
        # but before the terminal pointer commit below.  Re-running the same
        # aggregation is therefore a safe recovery action, never a reselection.
        if decision["next_arm"] is None:
            _persist_final_readout_decision(stage, decision)
        return decision
    if (STUDY / FINAL_READOUT_DECISION_NAME).exists():
        _verified_final_readout_decision()
        raise RuntimeError("readout screen is terminal; refusing to aggregate another stage")
    parents = _stage_parent_references(stage)
    selected = _selected_loss()
    required_arms = tuple(READOUT_STAGE_SPECS[stage]["arms"])
    values = _collect_inner(
        "inner_screen", "row", required_arms,
        {arm: selected["selected_loss_recipe"] for arm in required_arms},
    )
    stage_values = _sorted_screen_values(values, required_arms)
    evidence_path = _readout_stage_path(stage, "results")
    evidence_sha256 = _persist_screen_evidence(evidence_path, stage_values)
    # This required study artifact is a readable latest-stage rollup. Selection
    # provenance instead points to the immutable stage-specific evidence above.
    write_frame(STUDY / "INNER_SCREEN_RESULTS.csv", stage_values)
    decision = _readout_stage_payload(stage, stage_values, selected, evidence_sha256, parents)
    decision = _persist_immutable_json(decision_path, decision, label=f"{stage} readout decision")
    if decision["next_arm"] is None:
        _persist_final_readout_decision(stage, decision)
    return decision


def aggregate_readout_screen(arms: Sequence[str]) -> dict[str, Any]:
    """Aggregate one state-machine stage under a study-level commit lock."""

    with _readout_state_lock():
        return _aggregate_readout_screen_unlocked(arms)


def _median_epoch(summary: pd.DataFrame, arm: str) -> int:
    values = summary.loc[summary.arm.eq(arm), "best_epoch"].to_numpy(float)
    if len(values) != 5:
        raise RuntimeError(f"expected five inner folds to select final epoch for {arm}")
    return int(math.floor(float(np.median(values)) + 0.5))


def _formal_selection(column: str, outer_protocol: str, seed: int, arm: str, recipe: str,
                      *, phase: str) -> int:
    selection_arm = "L0" if phase == "loss_screen" and arm == "R0" else arm
    run = _context_dir(phase, outer_protocol, column, seed)
    completed = _load_completed_inner_context(
        run=run, phase=phase, column=column, outer_protocol=outer_protocol, seed=seed,
        requested_arms=(selection_arm,), requested_recipes={selection_arm: recipe},
        protocol_sha256=sha(STUDY / "protocol.json"),
    )
    if completed is None or completed["missing_arms"]:
        raise RuntimeError(f"missing verified inner selection context: {run}")
    summary = completed["summary"]
    expected = summary.loc[summary.arm.eq(selection_arm), "loss_recipe"].unique().tolist()
    if expected != [recipe]:
        raise RuntimeError("selected final recipe does not match frozen inner-screen arm")
    return _median_epoch(summary, selection_arm)


def run_compound_selection_context(column: str, seed: int, candidate: str) -> Path:
    if candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("compound confirmation requires an architecture candidate R1-R4")
    provenance = _selected_candidate_provenance(candidate)
    _assert_row_continuation_authorized(candidate)
    recipe = str(provenance["loss_selection"]["selected_loss_recipe"])
    recipes = {"R0": recipe, candidate: recipe}
    return _inner_fit_context(phase="compound_selection", column=column, outer_protocol="compound", seed=seed,
                              arms=("R0", candidate), recipes=recipes)


def _final_dir(outer_protocol: str, column: str, seed: int, arm: str) -> Path:
    return STUDY / "runtime" / "formal" / outer_protocol / column / f"seed_{seed}" / arm


def _selected_candidate_provenance(candidate: str) -> dict[str, Any]:
    """Return the hash-locked ROW inner-screen decision for one candidate.

    Formal scoring is permitted only for the single candidate selected after
    the gated ROW screen has stopped.  This check deliberately reads no target
    endpoint and is shared by prediction freeze and score actions.
    """

    if candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("candidate provenance requires an R1-R4 arm")
    path = STUDY / FINAL_READOUT_DECISION_NAME
    decision = _verified_final_readout_decision()
    if decision.get("selection_boundary") != READOUT_SELECTION_BOUNDARY:
        raise RuntimeError("candidate provenance has an invalid selection boundary")
    if decision.get("next_arm") is not None:
        raise RuntimeError("candidate provenance is incomplete: the ROW screen authorized a later arm")
    if decision.get("selected_candidate_if_screen_stops") != candidate:
        raise RuntimeError("candidate does not match the frozen ROW inner-screen selection")
    gates = decision.get("gates")
    if not isinstance(gates, Mapping) or gates.get(candidate) is not True:
        raise RuntimeError("candidate did not pass its frozen ROW inner-screen gate")
    screened = decision.get("screened_arms")
    if not isinstance(screened, list) or "R0" not in screened or candidate not in screened:
        raise RuntimeError("candidate provenance lacks the matched R0/candidate screen")
    loss = decision.get("loss_selection")
    if not isinstance(loss, Mapping) or loss.get("selected_loss_recipe") not in a.LOSS_RECIPES:
        raise RuntimeError("candidate provenance lacks a hash-verified global loss recipe")
    return {
        "candidate": candidate,
        "inner_screen_decision": str(path.relative_to(ROOT)),
        "inner_screen_decision_sha256": sha(path),
        "stage": str(decision["stage"]),
        "stage_decision": dict(decision["stage_decision"]),
        "loss_selection": dict(loss),
    }


BLIND_PREDICTION_COLUMNS = ("sample_id", "V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90")
FINAL_FREEZE_FILENAMES = ("final.pt", "validation_predictions_blind.csv.gz", "test_predictions_blind.csv.gz")


def _expected_final_freeze_files(outer_protocol: str, arms: Sequence[str]) -> set[str]:
    return {
        str((_final_dir(outer_protocol, column, seed, arm) / filename).relative_to(ROOT))
        for column in COLUMNS for seed in SEEDS for arm in arms for filename in FINAL_FREEZE_FILENAMES
    }


def _validate_blind_prediction_file(path: Path, expected_ids: Sequence[str], *, role: str) -> None:
    """Validate the labels-free prediction artifact before committing its hash."""

    if not path.exists():
        raise RuntimeError(f"missing final {role} prediction before freeze: {path}")
    frame = pd.read_csv(path)
    missing = set(BLIND_PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"final {role} prediction has missing columns {sorted(missing)}: {path}")
    actual_ids = frame.sample_id.astype(str).tolist()
    required_ids = [str(value) for value in expected_ids]
    if len(required_ids) != len(set(required_ids)) or actual_ids != required_ids:
        raise RuntimeError(f"final {role} predictions do not match the frozen role order: {path}")
    values = frame.loc[:, list(BLIND_PREDICTION_COLUMNS[1:])].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"final {role} predictions contain non-finite values: {path}")


def _final_context_provenance(outer_protocol: str, arm: str, recipe: str,
                              selection_phase: str) -> dict[str, Any]:
    """Derive the only permitted final-fit contract from the frozen selection."""

    if outer_protocol not in PROTOCOLS:
        raise ValueError("final context requires a preregistered outer protocol")
    provenance = _selected_candidate_provenance(
        # R0 is paired with the selected candidate; all other arms must be it.
        _verified_final_readout_decision()["selected_candidate_if_screen_stops"]
        if arm == "R0" else arm
    )
    candidate = str(provenance["candidate"])
    if arm not in ("R0", candidate):
        raise RuntimeError("final context arm is not the frozen matched P0/candidate pair")
    expected_recipe = str(provenance["loss_selection"]["selected_loss_recipe"])
    expected_phase = "inner_screen" if outer_protocol == "row" else "compound_selection"
    if recipe != expected_recipe:
        raise RuntimeError("final context loss recipe does not match the hash-verified global loss selection")
    if selection_phase != expected_phase:
        raise RuntimeError("final context selection phase does not match the frozen protocol")
    if outer_protocol == "compound":
        _assert_row_continuation_authorized(candidate)
    return provenance


def _verify_final_fit_completion(*, column: str, outer_protocol: str, seed: int, arm: str,
                                 recipe: str, selection_phase: str,
                                 candidate_provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an existing final refit before resuming or freezing it."""

    run = _final_dir(outer_protocol, column, seed, arm)
    complete = run / "final_fit_complete.json"
    final_pt = run / "final.pt"
    validation = run / "validation_predictions_blind.csv.gz"
    test = run / "test_predictions_blind.csv.gz"
    if not complete.exists():
        raise RuntimeError(f"missing final fit completion record before validation: {run}")
    payload = json.loads(complete.read_text(encoding="utf-8"))
    identity = {
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "selection_phase": selection_phase,
        "outer_protocol": outer_protocol,
        "column": column,
        "outer_seed": int(seed),
        "arm": arm,
        "loss_recipe": recipe,
        "candidate_provenance": dict(candidate_provenance),
        "outer_validation_or_test_labels_used": False,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"final fit completion identity/label-use assertion failed: {run}")
    selected_epoch = _formal_selection(column, outer_protocol, seed, arm, recipe, phase=selection_phase)
    if payload.get("selected_epoch") != selected_epoch:
        raise RuntimeError(f"final fit completion epoch does not match frozen inner selection: {run}")
    expected_hashes = {
        "final_pt_sha256": final_pt,
        "validation_prediction_sha256": validation,
        "test_prediction_sha256": test,
    }
    for field, artifact in expected_hashes.items():
        if not artifact.exists() or payload.get(field) != sha(artifact):
            raise RuntimeError(f"final fit completion digest assertion failed for {field}: {run}")
    roles = _roles(column, outer_protocol, seed)
    _validate_blind_prediction_file(validation, roles["validation"], role="validation")
    _validate_blind_prediction_file(test, roles["test"], role="test")
    return payload


def run_final_context(column: str, outer_protocol: str, seed: int, arm: str, recipe: str, *, selection_phase: str) -> Path:
    """Fixed-epoch final refit and blind prediction freeze for one arm/context."""

    provenance = _final_context_provenance(outer_protocol, arm, recipe, selection_phase)
    run = _final_dir(outer_protocol, column, seed, arm)
    complete = run / "final_fit_complete.json"
    if complete.exists():
        _verify_final_fit_completion(
            column=column, outer_protocol=outer_protocol, seed=seed, arm=arm,
            recipe=recipe, selection_phase=selection_phase, candidate_provenance=provenance,
        )
        return run
    context = prepare_context(column, outer_protocol, seed)
    if _arm_requires_source_features(arm):
        _ensure_source_features(context)
    selected_epoch = _formal_selection(column, outer_protocol, seed, arm, recipe, phase=selection_phase)
    model, extras = build_arm_model(arm, context, initialization_seed=deterministic_seed("formal-init", outer_protocol, column, seed, arm))
    started = time.time()
    fit = a.train_target_adaptation_fixed_epochs(
        model, context["atoms"], context["angles"], context["positions"]["gradient_train"], selected_epoch,
        mode="historical_shallow", seed=deterministic_seed("formal-fit", outer_protocol, column, seed, arm),
        config={**_fit_config(recipe=recipe, additional_prefixes=extras), "target_scales": context["outer_scales"]},
    )
    validation = _blind_predict(model, context["atoms"], context["angles"], context["positions"]["validation"])
    test = _blind_predict(model, context["atoms"], context["angles"], context["positions"]["test"])
    run.mkdir(parents=True, exist_ok=True)
    write_frame(run / "validation_predictions_blind.csv.gz", _prediction_frame(context["roles"]["validation"], validation))
    write_frame(run / "test_predictions_blind.csv.gz", _prediction_frame(context["roles"]["test"], test))
    torch.save({"model_state_dict": model.state_dict(), "arm": arm, "loss_recipe": recipe,
                "selected_epoch": selected_epoch, "source_checkpoint_sha256": sha(SOURCE),
                "split_manifest_sha256": sha(FROZEN / "split_manifest.csv"), "outer_validation_or_test_used_for_selection": False},
               run / "final.pt")
    source_cache_audit = None
    if context.get("source_cache") is not None:
        cache_path = STUDY / "source_cache" / context["source_cache"]["source_checkpoint_sha256"] / f"{column}_source_features.pt"
        source_cache_audit = {"path": str(cache_path.relative_to(ROOT)), "sha256": sha(cache_path),
                              "sample_ids_sha256": context["source_cache"]["sample_ids_sha256"]}
    write_json(run / "final_fit_audit.json", {**context["audit"], "arm": arm, "loss_recipe": recipe,
                "selected_epoch": selected_epoch, "training": _fit_config(recipe=recipe, additional_prefixes=extras),
                "trainable_parameter_count": int(fit.trainable_parameters), "total_parameter_count": int(fit.total_parameters),
                "runtime_seconds": float(time.time() - started), "environment": _environment(),
                "git_sha": git_sha(), "source_cache": source_cache_audit, "candidate_provenance": provenance,
                "outer_validation_or_test_labels_used": False})
    write_json(complete, {"protocol_sha256": sha(STUDY / "protocol.json"), "selection_phase": selection_phase,
                "outer_protocol": outer_protocol, "column": column, "outer_seed": int(seed), "arm": arm,
                "loss_recipe": recipe, "selected_epoch": selected_epoch, "final_pt_sha256": sha(run / "final.pt"),
                "validation_prediction_sha256": sha(run / "validation_predictions_blind.csv.gz"),
                "test_prediction_sha256": sha(run / "test_predictions_blind.csv.gz"),
                "candidate_provenance": provenance,
                "outer_validation_or_test_labels_used": False})
    return run


def finalize_prediction_freeze(outer_protocol: str, arms: Sequence[str]) -> dict[str, Any]:
    frozen_arms = tuple(arms)
    if outer_protocol not in PROTOCOLS:
        raise ValueError("prediction freeze requires a preregistered outer protocol")
    if len(frozen_arms) != 2 or frozen_arms[0] != "R0" or frozen_arms[1] not in READOUT_ARMS or frozen_arms[1] == "R0":
        raise ValueError("prediction freeze requires arms in the exact order: R0 then the selected R1-R4 candidate")
    if outer_protocol == "compound":
        _assert_row_continuation_authorized(frozen_arms[1])
    candidate_provenance = _selected_candidate_provenance(frozen_arms[1])
    expected_recipe = str(candidate_provenance["loss_selection"]["selected_loss_recipe"])
    expected_selection_phase = "inner_screen" if outer_protocol == "row" else "compound_selection"
    protocol_sha256 = sha(STUDY / "protocol.json")
    files: dict[str, str] = {}
    for column in COLUMNS:
        for seed in SEEDS:
            for arm in frozen_arms:
                run = _final_dir(outer_protocol, column, seed, arm)
                final_pt = run / "final.pt"
                validation = run / "validation_predictions_blind.csv.gz"
                test = run / "test_predictions_blind.csv.gz"
                _verify_final_fit_completion(
                    column=column, outer_protocol=outer_protocol, seed=seed, arm=arm,
                    recipe=expected_recipe, selection_phase=expected_selection_phase,
                    candidate_provenance=candidate_provenance,
                )
                for artifact in (final_pt, validation, test):
                    files[str(artifact.relative_to(ROOT))] = sha(artifact)
    if set(files) != _expected_final_freeze_files(outer_protocol, frozen_arms):
        raise RuntimeError("final prediction freeze did not enumerate the complete expected artifact set")
    payload = {"protocol_sha256": protocol_sha256, "outer_protocol": outer_protocol,
               "arms": list(frozen_arms), "candidate_provenance": candidate_provenance,
               "files": files, "global_outer_truth_read_before_freeze": False}
    hashes_path = STUDY / "prediction_hashes.json"
    merged = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.exists() else {}
    if "stopping_decision" in merged:
        raise RuntimeError("prediction freeze is blocked by the terminal no-score stopping decision")
    existing = merged.get(outer_protocol)
    if existing is not None and existing != payload:
        raise RuntimeError(f"{outer_protocol} prediction-freeze hash entry drift; refusing to overwrite immutable evidence")
    manifest_path = STUDY / f"{outer_protocol.upper()}_PREDICTION_FREEZE_MANIFEST.json"
    _persist_immutable_json(manifest_path, payload, label=f"{outer_protocol} prediction-freeze manifest")
    if existing is None:
        merged[outer_protocol] = payload
        write_json(hashes_path, merged)
    return payload


def _assert_frozen_for_score(outer_protocol: str, arms: Sequence[str]) -> None:
    frozen_arms = tuple(arms)
    path = STUDY / f"{outer_protocol.upper()}_PREDICTION_FREEZE_MANIFEST.json"
    if not path.exists():
        raise RuntimeError("score refused: no global prediction-freeze manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    hashes_path = STUDY / "prediction_hashes.json"
    if not hashes_path.exists() or json.loads(hashes_path.read_text(encoding="utf-8")).get(outer_protocol) != manifest:
        raise RuntimeError("score refused: prediction-freeze manifest is not mirrored by the immutable hash record")
    if manifest.get("protocol_sha256") != sha(STUDY / "protocol.json"):
        raise RuntimeError("score refused: freeze manifest has a different study protocol")
    if manifest.get("outer_protocol") != outer_protocol or manifest.get("arms") != list(frozen_arms):
        raise RuntimeError("score refused: freeze manifest has different protocol or arms")
    if len(frozen_arms) != 2 or frozen_arms[0] != "R0":
        raise RuntimeError("score refused: freeze manifest does not contain matched P0 and one candidate")
    if manifest.get("candidate_provenance") != _selected_candidate_provenance(frozen_arms[1]):
        raise RuntimeError("score refused: freeze manifest candidate provenance changed")
    expected_files = _expected_final_freeze_files(outer_protocol, frozen_arms)
    if set(manifest.get("files", {})) != expected_files:
        raise RuntimeError("score refused: incomplete global blind prediction freeze")
    for relative, digest in manifest["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError("score refused: frozen final artifact digest changed")


def _final_prediction(column: str, outer_protocol: str, seed: int, arm: str, test_ids: Sequence[str]) -> np.ndarray:
    path = _final_dir(outer_protocol, column, seed, arm) / "test_predictions_blind.csv.gz"
    frame = pd.read_csv(path)
    if frame.sample_id.astype(str).tolist() != list(test_ids):
        raise RuntimeError(f"new {arm} predictions do not match frozen test order: {path}")
    return frame[["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"]].to_numpy(float)


def _historical_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing historical {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"historical {label} is not a JSON object: {path}")
    return payload


def _frozen_role_hashes(column: str, outer_protocol: str, seed: int) -> dict[str, str]:
    roles = _roles(column, outer_protocol, seed)
    return {role: stable_hash(sorted(ids)) for role, ids in roles.items()}


def _verify_full_data_reference_freeze(path: Path, *, column: str, outer_protocol: str, seed: int) -> None:
    """Verify the two-level freeze chain for paper/M3 reference predictions."""

    study = ROOT / "studies/transfer/full_data_baseline_finalization"
    manifest_path = study / "prediction_freeze_manifest.json"
    manifest = _historical_json(manifest_path, label="full-data prediction-freeze manifest")
    frozen_path = path.parent / "frozen.json"
    try:
        frozen_relative = str(frozen_path.relative_to(study))
    except ValueError as error:
        raise RuntimeError(f"full-data reference path is outside its frozen study: {path}") from error
    manifest_files = manifest.get("files")
    expected_frozen_digest = manifest_files.get(frozen_relative) if isinstance(manifest_files, Mapping) else None
    if not isinstance(expected_frozen_digest, str) or not frozen_path.exists() or sha(frozen_path) != expected_frozen_digest:
        raise RuntimeError(f"full-data reference freeze digest mismatch: {frozen_path}")
    frozen = _historical_json(frozen_path, label="full-data context freeze")
    contract = frozen.get("contract")
    role_hashes = _frozen_role_hashes(column, outer_protocol, seed)
    expected_contract = {
        "column": column,
        "protocol": outer_protocol,
        "seed": int(seed),
        "train_ids_hash": role_hashes["gradient_train"],
        "validation_ids_hash": role_hashes["validation"],
        "test_ids_hash": role_hashes["test"],
        "test_labels_used_for_fit_normalization_or_selection": 0,
        "donor_outer_test_rows_used": 0,
    }
    if not isinstance(contract, Mapping) or any(contract.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError(f"full-data reference freeze contract mismatch: {frozen_path}")
    frozen_files = frozen.get("files")
    expected_prediction_digest = frozen_files.get(path.name) if isinstance(frozen_files, Mapping) else None
    if not isinstance(expected_prediction_digest, str) or not path.exists() or sha(path) != expected_prediction_digest:
        raise RuntimeError(f"full-data reference prediction digest mismatch: {path}")


def _verify_hier_reference_freeze(path: Path, *, column: str, outer_protocol: str, seed: int) -> None:
    """Verify the direct prediction digest in the corrected-HIER freeze record."""

    study = ROOT / "studies/transfer/hier_cw_semantic_repair"
    manifest = _historical_json(study / "prediction_freeze_manifest.json", label="corrected-HIER prediction-freeze manifest")
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError as error:
        raise RuntimeError(f"corrected-HIER reference path is outside the repository: {path}") from error
    files = manifest.get("files")
    expected_digest = files.get(relative) if isinstance(files, Mapping) else None
    if not isinstance(expected_digest, str) or not path.exists() or sha(path) != expected_digest:
        raise RuntimeError(f"corrected-HIER reference prediction digest mismatch: {path}")
    # The digest is the primary proof; checking the context location makes a
    # copied prediction file unable to masquerade as another seed/role.
    expected_tail = Path("runtime") / "contexts" / column / outer_protocol / f"seed_{seed}" / "predictions_blind.csv.gz"
    if not str(path).endswith(str(expected_tail)):
        raise RuntimeError(f"corrected-HIER reference context mismatch: {path}")
    audit_path = path.with_name("fit_audit.json")
    audit_relative = str(audit_path.relative_to(ROOT))
    audit_digest = files.get(audit_relative) if isinstance(files, Mapping) else None
    if not isinstance(audit_digest, str) or not audit_path.exists() or sha(audit_path) != audit_digest:
        raise RuntimeError(f"corrected-HIER fit audit digest mismatch: {audit_path}")
    audit = _historical_json(audit_path, label="corrected-HIER fit audit")
    combined_gradient_train_ids = [
        sample_id
        for focal_column in COLUMNS
        for sample_id in _roles(focal_column, outer_protocol, seed)["gradient_train"]
    ]
    expected_audit = {
        "column": column,
        "protocol": outer_protocol,
        "seed": int(seed),
        "train_ids_sha256": stable_hash(sorted(combined_gradient_train_ids)),
        "test_ids_sha256": _frozen_role_hashes(column, outer_protocol, seed)["test"],
        "outer_validation_used": False,
        "test_truth_used": False,
    }
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise RuntimeError(f"corrected-HIER fit-role contract mismatch: {audit_path}")


def _align_historical_prediction_ids(frame: pd.DataFrame, test_ids: Sequence[str], *, path: Path,
                                     reference: str) -> pd.DataFrame:
    """Require an exact ID set, then put historical rows in current test order."""

    if "sample_id" not in frame.columns:
        raise RuntimeError(f"historical {reference} lacks sample_id: {path}")
    requested = [str(value) for value in test_ids]
    if not requested or len(requested) != len(set(requested)):
        raise RuntimeError("frozen test IDs must be non-empty and unique")
    aligned = frame.copy()
    aligned["sample_id"] = aligned.sample_id.astype(str)
    actual = aligned.sample_id.tolist()
    if len(actual) != len(requested) or len(actual) != len(set(actual)) or set(actual) != set(requested):
        raise RuntimeError(f"historical {reference} IDs do not match frozen test population: {path}")
    # Uniqueness was asserted above; avoid pandas' deprecated `verify_integrity`
    # flag while retaining a deterministic ID-indexed reorder.
    return aligned.set_index("sample_id", drop=False).loc[requested].reset_index(drop=True)


def _reference_prediction(column: str, outer_protocol: str, seed: int, reference: str,
                          test_ids: Sequence[str]) -> np.ndarray:
    """Read a frozen, ID-aligned historical baseline without its metrics."""

    context = f"contexts/{column}/{outer_protocol}/seed_{seed}"
    if reference in ("paper_style_current_v2", "M3_CENTER_WIDTH_FULL"):
        path = ROOT / "studies/transfer/full_data_baseline_finalization/runtime" / context / "predictions_blind.csv.gz"
        _verify_full_data_reference_freeze(path, column=column, outer_protocol=outer_protocol, seed=seed)
    elif reference == "HIER_CW_SHARED_LAMBDA_CORRECTED":
        path = ROOT / "studies/transfer/hier_cw_semantic_repair/runtime" / context / "predictions_blind.csv.gz"
        _verify_hier_reference_freeze(path, column=column, outer_protocol=outer_protocol, seed=seed)
    else:
        raise ValueError(f"unknown historical reference: {reference}")
    frame = _align_historical_prediction_ids(pd.read_csv(path), test_ids, path=path, reference=reference)
    if reference == "paper_style_current_v2":
        columns = ["paper_style_current_v2_V1", "paper_style_current_v2_V2"]
    else:
        columns = [f"{reference}_V1", f"{reference}_V2"]
    if set(columns) - set(frame.columns):
        raise RuntimeError(f"historical {reference} prediction columns are incomplete: {path}")
    point = frame.loc[:, columns].to_numpy(float)
    if not np.isfinite(point).all():
        raise RuntimeError(f"historical {reference} predictions contain non-finite values: {path}")
    # Structured/paper references are point estimators. Repeat q50 into the
    # six-column layout solely for common point-metric bookkeeping.
    return np.column_stack([point[:, 0], point[:, 0], point[:, 0], point[:, 1], point[:, 1], point[:, 1]])


def _metric_values(truth: np.ndarray, prediction: np.ndarray, scales: Mapping[str, float]) -> dict[str, float | bool]:
    metrics = point_metrics(truth, prediction, scales)
    return {key: (float(value) if isinstance(value, (float, np.floating)) else bool(value)) for key, value in metrics.items()}


def _relative_change_pct(candidate: float, reference: float) -> float:
    """Return candidate-minus-reference percent change with an explicit zero rule."""

    if not (math.isfinite(candidate) and math.isfinite(reference)):
        raise RuntimeError("ROW continuation gate received a non-finite metric")
    if reference > 0:
        return float(100.0 * (candidate - reference) / reference)
    if candidate == reference:
        return 0.0
    return math.inf if candidate > reference else -math.inf


def _mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = values.to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("ROW continuation gate received a non-finite metric")
    return float(numeric.mean()), float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def evaluate_row_continuation(candidate: str, result: pd.DataFrame) -> dict[str, Any]:
    """Apply the fixed formal ROW continuation gate to matched P0 scores.

    This function consumes only a completed ROW score table.  It is never used
    for model/loss/epoch selection and does not access endpoint truth itself.
    """

    if candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("ROW continuation requires a selected non-R0 candidate")
    required = {"protocol", "column", "outer_seed", "method", "combined_normalized_rmse",
                "V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "all_outputs_finite"}
    if missing := sorted(required.difference(result.columns)):
        raise RuntimeError("ROW result table is missing continuation-gate columns: " + ", ".join(missing))
    row = result.loc[result.protocol.astype(str).eq("row")].copy()
    if row.empty:
        raise RuntimeError("ROW continuation requires completed ROW scores")
    columns: dict[str, dict[str, Any]] = {}
    endpoint_passes: list[bool] = []
    combined_passes: list[bool] = []
    seed_win_passes: list[bool] = []
    endpoint_metrics = ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae")
    expected_seeds = set(SEEDS)
    for column in COLUMNS:
        subset = row.loc[row.column.astype(str).eq(column)]
        p0 = subset.loc[subset.method.astype(str).eq("P0")].copy()
        current = subset.loc[subset.method.astype(str).eq(candidate)].copy()
        for method, frame in (("P0", p0), (candidate, current)):
            if len(frame) != len(SEEDS) or set(frame.outer_seed.astype(int)) != expected_seeds:
                raise RuntimeError(f"ROW continuation requires one {method} result for every frozen seed in {column}")
            if frame.duplicated("outer_seed").any():
                raise RuntimeError(f"ROW continuation found duplicate {method} seed rows in {column}")
            if not frame.all_outputs_finite.eq(True).all():
                raise RuntimeError(f"ROW continuation cannot use non-finite {method} outputs in {column}")
        paired = p0.merge(current, on="outer_seed", suffixes=("_p0", "_candidate"), validate="one_to_one")
        p0_combined_mean, p0_combined_std = _mean_std(paired["combined_normalized_rmse_p0"])
        candidate_combined_mean, candidate_combined_std = _mean_std(paired["combined_normalized_rmse_candidate"])
        mean_gain_pct = -_relative_change_pct(candidate_combined_mean, p0_combined_mean)
        seed_details = []
        for value in paired.sort_values("outer_seed").itertuples(index=False):
            p0_value = float(getattr(value, "combined_normalized_rmse_p0"))
            candidate_value = float(getattr(value, "combined_normalized_rmse_candidate"))
            gain_pct = -_relative_change_pct(candidate_value, p0_value)
            seed_details.append({
                "outer_seed": int(value.outer_seed), "p0_combined_nrmse": p0_value,
                "candidate_combined_nrmse": candidate_value, "gain_pct": gain_pct,
                "candidate_win": bool(candidate_value < p0_value),
            })
        seed_wins = int(sum(item["candidate_win"] for item in seed_details))
        combined_pass = bool(mean_gain_pct >= ROW_CONTINUATION_RULE["mean_combined_gain_pct_min"])
        seed_wins_pass = bool(seed_wins >= ROW_CONTINUATION_RULE["seed_wins_min"])
        combined_passes.append(combined_pass)
        seed_win_passes.append(seed_wins_pass)
        endpoint_summary: dict[str, dict[str, float | bool]] = {}
        for metric in endpoint_metrics:
            p0_mean, p0_std = _mean_std(paired[f"{metric}_p0"])
            candidate_mean, candidate_std = _mean_std(paired[f"{metric}_candidate"])
            regression_pct = _relative_change_pct(candidate_mean, p0_mean)
            endpoint_pass = bool(regression_pct <= ROW_CONTINUATION_RULE["endpoint_mean_regression_pct_max"])
            endpoint_passes.append(endpoint_pass)
            endpoint_summary[metric] = {
                "p0_mean": p0_mean, "p0_std": p0_std,
                "candidate_mean": candidate_mean, "candidate_std": candidate_std,
                "mean_relative_regression_pct": regression_pct,
                "within_regression_limit": endpoint_pass,
            }
        columns[column] = {
            "n_seeds": len(SEEDS),
            "combined_nrmse": {
                "p0_mean": p0_combined_mean, "p0_std": p0_combined_std,
                "candidate_mean": candidate_combined_mean, "candidate_std": candidate_combined_std,
                "mean_gain_pct": mean_gain_pct, "seed_wins": seed_wins,
                "mean_gain_pass": combined_pass, "seed_wins_pass": seed_wins_pass,
                "seed_details": seed_details,
            },
            "endpoint_metrics": endpoint_summary,
        }
    criteria = {
        "mean_combined_gain_both_columns": bool(all(combined_passes)),
        "seed_wins_both_columns": bool(all(seed_win_passes)),
        "no_endpoint_mean_rmse_or_mae_regression_above_limit": bool(all(endpoint_passes)),
    }
    authorized = bool(all(criteria.values()))
    return {
        "schema_version": 1,
        "protocol": "row",
        "candidate": candidate,
        "reference": "P0",
        "rule": ROW_CONTINUATION_RULE,
        "columns": columns,
        "criteria": criteria,
        "status": "ROW_CONTINUATION_PASSED" if authorized else "ROW_CONTINUATION_FAILED",
        "compound_confirmation_authorized": authorized,
        "decision_boundary": "Frozen ROW outer-test scores only; no score selected architecture, loss, source feature, or epoch.",
    }


def _row_prediction_freeze_reference(candidate: str) -> dict[str, str]:
    """Return the immutable ROW freeze that authorized this score table."""

    _assert_frozen_for_score("row", ("R0", candidate))
    path = STUDY / "ROW_PREDICTION_FREEZE_MANIFEST.json"
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def _persist_row_continuation_decision(candidate: str, _result: pd.DataFrame) -> dict[str, Any]:
    """Commit the formal ROW gate outcome after the ROW score table is written."""

    path = STUDY / ROW_CONTINUATION_DECISION_NAME
    row_path = STUDY / "ROW_RESULTS.csv"
    if not row_path.exists():
        raise RuntimeError("cannot persist a ROW continuation decision before ROW_RESULTS.csv exists")
    # Gate evidence must be exactly the serialized, hash-addressed table rather
    # than an in-memory frame whose CSV round-trip could differ in dtype/format.
    decision = evaluate_row_continuation(candidate, pd.read_csv(row_path))
    decision["row_results_sha256"] = sha(row_path)
    decision["row_prediction_freeze_manifest"] = _row_prediction_freeze_reference(candidate)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != decision:
            raise RuntimeError("ROW continuation decision drift; refusing to replace the formal gate record")
    else:
        write_json(path, decision)
    return decision


def _load_row_continuation_decision(candidate: str) -> dict[str, Any]:
    """Load and integrity-check the sole authorization record for COMPOUND."""

    path = STUDY / ROW_CONTINUATION_DECISION_NAME
    row_path = STUDY / "ROW_RESULTS.csv"
    if not path.exists() or not row_path.exists():
        raise RuntimeError("COMPOUND is blocked until the formal ROW continuation decision is recorded")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("candidate") != candidate or decision.get("protocol") != "row" or decision.get("reference") != "P0":
        raise RuntimeError("COMPOUND is blocked by a ROW continuation decision for a different candidate")
    if decision.get("row_prediction_freeze_manifest") != _row_prediction_freeze_reference(candidate):
        raise RuntimeError("COMPOUND is blocked because the ROW prediction-freeze evidence changed")
    if decision.get("row_results_sha256") != sha(row_path):
        raise RuntimeError("COMPOUND is blocked because the ROW continuation evidence changed")
    expected = evaluate_row_continuation(candidate, pd.read_csv(row_path))
    expected["row_results_sha256"] = sha(row_path)
    expected["row_prediction_freeze_manifest"] = _row_prediction_freeze_reference(candidate)
    if decision != expected:
        raise RuntimeError("COMPOUND is blocked because the ROW continuation decision does not match its evidence")
    return decision


def _assert_row_continuation_authorized(candidate: str) -> dict[str, Any]:
    decision = _load_row_continuation_decision(candidate)
    if decision.get("status") != "ROW_CONTINUATION_PASSED" or not decision.get("compound_confirmation_authorized"):
        raise RuntimeError("COMPOUND is blocked: the formal ROW continuation gate did not pass")
    return decision


def _artifact_reference(path: Path) -> dict[str, str]:
    """Return a path/digest pair for an immutable score-bound input."""

    if not path.exists():
        raise RuntimeError(f"missing promotion-decision input: {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def _serialized_score_summary(outer_protocol: str, candidate: str, path: Path) -> dict[str, Any]:
    """Summarize one serialized P0/candidate score table without reading truth.

    The promotion gate intentionally consumes CSV bytes that have already
    crossed the scoring boundary.  It does not call a prediction or truth
    reader, and therefore cannot alter the evaluated outcomes.
    """

    if outer_protocol not in PROTOCOLS:
        raise ValueError("promotion summary requires a preregistered outer protocol")
    if candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("promotion summary requires a selected non-R0 candidate")
    if not path.exists():
        raise RuntimeError(f"promotion decision requires a completed {outer_protocol.upper()} score table")
    result = pd.read_csv(path)
    if list(result.columns) != list(RESULT_COLUMNS):
        raise RuntimeError(f"promotion decision found a malformed {outer_protocol.upper()} score schema")
    if result.empty or not result.protocol.astype(str).eq(outer_protocol).all():
        raise RuntimeError(f"promotion decision requires only completed {outer_protocol.upper()} score rows")
    required = {
        "protocol", "column", "outer_seed", "method", "combined_normalized_rmse",
        "V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "all_outputs_finite",
    }
    if missing := sorted(required.difference(result.columns)):
        raise RuntimeError("promotion score table is missing required columns: " + ", ".join(missing))

    expected_seeds = set(SEEDS)
    endpoint_metrics = ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae")
    columns: dict[str, dict[str, Any]] = {}
    for column in COLUMNS:
        subset = result.loc[result.column.astype(str).eq(column)]
        p0 = subset.loc[subset.method.astype(str).eq("P0")].copy()
        current = subset.loc[subset.method.astype(str).eq(candidate)].copy()
        for method, frame in (("P0", p0), (candidate, current)):
            if len(frame) != len(SEEDS) or set(frame.outer_seed.astype(int)) != expected_seeds:
                raise RuntimeError(
                    f"promotion decision requires one {method} result for every frozen seed in {column}/{outer_protocol}"
                )
            if frame.duplicated("outer_seed").any():
                raise RuntimeError(f"promotion decision found duplicate {method} seed rows in {column}/{outer_protocol}")
            if not frame.all_outputs_finite.eq(True).all():
                raise RuntimeError(f"promotion decision cannot use non-finite {method} outputs in {column}/{outer_protocol}")
        paired = p0.merge(current, on="outer_seed", suffixes=("_p0", "_candidate"), validate="one_to_one")
        p0_combined_mean, p0_combined_std = _mean_std(paired["combined_normalized_rmse_p0"])
        candidate_combined_mean, candidate_combined_std = _mean_std(paired["combined_normalized_rmse_candidate"])
        mean_gain_pct = -_relative_change_pct(candidate_combined_mean, p0_combined_mean)
        seed_details = []
        for value in paired.sort_values("outer_seed").itertuples(index=False):
            p0_value = float(getattr(value, "combined_normalized_rmse_p0"))
            candidate_value = float(getattr(value, "combined_normalized_rmse_candidate"))
            gain_pct = -_relative_change_pct(candidate_value, p0_value)
            seed_details.append({
                "outer_seed": int(value.outer_seed),
                "p0_combined_nrmse": p0_value,
                "candidate_combined_nrmse": candidate_value,
                "gain_pct": gain_pct,
                "candidate_win": bool(candidate_value < p0_value),
            })
        endpoint_summary: dict[str, dict[str, float]] = {}
        for metric in endpoint_metrics:
            p0_mean, p0_std = _mean_std(paired[f"{metric}_p0"])
            candidate_mean, candidate_std = _mean_std(paired[f"{metric}_candidate"])
            endpoint_summary[metric] = {
                "p0_mean": p0_mean,
                "p0_std": p0_std,
                "candidate_mean": candidate_mean,
                "candidate_std": candidate_std,
                "mean_relative_regression_pct": _relative_change_pct(candidate_mean, p0_mean),
            }
        columns[column] = {
            "n_seeds": len(SEEDS),
            "combined_nrmse": {
                "p0_mean": p0_combined_mean,
                "p0_std": p0_combined_std,
                "candidate_mean": candidate_combined_mean,
                "candidate_std": candidate_combined_std,
                "mean_gain_pct": mean_gain_pct,
                "seed_wins": int(sum(item["candidate_win"] for item in seed_details)),
                "seed_details": seed_details,
            },
            "endpoint_metrics": endpoint_summary,
        }
    return {
        "protocol": outer_protocol,
        "candidate": candidate,
        "reference": "P0",
        "columns": columns,
    }


def _prediction_freeze_reference(outer_protocol: str, candidate: str) -> dict[str, str]:
    """Verify and address the freeze manifest that authorized a score table."""

    _assert_frozen_for_score(outer_protocol, ("R0", candidate))
    return _artifact_reference(STUDY / f"{outer_protocol.upper()}_PREDICTION_FREEZE_MANIFEST.json")


def _promotion_decision_payload(candidate: str) -> dict[str, Any]:
    """Derive the terminal promotion interpretation from already-scored evidence."""

    row_path = STUDY / "ROW_RESULTS.csv"
    compound_path = STUDY / "COMPOUND_RESULTS.csv"
    continuation_path = STUDY / ROW_CONTINUATION_DECISION_NAME
    row_continuation = _load_row_continuation_decision(candidate)
    if (row_continuation.get("status") != "ROW_CONTINUATION_PASSED"
            or not row_continuation.get("compound_confirmation_authorized")):
        raise RuntimeError("promotion decision is unavailable because the ROW continuation gate did not authorize COMPOUND")
    row_summary = _serialized_score_summary("row", candidate, row_path)
    compound_summary = _serialized_score_summary("compound", candidate, compound_path)
    row_freeze = _prediction_freeze_reference("row", candidate)
    compound_freeze = _prediction_freeze_reference("compound", candidate)

    row_columns = row_continuation["columns"]
    strong_row_gain = bool(all(
        float(row_columns[column]["combined_nrmse"]["mean_gain_pct"])
        >= PROMOTION_RULE["strong_row_mean_combined_gain_pct_min"]
        for column in COLUMNS
    ))
    strong_row_wins = bool(all(
        int(row_columns[column]["combined_nrmse"]["seed_wins"])
        >= PROMOTION_RULE["strong_row_seed_wins_min"]
        for column in COLUMNS
    ))
    row_endpoint_safe = bool(
        row_continuation["criteria"].get("no_endpoint_mean_rmse_or_mae_regression_above_limit")
    )
    compound_gains = {
        column: float(compound_summary["columns"][column]["combined_nrmse"]["mean_gain_pct"])
        for column in COLUMNS
    }
    compound_positive = bool(all(
        gain > PROMOTION_RULE["strong_compound_mean_combined_gain_pct_min"]
        for gain in compound_gains.values()
    ))
    compound_target = bool(all(
        gain >= PROMOTION_RULE["compound_target_mean_combined_gain_pct"]
        for gain in compound_gains.values()
    ))
    compound_noninferior = bool(all(
        gain >= -PROMOTION_RULE["promising_compound_noninferiority_loss_pct_max"]
        for gain in compound_gains.values()
    ))
    strong = bool(strong_row_gain and strong_row_wins and row_endpoint_safe and compound_positive)
    if strong:
        status = "STRONG_PROMOTION"
    elif compound_noninferior:
        status = "PROMISING_ROW_ONLY"
    else:
        status = "ROW_SPECIFIC_IMPROVEMENT__COMPOUND_DEGRADED"

    return {
        "schema_version": 1,
        "decision_kind": "PROMOTION_DECISION",
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "candidate": candidate,
        "reference": "P0",
        "status": status,
        "rule": PROMOTION_RULE,
        "criteria": {
            "row_continuation_authorized": True,
            "row_strong_mean_combined_gain_both_columns": strong_row_gain,
            "row_strong_seed_wins_both_columns": strong_row_wins,
            "row_endpoint_nonregression": row_endpoint_safe,
            "compound_positive_gain_both_columns": compound_positive,
            "compound_target_gain_both_columns": compound_target,
            "compound_noninferior_within_minus_2pct_both_columns": compound_noninferior,
            "strong_promotion": strong,
        },
        "evidence": {
            "row_results": _artifact_reference(row_path),
            "compound_results": _artifact_reference(compound_path),
            "row_continuation_decision": _artifact_reference(continuation_path),
            "row_prediction_freeze_manifest": row_freeze,
            "compound_prediction_freeze_manifest": compound_freeze,
        },
        "row_continuation": {
            "status": row_continuation["status"],
            "criteria": dict(row_continuation["criteria"]),
        },
        "row": row_summary,
        "compound": compound_summary,
        "decision_boundary": (
            "Serialized ROW/COMPOUND score CSVs, the immutable ROW continuation decision, and "
            "the two verified prediction-freeze manifests only; this action never reads endpoint truth."
        ),
    }


def finalize_promotion_decision(candidate: str) -> dict[str, Any]:
    """Commit the one immutable post-COMPOUND promotion interpretation."""

    with _promotion_decision_lock():
        return _persist_immutable_json(
            STUDY / PROMOTION_DECISION_NAME,
            _promotion_decision_payload(candidate),
            label="promotion decision",
        )


def _load_promotion_decision(candidate: str) -> dict[str, Any]:
    """Load and recompute-verify the report's sole final-promotion input."""

    path = STUDY / PROMOTION_DECISION_NAME
    if not path.exists():
        raise RuntimeError("cannot report a completed COMPOUND confirmation without PROMOTION_DECISION.json")
    decision = json.loads(path.read_text(encoding="utf-8"))
    expected = _promotion_decision_payload(candidate)
    if decision != expected:
        raise RuntimeError("promotion decision is mutable, unhashed, or inconsistent with its scored evidence")
    return decision


def _clustered_bootstrap_delta(truth: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
                               clusters: Sequence[str], *, seed: int, draws: int = 2000) -> list[dict[str, float | str | int]]:
    """Cluster-resample canonical SMILES for endpoint-wise ΔRMSE/ΔMAE CIs."""

    cluster_array = np.asarray([str(value) for value in clusters])
    if len(cluster_array) != len(truth):
        raise ValueError("bootstrap clusters do not match test rows")
    unique = np.unique(cluster_array)
    if len(unique) < 2:
        raise ValueError("clustered bootstrap requires at least two canonical SMILES")
    positions = {value: np.flatnonzero(cluster_array == value) for value in unique}
    rng = np.random.default_rng(int(seed))
    rows = []
    for target, index in (("V1", 0), ("V2", 1)):
        full_candidate_error = truth[:, index] - candidate[:, 3 * index + 1]
        full_reference_error = truth[:, index] - reference[:, 3 * index + 1]
        observed_rmse_delta = math.sqrt(float(np.mean(full_candidate_error ** 2))) - math.sqrt(float(np.mean(full_reference_error ** 2)))
        observed_mae_delta = float(np.mean(np.abs(full_candidate_error))) - float(np.mean(np.abs(full_reference_error)))
        rmse_delta = np.empty(draws, dtype=float)
        mae_delta = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            take = np.concatenate([positions[value] for value in sampled])
            cand_error = truth[take, index] - candidate[take, 3 * index + 1]
            ref_error = truth[take, index] - reference[take, 3 * index + 1]
            rmse_delta[draw] = math.sqrt(float(np.mean(cand_error ** 2))) - math.sqrt(float(np.mean(ref_error ** 2)))
            mae_delta[draw] = float(np.mean(np.abs(cand_error))) - float(np.mean(np.abs(ref_error)))
        rows.append({"endpoint": target, "metric": "delta_rmse", "estimate": observed_rmse_delta,
                     "ci95_low": float(np.quantile(rmse_delta, .025)), "ci95_high": float(np.quantile(rmse_delta, .975)),
                     "bootstrap_draws": int(draws), "cluster_count": int(len(unique))})
        rows.append({"endpoint": target, "metric": "delta_mae", "estimate": observed_mae_delta,
                     "ci95_low": float(np.quantile(mae_delta, .025)), "ci95_high": float(np.quantile(mae_delta, .975)),
                     "bootstrap_draws": int(draws), "cluster_count": int(len(unique))})
    return rows


def _replace_protocol_rows(path: Path, frame: pd.DataFrame, outer_protocol: str) -> None:
    """Persist one protocol without rewriting a completed other protocol's rows."""

    _persist_scored_protocol_rows(path, frame, outer_protocol)


RESULT_COLUMNS = (
    "protocol", "column", "outer_seed", "method", "n_test", "metric_scale_authority",
    "V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae",
    "combined_normalized_rmse", "all_outputs_finite",
)
PAIRED_COLUMNS = (
    "protocol", "column", "outer_seed", "candidate", "reference",
    "candidate_minus_reference_V1_rmse", "candidate_minus_reference_V2_rmse",
    "candidate_minus_reference_V1_mae", "candidate_minus_reference_V2_mae",
    "candidate_minus_reference_combined_nrmse",
)
BOOTSTRAP_COLUMNS = (
    "protocol", "column", "outer_seed", "candidate", "reference", "endpoint", "metric",
    "estimate", "ci95_low", "ci95_high", "bootstrap_draws", "cluster_count",
)


def _canonical_scored_frame(frame: pd.DataFrame, columns: Sequence[str], *, sort_by: Sequence[str],
                            path: Path) -> pd.DataFrame:
    """Normalize a score artifact for semantic equality checks, never for rewrites."""

    expected_columns = list(columns)
    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"scored artifact schema mismatch: {path}")
    return frame.loc[:, expected_columns].sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)


def _scored_sort_columns(columns: Sequence[str]) -> tuple[str, ...]:
    if tuple(columns) == RESULT_COLUMNS:
        return ("protocol", "column", "outer_seed", "method")
    if tuple(columns) == PAIRED_COLUMNS:
        return ("protocol", "column", "outer_seed", "candidate", "reference")
    if tuple(columns) == BOOTSTRAP_COLUMNS:
        return ("protocol", "column", "outer_seed", "candidate", "reference", "endpoint", "metric")
    raise ValueError("unknown scored artifact schema")


def _assert_same_scored_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, columns: Sequence[str],
                              path: Path) -> None:
    """Accept an idempotent re-score only when its serialized meaning agrees."""

    sort_by = _scored_sort_columns(columns)
    actual_canonical = _canonical_scored_frame(actual, columns, sort_by=sort_by, path=path)
    expected_canonical = _canonical_scored_frame(expected, columns, sort_by=sort_by, path=path)
    try:
        pd.testing.assert_frame_equal(
            actual_canonical, expected_canonical,
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(f"refusing to replace existing scored artifact: {path}") from error


def _persist_scored_frame(path: Path, frame: pd.DataFrame, columns: Sequence[str], *, dry_run: bool = False) -> None:
    """Write a score table once, or verify it without changing its existing bytes."""

    canonical = _canonical_scored_frame(frame, columns, sort_by=_scored_sort_columns(columns), path=path)
    if path.exists():
        _assert_same_scored_frame(pd.read_csv(path), canonical, columns=columns, path=path)
        return
    if not dry_run:
        write_frame(path, canonical)


def _persist_scored_protocol_rows(path: Path, frame: pd.DataFrame, outer_protocol: str, *, dry_run: bool = False) -> None:
    """Append a new protocol once while preserving prior scored rows byte-for-byte.

    `PAIRED_COMPARISON.csv` and `BOOTSTRAP_CI.csv` are shared ROW/COMPOUND
    artifacts.  A COMPOUND score is allowed to append after ROW, but a re-score
    may only verify an existing protocol slice; it can never replace it.
    """

    columns = PAIRED_COLUMNS if "candidate_minus_reference_combined_nrmse" in frame.columns else BOOTSTRAP_COLUMNS
    canonical = _canonical_scored_frame(frame, columns, sort_by=_scored_sort_columns(columns), path=path)
    if not canonical.protocol.astype(str).eq(outer_protocol).all():
        raise RuntimeError(f"scored protocol rows do not match requested protocol: {path}")
    if not path.exists():
        if not dry_run:
            write_frame(path, canonical)
        return
    existing = pd.read_csv(path)
    # Validate the complete existing schema before deciding whether it can be
    # extended.  A header-only no-score artifact is safely extended in the
    # same append-only fashion.
    _canonical_scored_frame(existing, columns, sort_by=_scored_sort_columns(columns), path=path)
    existing_protocol = existing.loc[existing.protocol.astype(str).eq(outer_protocol)]
    if not existing_protocol.empty:
        _assert_same_scored_frame(existing_protocol, canonical, columns=columns, path=path)
        return
    if not canonical.empty and not dry_run:
        _append_frame_without_rewriting_existing_rows(path, canonical)


def _validate_empty_scored_artifact(path: Path, columns: Sequence[str]) -> None:
    """Ensure an optional scored table can safely represent an unscored stop."""

    expected = list(columns)
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty or list(existing.columns) != expected:
            raise RuntimeError(f"refusing to replace existing scored artifact: {path}")


def _write_empty_scored_artifact(path: Path, columns: Sequence[str]) -> None:
    """Create a declared no-score table without replacing scored evidence.

    A failed ROW inner gate is an intentional terminal outcome.  The required
    report tables still need stable schemas, but inventing rows (or touching
    outer truth to fill them) would obscure that the score boundary was never
    crossed.  This helper is deliberately idempotent only for an identical
    header-only table and otherwise refuses to replace any evidence.
    """

    _validate_empty_scored_artifact(path, columns)
    if not path.exists():
        write_frame(path, pd.DataFrame(columns=list(columns)))


def _negative_inner_gate_decision() -> tuple[dict[str, Any], Path]:
    """Validate that the preregistered ROW screen, not an outer score, stopped."""

    path = STUDY / FINAL_READOUT_DECISION_NAME
    decision = _verified_final_readout_decision()
    screened = set(decision.get("screened_arms", ()))
    gates = decision.get("gates")
    if not {"R0", "R1", "R2"}.issubset(screened):
        raise RuntimeError("negative finalization requires the preregistered R0/R1/R2 ROW screen")
    if not isinstance(gates, dict) or not {"R1", "R2"}.issubset(gates) or any(bool(value) for value in gates.values()):
        raise RuntimeError("negative finalization is allowed only when no readout arm passed its gate")
    summary = decision.get("summary")
    if not isinstance(summary, list) or not {
        (arm, column) for arm in ("R1", "R2") for column in COLUMNS
    }.issubset({(str(item.get("arm")), str(item.get("column"))) for item in summary if isinstance(item, dict)}):
        raise RuntimeError("negative finalization requires two-column inner-screen summaries for R1 and R2")
    if decision.get("next_arm") is not None or decision.get("selected_candidate_if_screen_stops") is not None:
        raise RuntimeError("negative finalization refused: the screen selected or authorized a candidate")
    if decision.get("selection_boundary") != READOUT_SELECTION_BOUNDARY:
        raise RuntimeError("negative finalization refused: inner-screen selection boundary is not the frozen ROW boundary")
    return decision, path


def _negative_final_report(decision: Mapping[str, Any], stopping: Mapping[str, Any]) -> None:
    """Write a full, explicitly unscored report for a failed inner gate."""

    summary = decision.get("summary", [])
    lines = [
        "# Final report — conditioned source readout", "",
        "## Decision", "",
        "`NO_MATERIAL_ARCHITECTURE_GAIN`", "",
        "The preregistered ROW inner-CV gate found no eligible R1/R2 readout arm. "
        "Consequently this terminal report was created without reading outer validation/test truth, "
        "without making new outer predictions, and without running COMPOUND confirmation.", "",
        "## ROW inner-screen evidence", "",
        "| arm | column | mean relative improvement (%) | fold wins | seed wins | gate |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    gates = decision.get("gates", {})
    for item in summary:
        lines.append(
            f"| {item['arm']} | {item['column']} | {item['mean_relative_improvement_pct']:.3f} | "
            f"{item['fold_wins']}/{item['n_folds']} | {item['seed_wins']}/{item['n_seeds']} | "
            f"{'pass' if gates.get(item['arm'], False) else 'fail'} |"
        )
    lines.extend([
        "", "## Outer-score artifact status", "",
        "`ROW_RESULTS.csv`, `COMPOUND_RESULTS.csv`, `PAIRED_COMPARISON.csv`, and `BOOTSTRAP_CI.csv` "
        "are intentionally header-only. They record no synthetic or partially observed score; the machine-readable "
        "stopping decision and `prediction_hashes.json` attest that the outer-test score boundary was not crossed.", "",
        "## Required questions", "",
        "1. **Is fixed sum pooling a transfer bottleneck?** No qualifying evidence under the preregistered ROW inner screen.",
        f"2. **Does adaptive readout (R1) improve stably?** {'Yes' if gates.get('R1') else 'No'} under the fixed gate.",
        f"3. **Does condition-query readout (R2) improve over R1/R0?** {'Yes' if gates.get('R2') else 'No'} under the fixed gate.",
        "4. **Does source prediction (R3) add benefit?** Not run: R2 did not authorize source-prediction augmentation.",
        "5. **Does source embedding (R4) add incremental benefit?** Not run: R3 was not authorized.",
        "6. **Is improvement present in both 25g and 40g?** No arm met the two-column inner-CV rule.",
        "7. **Is improvement present in both ROW and COMPOUND?** Not assessed; COMPOUND is correctly blocked by the ROW stopping rule.",
        "8. **Does the new model exceed paper-style/P0/structured baselines?** Not assessed: no outer-test score was read.",
        "9. **Is any improvement driven by one endpoint?** Not assessed on outer truth; no endpoint-level outer score exists.",
        "10. **Does the study meet the promotion gate?** No — `NO_MATERIAL_ARCHITECTURE_GAIN`.", "",
        "## Reproducibility", "",
        f"Stopping-decision SHA256: `{stopping['inner_screen_decision_sha256']}`. "
        "The inner screen remains in `INNER_SCREEN_RESULTS.csv`; protocol and frozen split provenance remain in "
        "`protocol.json` and `run_manifest.csv`. This report action performs no endpoint-data read.", "",
    ])
    report_path = STUDY / "FINAL_REPORT.md"
    content = "\n".join(lines)
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != content:
            raise RuntimeError("negative finalization refused: final report drift")
    else:
        report_path.write_text(content, encoding="utf-8")


def finalize_negative_outcome() -> dict[str, Any]:
    """Materialize required top-level artifacts after an inner gate failure.

    This is the only terminal action that is authorized when the selected
    candidate is ``None``.  It intentionally never calls context preparation,
    final refitting, prediction freeze, or score, so outer endpoint truth is
    unavailable by construction.
    """

    decision, decision_path = _negative_inner_gate_decision()
    required = (
        "PREREGISTRATION.md", "IMPLEMENTATION_AUDIT.md", "LOSS_SCREEN.csv", "LOSS_SCREEN_REPORT.md",
        "INNER_SCREEN_RESULTS.csv", "protocol.json", "run_manifest.csv",
    )
    missing = [name for name in required if not (STUDY / name).exists()]
    if missing:
        raise RuntimeError("negative finalization requires completed prerequisite artifacts: " + ", ".join(missing))
    stopping = {
        "status": "NO_MATERIAL_ARCHITECTURE_GAIN",
        "stage": "ROW_INNER_SCREEN",
        "reason": "No R1/R2 arm satisfied the preregistered two-column 3% / 3-of-5 / 13-of-25 gate.",
        "screened_arms": decision["screened_arms"],
        "gates": decision["gates"],
        "inner_screen_decision": decision_path.name,
        "inner_screen_decision_sha256": sha(decision_path),
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "outer_test_truth_read": False,
        "outer_predictions_created": False,
        "compound_confirmation_authorized": False,
    }
    scored_artifacts = (
        (STUDY / "ROW_RESULTS.csv", RESULT_COLUMNS),
        (STUDY / "COMPOUND_RESULTS.csv", RESULT_COLUMNS),
        (STUDY / "PAIRED_COMPARISON.csv", PAIRED_COLUMNS),
        (STUDY / "BOOTSTRAP_CI.csv", BOOTSTRAP_COLUMNS),
    )
    # Validate all destinations before creating any terminal artifact, avoiding
    # a partial stopping record if a user has already placed scored evidence.
    for artifact_path, columns in scored_artifacts:
        _validate_empty_scored_artifact(artifact_path, columns)
    hashes_path = STUDY / "prediction_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.exists() else {}
    if any(protocol in hashes for protocol in PROTOCOLS):
        raise RuntimeError("negative finalization refused: a prediction-freeze manifest already exists")
    existing_hash_stopping = hashes.get("stopping_decision")
    if existing_hash_stopping is not None and existing_hash_stopping != stopping:
        raise RuntimeError("negative finalization refused: prediction-hash stopping decision drift")
    stopping_path = STUDY / "OUTER_TEST_STOPPING_DECISION.json"
    if stopping_path.exists():
        existing_stopping = json.loads(stopping_path.read_text(encoding="utf-8"))
        if existing_stopping != stopping:
            raise RuntimeError("negative finalization refused: outer-test stopping decision drift")
    else:
        write_json(stopping_path, stopping)
    hashes["stopping_decision"] = stopping
    write_json(hashes_path, hashes)
    for artifact_path, columns in scored_artifacts:
        _write_empty_scored_artifact(artifact_path, columns)
    _negative_final_report(decision, stopping)
    return stopping


def _preflight_score_inputs(outer_protocol: str, candidate: str,
                            references: Sequence[str]) -> list[dict[str, Any]]:
    """Load every prediction and comparator contract before any truth read."""

    prepared: list[dict[str, Any]] = []
    for column in COLUMNS:
        canonical = FROZEN / f"filtered_canonical_{column}.csv"
        feature = pd.read_csv(FROZEN / f"filtered_features_{column}.csv", usecols=["sample_id", "canonical_smiles"])
        feature["sample_id"] = feature.sample_id.astype(str)
        canonical_by_id = feature.set_index("sample_id").canonical_smiles.astype(str).to_dict()
        for seed in SEEDS:
            context = prepare_context(column, outer_protocol, seed)
            test_ids = [str(value) for value in context["roles"]["test"]]
            predictions: dict[str, np.ndarray] = {
                "P0": _final_prediction(column, outer_protocol, seed, "R0", test_ids),
                candidate: _final_prediction(column, outer_protocol, seed, candidate, test_ids),
            }
            for reference in references:
                predictions[reference] = _reference_prediction(column, outer_protocol, seed, reference, test_ids)
            try:
                clusters = [canonical_by_id[sample_id] for sample_id in test_ids]
            except KeyError as error:
                raise RuntimeError(
                    f"frozen test IDs are missing canonical-smiles clusters for {column}/{outer_protocol}/seed_{seed}"
                ) from error
            prepared.append({
                "column": column,
                "seed": seed,
                "canonical": canonical,
                "context": context,
                "test_ids": test_ids,
                "predictions": predictions,
                "clusters": clusters,
            })
    return prepared


def score(outer_protocol: str, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read outer truth only after all prediction and comparator preflight checks."""

    if outer_protocol not in PROTOCOLS or candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("score requires a selected non-R0 candidate and frozen protocol")
    if outer_protocol == "compound":
        # This guard executes before the freeze check, context construction, or
        # any endpoint-data read, so an unsuccessful ROW result cannot be
        # bypassed by invoking the COMPOUND score action directly.
        _assert_row_continuation_authorized(candidate)
    _selected_candidate_provenance(candidate)
    arms = ("R0", candidate)
    _assert_frozen_for_score(outer_protocol, arms)
    result_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    references = ["paper_style_current_v2"] if outer_protocol == "row" else [
        "M3_CENTER_WIDTH_FULL", "HIER_CW_SHARED_LAMBDA_CORRECTED"
    ]
    preflight = _preflight_score_inputs(outer_protocol, candidate, references)
    for item in preflight:
        column = str(item["column"])
        seed = int(item["seed"])
        canonical = Path(item["canonical"])
        context = item["context"]
        test_ids = list(item["test_ids"])
        predictions = item["predictions"]
        clusters = list(item["clusters"])
        truth = _read_authorized_truth(canonical, test_ids)
        for method, prediction in predictions.items():
            metrics = _metric_values(truth, prediction, context["outer_scales"])
            result_rows.append({"protocol": outer_protocol, "column": column, "outer_seed": int(seed),
                                "method": method, "n_test": len(test_ids),
                                "metric_scale_authority": "outer_gradient_train_ddof0", **metrics})
        for reference in ["P0", *references]:
            candidate_metrics = _metric_values(truth, predictions[candidate], context["outer_scales"])
            reference_metrics = _metric_values(truth, predictions[reference], context["outer_scales"])
            paired_rows.append({
                "protocol": outer_protocol, "column": column, "outer_seed": int(seed), "candidate": candidate,
                "reference": reference, "candidate_minus_reference_V1_rmse": candidate_metrics["V1_rmse"] - reference_metrics["V1_rmse"],
                "candidate_minus_reference_V2_rmse": candidate_metrics["V2_rmse"] - reference_metrics["V2_rmse"],
                "candidate_minus_reference_V1_mae": candidate_metrics["V1_mae"] - reference_metrics["V1_mae"],
                "candidate_minus_reference_V2_mae": candidate_metrics["V2_mae"] - reference_metrics["V2_mae"],
                "candidate_minus_reference_combined_nrmse": candidate_metrics["combined_normalized_rmse"] - reference_metrics["combined_normalized_rmse"],
            })
            for row in _clustered_bootstrap_delta(
                truth, predictions[candidate], predictions[reference], clusters,
                seed=deterministic_seed("bootstrap", outer_protocol, column, seed, candidate, reference),
            ):
                bootstrap_rows.append({"protocol": outer_protocol, "column": column, "outer_seed": int(seed),
                                       "candidate": candidate, "reference": reference, **row})
    result = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    paired = pd.DataFrame(paired_rows, columns=PAIRED_COLUMNS)
    bootstrap = pd.DataFrame(bootstrap_rows, columns=BOOTSTRAP_COLUMNS)
    target = STUDY / ("ROW_RESULTS.csv" if outer_protocol == "row" else "COMPOUND_RESULTS.csv")
    with _score_artifact_lock():
        # Validate every destination before changing any scored artifact.  This
        # makes a divergent re-score fail as a no-op rather than leave a newly
        # written result table beside an incompatible paired/bootstrap table.
        _persist_scored_frame(target, result, RESULT_COLUMNS, dry_run=True)
        _persist_scored_protocol_rows(STUDY / "PAIRED_COMPARISON.csv", paired, outer_protocol, dry_run=True)
        _persist_scored_protocol_rows(STUDY / "BOOTSTRAP_CI.csv", bootstrap, outer_protocol, dry_run=True)
        _persist_scored_frame(target, result, RESULT_COLUMNS)
        _replace_protocol_rows(STUDY / "PAIRED_COMPARISON.csv", paired, outer_protocol)
        _replace_protocol_rows(STUDY / "BOOTSTRAP_CI.csv", bootstrap, outer_protocol)
        if outer_protocol == "row":
            _persist_row_continuation_decision(candidate, pd.read_csv(target))
        else:
            # This is terminal interpretation only.  It receives completed,
            # serialized score artifacts and never reopens the truth boundary.
            finalize_promotion_decision(candidate)
    return result, paired, bootstrap


def _summary_table(result: pd.DataFrame) -> pd.DataFrame:
    metrics = ["V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse"]
    return result.groupby(["protocol", "column", "method"], as_index=False)[metrics].agg(["mean", "std"]).reset_index()


def generate_final_report(candidate: str) -> None:
    """Render only already-scored artifacts; it has no endpoint-data access."""

    row_path, compound_path = STUDY / "ROW_RESULTS.csv", STUDY / "COMPOUND_RESULTS.csv"
    if not row_path.exists():
        raise RuntimeError("cannot render final report before ROW score")
    row = pd.read_csv(row_path)
    compound = pd.read_csv(compound_path) if compound_path.exists() else pd.DataFrame()
    paired = pd.read_csv(STUDY / "PAIRED_COMPARISON.csv") if (STUDY / "PAIRED_COMPARISON.csv").exists() else pd.DataFrame()
    bootstrap = pd.read_csv(STUDY / "BOOTSTRAP_CI.csv") if (STUDY / "BOOTSTRAP_CI.csv").exists() else pd.DataFrame()
    continuation_decision = _load_row_continuation_decision(candidate)
    if not continuation_decision["compound_confirmation_authorized"] and not compound.empty:
        raise RuntimeError("cannot report COMPOUND evidence after a failed ROW continuation gate")
    promotion_decision: dict[str, Any] | None = None

    def mean_std_text(values: pd.Series) -> str:
        numeric = values.to_numpy(float)
        if not np.isfinite(numeric).all():
            return "non-finite"
        return f"{numeric.mean():.3f} ± {numeric.std(ddof=1) if len(numeric) > 1 else 0.0:.3f}"

    def compact(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_Not run because the preceding gate did not authorize it._\n"
        metrics = ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse")
        lines = ["| column | method | V1 RMSE | V1 MAE | V1 R² | V2 RMSE | V2 MAE | V2 R² | combined train-NRMSE |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for (column, method), group in frame.groupby(["column", "method"], sort=True):
            summary = [mean_std_text(group[metric]) for metric in metrics]
            lines.append(f"| {column} | {method} | " + " | ".join(summary) + " |")
        return "\n".join(lines) + "\n"

    if continuation_decision["compound_confirmation_authorized"] and compound.empty:
        status = "ROW_CONTINUATION_PASSED__COMPOUND_CONFIRMATION_PENDING"
    elif not continuation_decision["compound_confirmation_authorized"]:
        status = "ROW_CONTINUATION_FAILED__COMPOUND_BLOCKED"
    else:
        # The report is intentionally a consumer, never an alternative gate:
        # this reload recomputes and hash-verifies the immutable decision from
        # the serialized score/freeze evidence without reading endpoint truth.
        promotion_decision = _load_promotion_decision(candidate)
        status = str(promotion_decision["status"])

    lines = ["# Final report — conditioned source readout", "", f"## Decision\n\n`{status}`\n",
             "All outer results are developmental confirmation because these frozen outer identities were historically exposed. No score selected the loss, architecture, source feature, or epoch.",
             "", "## Formal ROW continuation gate", "",
             f"Outcome: **`{continuation_decision['status']}`**. COMPOUND confirmation authorized: **`{continuation_decision['compound_confirmation_authorized']}`**.",
             "",
             "| column | P0 combined NRMSE (mean ± std) | candidate combined NRMSE (mean ± std) | mean gain (%) | wins / 5 | gain ≥3% | wins ≥4/5 |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    if promotion_decision is not None:
        lines.extend([
            "", "## Post-COMPOUND promotion decision", "",
            f"Outcome: **`{promotion_decision['status']}`**. The immutable decision is bound to both score tables, the ROW continuation decision, and both prediction-freeze manifests.",
            f"Promotion-decision SHA256: `{sha(STUDY / PROMOTION_DECISION_NAME)}`.",
        ])
    for column in COLUMNS:
        combined = continuation_decision["columns"][column]["combined_nrmse"]
        lines.append(
            f"| {column} | {combined['p0_mean']:.3f} ± {combined['p0_std']:.3f} | "
            f"{combined['candidate_mean']:.3f} ± {combined['candidate_std']:.3f} | "
            f"{combined['mean_gain_pct']:.3f} | {combined['seed_wins']}/{continuation_decision['rule']['seed_count']} | "
            f"{'pass' if combined['mean_gain_pass'] else 'fail'} | {'pass' if combined['seed_wins_pass'] else 'fail'} |"
        )
    lines.extend([
        "", "The gate requires at least 3% mean combined gain in **both** columns, at least 4/5 matched seed wins in **both** columns, and no mean endpoint RMSE/MAE regression above 2%.",
        "", "| column | endpoint metric | P0 (mean ± std) | candidate (mean ± std) | mean regression (%) | within +2% |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for column in COLUMNS:
        for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae"):
            endpoint = continuation_decision["columns"][column]["endpoint_metrics"][metric]
            lines.append(
                f"| {column} | {metric} | {endpoint['p0_mean']:.3f} ± {endpoint['p0_std']:.3f} | "
                f"{endpoint['candidate_mean']:.3f} ± {endpoint['candidate_std']:.3f} | "
                f"{endpoint['mean_relative_regression_pct']:.3f} | "
                f"{'pass' if endpoint['within_regression_limit'] else 'fail'} |"
            )
    lines.extend(["", "## ROW results", "", compact(row), "", "## COMPOUND results", "", compact(compound),
                  "", "## Seed-wise matched comparison", ""])
    if paired.empty:
        lines.append("_No paired comparison has been scored._")
    else:
        lines.extend(["| protocol | column | candidate | reference | Δ combined NRMSE (mean ± std) | candidate wins / 5 |", "| --- | --- | --- | --- | ---: | ---: |"])
        for (protocol, column, compared_candidate, reference), group in paired.groupby(["protocol", "column", "candidate", "reference"], sort=True):
            wins = "—"
            if protocol == "row" and compared_candidate == candidate and reference == "P0":
                wins = f"{continuation_decision['columns'][column]['combined_nrmse']['seed_wins']}/5"
            lines.append(f"| {protocol} | {column} | {compared_candidate} | {reference} | {mean_std_text(group.candidate_minus_reference_combined_nrmse)} | {wins} |")
    lines.extend(["", "## Canonical-SMILES clustered bootstrap", ""])
    if bootstrap.empty:
        lines.append("_No clustered bootstrap intervals have been scored._")
    else:
        subset = bootstrap.loc[(bootstrap.candidate.astype(str).eq(candidate)) & (bootstrap.reference.astype(str).eq("P0"))]
        if subset.empty:
            lines.append("_No candidate-versus-P0 clustered bootstrap intervals have been scored._")
        else:
            lines.extend(["| protocol | column | seed | endpoint | metric | observed full-sample Δ | clustered 95% CI |",
                          "| --- | --- | ---: | --- | --- | ---: | ---: |"])
            for value in subset.sort_values(["protocol", "column", "outer_seed", "endpoint", "metric"]).itertuples(index=False):
                lines.append(f"| {value.protocol} | {value.column} | {value.outer_seed} | {value.endpoint} | {value.metric} | {value.estimate:.4f} | [{value.ci95_low:.4f}, {value.ci95_high:.4f}] |")
    lines.extend(["", "## Required questions", "",
                  f"1. Fixed sum pooling is a transfer bottleneck only if a gated arm passed the inner/ROW gates; current selection: `{candidate}`.",
                  "2. Adaptive readout (R1) is interpreted only from its matched inner summary, not a single outer score.",
                  "3. Condition-query superiority is interpreted only from R2 versus R1/R0 inner evidence.",
                  "4. Source prediction (R3) and 5. source embedding (R4) are reported only if their preceding gates authorized execution.",
                  f"6. Formal ROW gate outcome: `{continuation_decision['status']}`.",
                  "7. COMPOUND is mandatory when ROW continuation passes; its table above is never treated as a tuning screen.",
                  "8. Historical paper-style/P0/structured references were ID-checked and re-scored with common outer-train scales.",
                  "9. Endpoint-level RMSE/MAE are shown above; no aggregate result overrides a material endpoint regression.",
                  f"10. Final promotion status: `{status}`.", "",
                  "## Reproducibility", "",
                  "See `protocol.json`, `run_manifest.csv`, `source_cache/*_cache_manifest.json`, inner context audits, prediction-freeze manifests, `ROW_CONTINUATION_DECISION.json`, `PROMOTION_DECISION.json`, `PAIRED_COMPARISON.csv`, and `BOOTSTRAP_CI.csv`. Clustered bootstrap uses canonical SMILES, not IID rows.", ""])
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "prepare-source-cache", "loss-context", "loss-aggregate", "readout-context", "readout-aggregate", "compound-selection-context", "final-context", "freeze", "score", "promotion-decision", "report", "finalize-negative"), required=True)
    parser.add_argument("--column", choices=COLUMNS)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="row")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arms", nargs="+", choices=READOUT_ARMS)
    parser.add_argument("--candidate", choices=READOUT_ARMS)
    parser.add_argument("--arm", choices=READOUT_ARMS)
    parser.add_argument("--loss-recipe", choices=a.LOSS_RECIPES)
    parser.add_argument("--selection-phase", choices=("loss_screen", "inner_screen", "compound_selection"), default="inner_screen")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(); return
    if args.action == "prepare-source-cache":
        if args.column is None: parser.error("prepare-source-cache requires --column")
        prepare_source_cache(args.column); return
    if args.action == "loss-context":
        if args.column is None or args.seed is None: parser.error("loss-context requires --column and --seed")
        run_loss_context(args.column, args.seed); return
    if args.action == "loss-aggregate":
        aggregate_loss_screen(); return
    if args.action == "readout-context":
        if args.column is None or args.seed is None or not args.arms: parser.error("readout-context requires --column --seed --arms")
        run_readout_context(args.column, args.seed, args.arms); return
    if args.action == "readout-aggregate":
        if not args.arms: parser.error("readout-aggregate requires --arms")
        aggregate_readout_screen(args.arms); return
    if args.action == "finalize-negative":
        finalize_negative_outcome(); return
    if args.action == "compound-selection-context":
        if args.column is None or args.seed is None or args.candidate is None: parser.error("compound-selection-context requires --column --seed --candidate")
        run_compound_selection_context(args.column, args.seed, args.candidate); return
    if args.action == "final-context":
        if args.column is None or args.seed is None or args.arm is None or args.loss_recipe is None: parser.error("final-context requires --column --seed --arm --loss-recipe")
        run_final_context(args.column, args.protocol, args.seed, args.arm, args.loss_recipe, selection_phase=args.selection_phase); return
    if args.action == "freeze":
        if not args.arms: parser.error("freeze requires --arms")
        finalize_prediction_freeze(args.protocol, args.arms); return
    if args.action == "score":
        if args.candidate is None: parser.error("score requires --candidate")
        score(args.protocol, args.candidate); return
    if args.action == "promotion-decision":
        if args.candidate is None: parser.error("promotion-decision requires --candidate")
        finalize_promotion_decision(args.candidate); return
    if args.action == "report":
        if args.candidate is None: parser.error("report requires --candidate")
        generate_final_report(args.candidate); return


if __name__ == "__main__":
    main()

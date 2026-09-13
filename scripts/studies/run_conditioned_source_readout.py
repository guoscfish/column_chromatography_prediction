#!/usr/bin/env python3
"""ROW-first conditioned source-readout transfer study.

Every fitting action is deliberately separated from outer-test scoring.  The
script is resumable by outer context so the preregistered inner screens can be
run on independent workers without changing the split or selection boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
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


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def deterministic_seed(*parts: object) -> int:
    return int(stable_hash([str(value) for value in parts])[:8], 16) & 0x7FFFFFFF


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


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
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("conditioned-source-readout protocol drift; refusing to overwrite")
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


def _inner_fit_context(*, phase: str, column: str, outer_protocol: str, seed: int,
                       arms: Sequence[str], recipes: Mapping[str, str]) -> Path:
    """Run all requested arms' five inner folds inside one outer context."""

    run = _context_dir(phase, outer_protocol, column, seed)
    complete = run / "context_complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("arms") == list(arms) and payload.get("recipes") == dict(recipes):
            return run
        raise RuntimeError(f"existing completed context has a different arm contract: {run}")
    context = prepare_context(column, outer_protocol, seed)
    if any(_arm_requires_source_features(arm) for arm in arms):
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
        for arm in arms:
            recipe = recipes[arm]
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
    write_frame(run / "inner_cv_summary.csv", pd.DataFrame(rows))
    write_frame(run / "inner_cv_history.csv", pd.DataFrame(histories))
    write_json(run / "context_audit.json", context["audit"])
    write_json(complete, {
        "protocol_sha256": sha(STUDY / "protocol.json"), "phase": phase, "column": column,
        "outer_protocol": outer_protocol, "outer_seed": int(seed), "arms": list(arms), "recipes": dict(recipes),
        "summary_sha256": sha(run / "inner_cv_summary.csv"), "history_sha256": sha(run / "inner_cv_history.csv"),
        "outer_validation_or_test_labels_used": False,
    })
    return run


def run_loss_context(column: str, seed: int) -> Path:
    return _inner_fit_context(phase="loss_screen", column=column, outer_protocol="row", seed=seed,
                              arms=tuple(LOSS_ARMS), recipes=LOSS_ARMS)


def _selected_loss() -> dict[str, Any]:
    path = STUDY / "loss_selection.json"
    if not path.exists():
        raise RuntimeError("loss screen has not selected one global recipe")
    return json.loads(path.read_text(encoding="utf-8"))


def run_readout_context(column: str, seed: int, arms: Sequence[str]) -> Path:
    selected = _selected_loss()
    recipe = selected["selected_loss_recipe"]
    if any(arm not in READOUT_ARMS for arm in arms):
        raise ValueError("unknown readout arm")
    return _inner_fit_context(phase="inner_screen", column=column, outer_protocol="row", seed=seed,
                              arms=tuple(arms), recipes={arm: recipe for arm in arms})


def _collect_inner(phase: str, outer_protocol: str, required_arms: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing = []
    for column in COLUMNS:
        for seed in SEEDS:
            path = _context_dir(phase, outer_protocol, column, seed) / "inner_cv_summary.csv"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            frame = pd.read_csv(path)
            if set(required_arms) - set(frame.arm):
                missing.append(f"{path.relative_to(ROOT)} missing arms {sorted(set(required_arms) - set(frame.arm))}")
            frames.append(frame)
    if missing:
        raise RuntimeError("incomplete preregistered inner screen:\n" + "\n".join(missing))
    result = pd.concat(frames, ignore_index=True)
    expected = len(COLUMNS) * len(SEEDS) * 5 * len(required_arms)
    if len(result.loc[result.arm.isin(required_arms)]) != expected:
        raise RuntimeError("inner screen row count does not equal frozen 2x5x5 schedule")
    return result


def aggregate_loss_screen() -> dict[str, Any]:
    values = _collect_inner("loss_screen", "row", tuple(LOSS_ARMS))
    write_frame(STUDY / "LOSS_SCREEN.csv", values.sort_values(["column", "outer_seed", "inner_fold", "arm"]))
    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    base = values.loc[values.arm.eq("L0"), ["column", "outer_seed", "inner_fold", "validation_score"]].rename(columns={"validation_score": "L0_score"})
    for arm in ("L1", "L2"):
        current = values.loc[values.arm.eq(arm), ["column", "outer_seed", "inner_fold", "validation_score"]]
        merged = base.merge(current, on=["column", "outer_seed", "inner_fold"], validate="one_to_one")
        merged["relative_improvement_pct"] = 100 * (merged.L0_score - merged.validation_score) / merged.L0_score
        for column, group in merged.groupby("column", sort=True):
            seed_mean = group.groupby("outer_seed", as_index=False).relative_improvement_pct.mean()
            row = {"arm": arm, "column": column, "mean_relative_improvement_pct": float(group.relative_improvement_pct.mean()),
                   "std_relative_improvement_pct": float(group.relative_improvement_pct.std(ddof=1)),
                   "fold_wins": int((group.relative_improvement_pct > 0).sum()),
                   "seed_wins": int((seed_mean.relative_improvement_pct > 0).sum()),
                   "n_folds": int(len(group)), "n_seeds": int(len(seed_mean))}
            summary_rows.append(row)
            paired_rows.extend({"arm": arm, **record} for record in merged.loc[merged.column.eq(column)].to_dict("records"))
    summary = pd.DataFrame(summary_rows)
    eligible = []
    for arm in ("L1", "L2"):
        subset = summary.loc[summary.arm.eq(arm)]
        if len(subset) != 2:
            continue
        stable = bool((subset.mean_relative_improvement_pct >= 1.0).all()
                      and (subset.seed_wins >= 3).all() and (subset.fold_wins >= 13).all())
        if stable:
            eligible.append((float(subset.mean_relative_improvement_pct.mean()), arm))
    selected_arm = max(eligible)[1] if eligible else "L0"
    decision = {"selected_loss_arm": selected_arm, "selected_loss_recipe": LOSS_ARMS[selected_arm],
                "selection_boundary": "ROW inner GroupKFold only; no outer validation/test truth", "summary": summary_rows,
                "rule": _protocol()["loss_selection_rule"], "eligible_nonbaseline": [arm for _, arm in eligible]}
    write_json(STUDY / "loss_selection.json", decision)
    lines = ["# Loss screen report", "", "The three fixed loss recipes were evaluated only in 5-fold canonical-SMILES GroupKFold inside each ROW outer gradient-train context. No outer validation/test endpoint was available to the screen.", "",
             f"Selected global recipe: **{selected_arm} / `{LOSS_ARMS[selected_arm]}`**.", "", "| arm | column | mean relative improvement (%) | fold wins | seed wins |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary_rows:
        lines.append(f"| {row['arm']} | {row['column']} | {row['mean_relative_improvement_pct']:.3f} | {row['fold_wins']}/25 | {row['seed_wins']}/5 |")
    lines.extend(["", "A non-L0 recipe required >=1% mean improvement in both columns, >=13/25 fold wins and >=3/5 seed wins in both columns. Otherwise the simpler raw P0 recipe remains selected.", ""])
    (STUDY / "LOSS_SCREEN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def aggregate_readout_screen(arms: Sequence[str]) -> dict[str, Any]:
    if "R0" not in arms:
        raise ValueError("R0 is required for every readout screen comparison")
    values = _collect_inner("inner_screen", "row", tuple(arms))
    write_frame(STUDY / "INNER_SCREEN_RESULTS.csv", values.sort_values(["column", "outer_seed", "inner_fold", "arm"]))
    base = values.loc[values.arm.eq("R0"), ["column", "outer_seed", "inner_fold", "validation_score"]].rename(columns={"validation_score": "R0_score"})
    rows: list[dict[str, Any]] = []
    arm_summary: dict[str, list[dict[str, Any]]] = {}
    for arm in (value for value in arms if value != "R0"):
        current = values.loc[values.arm.eq(arm), ["column", "outer_seed", "inner_fold", "validation_score"]]
        merged = base.merge(current, on=["column", "outer_seed", "inner_fold"], validate="one_to_one")
        merged["relative_improvement_pct"] = 100 * (merged.R0_score - merged.validation_score) / merged.R0_score
        per_arm = []
        for column, group in merged.groupby("column", sort=True):
            seed_mean = group.groupby("outer_seed", as_index=False).relative_improvement_pct.mean()
            record = {"arm": arm, "column": column, "mean_relative_improvement_pct": float(group.relative_improvement_pct.mean()),
                      "std_relative_improvement_pct": float(group.relative_improvement_pct.std(ddof=1)),
                      "fold_wins": int((group.relative_improvement_pct > 0).sum()), "seed_wins": int((seed_mean.relative_improvement_pct > 0).sum()),
                      "n_folds": int(len(group)), "n_seeds": int(len(seed_mean))}
            rows.append(record); per_arm.append(record)
        arm_summary[arm] = per_arm
    gates = {}
    for arm, records in arm_summary.items():
        frame = pd.DataFrame(records)
        gates[arm] = bool(len(frame) == 2 and (frame.mean_relative_improvement_pct >= 3.0).all()
                          and (frame.seed_wins >= 3).all() and (frame.fold_wins >= 13).all())
    r2_gate = gates.get("R2", False)
    r3_gate = gates.get("R3", False)
    next_arm = "R3" if set(("R0", "R1", "R2")).issubset(arms) and r2_gate and "R3" not in arms else (
        "R4" if "R3" in arms and r3_gate and "R4" not in arms else None
    )
    eligible = [arm for arm, passed in gates.items() if passed]
    selection = None
    if next_arm is None and eligible:
        selection = max(eligible, key=lambda arm: float(pd.DataFrame(arm_summary[arm]).mean_relative_improvement_pct.mean()))
    decision = {"screened_arms": list(arms), "summary": rows, "gates": gates, "next_arm": next_arm,
                "selected_candidate_if_screen_stops": selection, "rule": _protocol()["architecture_continuation_rule"],
                "selection_boundary": "ROW inner GroupKFold only; no outer validation/test truth"}
    write_json(STUDY / "inner_screen_decision.json", decision)
    return decision


def _median_epoch(summary: pd.DataFrame, arm: str) -> int:
    values = summary.loc[summary.arm.eq(arm), "best_epoch"].to_numpy(float)
    if len(values) != 5:
        raise RuntimeError(f"expected five inner folds to select final epoch for {arm}")
    return int(math.floor(float(np.median(values)) + 0.5))


def _formal_selection(column: str, outer_protocol: str, seed: int, arm: str, recipe: str,
                      *, phase: str) -> int:
    path = _context_dir(phase, outer_protocol, column, seed) / "inner_cv_summary.csv"
    if not path.exists():
        raise RuntimeError(f"missing inner selection context: {path}")
    summary = pd.read_csv(path)
    selection_arm = "L0" if phase == "loss_screen" and arm == "R0" else arm
    expected = summary.loc[summary.arm.eq(selection_arm), "loss_recipe"].unique().tolist()
    if expected != [recipe]:
        raise RuntimeError("selected final recipe does not match frozen inner-screen arm")
    return _median_epoch(summary, selection_arm)


def run_compound_selection_context(column: str, seed: int, candidate: str) -> Path:
    selected = _selected_loss()
    if candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("compound confirmation requires an architecture candidate R1-R4")
    recipes = {"R0": "raw_quantile", candidate: selected["selected_loss_recipe"]}
    return _inner_fit_context(phase="compound_selection", column=column, outer_protocol="compound", seed=seed,
                              arms=("R0", candidate), recipes=recipes)


def _final_dir(outer_protocol: str, column: str, seed: int, arm: str) -> Path:
    return STUDY / "runtime" / "formal" / outer_protocol / column / f"seed_{seed}" / arm


def run_final_context(column: str, outer_protocol: str, seed: int, arm: str, recipe: str, *, selection_phase: str) -> Path:
    """Fixed-epoch final refit and blind prediction freeze for one arm/context."""

    run = _final_dir(outer_protocol, column, seed, arm)
    complete = run / "final_fit_complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("arm") == arm and payload.get("loss_recipe") == recipe and payload.get("selection_phase") == selection_phase:
            return run
        raise RuntimeError("existing final fit has a different frozen contract")
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
                "git_sha": git_sha(), "source_cache": source_cache_audit,
                "outer_validation_or_test_labels_used": False})
    write_json(complete, {"protocol_sha256": sha(STUDY / "protocol.json"), "selection_phase": selection_phase,
                "outer_protocol": outer_protocol, "column": column, "outer_seed": int(seed), "arm": arm,
                "loss_recipe": recipe, "selected_epoch": selected_epoch, "final_pt_sha256": sha(run / "final.pt"),
                "validation_prediction_sha256": sha(run / "validation_predictions_blind.csv.gz"),
                "test_prediction_sha256": sha(run / "test_predictions_blind.csv.gz"),
                "outer_validation_or_test_labels_used": False})
    return run


def finalize_prediction_freeze(outer_protocol: str, arms: Sequence[str]) -> dict[str, Any]:
    files = {}
    for column in COLUMNS:
        for seed in SEEDS:
            for arm in arms:
                run = _final_dir(outer_protocol, column, seed, arm)
                complete = run / "final_fit_complete.json"
                prediction = run / "test_predictions_blind.csv.gz"
                if not complete.exists() or not prediction.exists():
                    raise RuntimeError(f"missing final blind prediction before freeze: {run}")
                payload = json.loads(complete.read_text(encoding="utf-8"))
                if payload["test_prediction_sha256"] != sha(prediction) or payload["outer_validation_or_test_labels_used"]:
                    raise RuntimeError("final prediction hash/label-use assertion failed")
                files[str(prediction.relative_to(ROOT))] = sha(prediction)
    payload = {"protocol_sha256": sha(STUDY / "protocol.json"), "outer_protocol": outer_protocol,
               "arms": list(arms), "files": files, "global_outer_truth_read_before_freeze": False}
    write_json(STUDY / f"{outer_protocol.upper()}_PREDICTION_FREEZE_MANIFEST.json", payload)
    hashes_path = STUDY / "prediction_hashes.json"
    merged = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.exists() else {}
    merged[outer_protocol] = payload
    write_json(hashes_path, merged)
    return payload


def _assert_frozen_for_score(outer_protocol: str, arms: Sequence[str]) -> None:
    path = STUDY / f"{outer_protocol.upper()}_PREDICTION_FREEZE_MANIFEST.json"
    if not path.exists():
        raise RuntimeError("score refused: no global prediction-freeze manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("outer_protocol") != outer_protocol or manifest.get("arms") != list(arms):
        raise RuntimeError("score refused: freeze manifest has different protocol or arms")
    expected = len(COLUMNS) * len(SEEDS) * len(arms)
    if len(manifest.get("files", {})) != expected:
        raise RuntimeError("score refused: incomplete global blind prediction freeze")
    for relative, digest in manifest["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError("score refused: frozen prediction file digest changed")


def _final_prediction(column: str, outer_protocol: str, seed: int, arm: str, test_ids: Sequence[str]) -> np.ndarray:
    path = _final_dir(outer_protocol, column, seed, arm) / "test_predictions_blind.csv.gz"
    frame = pd.read_csv(path)
    if frame.sample_id.astype(str).tolist() != list(test_ids):
        raise RuntimeError(f"new {arm} predictions do not match frozen test order: {path}")
    return frame[["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"]].to_numpy(float)


def _reference_prediction(column: str, outer_protocol: str, seed: int, reference: str,
                          test_ids: Sequence[str]) -> np.ndarray:
    """Read an existing, ID-checked baseline prediction without its metrics."""

    context = f"contexts/{column}/{outer_protocol}/seed_{seed}"
    if reference in ("paper_style_current_v2", "M3_CENTER_WIDTH_FULL"):
        path = ROOT / "studies/transfer/full_data_baseline_finalization/runtime" / context / "predictions_blind.csv.gz"
    elif reference == "HIER_CW_SHARED_LAMBDA_CORRECTED":
        path = ROOT / "studies/transfer/hier_cw_semantic_repair/runtime" / context / "predictions_blind.csv.gz"
    else:
        raise ValueError(f"unknown historical reference: {reference}")
    frame = pd.read_csv(path)
    if frame.sample_id.astype(str).tolist() != list(test_ids):
        raise RuntimeError(f"historical {reference} IDs do not match frozen test order: {path}")
    if reference == "paper_style_current_v2":
        point = frame[["paper_style_current_v2_V1", "paper_style_current_v2_V2"]].to_numpy(float)
    else:
        point = frame[[f"{reference}_V1", f"{reference}_V2"]].to_numpy(float)
    # Structured/paper references are point estimators. Repeat q50 into the
    # six-column layout solely for common point-metric bookkeeping.
    return np.column_stack([point[:, 0], point[:, 0], point[:, 0], point[:, 1], point[:, 1], point[:, 1]])


def _metric_values(truth: np.ndarray, prediction: np.ndarray, scales: Mapping[str, float]) -> dict[str, float | bool]:
    metrics = point_metrics(truth, prediction, scales)
    return {key: (float(value) if isinstance(value, (float, np.floating)) else bool(value)) for key, value in metrics.items()}


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
        rmse_delta = np.empty(draws, dtype=float)
        mae_delta = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            take = np.concatenate([positions[value] for value in sampled])
            cand_error = truth[take, index] - candidate[take, 3 * index + 1]
            ref_error = truth[take, index] - reference[take, 3 * index + 1]
            rmse_delta[draw] = math.sqrt(float(np.mean(cand_error ** 2))) - math.sqrt(float(np.mean(ref_error ** 2)))
            mae_delta[draw] = float(np.mean(np.abs(cand_error))) - float(np.mean(np.abs(ref_error)))
        rows.append({"endpoint": target, "metric": "delta_rmse", "estimate": float(np.mean(rmse_delta)),
                     "ci95_low": float(np.quantile(rmse_delta, .025)), "ci95_high": float(np.quantile(rmse_delta, .975)),
                     "bootstrap_draws": int(draws), "cluster_count": int(len(unique))})
        rows.append({"endpoint": target, "metric": "delta_mae", "estimate": float(np.mean(mae_delta)),
                     "ci95_low": float(np.quantile(mae_delta, .025)), "ci95_high": float(np.quantile(mae_delta, .975)),
                     "bootstrap_draws": int(draws), "cluster_count": int(len(unique))})
    return rows


def _replace_protocol_rows(path: Path, frame: pd.DataFrame, outer_protocol: str) -> None:
    """Preserve a completed other-protocol report when writing a later phase."""

    if path.exists():
        previous = pd.read_csv(path)
        if "protocol" in previous.columns:
            frame = pd.concat([previous.loc[~previous.protocol.eq(outer_protocol)], frame], ignore_index=True)
    write_frame(path, frame)


def score(outer_protocol: str, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read outer truth only after a global candidate/P0 prediction freeze."""

    if outer_protocol not in PROTOCOLS or candidate not in READOUT_ARMS or candidate == "R0":
        raise ValueError("score requires a selected non-R0 candidate and frozen protocol")
    arms = ("R0", candidate)
    _assert_frozen_for_score(outer_protocol, arms)
    result_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    references = ["paper_style_current_v2"] if outer_protocol == "row" else [
        "M3_CENTER_WIDTH_FULL", "HIER_CW_SHARED_LAMBDA_CORRECTED"
    ]
    for column in COLUMNS:
        canonical = FROZEN / f"filtered_canonical_{column}.csv"
        feature = pd.read_csv(FROZEN / f"filtered_features_{column}.csv", usecols=["sample_id", "canonical_smiles"])
        feature["sample_id"] = feature.sample_id.astype(str)
        canonical_by_id = feature.set_index("sample_id").canonical_smiles.astype(str).to_dict()
        for seed in SEEDS:
            context = prepare_context(column, outer_protocol, seed)
            test_ids = context["roles"]["test"]
            truth = _read_authorized_truth(canonical, test_ids)
            predictions: dict[str, np.ndarray] = {
                "P0": _final_prediction(column, outer_protocol, seed, "R0", test_ids),
                candidate: _final_prediction(column, outer_protocol, seed, candidate, test_ids),
            }
            for reference in references:
                predictions[reference] = _reference_prediction(column, outer_protocol, seed, reference, test_ids)
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
                clusters = [canonical_by_id[sample_id] for sample_id in test_ids]
                for row in _clustered_bootstrap_delta(truth, predictions[candidate], predictions[reference], clusters,
                                                     seed=deterministic_seed("bootstrap", outer_protocol, column, seed, candidate, reference)):
                    bootstrap_rows.append({"protocol": outer_protocol, "column": column, "outer_seed": int(seed),
                                           "candidate": candidate, "reference": reference, **row})
    result = pd.DataFrame(result_rows)
    paired = pd.DataFrame(paired_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    target = STUDY / ("ROW_RESULTS.csv" if outer_protocol == "row" else "COMPOUND_RESULTS.csv")
    write_frame(target, result.sort_values(["column", "outer_seed", "method"]))
    _replace_protocol_rows(STUDY / "PAIRED_COMPARISON.csv", paired, outer_protocol)
    _replace_protocol_rows(STUDY / "BOOTSTRAP_CI.csv", bootstrap, outer_protocol)
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
    decision = json.loads((STUDY / "inner_screen_decision.json").read_text(encoding="utf-8")) if (STUDY / "inner_screen_decision.json").exists() else {}
    def compact(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_Not run because the preceding gate did not authorize it._\n"
        subset = frame.groupby(["column", "method"], as_index=False)[["V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse"]].mean(numeric_only=True)
        lines = ["| column | method | V1 RMSE | V1 MAE | V1 R² | V2 RMSE | V2 MAE | V2 R² | combined train-NRMSE |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for value in subset.itertuples(index=False):
            lines.append(f"| {value.column} | {value.method} | {value.V1_rmse:.3f} | {value.V1_mae:.3f} | {value.V1_r2:.3f} | {value.V2_rmse:.3f} | {value.V2_mae:.3f} | {value.V2_r2:.3f} | {value.combined_normalized_rmse:.3f} |")
        return "\n".join(lines) + "\n"
    p0_pair = paired.loc[(paired.protocol.eq("row")) & (paired.reference.eq("P0"))] if not paired.empty else pd.DataFrame()
    row_improvement = {}
    if not p0_pair.empty:
        for column, group in p0_pair.groupby("column"):
            row_improvement[column] = float(100 * (-group.candidate_minus_reference_combined_nrmse.mean()) /
                                             (row.loc[(row.column.eq(column)) & (row.method.eq("P0")), "combined_normalized_rmse"].mean()))
    continuation = bool(len(row_improvement) == 2 and all(value >= 3.0 for value in row_improvement.values()))
    if continuation and compound.empty:
        status = "ROW_CONTINUATION_PASSED__COMPOUND_CONFIRMATION_PENDING"
    elif not continuation:
        status = "NO_MATERIAL_ARCHITECTURE_GAIN_OR_ROW_CONTINUATION_FAILURE"
    else:
        status = "COMPOUND_CONFIRMATION_COMPLETE__PROMOTION_REQUIRES_GATE_REVIEW"
    lines = ["# Final report — conditioned source readout", "", f"## Decision\n\n`{status}`\n",
             "All outer results are developmental confirmation because these frozen outer identities were historically exposed. No score selected the loss, architecture, source feature, or epoch.",
             "", "## ROW results", "", compact(row), "", "## COMPOUND results", "", compact(compound),
             "", "## Seed-wise matched comparison", ""]
    if paired.empty:
        lines.append("_No paired comparison has been scored._")
    else:
        lines.extend(["| protocol | column | candidate | reference | Δ combined NRMSE |", "| --- | --- | --- | --- | ---: |"])
        for value in paired.groupby(["protocol", "column", "candidate", "reference"], as_index=False).candidate_minus_reference_combined_nrmse.mean().itertuples(index=False):
            lines.append(f"| {value.protocol} | {value.column} | {value.candidate} | {value.reference} | {value.candidate_minus_reference_combined_nrmse:.4f} |")
    lines.extend(["", "## Required questions", "",
                  f"1. Fixed sum pooling is a transfer bottleneck only if a gated arm passed the inner/ROW gates; current selection: `{candidate}`.",
                  "2. Adaptive readout (R1) is interpreted only from its matched inner summary, not a single outer score.",
                  "3. Condition-query superiority is interpreted only from R2 versus R1/R0 inner evidence.",
                  "4. Source prediction (R3) and 5. source embedding (R4) are reported only if their preceding gates authorized execution.",
                  f"6. Current ROW P0-relative combined-NRMSE changes: `{row_improvement}`.",
                  "7. COMPOUND is mandatory when ROW continuation passes; its table above is never treated as a tuning screen.",
                  "8. Historical paper-style/P0/structured references were ID-checked and re-scored with common outer-train scales.",
                  "9. Endpoint-level RMSE/MAE are shown above; no aggregate result overrides a material endpoint regression.",
                  f"10. Final promotion status: `{status}`.", "",
                  "## Reproducibility", "",
                  "See `protocol.json`, `run_manifest.csv`, `source_cache/*_cache_manifest.json`, inner context audits, prediction-freeze manifests, `PAIRED_COMPARISON.csv`, and `BOOTSTRAP_CI.csv`. Clustered bootstrap uses canonical SMILES, not IID rows.", ""])
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "prepare-source-cache", "loss-context", "loss-aggregate", "readout-context", "readout-aggregate", "compound-selection-context", "final-context", "freeze", "score", "report"), required=True)
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
    if args.action == "report":
        if args.candidate is None: parser.error("report requires --candidate")
        generate_final_report(args.candidate); return


if __name__ == "__main__":
    main()

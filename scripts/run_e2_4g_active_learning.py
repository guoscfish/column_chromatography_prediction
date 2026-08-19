#!/usr/bin/env python3
"""Run E2 row pilots and bounded compound Round-0/acquisition preflight."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_acquisition import (
    batch_distance_summary,
    build_joint_representation,
    farthest_first_select,
    hybrid_select,
    mean_knn_distance,
    signal_agreement_rows,
    top_score_select,
)
from scripts.al_engine import (
    ActiveLearningState,
    FitResult,
    QGeoGNNActiveLearningEngine,
    SourceFreeTrainConfig,
    canonical_json_hash,
    initialize_round_state,
    load_round_state,
    random_query,
    save_round_state,
)
from scripts.run_e0_4g_baseline import (
    eluent_descriptor,
    minmax_fit,
    sha256_file,
    write_artifact_manifest,
    write_environment,
)
from scripts.run_e1_signal_qualification import condition_matrix, evaluate_signal


SOURCE_DIR = ROOT / "experiments" / "e0_4g_baseline"
D28_DIR = ROOT / "experiments" / "d28_al_engineering"
E1_DIR = ROOT / "experiments" / "e1_signal_qualification"
DEFAULT_OUTPUT = ROOT / "experiments" / "e2_4g_active_learning"
DEFAULT_COMPOUND_PREFLIGHT_OUTPUT = ROOT / "experiments" / "e2_4g_compound_preflight"
OUTER_SEEDS = (42, 525, 1101)
STRATEGIES = ("random", "coverage", "ensemble", "hybrid")
SIGNALS = ("quantile_width", "ensemble", "latent_distance", "random")
MEMBER_COUNT = 3


def git_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    return {"commit": commit, "dirty": bool(status.strip())}


def member_seeds(outer_seed: int) -> tuple[int, ...]:
    return tuple(outer_seed * 100 + index for index in range(MEMBER_COUNT))


def rows_for_ids(engine: QGeoGNNActiveLearningEngine, ids: list[str]) -> pd.DataFrame:
    positions = [engine._sample_to_index[value] for value in ids]
    return engine.data.iloc[positions].reset_index(drop=True)


def indices_for_ids(engine: QGeoGNNActiveLearningEngine, ids: list[str]) -> np.ndarray:
    return np.asarray([engine._sample_to_index[value] for value in ids], dtype=int)


def make_scaler(
    data: pd.DataFrame, graph_cache: dict, partition: pd.DataFrame
) -> dict:
    positions = partition.loc[
        partition["role"].eq("l0_train"), "canonical_index"
    ].to_numpy(dtype=int)
    descriptors = np.vstack(
        [graph_cache[value]["descriptor"] for value in data["canonical_smiles"]]
    ).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in data["PE/EA"]]).astype(
        np.float32
    )
    return {
        "fit_split": "e2_fixed_l0_train",
        "descriptor": minmax_fit(descriptors[positions]),
        "eluent": minmax_fit(eluents[positions]),
    }


def context_for_seed(
    outer_seed: int, output_dir: Path, config: SourceFreeTrainConfig, split_mode: str = "row"
) -> dict[str, Any]:
    data = pd.read_csv(SOURCE_DIR / "canonical_4g.csv")
    graph_cache = torch.load(SOURCE_DIR / "graph_cache_4g.pt", weights_only=False)
    if split_mode not in {"row", "compound"}:
        raise ValueError(f"Unsupported E2 split mode: {split_mode}")
    partition_path = D28_DIR / "partitions" / f"e2_4g_{split_mode}_seed_{outer_seed}.csv"
    partition = pd.read_csv(partition_path)
    scaler = make_scaler(data, graph_cache, partition)
    seed_dir = output_dir / f"{split_mode}_seed_{outer_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = seed_dir / "scaler.json"
    scaler_path.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    engine = QGeoGNNActiveLearningEngine(
        data,
        graph_cache,
        scaler,
        SOURCE_DIR / "checkpoints" / "best.pt",
    )
    role_ids = {
        role: partition.loc[partition["role"].eq(role), "sample_id"].astype(str).tolist()
        for role in ("l0_train", "l0_validation", "u0", "test")
    }
    l0_ids = role_ids["l0_train"] + role_ids["l0_validation"]
    scale_rows = rows_for_ids(engine, role_ids["l0_train"])
    target_scales = {
        target: max(float(scale_rows[f"{target}_ml"].std(ddof=0)), 1e-8)
        for target in ("V1", "V2")
    }
    protocol_hash = canonical_json_hash(
        {
            "train_config": asdict(config),
            "scaler": scaler,
            "partition_sha256": sha256_file(partition_path),
            "member_seeds": member_seeds(outer_seed),
        }
    )
    return {
        "data": data,
        "partition": partition,
        "partition_path": partition_path,
        "partition_hash": sha256_file(partition_path),
        "scaler": scaler,
        "scaler_path": scaler_path,
        "engine": engine,
        "seed_dir": seed_dir,
        "role_ids": role_ids,
        "l0_ids": l0_ids,
        "target_scales": target_scales,
        "protocol_hash": protocol_hash,
    }


def ensure_fit(
    engine: QGeoGNNActiveLearningEngine,
    labeled_ids: list[str],
    validation_ids: list[str],
    config: SourceFreeTrainConfig,
    seed: int,
    output_dir: Path,
) -> tuple[FitResult, float, bool]:
    result_path = output_dir / "fit_result.json"
    checkpoint_path = output_dir / "best.pt"
    expected_labeled = canonical_json_hash(sorted(labeled_ids))
    expected_validation = canonical_json_hash(sorted(validation_ids))
    expected_scaler = canonical_json_hash(engine.scaler)
    if result_path.exists() and checkpoint_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("train_config_hash") == config.config_hash
            and payload.get("labeled_ids_hash") == expected_labeled
            and payload.get("validation_ids_hash") == expected_validation
            and payload.get("scaler_hash") == expected_scaler
            and int(checkpoint.get("seed", -1)) == seed
            and sha256_file(checkpoint_path) == payload.get("checkpoint_sha256")
        ):
            return FitResult(**payload), 0.0, True
        raise ValueError(f"Refusing to overwrite incompatible completed fit: {output_dir}")
    start = time.perf_counter()
    result = engine.fit(
        labeled_ids,
        validation_ids,
        config,
        init_checkpoint=None,
        seed=seed,
        output_dir=output_dir,
    )
    return result, float(time.perf_counter() - start), False


def fit_ensemble(
    context: dict[str, Any],
    labeled_ids: list[str],
    config: SourceFreeTrainConfig,
    output_dir: Path,
    outer_seed: int,
) -> tuple[dict[int, Path], list[dict]]:
    checkpoints, records = {}, []
    for member_index, seed in enumerate(member_seeds(outer_seed)):
        result, seconds, reused = ensure_fit(
            context["engine"],
            labeled_ids,
            context["role_ids"]["l0_validation"],
            config,
            seed,
            output_dir / f"member_{member_index}_seed_{seed}",
        )
        checkpoints[seed] = Path(result.checkpoint)
        records.append(
            {
                "outer_seed": outer_seed,
                "member_index": member_index,
                "member_seed": seed,
                "fit_seconds_current_run": seconds,
                "reused_completed_fit": reused,
                "max_epoch": config.epochs,
                "early_stop_epoch": result.epochs_run,
                "hit_max_epoch": bool(result.epochs_run >= config.epochs),
                "best_epoch_ge_490": bool(result.best_epoch >= 490),
                **asdict(result),
            }
        )
    return checkpoints, records


def ensemble_predict(
    engine: QGeoGNNActiveLearningEngine,
    ids: list[str],
    checkpoints: dict[int, Path],
    target_scales: dict[str, float],
    return_primary_embedding: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    tables, primary_embedding = {}, None
    ordered_seeds = list(checkpoints)
    for member_index, seed in enumerate(ordered_seeds):
        result = engine.predict(
            ids,
            checkpoints[seed],
            return_quantiles=True,
            return_embedding=return_primary_embedding and member_index == 0,
            batch_size=256,
            chunk_size=1024,
        )
        tables[seed] = result.table
        if member_index == 0:
            primary_embedding = result.embeddings
    table = tables[ordered_seeds[0]].copy()
    v1 = np.column_stack(
        [tables[seed]["V1_q50"].to_numpy(dtype=float) for seed in ordered_seeds]
    )
    v2 = np.column_stack(
        [tables[seed]["V2_q50"].to_numpy(dtype=float) for seed in ordered_seeds]
    )
    table["ensemble_pred_V1"] = v1.mean(axis=1)
    table["ensemble_pred_V2"] = v2.mean(axis=1)
    standardized_v1 = v1 / target_scales["V1"]
    standardized_v2 = v2 / target_scales["V2"]
    table["ensemble_variance_V1"] = np.var(standardized_v1, axis=1, ddof=1)
    table["ensemble_variance_V2"] = np.var(standardized_v2, axis=1, ddof=1)
    table["ensemble_covariance_V1_V2"] = np.asarray(
        [np.cov(standardized_v1[row], standardized_v2[row], ddof=1)[0, 1] for row in range(len(table))]
    )
    table["ensemble_score"] = (
        table["ensemble_variance_V1"] + table["ensemble_variance_V2"]
    )
    table["V1_quantile_width"] = table["V1_q90"] - table["V1_q10"]
    table["V2_quantile_width"] = table["V2_q90"] - table["V2_q10"]
    table["quantile_width"] = 0.5 * (
        table["V1_quantile_width"] / target_scales["V1"]
        + table["V2_quantile_width"] / target_scales["V2"]
    )
    for member_index, seed in enumerate(ordered_seeds):
        table[f"member_{member_index}_seed"] = seed
        table[f"member_{member_index}_V1_q50"] = v1[:, member_index]
        table[f"member_{member_index}_V2_q50"] = v2[:, member_index]
    return table, primary_embedding


def add_truth_and_errors(
    context: dict[str, Any], table: pd.DataFrame, primary_member_error: bool
) -> pd.DataFrame:
    result = table.copy()
    truth = rows_for_ids(context["engine"], result["sample_id"].astype(str).tolist())
    result["true_V1"] = truth["V1_ml"].to_numpy(dtype=float)
    result["true_V2"] = truth["V2_ml"].to_numpy(dtype=float)
    pred_v1 = result["member_0_V1_q50"] if primary_member_error else result["ensemble_pred_V1"]
    pred_v2 = result["member_0_V2_q50"] if primary_member_error else result["ensemble_pred_V2"]
    result["standardized_abs_error"] = 0.5 * (
        np.abs(result["true_V1"] - pred_v1) / context["target_scales"]["V1"]
        + np.abs(result["true_V2"] - pred_v2) / context["target_scales"]["V2"]
    )
    result["ensemble_mean_standardized_abs_error"] = 0.5 * (
        np.abs(result["true_V1"] - result["ensemble_pred_V1"])
        / context["target_scales"]["V1"]
        + np.abs(result["true_V2"] - result["ensemble_pred_V2"])
        / context["target_scales"]["V2"]
    )
    result["tail_flag"] = False
    result["compound_id"] = truth["canonical_smiles"].astype(str).to_numpy()
    return result


def representations_and_latent(
    context: dict[str, Any],
    labeled_train_ids: list[str],
    pool_ids: list[str],
    checkpoint: Path,
    pool_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    engine = context["engine"]
    labeled_result = engine.predict(
        labeled_train_ids,
        checkpoint,
        return_quantiles=False,
        return_embedding=True,
        batch_size=256,
        chunk_size=1024,
    )
    labeled_indices = indices_for_ids(engine, labeled_train_ids)
    pool_indices = indices_for_ids(engine, pool_ids)
    labeled_conditions = condition_matrix(engine.data, labeled_indices)
    pool_conditions = condition_matrix(engine.data, pool_indices)
    labeled_repr, pool_repr, audit = build_joint_representation(
        labeled_result.embeddings,
        pool_embedding,
        labeled_conditions,
        pool_conditions,
    )
    latent = mean_knn_distance(labeled_repr, pool_repr, neighbors=5)
    return labeled_repr, pool_repr, latent, audit


def test_metrics(
    context: dict[str, Any], prediction: pd.DataFrame, metadata: dict
) -> dict:
    truth = rows_for_ids(context["engine"], prediction["sample_id"].astype(str).tolist())
    row = dict(metadata)
    for target in ("V1", "V2"):
        true = truth[f"{target}_ml"].to_numpy(dtype=float)
        pred = prediction[f"ensemble_pred_{target}"].to_numpy(dtype=float)
        residual = true - pred
        row[f"{target}_mae"] = float(np.abs(residual).mean())
        row[f"{target}_rmse"] = float(np.sqrt(np.square(residual).mean()))
        denominator = float(np.square(true - true.mean()).sum())
        row[f"{target}_r2"] = float(1.0 - np.square(residual).sum() / denominator) if denominator else float("nan")
    row["nrmse"] = 0.5 * (
        row["V1_rmse"] / context["target_scales"]["V1"]
        + row["V2_rmse"] / context["target_scales"]["V2"]
    )
    row["test_rows"] = len(prediction)
    return row


def random_score(outer_seed: int, ids: list[str]) -> np.ndarray:
    # Diagnostic-only score; acquisition Random uses the persisted AL RNG.
    rng = np.random.default_rng(outer_seed + 9_000_000)
    ordered = sorted(ids)
    mapping = dict(zip(ordered, rng.random(len(ordered))))
    return np.asarray([mapping[value] for value in ids], dtype=float)


def compute_e1_agreement() -> pd.DataFrame:
    frame = pd.read_csv(E1_DIR / "uq_predictions.csv.gz")
    rows = []
    for (split_mode, outer_seed), group in frame.groupby(["split_mode", "outer_seed"]):
        slices = {
            "full": group,
            "common": group[~group["tail_flag"].astype(bool)],
            "tail": group[group["tail_flag"].astype(bool)],
        }
        for slice_name, selected in slices.items():
            if len(selected) < 2:
                continue
            rows.extend(
                signal_agreement_rows(
                    selected,
                    {
                        "stage": "E1_4g_to_8g_transfer",
                        "split": split_mode,
                        "seed": int(outer_seed),
                        "slice": slice_name,
                        "n_samples": len(selected),
                    },
                )
            )
    return pd.DataFrame(rows)


def round0_for_seed(
    context: dict[str, Any],
    config: SourceFreeTrainConfig,
    outer_seed: int,
    split_mode: str = "row",
) -> tuple[dict[int, Path], pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame]:
    shared_dir = context["seed_dir"] / "shared_round_0"
    checkpoints, fit_records = fit_ensemble(
        context, context["l0_ids"], config, shared_dir, outer_seed
    )
    pool, pool_h = ensemble_predict(
        context["engine"],
        context["role_ids"]["u0"],
        checkpoints,
        context["target_scales"],
        True,
    )
    labeled_train_ids = context["role_ids"]["l0_train"]
    labeled_repr, pool_repr, latent, representation_audit = representations_and_latent(
        context,
        labeled_train_ids,
        context["role_ids"]["u0"],
        next(iter(checkpoints.values())),
        pool_h,
    )
    pool["latent_distance"] = latent
    pool["random_score"] = random_score(outer_seed, pool["sample_id"].astype(str).tolist())
    pool = add_truth_and_errors(context, pool, primary_member_error=True)
    pool["outer_seed"] = outer_seed
    pool["split_mode"] = split_mode
    pool["slice"] = "u0_full"
    pool.to_csv(shared_dir / "round0_pool_signals.csv.gz", index=False)
    np.savez_compressed(
        shared_dir / "round0_representations.npz",
        labeled_sample_ids=np.asarray(labeled_train_ids),
        pool_sample_ids=pool["sample_id"].astype(str).to_numpy(),
        labeled_representation=labeled_repr,
        pool_representation=pool_repr,
    )
    (shared_dir / "representation_audit.json").write_text(
        json.dumps(representation_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metric_rows, risk_rows = [], []
    for signal_index, signal in enumerate(SIGNALS):
        metric, risk = evaluate_signal(
            pool,
            signal,
            "full",
            bootstrap_iterations=1000,
            bootstrap_seed=outer_seed * 100 + signal_index,
        )
        metric.update({"split_mode": split_mode, "outer_seed": outer_seed, "evaluation_pool": "U0"})
        metric_rows.append(metric)
        for item in risk:
            item.update({"split_mode": split_mode, "outer_seed": outer_seed, "evaluation_pool": "U0"})
            risk_rows.append(item)
    agreement = pd.DataFrame(
        signal_agreement_rows(
            pool,
            {
                "stage": "E2_source_free_round0",
                "split": split_mode,
                "seed": outer_seed,
                "slice": "u0_full",
                "n_samples": len(pool),
            },
        )
    )
    return checkpoints, pool, fit_records, pd.DataFrame(metric_rows), agreement


def initialize_strategy_state(
    context: dict[str, Any], outer_seed: int, checkpoint: Path
) -> ActiveLearningState:
    return initialize_round_state(
        context["l0_ids"],
        context["role_ids"]["u0"],
        str(checkpoint),
        outer_seed,
        context["partition_hash"],
        context["protocol_hash"],
    )


def condition_distribution(frame: pd.DataFrame) -> dict:
    ratios = frame["PE/EA"].astype(str)
    return {
        "loading_solvent_counts": frame["loading solvent"].astype(str).value_counts().sort_index().to_dict(),
        "pe_ea_top_counts": ratios.value_counts().head(10).to_dict(),
        "mass_mean": float((frame["Density g/ml"] * frame["V/ul"]).mean()),
        "loading_volume_mean": float(frame["Volume of loading solvent/ul"].mean()),
    }


def select_batch(
    strategy: str,
    state: ActiveLearningState,
    pool: pd.DataFrame,
    pool_repr: np.ndarray,
    labeled_repr: np.ndarray,
    batch_size: int,
) -> tuple[ActiveLearningState, list[str], list[str] | None]:
    ids = pool["sample_id"].astype(str).tolist()
    if ids != state.pool_ids:
        raise AssertionError("Pool prediction order does not match persisted state")
    if strategy == "random":
        next_state = random_query(state, batch_size, checkpoint=state.checkpoint)
        return next_state, next_state.selected_ids, None
    if strategy == "coverage":
        selected = farthest_first_select(ids, pool_repr, labeled_repr, batch_size)
        subset = None
    elif strategy == "ensemble":
        selected = top_score_select(ids, pool["ensemble_score"].to_numpy(), batch_size)
        subset = None
    elif strategy == "hybrid":
        selected, subset = hybrid_select(
            ids,
            pool["ensemble_score"].to_numpy(),
            pool_repr,
            labeled_repr,
            batch_size,
            prefilter_fraction=0.25,
        )
    else:
        raise ValueError(strategy)
    selected_set = set(selected)
    next_state = ActiveLearningState(
        version=1,
        round=state.round + 1,
        labeled_ids=state.labeled_ids + selected,
        pool_ids=[value for value in state.pool_ids if value not in selected_set],
        selected_ids=selected,
        checkpoint=state.checkpoint,
        seed=state.seed,
        rng_state=state.rng_state,
        split_hash=state.split_hash,
        config_hash=state.config_hash,
    )
    next_state.validate()
    return next_state, selected, subset


def queried_diagnostics(
    context: dict[str, Any],
    strategy: str,
    outer_seed: int,
    next_state: ActiveLearningState,
    pool: pd.DataFrame,
    pool_repr: np.ndarray,
    selected_ids: list[str],
    fit_records: list[dict],
    hybrid_subset: list[str] | None,
) -> tuple[dict, pd.DataFrame]:
    position = {value: index for index, value in enumerate(pool["sample_id"].astype(str))}
    selected_positions = np.asarray([position[value] for value in selected_ids], dtype=int)
    selected = pool.iloc[selected_positions].copy().reset_index(drop=True)
    selected_repr = pool_repr[selected_positions]
    mean_distance, min_distance = batch_distance_summary(selected_repr)
    truth_rows = rows_for_ids(context["engine"], selected_ids)
    record = {
        "round": next_state.round,
        "strategy": strategy,
        "seed": outer_seed,
        "selected_total": len(selected_ids),
        "selected_sample_ids": json.dumps(selected_ids),
        "selected_unique_compounds": int(truth_rows["canonical_smiles"].nunique()),
        "selected_mean_true_error_after_reveal": float(
            selected["ensemble_mean_standardized_abs_error"].mean()
        ),
        "selected_mean_ensemble_uncertainty": float(selected["ensemble_score"].mean()),
        "selected_mean_quantile_width": float(selected["quantile_width"].mean()),
        "selected_mean_latent_distance": float(selected["latent_distance"].mean()),
        "batch_mean_pairwise_latent_distance": mean_distance,
        "batch_min_pairwise_latent_distance": min_distance,
        "condition_distribution": json.dumps(condition_distribution(truth_rows), ensure_ascii=False),
        "best_epoch": json.dumps([int(item["best_epoch"]) for item in fit_records]),
        "hit_max_epoch": bool(any(item["hit_max_epoch"] for item in fit_records)),
        "best_epoch_ge_490": bool(any(item["best_epoch_ge_490"] for item in fit_records)),
        "hybrid_prefilter_count": len(hybrid_subset) if hybrid_subset is not None else None,
        "hybrid_prefilter_fraction": 0.25 if hybrid_subset is not None else None,
    }
    selected["round"] = next_state.round
    selected["strategy"] = strategy
    selected["seed"] = outer_seed
    return record, selected


def run_strategy(
    context: dict[str, Any],
    config: SourceFreeTrainConfig,
    outer_seed: int,
    strategy: str,
    shared_checkpoints: dict[int, Path],
    shared_fit_records: list[dict],
    rounds: int,
    query_size: int,
) -> None:
    strategy_dir = context["seed_dir"] / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    state = initialize_strategy_state(
        context, outer_seed, next(iter(shared_checkpoints.values()))
    )
    for round_index in range(rounds + 1):
        round_dir = strategy_dir / f"round_{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        fit_state_path = round_dir / "state_after_fit.json"
        metrics_path = round_dir / "round_metrics.json"
        query_state_path = round_dir / "state_after_query.json"
        round_complete = (
            fit_state_path.exists()
            and metrics_path.exists()
            and (round_index == rounds or query_state_path.exists())
        )
        if round_complete:
            continue
        if round_index == 0:
            checkpoints, fit_records = shared_checkpoints, shared_fit_records
        else:
            previous_state_path = strategy_dir / f"round_{round_index - 1}" / "state_after_query.json"
            state = load_round_state(
                previous_state_path,
                context["partition_hash"],
                context["protocol_hash"],
            )
            checkpoints, fit_records = fit_ensemble(
                context, state.labeled_ids, config, round_dir / "checkpoints", outer_seed
            )
        state = replace(state, checkpoint=str(next(iter(checkpoints.values()))))
        save_round_state(round_dir / "state_after_fit.json", state)
        pd.DataFrame(fit_records).assign(strategy=strategy, round=round_index).to_csv(
            round_dir / "convergence.csv", index=False
        )

        test_prediction, _ = ensemble_predict(
            context["engine"],
            context["role_ids"]["test"],
            checkpoints,
            context["target_scales"],
            False,
        )
        test_prediction = add_truth_and_errors(context, test_prediction, False)
        test_prediction["strategy"] = strategy
        test_prediction["outer_seed"] = outer_seed
        test_prediction["round"] = round_index
        test_prediction.to_csv(round_dir / "test_predictions.csv.gz", index=False)
        metrics = test_metrics(
            context,
            test_prediction,
            {
                "strategy": strategy,
                "outer_seed": outer_seed,
                "round": round_index,
                "labeled_total": len(state.labeled_ids),
                "train_labeled": len(state.labeled_ids) - len(context["role_ids"]["l0_validation"]),
                "validation_labeled": len(context["role_ids"]["l0_validation"]),
            },
        )
        (round_dir / "round_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if round_index == rounds:
            continue

        pool, pool_h = ensemble_predict(
            context["engine"],
            state.pool_ids,
            checkpoints,
            context["target_scales"],
            True,
        )
        labeled_train_ids = [
            value
            for value in state.labeled_ids
            if value not in set(context["role_ids"]["l0_validation"])
        ]
        labeled_repr, pool_repr, latent, representation_audit = representations_and_latent(
            context,
            labeled_train_ids,
            state.pool_ids,
            next(iter(checkpoints.values())),
            pool_h,
        )
        pool["latent_distance"] = latent
        pool["random_score"] = random_score(
            outer_seed + round_index * 10_000, pool["sample_id"].astype(str).tolist()
        )
        pool = add_truth_and_errors(context, pool, False)
        pool.to_csv(round_dir / "pool_predictions.csv.gz", index=False)
        (round_dir / "representation_audit.json").write_text(
            json.dumps(representation_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        next_state, selected_ids, subset = select_batch(
            strategy, state, pool, pool_repr, labeled_repr, query_size
        )
        diagnostic, selected = queried_diagnostics(
            context,
            strategy,
            outer_seed,
            next_state,
            pool,
            pool_repr,
            selected_ids,
            fit_records,
            subset,
        )
        selected.to_csv(round_dir / "selected_after_reveal.csv", index=False)
        (round_dir / "queried_batch_diagnostics.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_round_state(round_dir / "state_after_query.json", next_state)
        print(
            json.dumps(
                {
                    "completed_acquisition": True,
                    "strategy": strategy,
                    "outer_seed": outer_seed,
                    "round": next_state.round,
                    "labeled_total": len(next_state.labeled_ids),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def compound_preflight(output_dir: Path, config: SourceFreeTrainConfig) -> None:
    """Run only compound seed-42 Round-0 and Round-1 acquisition dry-run."""
    split_mode = "compound"
    seed = 42
    context = context_for_seed(seed, output_dir, config, split_mode=split_mode)
    partition = context["partition"]
    role_sets = {
        role: set(partition.loc[partition["role"].eq(role), "canonical_smiles"].astype(str))
        for role in ("l0_train", "l0_validation", "u0", "test")
    }
    leakage = {
        "l0_train_test_overlap": sorted(role_sets["l0_train"] & role_sets["test"]),
        "l0_validation_test_overlap": sorted(role_sets["l0_validation"] & role_sets["test"]),
        "u0_test_overlap": sorted(role_sets["u0"] & role_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Compound partition leakage detected: {leakage}")
    counts = partition["role"].value_counts().to_dict()
    if counts != {"u0": 3375, "test": 413, "l0_train": 318, "l0_validation": 57}:
        raise RuntimeError(f"Unexpected compound seed42 counts: {counts}")
    if partition["sample_id"].astype(str).nunique() != len(partition):
        raise RuntimeError("Compound partition sample_id values are not stable and unique")
    write_environment(output_dir)
    (output_dir / "config.json").write_text(
        json.dumps({
            "stage": "E2_compound_seed42_preflight",
            "split_mode": "compound",
            "outer_seed": 42,
            "train_config": asdict(config),
            "member_count": MEMBER_COUNT,
            "query_size": 25,
            "stop_after": "Round-0 training and Round-1 acquisition dry-run",
            "partition_path": str(context["partition_path"].relative_to(ROOT)),
            "partition_sha256": context["partition_hash"],
            "scaler_policy": "fixed L0_train only",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoints, pool, fit_records, metrics, agreement = round0_for_seed(
        context, config, seed, split_mode=split_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "round0_signal_diagnostics.csv", index=False)
    agreement.to_csv(output_dir / "signal_agreement.csv", index=False)
    pd.DataFrame(fit_records).to_csv(output_dir / "round0_convergence_audit.csv", index=False)
    signal_summary = metrics.groupby("signal", as_index=False).agg(
        mean_spearman=("spearman", "mean"),
        mean_auroc=("hard_error_auroc", "mean"),
        mean_enrichment=("enrichment", "mean"),
        mean_ause=("ause", "mean"),
    )
    signal_summary.to_csv(output_dir / "round0_signal_summary.csv", index=False)
    state = initialize_strategy_state(context, seed, next(iter(checkpoints.values())))
    labeled_train_ids = context["role_ids"]["l0_train"]
    pool_h = context["engine"].predict(
        context["role_ids"]["u0"], next(iter(checkpoints.values())),
        return_quantiles=False, return_embedding=True, batch_size=256, chunk_size=1024,
    ).embeddings
    labeled_repr, pool_repr, latent, representation_audit = representations_and_latent(
        context, labeled_train_ids, context["role_ids"]["u0"],
        next(iter(checkpoints.values())), pool_h
    )
    pool = pool.copy()
    pool["latent_distance"] = latent
    pool["random_score"] = random_score(seed, pool["sample_id"].astype(str).tolist())
    pool.to_csv(output_dir / "round0_pool_signals.csv.gz", index=False)
    preflight_rows = []
    selected_by_strategy = {}
    test_ids = set(context["role_ids"]["test"])
    for strategy in STRATEGIES:
        next_state, selected_ids, subset = select_batch(
            strategy, state, pool, pool_repr, labeled_repr, 25
        )
        diagnostic, selected = queried_diagnostics(
            context, strategy, seed, next_state, pool, pool_repr, selected_ids, fit_records, subset
        )
        if len(selected_ids) != 25 or len(set(selected_ids)) != 25:
            raise RuntimeError(f"{strategy} dry-run did not select 25 unique samples")
        if set(selected_ids) & test_ids:
            raise RuntimeError(f"{strategy} dry-run leaked test sample IDs")
        if strategy == "hybrid" and not set(selected_ids).issubset(set(subset or [])):
            raise RuntimeError("Hybrid selection escaped its Top-25% prefilter")
        selected_by_strategy[strategy] = selected_ids
        selected.to_csv(output_dir / f"round1_{strategy}_selected.csv", index=False)
        preflight_rows.append(diagnostic)
    coverage_repeat, _, _ = select_batch(
        "coverage", state, pool, pool_repr, labeled_repr, 25
    )
    if coverage_repeat.selected_ids != selected_by_strategy["coverage"]:
        raise RuntimeError("Coverage acquisition is not deterministic")
    pd.DataFrame(preflight_rows).to_csv(output_dir / "round1_acquisition_dry_run.csv", index=False)
    (output_dir / "representation_audit.json").write_text(
        json.dumps(representation_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "preflight_decision.json").write_text(
        json.dumps({
            "stage": "E2_compound_seed42_preflight",
            "complete": True,
            "split_mode": "compound",
            "outer_seed": 42,
            "round0_training": "K=3 source-free members",
            "round1_acquisition_only": True,
            "full_compound_pilot_started": False,
            "compound_leakage": leakage,
            "counts": counts,
            "validation_counted_in_label_budget": True,
            "l0_total": counts["l0_train"] + counts["l0_validation"],
            "sample_ids_unique": True,
            "coverage_deterministic": True,
            "all_selected_batches_unique": True,
            "all_selected_batches_test_leakage_free": True,
            "hybrid_prefilter_valid": True,
            "next_action": "review preflight before full compound pilot",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# E2 Compound Seed-42 Preflight\n\n"
        "This directory contains only source-free Round-0 K=3 diagnostics and a Round-1 "
        "acquisition dry-run for Random, Coverage, Ensemble and Hybrid. No Round-1 model was "
        "trained and the full compound pilot was not started.\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output_dir)


def collect_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, queried, convergence = [], [], []
    for path in output_dir.glob("row_seed_*/[a-z]*/round_*/round_metrics.json"):
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    for path in output_dir.glob("row_seed_*/[a-z]*/round_*/queried_batch_diagnostics.json"):
        queried.append(json.loads(path.read_text(encoding="utf-8")))
    for path in output_dir.glob("row_seed_*/[a-z]*/round_*/convergence.csv"):
        convergence.append(pd.read_csv(path))
    metrics_df = pd.DataFrame(metrics).sort_values(["outer_seed", "strategy", "round"])
    queried_df = pd.DataFrame(queried).sort_values(["seed", "strategy", "round"])
    convergence_df = pd.concat(convergence, ignore_index=True).sort_values(
        ["outer_seed", "strategy", "round", "member_index"]
    )
    metrics_df.to_csv(output_dir / "round_metrics.csv", index=False)
    queried_df.to_csv(output_dir / "queried_batch_diagnostics.csv", index=False)
    convergence_df.to_csv(output_dir / "convergence_audit.csv", index=False)
    return metrics_df, queried_df, convergence_df


def calculate_aulc(metrics: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (seed, strategy), group in metrics.groupby(["outer_seed", "strategy"]):
        group = group.sort_values("labeled_total")
        labels = group["labeled_total"].to_numpy(dtype=float)
        values = group["nrmse"].to_numpy(dtype=float)
        raw = float(np.trapz(values, labels))
        normalized = float(raw / (labels[-1] - labels[0]))
        rows.append(
            {
                "outer_seed": seed,
                "strategy": strategy,
                "aulc_raw": raw,
                "aulc_normalized": normalized,
                "final_nrmse": float(values[-1]),
            }
        )
    aulc = pd.DataFrame(rows)
    random = aulc[aulc["strategy"].eq("random")].set_index("outer_seed")
    effects = []
    for strategy in STRATEGIES[1:]:
        selected = aulc[aulc["strategy"].eq(strategy)].set_index("outer_seed")
        differences = selected["aulc_normalized"] - random["aulc_normalized"]
        mean = float(differences.mean())
        sd = float(differences.std(ddof=1))
        margin = 4.303 * sd / math.sqrt(len(differences))
        effects.append(
            {
                "strategy": strategy,
                "comparison": f"{strategy}-random",
                "mean_paired_difference": mean,
                "standard_deviation": sd,
                "paired_ci_low": mean - margin,
                "paired_ci_high": mean + margin,
                "win_count": int((differences < 0).sum()),
                "seeds": len(differences),
                "differences_by_seed": json.dumps(differences.to_dict()),
            }
        )
    aulc.to_csv(output_dir / "aulc_summary.csv", index=False)
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(output_dir / "paired_effects.csv", index=False)
    return aulc, effects_df


def label_efficiency(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in OUTER_SEEDS:
        random_curve = metrics[
            metrics["outer_seed"].eq(seed) & metrics["strategy"].eq("random")
        ].sort_values("labeled_total")
        threshold = float(random_curve.iloc[-1]["nrmse"])
        random_final_labels = int(random_curve.iloc[-1]["labeled_total"])
        for strategy in STRATEGIES:
            curve = metrics[
                metrics["outer_seed"].eq(seed) & metrics["strategy"].eq(strategy)
            ].sort_values("labeled_total")
            cumulative_best = np.minimum.accumulate(curve["nrmse"].to_numpy(dtype=float))
            hit = np.flatnonzero(cumulative_best <= threshold)
            labels = int(curve.iloc[hit[0]]["labeled_total"]) if len(hit) else None
            rows.append(
                {
                    "outer_seed": seed,
                    "strategy": strategy,
                    "target": "random_final_nrmse",
                    "target_nrmse": threshold,
                    "labels_required": labels,
                    "labels_saved_vs_random_final": random_final_labels - labels if labels is not None else None,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "label_efficiency.csv", index=False)
    return result


def make_plots(
    output_dir: Path,
    metrics: pd.DataFrame,
    aulc: pd.DataFrame,
    round0: pd.DataFrame,
    agreement: pd.DataFrame,
    queried: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = {"random": "#9D9D9D", "coverage": "#54A24B", "ensemble": "#F58518", "hybrid": "#4C78A8"}
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for strategy in STRATEGIES:
        selected = metrics[metrics["strategy"].eq(strategy)]
        for _, seed_curve in selected.groupby("outer_seed"):
            seed_curve = seed_curve.sort_values("labeled_total")
            ax.plot(seed_curve["labeled_total"], seed_curve["nrmse"], color=colors[strategy], alpha=0.22)
        mean = selected.groupby("labeled_total", as_index=False)["nrmse"].mean()
        ax.plot(mean["labeled_total"], mean["nrmse"], color=colors[strategy], label=strategy, linewidth=2)
    ax.set(xlabel="number of labeled samples", ylabel="normalized RMSE", title="E2 row pilot learning curves")
    ax.legend()
    fig.savefig(plot_dir / "learning_curve_nrmse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    pivot = aulc.pivot(index="outer_seed", columns="strategy", values="aulc_normalized")
    x = np.arange(len(STRATEGIES))
    for _, row in pivot.iterrows():
        ax.plot(x, [row[s] for s in STRATEGIES], marker="o", alpha=0.5)
    ax.set(xticks=x, xticklabels=STRATEGIES, ylabel="normalized AULC", title="Paired AULC by outer seed")
    fig.savefig(plot_dir / "paired_aulc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    summary = round0.groupby("signal")["spearman"].agg(["mean", "std"]).reindex(SIGNALS)
    ax.bar(summary.index, summary["mean"], yerr=summary["std"], color=["#4C78A8", "#F58518", "#54A24B", "#9D9D9D"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(ylabel="Spearman(signal, error)", title="E2 source-free Round-0 signal sanity")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(plot_dir / "round0_signal_error.png", dpi=180)
    plt.close(fig)

    e2_agreement = agreement[agreement["stage"].eq("E2_source_free_round0")]
    pair_labels = [f"{a}\nvs\n{b}" for a, b in zip(e2_agreement["signal_A"], e2_agreement["signal_B"])]
    matrix = e2_agreement.pivot_table(index="seed", columns=["signal_A", "signal_B"], values="spearman")
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set(yticks=np.arange(len(matrix.index)), yticklabels=matrix.index, xticks=np.arange(len(matrix.columns)), xticklabels=[f"{a}/{b}" for a, b in matrix.columns])
    ax.tick_params(axis="x", rotation=20)
    fig.colorbar(image, ax=ax, label="Spearman")
    fig.savefig(plot_dir / "signal_agreement_heatmap.png", dpi=180)
    plt.close(fig)

    overlap = e2_agreement.groupby(["signal_A", "signal_B"])[["top5_overlap", "top10_overlap", "top20_overlap"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    overlap.plot(kind="bar", ax=ax)
    ax.set(ylabel="Top-K overlap fraction", xlabel="signal pair", ylim=(0, 1), title="E2 Round-0 signal overlap")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(plot_dir / "topk_signal_overlap.png", dpi=180)
    plt.close(fig)

    for column, filename, ylabel in (
        ("batch_mean_pairwise_latent_distance", "batch_diversity_by_round.png", "mean pairwise latent distance"),
        ("selected_mean_true_error_after_reveal", "selected_true_error_by_round.png", "selected true standardized error"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for strategy in STRATEGIES:
            selected = queried[queried["strategy"].eq(strategy)]
            mean = selected.groupby("round", as_index=False)[column].mean()
            ax.plot(mean["round"], mean[column], marker="o", color=colors[strategy], label=strategy)
        ax.set(xlabel="acquisition round", ylabel=ylabel)
        ax.legend()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for strategy, selected in convergence.groupby("strategy"):
        ax.hist(selected["best_epoch"], bins=20, alpha=0.4, label=strategy, color=colors[strategy])
    ax.axvline(490, color="black", linestyle="--", label="490")
    ax.set(xlabel="best epoch", ylabel="fit count", title="Convergence audit")
    ax.legend()
    fig.savefig(plot_dir / "best_epoch_distribution.png", dpi=180)
    plt.close(fig)


def finalize(output_dir: Path) -> None:
    metrics, queried, convergence = collect_outputs(output_dir)
    expected_metrics = len(OUTER_SEEDS) * len(STRATEGIES) * 9
    expected_queries = len(OUTER_SEEDS) * len(STRATEGIES) * 8
    if len(metrics) != expected_metrics or len(queried) != expected_queries:
        raise RuntimeError(
            f"Pilot incomplete: metrics {len(metrics)}/{expected_metrics}, queries {len(queried)}/{expected_queries}"
        )
    aulc, effects = calculate_aulc(metrics, output_dir)
    label_efficiency(metrics, output_dir)
    round0 = pd.read_csv(output_dir / "round0_signal_diagnostics.csv")
    agreement = pd.read_csv(output_dir / "signal_agreement.csv")
    make_plots(output_dir, metrics, aulc, round0, agreement, queried, convergence)
    late_fraction = float(convergence["best_epoch_ge_490"].mean())
    best_strategy = str(
        aulc.groupby("strategy")["aulc_normalized"].mean().sort_values().index[0]
    )
    active_wins = effects.loc[effects["mean_paired_difference"] < 0, "strategy"].tolist()
    decision = {
        "stage": "E2_row_3seed_pilot",
        "complete": True,
        "scientific_result_interpretable": True,
        "best_mean_aulc_strategy": best_strategy,
        "strategies_with_mean_aulc_better_than_random": active_wins,
        "any_active_strategy_wins_all_three_seeds": bool((effects["win_count"] == 3).any()),
        "hybrid_better_than_both_single_strategies_mean_aulc": bool(
            aulc.groupby("strategy")["aulc_normalized"].mean()["hybrid"]
            < min(
                aulc.groupby("strategy")["aulc_normalized"].mean()["coverage"],
                aulc.groupby("strategy")["aulc_normalized"].mean()["ensemble"],
            )
        ),
        "best_epoch_ge_490_fraction": late_fraction,
        "convergence_decision_required": bool(late_fraction > 0.20),
        "tail_interpreted": False,
        "new_methods_added": False,
        "next_stage": "E2 compound pilot" if active_wins else "E2 failure audit before compound",
    }
    (output_dir / "e2_row_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mean_aulc = aulc.groupby("strategy")["aulc_normalized"].mean().sort_values()
    readme = f"""# E2 4g Active Learning — Row 3-seed Pilot

## 状态

本目录是正式E2 row协议结果，不与`e2_random_smoke`混淆。3个paired outer seeds、4种策略、K=3、L0=375、B=25、8轮均已完成；每个预算点都从seeded random source-free初始化重新训练，固定L0-train scaler与validation。

平均normalized AULC（越低越好）：

```text
{mean_aulc.to_string()}
```

- 最低平均AULC：**{best_strategy}**。
- 平均AULC优于Random的策略：**{', '.join(active_wins) if active_wins else '无'}**。
- Hybrid平均AULC同时优于Coverage和Ensemble：**{decision['hybrid_better_than_both_single_strategies_mean_aulc']}**。
- best_epoch>=490比例：**{late_fraction:.1%}**；是否触发单独convergence decision：**{decision['convergence_decision_required']}**。

## 解释边界

Round-0离线signal诊断只回答E1信号能否迁移到source-free regime，不是主动学习结论。正式科学比较来自3-seed完整learning curve与paired AULC。4g canonical数据历史上已删除60/120 mL tail，因此E2不解释tail acquisition；该机制留到E4 no-threshold 8g。Quantile Width保留在离线诊断与后续E4 Protocol B legacy baseline中，但不是E2第五种策略。

完整数值见`round_metrics.csv`、`aulc_summary.csv`、`paired_effects.csv`、`label_efficiency.csv`、`round0_signal_diagnostics.csv`、`signal_agreement.csv`、`queried_batch_diagnostics.csv`与`convergence_audit.csv`。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def write_static_config(
    output_dir: Path, config: SourceFreeTrainConfig, rounds: int, query_size: int
) -> None:
    partitions = {}
    for seed in OUTER_SEEDS:
        path = D28_DIR / "partitions" / f"e2_4g_row_seed_{seed}.csv"
        partitions[str(seed)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    payload = {
        "stage": "E2_4g_active_learning_row_pilot",
        "outer_seeds": list(OUTER_SEEDS),
        "strategies": list(STRATEGIES),
        "member_count": MEMBER_COUNT,
        "member_seed_rule": "outer_seed*100 + member_index",
        "rounds": rounds,
        "query_size": query_size,
        "budgets": [375 + query_size * value for value in range(rounds + 1)],
        "train_config": asdict(config),
        "source_free": True,
        "scaler": "fit fixed L0_train once per outer seed; freeze across rounds and strategies",
        "performance_primary": "0.5*(RMSE_V1/L0_train_sd_V1 + RMSE_V2/L0_train_sd_V2)",
        "aulc": "trapezoid over labels; normalized by label range",
        "round0_primary_error": "member0 mean standardized V1/V2 absolute error, matching E1; ensemble-mean error retained as sensitivity",
        "coverage": {
            "representation": "[128D h_graph; 9D conditions]",
            "normalization": "featurewise z-score of each block fitted on current non-validation labeled train",
            "distance": "Euclidean",
            "batch": "sequential farthest-first/k-center",
            "first_center": "candidate farthest from current labeled train",
            "tie_breaking": "lexicographically smallest sample_id",
            "within_batch_update": True,
        },
        "ensemble": {
            "score": "trace of sample covariance of L0-train-scale standardized V1/V2 q50",
            "members": 3,
            "independence": "separate seeded random initialization; current Lt only",
        },
        "hybrid": {
            "prefilter": "top 25% by frozen ensemble score",
            "selection": "same farthest-first implementation as Coverage",
        },
        "partitions": partitions,
        "source_files_sha256": {
            "scripts/al_engine.py": sha256_file(ROOT / "scripts" / "al_engine.py"),
            "scripts/al_acquisition.py": sha256_file(ROOT / "scripts" / "al_acquisition.py"),
            "scripts/run_e2_4g_active_learning.py": sha256_file(
                ROOT / "scripts" / "run_e2_4g_active_learning.py"
            ),
        },
        "scheduler": "outer-seed workers may run in parallel; each seed directory is isolated and scientific configuration is unchanged",
        "git": git_metadata(),
        "parquet_status": "csv.gz used because fish environment has no pyarrow/fastparquet",
        "tail": "not analyzed; absent structurally from legacy-threshold 4g canonical data",
    }
    config_path = output_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        runtime_keys = {"git", "source_files_sha256", "scheduler"}
        comparable_previous = {
            key: value for key, value in previous.items() if key not in runtime_keys
        }
        comparable_new = {
            key: value for key, value in payload.items() if key not in runtime_keys
        }
        if comparable_previous != comparable_new:
            raise ValueError("Existing E2 output config differs; use a new output directory")
        previous.update({key: payload[key] for key in runtime_keys})
        config_path.write_text(
            json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "partition_manifest.json").write_text(
        json.dumps(partitions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_round0(output_dir: Path, config: SourceFreeTrainConfig) -> None:
    all_metrics, all_agreement, all_fits = [], [], []
    for seed in OUTER_SEEDS:
        context = context_for_seed(seed, output_dir, config)
        _, _, fit_records, metrics, agreement = round0_for_seed(context, config, seed)
        all_metrics.append(metrics)
        all_agreement.append(agreement)
        all_fits.extend([{**item, "strategy": "shared_round0", "round": 0} for item in fit_records])
        print(json.dumps({"round0_completed": True, "outer_seed": seed}, ensure_ascii=False), flush=True)
    round0 = pd.concat(all_metrics, ignore_index=True)
    round0.to_csv(output_dir / "round0_signal_diagnostics.csv", index=False)
    e1 = compute_e1_agreement()
    agreement = pd.concat([e1, *all_agreement], ignore_index=True)
    agreement.to_csv(output_dir / "signal_agreement.csv", index=False)
    pd.DataFrame(all_fits).to_csv(output_dir / "round0_convergence_audit.csv", index=False)
    signal_summary = (
        round0.groupby("signal")
        .agg(
            seeds=("outer_seed", "nunique"),
            positive_spearman_seeds=("spearman", lambda values: int((values > 0).sum())),
            mean_spearman=("spearman", "mean"),
            mean_auroc=("hard_error_auroc", "mean"),
            mean_enrichment=("enrichment", "mean"),
            mean_ause=("ause", "mean"),
        )
        .reset_index()
    )
    signal_summary.to_csv(output_dir / "round0_signal_summary.csv", index=False)
    signal_key = signal_summary.set_index("signal")
    decision = {
        "stage": "E2_source_free_round0_diagnostic",
        "complete": True,
        "active_learning_scientific_conclusion": False,
        "ensemble_positive_spearman_seeds": int(
            signal_key.loc["ensemble", "positive_spearman_seeds"]
        ),
        "latent_positive_spearman_seeds": int(
            signal_key.loc["latent_distance", "positive_spearman_seeds"]
        ),
        "ensemble_regime_failure_flag": bool(
            signal_key.loc["ensemble", "positive_spearman_seeds"] < 2
            or signal_key.loc["ensemble", "mean_spearman"] <= 0.05
        ),
        "latent_regime_failure_flag": bool(
            signal_key.loc["latent_distance", "positive_spearman_seeds"] < 2
            or signal_key.loc["latent_distance", "mean_spearman"] <= 0.05
        ),
        "failure_rule": "diagnostic flag if positive Spearman in fewer than 2/3 seeds or mean Spearman <= 0.05; frozen E1 definitions are not changed",
        "next_stage_regardless_of_flag": "E2 row Random/Coverage/Ensemble/Hybrid pilot",
    }
    (output_dir / "round0_regime_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#9D9D9D"]
    ordered = signal_summary.set_index("signal").reindex(SIGNALS)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(ordered.index, ordered["mean_spearman"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(ylabel="mean Spearman(signal, error)", title="E2 source-free Round-0 signal sanity")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(plot_dir / "round0_signal_error.png", dpi=180)
    plt.close(fig)
    e2_agreement = agreement[agreement["stage"].eq("E2_source_free_round0")]
    matrix = e2_agreement.pivot_table(
        index="seed", columns=["signal_A", "signal_B"], values="spearman"
    )
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set(
        yticks=np.arange(len(matrix.index)),
        yticklabels=matrix.index,
        xticks=np.arange(len(matrix.columns)),
        xticklabels=[f"{left}/{right}" for left, right in matrix.columns],
    )
    ax.tick_params(axis="x", rotation=20)
    fig.colorbar(image, ax=ax, label="Spearman")
    fig.savefig(plot_dir / "signal_agreement_heatmap.png", dpi=180)
    plt.close(fig)
    overlap = e2_agreement.groupby(["signal_A", "signal_B"])[
        ["top5_overlap", "top10_overlap", "top20_overlap"]
    ].mean()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    overlap.plot(kind="bar", ax=ax)
    ax.set(
        ylabel="Top-K overlap fraction",
        xlabel="signal pair",
        ylim=(0, 1),
        title="E2 Round-0 signal overlap",
    )
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(plot_dir / "topk_signal_overlap.png", dpi=180)
    plt.close(fig)
    progress_readme = f"""# E2 4g Active Learning — in progress

Round-0 source-free signal diagnostic已完成；它只检验E1信号能否迁移到E2模型regime，不是主动学习科学结论。

- Ensemble regime failure flag：**{decision['ensemble_regime_failure_flag']}**。
- Latent Distance regime failure flag：**{decision['latent_regime_failure_flag']}**。
- 无论flag结果如何，下一步仍按冻结协议运行Random/Coverage/Ensemble/Hybrid三seed完整row learning curves。

正式E2结论必须等待3 seeds × 4 strategies × 9 budgets × K=3全部完成后，由paired AULC给出。
"""
    (output_dir / "README.md").write_text(progress_readme, encoding="utf-8")
    write_artifact_manifest(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--phase", choices=("round0", "pilot", "worker", "finalize", "preflight", "all"), default="all"
    )
    parser.add_argument("--split-mode", choices=("row", "compound"), default="row")
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument(
        "--worker-strategy",
        choices=STRATEGIES,
        help="Run only one strategy for an isolated seed worker (scheduler-only).",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--query-size", type=int, default=25)
    args = parser.parse_args()
    if args.rounds != 8 or args.query_size != 25:
        raise ValueError("Formal E2 protocol freezes rounds=8 and query_size=25")
    output_dir = (
        args.output_dir
        or (DEFAULT_COMPOUND_PREFLIGHT_OUTPUT if args.phase == "preflight" else DEFAULT_OUTPUT)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)
    config = SourceFreeTrainConfig(epochs=args.epochs, patience=args.patience)
    config.validate_frozen_predictor()
    if args.phase == "preflight":
        if args.split_mode != "compound":
            raise ValueError("--phase preflight requires --split-mode compound")
        compound_preflight(output_dir, config)
        print(json.dumps({"preflight_complete": True, "split_mode": "compound", "outer_seed": 42}), flush=True)
        return
    write_static_config(output_dir, config, args.rounds, args.query_size)

    if args.phase == "worker":
        if args.worker_seed not in OUTER_SEEDS:
            raise ValueError(f"--worker-seed must be one of {OUTER_SEEDS}")
        if not (output_dir / "round0_signal_diagnostics.csv").exists():
            raise RuntimeError("Run --phase round0 before workers")
        context = context_for_seed(args.worker_seed, output_dir, config)
        shared_checkpoints, shared_fit_records = fit_ensemble(
            context,
            context["l0_ids"],
            config,
            context["seed_dir"] / "shared_round_0",
            args.worker_seed,
        )
        worker_strategies = (
            [args.worker_strategy] if args.worker_strategy is not None else STRATEGIES
        )
        for strategy in worker_strategies:
            run_strategy(
                context,
                config,
                args.worker_seed,
                strategy,
                shared_checkpoints,
                shared_fit_records,
                args.rounds,
                args.query_size,
            )
        print(json.dumps({"worker_complete": True, "outer_seed": args.worker_seed}), flush=True)
        return

    if args.phase in ("round0", "all"):
        run_round0(output_dir, config)
    if args.phase in ("pilot", "all"):
        if not (output_dir / "round0_signal_diagnostics.csv").exists():
            raise RuntimeError("Run --phase round0 before the formal pilot")
        for outer_seed in OUTER_SEEDS:
            context = context_for_seed(outer_seed, output_dir, config)
            shared_checkpoints, shared_fit_records = fit_ensemble(
                context,
                context["l0_ids"],
                config,
                context["seed_dir"] / "shared_round_0",
                outer_seed,
            )
            for strategy in STRATEGIES:
                run_strategy(
                    context,
                    config,
                    outer_seed,
                    strategy,
                    shared_checkpoints,
                    shared_fit_records,
                    args.rounds,
                    args.query_size,
                )
        finalize(output_dir)
    elif args.phase == "finalize":
        finalize(output_dir)


if __name__ == "__main__":
    main()

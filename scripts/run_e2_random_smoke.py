#!/usr/bin/env python3
"""Run the minimal E2 source-free Random active-learning chain smoke test."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_engine import (
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
from scripts.run_e1_signal_qualification import condition_matrix, standardize


SOURCE_DIR = ROOT / "experiments" / "e0_4g_baseline"
D28_DIR = ROOT / "experiments" / "d28_al_engineering"
DEFAULT_OUTPUT = ROOT / "experiments" / "e2_random_smoke"
MEMBER_SEEDS = (42, 525)


def true_rows(engine: QGeoGNNActiveLearningEngine, ids: list[str]) -> pd.DataFrame:
    indices = [engine._sample_to_index[value] for value in ids]
    return engine.data.iloc[indices].reset_index(drop=True)


def ensemble_table(
    engine: QGeoGNNActiveLearningEngine,
    ids: list[str],
    checkpoints: dict[int, Path],
    return_primary_embedding: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    member_tables, primary_embedding = {}, None
    for member_seed, checkpoint in checkpoints.items():
        result = engine.predict(
            ids,
            checkpoint,
            return_quantiles=True,
            return_embedding=return_primary_embedding and member_seed == MEMBER_SEEDS[0],
            batch_size=256,
            chunk_size=1024,
        )
        member_tables[member_seed] = result.table
        if member_seed == MEMBER_SEEDS[0]:
            primary_embedding = result.embeddings
    table = member_tables[MEMBER_SEEDS[0]][
        ["sample_id", "canonical_index", "source_canonical_index"]
    ].copy()
    v1 = np.column_stack(
        [member_tables[seed]["V1_q50"].to_numpy(dtype=float) for seed in MEMBER_SEEDS]
    )
    v2 = np.column_stack(
        [member_tables[seed]["V2_q50"].to_numpy(dtype=float) for seed in MEMBER_SEEDS]
    )
    table["pred_V1"] = v1.mean(axis=1)
    table["pred_V2"] = v2.mean(axis=1)
    for member_index, seed in enumerate(MEMBER_SEEDS):
        table[f"member_{member_index}_seed"] = seed
        table[f"member_{member_index}_V1_q50"] = v1[:, member_index]
        table[f"member_{member_index}_V2_q50"] = v2[:, member_index]
    table["ensemble_variance_V1"] = np.var(v1, axis=1, ddof=1)
    table["ensemble_variance_V2"] = np.var(v2, axis=1, ddof=1)
    return table, primary_embedding


def evaluate_test(
    engine: QGeoGNNActiveLearningEngine,
    test_ids: list[str],
    checkpoints: dict[int, Path],
    scales: dict[str, float],
    round_index: int,
) -> tuple[dict, pd.DataFrame]:
    prediction, _ = ensemble_table(engine, test_ids, checkpoints, False)
    truth = true_rows(engine, test_ids)
    prediction["true_V1"] = truth["V1_ml"].to_numpy(dtype=float)
    prediction["true_V2"] = truth["V2_ml"].to_numpy(dtype=float)
    error_v1 = prediction["true_V1"] - prediction["pred_V1"]
    error_v2 = prediction["true_V2"] - prediction["pred_V2"]
    rmse_v1 = float(np.sqrt(np.mean(np.square(error_v1))))
    rmse_v2 = float(np.sqrt(np.mean(np.square(error_v2))))
    prediction["round"] = round_index
    return {
        "round": round_index,
        "labeled_total": None,
        "test_rows": len(prediction),
        "V1_rmse": rmse_v1,
        "V2_rmse": rmse_v2,
        "normalized_rmse_score": float(
            math.sqrt(0.5 * ((rmse_v1 / scales["V1"]) ** 2 + (rmse_v2 / scales["V2"]) ** 2))
        ),
    }, prediction


def selected_diagnostics(
    engine: QGeoGNNActiveLearningEngine,
    state_before,
    state_after,
    pool_prediction: pd.DataFrame,
    pool_embedding: np.ndarray,
    labeled_embedding: np.ndarray,
    scales: dict[str, float],
) -> tuple[dict, pd.DataFrame]:
    selected_ids = state_after.selected_ids
    selected_positions = {value: index for index, value in enumerate(state_before.pool_ids)}
    positions = np.asarray([selected_positions[value] for value in selected_ids], dtype=int)
    selected = pool_prediction.iloc[positions].copy().reset_index(drop=True)
    truth = true_rows(engine, selected_ids)
    selected["true_V1"] = truth["V1_ml"].to_numpy(dtype=float)
    selected["true_V2"] = truth["V2_ml"].to_numpy(dtype=float)
    selected["tail_flag"] = (selected["true_V1"] > 60.0) | (selected["true_V2"] > 120.0)
    selected["compound_id"] = truth["canonical_smiles"].astype(str).to_numpy()
    selected["standardized_abs_error_before_reveal"] = 0.5 * (
        np.abs(selected["true_V1"] - selected["pred_V1"]) / scales["V1"]
        + np.abs(selected["true_V2"] - selected["pred_V2"]) / scales["V2"]
    )
    selected["ensemble_score"] = (
        selected["ensemble_variance_V1"] / scales["V1"] ** 2
        + selected["ensemble_variance_V2"] / scales["V2"] ** 2
    )

    labeled_indices = np.asarray(
        [engine._sample_to_index[value] for value in state_before.labeled_ids], dtype=int
    )
    pool_indices = np.asarray(
        [engine._sample_to_index[value] for value in state_before.pool_ids], dtype=int
    )
    labeled_conditions = condition_matrix(engine.data, labeled_indices)
    pool_conditions = condition_matrix(engine.data, pool_indices)
    _, selected_h = standardize(labeled_embedding, pool_embedding[positions])
    _, selected_c = standardize(labeled_conditions, pool_conditions[positions])
    selected_representation = np.column_stack([selected_h, selected_c])
    mean_pairwise = float(pdist(selected_representation).mean()) if len(selected_ids) > 1 else 0.0
    diagnostics = {
        "round": state_after.round,
        "strategy": "random",
        "selected_total": len(selected),
        "selected_tail_count": int(selected["tail_flag"].sum()),
        "selected_tail_fraction": float(selected["tail_flag"].mean()),
        "selected_compounds": int(selected["compound_id"].nunique()),
        "mean_pairwise_latent_distance": mean_pairwise,
        "mean_uncertainty": float(selected["ensemble_score"].mean()),
        "mean_true_error_after_reveal": float(
            selected["standardized_abs_error_before_reveal"].mean()
        ),
    }
    return diagnostics, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--query-size", type=int, default=25)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)
    partition_path = D28_DIR / "partitions" / "e2_4g_row_seed_42.csv"
    partition = pd.read_csv(partition_path)
    data = pd.read_csv(SOURCE_DIR / "canonical_4g.csv")
    graph_cache = torch.load(SOURCE_DIR / "graph_cache_4g.pt", weights_only=False)
    l0_train_positions = partition.loc[
        partition["role"].eq("l0_train"), "canonical_index"
    ].to_numpy(dtype=int)
    descriptors = np.vstack(
        [graph_cache[smiles]["descriptor"] for smiles in data["canonical_smiles"]]
    ).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in data["PE/EA"]]).astype(
        np.float32
    )
    scaler = {
        "fit_split": "e2_fixed_l0_train",
        "descriptor": minmax_fit(descriptors[l0_train_positions]),
        "eluent": minmax_fit(eluents[l0_train_positions]),
    }
    (output_dir / "scaler.json").write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    engine = QGeoGNNActiveLearningEngine(
        data,
        graph_cache,
        scaler,
        SOURCE_DIR / "checkpoints" / "best.pt",
    )
    validation_ids = partition.loc[
        partition["role"].eq("l0_validation"), "sample_id"
    ].astype(str).tolist()
    labeled_ids = partition.loc[
        partition["role"].isin(["l0_train", "l0_validation"]), "sample_id"
    ].astype(str).tolist()
    pool_ids = partition.loc[partition["role"].eq("u0"), "sample_id"].astype(str).tolist()
    test_ids = partition.loc[partition["role"].eq("test"), "sample_id"].astype(str).tolist()
    l0_train = true_rows(
        engine,
        partition.loc[partition["role"].eq("l0_train"), "sample_id"].astype(str).tolist(),
    )
    scales = {
        target: max(float(l0_train[f"{target}_ml"].std(ddof=0)), 1e-8)
        for target in ("V1", "V2")
    }
    config = SourceFreeTrainConfig(epochs=args.epochs, patience=args.patience)
    config.validate_frozen_predictor()
    split_hash = sha256_file(partition_path)
    protocol_config_hash = canonical_json_hash(
        {"train_config": asdict(config), "scaler": scaler}
    )
    state = initialize_round_state(
        labeled_ids,
        pool_ids,
        "source_free_untrained",
        42,
        split_hash,
        protocol_config_hash,
    )

    metrics, diagnostics, all_predictions, checkpoint_records = [], [], [], []
    previous_prediction, previous_hashes = None, None
    for round_index in range(args.rounds + 1):
        round_dir = output_dir / f"round_{round_index}"
        checkpoints = {}
        fit_results = {}
        for member_seed in MEMBER_SEEDS:
            fit = engine.fit(
                state.labeled_ids,
                validation_ids,
                config,
                init_checkpoint=None,
                seed=member_seed,
                output_dir=round_dir / f"member_seed_{member_seed}",
            )
            checkpoints[member_seed] = Path(fit.checkpoint)
            fit_results[member_seed] = fit
            checkpoint_records.append(
                {"round": round_index, "member_seed": member_seed, **asdict(fit)}
            )
        state = replace(state, checkpoint=str(checkpoints[MEMBER_SEEDS[0]]))
        save_round_state(round_dir / "state_after_fit.json", state)
        reloaded = load_round_state(
            round_dir / "state_after_fit.json", split_hash, protocol_config_hash
        )
        if reloaded != state:
            raise AssertionError("Round state changed after persistence")

        row_metrics, test_prediction = evaluate_test(
            engine, test_ids, checkpoints, scales, round_index
        )
        row_metrics["labeled_total"] = len(state.labeled_ids)
        hashes = [fit_results[seed].checkpoint_sha256 for seed in MEMBER_SEEDS]
        row_metrics["checkpoint_changed_from_previous"] = (
            None if previous_hashes is None else hashes != previous_hashes
        )
        current_prediction = test_prediction[["pred_V1", "pred_V2"]].to_numpy(dtype=float)
        row_metrics["prediction_max_abs_change_from_previous"] = (
            None
            if previous_prediction is None
            else float(np.max(np.abs(current_prediction - previous_prediction)))
        )
        metrics.append(row_metrics)
        all_predictions.append(test_prediction)
        previous_prediction, previous_hashes = current_prediction, hashes

        if round_index == args.rounds:
            continue
        pool_prediction, pool_embedding = ensemble_table(
            engine, state.pool_ids, checkpoints, True
        )
        labeled_prediction = engine.predict(
            state.labeled_ids,
            checkpoints[MEMBER_SEEDS[0]],
            return_quantiles=False,
            return_embedding=True,
            batch_size=256,
            chunk_size=1024,
        )
        next_state = random_query(
            state, args.query_size, checkpoint=str(checkpoints[MEMBER_SEEDS[0]])
        )
        query_diagnostics, selected = selected_diagnostics(
            engine,
            state,
            next_state,
            pool_prediction,
            pool_embedding,
            labeled_prediction.embeddings,
            scales,
        )
        diagnostics.append(query_diagnostics)
        selected["round"] = next_state.round
        selected["strategy"] = "random"
        selected.to_csv(round_dir / "selected_after_reveal.csv", index=False)
        save_round_state(round_dir / "state_after_query.json", next_state)
        state = next_state

    metrics_df = pd.DataFrame(metrics)
    diagnostics_df = pd.DataFrame(diagnostics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(output_dir / "learning_curve_smoke.csv", index=False)
    diagnostics_df.to_csv(output_dir / "queried_slice_diagnostics.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions_by_round.csv.gz", index=False)
    pd.DataFrame(checkpoint_records).to_csv(output_dir / "checkpoint_manifest.csv", index=False)

    selected_across_rounds = [
        value
        for round_index in range(args.rounds)
        for value in load_round_state(
            output_dir / f"round_{round_index}" / "state_after_query.json",
            split_hash,
            protocol_config_hash,
        ).selected_ids
    ]
    passed = bool(
        len(selected_across_rounds) == args.rounds * args.query_size
        and len(selected_across_rounds) == len(set(selected_across_rounds))
        and all(metrics_df.loc[1:, "checkpoint_changed_from_previous"].astype(bool))
        and (metrics_df.loc[1:, "prediction_max_abs_change_from_previous"] > 0).all()
        and diagnostics_df["selected_total"].eq(args.query_size).all()
        and predictions_df.groupby("round")["sample_id"].nunique().eq(len(test_ids)).all()
    )
    decision = {
        "passed": passed,
        "stage": "E2_random_chain_smoke",
        "scientific_result_interpretable": False,
        "strategy": "random_only",
        "outer_protocol": "row_seed_42",
        "rounds": args.rounds,
        "query_size": args.query_size,
        "members": list(MEMBER_SEEDS),
        "source_free_no_full_4g_checkpoint_labels": True,
        "predictor_gate0_8g_transfer_changed": False,
        "unique_queries": len(set(selected_across_rounds)),
        "expected_queries": args.rounds * args.query_size,
        "fixed_test_ids": True,
        "fixed_l0_validation_ids": True,
        "next_stage": "E2 four-strategy three-seed pilot" if passed else "remain at E2 smoke",
        "tail_diagnostic_limitation": "4g canonical data already applies legacy 60/120 thresholds, so selected_tail_count is structurally zero",
    }
    (output_dir / "e2_smoke_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    experiment_config = {
        "stage": decision["stage"],
        "partition": str(partition_path.relative_to(ROOT)),
        "partition_sha256": split_hash,
        "train_config": asdict(config),
        "train_config_hash": config.config_hash,
        "protocol_config_hash": protocol_config_hash,
        "scaler_sha256": sha256_file(output_dir / "scaler.json"),
        "evaluation_scales_from": "fixed L0_train only",
        "evaluation_scales": scales,
        "selection": "Random with persisted numpy Generator state",
        "test_role": "reporting only; smoke result cannot select scientific methods",
        "source_free_protocol": "seeded random QGeoGNN + monotonic head; full model trainable; no full-data 4g checkpoint weights loaded",
    }
    (output_dir / "config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# E2 Random active-learning chain smoke

状态：**{'通过' if passed else '失败'}**。本目录只验证E2工程闭环，不用于判断Random或主动策略的科学优劣。

- 固定D28 `e2_4g_row_seed_42` partition；L0={len(labeled_ids)}，其中validation={len(validation_ids)}，test={len(test_ids)}。
- Random查询{args.rounds}轮，每轮{args.query_size}条；K=2成员每轮从seeded random source-free初始化重新训练。
- 不加载训练过完整4g标签的checkpoint，避免L0/U0/test标签泄漏；这是一条E2专用source-free协议，不修改Gate 0冻结的4g→8g `last2+head`迁移合同。
- 输入scaler只用固定L0-train拟合一次，随后跨轮次与策略冻结；validation、U0和test不参与拟合。
- test身份每轮固定，checkpoint与test prediction均发生变化，round state可按split/config hash恢复。
- `queried_slice_diagnostics.csv`含预注册的selected总数、tail、compound、批内表示距离、uncertainty与reveal后真实误差。

限制：当前4g canonical数据在历史reader阶段已经应用60/120 mL阈值，因此本smoke的tail计数结构性为0；tail acquisition机制只能在后续保留完整574行的8g E4中正式解释。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

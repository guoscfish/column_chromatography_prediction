#!/usr/bin/env python3
"""Run D28 AL-engineering qualification and freeze paired L0/U0/test partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_engine import (
    QGeoGNNActiveLearningEngine,
    TrainConfig,
    canonical_json_hash,
    initialize_round_state,
    load_round_state,
    random_query,
    save_round_state,
)
from scripts.run_e0_4g_baseline import sha256_file, write_artifact_manifest, write_environment
from scripts.run_e0_8g_controls import load_graph_cache


SEEDS = (42, 525, 1101)
SOURCE_DIR = ROOT / "experiments" / "e0_4g_baseline"
G03_DIR = ROOT / "experiments" / "g0_3_threshold_sensitivity"
G04_DIR = ROOT / "experiments" / "g0_4_paper_style_transfer"
FROZEN_8G_CHECKPOINT = (
    G04_DIR / "runs" / "row_seed_42" / "last2_head" / "checkpoints" / "last2_head.pt"
)


def row_test_mask(rows: int, seed: int, fraction: float) -> np.ndarray:
    rng = np.random.RandomState(seed)
    order = rng.permutation(rows)
    test_count = max(1, int(math.ceil(fraction * rows)))
    mask = np.zeros(rows, dtype=bool)
    mask[order[-test_count:]] = True
    return mask


def compound_test_mask(data: pd.DataFrame, seed: int, fraction: float) -> np.ndarray:
    sizes = data.groupby("canonical_smiles", sort=True).size()
    groups = sizes.index.to_numpy(copy=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    cumulative = np.cumsum([int(sizes[group]) for group in groups])
    keep_target = (1.0 - fraction) * len(data)
    cut = int(np.argmin(np.abs(cumulative - keep_target))) + 1
    test_groups = set(groups[cut:])
    if not test_groups:
        test_groups = {groups[-1]}
    return data["canonical_smiles"].isin(test_groups).to_numpy()


def make_partition(
    data: pd.DataFrame,
    stage: str,
    protocol: str,
    seed: int,
    test_fraction: float,
    l0_size: int | None,
    l0_fraction: float | None,
    validation_size: int | None,
    validation_fraction: float | None,
) -> pd.DataFrame:
    if protocol.endswith("compound") or protocol == "compound":
        test_mask = compound_test_mask(data, seed, test_fraction)
    else:
        test_mask = row_test_mask(len(data), seed, test_fraction)
    available = np.flatnonzero(~test_mask)
    if l0_size is None:
        if l0_fraction is None:
            raise ValueError("Either l0_size or l0_fraction is required")
        resolved_l0 = int(math.ceil(l0_fraction * len(available)))
    else:
        resolved_l0 = int(l0_size)
    if resolved_l0 >= len(available):
        raise ValueError("L0 must leave a non-empty U0")
    rng = np.random.RandomState(seed + 100_003)
    l0 = rng.choice(available, size=resolved_l0, replace=False)
    if validation_size is None:
        if validation_fraction is None:
            raise ValueError("Either validation_size or validation_fraction is required")
        resolved_validation = int(math.ceil(validation_fraction * len(l0)))
    else:
        resolved_validation = int(validation_size)
    if resolved_validation < 2 or resolved_validation >= len(l0):
        raise ValueError("Initial validation must have >=2 rows and leave L0 training rows")
    validation = set(rng.choice(l0, size=resolved_validation, replace=False).tolist())
    l0_set = set(int(index) for index in l0)
    roles = []
    for index in range(len(data)):
        if test_mask[index]:
            roles.append("test")
        elif index in validation:
            roles.append("l0_validation")
        elif index in l0_set:
            roles.append("l0_train")
        else:
            roles.append("u0")
    partition = data[["sample_id", "canonical_index", "canonical_smiles"]].copy()
    partition["role"] = roles
    partition["stage"] = stage
    partition["protocol"] = protocol
    partition["seed"] = seed
    if set(partition["role"]) != {"test", "l0_validation", "l0_train", "u0"}:
        raise AssertionError("Partition is missing a required role")
    if partition["sample_id"].duplicated().any() or partition["canonical_index"].duplicated().any():
        raise AssertionError("Partition identity is not one-to-one")
    if "compound" in protocol:
        test_compounds = set(partition.loc[partition["role"].eq("test"), "canonical_smiles"])
        development_compounds = set(partition.loc[~partition["role"].eq("test"), "canonical_smiles"])
        if test_compounds & development_compounds:
            raise AssertionError("Compound test leakage detected")
    return partition


def freeze_partitions(output_dir: Path) -> tuple[dict, dict[tuple[str, str, int], Path]]:
    partition_dir = output_dir / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)
    data_4g = pd.read_csv(SOURCE_DIR / "canonical_4g.csv").reset_index(drop=True)
    data_4g["canonical_index"] = np.arange(len(data_4g), dtype=int)
    data_8g = pd.read_csv(G03_DIR / "canonical_8g_no_threshold.csv").reset_index(drop=True)
    data_8g["canonical_index"] = np.arange(len(data_8g), dtype=int)
    specifications = [
        ("e2_4g", "row", data_4g, 0.10, None, 0.10, None, 0.15),
        ("e2_4g", "compound", data_4g, 0.10, None, 0.10, None, 0.15),
        ("e4_8g", "protocol_a_row", data_8g, 0.10, 50, None, 8, None),
        ("e4_8g", "protocol_b_compound", data_8g, 0.15, 50, None, 8, None),
    ]
    manifest, paths = {}, {}
    for stage, protocol, data, test_fraction, l0_size, l0_fraction, val_size, val_fraction in specifications:
        for seed in SEEDS:
            partition = make_partition(
                data,
                stage,
                protocol,
                seed,
                test_fraction,
                l0_size,
                l0_fraction,
                val_size,
                val_fraction,
            )
            path = partition_dir / f"{stage}_{protocol}_seed_{seed}.csv"
            partition.to_csv(path, index=False)
            key = f"{stage}/{protocol}/seed={seed}"
            counts = {name: int(value) for name, value in partition["role"].value_counts().items()}
            manifest[key] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "counts": counts,
                "l0_total": counts["l0_train"] + counts["l0_validation"],
                "test_compound_leakage": bool(
                    set(partition.loc[partition["role"].eq("test"), "canonical_smiles"])
                    & set(partition.loc[~partition["role"].eq("test"), "canonical_smiles"])
                )
                if "compound" in protocol
                else None,
            }
            paths[(stage, protocol, seed)] = path
    return manifest, paths


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run_stress_inference(
    engine: QGeoGNNActiveLearningEngine,
    checkpoint: Path,
    output_dir: Path,
    candidate_count: int,
) -> dict:
    source = np.arange(candidate_count, dtype=int) % len(engine.data)
    candidates = pd.DataFrame(
        {
            "sample_id": [
                f"d28-stress-{index:05d}-{engine.data.iloc[source_index]['sample_id'][:8]}"
                for index, source_index in enumerate(source)
            ],
            "canonical_index": np.arange(candidate_count, dtype=int),
            "source_canonical_index": source,
        }
    )
    maximum_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    first = engine.predict(
        candidates,
        checkpoint,
        return_quantiles=True,
        return_embedding=True,
        batch_size=64,
        chunk_size=512,
    )
    first_seconds = time.perf_counter() - start
    maximum_rss_mid = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    second = engine.predict(
        candidates,
        checkpoint,
        return_quantiles=True,
        return_embedding=True,
        batch_size=257,
        chunk_size=1024,
    )
    second_seconds = time.perf_counter() - start
    maximum_rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    identity_equal = first.table[IDENTITY_COLUMNS].equals(second.table[IDENTITY_COLUMNS])
    request_order_equal = first.table["sample_id"].tolist() == candidates["sample_id"].tolist()
    prediction_columns = ["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"]
    first_values = first.table[prediction_columns].to_numpy(dtype=np.float32)
    second_values = second.table[prediction_columns].to_numpy(dtype=np.float32)
    prediction_max_abs_diff = float(np.max(np.abs(first_values - second_values)))
    embedding_max_abs_diff = float(np.max(np.abs(first.embeddings - second.embeddings)))
    crossing_rate = float(
        np.mean(
            (first.table["V1_q10"] > first.table["V1_q50"])
            | (first.table["V1_q50"] > first.table["V1_q90"])
            | (first.table["V2_q10"] > first.table["V2_q50"])
            | (first.table["V2_q50"] > first.table["V2_q90"])
        )
    )
    tolerance = 1e-5
    passed = bool(
        len(first.table) == candidate_count
        and first.table["sample_id"].is_unique
        and first.table["canonical_index"].is_unique
        and identity_equal
        and request_order_equal
        and prediction_max_abs_diff <= tolerance
        and embedding_max_abs_diff <= tolerance
        and crossing_rate == 0.0
    )
    first.table.to_csv(output_dir / "stress_predictions_10240.csv.gz", index=False, compression="gzip")
    return {
        "passed": passed,
        "candidate_count": candidate_count,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "run_a": {"batch_size": 64, "chunk_size": 512, "seconds": first_seconds},
        "run_b": {"batch_size": 257, "chunk_size": 1024, "seconds": second_seconds},
        "identity_equal": identity_equal,
        "request_order_equal": request_order_equal,
        "unique_sample_ids": bool(first.table["sample_id"].is_unique),
        "unique_canonical_indices": bool(first.table["canonical_index"].is_unique),
        "prediction_max_abs_diff": prediction_max_abs_diff,
        "embedding_max_abs_diff": embedding_max_abs_diff,
        "numeric_tolerance": tolerance,
        "prediction_sha256_a": array_sha256(first_values),
        "prediction_sha256_b": array_sha256(second_values),
        "embedding_sha256_a": array_sha256(first.embeddings),
        "embedding_sha256_b": array_sha256(second.embeddings),
        "quantile_crossing_rate": crossing_rate,
        "max_rss_before": maximum_rss_before,
        "max_rss_after_run_a": maximum_rss_mid,
        "max_rss_after_run_b": maximum_rss_after,
    }


# Kept local to make identity columns explicit in the stress audit.
IDENTITY_COLUMNS = ["sample_id", "canonical_index", "source_canonical_index"]


def run_resume_check(
    partition_path: Path, checkpoint: Path, output_dir: Path, config: TrainConfig
) -> dict:
    partition = pd.read_csv(partition_path)
    labeled = partition.loc[
        partition["role"].isin(["l0_train", "l0_validation"]), "sample_id"
    ].astype(str).tolist()
    pool = partition.loc[partition["role"].eq("u0"), "sample_id"].astype(str).tolist()
    split_hash = sha256_file(partition_path)
    initial = initialize_round_state(
        labeled,
        pool,
        str(checkpoint),
        42,
        split_hash,
        config.config_hash,
    )
    continuous_round_1 = random_query(initial, 10, checkpoint="round_1.pt")
    continuous_round_2 = random_query(continuous_round_1, 10, checkpoint="round_2.pt")
    state_dir = output_dir / "state"
    save_round_state(state_dir / "round_0.json", initial)
    save_round_state(state_dir / "round_1.json", continuous_round_1)
    resumed_round_1 = load_round_state(
        state_dir / "round_1.json", split_hash, config.config_hash
    )
    resumed_round_2 = random_query(resumed_round_1, 10, checkpoint="round_2.pt")
    save_round_state(state_dir / "round_2.json", resumed_round_2)
    hash_guard_passed = False
    try:
        load_round_state(state_dir / "round_1.json", "changed-split", config.config_hash)
    except ValueError:
        hash_guard_passed = True
    continuous_payload = asdict(continuous_round_2)
    resumed_payload = asdict(resumed_round_2)
    passed = bool(
        continuous_payload == resumed_payload
        and not (set(resumed_round_2.labeled_ids) & set(resumed_round_2.pool_ids))
        and len(resumed_round_2.labeled_ids) == len(set(resumed_round_2.labeled_ids))
        and hash_guard_passed
    )
    return {
        "passed": passed,
        "partition": str(partition_path.relative_to(ROOT)),
        "split_hash": split_hash,
        "config_hash": config.config_hash,
        "rounds_compared": 2,
        "continuous_equals_resumed": continuous_payload == resumed_payload,
        "hash_guard_passed": hash_guard_passed,
        "initial_labeled": len(initial.labeled_ids),
        "initial_pool": len(initial.pool_ids),
        "final_labeled": len(resumed_round_2.labeled_ids),
        "final_pool": len(resumed_round_2.pool_ids),
        "round_1_selected_ids": continuous_round_1.selected_ids,
        "round_2_selected_ids": continuous_round_2.selected_ids,
        "duplicate_queries": len(resumed_round_2.labeled_ids) - len(set(resumed_round_2.labeled_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "d28_al_engineering"
    )
    parser.add_argument("--stress-candidates", type=int, default=10240)
    parser.add_argument("--fit-smoke-epochs", type=int, default=2)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if (output_dir / "d28_decision.json").exists():
        raise FileExistsError(f"Refusing to overwrite finalized D28 experiment: {output_dir}")
    if args.stress_candidates < 10000:
        raise ValueError("D28 requires at least 10,000 stress candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)
    partition_manifest, partition_paths = freeze_partitions(output_dir)
    (output_dir / "partition_manifest.json").write_text(
        json.dumps(partition_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data = pd.read_csv(G03_DIR / "canonical_8g_no_threshold.csv")
    scaler = json.loads((SOURCE_DIR / "scaler.json").read_text(encoding="utf-8"))
    engine = QGeoGNNActiveLearningEngine(
        data,
        load_graph_cache(),
        scaler,
        SOURCE_DIR / "checkpoints" / "best.pt",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    smoke_partition_path = partition_paths[("e4_8g", "protocol_a_row", 42)]
    smoke_partition = pd.read_csv(smoke_partition_path)
    labeled_ids = smoke_partition.loc[
        smoke_partition["role"].isin(["l0_train", "l0_validation"]), "sample_id"
    ].astype(str).tolist()
    validation_ids = smoke_partition.loc[
        smoke_partition["role"].eq("l0_validation"), "sample_id"
    ].astype(str).tolist()
    smoke_config = TrainConfig(
        epochs=args.fit_smoke_epochs,
        patience=args.fit_smoke_epochs,
        batch_size=2048,
    )
    fit_result = engine.fit(
        labeled_ids,
        validation_ids,
        smoke_config,
        SOURCE_DIR / "checkpoints" / "best.pt",
        42,
        output_dir / "fit_smoke",
    )
    smoke_pool_ids = smoke_partition.loc[smoke_partition["role"].eq("u0"), "sample_id"].head(20)
    smoke_prediction = engine.predict(
        smoke_pool_ids.astype(str).tolist(),
        Path(fit_result.checkpoint),
        return_quantiles=True,
        return_embedding=True,
        batch_size=7,
        chunk_size=11,
    )
    smoke_prediction.table.to_csv(output_dir / "fit_smoke" / "pool_predictions.csv", index=False)
    np.save(output_dir / "fit_smoke" / "pool_embeddings.npy", smoke_prediction.embeddings)
    fit_audit = {
        "passed": bool(
            fit_result.train_rows == 42
            and fit_result.validation_rows == 8
            and len(smoke_prediction.table) == 20
            and smoke_prediction.embeddings.shape == (20, 128)
            and smoke_prediction.table["sample_id"].tolist()
            == smoke_pool_ids.astype(str).tolist()
        ),
        **asdict(fit_result),
        "prediction_rows": len(smoke_prediction.table),
        "embedding_shape": list(smoke_prediction.embeddings.shape),
        "scientific_role": "engineering smoke only; two epochs do not create a scientific baseline",
    }
    (output_dir / "fit_predict_audit.json").write_text(
        json.dumps(fit_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stress_audit = run_stress_inference(
        engine, FROZEN_8G_CHECKPOINT, output_dir, args.stress_candidates
    )
    (output_dir / "batched_inference_audit.json").write_text(
        json.dumps(stress_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frozen_config = TrainConfig()
    resume_audit = run_resume_check(
        smoke_partition_path, FROZEN_8G_CHECKPOINT, output_dir, frozen_config
    )
    (output_dir / "resume_audit.json").write_text(
        json.dumps(resume_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    partition_passed = bool(
        len(partition_manifest) == 12
        and all(not entry["test_compound_leakage"] for entry in partition_manifest.values() if entry["test_compound_leakage"] is not None)
        and all(entry["l0_total"] > 0 for entry in partition_manifest.values())
    )
    decision = {
        "stage": "D28_active_learning_engineering",
        "passed": bool(fit_audit["passed"] and stress_audit["passed"] and resume_audit["passed"] and partition_passed),
        "fit_predict_interface_passed": fit_audit["passed"],
        "batched_inference_10k_passed": stress_audit["passed"],
        "resume_rng_identity_passed": resume_audit["passed"],
        "paired_partitions_passed": partition_passed,
        "gate_0_scientific_config_changed": False,
        "next_stage": "E1 acquisition-signal qualification" if fit_audit["passed"] and stress_audit["passed"] and resume_audit["passed"] and partition_passed else "remain at D28",
    }
    (output_dir / "d28_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "stage": "D28_active_learning_engineering",
        "stress_candidates": args.stress_candidates,
        "fit_smoke_epochs": args.fit_smoke_epochs,
        "canonical_8g": str((G03_DIR / "canonical_8g_no_threshold.csv").relative_to(ROOT)),
        "canonical_8g_sha256": sha256_file(G03_DIR / "canonical_8g_no_threshold.csv"),
        "source_anchor": str((SOURCE_DIR / "checkpoints" / "best.pt").relative_to(ROOT)),
        "source_anchor_sha256": sha256_file(SOURCE_DIR / "checkpoints" / "best.pt"),
        "frozen_inference_checkpoint": str(FROZEN_8G_CHECKPOINT.relative_to(ROOT)),
        "frozen_inference_checkpoint_sha256": sha256_file(FROZEN_8G_CHECKPOINT),
        "frozen_train_config": asdict(frozen_config),
        "frozen_train_config_hash": frozen_config.config_hash,
        "identity_policy": "sample_id primary; canonical_index frozen-table audit position; source_canonical_index explicit only for repeated stress fixtures",
        "test_selection_policy": "test never enters fit, validation, checkpoint, calibration or state selection",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# D28：主动学习工程底座检查

## 决定

**{'通过' if decision['passed'] else '未通过'}**。本阶段只验证接口与状态，不修改Gate 0冻结科学配置，也不把2-epoch fit smoke作为科学结果。

- 统一`fit/predict`：{'通过' if fit_audit['passed'] else '失败'}；L0=50，其中train=42、validation=8，validation计入标签预算。
- 10k+分块推理：{'通过' if stress_audit['passed'] else '失败'}；{stress_audit['candidate_count']}个candidate，两套batch/chunk配置的prediction最大绝对差为{stress_audit['prediction_max_abs_diff']:.3g}，embedding为{stress_audit['embedding_max_abs_diff']:.3g}，顺序与身份完全一致。
- Round resume：{'通过' if resume_audit['passed'] else '失败'}；连续2轮与round 1落盘后恢复的selected/labeled/pool/RNG状态完全一致，重复query={resume_audit['duplicate_queries']}。
- 固定partition：{'通过' if partition_passed else '失败'}；已冻结E2 4g row/compound与E4 8g Protocol A/B各3个outer seeds，共12份partition。

## 身份和泄漏约束

长期身份只使用`sample_id`。`canonical_index`是冻结canonical表中的审计位置；任何过滤后的临时DataFrame行号都不能写入AL状态。Stress fixture因复用574条真实图构造10240个虚拟candidate，额外显式保存`source_canonical_index`，不会冒充新的实验样本。

`fit`只接收当前labeled budget与其中固定validation subset；test不进入训练、早停、checkpoint或状态选择。`predict`按请求顺序返回quantiles和128维`h_graph`，并在每个chunk核对位置。

## 下一步

{'Gate 0工程检查完成，可以进入E1 signal qualification；E1不做主动学习重训。' if decision['passed'] else '保持在D28，修复失败项后才能进入E1。'}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps({"decision": decision, "stress": stress_audit, "resume": resume_audit}, ensure_ascii=False), flush=True)
    if not decision["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

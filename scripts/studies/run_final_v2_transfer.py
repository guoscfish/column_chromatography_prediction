#!/usr/bin/env python3
"""Ordinary 4g-to-8g transfer baseline using the final standalone predictor."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.evaluation.reporting import markdown_table
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data, load_combined_graph_cache
from src.qgeognn_al.transfer.baseline import configure_trainable, set_training_mode
from src.qgeognn_al.models import load_predictor_checkpoint, predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE, TARGET_DATA, TARGET_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json, loader_pair, point_metrics, predict, seed_everything, stable_hash, target_loss

STUDY = ROOT / "studies/transfer/4g_to_8g"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
QUALIFICATION = ROOT / "studies/predictor/final_4g_qualification/decision.json"
OLD_STUDY = ROOT / "studies/track_b_transfer/t1_low_label_adaptation"
METHODS = ("zero_shot", "affine", "target_head_only", "last2", "full_finetune")
NEURAL = {"target_head_only": "head_only", "last2": "last2", "full_finetune": "full"}
CONFIG = {"learning_rate": 1e-4, "weight_decay": 1e-5, "maximum_epochs": 500, "patience": 100,
          "batch_size": 2048, "checkpoint_selection": "target_validation_combined_source_normalized_rmse",
          "target_normalization": "source_4g_train_only", "active_acquisition": False,
          "cpu_threads": 1, "loss_weights": {"V1": 1., "V2": 1.},
          "head": "unchanged_current_Linear_ReLU", "shuffle": "deterministic_each_epoch"}


def prepare():
    decision = json.loads(QUALIFICATION.read_text())
    if decision["decision"] != "4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES":
        raise RuntimeError("4g qualification gate is not open")
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source_payload["model_variant"] != "qgeognn_v2":
        raise RuntimeError("transfer source is not standalone QGeoGNN-V2")
    STUDY.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((OLD_STUDY / "config.json").read_text())
    assert sha256_file(TARGET_DATA) == frozen["target_sha256"]
    for name in ("partition_manifest.csv", "schedule_manifest.csv"):
        hash_key = "partition_sha256" if name.startswith("partition") else "schedule_sha256"
        assert sha256_file(OLD_STUDY / name) == frozen[hash_key]
        destination = STUDY / "splits" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD_STUDY / name, destination)
    protocol = {
        **CONFIG, "methods": METHODS, "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(SOURCE), "source_selection": "row_seed_42 preregistered before qualification",
        "target_data": str(TARGET_DATA.relative_to(ROOT)), "target_data_sha256": sha256_file(TARGET_DATA),
        "partition_sha256": sha256_file(STUDY / "splits/partition_manifest.csv"),
        "schedule_sha256": sha256_file(STUDY / "splits/schedule_manifest.csv"),
        "partition_origin": "frozen historical T1 identity-only schedule", "test_during_fit": False,
        "target_threshold": None, "adapter_sweep": False, "uq_acquisition": False,
        "outer_seeds": frozen["outer_seeds"], "budgets": frozen["target_label_budgets"],
        "fixed_validation": 8, "target_rows": 574,
        "primary_aulc_metric": "T1 arithmetic mean of V1/source_std and V2/source_std RMSE; trapezoidal average over budgets 30..100",
        "checkpoint_metric": "root mean square of the two source-normalized RMSEs (same ordering as T1 validation normalized MSE)",
        "stability_rule": "negative mean and median paired AULC delta; wins in at least 4/5 outer seeds",
        "comparators": ["zero_shot", "affine"], "preprocessing_target_rows_used": 0,
    }
    if (STUDY / "protocol.json").exists() and json.loads((STUDY / "protocol.json").read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("frozen transfer protocol changed")
    atomic_json(STUDY / "protocol.json", protocol)
    (STUDY / "PREREGISTRATION.md").write_text(
        "# Final QGeoGNN-V2 4g-to-8g transfer baseline\n\n"
        "Use the preregistered row-seed-42 standalone 4g checkpoint and the frozen T1 target partitions, budgets, and source-only preprocessing. Compare only zero-shot, affine, target-head-only, last-two effective message layers, and full fine-tuning. Validation selects neural checkpoints; target test labels are evaluated only after fits freeze. No target threshold, adapter sweep, uncertainty acquisition, or post-test tuning is allowed.\n"
    )
    return protocol


def train_adaptation(mode, seed, atom, angle, train_idx, valid_idx, preprocessing, output, contract=None):
    if set(train_idx) & set(valid_idx) or not len(train_idx) or not len(valid_idx):
        raise ValueError("invalid adaptation train/validation split")
    contract = {} if contract is None else contract
    finished_path = output.parent / "fit_summary.json"
    if finished_path.exists():
        finished = json.loads(finished_path.read_text())
        if finished["contract"] != contract or sha256_file(output) != finished["checkpoint_sha256"]:
            raise RuntimeError("completed transfer fit contract/hash mismatch")
        return {**finished, "history": pd.read_csv(output.parent / "history.csv").to_dict("records")}
    seed_everything(seed)
    model = load_predictor_checkpoint(SOURCE)
    trainable, total = configure_trainable(model, mode)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    best, best_epoch, stale, history = float("inf"), None, 0, []
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_path = output.parent / "last.pt"
    start_epoch = 1
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume["contract"] != contract:
            raise RuntimeError("interrupted transfer fit contract mismatch")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        torch.set_rng_state(resume["rng_state"])
        best, best_epoch, stale, history = resume["best"], resume["best_epoch"], resume["stale"], resume["history"]
        start_epoch = resume["epoch"]+1
    for epoch in range(start_epoch, CONFIG["maximum_epochs"] + 1):
        set_training_mode(model)
        order = np.random.default_rng(seed * 10000 + epoch).permutation(train_idx)
        losses = []
        for a, b in zip(*loader_pair(atom, angle, order, CONFIG["batch_size"])):
            pred = model(a, b)
            loss = target_loss(a.y[:, 0], pred[:, :3]) + target_loss(a.y[:, 1], pred[:, 3:])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite target training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        truth, pred, _ = predict(model, atom, angle, valid_idx)
        metrics = point_metrics(truth, pred, preprocessing["target_scales"])
        score = metrics["combined_normalized_rmse"]
        if not math.isfinite(score):
            raise RuntimeError("non-finite target validation score")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_score": score})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(predictor_checkpoint(model, preprocessing=preprocessing,
                       training_config={**CONFIG, "transfer_mode": mode, "seed": seed},
                       provenance={"source_checkpoint_sha256": sha256_file(SOURCE), "best_epoch": epoch,
                                   "target_validation_score": score, "fit_contract": contract}), output.with_suffix(".tmp"))
            output.with_suffix(".tmp").replace(output)
        else:
            stale += 1
        if epoch % 25 == 0:
            torch.save({"contract": contract, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(), "best": best, "best_epoch": best_epoch, "stale": stale,
                        "history": history, "epoch": epoch}, resume_path.with_suffix(".tmp"))
            resume_path.with_suffix(".tmp").replace(resume_path)
            atomic_json(output.parent / "progress.json", {"epoch": epoch, "best_epoch": best_epoch, "mode": mode})
        if stale >= CONFIG["patience"]:
            break
    result = {"best_epoch": best_epoch, "epochs_run": len(history), "validation_score": best,
            "trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / total, "contract": contract, "checkpoint_sha256": sha256_file(output)}
    pd.DataFrame(history).to_csv(output.parent / "history.csv", index=False)
    atomic_json(finished_path, result)
    return {**result, "history": history}


def affine_fit(train_truth, train_prediction, values):
    output = np.empty_like(values)
    for i in range(2):
        design = np.column_stack([train_prediction[:, i], np.ones(len(train_prediction))])
        coefficient = np.linalg.lstsq(design, train_truth[:, i], rcond=None)[0]
        output[:, i] = values[:, i] * coefficient[0] + coefficient[1]
    return output


def run_context(seed, budget):
    torch.set_num_threads(CONFIG["cpu_threads"])
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if sha256_file(SOURCE) != protocol["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed")
    result = STUDY / f"results/seed_{seed}/budget_{budget}"
    context_contract = {"protocol_hash": stable_hash(protocol), "seed": seed, "budget": budget}
    done = result / "completion.json"
    if done.exists():
        completion = json.loads(done.read_text())
        assert completion["contract"] == context_contract
        for name, digest in completion["files"].items():
            assert sha256_file(result/name) == digest
        print(f"seed={seed} budget={budget}: reusing complete context", flush=True); return
    for path, key in ((TARGET_DATA, "target_data_sha256"), (STUDY / "splits/partition_manifest.csv", "partition_sha256"),
                      (STUDY / "splits/schedule_manifest.csv", "schedule_sha256")):
        assert sha256_file(path) == protocol[key]
    target = pd.read_csv(TARGET_DATA).reset_index(drop=True)
    schedule = pd.read_csv(STUDY / "splits/schedule_manifest.csv")
    context = schedule.loc[(schedule.outer_seed == seed) & (schedule.budget == budget)]
    roles = target[["sample_id"]].merge(context[["sample_id", "role"]], on="sample_id", validate="one_to_one", how="left")
    assert len(roles) == 574 and not roles.role.isna().any()
    indices = {role: np.flatnonzero(roles.role.eq(role)) for role in ("gradient_train", "validation", "test")}
    if any(set(indices[a]) & set(indices[b]) for a, b in (("gradient_train", "validation"), ("gradient_train", "test"), ("validation", "test"))):
        raise RuntimeError("target role overlap")
    assert len(indices["gradient_train"]) == budget-8 and len(indices["validation"]) == 8
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    preprocessing = source_payload["preprocessing"]
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    # Assemble model data with no test/unlabeled truth available to fitting.
    fitting_data = target.copy()
    fitting_data[["V1_ml", "V2_ml"]] = 0.
    revealed = np.concatenate([indices["gradient_train"], indices["validation"]])
    fitting_data.loc[revealed, ["V1_ml", "V2_ml"]] = target.loc[revealed, ["V1_ml", "V2_ml"]]
    atom, angle = build_model_data(fitting_data, cache, pd.DataFrame(), preprocessing["scaler"])
    source_model = load_predictor_checkpoint(SOURCE)
    train_truth, train_six, _ = predict(source_model, atom, angle, indices["gradient_train"])
    fit_audit = {}
    checkpoints = {}
    for method, mode in NEURAL.items():
        checkpoint = STUDY / f"runtime/seed_{seed}/budget_{budget}/{method}/best.pt"
        contract = {**context_contract, "method": method, "source_sha256": protocol["source_checkpoint_sha256"],
                    "train_ids_hash": stable_hash(sorted(target.iloc[indices["gradient_train"]].sample_id.tolist())),
                    "validation_ids_hash": stable_hash(sorted(target.iloc[indices["validation"]].sample_id.tolist()))}
        fit = train_adaptation(mode, seed, atom, angle, indices["gradient_train"], indices["validation"], preprocessing, checkpoint, contract)
        checkpoints[method] = checkpoint
        fit_audit[method] = {k: v for k, v in fit.items() if k != "history"}
        pd.DataFrame(fit["history"]).to_csv(checkpoint.parent / "history.csv", index=False)
    # Test outcomes become available only after all adaptation checkpoints freeze.
    test_truth = target.iloc[indices["test"]][["V1_ml", "V2_ml"]].to_numpy(dtype=np.float32)
    test_six = predict(source_model, atom, angle, indices["test"])[1]
    predictions = {"zero_shot": test_six[:, [1, 4]],
                   "affine": affine_fit(train_truth, train_six[:, [1, 4]], test_six[:, [1, 4]])}
    for method, checkpoint in checkpoints.items():
        predictions[method] = predict(load_predictor_checkpoint(checkpoint), atom, angle, indices["test"])[1][:, [1, 4]]
    metrics = {}
    for method, pred2 in predictions.items():
        six = np.column_stack([pred2[:, 0], pred2[:, 0], pred2[:, 0], pred2[:, 1], pred2[:, 1], pred2[:, 1]])
        metrics[method] = point_metrics(test_truth, six, preprocessing["target_scales"])
        metrics[method]["T1_combined_NRMSE"] = float(.5*sum(metrics[method][f"{t}_rmse"]/preprocessing["target_scales"][t] for t in ("V1", "V2")))
    result.mkdir(parents=True, exist_ok=True)
    atomic_json(result / "metrics.json", metrics)
    atomic_json(result / "fit_audit.json", fit_audit)
    pd.DataFrame({"sample_id": target.iloc[indices["test"]].sample_id,
                  "V1_true": test_truth[:, 0], "V2_true": test_truth[:, 1],
                  **{f"{method}_{target_name}": values[:, i] for method, values in predictions.items() for i, target_name in enumerate(("V1", "V2"))}}).to_csv(result / "predictions.csv.gz", index=False, compression="gzip")
    atomic_json(result / "label_usage.json", {"train_rows": budget-8, "validation_rows": 8, "test_rows": len(indices["test"]),
                "test_rows_used_for_fit": 0, "test_rows_used_for_checkpoint_selection": 0, "target_rows_used_for_preprocessing_fit": 0,
                "train_ids_hash": contract["train_ids_hash"], "validation_ids_hash": contract["validation_ids_hash"],
                "test_ids_hash": stable_hash(sorted(target.iloc[indices["test"]].sample_id.tolist()))})
    atomic_json(done, {"contract": context_contract, "files": {name: sha256_file(result/name) for name in
                ("metrics.json", "fit_audit.json", "predictions.csv.gz", "label_usage.json")}})
    print(json.dumps({"seed": seed, "budget": budget, "metrics": metrics}), flush=True)


def summarize():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    partition = pd.read_csv(STUDY / "splits/partition_manifest.csv")
    seeds = list(dict.fromkeys(partition.outer_seed.astype(int)))
    budgets = sorted(pd.read_csv(STUDY / "splits/schedule_manifest.csv").budget.astype(int).unique())
    rows, params = [], {
        "zero_shot": {"trainable_parameters": 0, "total_parameters": 458952, "trainable_fraction": 0.},
        "affine": {"trainable_parameters": 4, "total_parameters": 458956, "trainable_fraction": 4/458956,
                   "neural_parameters_trainable": 0, "external_affine_coefficients": 4},
    }
    for seed in seeds:
        for budget in budgets:
            base = STUDY / f"results/seed_{seed}/budget_{budget}"
            metrics = json.loads((base / "metrics.json").read_text())
            audit = json.loads((base / "fit_audit.json").read_text())
            for method in METHODS:
                rows.append({"seed": seed, "budget": budget, "method": method, **{k: v for k, v in metrics[method].items() if k != "all_outputs_finite"}})
            for method, values in audit.items(): params[method] = {k: values[k] for k in ("trainable_parameters", "total_parameters", "trainable_fraction")}
    table = pd.DataFrame(rows); table.to_csv(STUDY / "results/per_seed_budget_metrics.csv", index=False)
    summary = table.groupby(["budget", "method"]).agg({k: ["mean", "std", "min", "max"] for k in ("V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae", "combined_normalized_rmse", "T1_combined_NRMSE")})
    summary.to_csv(STUDY / "results/aggregate_metrics.csv")
    aulc_rows = []
    for (seed, method), group in table.groupby(["seed", "method"]):
        group = group.sort_values("budget")
        score = float(np.trapezoid(group.T1_combined_NRMSE, group.budget) / (max(budgets)-min(budgets)))
        aulc_rows.append({"seed": seed, "method": method, "normalized_aulc": score})
    aulc = pd.DataFrame(aulc_rows); aulc.to_csv(STUDY / "results/aulc_by_seed.csv", index=False)
    ranked = aulc.groupby("method").normalized_aulc.agg(["mean", "std", "min", "max"]).sort_values("mean")
    ranked.to_csv(STUDY / "results/aulc_ranking.csv")
    best = str(ranked.index[0])
    zero = aulc.loc[aulc.method.eq("zero_shot")].set_index("seed").normalized_aulc
    candidate = aulc.loc[aulc.method.eq(best)].set_index("seed").normalized_aulc
    wins = int((candidate < zero).sum()) if best != "zero_shot" else 0
    paired = []
    for method in METHODS:
        for reference in ("zero_shot", "affine"):
            if method == reference:
                continue
            values = aulc.loc[aulc.method.eq(method)].set_index("seed").normalized_aulc
            baseline = aulc.loc[aulc.method.eq(reference)].set_index("seed").normalized_aulc
            delta = values-baseline
            paired.append({"method": method, "reference": reference, "mean_delta": float(delta.mean()),
                           "median_delta": float(delta.median()), "std_delta": float(delta.std(ddof=1)),
                           "wins": int((delta < 0).sum()), "seeds": len(delta),
                           "stable_improvement": bool(delta.mean() < 0 and delta.median() < 0 and (delta < 0).sum() >= 4),
                           "per_seed_deltas": {str(k): float(v) for k, v in delta.items()}})
    atomic_json(STUDY / "results/paired_aulc_effects.json", paired)
    stable = best != "zero_shot" and next(row for row in paired if row["method"] == best and row["reference"] == "zero_shot")["stable_improvement"]
    decision = {"decision": "TRANSFER_BASELINE_CANDIDATE" if stable else "SOURCE_TARGET_SHIFT_DIAGNOSIS_REQUIRED",
                "best_method": best, "stable_improvement_vs_zero_shot": stable, "wins_vs_zero_shot": wins,
                "source_checkpoint_sha256": protocol["source_checkpoint_sha256"], "parameter_counts": params,
                "next_action": "INDEPENDENT_TRANSFER_VALIDATION_WITH_PARALLEL_UQ_QUALIFICATION" if stable else "SOURCE_TARGET_SHIFT_AND_OUTPUT_CALIBRATION_DIAGNOSIS",
                "active_transfer_executed": False, "candidate_is_developmental": True,
                "ranking_estimand": "fixed-source checkpoint, 5 target row partitions, 4 nested random label budgets",
                "historical_transfer_results_role": "HISTORICAL_LEGACY_PREDICTOR_EVIDENCE"}
    atomic_json(STUDY / "decision.json", decision)
    (STUDY / "TRANSFER_BASELINE_REPORT.md").write_text(
        "# Final QGeoGNN-V2 4g-to-8g transfer baseline\n\n"
        f"Decision: `{decision['decision']}`. Best mean normalized AULC: `{best}`.\n\n"
        "The source is the preregistered standalone 4g row-seed-42 checkpoint. Target data: 574 rows without thresholding. Exact frozen T1 manifests define five outer seeds and budgets 30/50/70/100, each including eight fixed validation labels. All methods share the same random labels. Preprocessing stays at source-train statistics. Target test outcomes are accessed after all neural checkpoints freeze. Head-only freezes backbone BN statistics; last2 updates two final node layers and their one effective geometry update; full fine-tuning updates all parameters. Affine fits four external coefficients on target training rows only.\n\n"
        "Primary normalized AULC uses the T1 arithmetic mean of two source-standard-deviation-normalized RMSEs, integrated over budgets and divided by 70. The also-reported combined normalized RMSE is their root-mean-square. Validation RMS has the same ordering as normalized MSE. Across-seed std uses ddof=1.\n\n"
        "## AULC summary\n\n" + markdown_table(ranked) + "\n\n## Paired effects\n\n"
        + markdown_table(pd.DataFrame([{k: v for k, v in row.items() if k != "per_seed_deltas"} for row in paired]), index=False)
        + "\n\nStable improvement means negative mean and median paired AULC differences and at least 4/5 seed wins. Choosing a candidate from this fixed set is developmental; independent target/compound validation remains necessary. No old Legacy ranking is treated as a new result. No acquisition, adapter sweep, active learning or active transfer was performed.\n\n"
        + "## Budget aggregates\n\n" + markdown_table(summary) + "\n\n## Per-seed, per-budget metrics\n\n" + markdown_table(table, index=False) + "\n"
    )
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-context", nargs=2, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.run_context: run_context(*args.run_context)
    elif args.execute:
        prepare()
        partition = pd.read_csv(STUDY / "splits/partition_manifest.csv")
        schedule = pd.read_csv(STUDY / "splits/schedule_manifest.csv")
        contexts = [(seed, budget) for seed in dict.fromkeys(partition.outer_seed.astype(int))
                    for budget in sorted(schedule.budget.astype(int).unique())]
        logdir = STUDY / "runtime/logs"
        logdir.mkdir(parents=True, exist_ok=True)
        def launch(pair):
            seed, budget = pair
            with (logdir/f"{seed}_{budget}.log").open("a") as stream:
                completed = subprocess.run([sys.executable, __file__, "--run-context", str(seed), str(budget)], stdout=stream, stderr=subprocess.STDOUT)
            return seed, budget, completed.returncode
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            status = list(pool.map(launch, contexts))
        atomic_json(STUDY/"results/execution_audit.json", {"contexts": status, "failed": sum(r[2]!=0 for r in status)})
        if any(r[2]!=0 for r in status):
            raise RuntimeError("transfer context failure; recorded for identical-contract resume")
        summarize()
    elif args.summarize: summarize()
    else: prepare()


if __name__ == "__main__":
    main()

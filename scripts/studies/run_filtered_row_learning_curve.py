#!/usr/bin/env python3
"""Run B=100/300 on the exact filtered FULL row test/validation populations.

Budget counts gradient-train + validation labels. Recover the original seed
permutation without reading target labels, intersect its train order with the
filtered FULL eligible train set, and use nested prefixes. Every new prediction
is globally frozen before new test evaluation. FULL predictions are reused,
hash-verified, and never retrained. This is a descriptive learning curve on
historically examined tests, not independent confirmation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.studies import run_row_full_data_rmse_completion as full
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json
from src.qgeognn_al.transfer import (
    absolute_error_metrics, ea_fraction, fit_affine, fit_scale_only,
    fit_paper_style_current_v2_full, make_label_scrubbed_graphs, select_conditional_ea,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash

STUDY = ROOT / "studies/transfer/filtered_row_learning_curve_20261013"
COLUMNS, METHODS, SEEDS = full.COLUMNS, full.METHODS, full.SEEDS
BUDGETS = (100, 300)
require, read_json, freeze_json = full.require, full.read_json, full.freeze_json


def nested_schedule(full_schedule, original_ids_by_column, budgets=BUDGETS, columns=COLUMNS, seeds=SEEDS):
    """Label-free split construction; exact held-out roles and nested train sets."""
    blocks, accounts = [], []
    for column in columns:
        original_ids = list(map(str, original_ids_by_column[column]))
        require(len(original_ids) == len(set(original_ids)), "Duplicate original IDs")
        for seed in seeds:
            block = full_schedule.loc[full_schedule.column.eq(column) & full_schedule.protocol.eq("row")
                                      & full_schedule.outer_seed.eq(seed)].copy()
            require(not block.empty and block.sample_id.is_unique, "Missing/duplicate FULL context")
            role_sets = {role: set(block.loc[block.role.eq(role), "sample_id"].astype(str))
                         for role in ("gradient_train", "validation", "test")}
            require(all(role_sets.values()), "FULL role is empty")
            permutation = np.random.default_rng(seed).permutation(len(original_ids))
            train_order = [original_ids[i] for i in permutation if original_ids[i] in role_sets["gradient_train"]]
            require(set(train_order) == role_sets["gradient_train"], "Original IDs do not cover eligible FULL train")
            previous = set()
            for budget in budgets:
                n_train = budget - len(role_sets["validation"])
                require(2 <= n_train < len(train_order), "Budget must be below FULL and leave at least two train rows")
                selected = set(train_order[:n_train])
                require(previous <= selected, "Train prefixes are not nested")
                require(not (selected & role_sets["validation"] or selected & role_sets["test"]), "Held-out leakage")
                current = block.copy()
                current["full_role"] = current.role
                current.loc[current.role.eq("gradient_train"), "role"] = "pool"
                current.loc[current.sample_id.astype(str).isin(selected), "role"] = "gradient_train"
                current["budget"] = int(budget)
                current["actual_budget"] = int(budget)
                current["planned_budget"] = int(budget)
                current["training_label_count"] = n_train
                rank = {value: i + 1 for i, value in enumerate(train_order)}
                current["eligible_train_rank"] = current.sample_id.astype(str).map(rank).fillna(0).astype(int)
                blocks.append(current)
                accounts.append({"column": column, "protocol": "row", "seed": seed, "budget": budget,
                                 "n_train": n_train, "n_valid": len(role_sets["validation"]), "n_test": len(role_sets["test"]),
                                 "actual_budget": n_train + len(role_sets["validation"]),
                                 "full_actual_budget": len(train_order) + len(role_sets["validation"]),
                                 "train_ids_hash": stable_hash(sorted(selected)),
                                 "validation_ids_hash": stable_hash(sorted(role_sets["validation"])),
                                 "test_ids_hash": stable_hash(sorted(role_sets["test"])),
                                 "eligible_train_order_hash": stable_hash(train_order)})
                previous = selected
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(accounts)


def context(column, seed, budget):
    schedule = pd.read_csv(STUDY / "split_manifest.csv")
    block = schedule.loc[schedule.column.eq(column) & schedule.outer_seed.eq(seed) & schedule.budget.eq(budget)]
    require(not block.empty and block.sample_id.is_unique and set(block.protocol) == {"row"}, "Invalid context")
    return block


def role_ids(column, seed, budget, role):
    return context(column, seed, budget).loc[lambda f: f.role.eq(role), "sample_id"].astype(str).tolist()


def truth_for_role(column, seed, budget, role):
    require(role in ("gradient_train", "validation", "test"), "Unknown truth role")
    if role == "test":
        validate_global_freeze()
    return full.parent._read_authorized_truth(full.STUDY / "inputs" / f"filtered_canonical_{column}.csv",
                                              role_ids(column, seed, budget, role))


def runtime_path(column, seed, budget):
    return STUDY / "runtime/contexts" / column / "row" / f"seed_{seed}" / f"budget_{budget}"


def prepare():
    full.validate_inputs()
    full.validate_global_freeze()
    parent_protocol = read_json(full.STUDY / "protocol.json")
    originals = {column: pd.read_csv(full.parent.canonical_path(column), usecols=["sample_id"]).sample_id.astype(str).tolist()
                 for column in COLUMNS}
    schedule, accounting = nested_schedule(pd.read_csv(full.STUDY / "split_manifest.csv"), originals)
    require(len(accounting) == 30 and set(accounting.actual_budget) == set(BUDGETS), "Incomplete budget grid")
    STUDY.mkdir(parents=True, exist_ok=True)
    for frame, name in ((schedule, "split_manifest.csv"), (accounting, "label_accounting.csv")):
        path = STUDY / name
        if path.exists():
            require(pd.read_csv(path).equals(frame), f"Frozen schedule changed: {name}")
        else:
            frame.to_csv(path, index=False)
    code_paths = [str(Path(__file__).relative_to(ROOT)), *parent_protocol["implementation_sha256"].keys()]
    protocol = {
        "study_name": "FILTERED_ROW_LEARNING_CURVE_20261013", "status": "FROZEN_BEFORE_NEW_TEST_EVALUATION",
        "columns": list(COLUMNS), "protocols": ["row"], "outer_seeds": list(SEEDS), "methods": list(METHODS),
        "budgets": list(BUDGETS), "reused_budget": "FULL", "new_neural_fits": 30,
        "budget_definition": "gradient_train plus validation revealed target rows",
        "sampling": "original default_rng(seed) canonical row permutation; intersect eligible filtered FULL train; nested prefixes",
        "validation_test": "byte-identical ID sets to filtered FULL for every column/seed and every budget",
        "thresholds_ml": parent_protocol["thresholds_ml"], "threshold_selection": "none; inherited frozen thresholds",
        "source_checkpoint_sha256": full.parent.SOURCE_SHA256,
        "paper_style_current_v2": parent_protocol["paper_style_current_v2"],
        "conditional_EA": parent_protocol["conditional_EA"],
        "scale_only_affine": "same source q50, same gradient-train-only OLS implementations as FULL",
        "zero_shot": "reuse identical FULL predictions at each context; zero target training labels",
        "test_labels_used_for_fit_or_selection": False,
        "evaluation_boundary": "all 30 new contexts hash-frozen plus 15 reused FULL contexts verified before new test evaluation",
        "full_study": str(full.STUDY.relative_to(ROOT)),
        "full_protocol_sha256": sha256_file(full.STUDY / "protocol.json"),
        "full_global_freeze_sha256": sha256_file(full.STUDY / "all_predictions_frozen.json"),
        "full_metrics_sha256": sha256_file(full.STUDY / "all_metrics.csv"),
        "full_summary_sha256": sha256_file(full.STUDY / "summary.csv"),
        "canonical_id_order_hashes": {c: stable_hash(ids) for c, ids in originals.items()},
        "implementation_sha256": {rel: sha256_file(ROOT / rel) for rel in code_paths},
        "split_manifest_sha256": sha256_file(STUDY / "split_manifest.csv"),
        "label_accounting_sha256": sha256_file(STUDY / "label_accounting.csv"),
        "source_main_commit": parent_protocol["source_main_commit"],
        "interpretation": "descriptive learning curve on historically examined tests; no independent-confirmation claim",
        "plot_x": "actual revealed label count; FULL mean across five seeds, never equal categorical spacing",
    }
    freeze_json(STUDY / "protocol.json", protocol)
    print(accounting.groupby(["column", "budget"])[["n_train", "n_valid", "n_test", "full_actual_budget"]].mean().to_string(), flush=True)
    return protocol


def validate_inputs():
    full.validate_inputs()
    p = read_json(STUDY / "protocol.json")
    for name in ("protocol", "all_predictions_frozen", "all_metrics", "summary"):
        key = {"protocol": "full_protocol_sha256", "all_predictions_frozen": "full_global_freeze_sha256",
               "all_metrics": "full_metrics_sha256", "summary": "full_summary_sha256"}[name]
        suffix = ".csv" if name in ("all_metrics", "summary") else ".json"
        require(sha256_file(full.STUDY / (name + suffix)) == p[key], f"FULL artifact changed: {name}")
    for rel, digest in p["implementation_sha256"].items():
        require(sha256_file(ROOT / rel) == digest, f"Implementation changed: {rel}")
    require(sha256_file(STUDY / "split_manifest.csv") == p["split_manifest_sha256"], "Split changed")
    require(sha256_file(STUDY / "label_accounting.csv") == p["label_accounting_sha256"], "Accounting changed")
    return p


def fit_context(column, seed, budget):
    started = time.perf_counter()
    validate_inputs()
    ids = {role: role_ids(column, seed, budget, role) for role in ("gradient_train", "validation", "test")}
    for role in ("validation", "test"):
        require(ids[role] == full.role_ids(column, seed, role), f"Held-out identity/order mismatch: {role}")
    require(len(ids["gradient_train"]) + len(ids["validation"]) == budget, "Budget accounting mismatch")
    runtime = runtime_path(column, seed, budget)
    contract = {"column": column, "protocol": "row", "seed": seed, "budget": budget,
                "protocol_sha256": sha256_file(STUDY / "protocol.json"),
                **{f"{role}_ids_hash": stable_hash(sorted(values)) for role, values in ids.items()},
                "test_labels_used_for_fit_or_selection": 0}
    if (runtime / "frozen.json").exists():
        frozen = read_json(runtime / "frozen.json")
        require(frozen["contract"] == contract, "Frozen contract mismatch")
        for name, digest in frozen["files"].items():
            require(sha256_file(runtime / name) == digest, f"Frozen prediction changed: {name}")
        return {"column": column, "seed": seed, "budget": budget, "status": "REUSED_COMPLETED"}
    feature = pd.read_csv(full.STUDY / "inputs" / f"filtered_features_{column}.csv")
    require(not {"V1_ml", "V2_ml"} & set(feature.columns), "Feature input contains truth")
    positions = {r: full.parent._positions(feature, values) for r, values in ids.items()}
    cache_source = pd.read_csv(full.STUDY / "inputs" / f"source_{column}.csv.gz").set_index("sample_id")
    source = cache_source.loc[feature.sample_id.astype(str), ["V1_source", "V2_source"]].to_numpy(float)
    train_truth = truth_for_role(column, seed, budget, "gradient_train")
    valid_truth = truth_for_role(column, seed, budget, "validation")
    test_source = source[positions["test"]]
    scale = fit_scale_only(train_truth, source[positions["gradient_train"]], test_source)
    affine = fit_affine(train_truth, source[positions["gradient_train"]], test_source)
    payload = torch.load(full.parent.SOURCE, map_location="cpu", weights_only=False)
    preprocessing = payload["preprocessing"]
    mass = float(column[:-1])
    conditional, conditional_audit = select_conditional_ea(
        train_truth=train_truth, train_source=source[positions["gradient_train"]],
        train_ea=ea_fraction(feature.iloc[positions["gradient_train"]]["PE/EA"]),
        validation_truth=valid_truth, validation_source=source[positions["validation"]],
        validation_ea=ea_fraction(feature.iloc[positions["validation"]]["PE/EA"]),
        predict_source=test_source, predict_ea=ea_fraction(feature.iloc[positions["test"]]["PE/EA"]),
        source_scales=[preprocessing["target_scales"][t] for t in ("V1", "V2")],
        mass_ratio=mass / 4, penalties=full.parent.CONDITIONAL_PENALTIES,
    )
    cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
    cache.update(torch.load(full.parent.PARENT / "data_audit" / f"graph_cache_{column}_only.pt", weights_only=False))
    atom, angle = make_label_scrubbed_graphs(feature, cache, preprocessing["scaler"])
    require(all(torch.count_nonzero(g.y).item() == 0 for g in atom), "Graphs expose unrevealed labels")
    model = load_predictor_checkpoint(full.parent.SOURCE)
    torch.set_num_threads(1)
    fresh_source = full.parent._model_points(model, atom, angle, np.arange(len(feature)))
    np.testing.assert_allclose(fresh_source, source, rtol=1e-7, atol=1e-6)
    nominal_context = np.array([[mass / 4, float(column == "25g"), float(column == "40g")]], dtype=np.float32)
    runtime.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "neural_started", "column": column, "seed": seed, "budget": budget,
                      "n_train": len(ids["gradient_train"]), "n_valid": len(ids["validation"])}), flush=True)
    paper, paper_audit = fit_paper_style_current_v2_full(
        source_model=model, base_atom=atom, angle=angle, contexts=np.repeat(nominal_context, len(feature), axis=0),
        train_indices=positions["gradient_train"], validation_indices=positions["validation"], test_indices=positions["test"],
        train_truth=train_truth, validation_truth=valid_truth, source_scales=preprocessing["target_scales"],
        seed=seed, config=dict(full.parent.PAPER_CONFIG), runtime=runtime / "paper_style_current_v2",
        contract={**contract, "method": "paper_style_current_v2", "config": full.parent.PAPER_CONFIG,
                  "source_checkpoint_sha256": full.parent.SOURCE_SHA256},
    )
    inherited_prediction = pd.read_csv(full.runtime_path(column, seed) / "predictions_blind.csv.gz").set_index("sample_id")
    zero = inherited_prediction.loc[ids["test"], ["zero_shot_V1", "zero_shot_V2"]].to_numpy(float)
    np.testing.assert_allclose(zero, test_source, rtol=1e-7, atol=1e-6)
    predictions = {"zero_shot": zero, "scale_only": scale.prediction, "affine": affine.prediction,
                   "conditional_EA": conditional, "paper_style_current_v2": paper}
    frame = pd.DataFrame({"sample_id": ids["test"]})
    for method in METHODS:
        points = predictions[method]
        require(points.shape == (len(ids["test"]), 2) and np.isfinite(points).all(), "Invalid predictions")
        frame[f"{method}_V1"], frame[f"{method}_V2"] = points[:, 0], points[:, 1]
    frame.to_csv(runtime / "predictions_blind.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    audit = {"contract": contract, "zero_shot": {"target_training_labels": 0, "same_as_full": True},
             "scale_only": {"fit_role": "gradient_train", "coefficients": scale.coefficients.tolist()},
             "affine": {"fit_role": "gradient_train", "coefficients": affine.coefficients.tolist()},
             "conditional_EA": conditional_audit, "paper_style_current_v2": paper_audit,
             "test_graph_labels": "zero_sentinel", "test_labels_used_for_fit_or_selection": 0}
    atomic_json(runtime / "fit_audit.json", audit)
    names = ["predictions_blind.csv.gz", "fit_audit.json", *[f"paper_style_current_v2/{n}" for n in ("predictions.csv.gz", "fit_audit.json", "best.pt")]]
    freeze_json(runtime / "frozen.json", {"contract": contract, "methods": list(METHODS),
                                         "files": {n: sha256_file(runtime / n) for n in names}})
    return {"column": column, "seed": seed, "budget": budget, "status": "COMPLETED",
            "seconds": time.perf_counter() - started, "epochs": paper_audit["epochs_run"]}


def freeze_all():
    full.validate_global_freeze()
    records = {}
    for column in COLUMNS:
        for seed in SEEDS:
            for budget in BUDGETS:
                path = runtime_path(column, seed, budget) / "frozen.json"
                record = read_json(path)
                require(record["contract"]["protocol_sha256"] == sha256_file(STUDY / "protocol.json"), "Contract drift")
                require(record["methods"] == list(METHODS), "Missing methods")
                for name, digest in record["files"].items():
                    require(sha256_file(path.parent / name) == digest, f"Prediction changed: {name}")
                records[str(path.relative_to(STUDY))] = sha256_file(path)
    freeze_json(STUDY / "all_predictions_frozen.json", {
        "new_contexts": 30, "reused_full_contexts": 15, "methods": list(METHODS),
        "protocol_sha256": sha256_file(STUDY / "protocol.json"), "files": records,
        "full_global_freeze_sha256": sha256_file(full.STUDY / "all_predictions_frozen.json"),
        "test_truth_evaluation_started": False,
    })


def validate_global_freeze():
    path = STUDY / "all_predictions_frozen.json"
    require(path.exists(), "Test truth blocked: all budget predictions are not globally frozen")
    freeze = read_json(path)
    require(freeze["new_contexts"] == 30 and len(freeze["files"]) == 30, "Incomplete global freeze")
    require(freeze["protocol_sha256"] == sha256_file(STUDY / "protocol.json"), "Protocol drift")
    require(freeze["full_global_freeze_sha256"] == sha256_file(full.STUDY / "all_predictions_frozen.json"), "FULL freeze drift")
    full.validate_global_freeze()
    for rel, digest in freeze["files"].items():
        record_path = STUDY / rel
        require(sha256_file(record_path) == digest, f"Context freeze changed: {rel}")
        record = read_json(record_path)
        require(record["contract"]["test_labels_used_for_fit_or_selection"] == 0, "Fit leakage")
        for name, value in record["files"].items():
            require(sha256_file(record_path.parent / name) == value, f"Prediction changed: {name}")
    return freeze


def evaluate():
    validate_inputs()
    validate_global_freeze()
    scales = torch.load(full.parent.SOURCE, map_location="cpu", weights_only=False)["preprocessing"]["target_scales"]
    rows = []
    for column in COLUMNS:
        for seed in SEEDS:
            ids = role_ids(column, seed, BUDGETS[0], "test")
            truth = truth_for_role(column, seed, BUDGETS[0], "test")
            require(np.all(truth <= np.array(full.THRESHOLDS[column])), "Test outside threshold domain")
            for budget in BUDGETS:
                require(ids == role_ids(column, seed, budget, "test"), "Budget test order differs")
                frame = pd.read_csv(runtime_path(column, seed, budget) / "predictions_blind.csv.gz").set_index("sample_id").loc[ids]
                for method in METHODS:
                    rows.append({"column": column, "protocol": "row", "seed": seed, "budget": str(budget), "method": method,
                                 "n_train": len(role_ids(column, seed, budget, "gradient_train")),
                                 "n_valid": len(role_ids(column, seed, budget, "validation")), "n_test": len(ids),
                                 "actual_budget": budget,
                                 "provenance": "REUSED_ZERO_SHOT" if method == "zero_shot" else "NEW_FILTERED_BUDGET_FIT",
                                 **absolute_error_metrics(truth, frame[[f"{method}_V1", f"{method}_V2"]].to_numpy(float), scales)})
    inherited = pd.read_csv(full.STUDY / "all_metrics.csv")
    inherited["budget"] = "FULL"
    inherited["actual_budget"] = inherited.n_train + inherited.n_valid
    inherited["provenance"] = "REUSED_FROZEN_FULL"
    metrics = pd.concat([pd.DataFrame(rows), inherited], ignore_index=True)
    require(len(metrics) == 225, "Expected 225 method/seed/budget records")
    summary_frames = []
    for budget, block in metrics.groupby("budget", sort=False):
        summary = full.parent._summary_rows(block)
        summary["budget"] = budget
        summary["actual_budget_mean"] = summary.n_train_mean + summary.n_valid_mean
        summary_frames.append(summary)
    summary = pd.concat(summary_frames, ignore_index=True)
    require(len(summary) == 45 and set(summary.n_seeds) == {5}, "Summary grid incomplete")
    require(np.all(summary.loc[summary.budget.eq("100"), "actual_budget_mean"] == 100), "B100 count error")
    require(np.all(summary.loc[summary.budget.eq("300"), "actual_budget_mean"] == 300), "B300 count error")
    for column in COLUMNS:
        for seed in SEEDS:
            zero = metrics.loc[metrics.column.eq(column) & metrics.seed.eq(seed) & metrics.method.eq("zero_shot")]
            require(np.ptp(zero.V1_rmse.to_numpy()) < 1e-12 and np.ptp(zero.V2_rmse.to_numpy()) < 1e-12,
                    "Zero-shot changed with budget/test")
    metrics.to_csv(STUDY / "all_metrics.csv", index=False)
    summary.to_csv(STUDY / "summary.csv", index=False)
    atomic_json(STUDY / "completion_audit.json", {
        "new_contexts": 30, "reused_full_contexts": 15, "method_seed_budget_records": 225, "summary_rows": 45,
        "same_tests_all_budgets": True, "same_validation_all_budgets": True, "nested_train_prefixes": True,
        "zero_shot_identical_all_budgets": True, "full_results_reused_unchanged": True,
        "test_evaluation_after_global_freeze": True, "test_labels_used_for_fit_or_selection": 0,
        "summary_sha256": sha256_file(STUDY / "summary.csv"), "metrics_sha256": sha256_file(STUDY / "all_metrics.csv"),
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "global_freeze_sha256": sha256_file(STUDY / "all_predictions_frozen.json"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__},
    })
    print(summary[["column", "budget", "actual_budget_mean", "method", "V1_rmse_mean", "V2_rmse_mean"]].to_string(index=False), flush=True)


def execute(workers):
    prepare()
    def worker(column, seed, budget):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--fit-context", column, str(seed), str(budget)]
        log = STUDY / "runtime/logs" / f"{column}_{seed}_b{budget}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f"Fit failed: {column}/{seed}/{budget}; see {log}")
        lines = [line for line in log.read_text().splitlines() if line.startswith('{"column"')]
        return "\n".join(lines)
    tasks = [(column, seed, budget) for budget in BUDGETS for column in COLUMNS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, *task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    freeze_all()
    print("All 30 new contexts + 15 FULL contexts frozen; starting evaluation.", flush=True)
    evaluate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--fit-context", nargs=3, metavar=("COLUMN", "SEED", "BUDGET"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(args.workers)
    elif args.evaluate:
        evaluate()
    else:
        column, seed, budget = args.fit_context
        require(column in COLUMNS and int(seed) in SEEDS and int(budget) in BUDGETS, "Unknown context")
        print(json.dumps(fit_context(column, int(seed), int(budget))), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Complete the requested filtered FULL/row, three-column five-method figure.

Reuse frozen 25g/40g results; fit 8g and all missing affine arms only.
No existing experiment is overwritten. Every new prediction is globally frozen
before evaluation reads test labels. Inherited test populations have previously
been examined: this is a descriptive completion, not independent confirmation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.studies import run_filtered_full_data_benchmark as parent
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json
from src.qgeognn_al.transfer import (
    absolute_error_metrics, build_full_data_schedule, ea_fraction, fit_affine,
    fit_paper_style_current_v2_full, fit_scale_only, make_label_scrubbed_graphs,
    select_conditional_ea,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash

STUDY = ROOT / "studies/transfer/row_full_data_rmse_completion_20261013"
BASELINE = ROOT / "studies/transfer/full_data_baseline_finalization"
HEADROOM = ROOT / "studies/transfer/filtered_transfer_headroom_audit"
COLUMNS = ("8g", "25g", "40g")
METHODS = ("zero_shot", "scale_only", "affine", "conditional_EA", "paper_style_current_v2")
REUSED = tuple(m for m in METHODS if m != "affine")
SEEDS = parent.SEEDS
THRESHOLDS = {"8g": (60.0, 120.0), **parent.THRESHOLDS}
CODE_PATHS = (
    "scripts/studies/run_row_full_data_rmse_completion.py",
    "scripts/studies/run_filtered_full_data_benchmark.py",
    "src/qgeognn_al/transfer/full_data.py",
    "src/qgeognn_al/transfer/calibration.py",
    "src/qgeognn_al/transfer/conditional_scaling.py",
    "src/qgeognn_al/transfer/adaptation.py",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text())


def freeze_json(path, value):
    if path.exists():
        require(read_json(path) == value, f"Refusing to alter frozen record: {path}")
    else:
        atomic_json(path, value)


def context(column, seed):
    frame = pd.read_csv(STUDY / "split_manifest.csv")
    result = frame.loc[frame.column.eq(column) & frame.protocol.eq("row")
                       & frame.outer_seed.eq(seed)].copy()
    require(result.sample_id.is_unique and not result.empty, "Missing/duplicate context IDs")
    return result


def role_ids(column, seed, role):
    return context(column, seed).loc[lambda f: f.role.eq(role), "sample_id"].astype(str).tolist()


def truth_for_role(column, seed, role):
    require(role in ("gradient_train", "validation", "test"), "Unknown role")
    if role == "test":
        validate_global_freeze()
    return parent._read_authorized_truth(STUDY / "inputs" / f"filtered_canonical_{column}.csv",
                                         role_ids(column, seed, role))


def runtime_path(column, seed):
    return STUDY / "runtime" / "contexts" / column / "row" / f"seed_{seed}"


def validate_inputs():
    protocol = read_json(STUDY / "protocol.json")
    for rel, digest in protocol["inputs_sha256"].items():
        require(sha256_file(STUDY / rel) == digest, f"Frozen input changed: {rel}")
    for rel, digest in protocol["implementation_sha256"].items():
        require(sha256_file(ROOT / rel) == digest, f"Frozen implementation changed: {rel}")
    for rel, digest in protocol["inherited_artifacts_sha256"].items():
        require(sha256_file(ROOT / rel) == digest, f"Inherited artifact changed: {rel}")
    require(sha256_file(STUDY / "split_manifest.csv") == protocol["split_manifest_sha256"], "Split drift")
    require(sha256_file(STUDY / "label_accounting.csv") == protocol["label_accounting_sha256"], "Label accounting drift")
    require(sha256_file(parent.SOURCE) == parent.SOURCE_SHA256, "Source checkpoint drift")
    return protocol


def verify_parent_context(column, seed):
    old_runtime = parent._runtime_context(column, "row", seed)
    relative = str((old_runtime / "frozen.json").relative_to(parent.STUDY))
    global_parent = read_json(parent.STUDY / "all_predictions_frozen.json")
    require(sha256_file(old_runtime / "frozen.json") == global_parent["files"][relative], "Parent freeze drift")
    frozen = read_json(old_runtime / "frozen.json")
    require(frozen["contract"]["protocol_sha256"] == sha256_file(parent.STUDY / "protocol.json"), "Parent protocol drift")
    require(frozen["contract"]["test_labels_used_for_fit_or_selection"] == 0, "Parent fit leakage")
    for role in ("gradient_train", "validation", "test"):
        require(stable_hash(sorted(role_ids(column, seed, role))) == frozen["contract"][f"{role}_ids_hash"],
                f"Parent role mismatch: {column}/{seed}/{role}")
    for name, digest in frozen["files"].items():
        require(sha256_file(old_runtime / name) == digest, f"Parent runtime changed: {name}")
    return old_runtime, frozen


def prepare():
    require(sha256_file(parent.SOURCE) == parent.SOURCE_SHA256, "Source checkpoint drift")
    require(sha256_file(parent.PARENT / "splits/schedule_manifest.csv") == parent.SCHEDULE_SHA256,
            "Parent schedule drift")
    parent_protocol = read_json(parent.STUDY / "protocol.json")
    require(read_json(HEADROOM / "protocol.json")["source_checkpoint_sha256"] == parent.SOURCE_SHA256,
            "Inherited source cache checkpoint mismatch")
    base_manifest = read_json(BASELINE / "artifact_manifest.json")
    inputs = STUDY / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    inherited = {}
    features = {}
    for column in COLUMNS:
        source_dir = BASELINE if column == "8g" else parent.STUDY
        canonical = source_dir / f"filtered_canonical_{column}.csv"
        expected = (base_manifest["files"][canonical.name] if column == "8g"
                    else parent_protocol["filtered_canonical_sha256"][column])
        require(sha256_file(canonical) == expected, f"Filtered population drift: {column}")
        for kind in ("canonical", "features"):
            src = source_dir / f"filtered_{kind}_{column}.csv"
            dest = inputs / src.name
            if dest.exists():
                require(sha256_file(src) == sha256_file(dest), "Frozen input copy changed")
            else:
                shutil.copyfile(src, dest)
            inherited[str(src.relative_to(ROOT))] = sha256_file(src)
        features[column] = pd.read_csv(inputs / f"filtered_features_{column}.csv")
        require(not {"V1_ml", "V2_ml"} & set(features[column].columns), "Feature input exposes truth")
        require(len(features[column]) == {"8g": 552, "25g": 408, "40g": 456}[column], "Filtered row count drift")
        source_path = (BASELINE / "runtime/source_8g.csv.gz" if column == "8g"
                       else HEADROOM / "runtime" / f"source_{column}.csv.gz")
        meta = read_json(source_path.with_name(f"source_{column}.json"))
        require(sha256_file(source_path) == meta.get("prediction_sha256", meta.get("sha256")), "Source prediction cache drift")
        frozen_source = inputs / source_path.name
        if frozen_source.exists():
            require(sha256_file(frozen_source) == sha256_file(source_path), "Source copy drift")
        else:
            shutil.copyfile(source_path, frozen_source)
        inherited[str(source_path.relative_to(ROOT))] = sha256_file(source_path)
    schedule, accounting = build_full_data_schedule(
        pd.read_csv(parent.PARENT / "splits/schedule_manifest.csv"), features,
        columns=COLUMNS, protocols=("row",), seeds=SEEDS, parent_budget=100,
    )
    inherited_schedule = pd.read_csv(BASELINE / "split_manifest.csv")
    keys = ["column", "protocol", "outer_seed", "sample_id", "role"]
    expected = inherited_schedule.loc[inherited_schedule.protocol.eq("row"), keys].sort_values(keys).reset_index(drop=True)
    require(schedule[keys].sort_values(keys).reset_index(drop=True).equals(expected), "Inherited full roles differ")
    require(len(accounting) == 15, "Expected 15 row contexts")
    for role, minimum in parent.MINIMUM_ROLE_COUNTS.items():
        require(accounting[f"{role}_rows"].min() >= minimum, f"Insufficient {role} rows")
    for frame, name in ((schedule, "split_manifest.csv"), (accounting, "label_accounting.csv")):
        path = STUDY / name
        if path.exists():
            require(pd.read_csv(path).equals(frame.reset_index(drop=True)), f"Frozen table drift: {name}")
        else:
            frame.to_csv(path, index=False)
    # Source graph caches and checkpoint are verified; no target truth is parsed here.
    for column in COLUMNS:
        inherited[str((parent.PARENT / "data_audit" / f"graph_cache_{column}_only.pt").relative_to(ROOT))] = sha256_file(
            parent.PARENT / "data_audit" / f"graph_cache_{column}_only.pt")
    inherited[str(SOURCE_GRAPH_CACHE.relative_to(ROOT))] = sha256_file(SOURCE_GRAPH_CACHE)
    implementations = {rel: sha256_file(ROOT / rel) for rel in CODE_PATHS}
    protocol = {
        "study_name": "ROW_FILTERED_FULL_DATA_RMSE_COMPLETION_20261013",
        "status": "FROZEN_BEFORE_NEW_TEST_EVALUATION", "budget": "FULL",
        "columns": list(COLUMNS), "protocols": ["row"], "outer_seeds": list(SEEDS), "methods": list(METHODS),
        "thresholds_ml": {c: {"V1": v[0], "V2": v[1]} for c, v in THRESHOLDS.items()},
        "filter_definition": "inherited fixed legacy thresholds; V1<=limit and V2<=limit; no threshold selection",
        "full_data_definition": "all operationally filtered rows outside inherited validation/test are gradient_train",
        "source_checkpoint": str(parent.SOURCE.relative_to(ROOT)), "source_checkpoint_sha256": parent.SOURCE_SHA256,
        "parent_schedule_sha256": parent.SCHEDULE_SHA256,
        "parent_benchmark_protocol_sha256": sha256_file(parent.STUDY / "protocol.json"),
        "parent_benchmark_global_freeze_sha256": sha256_file(parent.STUDY / "all_predictions_frozen.json"),
        "source_main_commit": "42ab12600a353bba4a063ef5d1b623ee2f7b976a",
        "new_arms": {"8g": list(METHODS), "25g": ["affine"], "40g": ["affine"]},
        "reused_arms": {"25g": list(REUSED), "40g": list(REUSED)},
        "affine": "per-endpoint slope+intercept OLS, gradient_train only; same implementation as matched B100",
        "scale_only": "per-endpoint through-origin OLS, gradient_train only",
        "conditional_EA": {"penalties": list(parent.CONDITIONAL_PENALTIES), "selection": "validation only"},
        "paper_style_current_v2": {"config": parent.PAPER_CONFIG,
                                    "context": "[target_mass_g/4, is_25g, is_40g]",
                                    "selection": "validation only; project paper-inspired strategy, not original-paper result"},
        "zero_shot_target_training_labels": 0, "test_labels_used_for_fit_or_selection": False,
        "source_cache_verification": "float32 inference vs decimal CSV cache: rtol=1e-7, atol=1e-6 mL",
        "evaluation_boundary": "hash-verify all 15 context predictions before any new test evaluation",
        "interpretation": "descriptive completion on historically examined filtered tests; no independent-confirmation claim",
        "inherited_artifacts_sha256": inherited, "implementation_sha256": implementations,
        "inputs_sha256": {str(p.relative_to(STUDY)): sha256_file(p) for p in sorted(inputs.iterdir())},
        "split_manifest_sha256": sha256_file(STUDY / "split_manifest.csv"),
        "label_accounting_sha256": sha256_file(STUDY / "label_accounting.csv"),
    }
    freeze_json(STUDY / "protocol.json", protocol)
    for column in ("25g", "40g"):
        for seed in SEEDS:
            verify_parent_context(column, seed)
    print(accounting[["column", "seed", "gradient_train_rows", "validation_rows", "test_rows"]].to_string(index=False), flush=True)
    return protocol


def fit_context(column, seed):
    started = time.perf_counter()
    protocol = validate_inputs()
    require(sha256_file(STUDY / "split_manifest.csv") == protocol["split_manifest_sha256"], "Split drift")
    ids = {role: role_ids(column, seed, role) for role in ("gradient_train", "validation", "test")}
    runtime = runtime_path(column, seed)
    contract = {"column": column, "protocol": "row", "seed": seed,
                "protocol_sha256": sha256_file(STUDY / "protocol.json"),
                **{f"{role}_ids_hash": stable_hash(sorted(values)) for role, values in ids.items()},
                "test_labels_used_for_fit_or_selection": 0}
    if (runtime / "frozen.json").exists():
        frozen = read_json(runtime / "frozen.json")
        require(frozen["contract"] == contract, "Context contract drift")
        for name, digest in frozen["files"].items():
            require(sha256_file(runtime / name) == digest, f"Context drift: {name}")
        return {"column": column, "seed": seed, "status": "REUSED_COMPLETED"}
    feature = pd.read_csv(STUDY / "inputs" / f"filtered_features_{column}.csv")
    positions = {role: parent._positions(feature, values) for role, values in ids.items()}
    source_frame = pd.read_csv(STUDY / "inputs" / f"source_{column}.csv.gz").set_index("sample_id")
    source = source_frame.loc[feature.sample_id.astype(str), ["V1_source", "V2_source"]].to_numpy(float)
    require(np.isfinite(source).all(), "Invalid cached source")
    train_truth = truth_for_role(column, seed, "gradient_train")
    test_source = source[positions["test"]]
    affine = fit_affine(train_truth, source[positions["gradient_train"]], test_source)
    predictions = {"affine": affine.prediction}
    audit = {"contract": contract, "affine": {"coefficients": affine.coefficients.tolist(), "fit_role": "gradient_train"}}
    runtime.mkdir(parents=True, exist_ok=True)
    if column != "8g":
        old_runtime, old_freeze = verify_parent_context(column, seed)
        stored = pd.read_csv(old_runtime / "predictions_blind.csv.gz").set_index("sample_id")
        require(set(stored.index.astype(str)) == set(ids["test"]), "Reused test IDs differ")
        for method in REUSED:
            predictions[method] = stored.loc[ids["test"], [f"{method}_V1", f"{method}_V2"]].to_numpy(float)
        np.testing.assert_allclose(predictions["zero_shot"], test_source, rtol=0, atol=1e-7)
        audit["reused"] = {"runtime": str(old_runtime.relative_to(ROOT)), "files": old_freeze["files"],
                           "methods": list(REUSED), "frozen_json_sha256": sha256_file(old_runtime / "frozen.json")}
    else:
        validation_truth = truth_for_role(column, seed, "validation")
        payload = torch.load(parent.SOURCE, map_location="cpu", weights_only=False)
        preprocessing = payload["preprocessing"]
        scale = fit_scale_only(train_truth, source[positions["gradient_train"]], test_source)
        predictions.update({"zero_shot": test_source, "scale_only": scale.prediction})
        conditional, conditional_audit = select_conditional_ea(
            train_truth=train_truth, train_source=source[positions["gradient_train"]],
            train_ea=ea_fraction(feature.iloc[positions["gradient_train"]]["PE/EA"]),
            validation_truth=validation_truth, validation_source=source[positions["validation"]],
            validation_ea=ea_fraction(feature.iloc[positions["validation"]]["PE/EA"]),
            predict_source=test_source, predict_ea=ea_fraction(feature.iloc[positions["test"]]["PE/EA"]),
            source_scales=[preprocessing["target_scales"][target] for target in ("V1", "V2")],
            mass_ratio=2.0, penalties=parent.CONDITIONAL_PENALTIES,
        )
        predictions["conditional_EA"] = conditional
        cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
        cache.update(torch.load(parent.PARENT / "data_audit/graph_cache_8g_only.pt", weights_only=False))
        atom, angle = make_label_scrubbed_graphs(feature, cache, preprocessing["scaler"])
        require(all(torch.count_nonzero(g.y).item() == 0 for g in atom), "Graph labels not scrubbed")
        model = load_predictor_checkpoint(parent.SOURCE)
        torch.set_num_threads(1)
        fresh_source = parent._model_points(model, atom, angle, np.arange(len(feature)))
        # The inherited DataFrame contains float32 values serialized as their
        # shortest decimal round-trip strings (e.g. 4.702541). Compare at the
        # precision of that cache, not an unattainable float64 tolerance.
        np.testing.assert_allclose(fresh_source, source, rtol=1e-7, atol=1e-6)
        paper, paper_audit = fit_paper_style_current_v2_full(
            source_model=model, base_atom=atom, angle=angle,
            contexts=np.repeat(np.array([[2., 0., 0.]], dtype=np.float32), len(feature), axis=0),
            train_indices=positions["gradient_train"], validation_indices=positions["validation"],
            test_indices=positions["test"], train_truth=train_truth, validation_truth=validation_truth,
            source_scales=preprocessing["target_scales"], seed=seed, config=dict(parent.PAPER_CONFIG),
            runtime=runtime / "paper_style_current_v2",
            contract={**contract, "method": "paper_style_current_v2", "config": parent.PAPER_CONFIG,
                      "source_checkpoint_sha256": parent.SOURCE_SHA256},
        )
        predictions["paper_style_current_v2"] = paper
        audit.update({"zero_shot": {"target_labels_used": 0},
                      "scale_only": {"coefficients": scale.coefficients.tolist(), "fit_role": "gradient_train"},
                      "conditional_EA": conditional_audit, "paper_style_current_v2": paper_audit,
                      "source_cache_fresh_inference_verified": True, "test_graph_labels": "zero_sentinel"})
    require(set(predictions) == set(METHODS) and all(np.isfinite(p).all() for p in predictions.values()), "Incomplete predictions")
    frame = pd.DataFrame({"sample_id": ids["test"]})
    for method in METHODS:
        require(predictions[method].shape == (len(ids["test"]), 2), "Prediction shape mismatch")
        frame[f"{method}_V1"] = predictions[method][:, 0]
        frame[f"{method}_V2"] = predictions[method][:, 1]
    frame.to_csv(runtime / "predictions_blind.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    atomic_json(runtime / "fit_audit.json", audit)
    files = {name: sha256_file(runtime / name) for name in ("predictions_blind.csv.gz", "fit_audit.json")}
    if column == "8g":
        for name in ("best.pt", "predictions.csv.gz", "fit_audit.json"):
            rel = f"paper_style_current_v2/{name}"
            files[rel] = sha256_file(runtime / rel)
    freeze_json(runtime / "frozen.json", {"contract": contract, "files": files, "methods": list(METHODS)})
    return {"column": column, "seed": seed, "status": "COMPLETED", "seconds": time.perf_counter() - started,
            "paper_epochs": audit.get("paper_style_current_v2", {}).get("epochs_run")}


def freeze_all():
    records = {}
    for column in COLUMNS:
        for seed in SEEDS:
            runtime = runtime_path(column, seed)
            frozen = read_json(runtime / "frozen.json")
            require(frozen["contract"]["protocol_sha256"] == sha256_file(STUDY / "protocol.json"), "Freeze protocol drift")
            require(frozen["methods"] == list(METHODS), "Missing methods")
            for name, digest in frozen["files"].items():
                require(sha256_file(runtime / name) == digest, f"Modified prediction: {name}")
            records[str((runtime / "frozen.json").relative_to(STUDY))] = sha256_file(runtime / "frozen.json")
    freeze_json(STUDY / "all_predictions_frozen.json", {
        "contexts": 15, "methods": list(METHODS), "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "files": records, "test_truth_evaluation_started": False,
    })


def validate_global_freeze():
    path = STUDY / "all_predictions_frozen.json"
    require(path.exists(), "Test truth blocked: predictions are not globally frozen")
    freeze = read_json(path)
    require(freeze["contexts"] == 15 and freeze["methods"] == list(METHODS), "Incomplete global freeze")
    require(freeze["protocol_sha256"] == sha256_file(STUDY / "protocol.json"), "Global protocol drift")
    require(len(freeze["files"]) == 15, "Missing frozen contexts")
    for rel, digest in freeze["files"].items():
        require(sha256_file(STUDY / rel) == digest, f"Global context drift: {rel}")
        record_path = STUDY / rel
        record = read_json(record_path)
        require(record["contract"]["test_labels_used_for_fit_or_selection"] == 0, "Fit leakage")
        for name, value in record["files"].items():
            require(sha256_file(record_path.parent / name) == value, f"Global prediction drift: {name}")
    return freeze


def evaluate():
    validate_inputs()
    validate_global_freeze()
    source = torch.load(parent.SOURCE, map_location="cpu", weights_only=False)
    scales = source["preprocessing"]["target_scales"]
    rows = []
    # Nothing in evaluation can affect model selection, which is already frozen.
    for column in COLUMNS:
        for seed in SEEDS:
            ids = role_ids(column, seed, "test")
            truth = truth_for_role(column, seed, "test")
            require(np.all(truth <= np.asarray(THRESHOLDS[column])), "Test outside fixed threshold domain")
            predictions = pd.read_csv(runtime_path(column, seed) / "predictions_blind.csv.gz").set_index("sample_id").loc[ids]
            for method in METHODS:
                points = predictions[[f"{method}_V1", f"{method}_V2"]].to_numpy(float)
                rows.append({"column": column, "protocol": "row", "seed": seed, "method": method,
                             "n_train": len(role_ids(column, seed, "gradient_train")),
                             "n_valid": len(role_ids(column, seed, "validation")), "n_test": len(ids),
                             "provenance": "NEW_FULL_DATA_FIT" if column == "8g" or method == "affine" else "REUSED_MATCHED_FROZEN",
                             **absolute_error_metrics(truth, points, scales)})
    metrics = pd.DataFrame(rows)
    require(len(metrics) == 75 and set(metrics.protocol) == {"row"}, "Incomplete metrics")
    inherited = pd.read_csv(parent.STUDY / "all_metrics.csv")
    for row in metrics.loc[metrics.column.isin(("25g", "40g")) & metrics.method.isin(REUSED)].itertuples():
        previous = inherited.loc[inherited.column.eq(row.column) & inherited.protocol.eq("row")
                                 & inherited.seed.eq(row.seed) & inherited.method.eq(row.method)]
        require(len(previous) == 1, "Missing inherited metric")
        for target in ("V1", "V2"):
            require(abs(getattr(row, f"{target}_rmse") - previous[f"{target}_rmse"].iloc[0]) < 1e-9,
                    "Reused metric changed")
    summary = parent._summary_rows(metrics)
    require(len(summary) == 15 and set(summary.n_seeds) == {5}, "Five-seed summary incomplete")
    metrics.to_csv(STUDY / "all_metrics.csv", index=False)
    summary.to_csv(STUDY / "summary.csv", index=False)
    atomic_json(STUDY / "completion_audit.json", {
        "contexts": 15, "method_seed_metrics": 75, "summary_rows": 15,
        "new_method_seed_results": 35, "reused_method_seed_results": 40,
        "reused_rmse_matches_parent": True, "test_evaluation_after_global_freeze": True,
        "test_labels_used_for_fit_or_selection": 0,
        "summary_sha256": sha256_file(STUDY / "summary.csv"),
        "metrics_sha256": sha256_file(STUDY / "all_metrics.csv"),
        "global_freeze_sha256": sha256_file(STUDY / "all_predictions_frozen.json"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__},
    })
    print(summary[["column", "method", "n_train_mean", "n_seeds", "V1_rmse_mean", "V2_rmse_mean"]].to_string(index=False), flush=True)


def execute(workers):
    prepare()
    def worker(column, seed):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--fit-context", column, str(seed)]
        log = STUDY / "runtime" / "logs" / f"{column}_{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f"Fit failed: {column}/{seed}; see {log}")
        return log.read_text().strip()
    tasks = [(c, s) for c in COLUMNS for s in SEEDS]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, c, s) for c, s in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    freeze_all()
    print("All 15 contexts globally frozen; starting test evaluation.", flush=True)
    evaluate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--fit-context", nargs=2, metavar=("COLUMN", "SEED"))
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(args.workers)
    elif args.evaluate:
        evaluate()
    else:
        column, seed = args.fit_context
        require(column in COLUMNS and int(seed) in SEEDS, "Unknown context")
        print(json.dumps(fit_context(column, int(seed))), flush=True)


if __name__ == "__main__":
    main()

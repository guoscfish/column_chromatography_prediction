#!/usr/bin/env python3
"""Six frozen-manifest runs of the single standalone QGeoGNN-V2 predictor."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models import build_predictor, load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import (
    atomic_json, fit_source_preprocessing, point_metrics, predict, seed_everything,
    stable_hash, train_source,
)

STUDY = ROOT / "studies/predictor/final_4g_qualification"
HISTORICAL_SPLITS = ROOT / "studies/predictor/clean_4g_baseline_qualification/splits"
MODES = ("row", "compound")
SEEDS = (42, 525, 1101)
CONFIG = {"model_variant": "qgeognn_v2", "optimizer": "Adam", "learning_rate": .001,
          "weight_decay": 0., "batch_size": 2048, "maximum_epochs": 1000, "patience": 100,
          "shuffle": "deterministic_each_epoch", "loss_weights": {"V1": 1., "V2": 1.},
          "checkpoint_selection": "validation_combined_normalized_rmse",
          "normalization": "source_train_only", "test_during_training": False,
          "initialization": "direct_standalone_seeded_construction; no historical initializer used",
          "8g_rows_used": 0, "head_changed": False, "architecture_search": False}


def engineering_gate():
    path = ROOT / "studies/predictor/final_v2_engineering/equivalence_audit.json"
    gate = json.loads(path.read_text())
    if gate["status"] != "PASS":
        raise RuntimeError("standalone engineering gate did not pass")
    return sha256_file(path)


def prepare():
    engineering_sha = engineering_gate()
    frozen = json.loads((HISTORICAL_SPLITS / "split_manifest.json").read_text())
    data = pd.read_csv(SOURCE_DATA)
    assert len(data) == 4163 and data.canonical_smiles.nunique() == 217
    assert data.V1_ml.max() <= 60 and data.V2_ml.max() <= 120
    assert sha256_file(SOURCE_DATA) == frozen["canonical_filtered_source_sha256"]
    (STUDY / "splits").mkdir(parents=True, exist_ok=True)
    for item in frozen["splits"]:
        original = ROOT / item["path"]
        assert sha256_file(original) == item["sha256"]
        destination = STUDY / "splits" / original.name
        if destination.exists() and sha256_file(destination) != item["sha256"]:
            raise RuntimeError("existing final qualification split changed")
        shutil.copyfile(original, destination)
    shutil.copyfile(HISTORICAL_SPLITS / "qualification_dataset_manifest.csv", STUDY / "splits/qualification_dataset_manifest.csv")
    atomic_json(STUDY / "splits/split_manifest.json", {"source_manifest_sha256": sha256_file(HISTORICAL_SPLITS / "split_manifest.json"),
                "regenerated": False, "splits": frozen["splits"]})
    protocol = {**CONFIG, "modes": MODES, "seeds": SEEDS, "rows": 4163, "compounds": 217,
                "engineering_gate_sha256": engineering_sha, "source_sha256": sha256_file(SOURCE_DATA),
                "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
                "across_seed_std_ddof": 1,
                "source_selection_for_transfer": "row_seed_42 fixed before qualification results; no best-test selection",
                "gate": "numerical stability, no training collapse, row predictive signal, interpretable compound generalization and split gaps; no arbitrary R2 cutoff",
                "uq_audit_nonblocking_for_point_transfer": True}
    existing = STUDY / "protocol.json"
    if existing.exists() and json.loads(existing.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("frozen qualification protocol changed")
    atomic_json(existing, protocol)
    (STUDY / "PREREGISTRATION.md").write_text("# Final standalone 4g qualification\n\n"
        "Six runs only: row/compound × seeds 42, 525, 1101. Reuse the existing qualification manifests byte-for-byte; never regenerate. The 4163-row / 217-compound domain retains V1 ≤ 60 and V2 ≤ 120.\n\n"
        "Use direct seeded standalone initialization and the unchanged R2 effective network/head/loss. Adam lr 0.001, weight decay 0, batch 2048, max 1000 epochs, patience 100, deterministic epoch shuffle, target weights 1:1; fit all preprocessing on training rows. Only validation selects checkpoints; test is evaluated after reload of the frozen best checkpoint. The standalone initializer consumes RNG only for effective modules; it is not claimed to recreate the diagnostic canonical Legacy initialization.\n\n"
        "Judge stability, train/validation/test gaps, row predictive signal, and explainable compound generalization from the measured results; impose no arbitrary R² performance threshold. Quantile diagnostics do not block ordinary point transfer. Fix source selection now to row seed 42, regardless of the other runs' test results. Do not tune the head, backbone, data thresholds, or optimizer.\n")


def run_one(mode, seed, threads=2):
    engineering_gate()
    torch.set_num_threads(threads)
    path = STUDY / f"splits/{mode}_seed_{seed}.csv"
    manifest = json.loads((STUDY / "splits/split_manifest.json").read_text())
    expected = next(item for item in manifest["splits"] if item["split_mode"] == mode and item["seed"] == seed)
    assert sha256_file(path) == expected["sha256"]
    split = pd.read_csv(path)
    data = pd.read_csv(SOURCE_DATA).reset_index(drop=True)
    joined = data[["sample_id"]].merge(split[["sample_id", "split"]], on="sample_id", validate="one_to_one", how="left")
    assert len(joined) == 4163 and not joined["split"].isna().any()
    indices = {r: np.flatnonzero(joined["split"].eq(r)) for r in ("train", "validation", "test")}
    assert {r: len(i) for r, i in indices.items()} == expected["row_counts"]
    if mode == "compound":
        sets = {r: set(data.iloc[i].canonical_smiles) for r, i in indices.items()}
        assert not (sets["train"] & sets["test"] or sets["train"] & sets["validation"] or sets["test"] & sets["validation"])
    result = STUDY / f"results/{mode}/seed_{seed}"
    runtime = STUDY / f"runtime/{mode}/seed_{seed}"
    result.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    config = {**CONFIG, "mode": mode, "seed": seed, "split_sha256": sha256_file(path), "cpu_threads": threads}
    config["config_hash"] = stable_hash(config)
    if (result / "run_summary.json").exists():
        summary = json.loads((result / "run_summary.json").read_text())
        assert summary["config_hash"] == config["config_hash"]
        assert sha256_file(runtime / "best.pt") == summary["checkpoint_sha256"]
        print(f"{mode}/{seed}: complete; reusing verified run", flush=True)
        return
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    norm, preprocessing = fit_source_preprocessing(data, split, cache, runtime / "scaler.json")
    atom, angle = build_model_data(data, cache, pd.DataFrame(), preprocessing["scaler"])
    seed_everything(seed)
    model = build_predictor(norm)
    atomic_json(result / "config.json", config)
    atomic_json(result / "normalization.json", {**preprocessing, "condition": asdict(norm)})
    started = time.time()
    def progress(record):
        record.update(mode=mode, seed=seed)
        atomic_json(runtime / "progress.json", record)
        print(json.dumps(record), flush=True)
    history, best_epoch = train_source(model, atom, angle, indices["train"], indices["validation"], preprocessing,
                                      config, runtime / "best.pt", progress)
    pd.DataFrame(history).to_csv(result / "history.csv", index=False)
    # All test access happens after fitting and reloading the frozen checkpoint.
    model = load_predictor_checkpoint(runtime / "best.pt")
    metrics, predictions, diagnostics = {}, [], {}
    for role, positions in indices.items():
        truth, prediction, order = predict(model, atom, angle, positions)
        metrics[role] = point_metrics(truth, prediction, preprocessing["target_scales"])
        frame = pd.DataFrame(prediction, columns=[f"{t}_{q}" for t in ("V1", "V2") for q in ("q10", "q50", "q90")])
        frame["V1_true"], frame["V2_true"] = truth[:, 0], truth[:, 1]
        frame["sample_id"] = data.iloc[order].sample_id.to_numpy()
        frame["split"] = role
        predictions.append(frame)
        diagnostics[role] = {t: {"prediction_std": float(prediction[:, 3*i+1].std()), "truth_std": float(truth[:, i].std()),
                                 "zero_prediction_fraction": float((prediction[:, 3*i+1] == 0).mean())}
                             for i, t in enumerate(("V1", "V2"))}
    pd.concat(predictions).to_csv(result / "predictions.csv.gz", index=False, compression="gzip")
    atomic_json(result / "metrics.json", metrics)
    atomic_json(result / "diagnostics.json", diagnostics)
    summary = {"mode": mode, "seed": seed, "config_hash": config["config_hash"], "best_epoch": best_epoch,
               "epochs_run": len(history), "parameter_count": sum(p.numel() for p in model.parameters()),
               "checkpoint_sha256": sha256_file(runtime / "best.pt"), "checkpoint_path": str((runtime / "best.pt").relative_to(ROOT)),
               "metrics": metrics, "training_seconds": time.time()-started, "status": "COMPLETE"}
    atomic_json(result / "run_summary.json", summary)
    print(json.dumps({"completed": [mode, seed], "best_epoch": best_epoch, "epochs": len(history), "test": metrics["test"]}), flush=True)


def execute(workers):
    prepare()
    logdir = STUDY / "runtime/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    def launch(pair):
        mode, seed = pair
        with (logdir / f"{mode}_{seed}.log").open("w") as output:
            completed = subprocess.run([sys.executable, __file__, "--run", mode, str(seed)], stdout=output, stderr=subprocess.STDOUT)
        atomic_json(STUDY / f"runtime/{mode}/seed_{seed}/process_result.json", {"returncode": completed.returncode})
        return mode, seed, completed.returncode
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(launch, [(mode, seed) for mode in MODES for seed in SEEDS]))
    atomic_json(STUDY / "results/execution_audit.json", {"runs": results, "failed_count": sum(r[2] != 0 for r in results)})
    if any(r[2] != 0 for r in results):
        raise RuntimeError("qualification run failure; see complete logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run", nargs=2)
    args = parser.parse_args()
    if args.run:
        run_one(args.run[0], int(args.run[1]))
    elif args.execute:
        execute(args.workers)
    else:
        prepare()

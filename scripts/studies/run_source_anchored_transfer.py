#!/usr/bin/env python3
"""Blind, resumable 120-context matched representation-transfer experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies.run_scaling_failure_audit import FEATURES, selected_truth
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json, loader_pair, seed_everything, target_loss
from src.qgeognn_al.transfer.source_anchored import METHODS, SourceAnchoredTransfer

STUDY = ROOT / "studies/transfer/source_anchored_shared_transfer"
OLD = ROOT / "studies/transfer/cross_column"
CONDITIONAL = ROOT / "studies/transfer/scaling_failure_audit"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SPLIT = ROOT / "studies/predictor/final_4g_qualification/splits/row_seed_42.csv"
SCHEDULE = OLD / "splits/schedule_manifest.csv"
BASELINES = ("scale_only", "local_identity_shrinkage", "conditional_EA", "conditional_policy", "target_head_only")
COLUMNS = ("8g", "25g", "40g")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
BUDGETS = (30, 50, 70, 100)
CONFIG = {"learning_rate": 1e-4, "weight_decay": 1e-5, "maximum_epochs": 500,
          "patience": 100, "batch_size": 2048, "lambda_source": 1., "cpu_threads": 1,
          "target_order_rng": "seed*10000+epoch", "replay_rng": "SeedSequence([seed,epoch,41004])",
          "source_batch_size": "target_train_count", "source_head_frozen": True,
          "source_replay_bn": "eval_statistics_with_gradients", "relative_error_floor_ml": 1.}


def target_path(column):
    if column not in COLUMNS:
        raise ValueError("invalid focal column")
    return OLD / "data_audit" / f"canonical_{column}.csv"


def context_path(column, protocol, seed, budget):
    return Path(column) / protocol / f"seed_{seed}" / f"budget_{budget}"


def verify_manifest(path):
    record = json.loads(path.read_text())
    for name, digest in record["files"].items():
        if sha256_file(path.parent / name) != digest:
            raise RuntimeError(f"frozen artifact changed: {path.parent / name}")
    return record


def prepare():
    """Lock design, data, source and reference prediction provenance before fits."""
    paths = [Path(__file__), ROOT / "src/qgeognn_al/transfer/source_anchored.py",
             ROOT / "src/qgeognn_al/models/qgeognn_v2.py", ROOT / "src/qgeognn_al/training/predictor.py",
             ROOT / "src/qgeognn_al/data.py", ROOT / "src/qgeognn_al/transfer/baseline.py",
             STUDY / "MODEL_PREREGISTRATION.md", STUDY / "PRE_EXPERIMENT_AUDIT.md",
             SOURCE, SOURCE_DATA, SOURCE_GRAPH_CACHE, SOURCE_SPLIT, SCHEDULE]
    for c in COLUMNS:
        paths += [target_path(c), OLD / "data_audit" / f"graph_cache_{c}_only.pt",
                  CONDITIONAL / f"features_{c}.csv"]
    config = {"config": CONFIG, "methods": METHODS, "baselines": BASELINES,
              "columns": COLUMNS, "seeds": SEEDS, "budgets": BUDGETS, "contexts": 120,
              "new_fits": 480, "developmental_evidence": True, "donor_target_labels": 0,
              "test_used_for_selection": False,
              "hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}}
    config = json.loads(json.dumps(config))
    path = STUDY / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise RuntimeError("frozen protocol changed; never silently reuse or overwrite")
    else:
        atomic_json(path, config)
    return config


def ledger(column, protocol, seed, budget):
    schedule = pd.read_csv(SCHEDULE)
    context = schedule.loc[schedule.column.eq(column) & schedule.protocol.eq(protocol)
                           & schedule.outer_seed.eq(seed) & schedule.planned_budget.eq(budget)]
    if context.empty or context.sample_id.duplicated().any():
        raise ValueError("missing or duplicate context")
    ids = {r: sorted(context.loc[context.role.eq(r), "sample_id"].astype(str))
           for r in ("gradient_train", "validation", "test", "pool")}
    all_ids = sum(ids.values(), [])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("overlapping roles")
    actual = int(context.actual_budget.iloc[0])
    if actual != len(ids["gradient_train"])+len(ids["validation"]):
        raise ValueError("purchased budget != train+validation")
    if set(all_ids) != set(pd.read_csv(target_path(column), usecols=["sample_id"]).sample_id):
        raise ValueError("focal-column identity mismatch / donor leakage")
    if protocol == "compound":
        sets = [set(context.loc[context.role.eq(r), "canonical_smiles"])
                for r in ("gradient_train", "validation", "test")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("compound leakage")
    return {"column": column, "protocol": protocol, "seed": seed, "budget": budget,
            "actual_budget": actual, **ids, "other_target_column_labels_used": 0,
            "test_labels_used_for_training_or_selection": 0}


def read_graphs(path, ids, cache, preprocessing, reveal=False):
    frame = pd.read_csv(path, usecols=FEATURES).set_index("sample_id").loc[ids].reset_index()
    frame[["V1_ml", "V2_ml"]] = selected_truth(path, ids) if reveal else 0.
    return build_model_data(frame, cache, pd.DataFrame(), preprocessing["scaler"])


def load_fitting_inputs(usage, preprocessing):
    """Construct separate purchased-train/validation and source-train objects only."""
    column = usage["column"]
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    focal_cache = {**cache, **torch.load(OLD / "data_audit" / f"graph_cache_{column}_only.pt", weights_only=False)}
    target = {role: read_graphs(target_path(column), usage[role], focal_cache, preprocessing, True)
              for role in ("gradient_train", "validation")}
    split = pd.read_csv(SOURCE_SPLIT)
    source_ids = sorted(split.loc[split.split.eq("train"), "sample_id"].astype(str))
    source = read_graphs(SOURCE_DATA, source_ids, cache, preprocessing, True)
    return target, source, source_ids, focal_cache, cache


def batch(graphs, positions=None):
    positions = range(len(graphs[0])) if positions is None else positions
    return next(zip(*loader_pair(*graphs, positions, CONFIG["batch_size"])))


def loss_pair(truth, prediction):
    return target_loss(truth[:, 0], prediction[:, :3])+target_loss(truth[:, 1], prediction[:, 3:])


def blind_predict(model, graphs, task="target"):
    model.eval()
    with torch.no_grad():
        return np.vstack([model(a, b, task=task).numpy()
                          for a, b in zip(*loader_pair(*graphs, range(len(graphs[0])), 128))])


def save_torch(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def fit(method, seed, target, source_graphs, source_ids, preprocessing, output, contract):
    """No test/probe graphs or other-target data enter this fitting interface."""
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "fit_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["contract"] != contract or sha256_file(output / "best.pt") != summary["checkpoint_sha256"]:
            raise RuntimeError("completed fit changed")
        return summary
    seed_everything(seed)
    scope, anchored = METHODS[method]
    model = SourceAnchoredTransfer(load_predictor_checkpoint(SOURCE), scope)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    valid = batch(target["validation"])
    count = len(target["gradient_train"][0])
    history, used, draws = [], set(), 0
    best, best_epoch, stale, start, elapsed = float("inf"), 0, 0, 1, 0.
    resume_path = output / "last.pt"
    if resume_path.exists():
        state = torch.load(resume_path, weights_only=False)
        if state["contract"] != contract:
            raise RuntimeError("resume contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, used, draws = state["history"], set(state["used"]), state["draws"]
        best, best_epoch, stale = state["best"], state["best_epoch"], state["stale"]
        start, elapsed = state["epoch"]+1, state["elapsed"]
    start_time = time.monotonic()
    for epoch in range(start, CONFIG["maximum_epochs"]+1):
        if stale >= CONFIG["patience"]:
            break
        model.training_target()
        order = np.random.default_rng(seed*10000+epoch).permutation(count)
        a, b = batch(target["gradient_train"], order)
        target_value = loss_pair(a.y, model(a, b))
        source_value = torch.tensor(0.)
        if anchored:
            indices = np.random.default_rng(np.random.SeedSequence([seed, epoch, 41004])).choice(len(source_ids), count, replace=False)
            sa, sb = batch(source_graphs, indices)
            with model.replay_mode():
                source_value = loss_pair(sa.y, model(sa, sb, task="source"))
            used.update(int(i) for i in indices)
            draws += count
        loss = target_value+CONFIG["lambda_source"]*source_value
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(*valid)
            score = point_metrics(valid[0].y.numpy(), prediction.numpy(), preprocessing["target_scales"])["combined_normalized_rmse"]
            valid_loss = float(loss_pair(valid[0].y, prediction))
        if not np.isfinite(score) or not np.isfinite(valid_loss):
            raise RuntimeError("nonfinite validation")
        history.append({"epoch": epoch, "target_train_loss": float(target_value.detach()),
                        "source_train_loss": float(source_value.detach()), "validation_loss": valid_loss,
                        "validation_score": score})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            save_torch(output / "best.pt", {"model": model.state_dict(), "contract": contract})
        else:
            stale += 1
        if epoch % 25 == 0 or stale >= CONFIG["patience"] or epoch == CONFIG["maximum_epochs"]:
            save_torch(resume_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "contract": contract, "history": history, "used": sorted(used), "draws": draws,
                       "best": best, "best_epoch": best_epoch, "stale": stale, "epoch": epoch,
                       "elapsed": elapsed+time.monotonic()-start_time})
            atomic_json(output / "progress.json", {"epoch": epoch, "best_epoch": best_epoch, "method": method})
        if stale >= CONFIG["patience"]:
            break
    model.load_state_dict(torch.load(output / "best.pt", weights_only=False)["model"])
    drift = model.parameter_drift(initial)
    if drift["source_head_l2_drift"] != 0:
        raise RuntimeError("frozen source head changed")
    summary = {"contract": contract, "method": method, "best_epoch": best_epoch, "epochs_run": len(history),
               "validation_score": best, "best_epoch_losses": history[best_epoch-1],
               "initial_epoch_losses": history[0], "final_epoch_losses": history[-1],
               "source_replay_draws": draws, "source_replay_unique_rows": len(used),
               "source_replay_ids": [source_ids[i] for i in sorted(used)],
               "target_train_count": count, "target_validation_count": len(target["validation"][0]),
               "wall_seconds": elapsed+time.monotonic()-start_time, "fit_success": True,
               "checkpoint_sha256": sha256_file(output / "best.pt"),
               **model.parameter_inventory(), **drift}
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    atomic_json(summary_path, summary)
    return summary


def freeze_references(relative, ids):
    path = CONDITIONAL / "models" / relative
    verify_manifest(path / "frozen.json")
    names = [n for n in BASELINES if n != "target_head_only"]
    cols = [f"{m}_{t}" for m in names for t in ("V1", "V2")]
    table = pd.read_csv(path / "predictions_blind.csv.gz", usecols=["sample_id", *cols]).set_index("sample_id").loc[ids]
    path = OLD / relative
    verify_manifest(path / "completion.json")
    cols = [f"target_head_only_{t}" for t in ("V1", "V2")]
    # This historical file also contains truth; never parse those columns.
    table[cols] = pd.read_csv(path / "predictions.csv.gz", usecols=["sample_id", *cols]).set_index("sample_id").loc[ids, cols]
    return table


def run_context(column, protocol, seed, budget):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepare()
    relative = context_path(column, protocol, seed, budget)
    output = STUDY / "contexts" / relative
    done = output / "frozen.json"
    if done.exists():
        verify_manifest(done)
        return
    usage = ledger(column, protocol, seed, budget)
    preprocessing = torch.load(SOURCE, weights_only=False)["preprocessing"]
    target, source_graphs, source_ids, cache, source_cache = load_fitting_inputs(usage, preprocessing)
    contract = {"protocol_sha256": sha256_file(STUDY / "protocol.json"),
                "column": column, "protocol": protocol, "seed": seed, "budget": budget,
                "target_train_ids": usage["gradient_train"], "target_validation_ids": usage["validation"]}
    fits = {}
    for method in METHODS:
        runtime = STUDY / "runtime" / relative / method
        fits[method] = fit(method, seed, target, source_graphs, source_ids, preprocessing, runtime, {**contract, "method": method})
        print(f"FIT {column}/{protocol}/{seed}/{budget} {method} epochs={fits[method]['epochs_run']}", flush=True)
    # Inference objects are constructed only after fitting; neither has truth.
    test_graphs = read_graphs(target_path(column), usage["test"], cache, preprocessing)
    split = pd.read_csv(SOURCE_SPLIT)
    probe_ids = sorted(split.loc[split.split.eq("validation"), "sample_id"].astype(str))
    probe_graphs = read_graphs(SOURCE_DATA, probe_ids, source_cache, preprocessing)
    table = freeze_references(relative, usage["test"])
    source_model = load_predictor_checkpoint(SOURCE)
    initial_model = SourceAnchoredTransfer(source_model, "shallow")
    probe = pd.DataFrame({"sample_id": probe_ids}).set_index("sample_id")
    probe[["initial_V1", "initial_V2"]] = blind_predict(initial_model, probe_graphs, "source")[:, [1, 4]]
    for method, (scope, _) in METHODS.items():
        model = SourceAnchoredTransfer(source_model, scope)
        model.load_state_dict(torch.load(STUDY / "runtime" / relative / method / "best.pt", weights_only=False)["model"])
        table[[f"{method}_V1", f"{method}_V2"]] = blind_predict(model, test_graphs)[:, [1, 4]]
        probe[[f"{method}_V1", f"{method}_V2"]] = blind_predict(model, probe_graphs, "source")[:, [1, 4]]
    if not np.isfinite(table.to_numpy()).all() or not np.isfinite(probe.to_numpy()).all():
        raise RuntimeError("nonfinite blind predictions")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "predictions_blind.csv.gz", compression={"method": "gzip", "mtime": 0})
    probe.to_csv(output / "source_probe_blind.csv.gz", compression={"method": "gzip", "mtime": 0})
    atomic_json(output / "fit_audit.json", fits)
    atomic_json(output / "label_usage.json", {**usage, "source_train_eligible_rows": len(source_ids),
                "source_probe_ids": probe_ids, "source_probe_used_for_selection": False})
    files = ["predictions_blind.csv.gz", "source_probe_blind.csv.gz", "fit_audit.json", "label_usage.json"]
    atomic_json(done, {"protocol_sha256": contract["protocol_sha256"], "frozen_at_unix": time.time(),
                       "files": {n: sha256_file(output/n) for n in files}})


def expected_contexts():
    return [(c, p, s, b) for c in COLUMNS for p in ("row", "compound") for s in SEEDS for b in BUDGETS]


def freeze_all():
    prepare()
    manifest = {}
    for context in expected_contexts():
        path = STUDY / "contexts" / context_path(*context) / "frozen.json"
        record = verify_manifest(path)
        if record["protocol_sha256"] != sha256_file(STUDY / "protocol.json"):
            raise RuntimeError("context protocol mismatch")
        manifest[str(path.relative_to(STUDY))] = sha256_file(path)
    freeze = {"contexts": 120, "fits": 480, "files": manifest,
              "protocol_sha256": sha256_file(STUDY / "protocol.json"), "target_test_evaluation_started": False}
    path = STUDY / "all_predictions_frozen.json"
    if path.exists() and json.loads(path.read_text()) != freeze:
        raise RuntimeError("global prediction freeze changed")
    atomic_json(path, freeze)


def execute(workers):
    prepare()
    logdir = STUDY / "runtime/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    def worker(context):
        column, protocol, seed, budget = context
        log = logdir / f"{column}_{protocol}_{seed}_{budget}.log"
        with log.open("a") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__)), "--context", *map(str, context)],
                                    stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"context failed: {context}; see {log}")
        return context
    contexts = expected_contexts()
    pending = [c for c in contexts if not (STUDY / "contexts" / context_path(*c) / "frozen.json").exists()]
    print(f"remaining contexts={len(pending)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, c) for c in pending]
        for n, future in enumerate(as_completed(futures), 1):
            context = future.result()
            print(f"FROZEN {len(contexts)-len(pending)+n}/120 {context}", flush=True)
    freeze_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--context", nargs=4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.context:
        c, p, s, b = args.context
        run_context(c, p, int(s), int(b))
    elif args.execute:
        execute(args.workers)
    elif args.freeze:
        freeze_all()
    else:
        prepare()

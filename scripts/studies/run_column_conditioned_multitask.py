#!/usr/bin/env python3
"""Preregistered A1/A2 study with separate inner, freeze, and score actions."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.transfer.adaptation import set_training_mode, target_loss_recipe
from src.qgeognn_al.transfer.column_conditioned_multitask import (
    ARCHITECTURES, TASKS, BalancedTaskBatcher, build_multitask_model,
    film_is_zero_initialized, global_target_group_folds, gradient_geometry,
    head_copies_are_exact, task_ids, cache_frozen_prefix,
)
from scripts.studies.run_filtered_full_data_benchmark import _read_authorized_truth

STUDY = ROOT / "studies/transfer/column_conditioned_multitask"
PARENT = ROOT / "studies/transfer/filtered_full_data_benchmark"
QUALIFIED = ROOT / "studies/predictor/final_4g_qualification"
SOURCE_DECISION = QUALIFIED / "decision.json"
SOURCE_SHA = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
SPLIT_SHA = "33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3"
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
FEATURES = ["sample_id", "canonical_smiles", "PE/EA", "loading solvent", "Density g/ml", "V/ul", "Volume of loading solvent/ul"]
QUANTILES = [f"{endpoint}_q{q}" for endpoint in ("V1", "V2") for q in (10, 50, 90)]
ARMS = dict(zip(("A1", "A2"), ARCHITECTURES))
TRAINING = {"maximum_epochs": 500, "patience": 80, "pretrained_lr": 1e-4,
            "new_lr": 3e-4, "weight_decay": 1e-5, "per_task_batch_cap": 682,
            "loss": "raw_quantile", "cpu_threads": 1, "gradient_diagnostic_interval": 10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value, *, immutable=False):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def source_path():
    decision = json.loads(SOURCE_DECISION.read_text())
    path = ROOT / decision["source_checkpoint_for_transfer"]
    if decision["source_checkpoint_sha256"] != SOURCE_SHA or sha(path) != SOURCE_SHA:
        raise RuntimeError("qualified source checkpoint digest mismatch")
    return path


def verify_schedule():
    if sha(PARENT / "split_manifest.csv") != SPLIT_SHA:
        raise RuntimeError("frozen outer split authority changed")
    frame = pd.read_csv(PARENT / "split_manifest.csv")
    if set(frame.outer_seed) != set(SEEDS):
        raise RuntimeError("outer seed schedule changed")
    if (STUDY / "split_manifest.csv").exists() and sha(STUDY / "split_manifest.csv") != SPLIT_SHA:
        raise RuntimeError("study outer identities differ from authority")
    return frame


def protocol_payload():
    parent = json.loads((PARENT / "protocol.json").read_text())
    return {
        "study": "COLUMN_CONDITIONED_MULTITASK_QGEOGNN", "arms": ARMS,
        "source_checkpoint": str(source_path().relative_to(ROOT)), "source_sha256": SOURCE_SHA,
        "split_manifest_sha256": SPLIT_SHA, "outer_seeds": list(SEEDS),
        "columns": ["25g", "40g"], "source_task": "4g", "primary_protocol": "compound",
        "secondary_protocol": "row", "training": TRAINING,
        "population_sha256": parent["filtered_canonical_sha256"],
        "features_sha256": {c: sha(PARENT / f"filtered_features_{c}.csv") for c in TASKS[1:]},
        "source_data_sha256": sha(SOURCE_DATA), "source_graph_cache_sha256": sha(SOURCE_GRAPH_CACHE),
        "target_graph_cache_sha256": {c: sha(ROOT / f"studies/transfer/cross_column/data_audit/graph_cache_{c}_only.pt") for c in TASKS[1:]},
        "source_split_sha256": sha(QUALIFIED / "splits/row_seed_42.csv"),
        "film": {"layers": [3, 4], "embedding_dim": 16, "placement": "after GIN output, before existing activation", "gamma_beta_initialization": "zero", "meaning": "categorical related task identity only"},
        "inner_cv": {"folds": 5, "group": "canonical_smiles", "grouping": "GLOBAL across 25g and 40g outer gradient_train", "selector": "equal-column arithmetic mean of inner-train-scaled combined NRMSE", "tie": "earliest epoch"},
        "batching": "per-task size=min(682,largest target training population); ceil(largest target/size) steps; independent epoch/task-seeded permutation cycles; equal task loss mean; mixed-task BN in a single forward",
        "bn_policy": "current; early frozen BN eval; late trainable BN mixed-task train",
        "source_supervision": "qualified row-seed-42 source_train only; source validation diagnostic only",
        "preprocessing": "unchanged qualified source preprocessing; inner metric scales from each task inner_train only",
        "tail_threshold": "each endpoint outer gradient_train 80th percentile; inner gate compares pooled OOF tail SSE/count within seed then seed means",
        "regimes": "outer gradient_train center 1/3 and 2/3 quantiles; descriptive only",
        "epoch_refit": "round-half-up median of five joint inner best epochs, per arm and seed",
        "gate": {"mean_fold_relative_nrmse_gain_min": .03, "seed_mean_wins_min": 3, "fold_wins_min": 13, "endpoint_mean_deterioration_max": .02, "tail_mean_deterioration_max": .02, "required_columns": ["25g", "40g"]},
        "outer_validation_or_test_used_for_fit_or_selection": False,
        "freeze": "all A1/A2 predictions for five seeds and both columns within protocol frozen before separate score action; compound first, row second",
        "interpretation": "DEVELOPMENTAL CONFIRMATION; target-compound holdout is not source-unseen OOD",
        "historical_baselines": ["P0 (ROW only; no converged COMPOUND artifact)", "paper_style_current_v2", "HIER_CW_SHARED_LAMBDA_CORRECTED"],
        "promotion": "both columns improve vs every available legitimate reference on compound with >=3/5 paired wins; no endpoint or tail mean >2% worse; no ROW mean NRMSE >2% worse; absent compound P0 explicitly reported",
        "no_extra_candidates": True,
    }


def prepare():
    verify_schedule()
    payload = protocol_payload()
    write_json(STUDY / "protocol.json", payload, immutable=True)
    if not (STUDY / "split_manifest.csv").exists():
        (STUDY / "split_manifest.csv").write_bytes((PARENT / "split_manifest.csv").read_bytes())
    return payload


def execution_provenance():
    paths = [Path(__file__).resolve(), ROOT / "src/qgeognn_al/transfer/column_conditioned_multitask.py",
             ROOT / "src/qgeognn_al/models/qgeognn_v2.py", ROOT / "src/qgeognn_al/transfer/adaptation.py",
             ROOT / "src/qgeognn_al/data.py", ROOT / "src/qgeognn_al/evaluation/point.py"]
    payload = {"files": {str(p.relative_to(ROOT)): sha(p) for p in paths},
               "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "cpu_threads": torch.get_num_threads(), "protocol_sha256": sha(STUDY / "protocol.json"),
               "frozen_prefix_cache": "qualified-source eval blocks 0-2; tested prediction/gradient/update equivalence"}
    write_json(STUDY / "EXECUTION_PROVENANCE.json", payload, immutable=True)
    return sha(STUDY / "EXECUTION_PROVENANCE.json")


def task_data(column, protocol, seed, specification):
    if column == "4g":
        feature_path, canonical_path = SOURCE_DATA, SOURCE_DATA
        split = pd.read_csv(QUALIFIED / "splits/row_seed_42.csv")
        roles = {r: split.loc[split.split.eq(s), "sample_id"].astype(str).tolist()
                 for r, s in (("gradient_train", "train"), ("source_validation", "validation"))}
        allowed = roles["gradient_train"] + roles["source_validation"]
    else:
        feature_path = PARENT / f"filtered_features_{column}.csv"
        canonical_path = PARENT / f"filtered_canonical_{column}.csv"
        if sha(canonical_path) != specification["population_sha256"][column] or sha(feature_path) != specification["features_sha256"][column]:
            raise RuntimeError("target data authority mismatch")
        schedule = verify_schedule()
        split = schedule.loc[schedule.column.eq(column) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed)]
        roles = {r: split.loc[split.role.eq(r), "sample_id"].astype(str).tolist() for r in ("gradient_train", "validation", "test")}
        allowed = roles["gradient_train"]
    feature = pd.read_csv(feature_path, usecols=FEATURES)
    feature.sample_id = feature.sample_id.astype(str)
    if not feature.sample_id.is_unique or any(not ids for ids in roles.values()):
        raise RuntimeError("missing or duplicate identities")
    if column != "4g" and set(feature.sample_id) != set().union(*map(set, roles.values())):
        raise RuntimeError("feature identities differ from frozen schedule")
    feature["V1_ml"], feature["V2_ml"] = 0., 0.
    lookup = {sid: i for i, sid in enumerate(feature.sample_id)}
    positions = {r: [lookup[sid] for sid in ids] for r, ids in roles.items()}
    labels = _read_authorized_truth(canonical_path, allowed)
    feature.loc[[lookup[sid] for sid in allowed], ["V1_ml", "V2_ml"]] = labels
    cache = dict(torch.load(SOURCE_GRAPH_CACHE, map_location="cpu", weights_only=False))
    if column != "4g":
        cache.update(torch.load(ROOT / f"studies/transfer/cross_column/data_audit/graph_cache_{column}_only.pt", map_location="cpu", weights_only=False))
    preprocessing = torch.load(source_path(), map_location="cpu", weights_only=False)["preprocessing"]
    atoms, angles = build_model_data(feature, cache, None, preprocessing["scaler"])
    cache_frozen_prefix(load_predictor_checkpoint(source_path()), atoms, angles)
    truth = feature[["V1_ml", "V2_ml"]].to_numpy(float)
    training_truth = truth[positions["gradient_train"]]
    return {"column": column, "frame": feature, "positions": positions, "roles": roles,
            "atoms": atoms, "angles": angles, "truth": truth,
            "scales": scales(training_truth), "tail": np.quantile(training_truth, .8, axis=0),
            "regimes": np.quantile(training_truth.mean(axis=1), [1/3, 2/3])}


def scales(truth):
    values = np.std(np.asarray(truth, dtype=float), axis=0, ddof=0)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("nonpositive training scales")
    return dict(zip(("V1", "V2"), map(float, values)))


def joint_data(protocol, seed):
    if protocol not in ("compound", "row") or seed not in SEEDS:
        raise ValueError("context outside preregistered schedule")
    specification = prepare()
    if protocol == "row":
        require_gate()
    return {c: task_data(c, protocol, seed, specification) for c in TASKS}


def fold_schedules(data):
    identities = [(c, i) for c in TASKS[1:] for i in data[c]["positions"]["gradient_train"]]
    groups = [str(data[c]["frame"].iloc[i].canonical_smiles) for c, i in identities]
    folds = global_target_group_folds(groups, [c for c, _ in identities])
    result = []
    for training, validation in folds:
        train = {"4g": data["4g"]["positions"]["gradient_train"]}
        valid = {"4g": data["4g"]["positions"]["source_validation"]}
        for c in TASKS[1:]:
            train[c] = [identities[i][1] for i in training if identities[i][0] == c]
            valid[c] = [identities[i][1] for i in validation if identities[i][0] == c]
            if not train[c] or not valid[c]:
                raise RuntimeError("a global inner fold is missing a target task")
        result.append((train, valid))
    return result


def authorize_fit(data, train, valid=None):
    for c in TASKS:
        allowed = set(data[c]["positions"]["gradient_train"])
        if not set(train[c]) <= allowed or len(train[c]) != len(set(train[c])):
            raise RuntimeError("unauthorized fitting identities")
        if valid is not None:
            val_allowed = set(data[c]["positions"]["source_validation"]) if c == "4g" else allowed
            if not set(valid[c]) <= val_allowed or set(train[c]) & set(valid[c]):
                raise RuntimeError("unauthorized selection identities")
    if valid is not None:
        train_groups = {str(data[c]["frame"].iloc[i].canonical_smiles) for c in TASKS[1:] for i in train[c]}
        valid_groups = {str(data[c]["frame"].iloc[i].canonical_smiles) for c in TASKS[1:] for i in valid[c]}
        if train_groups & valid_groups:
            raise RuntimeError("global target molecule leakage")


def make_batch(data, indices):
    pairs = [(c, int(i)) for c in TASKS for i in indices.get(c, [])]
    atoms = Batch.from_data_list([data[c]["atoms"][i] for c, i in pairs])
    angles = Batch.from_data_list([data[c]["angles"][i] for c, i in pairs])
    return atoms, angles, task_ids([c for c, _ in pairs])


def predict(model, data, column, positions):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(positions), 682):
            atoms, angles, tasks = make_batch(data, {column: positions[start:start + 682]})
            output.append(model(atoms, angles, tasks).numpy())
    prediction = np.vstack(output)
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite prediction")
    return prediction


def metric_rows(truth, prediction, metric_scales, thresholds, regimes):
    result = point_metrics(truth, prediction, metric_scales)
    point = prediction[:, [1, 4]]
    errors = point - truth
    ec, ew = errors.mean(axis=1), errors[:, 1] - errors[:, 0]
    result.update(Center_rmse=float(np.sqrt(np.mean(ec ** 2))), Width_rmse=float(np.sqrt(np.mean(ew ** 2))),
                  Center_error_variance=float(np.var(ec)), Width_error_variance=float(np.var(ew)),
                  Center_Width_error_covariance=float(np.mean((ec - ec.mean()) * (ew - ew.mean()))))
    tails = []
    for j, endpoint in enumerate(("V1", "V2")):
        mask = truth[:, j] >= thresholds[j]
        tail_sse = float(np.sum(errors[mask, j] ** 2))
        total_sse = float(np.sum(errors[:, j] ** 2))
        tails.append({"endpoint": endpoint, "threshold": float(thresholds[j]), "tail_count": int(mask.sum()),
                      "tail_sse": tail_sse, "total_sse": total_sse,
                      "tail_rmse": float(np.sqrt(tail_sse / mask.sum())) if mask.any() else np.nan,
                      "tail_sse_fraction": tail_sse / total_sse if total_sse else 0.})
    strata = []
    bins = np.searchsorted(regimes, truth.mean(axis=1), side="right")
    for k, name in enumerate(("low", "mid", "high")):
        mask = bins == k
        strata.append({"regime": name, "n": int(mask.sum()),
                       **{f"{t}_rmse": float(np.sqrt(np.mean(errors[mask, j] ** 2))) if mask.any() else np.nan for j, t in enumerate(("V1", "V2"))}})
    return result, tails, strata


def fit_model(data, train, valid, arm, seed, run, *, fixed_epochs=None):
    authorize_fit(data, train, valid)
    torch.manual_seed(int(seed))
    model = build_multitask_model(source_path(), ARMS[arm])
    optimizer = model.optimizer(pretrained_lr=TRAINING["pretrained_lr"], new_lr=TRAINING["new_lr"], weight_decay=TRAINING["weight_decay"])
    size = min(TRAINING["per_task_batch_cap"], max(len(train[c]) for c in TASKS[1:]))
    batcher = BalancedTaskBatcher(train, per_task_batch_size=size, seed=seed)
    metric_scales = {c: scales(data[c]["truth"][train[c]]) for c in TASKS}
    history, diagnostics, best, best_epoch, best_state = [], [], float("inf"), 0, None
    maximum = fixed_epochs or TRAINING["maximum_epochs"]
    start = time.monotonic()
    for epoch in range(1, maximum + 1):
        set_training_mode(model)
        losses_epoch = {c: [] for c in TASKS}
        for step, indices in enumerate(batcher.epoch_batches(epoch)):
            atoms, angles, tasks = make_batch(data, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(atoms, angles, tasks)
            losses = {c: target_loss_recipe(atoms.y[tasks.eq(j), 0], atoms.y[tasks.eq(j), 1], prediction[tasks.eq(j)], {}, recipe="raw_quantile") for j, c in enumerate(TASKS)}
            if epoch == 1 or epoch % TRAINING["gradient_diagnostic_interval"] == 0:
                diagnostics.append({"epoch": epoch, "step": step, **gradient_geometry(losses, model.shared_gradient_parameters())})
            objective = torch.stack(list(losses.values())).mean()
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite joint training loss")
            objective.backward()
            optimizer.step()
            for c in TASKS:
                losses_epoch[c].append(float(losses[c].detach()))
        row = {"epoch": epoch, **{f"train_loss_{c}": float(np.mean(v)) for c, v in losses_epoch.items()}}
        if valid is not None:
            for c in TASKS:
                values = point_metrics(data[c]["truth"][valid[c]], predict(model, data, c, valid[c]), metric_scales[c])
                row.update({f"{c}_{key}": value for key, value in values.items()})
            score = float(np.mean([row[f"{c}_combined_normalized_rmse"] for c in TASKS[1:]]))
            row["joint_validation_score"] = score
            if score < best:
                best, best_epoch, best_state = score, epoch, deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 50 == 0:
            print(json.dumps({"run": str(run.relative_to(STUDY)), "epoch": epoch, "best_epoch": best_epoch, "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)
        if valid is not None and epoch - best_epoch >= TRAINING["patience"]:
            break
    if valid is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(run / "gradients.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "arm": arm, "best_epoch": best_epoch,
                "source_sha256": SOURCE_SHA, "protocol_sha256": sha(STUDY / "protocol.json")}, run / "best.pt")
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "runtime_seconds": time.monotonic() - start,
                   "per_task_batch_size": size, "steps_per_epoch": batcher.steps_per_epoch,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "total_parameters": sum(p.numel() for p in model.parameters())}, metric_scales


def identity_records(data, train, valid, protocol, seed, fold):
    rows = []
    for role, indices in (("inner_train", train), ("inner_validation", valid)):
        for c in TASKS:
            for i in indices[c]:
                frame = data[c]["frame"].iloc[i]
                rows.append({"protocol": protocol, "outer_seed": seed, "inner_fold": fold, "column": c,
                             "sample_id": str(frame.sample_id), "canonical_smiles": str(frame.canonical_smiles), "role": role})
    return pd.DataFrame(rows)


def verify_completed(run):
    path = run / "complete.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["protocol_sha256"] != sha(STUDY / "protocol.json"):
        raise RuntimeError("completed fit protocol changed")
    if record.get("execution_provenance_sha256") != execution_provenance():
        raise RuntimeError("completed fit implementation or environment changed")
    for name, expected in record["files"].items():
        if sha(run / name) != expected:
            raise RuntimeError(f"completed fit artifact changed: {run / name}")
    return True


def inner_context(protocol, seed):
    data = joint_data(protocol, seed)
    provenance = execution_provenance()
    source_audit = json.loads((STUDY / "SOURCE_VERIFICATION.json").read_text())
    if source_audit["source_sha256"] != SOURCE_SHA or not source_audit["heads_exact"] or not source_audit["film_zero"]:
        raise RuntimeError("source initialization not verified")
    for fold, (train, valid) in enumerate(fold_schedules(data), 1):
        identities = identity_records(data, train, valid, protocol, seed, fold)
        for arm in ARMS:
            run = STUDY / f"runtime/inner/{protocol}/seed_{seed}/fold_{fold}/{arm}"
            if verify_completed(run):
                continue
            run.mkdir(parents=True, exist_ok=True)
            identities.to_csv(run / "identities.csv", index=False)
            model, fitted, metric_scales = fit_model(data, train, valid, arm, seed + fold * 1009, run)
            rows, tails, regimes, predictions = [], [], [], []
            for c in TASKS:
                pred = predict(model, data, c, valid[c])
                values, tail, strata = metric_rows(data[c]["truth"][valid[c]], pred, metric_scales[c], data[c]["tail"], data[c]["regimes"])
                meta = {"protocol": protocol, "outer_seed": seed, "inner_fold": fold, "arm": arm, "column": c}
                rows.append({**meta, **fitted, **values, "inner_train_count": len(train[c]), "inner_validation_count": len(valid[c]),
                             "inner_train_ids_sha256": digest(sorted(data[c]["frame"].iloc[train[c]].sample_id.tolist())),
                             "inner_validation_ids_sha256": digest(sorted(data[c]["frame"].iloc[valid[c]].sample_id.tolist())),
                             "scale_V1": metric_scales[c]["V1"], "scale_V2": metric_scales[c]["V2"]})
                tails.extend([{**meta, **item} for item in tail])
                regimes.extend([{**meta, **item} for item in strata])
                predictions.extend([{**meta, "sample_id": str(data[c]["frame"].iloc[i].sample_id), **dict(zip(QUANTILES, map(float, p)))} for i, p in zip(valid[c], pred)])
            for name, values in (("results.csv", rows), ("tails.csv", tails), ("regimes.csv", regimes), ("predictions.csv", predictions)):
                pd.DataFrame(values).to_csv(run / name, index=False)
            files = ["results.csv", "tails.csv", "regimes.csv", "predictions.csv", "identities.csv", "history.csv", "gradients.csv", "best.pt"]
            write_json(run / "complete.json", {"protocol_sha256": sha(STUDY / "protocol.json"), "execution_provenance_sha256": provenance, "files": {f: sha(run / f) for f in files}}, immutable=True)
            print(f"COMPLETED {protocol} seed={seed} fold={fold} {arm}", flush=True)


def gate_decision(results, tails):
    target = results.loc[results.column.isin(TASKS[1:])]
    expected = {(seed, fold, arm, c) for seed in SEEDS for fold in range(1, 6) for arm in ARMS for c in TASKS[1:]}
    actual = list(target[["outer_seed", "inner_fold", "arm", "column"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("gate requires exactly all 100 target fold/arm observations")
    decisions = []
    for c in TASKS[1:]:
        values = target.loc[target.column.eq(c)]
        left = values.loc[values.arm.eq("A1")].set_index(["outer_seed", "inner_fold"])
        right = values.loc[values.arm.eq("A2")].set_index(["outer_seed", "inner_fold"]).loc[left.index]
        gains = 1 - right.combined_normalized_rmse / left.combined_normalized_rmse
        seed_left = left.combined_normalized_rmse.groupby(level=0).mean()
        seed_right = right.combined_normalized_rmse.groupby(level=0).mean()
        endpoint_changes = {e: float(right[f"{e}_rmse"].mean() / left[f"{e}_rmse"].mean() - 1) for e in ("V1", "V2")}
        tail_changes = {}
        for endpoint in ("V1", "V2"):
            tail = tails.loc[tails.column.eq(c) & tails.endpoint.eq(endpoint)]
            grouped = tail.groupby(["outer_seed", "arm"])[["tail_sse", "tail_count"]].sum()
            if len(grouped) != 10 or (grouped.tail_count <= 0).any():
                raise RuntimeError("missing seed-level tail observations")
            rmse = np.sqrt(grouped.tail_sse / grouped.tail_count).unstack("arm")
            tail_changes[endpoint] = float(rmse.A2.mean() / rmse.A1.mean() - 1)
        conditions = {"mean_improvement": float(gains.mean()) >= .03,
                      "seed_wins": int((seed_right < seed_left).sum()) >= 3,
                      "fold_wins": int((gains > 0).sum()) >= 13,
                      "endpoint_guard": max(endpoint_changes.values()) <= .02,
                      "tail_guard": max(tail_changes.values()) <= .02}
        decisions.append({"column": c, "mean_relative_nrmse_gain": float(gains.mean()),
                          "seed_wins": int((seed_right < seed_left).sum()), "fold_wins": int((gains > 0).sum()),
                          "endpoint_changes": endpoint_changes, "tail_changes": tail_changes,
                          "conditions": conditions, "passed": all(conditions.values())})
    passed = all(row["passed"] for row in decisions)
    return {"status": "REPRESENTATION_GATE_PASSED" if passed else "NO_REPLICATED_COLUMN_FILM_GAIN",
            "passed": passed, "columns": decisions, "outer_scoring_authorized": passed,
            "outer_truth_used_for_fit_or_selection": False, "additional_candidates_authorized": False}


def aggregate(protocol="compound"):
    collections = {name: [] for name in ("results", "tails", "regimes", "gradients", "identities")}
    completions = {}
    for seed in SEEDS:
        for fold in range(1, 6):
            for arm in ARMS:
                run = STUDY / f"runtime/inner/{protocol}/seed_{seed}/fold_{fold}/{arm}"
                if not verify_completed(run):
                    raise RuntimeError(f"inner context incomplete: {run}")
                completions[str(run.relative_to(STUDY))] = sha(run / "complete.json")
                for name in collections:
                    frame = pd.read_csv(run / f"{name}.csv")
                    if name == "gradients":
                        frame = frame.assign(protocol=protocol, outer_seed=seed, inner_fold=fold, arm=arm)
                    collections[name].append(frame)
    frames = {key: pd.concat(values, ignore_index=True) for key, values in collections.items()}
    prefix = "" if protocol == "compound" else "ROW_"
    for name, file in (("results", "INNER_RESULTS.csv"), ("tails", "TAIL_METRICS.csv"), ("regimes", "RETENTION_REGIMES.csv"), ("gradients", "GRADIENT_DIAGNOSTICS.csv"), ("identities", "INNER_IDENTITIES.csv")):
        frames[name].to_csv(STUDY / (prefix + file), index=False)
    frames["results"].groupby(["column", "arm"]).mean(numeric_only=True).reset_index().to_csv(STUDY / (prefix + "INNER_SUMMARY.csv"), index=False)
    manifest = frames["results"][["protocol", "outer_seed", "inner_fold", "arm", "column", "inner_train_count", "inner_validation_count", "inner_train_ids_sha256", "inner_validation_ids_sha256", "best_epoch", "epochs_run"]]
    manifest.to_csv(STUDY / (prefix + "run_manifest.csv"), index=False)
    decision = gate_decision(frames["results"], frames["tails"])
    decision.update(protocol_sha256=sha(STUDY / "protocol.json"), inner_results_sha256=sha(STUDY / (prefix + "INNER_RESULTS.csv")), tail_metrics_sha256=sha(STUDY / (prefix + "TAIL_METRICS.csv")), completion_hashes=completions)
    write_json(STUDY / (prefix + "INNER_DECISION.json"), decision, immutable=True)
    return decision


def require_gate():
    path = STUDY / "INNER_DECISION.json"
    if not path.exists():
        raise RuntimeError("outer action blocked: no COMPOUND inner gate")
    decision = json.loads(path.read_text())
    if not decision["passed"] or decision["protocol_sha256"] != sha(STUDY / "protocol.json"):
        raise RuntimeError("outer action blocked: COMPOUND inner gate failed or changed")
    if sha(STUDY / "INNER_RESULTS.csv") != decision["inner_results_sha256"] or sha(STUDY / "TAIL_METRICS.csv") != decision["tail_metrics_sha256"]:
        raise RuntimeError("gate evidence changed")
    recomputed = gate_decision(pd.read_csv(STUDY / "INNER_RESULTS.csv"), pd.read_csv(STUDY / "TAIL_METRICS.csv"))
    if not recomputed["passed"]:
        raise RuntimeError("gate cannot be reproduced")
    for relative, expected in decision["completion_hashes"].items():
        run = STUDY / relative
        if sha(run / "complete.json") != expected or not verify_completed(run):
            raise RuntimeError("gate source completion changed")
    return decision


def final_context(protocol, seed):
    require_gate()
    if protocol == "row" and not (STUDY / "COMPOUND_SCORE_MANIFEST.json").exists():
        raise RuntimeError("COMPOUND must be scored before ROW")
    data = joint_data(protocol, seed)
    prefix = "" if protocol == "compound" else "ROW_"
    selections = pd.read_csv(STUDY / (prefix + "INNER_RESULTS.csv"))
    train = {c: data[c]["positions"]["gradient_train"] for c in TASKS}
    for arm in ARMS:
        run = STUDY / f"runtime/final/{protocol}/seed_{seed}/{arm}"
        if verify_completed(run):
            continue
        chosen = selections.loc[selections.outer_seed.eq(seed) & selections.arm.eq(arm) & selections.column.eq("25g"), "best_epoch"]
        if len(chosen) != 5:
            raise RuntimeError("missing joint epoch selections")
        epochs = int(np.floor(chosen.median() + .5))
        model, fitted, _ = fit_model(data, train, None, arm, seed, run, fixed_epochs=epochs)
        output = []
        for c in TASKS[1:]:
            for role in ("validation", "test"):
                positions = data[c]["positions"][role]
                prediction = predict(model, data, c, positions)
                output.extend([{"column": c, "role": role, "sample_id": data[c]["frame"].iloc[i].sample_id,
                                **dict(zip(QUANTILES, map(float, p)))} for i, p in zip(positions, prediction)])
        pd.DataFrame(output).to_csv(run / "predictions_blind.csv", index=False)
        write_json(run / "fit.json", {**fitted, "outer_truth_read": False, "selection_sha256": sha(STUDY / (prefix + "INNER_RESULTS.csv"))})
        write_json(run / "complete.json", {"protocol_sha256": sha(STUDY / "protocol.json"), "execution_provenance_sha256": execution_provenance(), "files": {f: sha(run / f) for f in ("predictions_blind.csv", "best.pt", "fit.json", "history.csv", "gradients.csv")}}, immutable=True)


def freeze(protocol):
    require_gate()
    files = {}
    for seed in SEEDS:
        for arm in ARMS:
            run = STUDY / f"runtime/final/{protocol}/seed_{seed}/{arm}"
            if not verify_completed(run):
                raise RuntimeError("all predictions must exist before global freeze")
            files[str((run / "complete.json").relative_to(STUDY))] = sha(run / "complete.json")
            files[str((run / "predictions_blind.csv").relative_to(STUDY))] = sha(run / "predictions_blind.csv")
    path = STUDY / ("PREDICTION_FREEZE_MANIFEST.json" if protocol == "compound" else "ROW_PREDICTION_FREEZE_MANIFEST.json")
    write_json(path, {"status": "FROZEN_BEFORE_OUTER_TRUTH", "protocol": protocol, "inner_decision_sha256": sha(STUDY / "INNER_DECISION.json"), "files": files}, immutable=True)
    return path


def verify_freeze(protocol):
    require_gate()
    path = STUDY / ("PREDICTION_FREEZE_MANIFEST.json" if protocol == "compound" else "ROW_PREDICTION_FREEZE_MANIFEST.json")
    if not path.exists():
        raise RuntimeError("outer truth blocked until all predictions are frozen")
    manifest = json.loads(path.read_text())
    if manifest["status"] != "FROZEN_BEFORE_OUTER_TRUTH" or manifest["inner_decision_sha256"] != sha(STUDY / "INNER_DECISION.json"):
        raise RuntimeError("invalid freeze provenance")
    if len(manifest["files"]) != 20:
        raise RuntimeError("incomplete freeze schedule")
    for relative, expected in manifest["files"].items():
        if sha(STUDY / relative) != expected:
            raise RuntimeError("frozen prediction artifact changed")
        if relative.endswith("complete.json") and not verify_completed((STUDY / relative).parent):
            raise RuntimeError("frozen fit is incomplete")
    return path


def score(protocol):
    freeze_path = verify_freeze(protocol)
    from scripts.studies.run_conditioned_source_readout import _reference_prediction, _align_historical_prediction_ids
    rows, tails, paired = [], [], []
    schedule = verify_schedule()
    for seed in SEEDS:
        for c in TASKS[1:]:
            context = schedule.loc[schedule.column.eq(c) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed)]
            ids = context.loc[context.role.eq("test"), "sample_id"].astype(str).tolist()
            train_ids = context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
            canonical = PARENT / f"filtered_canonical_{c}.csv"
            train_truth = _read_authorized_truth(canonical, train_ids)
            truth = _read_authorized_truth(canonical, ids)
            predictions = {}
            for arm in ARMS:
                path = STUDY / f"runtime/final/{protocol}/seed_{seed}/{arm}/predictions_blind.csv"
                frame = pd.read_csv(path)
                frame = frame.loc[frame.column.eq(c) & frame.role.eq("test")]
                predictions[arm] = _align_historical_prediction_ids(frame, ids, path=path, reference=arm)[QUANTILES].to_numpy(float)
            for reference in ("paper_style_current_v2", "HIER_CW_SHARED_LAMBDA_CORRECTED"):
                predictions[reference] = _reference_prediction(c, protocol, seed, reference, ids)
            if protocol == "row":
                p0_study = ROOT / "studies/transfer/traditional_transfer_converged_baseline_v1"
                frozen = json.loads((p0_study / "PREDICTION_FREEZE_MANIFEST.json").read_text())
                candidates = list(p0_study.glob(f"runtime/**/{c}/seed_{seed}/P0/test_predictions_blind.csv.gz"))
                if len(candidates) != 1:
                    raise RuntimeError("cannot resolve frozen ROW P0 prediction")
                path = candidates[0]
                complete = path.parent.parent / "context_fit_complete.json"
                if frozen["context_fit_complete_sha256"].get(str(complete.relative_to(p0_study))) != sha(complete):
                    raise RuntimeError("P0 context digest mismatch")
                complete_record = json.loads(complete.read_text())
                if sha(path) not in json.dumps(complete_record):
                    raise RuntimeError("P0 prediction digest absent from context freeze")
                predictions["P0"] = _align_historical_prediction_ids(pd.read_csv(path), ids, path=path, reference="P0")[QUANTILES].to_numpy(float)
            measured = {}
            for method, prediction in predictions.items():
                values, tail, _ = metric_rows(truth, prediction, scales(train_truth), np.quantile(train_truth, .8, axis=0), np.quantile(train_truth.mean(axis=1), [1/3, 2/3]))
                meta = {"protocol": protocol, "outer_seed": seed, "column": c, "method": method, "classification": "DEVELOPMENTAL CONFIRMATION"}
                measured[method] = values
                rows.append({**meta, **values})
                tails.extend([{**meta, **v} for v in tail])
            for reference in predictions:
                if reference == "A2":
                    continue
                paired.append({"protocol": protocol, "outer_seed": seed, "column": c, "candidate": "A2", "reference": reference,
                               **{key + "_relative_gain": 1 - measured["A2"][key] / measured[reference][key] for key in ("combined_normalized_rmse", "V1_rmse", "V2_rmse", "Center_rmse", "Width_rmse")}})
    prefix = "" if protocol == "compound" else "ROW_"
    for filename, values in (("OUTER_RESULTS.csv", rows), ("OUTER_TAIL_METRICS.csv", tails), ("PAIRED_COMPARISONS.csv", paired)):
        pd.DataFrame(values).to_csv(STUDY / (prefix + filename), index=False)
    write_json(STUDY / (protocol.upper() + "_SCORE_MANIFEST.json"), {"prediction_freeze_sha256": sha(freeze_path), "classification": "DEVELOPMENTAL CONFIRMATION", "files": {prefix + f: sha(STUDY / (prefix + f)) for f in ("OUTER_RESULTS.csv", "OUTER_TAIL_METRICS.csv", "PAIRED_COMPARISONS.csv")}}, immutable=True)


def source_check():
    prepare()
    specification = protocol_payload()
    source = task_data("4g", "compound", SEEDS[0], specification)
    data = {"4g": source}
    original = load_predictor_checkpoint(source_path())
    a1 = build_multitask_model(source_path(), ARMS["A1"]).eval()
    torch.manual_seed(42)
    a2 = build_multitask_model(source_path(), ARMS["A2"]).eval()
    indices = list(range(len(source["frame"])))
    actual = predict(a1, data, "4g", indices)
    reference = pd.read_csv(QUALIFIED / "results/row/seed_42/predictions.csv.gz").set_index("sample_id").loc[source["frame"].sample_id, QUANTILES].to_numpy(float)
    maximum = float(np.max(np.abs(actual - reference)))
    atoms, angles, tasks = make_batch(data, {"4g": indices[:24]})
    with torch.no_grad():
        source_difference = float(torch.max(torch.abs(a1(atoms, angles, tasks) - original(atoms, angles))))
        difference = max(float(torch.max(torch.abs(a2(atoms, angles, torch.full_like(tasks, j)) - a1(atoms, angles, torch.full_like(tasks, j))))) for j in range(3))
    if maximum > 1e-4 or source_difference > 1e-6 or difference > 1e-6:
        raise RuntimeError("source or epoch-zero equivalence failed")
    write_json(STUDY / "SOURCE_VERIFICATION.json", {"source_sha256": SOURCE_SHA, "rows": len(actual), "frozen_prediction_max_abs_difference": maximum,
               "active_source_vs_A1_max_abs_difference": source_difference, "A1_A2_max_abs_difference": difference,
               "heads_exact": head_copies_are_exact(a1) and head_copies_are_exact(a2), "film_zero": film_is_zero_initialized(a2), "all_finite": bool(np.isfinite(actual).all())}, immutable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "source-check", "inner", "aggregate", "final", "freeze", "score", "execute"), required=True)
    parser.add_argument("--protocol", choices=("compound", "row"), default="compound")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(TRAINING["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    if args.action == "prepare": prepare()
    elif args.action == "source-check": source_check()
    elif args.action == "inner":
        if args.seed is None: parser.error("inner requires --seed")
        inner_context(args.protocol, args.seed)
    elif args.action == "aggregate": print(json.dumps(aggregate(args.protocol), indent=2))
    elif args.action == "final":
        if args.seed is None: parser.error("final requires --seed")
        final_context(args.protocol, args.seed)
    elif args.action == "freeze": freeze(args.protocol)
    elif args.action == "score": score(args.protocol)
    elif args.action == "execute":
        source_check()
        for seed in SEEDS: inner_context("compound", seed)
        decision = aggregate("compound")
        if not decision["passed"]:
            print(decision["status"], flush=True)
            return
        for protocol in ("compound", "row"):
            if protocol == "row":
                for seed in SEEDS: inner_context(protocol, seed)
                aggregate(protocol)
            for seed in SEEDS: final_context(protocol, seed)
            freeze(protocol)
            print(f"Frozen {protocol}; run separate --action score --protocol {protocol}", flush=True)
            return


if __name__ == "__main__":
    main()

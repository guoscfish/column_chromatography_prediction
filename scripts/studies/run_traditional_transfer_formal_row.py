#!/usr/bin/env python3
"""Formal 5-seed ROW P0--P3 study with a hard prediction-freeze boundary.

``--fit`` never reads test endpoint cells.  ``--freeze`` hashes all forty
blind-prediction exports.  Only ``--score`` may read test truth, and it first
requires that complete global freeze manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.transfer import adaptation as a
from src.qgeognn_al.evaluation.point import point_metrics
from scripts.studies.run_filtered_full_data_benchmark import _read_authorized_truth

STUDY = ROOT / "studies/transfer/traditional_transfer_improvement/formal_row_5seed"
FROZEN = ROOT / "studies/transfer/filtered_full_data_benchmark"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
COLUMNS = ("25g", "40g")
METHODS = {"P0": ("current", "adam"), "P1": ("current", "adam"),
           "P2": ("source_stats", "adam"), "P3": ("source_stats", "lbfgs")}
FEATURES = ["sample_id", "canonical_smiles", "PE/EA", "loading solvent", "Density g/ml",
            "V/ul", "Volume of loading solvent/ul"]
ADAM = {"maximum_epochs": 150, "patience": 40, "batch_size": 2048,
        "learning_rate": 1e-4, "weight_decay": 1e-5}
LBFGS = {"maximum_epochs": 30, "patience": 12, "batch_size": 2048,
         "learning_rate": 0.1, "weight_decay": 0.0, "lbfgs_max_iter": 5,
         "lbfgs_history_size": 10, "lbfgs_line_search_fn": "strong_wolfe"}

def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def _protocol() -> dict:
    frozen = json.loads((FROZEN / "protocol.json").read_text())
    return {"study": "FORMAL_TRADITIONAL_PARAMETER_TRANSFER_5SEED_ROW", "status": "FROZEN_PRE_FIT",
      "columns": list(COLUMNS), "protocol": "row", "outer_seeds": frozen["outer_seeds"],
      "methods": {name: list(recipe) for name, recipe in METHODS.items()}, "source_checkpoint_seed": 42, "source_checkpoint_sha256": sha(SOURCE),
      "filtered_protocol_sha256": sha(FROZEN / "protocol.json"),
      "scope": "backbone.convs.4 + condition_branch + head", "trainable_parameters": 36387,
      "selection": "validation combined normalized RMSE; target scales from gradient_train only",
      "adam_budget": ADAM, "lbfgs_budget": LBFGS,
      "test_truth_rule": "forbidden during fit and freeze; one-shot scoring only after complete prediction freeze"}

def ensure_protocol() -> dict:
    payload = _protocol(); path = STUDY / "FORMAL_ROW_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("FORMAL_ROW_PROTOCOL.json differs from frozen requested protocol")
    write_json(path, payload)
    write_json(STUDY / "FORMAL_ROW_PROTOCOL.sha256.json", {"sha256": sha(path)})
    return payload

def _predict(model, atoms, angles, indices):
    model.eval(); output = []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*a.loader_pair(atoms, angles, indices, 2048)):
            value = model(atom_batch, angle_batch)
            if isinstance(value, (tuple, list)): value = value[0]
            output.append(value.detach().cpu().numpy())
    return np.vstack(output)

def prepare_context(column: str, seed: int):
    frozen = json.loads((FROZEN / "protocol.json").read_text())
    if seed not in frozen["outer_seeds"]: raise RuntimeError("seed absent from frozen schedule")
    feature = pd.read_csv(FROZEN / f"filtered_features_{column}.csv", usecols=FEATURES)
    canonical = FROZEN / f"filtered_canonical_{column}.csv"
    if sha(canonical) != frozen["filtered_canonical_sha256"][column]: raise RuntimeError("filtered population hash mismatch")
    schedule = pd.read_csv(FROZEN / "split_manifest.csv")
    context = schedule.loc[(schedule.column == column) & (schedule.protocol == "row") & (schedule.outer_seed == seed),
                           ["column", "protocol", "outer_seed", "sample_id", "role"]].reset_index(drop=True)
    expected = pd.read_csv(canonical, usecols=["sample_id"]).sample_id.astype(str).tolist()
    feature.sample_id = feature.sample_id.astype(str); context.sample_id = context.sample_id.astype(str)
    if not (feature.sample_id.is_unique and context.sample_id.is_unique and set(feature.sample_id) == set(expected) == set(context.sample_id)):
        raise RuntimeError("population identity mismatch")
    roles = {r: context.loc[context.role == r, "sample_id"].tolist() for r in ("gradient_train", "validation", "test")}
    if set(context.role) != set(roles) or any(set(roles[x]) & set(roles[y]) for x in roles for y in roles if x < y):
        raise RuntimeError("split roles are incomplete or overlap")
    cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False)); cache.update(torch.load(ROOT / f"studies/transfer/cross_column/data_audit/graph_cache_{column}_only.pt", weights_only=False))
    actual = set(feature.loc[feature.canonical_smiles.isin(cache), "sample_id"])
    if actual != set(expected): raise RuntimeError("graph coverage is not 100%")
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_pre = source_payload["preprocessing"]
    # This is the sole target-truth read in the fit phase, restricted to train+validation IDs.
    revealed = roles["gradient_train"] + roles["validation"]
    labels = _read_authorized_truth(canonical, revealed)
    frame = feature.copy(); frame["V1_ml"] = 0.0; frame["V2_ml"] = 0.0
    lookup = {s: i for i, s in enumerate(frame.sample_id)}
    for sample_id, target in zip(revealed, labels): frame.loc[lookup[sample_id], ["V1_ml", "V2_ml"]] = target
    atoms, angles = build_model_data(frame, cache, None, source_pre["scaler"])
    positions = {r: [lookup[s] for s in ids] for r, ids in roles.items()}
    preprocessing = {"scaler": source_pre["scaler"], "source_preprocessing": source_pre,
                     "target_scales": a.fit_target_scales(atoms, positions["gradient_train"]),
                     "fit_role": "gradient_train", "validation_rows_used": 0, "test_rows_used": 0}
    audit = {"column": column, "seed": seed, "population_sha256": sha(canonical),
             "split_manifest_sha256": sha(FROZEN / "split_manifest.csv"), "source_checkpoint_sha256": sha(SOURCE),
             "source_preprocessing_hash": digest(source_pre), "scaler_hash": digest(source_pre["scaler"]),
             "roles": {r: {"count": len(v), "ids_hash": digest(sorted(v))} for r, v in roles.items()},
             "target_scales": preprocessing["target_scales"], "graph_coverage": {"expected": len(expected), "actual": len(actual)},
             "test_truth_used_for_fit": False}
    return roles, positions, atoms, angles, preprocessing, audit

def fit_context(column: str, seed: int) -> None:
    protocol = ensure_protocol(); out = STUDY / "runtime" / column / f"seed_{seed}"
    frozen = out / "context_freeze.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text())
        for method in METHODS:
            if not (out / method / "test_predictions_blind.csv.gz").exists(): raise RuntimeError("partial frozen context")
        return
    roles, positions, atoms, angles, pre, audit = prepare_context(column, seed)
    out.mkdir(parents=True, exist_ok=True); histories=[]; metrics=[]; drifts=[]; inventory=[]; started=time.time()
    for method, (bn, optimizer) in METHODS.items():
        run = out / method; model = load_predictor_checkpoint(SOURCE); a.configure_trainable(model, "historical_shallow")
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}; bn_before = a.snapshot_bn_buffers(model)
        affine = {n:p.detach().clone() for n,p in model.named_parameters() if any(n.startswith(f"{m}.") for m,x in model.named_modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm))}
        config = dict(LBFGS if optimizer == "lbfgs" else ADAM, bn_policy=bn, optimizer=optimizer)
        if method == "P0": stages={"B": a.train_target_adaptation(model, atoms, angles, positions["gradient_train"], positions["validation"], pre, mode="historical_shallow", seed=seed, config=config, checkpoint_path=run / "best.pt")}
        else:
            staged=a.train_staged_target_adaptation(model, atoms, angles, positions["gradient_train"], positions["validation"], pre, stage_b_mode="historical_shallow", seed=seed, stage_a_config=config, stage_b_config=config, checkpoint_dir=run)
            stages={"A": staged.stage_a, "B": staged.stage_b}
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable != 36387: raise RuntimeError("historical shallow inventory changed")
        valid_pred = _predict(model, atoms, angles, positions["validation"]); test_pred = _predict(model, atoms, angles, positions["test"])
        if not (np.isfinite(valid_pred).all() and np.isfinite(test_pred).all()): raise RuntimeError("nonfinite prediction")
        np.save(run / "validation_predictions.npy", valid_pred)
        pd.DataFrame({"sample_id": roles["test"], "V1_q10":test_pred[:,0], "V1_q50":test_pred[:,1], "V1_q90":test_pred[:,2], "V2_q10":test_pred[:,3], "V2_q50":test_pred[:,4], "V2_q90":test_pred[:,5]}).to_csv(run / "test_predictions_blind.csv.gz", index=False, compression={"method":"gzip","mtime":0})
        truth = np.vstack([atoms[i].y.numpy().reshape(-1)[:2] for i in positions["validation"]]); vmetrics=point_metrics(truth, valid_pred, pre["target_scales"])
        for stage, fit in stages.items():
            histories.extend({"method":method,"stage":stage,**x} for x in fit.history); metrics.append({"method":method,"stage":stage,**fit.as_dict() | {"history": None}})
        drifts.append({"method":method,"parameter_drift":a.parameter_drift(model, initial, trainable_only=False),"bn_buffer_drift":a.bn_buffer_drift(model,bn_before),"bn_affine_drift":a.parameter_drift(model,affine,trainable_only=False),"quantile_crossing_count":int(((test_pred[:,0]>test_pred[:,1])|(test_pred[:,1]>test_pred[:,2])|(test_pred[:,3]>test_pred[:,4])|(test_pred[:,4]>test_pred[:,5])).sum()),"runtime_seconds":time.time()-started})
        inventory.extend({"method":method,"parameter":n,"numel":p.numel(),"trainable":p.requires_grad} for n,p in model.named_parameters())
        write_json(run / "fit_audit.json", {"method":method,"config":config,"validation_metrics":vmetrics,"test_truth_used_for_fit_or_selection":False,"prediction_sha256":sha(run / "test_predictions_blind.csv.gz")})
    pd.DataFrame(histories).to_csv(out / "training_history.csv",index=False); pd.DataFrame(metrics).to_csv(out / "stage_metrics.csv",index=False); pd.DataFrame(drifts).to_csv(out / "drift_metrics.csv",index=False); pd.DataFrame(inventory).to_csv(out / "parameter_inventory.csv",index=False)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()}
    write_json(frozen,{"protocol_sha256":sha(STUDY / "FORMAL_ROW_PROTOCOL.json"),"audit":audit,"files":files,"methods":list(METHODS),"test_truth_used_for_fit_or_selection":False})

def freeze() -> None:
    ensure_protocol(); records={}
    seeds=_protocol()["outer_seeds"]
    for column in COLUMNS:
        for seed in seeds:
            path=STUDY/"runtime"/column/f"seed_{seed}"/"context_freeze.json"
            if not path.exists(): raise RuntimeError(f"missing fit freeze: {path}")
            value=json.loads(path.read_text())
            if value["test_truth_used_for_fit_or_selection"] or value["methods"] != list(METHODS): raise RuntimeError("invalid context freeze")
            for relative, expected in value["files"].items():
                if sha(path.parent / relative) != expected: raise RuntimeError("frozen artifact changed")
            records[str(path.relative_to(STUDY))]=sha(path)
    write_json(STUDY/"PREDICTION_FREEZE_MANIFEST.json", {"contexts":10,"method_context_fits":40,"protocol_sha256":sha(STUDY/"FORMAL_ROW_PROTOCOL.json"),"context_freeze_sha256":records,"test_truth_evaluation_started":False})

def score() -> None:
    manifest=STUDY/"PREDICTION_FREEZE_MANIFEST.json"
    if not manifest.exists(): raise RuntimeError("PREDICTION_FREEZE_MANIFEST.json required before test scoring")
    frozen=json.loads(manifest.read_text())
    if frozen["contexts"] != 10 or frozen["protocol_sha256"] != sha(STUDY/"FORMAL_ROW_PROTOCOL.json"): raise RuntimeError("incomplete or changed prediction freeze")
    for relative, expected in frozen["context_freeze_sha256"].items():
        if sha(STUDY / relative) != expected: raise RuntimeError("context freeze changed after global freeze")
    rows=[]
    for column in COLUMNS:
        canonical=FROZEN/f"filtered_canonical_{column}.csv"
        for seed in _protocol()["outer_seeds"]:
            context=pd.read_csv(FROZEN/"split_manifest.csv"); ids=context.loc[(context.column==column)&(context.protocol=="row")&(context.outer_seed==seed)&(context.role=="test"),"sample_id"].astype(str).tolist()
            truth=_read_authorized_truth(canonical,ids); audit=json.loads((STUDY/"runtime"/column/f"seed_{seed}"/"context_freeze.json").read_text()); scales=audit["audit"]["target_scales"]
            for method in METHODS:
                prediction=pd.read_csv(STUDY/"runtime"/column/f"seed_{seed}"/method/"test_predictions_blind.csv.gz")
                if prediction.sample_id.astype(str).tolist()!=ids: raise RuntimeError("test prediction ID order mismatch")
                point=prediction[["V1_q50","V2_q50"]].to_numpy(float); metric=point_metrics(truth,np.column_stack([prediction[["V1_q10","V1_q50","V1_q90"]],prediction[["V2_q10","V2_q50","V2_q90"]]]),scales)
                rows.append({"column":column,"seed":seed,"method":method,"n_test":len(ids),**metric,"quantile_crossing_count":int(((prediction.V1_q10>prediction.V1_q50)|(prediction.V1_q50>prediction.V1_q90)|(prediction.V2_q10>prediction.V2_q50)|(prediction.V2_q50>prediction.V2_q90)).sum())})
    metrics=pd.DataFrame(rows); metrics.to_csv(STUDY/"test_metrics.csv",index=False)
    numeric=["V1_rmse","V1_mae","V1_r2","V2_rmse","V2_mae","V2_r2","combined_normalized_rmse"]
    summary=metrics.groupby(["column","method"])[numeric].agg(["mean","std","median","min","max"]); summary.columns=["_".join(x) for x in summary.columns]; summary.reset_index().to_csv(STUDY/"summary.csv",index=False)
    paired=[]
    for column, group in metrics.groupby("column"):
        for candidate, reference in (("P1","P0"),("P2","P1"),("P2","P0"),("P3","P2"),("P3","P0")):
            left=group[group.method==candidate].set_index("seed"); right=group[group.method==reference].set_index("seed")
            for metric in numeric:
                delta=left[metric]-right[metric]; paired.append({"column":column,"candidate":candidate,"reference":reference,"metric":metric,"mean_paired_delta":delta.mean(),"median_paired_delta":delta.median(),"std":delta.std(ddof=1),"seed_wins":int((delta<0).sum()),"n":len(delta)})
    pd.DataFrame(paired).to_csv(STUDY/"paired_comparisons.csv",index=False)
    write_json(STUDY/"TEST_SCORE_MANIFEST.json", {"prediction_freeze_sha256":sha(manifest),"test_metrics_sha256":sha(STUDY/"test_metrics.csv"),"status":"ONE_SHOT_TEST_SCORED"})

def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True); group.add_argument("--fit",action="store_true"); group.add_argument("--freeze",action="store_true"); group.add_argument("--score",action="store_true"); args=parser.parse_args()
    if args.fit:
        torch.set_num_threads(1)
        for column in COLUMNS:
            for seed in _protocol()["outer_seeds"]: fit_context(column,int(seed))
    elif args.freeze: freeze()
    else: score()
if __name__ == "__main__": main()

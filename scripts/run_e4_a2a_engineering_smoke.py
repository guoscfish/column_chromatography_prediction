#!/usr/bin/env python3
"""Bounded E4-A2a engineering smoke (seed 42 only; never formal training)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.al_engine import (QGeoGNNActiveLearningEngine, TrainConfig, initialize_round_state,
                               load_round_state, random_query, save_round_state)
from scripts.run_e4_active_transfer import (SOURCES, TARGET, SCALER, SOURCE_SCALES, STRATEGIES,
    acquire, ensemble_scores, fit_members, primary_quantile_width, representation_from_primary,
    score_round, validate_dry_run)
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e4_a2a_low_budget import OUT as PARTITIONS
from scripts.run_e4_a2a_low_budget import sha256 as file_sha
from scripts.run_e4_a2a_low_budget import EXPECTED
from scripts.run_e4_a2a_low_budget import SEEDS as ALL_SEEDS
from scripts.run_e4_a2a_low_budget import make_partition
from scripts.run_e0_8g_controls import load_graph_cache

OUT = ROOT / "experiments/e4_a2a_engineering_smoke"

def partition_context(seed: int):
    if seed not in ALL_SEEDS: raise ValueError("A2a smoke permits frozen seeds only")
    path = PARTITIONS / f"e4_a2a_protocol_a_seed_{seed}.csv"
    if not path.exists(): raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"sample_id": str})
    roles = {r: frame.loc[frame.role == r, "sample_id"].astype(str).tolist()
             for r in ("l0_train", "l0_validation", "u0", "test")}
    counts = {"old_l0_train_count": 42, "new_l0_train_count": len(roles["l0_train"]),
              "validation_count": len(roles["l0_validation"]), "new_u0_count": len(roles["u0"]),
              "test_count": len(roles["test"]), "union_rows": len(frame)}
    if len(frame) != 574 or frame.sample_id.nunique() != 574 or any(len(set(roles[a]) & set(roles[b])) for i,a in enumerate(roles) for b in list(roles)[i+1:]):
        raise ValueError("A2a partition contract failed")
    if any(counts[k] != v for k,v in EXPECTED.items()): raise ValueError(counts)
    return frame, roles, file_sha(path)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--protocol", default="A"); ap.add_argument("--outer-seed", type=int, default=42)
    ap.add_argument("--engineering-smoke", action="store_true"); args = ap.parse_args()
    if args.protocol != "A" or args.outer_seed != 42 or not args.engineering_smoke:
        raise SystemExit("bounded A2a smoke requires --protocol A --outer-seed 42 --engineering-smoke")
    OUT.mkdir(parents=True, exist_ok=True)
    frame, roles, split_hash = partition_context(42)
    audit = json.loads((PARTITIONS.parent / "partition_audit.json").read_text())
    (OUT / "partition_audit.json").write_text(json.dumps(audit, indent=2))
    pd.read_csv(PARTITIONS.parent / "partition_manifest.csv").to_csv(OUT / "partition_manifest.csv", index=False)
    data = pd.read_csv(TARGET); engine = QGeoGNNActiveLearningEngine(data, load_graph_cache(), json.loads(SCALER.read_text()), SOURCES[42], device=torch.device("cpu"))
    cfg = TrainConfig(epochs=50, patience=20); cfg.validate_frozen_predictor()
    labeled = roles["l0_train"] + roles["l0_validation"]
    ckpt0, fit0 = fit_members(engine, labeled, roles["l0_validation"], cfg, 42, 0, OUT / "round0")
    fit0_by_member = {r["member_seed"]: r for r in fit0}
    pd.DataFrame(fit0).to_csv(OUT / "round0_fit_audit.csv", index=False)
    freeze = []
    for r in fit0:
        freeze.append({"outer_seed":42, "round":r["round"], "member_seed":r["member_seed"], "init_source_sha256":r["init_source_sha256"], "expected_source_sha256":sha256_file(SOURCES[r["member_seed"]]), "source_reset_pass":r["init_source_sha256"]==sha256_file(SOURCES[r["member_seed"]]), "frozen_parameters_sha256_before":r["frozen_parameters_sha256_before"], "frozen_parameters_sha256_after":r["frozen_parameters_sha256_after"], "frozen_unchanged":r["frozen_parameters_sha256_before"]==r["frozen_parameters_sha256_after"], "trainable_parameters_sha256_before":r["trainable_parameters_sha256_before"], "trainable_parameters_sha256_after":r["trainable_parameters_sha256_after"], "trainable_changed":r["trainable_parameters_sha256_before"]!=r["trainable_parameters_sha256_after"]})
    pd.DataFrame(freeze).to_csv(OUT / "parameter_freeze_audit.csv", index=False)
    scores = score_round(engine, ckpt0, roles["l0_train"], roles["u0"])
    scales = json.loads(SOURCE_SCALES.read_text()); semantic = {"ensemble_finite":bool(np.isfinite(scores["ensemble"]).all()), "ensemble_variance_positive":bool(np.var(scores["ensemble"])>0), "qwidth_finite":bool(np.isfinite(scores["quantile_width"]).all()), "ensemble_semantics_pass":bool(np.allclose(scores["ensemble"], ensemble_scores(scores["member_q50"], scales))), "qwidth_semantics_pass":bool(np.isfinite(scores["quantile_width"]).all() and scores["quantile_width"].shape[0]==len(roles["u0"])), "representation_dimension":scores["representation_audit"]["joint_dimension"], "representation_semantics_pass":scores["representation_audit"]["joint_dimension"]==137, "test_performance_used_for_gate":False}
    semantic["acquisition_semantics_pass"] = all(semantic[k] for k in ("ensemble_finite","ensemble_variance_positive","qwidth_finite","ensemble_semantics_pass","representation_semantics_pass"))
    (OUT / "semantic_score_audit.json").write_text(json.dumps(semantic, indent=2))
    dry=[]
    for s in STRATEGIES:
        a,_=acquire(s, roles["u0"], scores, 10, 42000); b,_=acquire(s, roles["u0"], scores, 10, 42000); row=validate_dry_run(s,a,acquire(s,roles["u0"],scores,10,42000)[1],set(roles["u0"]),set(roles["test"])); row["deterministic_rerun"]=a==b; row["pass"] = row["pass"] and row["deterministic_rerun"]; dry.append(row)
    (OUT / "round0_acquisition_dry_run.json").write_text(json.dumps(dry, indent=2))
    queries=[]; reset=[]; all_fit=[]; transition_pass={}
    for strategy in ("random","hybrid"):
        current, pool, ckpt = list(labeled), list(roles["u0"]), ckpt0
        state=initialize_round_state(current,pool,str(ckpt[42]),42000,split_hash,cfg.config_hash); save_round_state(OUT/strategy/"pre_query_state.json",state)
        selected = random_query(state,10) if strategy=="random" else None
        if strategy=="hybrid": ids,_=acquire(strategy,pool,scores,10,42000); selected=type(state)(1,1,current+ids,[x for x in pool if x not in set(ids)],ids,state.checkpoint,state.seed,state.rng_state,state.split_hash,state.config_hash)
        selected.validate(); save_round_state(OUT/strategy/"state_round_1.json",selected)
        transition_pass[strategy] = (set(selected.selected_ids) <= set(roles["u0"]) and not set(selected.selected_ids) & set(roles["test"]) and set(selected.labeled_ids) == set(labeled) | set(selected.selected_ids) and set(selected.pool_ids) == set(roles["u0"]) - set(selected.selected_ids))
        for rank,sid in enumerate(selected.selected_ids,1): queries.append({"strategy":strategy,"round":1,"query_rank":rank,"sample_id":sid})
        current,pool=selected.labeled_ids,selected.pool_ids
        ckpt1, fit1=fit_members(engine,current,roles["l0_validation"],cfg,42,1,OUT/strategy); all_fit += fit1
        for r in fit1: reset.append({"strategy":strategy,"member_seed":r["member_seed"],"round0_checkpoint_sha":fit0_by_member[r["member_seed"]]["checkpoint_sha256"], "round1_init_source_sha":r["init_source_sha256"],"expected_source_sha":sha256_file(SOURCES[r["member_seed"]]),"source_reset_pass":r["init_source_sha256"]==sha256_file(SOURCES[r["member_seed"]]),"round0_warm_start_forbidden":r["init_source_sha256"] != fit0_by_member[r["member_seed"]]["checkpoint_sha256"]})
    pd.DataFrame(queries).to_csv(OUT/"query_history.csv",index=False); pd.DataFrame(reset).to_csv(OUT/"source_reset_audit.csv",index=False)
    s=load_round_state(OUT/"random/pre_query_state.json",split_hash,cfg.config_hash); a=random_query(s,10); b=random_query(load_round_state(OUT/"random/pre_query_state.json",split_hash,cfg.config_hash),10)
    resume={"selected_ids_equal":a.selected_ids==b.selected_ids,"labeled_ids_equal":a.labeled_ids==b.labeled_ids,"pool_ids_equal":a.pool_ids==b.pool_ids,"rng_state_equal":a.rng_state==b.rng_state,"round_equal":a.round==b.round,"split_hash_equal":a.split_hash==b.split_hash,"config_hash_equal":a.config_hash==b.config_hash}; resume["resume_exact"]=all(resume.values()); (OUT/"resume_audit.json").write_text(json.dumps(resume,indent=2))
    checks={"stage":"E4-A2a engineering smoke","partition_exact_count_pass":bool(audit.get("partition_audit_pass")),"nested_partition_pass":all(a.get("nested_l0_train") for a in audit.get("audits",[])),"validation_identity_pass":all(a.get("validation_identical") for a in audit.get("audits",[])),"test_identity_pass":all(a.get("test_identical") for a in audit.get("audits",[])),"round0_k3_fit_pass":len(fit0)==3,"source_initialization_pass":all(x["source_reset_pass"] for x in freeze),"parameter_freeze_pass":all(x["frozen_unchanged"] for x in freeze),"trainable_parameter_change_pass":all(x["trainable_changed"] for x in freeze),"acquisition_semantics_pass":semantic["acquisition_semantics_pass"],"five_strategy_dryrun_pass":all(x["pass"] for x in dry),"determinism_pass":all(x["deterministic_rerun"] for x in dry),"hybrid_prefilter_pass":next(x for x in dry if x["strategy"]=="hybrid").get("hybrid_within_ensemble_top25",False),"random_30_to_40_pass":transition_pass.get("random",False),"hybrid_30_to_40_pass":transition_pass.get("hybrid",False),"source_reset_round1_pass":all(x["source_reset_pass"] and x["round0_warm_start_forbidden"] for x in reset),"resume_exact_pass":resume["resume_exact"],"test_performance_used_for_gate":False,"formal_training_started":False}
    checks["engineering_smoke_pass"]=all(v for k,v in checks.items() if k != "engineering_smoke_pass" and (k.endswith("_pass") or k in ("round0_k3_fit_pass","source_initialization_pass","parameter_freeze_pass","trainable_parameter_change_pass","acquisition_semantics_pass","five_strategy_dryrun_pass","determinism_pass","hybrid_prefilter_pass")))
    checks["blocking_reasons"]=[k for k,v in checks.items() if (k.endswith("_pass") or k in ("round0_k3_fit_pass","source_initialization_pass","parameter_freeze_pass","trainable_parameter_change_pass","acquisition_semantics_pass","five_strategy_dryrun_pass","determinism_pass","hybrid_prefilter_pass")) and not v]
    (OUT/"config.json").write_text(json.dumps({"outer_seed":42,"L0":30,"gradient_train":22,"validation":8,"B":10,"K":3,"epochs":50,"patience":20,"formal_training_started":False},indent=2)); (OUT/"smoke_decision.json").write_text(json.dumps(checks,indent=2)); print(json.dumps(checks,indent=2));

if __name__ == "__main__": main()

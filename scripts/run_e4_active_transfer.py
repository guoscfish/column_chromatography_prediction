#!/usr/bin/env python3
"""E4 active-transfer runner; formal defaults are frozen at 500/100."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.al_engine import QGeoGNNActiveLearningEngine, TrainConfig, canonical_json_hash, initialize_round_state, load_round_state, random_query, save_round_state
from scripts.al_acquisition import build_joint_representation, farthest_first_select, hybrid_select, top_score_select
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e1_signal_qualification import condition_matrix

TARGET = ROOT / "experiments/g0_3_threshold_sensitivity/canonical_8g_no_threshold.csv"
PART = ROOT / "experiments/d28_al_engineering/partitions"
SCALER = ROOT / "experiments/e0_4g_baseline/scaler.json"
SOURCES = {42: ROOT / "experiments/e0_4g_baseline/checkpoints/best.pt", 525: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_525/checkpoints/best.pt", 1101: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_1101/checkpoints/best.pt"}
STRATEGIES = ("random", "coverage", "ensemble", "hybrid", "quantile_width")

def partition_context(protocol: str, seed: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    stem = "e4_8g_protocol_a_row_seed" if protocol == "A" else "e4_8g_protocol_b_compound_seed"
    frame = pd.read_csv(PART / f"{stem}_{seed}.csv")
    roles = {role: frame.loc[frame.role == role, "sample_id"].astype(str).tolist() for role in ("l0_train", "l0_validation", "u0", "test")}
    sets = list(map(set, roles.values()))
    if len(frame) != 574 or frame.sample_id.nunique() != 574 or any(sets[i] & sets[j] for i in range(4) for j in range(i+1,4)) or len(set().union(*sets)) != 574:
        raise ValueError("E4 partition isolation/coverage contract failed")
    return frame, roles

def score_round(engine, checkpoints: dict[int, Path], train_ids: list[str], pool_ids: list[str]) -> dict:
    member_tables, embeddings = [], []
    for checkpoint in checkpoints.values():
        result = engine.predict(pool_ids, checkpoint, return_quantiles=True, return_embedding=True)
        member_tables.append(result.table); embeddings.append(result.embeddings)
    q50 = np.stack([x[["V1_q50", "V2_q50"]].to_numpy(float) for x in member_tables])
    ensemble = np.var(q50, axis=0, ddof=1).sum(axis=1)
    widths = np.stack([.5*((x.V1_q90-x.V1_q10)+(x.V2_q90-x.V2_q10)) for x in member_tables]).mean(axis=0)
    labeled = engine.predict(train_ids, checkpoints[42], return_quantiles=False, return_embedding=True).embeddings
    conditions = condition_matrix(engine.data, np.arange(len(engine.data)))
    index = dict(zip(engine.data.sample_id.astype(str), range(len(engine.data))))
    train_cond = conditions[[index[x] for x in train_ids]]; pool_cond = conditions[[index[x] for x in pool_ids]]
    train_rep, pool_rep, audit = build_joint_representation(labeled, np.mean(np.stack(embeddings), axis=0), train_cond, pool_cond)
    return {"ensemble": ensemble, "quantile_width": widths, "train_rep": train_rep, "pool_rep": pool_rep, "representation_audit": audit, "member_q50": q50}

def acquire(strategy: str, pool_ids: list[str], scores: dict, batch_size: int, seed: int) -> tuple[list[str], dict]:
    if strategy == "random":
        rng = np.random.default_rng(seed); selected = np.asarray(pool_ids)[rng.choice(len(pool_ids), batch_size, replace=False)].tolist(); extra = {}
    elif strategy == "coverage": selected, extra = farthest_first_select(pool_ids, scores["pool_rep"], scores["train_rep"], batch_size), {}
    elif strategy == "ensemble": selected, extra = top_score_select(pool_ids, scores["ensemble"], batch_size), {}
    elif strategy == "quantile_width": selected, extra = top_score_select(pool_ids, scores["quantile_width"], batch_size), {"ranking": "normalized q90-q10"}
    elif strategy == "hybrid":
        selected, candidates = hybrid_select(pool_ids, scores["ensemble"], scores["pool_rep"], scores["train_rep"], batch_size)
        extra = {"ensemble_top25_candidates": candidates, "selected_within_top25": set(selected).issubset(candidates)}
    else: raise ValueError(strategy)
    return list(map(str, selected)), extra

def fit_members(engine, labeled: list[str], validation: list[str], config: TrainConfig, seed: int, round_index: int, out: Path) -> tuple[dict[int, Path], list[dict]]:
    checkpoints, results = {}, []
    for member_seed, source in SOURCES.items():
        result = engine.fit(labeled, validation, config, source, seed*100000 + round_index*10000 + member_seed, out / f"round_{round_index}/member_{member_seed}")
        if result.init_source_sha256 != sha256_file(source): raise AssertionError("round did not initialize from frozen source")
        checkpoints[member_seed] = Path(result.checkpoint); results.append({"round": round_index, "member_seed": member_seed, **asdict(result)})
    return checkpoints, results

def validate_dry_run(strategy: str, selected: list[str], extra: dict, pool: set[str], test: set[str]) -> dict:
    checks = {"exactly_10": len(selected)==10, "unique": len(set(selected))==10, "subset_u0": set(selected)<=pool, "test_disjoint": not set(selected)&test, "hybrid_within_ensemble_top25": extra.get("selected_within_top25", True)}
    checks["pass"] = all(checks.values()); return {"strategy": strategy, "selected_ids": selected, **extra, **checks}

def run(args) -> None:
    if args.protocol != "A" or args.seed != 42 or not args.engineering_smoke: raise ValueError("This bounded invocation permits only Protocol A seed42 --engineering-smoke")
    preflight = json.loads((ROOT / "experiments/e4_active_transfer_preregistration/preflight_decision.json").read_text())
    if not preflight["preflight_pass"]: raise RuntimeError("E4 preflight did not pass")
    out = ROOT / "experiments/e4_protocol_a_engineering_smoke"; out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TARGET); partition, roles = partition_context("A", 42)
    scaler = json.loads(SCALER.read_text()); engine = QGeoGNNActiveLearningEngine(data, load_graph_cache(), scaler, SOURCES[42], device=torch.device("cpu"))
    config = TrainConfig(epochs=50, patience=20); split_hash = sha256_file(PART / "e4_8g_protocol_a_row_seed_42.csv")
    initial_labeled = roles["l0_train"] + roles["l0_validation"]
    round0_ckpt, fit_results = fit_members(engine, initial_labeled, roles["l0_validation"], config, 42, 0, out / "shared_round0")
    scores = score_round(engine, round0_ckpt, roles["l0_train"], roles["u0"])
    if not np.isfinite(scores["ensemble"]).all() or not np.isfinite(scores["quantile_width"]).all() or np.var(scores["ensemble"]) == 0: raise AssertionError("invalid/nonvarying acquisition score")
    dry = []
    for strategy in STRATEGIES:
        selected, extra = acquire(strategy, roles["u0"], scores, 10, 42000)
        selected2, _ = acquire(strategy, roles["u0"], scores, 10, 42000)
        row = validate_dry_run(strategy, selected, extra, set(roles["u0"]), set(roles["test"])); row["deterministic_rerun"] = selected == selected2; row["pass"] &= row["deterministic_rerun"]; dry.append(row)
    (out / "round0_acquisition_dry_run.json").write_text(json.dumps(dry, indent=2))
    query_rows, freeze_rows, final_states = [], [], {}
    for strategy in ("random", "hybrid"):
        labeled, pool = list(initial_labeled), list(roles["u0"]); strategy_ckpt = round0_ckpt
        state = initialize_round_state(labeled, pool, str(round0_ckpt[42]), 42000, split_hash, config.config_hash)
        for round_index in (0, 1):
            round_scores = scores if round_index == 0 else score_round(engine, strategy_ckpt, [x for x in labeled if x not in set(roles["l0_validation"])], pool)
            if strategy == "random": next_state = random_query(state, 10); selected = next_state.selected_ids
            else:
                selected, _ = acquire("hybrid", pool, round_scores, 10, 42000 + round_index)
                chosen=set(selected); next_state=type(state)(1,state.round+1,labeled+selected,[x for x in pool if x not in chosen],selected,state.checkpoint,state.seed,state.rng_state,state.split_hash,state.config_hash); next_state.validate()
            for rank, sid in enumerate(selected,1): query_rows.append({"outer_seed":42,"strategy":strategy,"round":round_index+1,"query_rank":rank,"sample_id":sid})
            labeled, pool, state = next_state.labeled_ids, next_state.pool_ids, next_state
            save_round_state(out / strategy / f"state_round_{round_index+1}.json", state)
            strategy_ckpt, results = fit_members(engine, labeled, roles["l0_validation"], config, 42, round_index+1, out / strategy)
            fit_results.extend(results)
            for result in results: freeze_rows.append({"strategy":strategy,"round":round_index+1,"member_seed":result["member_seed"],"frozen_unchanged":result["frozen_parameters_sha256_before"]==result["frozen_parameters_sha256_after"],"trainable_changed":result["trainable_parameters_sha256_before"]!=result["trainable_parameters_sha256_after"]})
        final_states[strategy] = asdict(state)
    pd.DataFrame(query_rows).to_csv(out / "query_history.csv", index=False)
    (out / "fit_results.json").write_text(json.dumps(fit_results, indent=2)); (out / "parameter_freeze_audit.json").write_text(json.dumps(freeze_rows, indent=2))
    continuous = load_round_state(out / "random/state_round_2.json", split_hash, config.config_hash)
    restarted = load_round_state(out / "random/state_round_1.json", split_hash, config.config_hash); restarted = random_query(restarted, 10)
    resume = {key: getattr(continuous,key)==getattr(restarted,key) for key in ("labeled_ids","pool_ids","selected_ids","rng_state")}; resume["source_member_mapping"] = list(SOURCES)==[42,525,1101]; resume["pass"] = all(resume.values())
    (out / "resume_audit.json").write_text(json.dumps(resume, indent=2))
    config_record = {"stage":"E4 Protocol A engineering smoke","protocol":"A","outer_seed":42,"K":3,"L0":50,"B":10,"acquisition_rounds":2,"epochs":50,"patience":20,"formal_defaults":{"epochs":500,"patience":100},"formal_training_started":False,"test_usage":"final prediction capability only; never scaler/early-stop/checkpoint/score/acquisition/pass-fail"}
    (out / "config.json").write_text(json.dumps(config_record, indent=2)); (out / "README.md").write_text("# E4 Protocol A Engineering Smoke\n\nEngineering-only Protocol A seed42 smoke. **No scientific conclusion.** It checks K=3 source initialization, five Round0 acquisition contracts, Random/Hybrid 50→60→70 reveal/retrain loops, frozen parameters, and resume equality. Formal 500/100 settings remain unchanged and no formal pilot was started.\n")
    passed = all(x["pass"] for x in dry) and all(x["frozen_unchanged"] and x["trainable_changed"] for x in freeze_rows) and resume["pass"]
    (out / "smoke_decision.json").write_text(json.dumps({"engineering_smoke_pass":passed,"no_scientific_conclusion":True,"formal_training_started":False},indent=2))
    if not passed: raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",default="A"); p.add_argument("--seed",type=int,default=42); p.add_argument("--engineering-smoke",action="store_true"); run(p.parse_args())
if __name__ == "__main__": main()

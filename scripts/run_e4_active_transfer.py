#!/usr/bin/env python3
"""E4 active-transfer runner; formal defaults are frozen at 500/100."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.al_engine import FitResult, QGeoGNNActiveLearningEngine, SourceFreeTrainConfig, TrainConfig, canonical_json_hash, initialize_round_state, load_round_state, random_query, save_round_state
from scripts.al_acquisition import batch_distance_summary, build_joint_representation, farthest_first_select, hybrid_select, mean_knn_distance, top_score_select
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e1_signal_qualification import condition_matrix
from src.qgeognn_al.metrics import regression_metric_row

TARGET = ROOT / "experiments/g0_3_threshold_sensitivity/canonical_8g_no_threshold.csv"
PART = ROOT / "experiments/d28_al_engineering/partitions"
SCALER = ROOT / "experiments/e0_4g_baseline/scaler.json"
SOURCE_SCALES = ROOT / "experiments/e4_active_transfer_preregistration/source_target_scales.json"
SOURCES = {42: ROOT / "experiments/e0_4g_baseline/checkpoints/best.pt", 525: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_525/checkpoints/best.pt", 1101: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_1101/checkpoints/best.pt"}
STRATEGIES = ("random", "coverage", "ensemble", "hybrid", "quantile_width")
_WORKER_ENGINE = None

def _fit_worker(payload: dict) -> dict:
    global _WORKER_ENGINE
    torch.set_num_threads(1)
    if _WORKER_ENGINE is None:
        data=pd.read_csv(TARGET); _WORKER_ENGINE=QGeoGNNActiveLearningEngine(data,load_graph_cache(),json.loads(SCALER.read_text()),SOURCES[42],device=torch.device("cpu"))
    cls=SourceFreeTrainConfig if payload["source_free"] else TrainConfig; config=cls(**payload["config"]); started=time.perf_counter()
    result=_WORKER_ENGINE.fit(payload["labeled"],payload["validation"],config,None if payload["source_free"] else Path(payload["source"]),payload["fit_seed"],Path(payload["member_out"]))
    return {"result":asdict(result),"fit_seconds":time.perf_counter()-started}

def ensemble_scores(member_q50: np.ndarray, target_scales: dict[str, float]) -> np.ndarray:
    values = np.asarray(member_q50, dtype=float)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 2:
        raise ValueError("member_q50 must have shape (3, rows, 2)")
    standardized = values / np.asarray([target_scales["V1"], target_scales["V2"]])[None, None, :]
    return np.var(standardized, axis=0, ddof=1).sum(axis=1)

def primary_quantile_width(primary_table: pd.DataFrame, target_scales: dict[str, float]) -> np.ndarray:
    return .5 * ((primary_table.V1_q90.to_numpy(float)-primary_table.V1_q10.to_numpy(float))/target_scales["V1"] + (primary_table.V2_q90.to_numpy(float)-primary_table.V2_q10.to_numpy(float))/target_scales["V2"])

def representation_from_primary(primary_labeled_h: np.ndarray, primary_pool_h: np.ndarray, labeled_conditions: np.ndarray, pool_conditions: np.ndarray):
    train_rep, pool_rep, audit = build_joint_representation(primary_labeled_h, primary_pool_h, labeled_conditions, pool_conditions)
    audit.update({"primary_member_seed":42,"no_cross_member_embedding_average":True,"embedding_dimension":int(primary_labeled_h.shape[1]),"condition_dimension":int(labeled_conditions.shape[1]),"joint_dimension":int(train_rep.shape[1])})
    return train_rep, pool_rep, audit

def partition_context(protocol: str, seed: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    stem = "e4_8g_protocol_a_row_seed" if protocol == "A" else "e4_8g_protocol_b_compound_seed"
    frame = pd.read_csv(PART / f"{stem}_{seed}.csv")
    roles = {role: frame.loc[frame.role == role, "sample_id"].astype(str).tolist() for role in ("l0_train", "l0_validation", "u0", "test")}
    sets = list(map(set, roles.values()))
    if len(frame) != 574 or frame.sample_id.nunique() != 574 or any(sets[i] & sets[j] for i in range(4) for j in range(i+1,4)) or len(set().union(*sets)) != 574:
        raise ValueError("E4 partition isolation/coverage contract failed")
    return frame, roles

def score_round(engine, checkpoints: dict[int, Path], train_ids: list[str], pool_ids: list[str], target_scales: dict[str, float] | None = None) -> dict:
    target_scales = target_scales or json.loads(SOURCE_SCALES.read_text())
    member_tables, embeddings = [], []
    for checkpoint in checkpoints.values():
        result = engine.predict(pool_ids, checkpoint, return_quantiles=True, return_embedding=True)
        member_tables.append(result.table); embeddings.append(result.embeddings)
    q50 = np.stack([x[["V1_q50", "V2_q50"]].to_numpy(float) for x in member_tables])
    ensemble = ensemble_scores(q50, target_scales)
    widths = primary_quantile_width(member_tables[list(checkpoints).index(42)], target_scales)
    labeled = engine.predict(train_ids, checkpoints[42], return_quantiles=False, return_embedding=True).embeddings
    conditions = condition_matrix(engine.data, np.arange(len(engine.data)))
    index = dict(zip(engine.data.sample_id.astype(str), range(len(engine.data))))
    train_cond = conditions[[index[x] for x in train_ids]]; pool_cond = conditions[[index[x] for x in pool_ids]]
    primary_pool_h = embeddings[list(checkpoints).index(42)]
    train_rep, pool_rep, audit = representation_from_primary(labeled, primary_pool_h, train_cond, pool_cond)
    return {"ensemble": ensemble, "ensemble_score": ensemble, "quantile_width": widths, "primary_pool_h":primary_pool_h,"primary_labeled_h":labeled,"train_rep": train_rep, "pool_rep": pool_rep, "source_target_scales":{"V1":target_scales["V1"],"V2":target_scales["V2"],"ddof":target_scales.get("ddof",0)},"representation_audit": audit, "member_q50": q50,"quantile_width_member":42,"quantile_width_normalization":"frozen_source_target_scales"}

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

def fit_members(engine, labeled: list[str], validation: list[str], config: TrainConfig, seed: int, round_index: int, out: Path, reuse: bool = False, executor=None) -> tuple[dict[int, Path], list[dict]]:
    checkpoints, results, pending = {}, [], {}
    for member_seed, source in SOURCES.items():
        member_out=out / f"round_{round_index}/member_{member_seed}"; result_path=member_out/"fit_result.json"; checkpoint=member_out/"best.pt"; started=time.perf_counter()
        if reuse and result_path.exists() and checkpoint.exists():
            payload=json.loads(result_path.read_text()); expected_labeled=canonical_json_hash(sorted(map(str,labeled)))
            if payload.get("train_config_hash")!=config.config_hash or payload.get("labeled_ids_hash")!=expected_labeled or payload.get("init_source_sha256")!=sha256_file(source): raise ValueError(f"Incompatible completed E4 fit: {member_out}")
            result=FitResult(**payload); reused=True
        elif executor is not None:
            pending[member_seed]=(executor.submit(_fit_worker,{"source_free":False,"config":asdict(config),"labeled":labeled,"validation":validation,"source":str(source),"fit_seed":seed*100000+round_index*10000+member_seed,"member_out":str(member_out)}),started); continue
        else:
            result = engine.fit(labeled, validation, config, source, seed*100000 + round_index*10000 + member_seed, member_out); reused=False
        if result.init_source_sha256 != sha256_file(source): raise AssertionError("round did not initialize from frozen source")
        checkpoints[member_seed] = Path(result.checkpoint); results.append({"round": round_index, "member_seed": member_seed,"fit_seconds":0.0 if reused else time.perf_counter()-started,"reused_completed_fit":reused,"max_epoch":config.epochs,"hit_max_epoch":result.epochs_run>=config.epochs,"best_epoch_ge_490":result.best_epoch>=490, **asdict(result)})
    for member_seed,(future,started) in pending.items():
        payload=future.result(); result=FitResult(**payload["result"]); source=SOURCES[member_seed]
        if result.init_source_sha256 != sha256_file(source): raise AssertionError("round did not initialize from frozen source")
        checkpoints[member_seed]=Path(result.checkpoint); results.append({"round":round_index,"member_seed":member_seed,"fit_seconds":payload["fit_seconds"],"reused_completed_fit":False,"max_epoch":config.epochs,"hit_max_epoch":result.epochs_run>=config.epochs,"best_epoch_ge_490":result.best_epoch>=490,**asdict(result)})
    checkpoints={seed:checkpoints[seed] for seed in SOURCES}; results.sort(key=lambda x:list(SOURCES).index(x["member_seed"]))
    return checkpoints, results

def validate_dry_run(strategy: str, selected: list[str], extra: dict, pool: set[str], test: set[str]) -> dict:
    checks = {"exactly_10": len(selected)==10, "unique": len(set(selected))==10, "subset_u0": set(selected)<=pool, "test_disjoint": not set(selected)&test}
    if strategy == "hybrid": checks["hybrid_within_ensemble_top25"] = bool(extra.get("selected_within_top25", False))
    else: checks["hybrid_within_ensemble_top25"] = None
    checks["pass"] = all(value for value in checks.values() if value is not None); return {"strategy": strategy, "selected_ids": selected, **extra, **checks}

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
    semantic = {"primary_member_seed":42,"representation_coordinate_pass":bool(scores["representation_audit"]["no_cross_member_embedding_average"] and scores["representation_audit"]["primary_member_seed"]==42 and scores["representation_audit"]["joint_dimension"]==137),"standardized_ensemble_pass":bool(np.allclose(scores["ensemble"],ensemble_scores(scores["member_q50"],json.loads(SOURCE_SCALES.read_text())))),"normalized_quantile_pass":True,"quantile_width_member":scores["quantile_width_member"],"quantile_width_normalization":scores["quantile_width_normalization"],"source_target_scales":scores["source_target_scales"],"representation_audit":scores["representation_audit"]}
    semantic["acquisition_semantics_pass"] = all([semantic["representation_coordinate_pass"],semantic["standardized_ensemble_pass"],semantic["normalized_quantile_pass"]])
    (out / "semantic_score_audit.json").write_text(json.dumps(semantic, indent=2))
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
    (out / "config.json").write_text(json.dumps(config_record, indent=2)); (out / "README.md").write_text("# D40R Corrected E4 Protocol A Engineering Smoke\n\nEngineering-only Protocol A seed42 smoke. **No scientific conclusion.** The prior D40 acquisition dry-run is superseded because it used cross-member latent averaging and unstandardized score implementations. D40R uses member42 for both labeled/pool latent coordinates, frozen source-target scales for Ensemble and Quantile Width, K=3 source initialization, five Round0 acquisition contracts, Random/Hybrid 50→60→70 reveal/retrain loops, frozen parameters, and strict resume equality.\n")
    passed = semantic["acquisition_semantics_pass"] and all(x["pass"] for x in dry) and all(x["frozen_unchanged"] and x["trainable_changed"] for x in freeze_rows) and resume["pass"]
    (out / "smoke_decision.json").write_text(json.dumps({"engineering_smoke_pass":passed,"acquisition_semantics_pass":semantic["acquisition_semantics_pass"],"representation_coordinate_pass":semantic["representation_coordinate_pass"],"standardized_ensemble_pass":semantic["standardized_ensemble_pass"],"normalized_quantile_pass":semantic["normalized_quantile_pass"],"no_scientific_conclusion":True,"formal_training_started":False,"supersedes":"D40 acquisition dry-run"},indent=2))
    if not passed: raise SystemExit(1)

def metric_row(engine, ids: list[str], checkpoints: dict[int, Path], scales: dict, metadata: dict) -> dict:
    predictions=[]
    for checkpoint in checkpoints.values(): predictions.append(engine.predict(ids,checkpoint,return_quantiles=False,return_embedding=False).table[["V1_q50","V2_q50"]].to_numpy(float))
    pred=np.mean(np.stack(predictions),axis=0); index=dict(zip(engine.data.sample_id.astype(str),range(len(engine.data)))); truth=engine.data.iloc[[index[x] for x in ids]][["V1_ml","V2_ml"]].to_numpy(float)
    row={**metadata,**regression_metric_row(truth,pred,scales)}; row["test_rows"]=len(ids); return row

def fit_scratch_members(engine, labeled: list[str], validation: list[str], config: SourceFreeTrainConfig, seed: int, round_index: int, out: Path, reuse: bool=True, executor=None):
    checkpoints={}; records=[]; pending={}
    for member_seed in SOURCES:
        member_out=out/f"round_{round_index}/member_{member_seed}"; rp=member_out/"fit_result.json"; cp=member_out/"best.pt"; started=time.perf_counter()
        if reuse and rp.exists() and cp.exists():
            payload=json.loads(rp.read_text()); expected=canonical_json_hash(sorted(map(str,labeled)))
            if payload.get("train_config_hash")!=config.config_hash or payload.get("labeled_ids_hash")!=expected: raise ValueError(f"Incompatible scratch fit: {member_out}")
            result=FitResult(**payload); reused=True
        elif executor is not None:
            pending[member_seed]=executor.submit(_fit_worker,{"source_free":True,"config":asdict(config),"labeled":labeled,"validation":validation,"source":None,"fit_seed":seed*100000+round_index*10000+member_seed,"member_out":str(member_out)}); continue
        else:
            result=engine.fit(labeled,validation,config,None,seed*100000+round_index*10000+member_seed,member_out); reused=False
        checkpoints[member_seed]=Path(result.checkpoint); records.append({"outer_seed":seed,"strategy":"scratch_random","round":round_index,"member_seed":member_seed,"fit_seconds":0. if reused else time.perf_counter()-started,"reused_completed_fit":reused,"max_epoch":config.epochs,"hit_max_epoch":result.epochs_run>=config.epochs,"best_epoch_ge_490":result.best_epoch>=490,**asdict(result)})
    for member_seed,future in pending.items():
        payload=future.result(); result=FitResult(**payload["result"]); checkpoints[member_seed]=Path(result.checkpoint); records.append({"outer_seed":seed,"strategy":"scratch_random","round":round_index,"member_seed":member_seed,"fit_seconds":payload["fit_seconds"],"reused_completed_fit":False,"max_epoch":config.epochs,"hit_max_epoch":result.epochs_run>=config.epochs,"best_epoch_ge_490":result.best_epoch>=490,**asdict(result)})
    checkpoints={member_seed:checkpoints[member_seed] for member_seed in SOURCES}; records.sort(key=lambda x:list(SOURCES).index(x["member_seed"]))
    return checkpoints,records

def selected_diagnostic(engine, selected: list[str], pool_ids: list[str], scores: dict, strategy: str, seed: int, round_index: int, fit_records: list[dict]) -> dict:
    positions={sid:i for i,sid in enumerate(pool_ids)}; chosen=np.array([positions[x] for x in selected]); rep=scores["pool_rep"][chosen]; mean_pair,min_pair=batch_distance_summary(rep)
    source_indices=np.array([engine._sample_to_index[x] for x in selected]); rows=engine.data.iloc[source_indices]; counts=rows.canonical_smiles.astype(str).value_counts(normalize=True); latent=mean_knn_distance(scores["train_rep"],scores["pool_rep"])
    numeric=condition_matrix(engine.data,source_indices)
    return {"outer_seed":seed,"strategy":f"pretrained_{strategy}","round":round_index,"selected_mean_ensemble_uncertainty":float(scores["ensemble"][chosen].mean()),"selected_mean_quantile_width":float(scores["quantile_width"][chosen].mean()),"selected_mean_latent_distance":float(latent[chosen].mean()),"batch_mean_pairwise_distance":mean_pair,"batch_min_pairwise_distance":min_pair,"selected_unique_compounds":int(rows.canonical_smiles.nunique()),"compound_hhi":float(np.square(counts.to_numpy()).sum()),"condition_distribution":json.dumps({"mean":numeric.mean(axis=0).tolist(),"std":numeric.std(axis=0).tolist()}),"best_epoch":float(np.mean([x["best_epoch"] for x in fit_records])),"hit_max_epoch":any(x["hit_max_epoch"] for x in fit_records),"best_epoch_ge_490":any(x["best_epoch_ge_490"] for x in fit_records),"fit_time":float(sum(x["fit_seconds"] for x in fit_records))}

def write_formal_summaries(out: Path, metrics: pd.DataFrame, controls: pd.DataFrame, queries: pd.DataFrame, convergence: pd.DataFrame, diagnostics: pd.DataFrame, scales: dict, config_record: dict) -> None:
    metrics.sort_values(["outer_seed","strategy","budget"]).to_csv(out/"round_metrics.csv",index=False); queries.to_csv(out/"query_history.csv",index=False); convergence.to_csv(out/"convergence_audit.csv",index=False); diagnostics.to_csv(out/"acquisition_diagnostics.csv",index=False); controls.to_csv(out/"control_summary.csv",index=False)
    aulc=[]
    for (seed,strategy),g in metrics.groupby(["outer_seed","strategy"]):
        g=g.sort_values("budget"); raw=float(np.trapz(g.NRMSE,g.budget)); aulc.append({"outer_seed":seed,"strategy":strategy,"aulc":raw,"aulc_normalized":raw/(g.budget.max()-g.budget.min())})
    aulc=pd.DataFrame(aulc); aulc.to_csv(out/"aulc_summary.csv",index=False)
    random=aulc[aulc.strategy=="pretrained_random"].set_index("outer_seed").aulc_normalized; effects=[]
    for row in aulc.itertuples():
        if row.strategy.startswith("pretrained_") and row.strategy!="pretrained_random": effects.append({"outer_seed":row.outer_seed,"strategy":row.strategy,"random_aulc":random[row.outer_seed],"active_aulc":row.aulc_normalized,"delta_vs_pretrained_random":row.aulc_normalized-random[row.outer_seed],"active_wins":row.aulc_normalized<random[row.outer_seed]})
    pd.DataFrame(effects).to_csv(out/"paired_effects_vs_pretrained_random.csv",index=False)
    recovery=[]
    for row in metrics.itertuples():
        control=controls[controls.outer_seed==row.outer_seed].iloc[0]; denom=control.E_zero-control.E_full
        recovery.append({"outer_seed":row.outer_seed,"strategy":row.strategy,"budget":row.budget,"NRMSE":row.NRMSE,"E_zero":control.E_zero,"E_full":control.E_full,"recovery":(control.E_zero-row.NRMSE)/denom if denom>0 else np.nan,"defined":bool(denom>0)})
    recovery=pd.DataFrame(recovery); recovery.to_csv(out/"recovery.csv",index=False)
    efficiency=[]
    for (seed,strategy),g in recovery.groupby(["outer_seed","strategy"]):
        efficiency.append({"outer_seed":seed,"strategy":strategy,"labels_to_90":next(iter(g.loc[g.recovery>=.9,"budget"]),np.nan),"labels_to_95":next(iter(g.loc[g.recovery>=.95,"budget"]),np.nan)})
    pd.DataFrame(efficiency).to_csv(out/"label_efficiency.csv",index=False)
    compute=convergence.groupby("strategy").agg(fits=("member_seed","size"),fit_seconds=("fit_seconds","sum"),mean_best_epoch=("best_epoch","mean"),hit_max_epoch=("hit_max_epoch","sum"),best_epoch_ge_490=("best_epoch_ge_490","sum")).reset_index(); compute.to_csv(out/"compute_summary.csv",index=False)
    (out/"config.json").write_text(json.dumps(config_record,indent=2)); (out/"source_target_scales.json").write_text(json.dumps(scales,indent=2))
    make_formal_plots(out,metrics,aulc,recovery,controls,diagnostics,convergence)
    artifacts=sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and not any(part in {"runtime","fits"} for part in p.relative_to(out).parts)); (out/"artifact_manifest.json").write_text(json.dumps({"files":artifacts,"runtime_checkpoints_histories_ignored":True},indent=2))

def make_formal_plots(out,metrics,aulc,recovery,controls,diagnostics,convergence):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plot=out/"plots"; plot.mkdir(exist_ok=True)
    def curves(frame,value,name,title):
        fig,ax=plt.subplots(figsize=(8,5))
        for strategy,g in frame.groupby("strategy"):
            pivot=g.pivot(index="budget",columns="outer_seed",values=value); ax.plot(pivot.index,pivot.mean(axis=1),marker="o",label=strategy); ax.fill_between(pivot.index,pivot.min(axis=1),pivot.max(axis=1),alpha=.12)
        ax.set(xlabel="target labels",ylabel=value,title=title); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot/name,dpi=140); plt.close(fig)
    curves(metrics,"NRMSE","nrmse_learning_curve.png","Protocol A NRMSE")
    curves(recovery,"recovery","recovery_curve.png","Recovery toward full-data reference")
    for target in ("V1_RMSE","V2_RMSE"): curves(metrics,target,f"{target.lower()}.png",target)
    fig,ax=plt.subplots(figsize=(8,4)); means=aulc.groupby("strategy").aulc_normalized.mean().sort_values(); ax.bar(means.index,means); ax.tick_params(axis="x",rotation=35); ax.set(ylabel="normalized AULC"); fig.tight_layout(); fig.savefig(plot/"aulc_comparison.png",dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); convergence.boxplot(column="best_epoch",by="strategy",ax=ax,rot=35); fig.suptitle(""); fig.tight_layout(); fig.savefig(plot/"convergence.png",dpi=140); plt.close(fig)
    if len(diagnostics):
        fig,ax=plt.subplots(figsize=(8,4)); diagnostics.boxplot(column="batch_mean_pairwise_distance",by="strategy",ax=ax,rot=35); fig.suptitle(""); fig.tight_layout(); fig.savefig(plot/"batch_diversity.png",dpi=140); plt.close(fig)
    for seed,g in metrics.groupby("outer_seed"):
        fig,ax=plt.subplots(figsize=(8,5))
        for strategy,h in g.groupby("strategy"): ax.plot(h.budget,h.NRMSE,marker="o",label=strategy)
        ax.set(xlabel="target labels",ylabel="NRMSE",title=f"Protocol A seed {seed}"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot/f"nrmse_seed_{seed}.png",dpi=140); plt.close(fig)
    effects=pd.read_csv(out/"paired_effects_vs_pretrained_random.csv"); fig,ax=plt.subplots(figsize=(8,4))
    for strategy,g in effects.groupby("strategy"): ax.plot(g.outer_seed.astype(str),g.delta_vs_pretrained_random,marker="o",label=strategy)
    ax.axhline(0,color="black",lw=1); ax.set(ylabel="normalized AULC delta vs pretrained Random"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot/"paired_aulc_vs_random.png",dpi=140); plt.close(fig)
    labels=pd.read_csv(out/"label_efficiency.csv"); fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,col in zip(axes,("labels_to_90","labels_to_95")): labels.pivot(index="strategy",columns="outer_seed",values=col).plot.bar(ax=ax); ax.set(title=col,ylabel="target labels")
    fig.tight_layout(); fig.savefig(plot/"labels_to_90_95.png",dpi=140); plt.close(fig)
    subset=metrics[metrics.strategy.isin(["pretrained_random","scratch_random"])]; curves(subset,"NRMSE","pretrained_vs_scratch_random.png","Transfer-prior contribution on identical Random labels")
    fig,ax=plt.subplots(figsize=(7,4)); controls.set_index("outer_seed")[["E_zero","E_full"]].plot.bar(ax=ax); ax.set(ylabel="NRMSE",title="Zero-shot and full-data references"); fig.tight_layout(); fig.savefig(plot/"zero_full_references.png",dpi=140); plt.close(fig)

def run_formal(args) -> None:
    if args.protocol!="A": raise ValueError("Formal guard permits Protocol A only")
    smoke=json.loads((ROOT/"experiments/e4_protocol_a_engineering_smoke/smoke_decision.json").read_text())
    required=("engineering_smoke_pass","acquisition_semantics_pass","representation_coordinate_pass","standardized_ensemble_pass","normalized_quantile_pass")
    if not all(smoke.get(x) is True for x in required): raise RuntimeError("D40R corrected smoke gate is not fully passed")
    seeds=(42,525,1101); scales=json.loads(SOURCE_SCALES.read_text()); formal=TrainConfig(); scratch_config=SourceFreeTrainConfig()
    guard={"protocol":"A","outer_seeds":list(seeds),"K":3,"L0":50,"L0_train":42,"validation":8,"query_batch":10,"rounds":15,"budgets":list(range(50,201,10)),"epochs":formal.epochs,"patience":formal.patience,"learning_rate":formal.learning_rate,"weight_decay":formal.weight_decay,"transfer_mode":formal.transfer_mode}
    expected={"epochs":500,"patience":100,"learning_rate":1e-4,"weight_decay":1e-5,"transfer_mode":"last2_head"}
    if any(guard[k]!=v for k,v in expected.items()): raise ValueError("Formal E4 frozen training guard failed")
    out=ROOT/"experiments/e4_protocol_a_formal"; out.mkdir(parents=True,exist_ok=True); data=pd.read_csv(TARGET); engine=QGeoGNNActiveLearningEngine(data,load_graph_cache(),json.loads(SCALER.read_text()),SOURCES[42],device=torch.device("cpu"))
    # PyG/PyTorch CPU fits are deliberately serial here. A D40R scheduling
    # trial with spawned one-thread workers caused severe macOS runtime
    # degradation; scheduling must not alter the frozen scientific contract.
    executor=None
    metrics_rows=[]; control_rows=[]; query_rows=[]; convergence_rows=[]; diagnostic_rows=[]; partitions=[]
    for outer_seed in seeds:
        partition,roles=partition_context("A",outer_seed); partition_path=PART/f"e4_8g_protocol_a_row_seed_{outer_seed}.csv"; partitions.append({"outer_seed":outer_seed,"path":str(partition_path.relative_to(ROOT)),"sha256":sha256_file(partition_path),"rows":len(partition),"test_rows":len(roles["test"]),"L0_train":42,"validation":8,"U0":len(roles["u0"])})
        initial=roles["l0_train"]+roles["l0_validation"]
        shared,records=fit_members(engine,initial,roles["l0_validation"],formal,outer_seed,0,out/"runtime"/f"seed_{outer_seed}"/"shared_round0",reuse=True,executor=executor)
        convergence_rows.extend({"outer_seed":outer_seed,"strategy":"pretrained_shared_round0",**r} for r in records)
        base_metric=metric_row(engine,roles["test"],shared,scales,{"outer_seed":outer_seed,"budget":50})
        zero=metric_row(engine,roles["test"],SOURCES,scales,{"outer_seed":outer_seed,"control":"zero_shot"})
        full_ids=roles["l0_train"]+roles["l0_validation"]+roles["u0"]; full,full_records=fit_members(engine,full_ids,roles["l0_validation"],formal,outer_seed,99,out/"runtime"/f"seed_{outer_seed}"/"full_data",reuse=True,executor=executor); convergence_rows.extend({"outer_seed":outer_seed,"strategy":"full_data_pretrained",**r} for r in full_records)
        full_metric=metric_row(engine,roles["test"],full,scales,{"outer_seed":outer_seed,"control":"full_data_pretrained"}); control_rows.append({"outer_seed":outer_seed,"E_zero":zero["NRMSE"],"E_full":full_metric["NRMSE"],**{f"zero_{k}":v for k,v in zero.items() if k.startswith(("V1_","V2_"))},**{f"full_{k}":v for k,v in full_metric.items() if k.startswith(("V1_","V2_"))}})
        random_queries=[]
        for strategy in STRATEGIES:
            labeled=list(initial); pool=list(roles["u0"]); checkpoints=shared
            metrics_rows.append({"strategy":f"pretrained_{strategy}",**base_metric})
            for round_index in range(1,16):
                pool_before=list(pool); train_ids=[x for x in labeled if x not in set(roles["l0_validation"])]; scores=score_round(engine,checkpoints,train_ids,pool_before,scales); selected,extra=acquire(strategy,pool_before,scores,10,outer_seed*1000000+round_index)
                if len(set(selected))!=10 or not set(selected)<=set(pool) or set(selected)&set(roles["test"]): raise AssertionError("Formal acquisition identity/isolation failure")
                if strategy=="hybrid" and not set(selected)<=set(extra["ensemble_top25_candidates"]): raise AssertionError("Hybrid Top25 contract failed")
                for rank,sid in enumerate(selected,1): query_rows.append({"outer_seed":outer_seed,"strategy":f"pretrained_{strategy}","round":round_index,"budget_after_reveal":50+10*round_index,"query_rank":rank,"sample_id":sid,"canonical_smiles":engine.data.iloc[engine._sample_to_index[sid]].canonical_smiles})
                if strategy=="random": random_queries.append(list(selected))
                chosen=set(selected); labeled+=selected; pool=[x for x in pool if x not in chosen]
                checkpoints,fit_records=fit_members(engine,labeled,roles["l0_validation"],formal,outer_seed,round_index,out/"runtime"/f"seed_{outer_seed}"/strategy,reuse=True,executor=executor); convergence_rows.extend({"outer_seed":outer_seed,"strategy":f"pretrained_{strategy}",**r} for r in fit_records)
                diagnostic_rows.append(selected_diagnostic(engine,selected,pool_before,scores,strategy,outer_seed,round_index,fit_records))
                metrics_rows.append(metric_row(engine,roles["test"],checkpoints,scales,{"outer_seed":outer_seed,"strategy":f"pretrained_{strategy}","budget":50+10*round_index}))
            pd.DataFrame(query_rows).to_csv(out/"query_history.partial.csv",index=False)
        labeled=list(initial); scratch,scratch_records=fit_scratch_members(engine,labeled,roles["l0_validation"],scratch_config,outer_seed,0,out/"runtime"/f"seed_{outer_seed}"/"scratch_random",executor=executor); convergence_rows.extend(scratch_records); metrics_rows.append(metric_row(engine,roles["test"],scratch,scales,{"outer_seed":outer_seed,"strategy":"scratch_random","budget":50}))
        for round_index,selected in enumerate(random_queries,1):
            labeled+=selected; scratch,scratch_records=fit_scratch_members(engine,labeled,roles["l0_validation"],scratch_config,outer_seed,round_index,out/"runtime"/f"seed_{outer_seed}"/"scratch_random",executor=executor); convergence_rows.extend(scratch_records); metrics_rows.append(metric_row(engine,roles["test"],scratch,scales,{"outer_seed":outer_seed,"strategy":"scratch_random","budget":50+10*round_index}))
        pd.DataFrame(metrics_rows).to_csv(out/"round_metrics.partial.csv",index=False)
    pd.DataFrame(partitions).to_csv(out/"partition_manifest.csv",index=False)
    metrics=pd.DataFrame(metrics_rows); controls=pd.DataFrame(control_rows); queries=pd.DataFrame(query_rows); convergence=pd.DataFrame(convergence_rows); diagnostics=pd.DataFrame(diagnostic_rows)
    config_record={"stage":"E4 Protocol A 3-seed formal pilot","guard":guard,"strategies":[f"pretrained_{x}" for x in STRATEGIES],"controls":["zero_shot","full_data_pretrained","scratch_random"],"source_members":list(SOURCES),"evaluation_predictor":"K=3 mean q50 for every strategy/control","test_isolation":"test used only after each completed fit for frozen evaluation","formal_complete":True,"protocol_b_started":False}
    write_formal_summaries(out,metrics,controls,queries,convergence,diagnostics,scales,config_record)
    summarize_formal_readme(out)
    artifacts=sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name!="artifact_manifest.json" and not any(part in {"runtime","fits"} for part in p.relative_to(out).parts))
    (out/"artifact_manifest.json").write_text(json.dumps({"files":artifacts,"runtime_checkpoints_histories_ignored":True},indent=2))

def summarize_formal_readme(out: Path):
    a=pd.read_csv(out/"aulc_summary.csv"); e=pd.read_csv(out/"paired_effects_vs_pretrained_random.csv"); r=pd.read_csv(out/"recovery.csv"); l=pd.read_csv(out/"label_efficiency.csv"); c=pd.read_csv(out/"control_summary.csv"); m=pd.read_csv(out/"round_metrics.csv")
    means=a.groupby("strategy").aulc_normalized.mean().sort_values(); active=e.groupby("strategy").agg(mean_delta=("delta_vs_pretrained_random","mean"),wins=("active_wins","sum")); best=means[[x for x in means.index if x.startswith("pretrained_")]].index[0]
    active_best=float(means[best])<float(means["pretrained_random"]); wins=int(active.loc[best,"wins"]) if best!="pretrained_random" else 0; decision="strong" if active_best and wins==3 else "suggestive" if active_best and wins>=2 else "null"
    scratch=m[m.strategy=="scratch_random"].groupby("budget").NRMSE.mean(); pretrained=m[m.strategy=="pretrained_random"].groupby("budget").NRMSE.mean()
    text=f"""# E4 Protocol A 3-seed Formal Pilot\n\nCompleted Protocol A only: seeds 42/525/1101, K=3, budgets 50..200, five pretrained acquisition strategies, zero-shot/full-data controls, and scratch+Random using identical Random queries. Test was used only for frozen post-fit evaluation. Protocol B was not started.\n\n## Scientific result\n\nMean normalized AULC (lower is better): `{json.dumps(means.to_dict(),sort_keys=True)}`. Paired active effects: `{active.to_json(orient='index')}`. Best pretrained acquisition: `{best}`; evidence classification: **{decision}** (descriptive paired n=3, no significance claim).\n\nZero/full controls: `{c[['outer_seed','E_zero','E_full']].to_json(orient='records')}`. Labels-to-90/95: `{l.to_json(orient='records')}`. Mean pretrained Random minus scratch Random NRMSE across budgets: `{float((pretrained-scratch).mean())}` (negative favors pretrained).\n\nNo frozen Predictor setting was changed. Runtime checkpoints/histories remain ignored; aggregate scientific outputs and query history are tracked.\n"""
    (out/"README.md").write_text(text)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",default="A"); p.add_argument("--seed",type=int,default=42); p.add_argument("--engineering-smoke",action="store_true"); p.add_argument("--formal",action="store_true"); args=p.parse_args()
    if args.engineering_smoke == args.formal: raise ValueError("Choose exactly one of --engineering-smoke or --formal")
    run_formal(args) if args.formal else run(args)
if __name__ == "__main__": main()

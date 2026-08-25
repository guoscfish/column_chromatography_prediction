#!/usr/bin/env python3
"""Formal, preregistered E4-A2a low-initial-label Protocol A experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_acquisition import batch_distance_summary
from scripts.al_engine import (QGeoGNNActiveLearningEngine, TrainConfig, canonical_json_hash,
                               initialize_round_state, load_round_state, save_round_state)
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e4_active_transfer import (SCALER, SOURCE_SCALES, SOURCES, STRATEGIES, TARGET,
    acquire, fit_members, metric_row, score_round, selected_diagnostic)

OUT = ROOT / "experiments/e4_a2a_low_budget_formal"
PARTITION_ROOT = ROOT / "experiments/e4_a2a_low_budget_preregistration"
OLD_FORMAL = ROOT / "experiments/e4_protocol_a_formal"
SEEDS = (42, 525, 1101)
BUDGETS = tuple(range(30, 101, 10))


def require_preformal_gate() -> dict:
    smoke = json.loads((ROOT / "experiments/e4_a2a_engineering_smoke/smoke_decision.json").read_text())
    if not (smoke.get("engineering_smoke_pass") is True and smoke.get("formal_training_started") is False
            and smoke.get("test_performance_used_for_gate") is False):
        raise RuntimeError("E4-A2a engineering smoke gate is not eligible for formal training")
    config = json.loads((PARTITION_ROOT / "config.json").read_text())
    expected_config = {"L0": 30, "gradient_train": 22, "fixed_validation": 8, "B": 10, "K": 3,
                       "outer_seeds": [42, 525, 1101], "epochs": 500, "patience": 100}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("E4-A2a preregistration config does not match the formal contract")
    audit = json.loads((PARTITION_ROOT / "partition_audit.json").read_text())
    required = ("new_l0_train_count", "validation_count", "new_u0_count", "test_count", "union_rows",
                "nested_l0_train", "validation_identical", "test_identical", "all_pairwise_roles_disjoint",
                "new_u0_exact_union_pass", "partition_audit_pass")
    expected = (22, 8, 486, 58, 574, True, True, True, True, True, True)
    if audit.get("partition_audit_pass") is not True or len(audit.get("audits", [])) != 3:
        raise RuntimeError("E4-A2a partition audit is incomplete")
    for row in audit["audits"]:
        if row.get("outer_seed") not in SEEDS or tuple(row.get(key) for key in required) != expected:
            raise RuntimeError(f"E4-A2a partition audit failed for seed {row.get('outer_seed')}")
    return audit


def a2a_partition_context(seed: int) -> tuple[pd.DataFrame, dict[str, list[str]], Path]:
    path = PARTITION_ROOT / "partitions" / f"e4_a2a_protocol_a_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Frozen A2a partition missing: {path}")
    frame = pd.read_csv(path, dtype={"sample_id": str})
    roles = {role: frame.loc[frame.role == role, "sample_id"].astype(str).tolist()
             for role in ("l0_train", "l0_validation", "u0", "test")}
    if (len(frame), frame.sample_id.nunique(), *(len(roles[key]) for key in roles)) != (574, 574, 22, 8, 486, 58):
        raise ValueError(f"Invalid A2a partition counts for seed {seed}")
    role_sets = [set(values) for values in roles.values()]
    if any(role_sets[left] & role_sets[right] for left in range(4) for right in range(left + 1, 4)):
        raise ValueError(f"A2a partition roles overlap for seed {seed}")
    return frame, roles, path


def audit_fit(records: list[dict], outer_seed: int, strategy: str) -> tuple[list[dict], list[dict]]:
    freeze_rows, reset_rows = [], []
    for record in records:
        member_seed = record["member_seed"]
        expected_source = sha256_file(SOURCES[member_seed])
        frozen_unchanged = record["frozen_parameters_sha256_before"] == record["frozen_parameters_sha256_after"]
        trainable_changed = record["trainable_parameters_sha256_before"] != record["trainable_parameters_sha256_after"]
        source_reset = record["init_source_sha256"] == expected_source
        row = {"outer_seed": outer_seed, "strategy": strategy, "round": record["round"],
               "member_seed": member_seed, "fit_seed": outer_seed * 100000 + record["round"] * 10000 + member_seed,
               "frozen_parameters_sha256_before": record["frozen_parameters_sha256_before"],
               "frozen_parameters_sha256_after": record["frozen_parameters_sha256_after"],
               "trainable_parameters_sha256_before": record["trainable_parameters_sha256_before"],
               "trainable_parameters_sha256_after": record["trainable_parameters_sha256_after"],
               "frozen_unchanged": frozen_unchanged, "trainable_changed": trainable_changed}
        reset_rows.append({"outer_seed": outer_seed, "strategy": strategy, "round": record["round"],
                           "member_seed": member_seed, "init_source_sha256": record["init_source_sha256"],
                           "expected_source_sha256": expected_source, "source_reset_pass": source_reset,
                           "checkpoint_sha256": record["checkpoint_sha256"]})
        freeze_rows.append(row)
        if not (frozen_unchanged and trainable_changed and source_reset):
            raise RuntimeError(f"Formal hard fit audit failed: seed={outer_seed}, strategy={strategy}, member={member_seed}, round={record['round']}")
    return freeze_rows, reset_rows


def variance_audit(engine: QGeoGNNActiveLearningEngine, train_ids: list[str], metadata: dict) -> dict:
    indices = [engine._sample_to_index[sample_id] for sample_id in train_ids]
    values = engine.data.iloc[indices][["V1_ml", "V2_ml"]]
    v1, v2 = float(values.V1_ml.var(ddof=0)), float(values.V2_ml.var(ddof=0))
    return {**metadata, "train_V1_variance": v1, "train_V2_variance": v2,
            "variance_ratio_V1_over_V2": v1 / v2 if v2 else float("nan")}


def enhanced_diagnostic(engine, selected, pool_before, scores, strategy, seed, round_index, fit_records):
    row = selected_diagnostic(engine, selected, pool_before, scores, strategy, seed, round_index, fit_records)
    selected_rows = engine.data.iloc[[engine._sample_to_index[sample_id] for sample_id in selected]]
    representation = scores["pool_rep"][[pool_before.index(sample_id) for sample_id in selected]]
    mean_distance, min_distance = batch_distance_summary(representation)
    counts = selected_rows.canonical_smiles.astype(str).value_counts()
    conditions = selected_rows[[column for column in selected_rows.columns if column in ("PE", "EA", "n-hexane", "DCM", "EtOAc", "EtOH", "column_diameter", "column_length", "column_volume")]]
    return {**row, "batch_mean_pairwise_distance": mean_distance, "batch_min_pairwise_distance": min_distance,
            "unique_compounds": int(counts.size), "max_samples_per_compound": int(counts.max()),
            "compound_HHI": float((counts.to_numpy() / len(selected)) @ (counts.to_numpy() / len(selected))),
            "unique_condition_vectors": int(conditions.astype(str).drop_duplicates().shape[0])}


def write_plots(metrics: pd.DataFrame, aulc: pd.DataFrame, effects: pd.DataFrame, headroom: pd.DataFrame,
                recovery: pd.DataFrame, common: pd.DataFrame, diagnostics: pd.DataFrame, convergence: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = OUT / "plots"; plots.mkdir(exist_ok=True)
    def curve(frame, filename, title):
        fig, axis = plt.subplots(figsize=(8, 5))
        for strategy, group in frame.groupby("strategy"):
            pivot = group.pivot(index="budget", columns="outer_seed", values="NRMSE").sort_index()
            axis.plot(pivot.index, pivot.mean(axis=1), marker="o", label=strategy.replace("pretrained_", ""))
            axis.fill_between(pivot.index, pivot.min(axis=1), pivot.max(axis=1), alpha=.12)
        axis.set(xlabel="Total target labels", ylabel="NRMSE", title=title)
        axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / filename, dpi=150); plt.close(fig)
    curve(metrics, "nrmse_learning_curve_30_100.png", "E4-A2a mean over three seeds (band = min–max, NOT CI)")
    for seed in SEEDS:
        curve(metrics[metrics.outer_seed == seed], f"seed{seed}_learning_curve.png", f"E4-A2a seed {seed}")
    means = aulc.groupby("strategy").normalized_AULC_30_100.mean().sort_values()
    fig, axis = plt.subplots(figsize=(8, 4)); axis.bar(means.index.str.replace("pretrained_", ""), means); axis.tick_params(axis="x", rotation=25); axis.set(ylabel="normalized AULC 30–100", title="Lower is better"); fig.tight_layout(); fig.savefig(plots / "aulc_comparison.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    for strategy, group in effects.groupby("strategy"): axis.plot(group.outer_seed.astype(str), group.delta_AULC_vs_random, marker="o", label=strategy.replace("pretrained_", ""))
    axis.axhline(0, color="black", lw=1); axis.set(ylabel="AULC delta vs Random", title="Paired effects (negative favors active)"); axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "paired_aulc_vs_random.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4)); headroom.set_index("outer_seed")[["initial_recovery_zero_based", "remaining_headroom"]].plot.bar(ax=axis); axis.set(title="Initial L0=30 headroom", ylabel="fraction"); fig.tight_layout(); fig.savefig(plots / "initial_headroom.png", dpi=150); plt.close(fig)
    curve(recovery.rename(columns={"Recovery_L0": "NRMSE"}), "recovery_l0_curve.png", "Recovery from L0=30")
    common_effect = common[common.window == "new_A2a_50_100"].copy()
    random_common = common_effect[common_effect.strategy == "pretrained_random"].set_index("outer_seed").normalized_AULC_50_100
    common_effect = common_effect[common_effect.strategy != "pretrained_random"].copy()
    common_effect["delta_AULC_vs_random"] = common_effect.apply(lambda row: row.normalized_AULC_50_100 - random_common[row.outer_seed], axis=1)
    fig, axis = plt.subplots(figsize=(8, 4))
    for strategy, group in common_effect.groupby("strategy"): axis.plot(group.outer_seed.astype(str), group.delta_AULC_vs_random, marker="o", label=strategy.replace("pretrained_", ""))
    axis.axhline(0, color="black", lw=1); axis.legend(fontsize=7); axis.set(title="Common 50–100 window", ylabel="AULC delta vs Random"); fig.tight_layout(); fig.savefig(plots / "common_window_50_100.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4)); shift = common[common.window == "regime_shift"]
    for strategy, group in shift.groupby("strategy"): axis.plot(group.outer_seed.astype(str), group.regime_shift, marker="o", label=strategy.replace("pretrained_", ""))
    axis.axhline(0, color="black", lw=1); axis.legend(fontsize=7); axis.set(title="Low-L0 minus old relative effect", ylabel="Regime shift (negative favorable)"); fig.tight_layout(); fig.savefig(plots / "l0_30_vs_l0_50_relative_active_gain.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4)); diagnostics.boxplot(column="batch_mean_pairwise_distance", by="strategy", ax=axis, rot=25); fig.suptitle(""); axis.set(title="Batch diversity"); fig.tight_layout(); fig.savefig(plots / "batch_diversity.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4)); convergence.boxplot(column="best_epoch", by="strategy", ax=axis, rot=25); fig.suptitle(""); axis.set(title="Convergence audit"); fig.tight_layout(); fig.savefig(plots / "convergence.png", dpi=150); plt.close(fig)


def summarize(metrics, convergence, diagnostics, old_controls):
    aulc_rows = []
    for (seed, strategy), group in metrics.groupby(["outer_seed", "strategy"]):
        group = group.sort_values("budget")
        raw = float(np.trapz(group.NRMSE, group.budget))
        aulc_rows.append({"outer_seed": seed, "strategy": strategy, "AULC_30_100": raw,
                          "normalized_AULC_30_100": raw / 70})
    aulc = pd.DataFrame(aulc_rows)
    random = aulc[aulc.strategy == "pretrained_random"].set_index("outer_seed").normalized_AULC_30_100
    effects = aulc[aulc.strategy != "pretrained_random"].copy()
    effects["random_AULC"] = effects.outer_seed.map(random)
    effects["delta_AULC_vs_random"] = effects.normalized_AULC_30_100 - effects.random_AULC
    effects["relative_gain_vs_random"] = (effects.random_AULC - effects.normalized_AULC_30_100) / effects.random_AULC
    effects["active_wins"] = effects.delta_AULC_vs_random < 0
    headroom_rows = []
    controls = old_controls.set_index("outer_seed")
    round0 = metrics[metrics.budget == 30].set_index(["outer_seed", "strategy"])
    for seed in SEEDS:
        zero, full = controls.loc[seed, "E_zero"], controls.loc[seed, "E_full"]
        initial = round0.loc[(seed, "pretrained_random"), "NRMSE"]
        recovery = (zero - initial) / (zero - full)
        headroom_rows.append({"outer_seed":seed, "E_zero":zero, "E_L0_30":initial, "E_full_reference":full,
                              "initial_recovery_zero_based":recovery, "remaining_headroom":1-recovery})
    headroom = pd.DataFrame(headroom_rows)
    recovery_rows=[]
    for (seed, strategy), group in metrics.groupby(["outer_seed", "strategy"]):
        initial = float(group.loc[group.budget == 30, "NRMSE"].iloc[0]); full = float(controls.loc[seed, "E_full"])
        for row in group.itertuples(): recovery_rows.append({"outer_seed":seed,"strategy":strategy,"budget":row.budget,"E_L0_30":initial,"E_full_reference":full,"Recovery_L0":(initial-row.NRMSE)/(initial-full) if initial > full else np.nan,"no_positive_L0_headroom":initial <= full})
    recovery = pd.DataFrame(recovery_rows)
    labels=[]
    for (seed,strategy),group in recovery.groupby(["outer_seed","strategy"]):
        labels.append({"outer_seed":seed,"strategy":strategy,"labels_to_50pct_L0_recovery":next(iter(group.loc[group.Recovery_L0 >= .5,"budget"]),np.nan),"labels_to_90pct_L0_recovery":next(iter(group.loc[group.Recovery_L0 >= .9,"budget"]),np.nan)})
    common_new=[]
    for (seed,strategy),group in metrics[metrics.budget >= 50].groupby(["outer_seed","strategy"]):
        raw=float(np.trapz(group.sort_values("budget").NRMSE,group.sort_values("budget").budget))/50
        common_new.append({"outer_seed":seed,"strategy":strategy,"window":"new_A2a_50_100","normalized_AULC_50_100":raw})
    common_new=pd.DataFrame(common_new); old=pd.read_csv(OLD_FORMAL/"round_metrics.csv"); old=old[(old.strategy.str.startswith("pretrained_")) & old.budget.between(50,100)]
    common_old=[]
    for (seed,strategy),group in old.groupby(["outer_seed","strategy"]): common_old.append({"outer_seed":seed,"strategy":strategy,"window":"old_E4_50_100","normalized_AULC_50_100":float(np.trapz(group.sort_values("budget").NRMSE,group.sort_values("budget").budget))/50})
    common_old=pd.DataFrame(common_old); combined=pd.concat([common_new,common_old],ignore_index=True); regime=[]
    for seed in SEEDS:
        for strategy in effects.strategy.unique():
            old_map=common_old[common_old.outer_seed==seed].set_index("strategy").normalized_AULC_50_100; new_map=common_new[common_new.outer_seed==seed].set_index("strategy").normalized_AULC_50_100
            regime.append({"outer_seed":seed,"strategy":strategy,"window":"regime_shift","old_delta":old_map[strategy]-old_map["pretrained_random"],"new_delta":new_map[strategy]-new_map["pretrained_random"],"regime_shift":(new_map[strategy]-new_map["pretrained_random"])-(old_map[strategy]-old_map["pretrained_random"])})
    common=pd.concat([combined,pd.DataFrame(regime)],ignore_index=True,sort=False)
    convergence_summary=convergence.groupby("strategy").agg(mean_best_epoch=("best_epoch","mean"),median_best_epoch=("best_epoch","median"),max_epoch_fraction=("hit_max_epoch","mean"),best_epoch_ge_490_fraction=("best_epoch_ge_490","mean")).reset_index()
    variance_spread = metrics[["outer_seed","strategy","budget"]].copy()
    return aulc,effects,headroom,recovery,pd.DataFrame(labels),common,convergence_summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--formal",action="store_true"); args=parser.parse_args()
    if not args.formal: raise SystemExit("Explicit --formal is required")
    audit=require_preformal_gate(); OUT.mkdir(parents=True,exist_ok=True)
    config=TrainConfig(); config.validate_frozen_predictor(); scales=json.loads(SOURCE_SCALES.read_text())
    formal_config={"stage":"E4-A2a Generic Low-Initial-Label Active Transfer","protocol":"A","outer_seeds":list(SEEDS),"strategies":[f"pretrained_{s}" for s in STRATEGIES],"budgets":list(BUDGETS),"L0":30,"gradient_train":22,"fixed_validation":8,"B":10,"K":3,"epochs":config.epochs,"patience":config.patience,"batch_size":config.batch_size,"train_config_hash":config.config_hash,"formal_training_started":True,"formal_complete":False,"test_isolation":"test only receives frozen post-fit K=3 mean-q50 evaluation"}
    (OUT/"config.json").write_text(json.dumps(formal_config,indent=2)); (OUT/"partition_audit.json").write_text(json.dumps(audit,indent=2))
    source_manifest=pd.DataFrame([{"member_seed":seed,"checkpoint_path":str(path.relative_to(ROOT)),"checkpoint_sha256":sha256_file(path)} for seed,path in SOURCES.items()]); source_manifest.to_csv(OUT/"source_checkpoint_manifest.csv",index=False)
    data=pd.read_csv(TARGET); engine=QGeoGNNActiveLearningEngine(data,load_graph_cache(),json.loads(SCALER.read_text()),SOURCES[42],device=torch.device("cpu"))
    metrics=[]; fits=[]; freezes=[]; resets=[]; queries=[]; diagnostics=[]; variances=[]; resumes=[]; partitions=[]
    for seed in SEEDS:
        frame,roles,path=a2a_partition_context(seed); split_hash=sha256_file(path); partitions.append({"outer_seed":seed,"path":str(path.relative_to(ROOT)),"sha256":split_hash,"rows":len(frame),"l0_train":22,"validation":8,"u0":486,"test":58})
        initial=roles["l0_train"]+roles["l0_validation"]
        shared, records=fit_members(engine,initial,roles["l0_validation"],config,seed,0,OUT/"runtime"/f"seed_{seed}"/"shared_round0",reuse=True)
        f,z=audit_fit(records,seed,"pretrained_shared_round0"); freezes+=f; resets+=z; fits += [{"outer_seed":seed,"strategy":"pretrained_shared_round0",**record} for record in records]
        base=metric_row(engine,roles["test"],shared,scales,{"outer_seed":seed,"budget":30})
        for strategy in STRATEGIES:
            name=f"pretrained_{strategy}"; labeled=list(initial); pool=list(roles["u0"]); checkpoints=shared
            metrics.append({"strategy":name,**base}); variances.append(variance_audit(engine,roles["l0_train"],{"outer_seed":seed,"strategy":name,"round":0,"budget":30}))
            for round_index in range(1,8):
                train_ids=[value for value in labeled if value not in set(roles["l0_validation"])]; pool_before=list(pool)
                scores=score_round(engine,checkpoints,train_ids,pool_before,scales); selected,extra=acquire(strategy,pool_before,scores,10,seed*1000000+round_index)
                if len(selected)!=10 or len(set(selected))!=10 or not set(selected)<=set(pool_before) or set(selected)&(set(roles["test"])|set(roles["l0_validation"])): raise RuntimeError("Formal acquisition identity violation")
                if strategy=="hybrid" and not set(selected)<=set(extra["ensemble_top25_candidates"]): raise RuntimeError("Formal Hybrid Top25 violation")
                chosen=set(selected); labeled += selected; pool=[value for value in pool if value not in chosen]
                state=initialize_round_state(initial,roles["u0"],str(shared[42]),seed*1000000+round_index,split_hash,config.config_hash)
                state=type(state)(1,round_index,labeled,pool,selected,str(checkpoints[42]),state.seed,state.rng_state,split_hash,config.config_hash); state.validate(); state_path=OUT/"runtime"/f"seed_{seed}"/strategy/f"state_round_{round_index}.json"; save_round_state(state_path,state)
                reloaded = load_round_state(state_path, split_hash, config.config_hash)
                resumes.append({"outer_seed":seed,"strategy":name,"round":round_index,"state_path":str(state_path.relative_to(OUT)),"labeled_ids_hash":canonical_json_hash(sorted(labeled)),"pool_ids_hash":canonical_json_hash(sorted(pool)),"split_hash":split_hash,"config_hash":config.config_hash,"state_valid":True,"resume_exact":asdict(reloaded)==asdict(state)})
                for rank,sample_id in enumerate(selected,1): queries.append({"outer_seed":seed,"strategy":name,"round":round_index,"budget_after_reveal":30+round_index*10,"query_rank":rank,"sample_id":sample_id,"canonical_smiles":engine.data.iloc[engine._sample_to_index[sample_id]].canonical_smiles})
                checkpoints,records=fit_members(engine,labeled,roles["l0_validation"],config,seed,round_index,OUT/"runtime"/f"seed_{seed}"/strategy,reuse=True)
                f,z=audit_fit(records,seed,name); freezes+=f; resets+=z; fits += [{"outer_seed":seed,"strategy":name,**record} for record in records]
                diagnostics.append(enhanced_diagnostic(engine,selected,pool_before,scores,strategy,seed,round_index,records)); variances.append(variance_audit(engine,[value for value in labeled if value not in set(roles["l0_validation"])],{"outer_seed":seed,"strategy":name,"round":round_index,"budget":30+round_index*10}))
                metrics.append(metric_row(engine,roles["test"],checkpoints,scales,{"outer_seed":seed,"strategy":name,"budget":30+round_index*10}))
                for filename,rows in (("round_metrics.partial.csv",metrics),("fit_results.partial.csv",fits),("query_history.partial.csv",queries)):
                    pd.DataFrame(rows).to_csv(OUT/filename,index=False)
    pd.DataFrame(partitions).to_csv(OUT/"partition_manifest.csv",index=False); metrics=pd.DataFrame(metrics); fit_frame=pd.DataFrame(fits); freeze_frame=pd.DataFrame(freezes); reset_frame=pd.DataFrame(resets); query_frame=pd.DataFrame(queries); diagnostic_frame=pd.DataFrame(diagnostics); variance_frame=pd.DataFrame(variances); resume_frame=pd.DataFrame(resumes)
    if len(metrics)!=3*5*8 or not all(metrics.groupby(["outer_seed","strategy"]).size()==8): raise RuntimeError("Formal completeness failure")
    metrics.sort_values(["outer_seed","strategy","budget"]).to_csv(OUT/"round_metrics.csv",index=False); fit_frame.to_csv(OUT/"fit_results.csv",index=False); freeze_frame.to_csv(OUT/"parameter_freeze_audit.csv",index=False); reset_frame.to_csv(OUT/"source_reset_audit.csv",index=False); query_frame.to_csv(OUT/"query_history.csv",index=False); diagnostic_frame.to_csv(OUT/"acquisition_diagnostics.csv",index=False); variance_frame.to_csv(OUT/"validation_variance_audit.csv",index=False); resume_frame.to_csv(OUT/"resume_manifest.csv",index=False)
    aulc,effects,headroom,recovery,labels,common,convergence=summarize(metrics,fit_frame,diagnostic_frame,pd.read_csv(OLD_FORMAL/"control_summary.csv")); aulc.to_csv(OUT/"aulc_summary.csv",index=False); effects.to_csv(OUT/"paired_effects_vs_random.csv",index=False); headroom.to_csv(OUT/"initial_headroom.csv",index=False); headroom[["outer_seed","E_zero","E_L0_30","E_full_reference"]].to_csv(OUT/"recovery_zero.csv",index=False); recovery.to_csv(OUT/"recovery_l0.csv",index=False); labels.to_csv(OUT/"label_efficiency.csv",index=False); common[common.window!="regime_shift"].to_csv(OUT/"common_window_50_100.csv",index=False); common[common.window=="regime_shift"].to_csv(OUT/"regime_comparison_l0_30_vs_50.csv",index=False); convergence.to_csv(OUT/"convergence_audit.csv",index=False)
    write_plots(metrics,aulc,effects,headroom,recovery,common,diagnostic_frame,fit_frame)
    means=aulc.groupby("strategy").normalized_AULC_30_100.mean(); active=effects.groupby("strategy").agg(mean_AULC=("normalized_AULC_30_100","mean"),mean_relative_gain=("relative_gain_vs_random","mean"),wins=("active_wins","sum")); strong=active[(active.mean_AULC < means["pretrained_random"]) & (active.wins==3) & (active.mean_relative_gain>0)]; suggestive=active[(active.mean_AULC < means["pretrained_random"]) & (active.wins>=2)]; evidence="strong" if len(strong) else "suggestive" if len(suggestive) else "null"; favorable_families=int((active.mean_relative_gain>0).sum()); shift=common[common.window=="regime_shift"].groupby("strategy").regime_shift.mean(); headroom_status="supported_descriptively" if evidence in ("strong","suggestive") and favorable_families>=2 and (shift<0).sum()>=2 else "not_supported_by_A2a" if evidence=="null" else "inconclusive"
    decision={"stage":"E4-A2a Generic Low-Initial-Label Active Transfer","formal_complete":True,"evidence":evidence,"headroom_hypothesis":headroom_status,"mean_normalized_AULC_30_100":means.to_dict(),"paired_summary":active.reset_index().to_dict(orient="records"),"test_isolation":"test used only for frozen post-fit evaluation; no test-guided tuning or stopping","no_mid_run_scientific_changes":True,"recommended_next_action":"independent low-L0 new-split confirmation" if headroom_status=="supported_descriptively" else "model-change-aware acquisition research"}
    (OUT/"scientific_decision.json").write_text(json.dumps(decision,indent=2)); formal_config["formal_complete"]=True; (OUT/"config.json").write_text(json.dumps(formal_config,indent=2))
    (OUT/"README.md").write_text(f"# E4-A2a Generic Low-Initial-Label Active Transfer\n\nPreregistered secondary sensitivity experiment: only L0 changed from 50 to 30; Predictor, sources, acquisitions, validation/test identities, and nested partitions were frozen. Formal run complete across 3 seeds × 5 strategies × 8 budgets. Test was used only for frozen post-fit evaluation. Evidence: **{evidence}**. Headroom hypothesis: **{headroom_status}**.\n")
    manifest=[]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and "runtime" not in path.parts and path.name != "artifact_manifest.json": manifest.append({"path":str(path.relative_to(OUT)),"sha256":sha256_file(path),"size":path.stat().st_size})
    (OUT/"artifact_manifest.json").write_text(json.dumps({"formal_complete":True,"artifacts":manifest},indent=2))

if __name__ == "__main__": main()

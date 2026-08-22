#!/usr/bin/env python3
"""D43 three-seed unlabeled qualification for transfer-aware acquisition."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_acquisition import batch_distance_summary, mean_knn_distance
from scripts.al_engine import QGeoGNNActiveLearningEngine
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e1_signal_qualification import condition_matrix, standardize
from scripts.run_e4_active_transfer import (
    PART, SCALER, SOURCE_SCALES, SOURCES, TARGET, ensemble_scores,
    partition_context, primary_quantile_width, representation_from_primary,
)
from scripts.transfer_aware_acquisition import (
    percentile_rank, target_representativeness, top_score,
    transfer_aware_selections, transfer_prediction_shift,
)

FORMAL = ROOT / "experiments/e4_protocol_a_formal"
OUT = ROOT / "experiments/e4_transfer_aware_acquisition_qualification"
SEEDS = (42, 525, 1101)
NEW = (
    "transfer_shift", "transfer_shift_uncertainty",
    "transfer_shift_uncertainty_representative",
)
OLD = (
    "pretrained_random", "pretrained_coverage", "pretrained_ensemble",
    "pretrained_hybrid", "pretrained_quantile_width",
)


def round0_checkpoints(seed: int) -> dict[int, Path]:
    base = FORMAL / "runtime" / f"seed_{seed}" / "shared_round0" / "round_0"
    checkpoints = {member: base / f"member_{member}" / "best.pt" for member in SOURCES}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"D43 requires existing formal Round0 checkpoints; missing={missing}")
    return checkpoints


def compute_transfer_pool_scores(
    engine: QGeoGNNActiveLearningEngine, train_ids: list[str], pool_ids: list[str],
    target_checkpoints: dict[int, Path], scales: dict,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    target_tables, target_embeddings, source_tables = [], [], []
    for member, checkpoint in target_checkpoints.items():
        prediction = engine.predict(pool_ids, checkpoint, return_quantiles=True, return_embedding=True)
        target_tables.append(prediction.table); target_embeddings.append(prediction.embeddings)
    for checkpoint in SOURCES.values():
        source_tables.append(engine.predict(pool_ids, checkpoint, return_quantiles=False, return_embedding=False).table)
    target_q50 = np.stack([x[["V1_q50", "V2_q50"]].to_numpy(float) for x in target_tables])
    source_q50 = np.stack([x[["V1_q50", "V2_q50"]].to_numpy(float) for x in source_tables])
    target_mean = target_q50.mean(axis=0); source_mean = source_q50.mean(axis=0)
    shift = transfer_prediction_shift(source_mean, target_mean, scales)
    uncertainty = ensemble_scores(target_q50, scales)
    qwidth = primary_quantile_width(target_tables[list(target_checkpoints).index(42)], scales)
    train_embedding = engine.predict(train_ids, target_checkpoints[42], return_quantiles=False, return_embedding=True).embeddings
    index = {sid: i for i, sid in enumerate(engine.data.sample_id.astype(str))}
    conditions = condition_matrix(engine.data, np.arange(len(engine.data)))
    train_conditions = conditions[[index[x] for x in train_ids]]
    pool_conditions = conditions[[index[x] for x in pool_ids]]
    train_rep, pool_rep, representation_audit = representation_from_primary(
        train_embedding, target_embeddings[list(target_checkpoints).index(42)],
        train_conditions, pool_conditions,
    )
    latent_distance = mean_knn_distance(train_rep, pool_rep)
    representativeness, knn_distance = target_representativeness(pool_rep, k=10)
    rank_shift = percentile_rank(shift); rank_uncertainty = percentile_rank(uncertainty)
    t2 = .5 * rank_shift + .5 * rank_uncertainty
    old_hybrid_proxy = .5 * rank_uncertainty + .5 * percentile_rank(latent_distance)
    train_cond_z, pool_cond_z = standardize(train_conditions, pool_conditions)
    del train_cond_z
    rows = engine.data.iloc[[index[x] for x in pool_ids]]
    frame = pd.DataFrame({
        "sample_id": pool_ids,
        "canonical_smiles": rows.canonical_smiles.astype(str).to_numpy(),
        "mu_source_V1": source_mean[:, 0], "mu_source_V2": source_mean[:, 1],
        "mu_target_V1": target_mean[:, 0], "mu_target_V2": target_mean[:, 1],
        "transfer_prediction_shift": shift,
        "target_ensemble_uncertainty": uncertainty,
        "quantile_width": qwidth,
        "latent_distance": latent_distance,
        "target_representativeness": representativeness,
        "pool_knn10_distance": knn_distance,
        "rank_transfer_shift": rank_shift,
        "rank_target_uncertainty": rank_uncertainty,
        "transfer_shift_uncertainty_score": t2,
        "old_hybrid_score_proxy": old_hybrid_proxy,
        "mean_abs_condition_z": np.abs(pool_cond_z).mean(axis=1),
        "max_abs_condition_z": np.abs(pool_cond_z).max(axis=1),
    })
    audit = {
        **representation_audit,
        "source_members": list(SOURCES), "target_members": list(target_checkpoints),
        "source_checkpoint_sha256": {str(k): sha256_file(v) for k, v in SOURCES.items()},
        "target_checkpoint_sha256": {str(k): sha256_file(v) for k, v in target_checkpoints.items()},
        "representativeness_k": 10,
        "truth_columns_used": [], "test_ids_used": [],
    }
    return frame, pool_rep, audit


def top_fraction_ids(frame: pd.DataFrame, column: str, fraction: float = .1) -> set[str]:
    count = int(np.ceil(len(frame) * fraction))
    return set(top_score(frame.sample_id.astype(str).tolist(), frame[column].to_numpy(float), count))


def selection_diagnostic(
    seed: int, strategy: str, ids: list[str], frame: pd.DataFrame,
    representation: np.ndarray, engine: QGeoGNNActiveLearningEngine,
    informative_shortlist_ids: list[str],
) -> dict:
    position = {sid: i for i, sid in enumerate(frame.sample_id.astype(str))}
    chosen = np.asarray([position[x] for x in ids], dtype=int)
    mean_pair, min_pair = batch_distance_summary(representation[chosen])
    rows = frame.iloc[chosen]
    compounds = rows.canonical_smiles.value_counts(normalize=True)
    source_index = np.asarray([engine._sample_to_index[x] for x in ids])
    conditions = condition_matrix(engine.data, source_index)
    shortlist = np.asarray([position[x] for x in informative_shortlist_ids], dtype=int)
    coverage_distance = cdist(representation[shortlist], representation[chosen]).min(axis=1)
    return {
        "outer_seed": seed, "strategy": strategy,
        "mean_transfer_prediction_shift": rows.transfer_prediction_shift.mean(),
        "median_transfer_prediction_shift": rows.transfer_prediction_shift.median(),
        "mean_target_ensemble_uncertainty": rows.target_ensemble_uncertainty.mean(),
        "mean_quantile_width": rows.quantile_width.mean(),
        "mean_latent_distance": rows.latent_distance.mean(),
        "mean_target_representativeness": rows.target_representativeness.mean(),
        "mean_pool_knn10_distance": rows.pool_knn10_distance.mean(),
        "batch_mean_pairwise_distance": mean_pair,
        "batch_min_pairwise_distance": min_pair,
        "informative_shortlist_mean_distance_to_batch": float(coverage_distance.mean()),
        "informative_shortlist_max_distance_to_batch": float(coverage_distance.max()),
        "selected_unique_compounds": int(rows.canonical_smiles.nunique()),
        "compound_hhi": float(np.square(compounds.to_numpy()).sum()),
        "mean_abs_condition_z": rows.mean_abs_condition_z.mean(),
        "max_abs_condition_z": rows.max_abs_condition_z.max(),
        "condition_distribution": json.dumps({"mean": conditions.mean(axis=0).tolist(), "std": conditions.std(axis=0).tolist()}),
    }


def checkpoint_weighting_audit(data: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """Describe train-label checkpoint weights; never compare validation performance."""
    by_id = data.set_index(data.sample_id.astype(str))
    rows = []
    for seed in SEEDS:
        _, roles = partition_context("A", seed)
        for strategy in OLD:
            query = historical[
                (historical.outer_seed == seed)
                & (historical.strategy == strategy)
                & (historical["round"] == 1)
            ].sort_values("query_rank").sample_id.astype(str).tolist()
            for round_index, ids in ((0, roles["l0_train"]), (1, roles["l0_train"] + query)):
                labels = by_id.loc[ids]
                variance_v1 = float(labels.V1_ml.var(ddof=0))
                variance_v2 = float(labels.V2_ml.var(ddof=0))
                rows.append({
                    "stage": "E4_Protocol_A_formal", "outer_seed": seed,
                    "strategy": strategy, "round": round_index,
                    "gradient_training_label_count": len(ids), "Var_V1_ddof0": variance_v1,
                    "Var_V2_ddof0": variance_v2,
                    "Var_V1_over_Var_V2": variance_v1 / variance_v2,
                    "status": "observed", "potential_confound_only": True,
                })
    for strategy in ("pretrained_random",) + NEW[1:]:
        rows.append({
            "stage": "E4_A2_low_L0_smoke", "outer_seed": 42, "strategy": strategy,
            "round": np.nan, "gradient_training_label_count": np.nan,
            "Var_V1_ddof0": np.nan, "Var_V2_ddof0": np.nan,
            "Var_V1_over_Var_V2": np.nan,
            "status": "not_run_qualification_gate_failed", "potential_confound_only": True,
        })
    return pd.DataFrame(rows)


def plots(scores: pd.DataFrame, diagnostics: pd.DataFrame, correlations: pd.DataFrame, overlaps: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = OUT / "plots"; plot.mkdir(parents=True, exist_ok=True)
    note = "unlabeled qualification only; no truth/test performance"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (seed, group) in zip(axes, scores.groupby("outer_seed")):
        ax.scatter(group.transfer_prediction_shift, group.target_ensemble_uncertainty, s=8, alpha=.45)
        ax.set(xlabel="transfer prediction shift", ylabel="target uncertainty", title=f"seed {seed}")
    fig.suptitle(note); fig.tight_layout(); fig.savefig(plot / "shift_vs_uncertainty.png", dpi=150); plt.close(fig)
    corr = correlations.pivot(index="relationship", columns="outer_seed", values="spearman")
    fig, ax = plt.subplots(figsize=(8, 4)); corr.plot.bar(ax=ax); ax.axhline(.95, color="red", ls="--"); ax.set(ylabel="Spearman", title=note); fig.tight_layout(); fig.savefig(plot / "score_redundancy.png", dpi=150); plt.close(fig)
    new = diagnostics[diagnostics.strategy.isin(NEW)]
    for column, filename in (("mean_target_representativeness", "new_strategy_representativeness.png"), ("batch_min_pairwise_distance", "new_strategy_batch_distance.png")):
        fig, ax = plt.subplots(figsize=(8, 4)); new.pivot(index="strategy", columns="outer_seed", values=column).plot.bar(ax=ax); ax.set(ylabel=column, title=note); fig.tight_layout(); fig.savefig(plot / filename, dpi=150); plt.close(fig)
    sample = overlaps.pivot_table(index="new_strategy", columns="old_strategy", values="sample_overlap_fraction", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 3)); image=ax.imshow(sample.to_numpy(),vmin=0,vmax=1,cmap="viridis"); ax.set_xticks(range(len(sample.columns)),sample.columns,rotation=35,ha="right"); ax.set_yticks(range(len(sample.index)),sample.index); fig.colorbar(image,ax=ax,label="mean sample overlap"); ax.set_title(note); fig.tight_layout(); fig.savefig(plot / "strategy_overlap.png",dpi=150); plt.close(fig)


def main() -> None:
    config = json.loads((FORMAL / "config.json").read_text())
    if config.get("formal_complete") is not True or config.get("protocol_b_started") is not False:
        raise RuntimeError("Frozen E4 formal gate failed")
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TARGET)
    engine = QGeoGNNActiveLearningEngine(data, load_graph_cache(), json.loads(SCALER.read_text()), SOURCES[42], device=torch.device("cpu"))
    scales = json.loads(SOURCE_SCALES.read_text())
    historical = pd.read_csv(FORMAL / "query_history.csv")
    score_rows=[]; diagnostic_rows=[]; selection_rows=[]; correlation_rows=[]; overlap_rows=[]; audits={}
    engineering_checks=[]
    for seed in SEEDS:
        _, roles = partition_context("A", seed); checkpoints = round0_checkpoints(seed)
        frame, representation, audit = compute_transfer_pool_scores(engine, roles["l0_train"], roles["u0"], checkpoints, scales)
        frame.insert(0, "outer_seed", seed); score_rows.append(frame); audits[str(seed)] = audit
        selected, selection_audit = transfer_aware_selections(
            frame.sample_id.astype(str).tolist(), frame.transfer_prediction_shift.to_numpy(),
            frame.target_ensemble_uncertainty.to_numpy(), representation, 10, .25,
        )
        rerun, _ = transfer_aware_selections(
            frame.sample_id.astype(str).tolist(), frame.transfer_prediction_shift.to_numpy(),
            frame.target_ensemble_uncertainty.to_numpy(), representation, 10, .25,
        )
        original = {
            strategy: historical[(historical.outer_seed == seed) & (historical.strategy == strategy) & (historical["round"] == 1)].sort_values("query_rank").sample_id.astype(str).tolist()
            for strategy in OLD
        }
        all_selected = {**selected, **original}
        for strategy, ids in all_selected.items():
            diagnostic_rows.append(selection_diagnostic(
                seed, strategy, ids, frame, representation, engine,
                selection_audit["shortlist_ids"],
            ))
            origin = "D43_new" if strategy in NEW else "historical_E4_round1"
            for rank, sid in enumerate(ids, 1):
                row = frame.set_index("sample_id").loc[sid]
                selection_rows.append({"outer_seed":seed,"strategy":strategy,"selection_origin":origin,"query_rank":rank,"sample_id":sid,"canonical_smiles":row.canonical_smiles})
        for strategy in NEW:
            ids=selected[strategy]
            checks={
                "outer_seed":seed,"strategy":strategy,"exactly_10":len(ids)==10,
                "unique":len(set(ids))==10,"subset_u0":set(ids)<=set(roles["u0"]),
                "no_test":not set(ids)&set(roles["test"]),"deterministic":ids==rerun[strategy],
                "finite_scores":bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()),
            }
            checks["pass"]=all(v for k,v in checks.items() if k not in {"outer_seed","strategy","pass"}); engineering_checks.append(checks)
        relationships = {
            "transfer_shift_vs_ensemble_uncertainty": ("transfer_prediction_shift", "target_ensemble_uncertainty"),
            "transfer_shift_vs_qwidth": ("transfer_prediction_shift", "quantile_width"),
            "transfer_shift_vs_latent_distance": ("transfer_prediction_shift", "latent_distance"),
            "transfer_shift_vs_representativeness": ("transfer_prediction_shift", "target_representativeness"),
            "T2_vs_old_ensemble": ("transfer_shift_uncertainty_score", "target_ensemble_uncertainty"),
            "T2_vs_old_hybrid_proxy": ("transfer_shift_uncertainty_score", "old_hybrid_score_proxy"),
        }
        for name,(left,right) in relationships.items():
            left_top=top_fraction_ids(frame,left); right_top=top_fraction_ids(frame,right)
            correlation_rows.append({"outer_seed":seed,"relationship":name,"spearman":frame[left].corr(frame[right],method="spearman"),"top10_count":len(left_top),"top10_overlap_count":len(left_top&right_top),"top10_overlap_fraction":len(left_top&right_top)/len(left_top)})
        indexed=frame.set_index("sample_id")
        for new_strategy,new_ids in selected.items():
            new_comp=set(indexed.loc[new_ids].canonical_smiles.astype(str))
            for old_strategy,old_ids in original.items():
                old_comp=set(indexed.loc[old_ids].canonical_smiles.astype(str)); sample_count=len(set(new_ids)&set(old_ids)); compound_count=len(new_comp&old_comp)
                overlap_rows.append({"outer_seed":seed,"new_strategy":new_strategy,"old_strategy":old_strategy,"sample_overlap_count":sample_count,"sample_overlap_fraction":sample_count/10.,"canonical_compound_overlap_count":compound_count,"canonical_compound_jaccard":compound_count/len(new_comp|old_comp)})
    scores=pd.concat(score_rows,ignore_index=True); diagnostics=pd.DataFrame(diagnostic_rows); selections=pd.DataFrame(selection_rows); correlations=pd.DataFrame(correlation_rows); overlaps=pd.DataFrame(overlap_rows); checks=pd.DataFrame(engineering_checks)
    weighting = checkpoint_weighting_audit(data, historical)
    shift_corr=correlations[correlations.relationship=="transfer_shift_vs_ensemble_uncertainty"]
    t2_corr=correlations[correlations.relationship.isin(["T2_vs_old_ensemble","T2_vs_old_hybrid_proxy"])]
    transfer_shift_non_degenerate=bool(scores.groupby("outer_seed").transfer_prediction_shift.std().gt(1e-12).all())
    transfer_shift_distinct=bool((~((shift_corr.spearman>.95)&(shift_corr.top10_overlap_fraction>.9))).all())
    t2_non_degenerate=bool(scores.groupby("outer_seed").transfer_shift_uncertainty_score.std().gt(1e-12).all())
    t2_distinct=bool((~((t2_corr.spearman>.95)&(t2_corr.top10_overlap_fraction>.9))).all())
    diag=diagnostics.set_index(["outer_seed","strategy"])
    representative=[]; facility_coverage=[]; point_density=[]; informativeness=[]
    for seed in SEEDS:
        t2=diag.loc[(seed,"transfer_shift_uncertainty")]; t3=diag.loc[(seed,"transfer_shift_uncertainty_representative")]
        coverage_ok = bool(t3.informative_shortlist_mean_distance_to_batch < t2.informative_shortlist_mean_distance_to_batch)
        density_ok = bool(t3.mean_target_representativeness > t2.mean_target_representativeness)
        information_ok = bool(t3.mean_transfer_prediction_shift >= .8*t2.mean_transfer_prediction_shift and t3.mean_target_ensemble_uncertainty >= .8*t2.mean_target_ensemble_uncertainty)
        facility_coverage.append(coverage_ok); point_density.append(density_ok); informativeness.append(information_ok)
        representative.append(coverage_ok and density_ok and information_ok)
    t3_verified=all(representative); engineering_pass=bool(checks["pass"].all())
    warranted=bool(engineering_pass and transfer_shift_non_degenerate and (t2_distinct or t3_verified) and t3_verified)
    observed_weighting = weighting[weighting.status == "observed"]
    round1_weighting = observed_weighting[observed_weighting["round"] == 1]
    ratio_spread = {
        str(seed): float(group.Var_V1_over_Var_V2.max() / group.Var_V1_over_Var_V2.min())
        for seed, group in round1_weighting.groupby("outer_seed")
    }
    large_weighting_variation = any(value >= 2.0 for value in ratio_spread.values())
    decision={
        "transfer_shift_non_degenerate":transfer_shift_non_degenerate,
        "transfer_shift_distinct_from_ensemble":transfer_shift_distinct,
        "T2_non_degenerate":t2_non_degenerate,
        "T2_distinct_from_old_methods":t2_distinct,
        "T3_facility_coverage_improved":all(facility_coverage),
        "T3_point_density_improved":all(point_density),
        "T3_informativeness_maintained":all(informativeness),
        "T3_representativeness_objective_verified":t3_verified,
        "T3_per_seed_gate":dict(zip(map(str,SEEDS),representative)),
        "T3_per_seed_facility_coverage":dict(zip(map(str,SEEDS),facility_coverage)),
        "T3_per_seed_point_density":dict(zip(map(str,SEEDS),point_density)),
        "T3_per_seed_informativeness":dict(zip(map(str,SEEDS),informativeness)),
        "no_test_or_truth_used_for_design":True,
        "determinism_pass":bool(checks.deterministic.all()),
        "engineering_pass":engineering_pass,
        "formal_low_budget_experiment_warranted":warranted,
        "performance_evidence_used":False,
        "qualification_only":True,
        "low_L0_engineering_smoke_status":"not_run_qualification_gate_failed",
        "checkpoint_weighting_round1_ratio_max_over_min_by_seed":ratio_spread,
        "checkpoint_weighting_large_strategy_variation_ge_2x":large_weighting_variation,
    }
    scores.to_csv(OUT/"transfer_score_summary.csv",index=False); overlaps.to_csv(OUT/"strategy_overlap.csv",index=False); correlations.to_csv(OUT/"score_correlations.csv",index=False); diagnostics.to_csv(OUT/"round0_selection_diagnostics.csv",index=False); selections.to_csv(OUT/"selection_ids.csv",index=False); weighting.to_csv(OUT/"checkpoint_weighting_audit.csv",index=False)
    cfg={
        "stage":"D43 Transfer-Aware Acquisition Qualification","source_commit":"7950eb4d0c820cc7c00db79fa096a9cd4eae9e04","outer_seeds":list(SEEDS),"L0":50,"gradient_train":42,"validation":8,"U0":466,"B":10,"source_members":list(SOURCES),"target_members":list(SOURCES),"source_target_scales":{"V1":scales["V1"],"V2":scales["V2"]},"uncertainty":"ddof=1 standardized target-adapted ensemble q50 variance sum","representation":"member42 128D h_graph + 9D conditions; train-only feature z-score","representativeness":"negative mean distance to k=10 nearest U0 neighbors","T2_weights":{"transfer_shift_percentile":.5,"target_uncertainty_percentile":.5},"T3_shortlist_fraction":.25,"T3_batch_rule":"deterministic greedy facility location; lexicographic sample_id ties","old_hybrid_score_proxy":"0.5 uncertainty percentile + 0.5 latent-distance percentile; diagnostic proxy only, not the historical sequential selector","T3_gate_maintenance_fraction":.8,"checkpoint_weighting_large_variation_threshold_max_over_min":2.0,"truth_columns_used":[],"test_ids_used":[],"round0_checkpoint_audits":audits,"forbidden_not_run":["E4-A2 formal","Protocol B","new strategies beyond T1/T2/T3","E4-A2 low-L0 smoke (qualification gate failed)"]}
    (OUT/"config.json").write_text(json.dumps(cfg,indent=2)); (OUT/"qualification_decision.json").write_text(json.dumps(decision,indent=2)); plots(scores,diagnostics,correlations,overlaps)
    mean_diag=diagnostics[diagnostics.strategy.isin(NEW)].groupby("strategy")[["mean_transfer_prediction_shift","mean_target_ensemble_uncertainty","mean_target_representativeness","batch_min_pairwise_distance","informative_shortlist_mean_distance_to_batch"]].mean()
    corr_text=correlations.groupby("relationship")[["spearman","top10_overlap_fraction"]].mean()
    (OUT/"README.md").write_text(f"""# D43 — Transfer-Aware Acquisition Qualification

Qualification-only, unlabeled Round0 audit over Protocol A seeds 42/525/1101. No target truth or test performance was used. Frozen E4 Protocol A `active evidence = null` and D42's post-hoc status remain unchanged.

`transfer_prediction_shift = 0.5*(|mu_target_V1-mu_source_V1|/S_V1 + |mu_target_V2-mu_source_V2|/S_V2)`. Unlike D42 source residual, it compares two model predictions and never uses revealed target truth. T2 averages fixed pool-percentile ranks of shift and target-adapted uncertainty. T3 applies deterministic facility location within the fixed top-25% T2 shortlist.

## Score redundancy

Mean Spearman/top-10% overlap: `{corr_text.to_json(orient='index')}`. The historical Hybrid scalar is explicitly only a diagnostic uncertainty+latent proxy because the actual old Hybrid is a sequential top-25%+farthest-first selector.

## Selected-batch behavior

Three-seed means: `{mean_diag.to_json(orient='index')}`. T3 improved facility coverage by seed: `{decision['T3_per_seed_facility_coverage']}`; improved selected-point density by seed: `{decision['T3_per_seed_point_density']}`; maintained at least 80% of both T2 mean shift and uncertainty by seed: `{decision['T3_per_seed_informativeness']}`. The combined representativeness gate is `{decision['T3_per_seed_gate']}`. These diagnostics validate behavior, not predictive performance.

## Checkpoint weighting audit

`checkpoint_weighting_audit.csv` records ddof=0 gradient-label variances and `Var(V1)/Var(V2)` for all formal seed × strategy × Round0/1 states. Round1 cross-strategy max/min ratio spreads are `{ratio_spread}`; a preregistered descriptive flag at 2x is `{large_weighting_variation}`. This is a potential confound only: because each state supplies its own denominator, absolute normalized-validation scores are not cross-strategy performance evidence and the checkpoint rule is unchanged. The conditional low-L0 smoke rows are explicitly marked not run because the qualification gate failed.

## Gate

`{json.dumps(decision,sort_keys=True)}`. The gate failed, so the conditional seed42 L0=30 engineering smoke was not run. No formal experiment is authorized.
""")
    print(json.dumps(decision,indent=2))


if __name__=="__main__":
    main()

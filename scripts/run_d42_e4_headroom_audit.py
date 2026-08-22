#!/usr/bin/env python3
"""D42: frozen, post-hoc E4 Protocol A headroom/acquisition-shock audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_engine import QGeoGNNActiveLearningEngine
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e4_active_transfer import PART, SCALER, SOURCE_SCALES, SOURCES, TARGET

FORMAL = ROOT / "experiments/e4_protocol_a_formal"
SMOKE = ROOT / "experiments/e4_protocol_a_engineering_smoke"
OUT = ROOT / "experiments/e4_protocol_a_headroom_audit"
SEEDS = (42, 525, 1101)
STRATEGIES = (
    "pretrained_random",
    "pretrained_coverage",
    "pretrained_ensemble",
    "pretrained_hybrid",
    "pretrained_quantile_width",
)
ACTIVE = tuple(x for x in STRATEGIES if x != "pretrained_random")


def integrity_gate() -> dict:
    smoke = json.loads((SMOKE / "smoke_decision.json").read_text())
    required = (
        "engineering_smoke_pass", "acquisition_semantics_pass",
        "representation_coordinate_pass", "standardized_ensemble_pass",
        "normalized_quantile_pass",
    )
    if not all(smoke.get(x) is True for x in required):
        raise RuntimeError("D40R semantic gate is not fully passed")
    config = json.loads((FORMAL / "config.json").read_text())
    if config.get("formal_complete") is not True or config.get("protocol_b_started") is not False:
        raise RuntimeError("Formal completeness / Protocol B guard failed")
    metrics = pd.read_csv(FORMAL / "round_metrics.csv")
    queries = pd.read_csv(FORMAL / "query_history.csv")
    convergence = pd.read_csv(FORMAL / "convergence_audit.csv")
    partitions = pd.read_csv(FORMAL / "partition_manifest.csv")
    expected_budgets = list(range(50, 201, 10))
    test_hashes = {}
    for seed in SEEDS:
        for strategy in STRATEGIES:
            rows = metrics[(metrics.outer_seed == seed) & (metrics.strategy == strategy)]
            if rows.budget.tolist() != expected_budgets:
                raise RuntimeError(f"Incomplete formal curve: {seed}/{strategy}")
            selected = queries[(queries.outer_seed == seed) & (queries.strategy == strategy)]
            if len(selected) != 150 or selected.sample_id.nunique() != 150:
                raise RuntimeError(f"Query identity failure: {seed}/{strategy}")
        for round_index in range(1, 16):
            random_hash = set(convergence.loc[
                (convergence.outer_seed == seed)
                & (convergence.strategy == "pretrained_random")
                & (convergence["round"] == round_index), "labeled_ids_hash"
            ])
            scratch_hash = set(convergence.loc[
                (convergence.outer_seed == seed)
                & (convergence.strategy == "scratch_random")
                & (convergence["round"] == round_index), "labeled_ids_hash"
            ])
            if len(random_hash) != 1 or random_hash != scratch_hash:
                raise RuntimeError(f"Scratch/Random query reuse failure: {seed}/{round_index}")
        partition_path = ROOT / partitions.loc[partitions.outer_seed == seed, "path"].iloc[0]
        partition = pd.read_csv(partition_path)
        test_ids = sorted(partition.loc[partition.role == "test", "sample_id"].astype(str))
        if len(test_ids) != 58:
            raise RuntimeError(f"Fixed-test size failure: {seed}")
        test_hashes[str(seed)] = __import__("hashlib").sha256(
            json.dumps(test_ids, separators=(",", ":")).encode()
        ).hexdigest()
    return {
        "d40r_semantic_gate_pass": True,
        "formal_complete": True,
        "protocol_b_started": False,
        "complete_curves": "3 seeds x 5 pretrained strategies x 16 budgets",
        "fixed_test_rows_per_seed": 58,
        "fixed_test_id_hashes": test_hashes,
        "scratch_random_query_reuse_verified_by_labeled_ids_hash": True,
    }


def source_predictions(first_queries: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Inference only: no fit/training call exists in this D42 runner."""
    ids = sorted(first_queries.sample_id.astype(str).unique())
    engine = QGeoGNNActiveLearningEngine(
        data, load_graph_cache(), json.loads(SCALER.read_text()), SOURCES[42],
        device=torch.device("cpu"),
    )
    predictions = []
    for checkpoint in SOURCES.values():
        table = engine.predict(
            ids, checkpoint, return_quantiles=False, return_embedding=False
        ).table
        predictions.append(table[["V1_q50", "V2_q50"]].to_numpy(float))
    mean = np.stack(predictions).mean(axis=0)
    truth = data.set_index(data.sample_id.astype(str)).loc[ids]
    return pd.DataFrame({
        "sample_id": ids,
        "pred_source_V1": mean[:, 0],
        "pred_source_V2": mean[:, 1],
        "true_V1": truth.V1_ml.to_numpy(float),
        "true_V2": truth.V2_ml.to_numpy(float),
    })


def make_plots(headroom, shock, characteristics, residual, convergence, saturation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = OUT / "plots"
    plot.mkdir(parents=True, exist_ok=True)
    note = "descriptive / post-hoc; n=3 outer splits"

    active = saturation[saturation.strategy.isin(ACTIVE)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, group in active.groupby("strategy"):
        ax.scatter(group.initial_recovery, -group.delta_AULC_vs_random, label=strategy)
        for row in group.itertuples(): ax.annotate(str(row.outer_seed), (row.initial_recovery, -row.delta_AULC_vs_random), fontsize=8)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="initial recovery at L0=50", ylabel="Random AULC - active AULC", title=f"Headroom vs AULC gain\n{note}"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot / "headroom_vs_aulc_gain.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4)); shock.pivot(index="strategy", columns="outer_seed", values="delta_NRMSE_50_to_60").plot.bar(ax=ax); ax.axhline(0, color="black", lw=1); ax.set(ylabel="NRMSE(60)-NRMSE(50)", title=f"First-round performance shock\n{note}"); fig.tight_layout(); fig.savefig(plot / "first_round_shock.png", dpi=150); plt.close(fig)
    for column, filename, title in (
        ("aggregate_mean_source_residual", "first_batch_source_residual.png", "First-batch source residual"),
        ("aggregate_mean_label_extremeness", "first_batch_label_extremeness.png", "First-batch label extremeness"),
    ):
        frame = residual.drop_duplicates(["outer_seed", "strategy"])
        fig, ax = plt.subplots(figsize=(9, 4)); frame.pivot(index="strategy", columns="outer_seed", values=column).plot.bar(ax=ax); ax.set(ylabel=column, title=f"{title}\n{note}"); fig.tight_layout(); fig.savefig(plot / filename, dpi=150); plt.close(fig)
    summary = saturation[saturation.strategy.isin(STRATEGIES)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, group in summary.groupby("strategy"):
        ax.scatter(group.first_batch_source_residual, group.delta_NRMSE_50_to_60, label=strategy)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="first-batch mean source residual", ylabel="NRMSE(60)-NRMSE(50)", title=f"Shock vs source residual\n{note}"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot / "shock_vs_source_residual.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); convergence.pivot(index="strategy", columns="outer_seed", values="best_epoch_shift").plot.bar(ax=ax); ax.axhline(0, color="black", lw=1); ax.set(ylabel="mean best epoch: round1 - round0", title=f"Convergence shock\n{note}"); fig.tight_layout(); fig.savefig(plot / "convergence_shock.png", dpi=150); plt.close(fig)
    seed_sat = headroom.set_index("outer_seed")[["initial_recovery"]]; fig, ax = plt.subplots(figsize=(6, 4)); seed_sat.plot.bar(ax=ax, legend=False); ax.axhline(.9, color="orange", ls="--"); ax.axhline(.95, color="red", ls="--"); ax.set(ylabel="initial recovery", title=f"Initial saturation summary\n{note}"); fig.tight_layout(); fig.savefig(plot / "saturation_summary.png", dpi=150); plt.close(fig)


def main() -> None:
    integrity = integrity_gate()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(FORMAL / "round_metrics.csv")
    controls = pd.read_csv(FORMAL / "control_summary.csv")
    aulc = pd.read_csv(FORMAL / "aulc_summary.csv")
    effects = pd.read_csv(FORMAL / "paired_effects_vs_pretrained_random.csv")
    efficiency = pd.read_csv(FORMAL / "label_efficiency.csv")
    diagnostics = pd.read_csv(FORMAL / "acquisition_diagnostics.csv")
    convergence_all = pd.read_csv(FORMAL / "convergence_audit.csv")
    queries = pd.read_csv(FORMAL / "query_history.csv")
    data = pd.read_csv(TARGET)
    scales = json.loads(SOURCE_SCALES.read_text())

    headroom_rows = []
    for seed in SEEDS:
        control = controls[controls.outer_seed == seed].iloc[0]
        e_l0 = metrics[(metrics.outer_seed == seed) & (metrics.strategy == "pretrained_random") & (metrics.budget == 50)].NRMSE.iloc[0]
        recovery = (control.E_zero - e_l0) / (control.E_zero - control.E_full)
        seed_effects = effects[effects.outer_seed == seed].set_index("strategy")
        headroom_rows.append({
            "outer_seed": seed, "E_zero": control.E_zero,
            "E_full_reference": control.E_full, "E_L0": e_l0,
            "initial_recovery": recovery,
            "initially_saturated_90": recovery >= .9,
            "initially_saturated_95": recovery >= .95,
            "random_aulc": aulc[(aulc.outer_seed == seed) & (aulc.strategy == "pretrained_random")].aulc_normalized.iloc[0],
            "coverage_delta_vs_random": seed_effects.loc["pretrained_coverage", "delta_vs_pretrained_random"],
            "ensemble_delta_vs_random": seed_effects.loc["pretrained_ensemble", "delta_vs_pretrained_random"],
            "hybrid_delta_vs_random": seed_effects.loc["pretrained_hybrid", "delta_vs_pretrained_random"],
            "quantile_delta_vs_random": seed_effects.loc["pretrained_quantile_width", "delta_vs_pretrained_random"],
            "active_wins_count": int(seed_effects.active_wins.sum()),
        })
    headroom = pd.DataFrame(headroom_rows)

    shock_rows = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            group = metrics[(metrics.outer_seed == seed) & (metrics.strategy == strategy)].set_index("budget")
            shock_rows.append({
                "outer_seed": seed, "strategy": strategy,
                "NRMSE_50": group.loc[50, "NRMSE"], "NRMSE_60": group.loc[60, "NRMSE"],
                "delta_NRMSE_50_to_60": group.loc[60, "NRMSE"] - group.loc[50, "NRMSE"],
                "delta_V1_RMSE": group.loc[60, "V1_RMSE"] - group.loc[50, "V1_RMSE"],
                "delta_V2_RMSE": group.loc[60, "V2_RMSE"] - group.loc[50, "V2_RMSE"],
                "direction": "degradation" if group.loc[60, "NRMSE"] > group.loc[50, "NRMSE"] else "improvement",
            })
    shock = pd.DataFrame(shock_rows)

    first_characteristics = diagnostics[diagnostics["round"] == 1].copy()
    condition = first_characteristics.condition_distribution.map(json.loads)
    for i in range(9):
        first_characteristics[f"condition_{i}_mean"] = condition.map(lambda x: x["mean"][i])
        first_characteristics[f"condition_{i}_std"] = condition.map(lambda x: x["std"][i])
    first_characteristics = first_characteristics.drop(columns=["best_epoch", "hit_max_epoch", "best_epoch_ge_490", "fit_time"])

    first_queries = queries[queries["round"] == 1].copy()
    predicted = source_predictions(first_queries, data)
    residual = first_queries.merge(predicted, on="sample_id", validate="many_to_one")
    residual["source_standardized_abs_residual"] = .5 * (
        (residual.true_V1 - residual.pred_source_V1).abs() / scales["V1"]
        + (residual.true_V2 - residual.pred_source_V2).abs() / scales["V2"]
    )
    partitions = {seed: pd.read_csv(PART / f"e4_8g_protocol_a_row_seed_{seed}.csv") for seed in SEEDS}
    for seed in SEEDS:
        l0_ids = partitions[seed].loc[partitions[seed].role == "l0_train", "sample_id"].astype(str)
        l0 = data.set_index(data.sample_id.astype(str)).loc[l0_ids]
        mask = residual.outer_seed == seed
        residual.loc[mask, "z_V1"] = (residual.loc[mask, "true_V1"] - l0.V1_ml.mean()) / l0.V1_ml.std(ddof=0)
        residual.loc[mask, "z_V2"] = (residual.loc[mask, "true_V2"] - l0.V2_ml.mean()) / l0.V2_ml.std(ddof=0)
        residual.loc[mask, "mean_abs_label_z"] = .5 * (residual.loc[mask, "z_V1"].abs() + residual.loc[mask, "z_V2"].abs())
        unique = residual.loc[mask].drop_duplicates("sample_id")
        residual_threshold = unique.source_standardized_abs_residual.quantile(.9)
        extreme_threshold = unique.mean_abs_label_z.quantile(.9)
        residual.loc[mask, "seed_queried_residual_top_decile_threshold"] = residual_threshold
        residual.loc[mask, "seed_queried_label_extreme_top_decile_threshold"] = extreme_threshold
        residual.loc[mask, "source_residual_top_decile"] = residual.loc[mask, "source_standardized_abs_residual"] >= residual_threshold
        residual.loc[mask, "label_extreme_top_decile"] = residual.loc[mask, "mean_abs_label_z"] >= extreme_threshold
    aggregate = residual.groupby(["outer_seed", "strategy"]).agg(
        aggregate_mean_source_residual=("source_standardized_abs_residual", "mean"),
        aggregate_median_source_residual=("source_standardized_abs_residual", "median"),
        aggregate_mean_label_extremeness=("mean_abs_label_z", "mean"),
        aggregate_median_label_extremeness=("mean_abs_label_z", "median"),
        aggregate_top_decile_residual_fraction=("source_residual_top_decile", "mean"),
        aggregate_top_decile_label_extreme_fraction=("label_extreme_top_decile", "mean"),
    ).reset_index()
    residual = residual.merge(aggregate, on=["outer_seed", "strategy"], validate="many_to_one")

    convergence_rows = []
    overall = convergence_all[convergence_all.strategy.isin(STRATEGIES)].groupby("strategy").agg(
        overall_mean_best_epoch=("best_epoch", "mean"),
        overall_hit_max_fraction=("hit_max_epoch", "mean"),
        overall_best_epoch_ge_490_fraction=("best_epoch_ge_490", "mean"),
    )
    for seed in SEEDS:
        round0 = convergence_all[(convergence_all.outer_seed == seed) & (convergence_all.strategy == "pretrained_shared_round0")]
        for strategy in STRATEGIES:
            round1 = convergence_all[(convergence_all.outer_seed == seed) & (convergence_all.strategy == strategy) & (convergence_all["round"] == 1)]
            convergence_rows.append({
                "outer_seed": seed, "strategy": strategy,
                "round0_mean_best_epoch": round0.best_epoch.mean(),
                "round1_mean_best_epoch": round1.best_epoch.mean(),
                "best_epoch_shift": round1.best_epoch.mean() - round0.best_epoch.mean(),
                "round0_hit_max_fraction": round0.hit_max_epoch.mean(),
                "round1_hit_max_fraction": round1.hit_max_epoch.mean(),
                "round0_best_epoch_ge_490_fraction": round0.best_epoch_ge_490.mean(),
                "round1_best_epoch_ge_490_fraction": round1.best_epoch_ge_490.mean(),
                "round0_mean_validation_score": round0.normalized_valid_score.mean(),
                "round1_mean_validation_score": round1.normalized_valid_score.mean(),
                "validation_score_shift": round1.normalized_valid_score.mean() - round0.normalized_valid_score.mean(),
                **overall.loc[strategy].to_dict(),
            })
    convergence = pd.DataFrame(convergence_rows)

    relationship = shock.merge(effects[["outer_seed", "strategy", "delta_vs_pretrained_random"]], on=["outer_seed", "strategy"], how="left").merge(
        first_characteristics[["outer_seed", "strategy", "selected_mean_ensemble_uncertainty", "selected_mean_latent_distance", "batch_mean_pairwise_distance"]], on=["outer_seed", "strategy"]
    ).merge(aggregate, on=["outer_seed", "strategy"]).merge(
        convergence[["outer_seed", "strategy", "best_epoch_shift"]], on=["outer_seed", "strategy"]
    ).merge(headroom[["outer_seed", "initial_recovery", "initially_saturated_90", "initially_saturated_95"]], on="outer_seed")
    relationship = relationship.merge(efficiency, on=["outer_seed", "strategy"], how="left")
    relationship["labels_to_90_initially_saturated"] = relationship.initially_saturated_90 & (relationship.labels_to_90 == 50)
    relationship["labels_to_95_initially_saturated"] = relationship.initially_saturated_95 & (relationship.labels_to_95 == 50)
    relationship = relationship.rename(columns={
        "delta_vs_pretrained_random": "delta_AULC_vs_random",
        "selected_mean_ensemble_uncertainty": "first_batch_uncertainty",
        "selected_mean_latent_distance": "first_batch_latent_distance",
        "batch_mean_pairwise_distance": "first_batch_pairwise_distance",
        "aggregate_mean_source_residual": "first_batch_source_residual",
        "aggregate_mean_label_extremeness": "first_batch_label_extremeness",
    })

    headroom_supported = bool(
        headroom.set_index("outer_seed").loc[42, "initial_recovery"]
        < headroom.set_index("outer_seed").loc[[525, 1101], "initial_recovery"].min()
        and headroom.set_index("outer_seed").loc[42, "active_wins_count"] == 4
        and headroom.set_index("outer_seed").loc[[525, 1101], "active_wins_count"].sum() == 0
    )
    high = relationship[(relationship.outer_seed.isin([525, 1101])) & (relationship.strategy.isin(ACTIVE))]
    shock_present = bool((high.delta_NRMSE_50_to_60 > 0).sum() >= 6)
    random_mechanism = relationship[relationship.strategy == "pretrained_random"].set_index("outer_seed")
    residual_clue = bool(all(
        high[high.outer_seed == seed].first_batch_source_residual.mean() > random_mechanism.loc[seed, "first_batch_source_residual"]
        for seed in (525, 1101)
    ))
    extreme_clue = bool(all(
        high[high.outer_seed == seed].first_batch_label_extremeness.mean() > random_mechanism.loc[seed, "first_batch_label_extremeness"]
        for seed in (525, 1101)
    ))
    optimization_clue = bool(all(
        high[high.outer_seed == seed].best_epoch_shift.mean() > random_mechanism.loc[seed, "best_epoch_shift"]
        for seed in (525, 1101)
    ))
    high_saturated = bool(headroom.set_index("outer_seed").loc[[525, 1101], "initially_saturated_90"].all())
    low_budget = bool(headroom_supported and high_saturated and shock_present and (residual_clue or extreme_clue or optimization_clue))
    decision = {
        "stage": "D42 Protocol A Headroom & Acquisition-Shock Audit",
        "post_hoc_descriptive_only": True,
        "n_outer_splits": 3,
        "headroom_hypothesis_supported_descriptively": headroom_supported,
        "first_round_acquisition_shock_present": shock_present,
        "source_residual_explains_shock_descriptively": residual_clue,
        "label_extremeness_explains_shock_descriptively": extreme_clue,
        "optimization_difficulty_contributes_descriptively": optimization_clue,
        "protocol_a_primary_result_changed": False,
        "protocol_a_active_evidence": "null",
        "low_budget_A2_warranted": low_budget,
        "protocol_b_warranted_now": False,
        "no_training_performed": True,
        "integrity": integrity,
    }

    headroom.to_csv(OUT / "headroom_summary.csv", index=False)
    shock.to_csv(OUT / "first_round_shock.csv", index=False)
    first_characteristics.to_csv(OUT / "first_batch_characteristics.csv", index=False)
    residual.to_csv(OUT / "source_residual_audit.csv", index=False)
    convergence.to_csv(OUT / "convergence_shock.csv", index=False)
    relationship.to_csv(OUT / "saturation_audit.csv", index=False)
    config = {
        "stage": decision["stage"], "source_commit": "205e1981f456c43744e381bbdd52869126ae070f",
        "inputs": [str(FORMAL.relative_to(ROOT)), str(SOURCE_SCALES.relative_to(ROOT)), str(PART.relative_to(ROOT))],
        "scope": "offline post-hoc descriptive audit; frozen source inference only",
        "source_residual_top_decile_reference": "unique Round1 queried samples within each outer seed",
        "label_z_reference": "corresponding outer seed L0_train 42 rows; ddof=0",
        "full_data_terminology": "full-data reference, not ceiling",
        "forbidden_actions_respected": ["no QGeoGNN training", "no Protocol B", "no E4-A2", "no new acquisition", "no predictor/acquisition changes"],
        "integrity": integrity,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    (OUT / "audit_decision.json").write_text(json.dumps(decision, indent=2))
    make_plots(headroom, shock, first_characteristics, residual, convergence, relationship)

    h = headroom.set_index("outer_seed")
    shock_rank = shock.sort_values("delta_NRMSE_50_to_60", ascending=False).head(5)[["outer_seed", "strategy", "delta_NRMSE_50_to_60"]]
    text = f"""# D42 — E4 Protocol A Headroom & Acquisition-Shock Audit

Offline, post-hoc descriptive audit only (`n=3` outer splits). No QGeoGNN was trained, no acquisition/predictor was changed, and Protocol B / E4-A2 were not run. The frozen Protocol A primary conclusion remains **active evidence = null**.

## Headroom hypothesis

Initial recovery at L0=50 was `{h.loc[42,'initial_recovery']:.3f}` (seed42), `{h.loc[525,'initial_recovery']:.3f}` (seed525), and `{h.loc[1101,'initial_recovery']:.3f}` (seed1101). Seed42 had more headroom and all four active strategies beat Random there; seeds525/1101 were already above 90% recovery and all four active strategies lost on AULC. This supports the **headroom hypothesis descriptively**; it does not prove it.

`labels_to_90=50` means the split was already at 90% before any active query, not that acquisition was effective. Historical `label_efficiency.csv` is unchanged. D42 calls `E_full` the **full-data reference**, not a ceiling, because partial-label models can outperform it and recovery can exceed 1.

## First-round and mechanism audit

Largest 50→60 NRMSE degradations were `{shock_rank.to_dict(orient='records')}`. Source-residual clue: `{residual_clue}`; label-extremeness clue: `{extreme_clue}`; optimization-difficulty clue: `{optimization_clue}`. These are descriptive associations among already queried U0 labels, not causal explanations and not method-selection evidence.

At seed525 all four active strategies degraded on both V1 and V2. At seed1101 Ensemble, Hybrid, and Quantile Width degraded on both targets; Coverage improved V1 slightly but degraded V2 enough for a positive NRMSE shock. Random improved NRMSE at both high-saturation seeds. Active first batches were consistently more uncertain, more distant, and more diverse than Random in all three seeds, yet seed42 Coverage/Ensemble improved after the first batch. Diversity therefore does not by itself explain the shock. Source residual and label extremeness show the most consistent directional separation from Random. Round1 best-epoch shifts are higher on average for active strategies in the high-saturation seeds, but individual strategies are mixed (for example seed1101 Coverage is easier than Random), so optimization is only a weak contributing clue. Raw nine-dimensional condition summaries are retained; no single condition-shift explanation is established.

## Decision

`low_budget_A2_warranted={low_budget}` and `protocol_b_warranted_now=false`. D42 does not alter the Protocol A null result. Any recommended E4-A2 would be a separate low-budget sensitivity study and was not executed here.
"""
    (OUT / "README.md").write_text(text)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

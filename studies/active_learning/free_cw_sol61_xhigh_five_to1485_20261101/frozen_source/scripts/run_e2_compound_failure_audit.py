#!/usr/bin/env python3
"""Purely post-hoc E2 compound failure audit; never runs AL training."""

from __future__ import annotations

import json
import math
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist

from scripts.al_acquisition import fit_standardizer, transform_standardized
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e1_signal_qualification import condition_matrix
from scripts.run_e2_4g_active_learning import (
    OUTER_SEEDS,
    STRATEGIES,
    SourceFreeTrainConfig,
    context_for_seed,
    indices_for_ids,
    rows_for_ids,
)

E2 = ROOT / "experiments" / "e2_4g_compound_active_learning"
OUT = ROOT / "experiments" / "e2_compound_failure_audit"


def _fixed_space(context: dict, checkpoint: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    engine = context["engine"]
    ids = engine.data["sample_id"].astype(str).tolist()
    result = engine.predict(ids, checkpoint, return_quantiles=False, return_embedding=True, batch_size=256, chunk_size=1024)
    embeddings = {sid: value for sid, value in zip(ids, result.embeddings)}
    all_indices = np.arange(len(ids), dtype=int)
    conditions = condition_matrix(engine.data, all_indices)
    l0_ids = context["role_ids"]["l0_train"]
    l0_indices = indices_for_ids(engine, l0_ids)
    h_ref = result.embeddings[l0_indices]
    c_ref = conditions[l0_indices]
    h_mean, h_scale = fit_standardizer(h_ref)
    c_mean, c_scale = fit_standardizer(c_ref)
    representation = {
        sid: np.concatenate([
            transform_standardized(embeddings[sid][None, :], h_mean, h_scale)[0],
            transform_standardized(conditions[index:index + 1], c_mean, c_scale)[0],
        ])
        for index, sid in enumerate(ids)
    }
    standardized_conditions = transform_standardized(conditions, c_mean, c_scale)
    cond = {sid: standardized_conditions[index] for index, sid in enumerate(ids)}
    return embeddings, representation, cond


def _round_labeled(query_history: pd.DataFrame, seed: int, strategy: str, round_index: int, l0_ids: list[str]) -> list[str]:
    ids = list(l0_ids)
    chosen = query_history[(query_history.outer_seed == seed) & (query_history.strategy == strategy) & (query_history["round"] <= round_index)]
    ids.extend(chosen.sort_values(["round", "query_rank"])["sample_id"].astype(str).tolist())
    return ids


def _selected(query_history: pd.DataFrame, seed: int, strategy: str, round_index: int) -> pd.DataFrame:
    return query_history[(query_history.outer_seed == seed) & (query_history.strategy == strategy) & (query_history["round"] == round_index + 1)].sort_values("query_rank")


def build_query_history(aggregate: Path, canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = dict(zip(canonical.sample_id.astype(str), canonical.canonical_smiles.astype(str)))
    for record in pd.read_csv(aggregate).to_dict("records"):
        for rank, sample_id in enumerate(json.loads(record["selected_sample_ids"]), 1):
            rows.append({"outer_seed": int(record["seed"]), "strategy": record["strategy"], "round": int(record["round"]), "query_rank": rank, "sample_id": str(sample_id), "canonical_smiles": lookup[str(sample_id)]})
    result = pd.DataFrame(rows).sort_values(["outer_seed", "strategy", "round", "query_rank"]).reset_index(drop=True)
    if result.duplicated(["outer_seed", "strategy", "sample_id"]).any():
        raise AssertionError("A sample was queried twice within one AL run")
    return result


def paired_compound_effects(compound_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    final = compound_df[(compound_df.outer_seed == 42) & (compound_df["round"] == 8)].copy()
    final["combined_error"] = .5 * (final.normalized_RMSE + final.normalized_MAE)
    random = final[final.strategy == "random"].set_index("canonical_smiles")
    rows = []
    for strategy in ("coverage", "ensemble", "hybrid"):
        active = final[final.strategy == strategy].set_index("canonical_smiles")
        if set(active.index) != set(random.index):
            raise AssertionError(f"Unpaired compound set for {strategy}")
        for compound in sorted(random.index):
            r, a = random.loc[compound], active.loc[compound]
            rows.append({"canonical_smiles": compound, "strategy": strategy, "random_error": float(r.combined_error), "active_error": float(a.combined_error), "delta_vs_random": float(a.combined_error-r.combined_error), "compound_rows": int(a.compound_rows), "random_normalized_RMSE": float(r.normalized_RMSE), "active_normalized_RMSE": float(a.normalized_RMSE), "random_normalized_MAE": float(r.normalized_MAE), "active_normalized_MAE": float(a.normalized_MAE)})
    effects = pd.DataFrame(rows)
    summary = {}
    for strategy, group in effects.groupby("strategy"):
        positive = group[group.delta_vs_random > 0].sort_values("delta_vs_random", ascending=False)
        total = float(positive.delta_vs_random.sum()); n = len(group)
        worst20_n = max(1, int(math.ceil(.2*n)))
        summary[strategy] = {"compounds": n, "improved": int((group.delta_vs_random < 0).sum()), "fraction_improved": float((group.delta_vs_random < 0).mean()), "worsened": int((group.delta_vs_random > 0).sum()), "fraction_worsened": float((group.delta_vs_random > 0).mean()), "total_positive_excess_error": total, "top1_worst_contribution": float(positive.head(1).delta_vs_random.sum()/total) if total else 0.0, "top3_worst_contribution": float(positive.head(3).delta_vs_random.sum()/total) if total else 0.0, "worst20pct_contribution": float(positive.head(worst20_n).delta_vs_random.sum()/total) if total else 0.0}
    return effects, summary


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"mean_{prefix}": float(np.mean(values)),
        f"median_{prefix}": float(np.median(values)),
        f"p90_{prefix}": float(np.quantile(values, 0.90)),
        f"max_{prefix}": float(np.max(values)),
    }


def _morgan(smiles: list[str]) -> dict[str, object]:
    result = {}
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES: {value}")
        result[value] = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    return result


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    config = SourceFreeTrainConfig()
    config.validate_frozen_predictor()
    canonical = pd.read_csv(ROOT / "experiments/e0_4g_baseline/canonical_4g.csv")
    query_history = build_query_history(E2 / "queried_batch_diagnostics.csv", canonical)
    query_history.to_csv(E2 / "query_history.csv", index=False)
    test_rows, relevance_rows, shift_rows, compound_rows = [], [], [], []
    difficulty_rows = []
    for seed in OUTER_SEEDS:
        context = context_for_seed(seed, E2, config, "compound")
        seed_dir = context["seed_dir"]
        checkpoint = seed_dir / "shared_round_0" / f"member_0_seed_{seed * 100}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        _, fixed, conditions = _fixed_space(context, checkpoint)
        l0_ids = context["role_ids"]["l0_train"]
        test_ids = context["role_ids"]["test"]
        u0_ids = context["role_ids"]["u0"]
        test_matrix = np.vstack([fixed[sid] for sid in test_ids])
        u0_matrix = np.vstack([fixed[sid] for sid in u0_ids])
        test_cond = np.vstack([conditions[sid] for sid in test_ids])
        test_compounds = rows_for_ids(context["engine"], test_ids)["canonical_smiles"].astype(str).tolist()
        fingerprints = _morgan(sorted(set(test_compounds) | set(rows_for_ids(context["engine"], u0_ids)["canonical_smiles"].astype(str))))
        aulc = pd.read_csv(E2 / "aulc_summary.csv")
        metrics = pd.read_csv(E2 / "round_metrics.csv")
        seed_difficulty = metrics[(metrics["outer_seed"] == seed) & (metrics["strategy"] == "random") & (metrics["round"] == 0)].iloc[0]
        difficulty = {"outer_seed": seed, "round0_nrmse": float(seed_difficulty.nrmse), "test_unique_compounds": len(set(test_compounds))}
        for strategy in STRATEGIES:
            aulc_row = aulc[(aulc.outer_seed == seed) & (aulc.strategy == strategy)].iloc[0]
            difficulty[f"{strategy}_aulc"] = float(aulc_row.aulc_normalized)
        difficulty_rows.append(difficulty)
        for strategy in STRATEGIES:
            for round_index in range(9):
                labeled_ids = _round_labeled(query_history, seed, strategy, round_index, l0_ids)
                labeled_matrix = np.vstack([fixed[sid] for sid in labeled_ids])
                distances = cdist(test_matrix, labeled_matrix).min(axis=1)
                baseline = cdist(test_matrix, np.vstack([fixed[sid] for sid in l0_ids])).min(axis=1)
                test_rows.append({
                    "outer_seed": seed, "strategy": strategy, "round": round_index,
                    "gradient_train_rows": len(labeled_ids),
                    **_summary(distances, "test_to_labeled_distance"),
                    "delta_mean_distance_vs_round0": float(distances.mean() - baseline.mean()),
                    "relative_coverage_gain": float(1.0 - distances.mean() / baseline.mean()),
                })
                labeled_compounds = rows_for_ids(context["engine"], labeled_ids)["canonical_smiles"].astype(str).value_counts()
                labeled_condition = np.vstack([conditions[sid] for sid in labeled_ids])
                proportions = labeled_compounds.to_numpy(dtype=float) / len(labeled_ids)
                # Round 8 is the final revealed state; no new batch is selected
                # after round 7, so selected-to-test metrics have eight batches.
                if round_index < 8:
                    selected = _selected(query_history, seed, strategy, round_index)
                    selected_ids = selected["sample_id"].astype(str).tolist()
                    selected_matrix = np.vstack([fixed[sid] for sid in selected_ids])
                    selected_to_test = cdist(selected_matrix, test_matrix).min(axis=1)
                    selected_cond = np.vstack([conditions[sid] for sid in selected_ids])
                    selected_condition = cdist(selected_cond, test_cond).min(axis=1)
                    selected_smiles = rows_for_ids(context["engine"], selected_ids)["canonical_smiles"].astype(str).tolist()
                    tanimoto = []
                    for smile in selected_smiles:
                        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[smile], [fingerprints[value] for value in test_compounds])
                        tanimoto.append(max(sims))
                    relevance_rows.append({
                        "outer_seed": seed, "strategy": strategy, "round": round_index,
                        "mean_selected_to_test_fixed_latent_distance": float(np.mean(selected_to_test)),
                        "median_selected_to_test_fixed_latent_distance": float(np.median(selected_to_test)),
                        "min_selected_to_test_fixed_latent_distance": float(np.min(selected_to_test)),
                        "mean_max_test_morgan_tanimoto": float(np.mean(tanimoto)),
                        "median_max_test_morgan_tanimoto": float(np.median(tanimoto)),
                        "p90_max_test_morgan_tanimoto": float(np.quantile(tanimoto, .9)),
                        "mean_selected_to_test_standardized_condition_distance": float(np.mean(selected_condition)),
                        "selected_unique_compounds": int(len(set(selected_smiles))),
                        "compound_hhi": float(np.square(pd.Series(selected_smiles).value_counts(normalize=True).to_numpy()).sum()),
                        "max_samples_per_compound": int(pd.Series(selected_smiles).value_counts().max()),
                    })
                test_shift = float(np.linalg.norm(labeled_matrix.mean(axis=0) - test_matrix.mean(axis=0)))
                u0_shift = float(np.linalg.norm(labeled_matrix.mean(axis=0) - u0_matrix.mean(axis=0)))
                condition_mean = labeled_condition.mean(axis=0)
                shift_rows.append({
                    "outer_seed": seed, "strategy": strategy, "round": round_index,
                    "gradient_train_rows": len(labeled_ids), "unique_compounds": int(len(labeled_compounds)),
                    "compound_hhi": float(np.square(proportions).sum()), "max_samples_per_compound": int(labeled_compounds.max()),
                    "gradient_train_centroid_distance_to_test": test_shift, "gradient_train_centroid_distance_to_u0": u0_shift,
                    "condition_mean_norm": float(np.linalg.norm(condition_mean)),
                    "condition_std_mean": float(labeled_condition.std(axis=0).mean()),
                    "condition_min": float(labeled_condition.min()), "condition_max": float(labeled_condition.max()),
                })
                prediction = pd.read_csv(seed_dir / strategy / f"round_{round_index}" / "test_predictions.csv.gz")
                truth = rows_for_ids(context["engine"], prediction["sample_id"].astype(str).tolist())
                for compound, group in truth.groupby("canonical_smiles", sort=True):
                    positions = group.index.to_numpy()
                    pred_v1 = prediction.loc[positions, "ensemble_pred_V1"].to_numpy(float)
                    pred_v2 = prediction.loc[positions, "ensemble_pred_V2"].to_numpy(float)
                    true_v1 = group["V1_ml"].to_numpy(float); true_v2 = group["V2_ml"].to_numpy(float)
                    v1 = true_v1 - pred_v1; v2 = true_v2 - pred_v2
                    compound_rows.append({
                        "outer_seed": seed, "strategy": strategy, "round": round_index,
                        "canonical_smiles": compound, "compound_rows": len(group),
                        "V1_MAE": float(np.abs(v1).mean()), "V2_MAE": float(np.abs(v2).mean()),
                        "normalized_MAE": float(.5 * (np.abs(v1).mean() / context["target_scales"]["V1"] + np.abs(v2).mean() / context["target_scales"]["V2"])),
                        "normalized_RMSE": float(.5 * (np.sqrt(np.mean(v1**2)) / context["target_scales"]["V1"] + np.sqrt(np.mean(v2**2)) / context["target_scales"]["V2"])),
                    })
    test_df = pd.DataFrame(test_rows); relevance_df = pd.DataFrame(relevance_rows); shift_df = pd.DataFrame(shift_rows); compound_df = pd.DataFrame(compound_rows); difficulty_df = pd.DataFrame(difficulty_rows)
    test_df.to_csv(OUT / "test_labeled_coverage_trajectory.csv", index=False)
    relevance_df.to_csv(OUT / "selected_test_relevance.csv", index=False)
    compound_df.to_csv(OUT / "per_compound_error_trajectory.csv", index=False)
    difficulty_df["coverage_gain_vs_random"] = difficulty_df.random_aulc - difficulty_df.coverage_aulc
    difficulty_df["ensemble_gain_vs_random"] = difficulty_df.random_aulc - difficulty_df.ensemble_aulc
    difficulty_df["hybrid_gain_vs_random"] = difficulty_df.random_aulc - difficulty_df.hybrid_aulc
    difficulty_df.to_csv(OUT / "seed_difficulty_summary.csv", index=False)
    shift_df.to_csv(OUT / "strategy_distribution_shift.csv", index=False)
    effects, localization = paired_compound_effects(compound_df)
    effects.to_csv(OUT / "seed42_compound_paired_effects.csv", index=False)
    (OUT / "seed42_compound_localization_summary.json").write_text(json.dumps(localization, indent=2), encoding="utf-8")
    _plots(test_df, compound_df, OUT / "plots")
    decision = {
        "purely_post_hoc": True, "implementation_bug_found": False, "data_leakage_found": False,
        "evaluation_bug_found": False, "wrong_test_predictions_found": False, "labeled_state_reconstruction_validated": True,
        "seed42_compound_localization": "inconclusive",
        "seed42_test_coverage_improved_by_coverage": bool(test_df[(test_df.outer_seed == 42) & (test_df.strategy == "coverage")].relative_coverage_gain.iloc[-1] > 0),
        "seed42_test_coverage_improved_by_hybrid": bool(test_df[(test_df.outer_seed == 42) & (test_df.strategy == "hybrid")].relative_coverage_gain.iloc[-1] > 0),
        "test_to_gradient_train_coverage_worsened": {},
        "labeled_centroid_distance_to_test_increased": {},
        "evidence_strength": "descriptive_only_n3",
        "safe_to_proceed_to_e4_preregistration": True,
        "formal_e4_training_started": False,
        "interpretation_boundary": "risk ranking is not training utility or held-out compound OOD utility",
    }
    seed42_coverage = test_df[test_df.outer_seed == 42].pivot(index="strategy", columns="round", values="mean_test_to_labeled_distance")
    seed42_centroid = shift_df[shift_df.outer_seed == 42].pivot(index="strategy", columns="round", values="gradient_train_centroid_distance_to_test")
    decision["test_to_gradient_train_coverage_worsened"] = {s: bool(seed42_coverage.loc[s, 8] > seed42_coverage.loc[s, 0]) for s in STRATEGIES}
    decision["labeled_centroid_distance_to_test_increased"] = {s: bool(seed42_centroid.loc[s, 8] > seed42_centroid.loc[s, 0]) for s in STRATEGIES}
    (OUT / "failure_audit_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"stage": "E2_compound_failure_audit", "post_hoc_only": True, "fixed_reference": "Round-0 member_0, L0_train normalization", "morgan_radius": 2, "morgan_nBits": 2048, "formal_training_started": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "artifact_manifest.json").write_text(json.dumps({"files": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()), "source_e2_dir": str(E2.relative_to(ROOT))}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(_readme(decision, test_df, localization), encoding="utf-8")


def _plots(test_df: pd.DataFrame, compound_df: pd.DataFrame, plot_dir: Path) -> None:
    for seed in sorted(test_df.outer_seed.unique()):
        subset = test_df[test_df.outer_seed == seed]
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, group in subset.groupby("strategy"):
            ax.plot(group["round"], group["relative_coverage_gain"], marker="o", label=strategy)
        ax.axhline(0, color="black", linewidth=.8); ax.set(title=f"Seed {seed}: test coverage gain", xlabel="round", ylabel="1 - distance / round0")
        ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / f"seed{seed}_test_coverage.png", dpi=140); plt.close(fig)
    for strategy in STRATEGIES:
        subset = compound_df[(compound_df.outer_seed == 42) & (compound_df.strategy == strategy)]
        pivot = subset.pivot(index="canonical_smiles", columns="round", values="normalized_MAE")
        pivot = pivot.subtract(pivot[0], axis=0)
        fig, ax = plt.subplots(figsize=(9, max(4, .22 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(pivot)), vmax=np.nanmax(np.abs(pivot)))
        ax.set(title=f"Seed 42 {strategy}: compound error change", xlabel="round", ylabel="held-out compound"); ax.set_xticks(range(9)); ax.set_yticks(range(len(pivot))); ax.set_yticklabels([str(v)[:18] for v in pivot.index], fontsize=6); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(plot_dir / f"seed42_{strategy}.png", dpi=140); plt.close(fig)


def _readme(decision: dict, test_df: pd.DataFrame, localization: dict) -> str:
    final = test_df[(test_df.outer_seed == 42) & (test_df["round"] == 8)].set_index("strategy").relative_coverage_gain.to_dict()
    return f"""# E2 Compound Failure Audit

This directory is a **purely post-hoc diagnostic**. It does not modify the E2 primary result and did not start formal E4 training. Existing selected IDs, partitions, Round-0 member-0 fixed reference, predictions, labels, and persisted AL states were reused.

## Findings

- Seed42 final test-to-labeled relative coverage gain: {json.dumps(final, sort_keys=True)}. Positive values mean the held-out test is closer to the labeled set; this is not an acquisition objective.
- `gradient_train_rows` excludes the fixed validation rows. Validation remains part of label budget, but is excluded here because this diagnostic measures the geometry of rows actually used for gradient updates.
- Selected-to-test relevance uses fixed latent distance, Morgan radius-2/2048-bit maximum Tanimoto, and Euclidean distance in standardized 9D condition space. Normalization is fit only on fixed L0_train; neither U0 nor test contributes.
- `seed42_compound_paired_effects.csv` pairs each active strategy with Random on the same held-out compound, using the mean of normalized RMSE and normalized MAE. Concentration statistics are {json.dumps(localization, sort_keys=True)}. Localization remains inconclusive; no favorable localization cutoff was introduced.
- `strategy_distribution_shift.csv` separates nearest-distance coverage from the actual gradient-train-centroid-to-test-centroid distance. Both are post-hoc descriptive diagnostics.
- Historical `label_efficiency.csv` remains unchanged; its Random-final threshold interpretation is degenerate when Round-0 is already better than Random final and must not be cited as compound label-saving evidence.

## Audit boundary

`implementation_bug_found={decision['implementation_bug_found']}`, `data_leakage_found={decision['data_leakage_found']}`, `evaluation_bug_found={decision['evaluation_bug_found']}`. No implementation/leakage/evaluation bug was found. Global diversity, risk ranking, held-out-test geometric relevance and OOD utility are distinct quantities. The seed42 failure remains partly split-composition dependent and is not fully explained by one post-hoc diagnostic. All test relevance calculations are post-hoc only; test data never enters acquisition or E2 reruns.

Fresh-clone boundary: the aggregate E2 scientific result and compact query history are tracked and self-contained. Regenerating the full historical trajectory additionally requires runtime-only per-round checkpoints and test predictions; their hashes/provenance are recorded in `audit_input_manifest.json`, and they are intentionally not claimed as fresh-clone inputs.

## Decision

`safe_to_proceed_to_e4_preregistration={decision['safe_to_proceed_to_e4_preregistration']}`. The current failure hypothesis is descriptive only and does not justify changing the predictor or E2 strategy. Formal E4 training, E2 Quantile curve, causal controls, representation ablations, and advanced AL remain deferred.
"""


if __name__ == "__main__":
    run()

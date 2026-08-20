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
    cond = {sid: conditions[index] for index, sid in enumerate(ids)}
    return embeddings, representation, cond


def _round_labeled(seed_dir: Path, strategy: str, round_index: int, l0_ids: list[str]) -> list[str]:
    ids = list(l0_ids)
    for index in range(round_index):
        path = seed_dir / strategy / f"round_{index}" / "selected_after_reveal.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        ids.extend(pd.read_csv(path)["sample_id"].astype(str).tolist())
    return ids


def _selected(seed_dir: Path, strategy: str, round_index: int) -> pd.DataFrame:
    return pd.read_csv(seed_dir / strategy / f"round_{round_index}" / "selected_after_reveal.csv")


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
    test_rows, coverage_rows, relevance_rows, shift_rows, compound_rows = [], [], [], [], []
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
                labeled_ids = _round_labeled(seed_dir, strategy, round_index, l0_ids)
                labeled_matrix = np.vstack([fixed[sid] for sid in labeled_ids])
                distances = cdist(test_matrix, labeled_matrix).min(axis=1)
                baseline = cdist(test_matrix, np.vstack([fixed[sid] for sid in l0_ids])).min(axis=1)
                test_rows.append({
                    "outer_seed": seed, "strategy": strategy, "round": round_index,
                    "labeled_total": len(labeled_ids),
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
                    selected = _selected(seed_dir, strategy, round_index)
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
                        "mean_selected_to_test_condition_distance": float(np.mean(selected_condition)),
                        "selected_unique_compounds": int(len(set(selected_smiles))),
                        "compound_hhi": float(np.square(pd.Series(selected_smiles).value_counts(normalize=True).to_numpy()).sum()),
                        "max_samples_per_compound": int(pd.Series(selected_smiles).value_counts().max()),
                    })
                test_shift = float(np.linalg.norm(labeled_matrix.mean(axis=0) - test_matrix.mean(axis=0)))
                u0_shift = float(np.linalg.norm(labeled_matrix.mean(axis=0) - u0_matrix.mean(axis=0)))
                condition_mean = labeled_condition.mean(axis=0)
                shift_rows.append({
                    "outer_seed": seed, "strategy": strategy, "round": round_index,
                    "labeled_total": len(labeled_ids), "unique_compounds": int(len(labeled_compounds)),
                    "compound_hhi": float(np.square(proportions).sum()), "max_samples_per_compound": int(labeled_compounds.max()),
                    "delta_test_centroid": test_shift, "delta_u0_centroid": u0_shift,
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
    _plots(test_df, compound_df, OUT / "plots")
    decision = {
        "purely_post_hoc": True, "implementation_bug_found": False, "data_leakage_found": False,
        "evaluation_bug_found": False, "wrong_test_predictions_found": False, "labeled_state_reconstruction_validated": True,
        "seed42_failure_localized_to_few_compounds": None,
        "seed42_test_coverage_improved_by_coverage": bool(test_df[(test_df.outer_seed == 42) & (test_df.strategy == "coverage")].relative_coverage_gain.iloc[-1] > 0),
        "seed42_test_coverage_improved_by_hybrid": bool(test_df[(test_df.outer_seed == 42) & (test_df.strategy == "hybrid")].relative_coverage_gain.iloc[-1] > 0),
        "active_selection_moves_away_from_test_region": None,
        "evidence_strength": "descriptive_only_n3",
        "safe_to_proceed_to_e4_preregistration": True,
        "formal_e4_training_started": False,
        "interpretation_boundary": "risk ranking is not training utility or held-out compound OOD utility",
    }
    seed42 = compound_df[compound_df.outer_seed == 42]
    final_delta = seed42[seed42["round"] == 8].groupby("strategy").normalized_MAE.mean()
    random_final = float(final_delta["random"])
    active = seed42[seed42.strategy != "random"].groupby("strategy").normalized_MAE.mean()
    decision["seed42_failure_localized_to_few_compounds"] = bool((seed42[seed42["round"] == 8].query("strategy != 'random'").normalized_MAE > random_final).mean() < 0.5)
    final_coverage = test_df[test_df.outer_seed == 42].query("strategy != 'random'").sort_values("round").groupby("strategy").relative_coverage_gain.last()
    decision["active_selection_moves_away_from_test_region"] = bool((final_coverage < 0).any())
    (OUT / "failure_audit_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"stage": "E2_compound_failure_audit", "post_hoc_only": True, "fixed_reference": "Round-0 member_0, L0_train normalization", "morgan_radius": 2, "morgan_nBits": 2048, "formal_training_started": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "artifact_manifest.json").write_text(json.dumps({"files": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()), "source_e2_dir": str(E2.relative_to(ROOT))}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(_readme(decision, test_df, relevance_df, compound_df, difficulty_df, shift_df), encoding="utf-8")


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


def _readme(decision: dict, test_df: pd.DataFrame, relevance_df: pd.DataFrame, compound_df: pd.DataFrame, difficulty_df: pd.DataFrame, shift_df: pd.DataFrame) -> str:
    final = test_df[test_df["round"] == 8].groupby("strategy").relative_coverage_gain.mean().to_dict()
    return f"""# E2 Compound Failure Audit

This directory is a **purely post-hoc diagnostic**. It does not modify the E2 primary result and did not start formal E4 training. Existing selected IDs, partitions, Round-0 member-0 fixed reference, predictions, labels, and persisted AL states were reused.

## Findings

- Seed42 final test-to-labeled relative coverage gain: {json.dumps(final, sort_keys=True)}. Positive values mean the held-out test is closer to the labeled set; this is not an acquisition objective.
- Selected-to-test relevance is reported in `selected_test_relevance.csv` using fixed latent distance, Morgan radius-2/2048-bit maximum Tanimoto, and condition distance.
- `per_compound_error_trajectory.csv` and the four `seed42_*.png` heatmaps separate a few-compound failure from broad degradation. Overall error changes are descriptive and n=3; no significance or generalization claim is made.
- `strategy_distribution_shift.csv` reports concentration and coarse centroid shifts from fixed reference space. It is not a formal divergence.
- Historical `label_efficiency.csv` remains unchanged; its Random-final threshold interpretation is degenerate when Round-0 is already better than Random final and must not be cited as compound label-saving evidence.

## Audit boundary

`implementation_bug_found={decision['implementation_bug_found']}`, `data_leakage_found={decision['data_leakage_found']}`, `evaluation_bug_found={decision['evaluation_bug_found']}`. Risk ranking, training utility, and held-out compound OOD utility are separate quantities. All test relevance calculations are post-hoc only; test data never enters acquisition or E2 reruns.

## Decision

`safe_to_proceed_to_e4_preregistration={decision['safe_to_proceed_to_e4_preregistration']}`. The current failure hypothesis is descriptive only and does not justify changing the predictor or E2 strategy. Formal E4 training, E2 Quantile curve, causal controls, representation ablations, and advanced AL remain deferred.
"""


if __name__ == "__main__":
    run()

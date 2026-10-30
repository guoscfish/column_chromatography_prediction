"""Join derived diagnostics and calculate seed-level descriptive associations.

No training imports, API calls, target data, label store, or selector execution.
Run --recompute with an environment containing NumPy, pandas, SciPy,
matplotlib and RDKit to repeat the read-only source analyses first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FREE = "free_llm32_scientist_v2"
KEY = ["seed", "method", "round"]


def checked_merge(left, right):
    if left.duplicated(KEY).any() or right.duplicated(KEY).any():
        raise ValueError("Duplicate seed/method/round keys")
    if set(map(tuple, left[KEY].values)) != set(map(tuple, right[KEY].values)):
        raise ValueError("Unmatched seed/method/round keys")
    common = (set(left) & set(right)) - set(KEY)
    aligned = left[KEY + sorted(common)].merge(right[KEY + sorted(common)], on=KEY, suffixes=("_l", "_r"), validate="one_to_one")
    for c in common:
        a, b = aligned[c + "_l"], aligned[c + "_r"]
        if not ((a == b) | (a.isna() & b.isna())).all():
            raise ValueError(f"Conflicting common column {c}")
    return left.merge(right.drop(columns=sorted(common)), on=KEY, validate="one_to_one")


def correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return len(x), None
    return len(x), float(spearmanr(x, y).statistic)


def associations(profiles):
    candidates = [
        "raw_remaining_U_coverage_distance_relative_drop", "raw_selected_distance_percentile_mean",
        "cw_selected_distance_percentile_mean", "raw_batch_nn_less_than_half_L_fraction",
        "raw_centered_batch_effective_rank", "raw_remaining_U_coverage_gain_to_bank_RMS",
        "raw_pool_coverage_mean_to_bank_RMS",
    ]
    # Behavior columns are explicitly selected to avoid fishing over all fields.
    candidates += [c for c in ("unique_molecules", "same_molecule_pairs", "known_molecule_fraction",
                               "pe_curve_extension_fraction", "amount_curve_extension_fraction",
                               "high8_same_molecule_enrichment", "high8_similarity_enrichment") if c in profiles]
    rows = []
    for (seed, method), trajectory in profiles.groupby(["seed", "method"]):
        for phase, g in [("333-1485", trajectory), ("333-653", trajectory[trajectory.labels_before < 653]),
                         ("653-1005", trajectory[trajectory.labels_before.between(653, 1004)]),
                         ("1005-1485", trajectory[trajectory.labels_before >= 1005])]:
            for feature in candidates:
                for adjustment in ("none", "within_six_stage_centering"):
                    x, y = g[feature].astype(float), g["next_fit_validation_gain"].astype(float)
                    if adjustment != "none":
                        x = x - g.groupby("stage")[feature].transform("mean")
                        y = y - g.groupby("stage")["next_fit_validation_gain"].transform("mean")
                    n, rho = correlation(x, y)
                    rows.append(dict(seed=seed, method=method, phase=phase, feature=feature,
                                     adjustment=adjustment, n_batches_descriptive_only=n,
                                     spearman_rho=rho))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "next_fit_associations_by_seed.csv", index=False)
    summary = []
    for keys, g in frame.groupby(["method", "phase", "feature", "adjustment"]):
        r = g.spearman_rho.dropna()
        summary.append(dict(zip(["method", "phase", "feature", "adjustment"], keys),
                            n_seeds=len(r), mean_seed_rho=r.mean(), sample_sd_seed_rho=r.std(ddof=1),
                            positive_seed_count=int((r > 0).sum()), negative_seed_count=int((r < 0).sum())))
    pd.DataFrame(summary).to_csv(HERE / "next_fit_association_summary.csv", index=False)


def assemble():
    geometry = pd.read_csv(HERE / "geometry_profiles.csv")
    behavior = pd.read_csv(HERE / "behavior_profiles.csv")
    profiles = checked_merge(geometry, behavior)
    curves = pd.read_csv(HERE / "round_metrics.csv")
    before = curves[curves["round"] < 36].rename(columns={"validation_nrmse": "validation_nrmse_before"})
    profiles = checked_merge(profiles, before[KEY + ["validation_nrmse_before", "model_provenance_stratum"]])
    after = curves[curves["round"] > 0].copy()
    after["round"] -= 1
    after = after.rename(columns={"validation_nrmse": "validation_nrmse_after", "adjacent_gain_positive_is_improvement": "next_fit_validation_gain"})
    profiles = checked_merge(profiles, after[KEY + ["validation_nrmse_after", "next_fit_validation_gain"]])
    assert len(profiles) == 360
    assert np.allclose(profiles.validation_nrmse_before - profiles.validation_nrmse_after, profiles.next_fit_validation_gain)
    profiles.sort_values(KEY).to_csv(HERE / "selection_profiles.csv", index=False)
    associations(profiles)
    integrity = json.loads((HERE / "runtime_integrity_audit.json").read_text())
    geom = json.loads((HERE / "geometry_availability.json").read_text())
    behavior_audit = json.loads((HERE / "behavior_data_audit.json").read_text())
    audit = {
        "status": "COMPLETE_WITH_EXPLICIT_STRUCTURAL_NA",
        "reporting_only": True,
        "repository_head": integrity["repository_head"],
        "remote_main": integrity["remote_main_checked_in_session"],
        "required_inputs": integrity["required_inputs"],
        "runtime_hash_audit": "runtime_integrity_audit.json",
        "runtime_source_hash_manifest": "runtime_source_hashes.json",
        "geometry_audit": "geometry_availability.json",
        "geometry_contract_robustness_audit": "geometry_contract_robustness_audit.json",
        "behavior_audit": "behavior_data_audit.json",
        "matched_condition_probe": {"audit": "matched_condition_geometry_probe.json", "rows": 72,
                                    "seeds": [157, 887], "labels_before": [333, 653, 973, 1293],
                                    "scope": "representative frozen states; not all five seeds or all rounds",
                                    "new_model_or_gradient_computation": False,
                                    "units_and_perturbation_ranges_not_equalized": True},
        "curves": {"verified_fits": 370, "seed_count": 5, "methods": 2, "points_per_trajectory": 37,
                   "runtime_equals_frozen_snapshot": True, "frozen_original_normalization_retained": True},
        "selections": {"verified_batches": len(profiles), "selected_records": int(profiles.batch_size.sum()),
                       "unique_trajectory_round_keys": int(profiles[KEY].drop_duplicates().shape[0])},
        "gradient_banks": {"raw_free": 180, "raw_lcmd": 180, "cw_free": 180, "cw_lcmd": 0,
                           "checkpoint_hashes_verified_for_each_state": geom["verified_checkpoint_hashes"]},
        "limitations": [
            {"metric": "LCMD CW geometry", "missing_batches": 180,
             "reason": "No frozen CW bank; non-invertible raw CountSketch cannot reconstruct CW; no recomputation performed."},
            {"metric": "LCMD history residual proximity", "missing_batches": 180,
             "reason": "The audited LCMD selection/receipt sources lack per-batch measured feedback payloads; hashes cannot recover values. No full truth table or foreign trajectory substituted."},
            {"metric": "Free r0 historical acquisition residual", "missing_batches": 5,
             "reason": "Initial L333 has no acquisition-time out-of-sample residual; NA, not zero."},
            {"metric": "CW causal contribution", "reason": "Observational exposure and text/behavior associations cannot identify contribution; NoCW experiment required."},
        ],
        "behavior_completeness_details": {k: v for k, v in behavior_audit.items() if k != "source_hashes"},
        "historical_model_version_confounding": "157/6101 retain Astra/high prefixes; pure Sol seed stratum has n=3 and is nonrandom",
        "new_training": 0, "new_llm_calls": 0, "gradient_recomputation": 0,
        "new_label_requests": 0, "test_truth_read": False,
    }
    (HERE / "data_availability_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"Assembled {len(profiles)} batches; {int(profiles.batch_size.sum())} selected records; seed-level associations only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Repeat existing read-only evidence analyses; never train or call a model")
    args = parser.parse_args()
    if args.recompute:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR="/tmp/free_lcmd_diagnostics_mpl")
        for script in ("evidence_audit.py", "learning_curve_analysis.py", "geometry_analysis.py", "selection_behavior_analysis.py", "condition_geometry_probe.py"):
            subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=ROOT, env=env)
    assemble()


if __name__ == "__main__":
    main()

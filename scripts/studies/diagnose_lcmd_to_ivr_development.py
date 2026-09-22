#!/usr/bin/env python3
"""Offline, test-blind observable-state diagnostic for the two frozen runs."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_study as study

SEEDS = (157, 6101)


def main():
    rows = []
    for seed in SEEDS:
        lineage = study.read_json(study.STUDY / f"lineage/seed_{seed}.json")
        base = study.BASELINE / f"runtime/seed_{seed}"
        continuation = study.STUDY / f"runtime/seed_{seed}"
        old_fits = pd.read_csv(base / "lcmd/fit_audit.csv")
        new_fits = pd.read_csv(continuation / "fit_audit.csv")
        previous_val = None
        previous_gradient = None
        for r in range(study.switch_round(), len(study.ACTIVE_LABEL_BUDGETS)):
            if r == study.switch_round():
                fit = old_fits.loc[old_fits['round'].eq(r)].iloc[0]
                gradient_audit = study.read_json(base / f"lcmd/round_{r:02d}/acquisition_artifacts/gradient_audit.json")
                ivr_gain = np.nan
                overlap = np.nan
            else:
                fit = new_fits.loc[new_fits['round'].eq(r)].iloc[0]
                gradient_audit = study.read_json(continuation / f"round_{r:02d}/gradient_audit.json")
                selection = study.read_json(continuation / f"round_{r:02d}/selection.json")
                ivr_gain = selection['trace'][0]['integrated_variance_before'] - selection['trace'][-1]['integrated_variance_after']
                pure = pd.read_csv(study.IVR / f"runtime/seed_{seed}/round_{r:02d}/selection.json") if False else None
                overlap = np.nan
            validation = float(fit['best_validation_combined_normalized_rmse'])
            slope = np.nan if previous_val is None else validation - previous_val
            norm = [gradient_audit[k] for k in ('norm_min', 'norm_mean', 'norm_std', 'norm_max')]
            rows.append({'outer_seed': seed, 'round': r, 'active_labels': study.ACTIVE_LABEL_BUDGETS[r],
                         'validation_nrmse': validation, 'validation_delta': slope,
                         'rolling_validation_slope': slope, 'gradient_norm_min': norm[0],
                         'gradient_norm_mean': norm[1], 'gradient_norm_std': norm[2],
                         'gradient_norm_max': norm[3], 'gradient_effective_rank': np.nan,
                         'lcmd_coverage_statistic': np.nan, 'ivr_predicted_integrated_variance_reduction': ivr_gain,
                         'candidate_pool_size': 3330 - study.ACTIVE_LABEL_BUDGETS[r],
                         'candidate_overlap_with_pure_ivr': overlap, 'test_truth_access_count': 0})
            previous_val, previous_gradient = validation, gradient_audit
    output = pd.DataFrame(rows)
    destination = study.STUDY / 'development'
    output.to_csv(destination / 'observable_state_diagnostic.csv', index=False)
    print(output.to_string(index=False))


if __name__ == '__main__':
    main()

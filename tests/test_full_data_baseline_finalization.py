from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.qgeognn_al.transfer.shared_center_width import (
    ALL_CONTEXT,
    MASS_FLOW_CONTEXT,
    fit_shared_center_width,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/transfer/full_data_baseline_finalization"
BENCHMARK = ROOT / "studies/transfer/filtered_full_data_benchmark"


def _synthetic(seed: int = 20260910):
    rng = np.random.default_rng(seed)
    source = np.sort(rng.uniform(1.0, 20.0, (120, 2)), axis=1)
    ea = rng.uniform(0.0, 1.0, len(source))
    columns = np.repeat(np.arange(3), 40)
    context = np.column_stack([
        np.asarray([8.0, 25.0, 40.0])[columns],
        np.asarray([10.0, 15.0, 30.0])[columns],
        np.asarray([1.5, 2.15, 2.15])[columns],
        np.asarray([13.2, 15.6, 15.6])[columns],
        np.asarray([0.4458, 0.5248, 0.5248])[columns],
    ])
    multiplier = 1.5 + 0.1 * ea + 0.02 * context[:, 0]
    truth = source * multiplier[:, None] + np.asarray([0.5, 1.0])
    return source, truth, ea, context


def test_shared_context_contracts_are_exact_and_no_feature_sweep_is_allowed():
    source, truth, ea, context = _synthetic()
    mass_flow = fit_shared_center_width(source, truth, ea, context[:, :2], MASS_FLOW_CONTEXT, 1.0)
    all_context = fit_shared_center_width(source, truth, ea, context, ALL_CONTEXT, 1.0)
    assert mass_flow.context_names == MASS_FLOW_CONTEXT
    assert mass_flow.free_coefficients == 10
    assert all_context.context_names == ALL_CONTEXT
    assert all_context.free_coefficients == 16
    assert all_context.center_fit.extra_names == ALL_CONTEXT


def test_shared_fit_is_deterministic_finite_and_normalization_is_fit_only():
    source, truth, ea, context = _synthetic()
    first = fit_shared_center_width(source[:80], truth[:80], ea[:80], context[:80], ALL_CONTEXT, 0.1)
    second = fit_shared_center_width(source[:80], truth[:80], ea[:80], context[:80], ALL_CONTEXT, 0.1)
    prediction = first.predict(source[80:], ea[80:], context[80:])
    np.testing.assert_array_equal(prediction, second.predict(source[80:], ea[80:], context[80:]))
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(first.center_fit.extra_means[0], context[:80].mean(axis=0))
    assert not np.allclose(first.center_fit.extra_means[0], context.mean(axis=0))


def test_frozen_focal_population_and_outer_roles_match_parent_exactly():
    if not (STUDY / "split_manifest.csv").exists():
        return
    current = pd.read_csv(STUDY / "split_manifest.csv")
    parent = pd.read_csv(BENCHMARK / "split_manifest.csv")
    keys = ["column", "protocol", "outer_seed", "sample_id", "role"]
    pd.testing.assert_frame_equal(
        current.loc[current.column.isin(["25g", "40g"]), keys].sort_values(keys).reset_index(drop=True),
        parent[keys].sort_values(keys).reset_index(drop=True),
    )


def test_label_accounting_and_shared_donor_exclusion():
    if not (STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv").exists():
        return
    accounting = pd.read_csv(STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")
    assert len(accounting) == 20
    assert (accounting.total_target_supervision_rows ==
            accounting[["8g_donor_rows", "25g_donor_rows", "40g_donor_rows"]].sum(axis=1)).all()
    assert (accounting.donor_outer_test_rows_used == 0).all()
    assert accounting.focal_target_train_rows.min() >= 300


def test_actual_legacy_values_and_fit_only_context_normalization():
    if not (STUDY / "prediction_freeze_manifest.json").exists():
        return
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert protocol["shared"]["context_values"] == {
        "8g": [8.0, 10.0, 1.5, 13.2, 0.4458],
        "25g": [25.0, 15.0, 2.15, 15.6, 0.5248],
        "40g": [40.0, 30.0, 2.15, 15.6, 0.5248],
    }
    accounting = pd.read_csv(STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")
    for _, row in accounting.iterrows():
        path = STUDY / f"runtime/contexts/{row.focal_column}/{row.protocol}/seed_{int(row.seed)}/fit_audit.json"
        audit = json.loads(path.read_text())
        assert audit["contract"]["test_labels_used_for_fit_normalization_or_selection"] == 0
        assert audit["contract"]["donor_outer_test_rows_used"] == 0
        methods = audit["methods"]
        assert methods["MASS_FLOW_SHARED"]["fit"]["context_names"] == list(MASS_FLOW_CONTEXT)
        assert methods["ALL_CONTEXT_SHARED_CENTER_WIDTH"]["fit"]["context_names"] == list(ALL_CONTEXT)
        counts = np.asarray([row[f"{column}_donor_rows"] for column in ("8g", "25g", "40g")])
        values = np.asarray([protocol["shared"]["context_values"][column] for column in ("8g", "25g", "40g")])
        expected = np.average(values, axis=0, weights=counts)
        actual = np.asarray(methods["ALL_CONTEXT_SHARED_CENTER_WIDTH"]["fit"]["center_fit"]["extra_means"])[0]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_source_hash_m3_equivalence_and_prediction_freeze():
    if not (STUDY / "prediction_freeze_manifest.json").exists():
        return
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert hashlib.sha256((ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt").read_bytes()).hexdigest() == protocol["source_checkpoint_sha256"]
    equivalence = json.loads((STUDY / "m3_historical_equivalence.json").read_text())
    assert equivalence["status"] == "PASS"
    assert equivalence["max_abs_prediction_difference"] < 1e-10
    frozen = json.loads((STUDY / "prediction_freeze_manifest.json").read_text())
    assert frozen["status"] == "FROZEN_BEFORE_TEST_EVALUATION"
    assert frozen["contexts"] == 20
    assert frozen["test_truth_evaluation_started"] is False
    for relative, digest in frozen["files"].items():
        assert hashlib.sha256((STUDY / relative).read_bytes()).hexdigest() == digest
    rerun = json.loads((STUDY / "deterministic_rerun.json").read_text())
    assert rerun["status"] == "PASS"
    assert rerun["prediction_freeze_manifest_byte_identical"] is True


def test_reference_metrics_are_reused_and_metric_arithmetic_matches_parent():
    if not (STUDY / "all_metrics.csv").exists():
        return
    summary = pd.read_csv(STUDY / "summary.csv")
    parent = pd.read_csv(BENCHMARK / "summary.csv")
    keys = ["column", "protocol", "method"]
    metrics = ["V1_rmse_mean", "V1_mae_mean", "V1_r2_mean", "V2_rmse_mean", "V2_mae_mean",
               "V2_r2_mean", "combined_normalized_rmse_mean"]
    methods = ["conditional_EA", "scale_only", "paper_style_current_v2"]
    left = summary.loc[summary.method.isin(methods)].set_index(keys).sort_index()
    right = parent.loc[parent.method.isin(methods)].set_index(keys).sort_index()
    np.testing.assert_allclose(left[metrics], right[metrics], atol=1e-12, rtol=0.0)
    decision = json.loads((STUDY / "decision.json").read_text())
    assert decision["prediction_freeze_verified_before_test_evaluation"] is True
    assert decision["active_learning_started"] is False

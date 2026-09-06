import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.transfer.calibration import fit_affine
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.transfer.residual_diagnostics import (
    donor_training_rows, fit_column_calibration, fit_monotone,
)


def test_monotone_affine_recovery_and_linear_tails():
    x = np.column_stack([np.linspace(0., 10., 81), np.linspace(1., 20., 81)])
    y = x * [2., 3.] + [4., -2.]
    model = fit_monotone(x, y, [4., 8.], .1)
    xx = np.column_stack([np.linspace(-10., 30., 401), np.linspace(-20., 60., 401)])
    prediction = model.predict(xx)
    np.testing.assert_allclose(prediction, xx * [2., 3.] + [4., -2.], atol=1e-10)
    assert np.all(np.diff(prediction, axis=0) >= -1e-10)
    np.testing.assert_allclose(np.diff(prediction[:20], n=2, axis=0), 0., atol=1e-10)
    np.testing.assert_allclose(np.diff(prediction[-20:], n=2, axis=0), 0., atol=1e-10)


def test_spline_recovers_monotone_curvature_and_rejects_decreasing_slope():
    x = np.tile(np.linspace(0., 9., 100)[:, None], (1, 2))
    y = 1 + x + 2*np.maximum(x-3, 0) + np.maximum(x-6, 0)
    curved = fit_monotone(x, y, [1., 1.], 0.)
    np.testing.assert_allclose(curved.predict(x), y, atol=1e-8)
    declining = fit_monotone(x, -x, [1., 1.], .1)
    assert np.all(np.diff(declining.predict(x), axis=0) >= -1e-9)
    assert np.all(declining.coefficients[:, 1:] >= 0)
    before = curved.audit()
    curved.predict(np.array([[1e10, -1e10]]))
    assert curved.audit() == before  # Prediction support never refits knots/scalers.


def fixtures():
    rng = np.random.default_rng(19)
    sources = [rng.uniform(1, 20, (30, 2)) for _ in range(3)]
    truth = [x * (i+2) + rng.normal(0, 1, x.shape) for i, x in enumerate(sources)]
    return sources, truth


def test_zero_coupling_exactly_independent_and_nonzero_uses_donors():
    x, y = fixtures()
    for shared in [True, False]:
        model = fit_column_calibration(x, y, [2., 6.25, 10.], [7., 16.], 0., shared=shared)
        for c in range(3):
            np.testing.assert_allclose(model.predict(x[c], c), fit_affine(y[c], x[c], x[c]).prediction, atol=1e-10)
    changed = [y[0], y[1] + 100, y[2] - 50]
    first = fit_column_calibration(x, y, [2., 6.25, 10.], [7., 16.], 1.)
    second = fit_column_calibration(x, changed, [2., 6.25, 10.], [7., 16.], 1.)
    assert np.max(np.abs(first.predict(x[0], 0) - second.predict(x[0], 0))) > .1
    # Matched local control must not acquire information from donor labels.
    a = fit_column_calibration(x, y, [2., 6.25, 10.], [7., 16.], 1., shared=False)
    b = fit_column_calibration(x, changed, [2., 6.25, 10.], [7., 16.], 1., shared=False)
    np.testing.assert_allclose(a.predict(x[0], 0), b.predict(x[0], 0), atol=1e-10)


def test_donor_purge_preserves_focal_train_and_excludes_all_holdout_compounds():
    context = pd.DataFrame([
        ("8g", "a", "test"), ("8g", "b", "validation"), ("8g", "c", "gradient_train"),
        ("25g", "a", "gradient_train"), ("25g", "b", "gradient_train"),
        ("25g", "d", "gradient_train"), ("40g", "e", "validation"),
        ("40g", "f", "gradient_train"),
    ], columns=["column", "canonical_smiles", "role"])
    comp = donor_training_rows(context, "8g", "compound")
    assert set(comp.canonical_smiles) == {"c", "d", "f"}
    row = donor_training_rows(context, "8g", "row")
    assert set(row.canonical_smiles) == {"a", "b", "c", "d", "f"}
    assert row.role.eq("gradient_train").all()


def test_real_frozen_schedules_are_feasible_without_extra_label_ids():
    schedule = pd.read_csv("studies/transfer/cross_column/splits/schedule_manifest.csv")
    for (protocol, _, _), context in schedule.groupby(["protocol", "outer_seed", "planned_budget"]):
        for column in ("8g", "25g", "40g"):
            allowed = donor_training_rows(context, column, protocol)
            assert set(allowed.sample_id) <= set(context.loc[context.role.eq("gradient_train"), "sample_id"])
            focal = context.loc[context.column.eq(column)]
            assert set(allowed.loc[allowed.column.eq(column), "sample_id"]) == set(focal.loc[focal.role.eq("gradient_train"), "sample_id"])
            assert (allowed.groupby("column").canonical_smiles.nunique() >= 2).all()
            if protocol == "compound":
                assert not set(allowed.canonical_smiles) & set(focal.loc[focal.role.isin(["validation", "test"]), "canonical_smiles"])


def test_degenerate_diagnostic_design_stops_without_silent_fallback():
    with pytest.raises(ValueError, match="degenerate source"):
        fit_monotone(np.ones((10, 2)), np.ones((10, 2)), [1., 1.], .1)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_column_calibration([np.ones((10, 2))]*3, [np.ones((10, 2))]*3, [2., 6.25, 10.], [7., 16.], 1.)


def test_completed_predictions_and_budget_ledgers_are_hash_locked():
    study = Path("studies/transfer/residual_diagnostics")
    freeze = json.loads((study/"all_predictions_frozen.json").read_text())
    assert freeze["contexts"] == len(freeze["files"]) == 120
    assert freeze["protocol_sha256"] == sha256_file(study/"protocol.json")
    config = json.loads((study/"protocol.json").read_text())
    for path, digest in {**config["protected_hashes"], **config["implementation_sha256"]}.items():
        assert sha256_file(Path(path)) == digest
    assert sha256_file(Path("NEXT_TRANSFER_MODEL_AUDIT.md")) == config["design_sha256"]
    schedule = pd.read_csv("studies/transfer/cross_column/splits/schedule_manifest.csv")
    for relative, digest in freeze["files"].items():
        done = study/relative
        assert sha256_file(done) == digest
        for name, expected in json.loads(done.read_text())["files"].items():
            assert sha256_file(done.parent/name) == expected
        usage = json.loads((done.parent/"label_usage.json").read_text())
        original = schedule.loc[schedule.protocol.eq(usage["protocol"]) & schedule.outer_seed.eq(usage["seed"]) &
                                schedule.planned_budget.eq(usage["planned_budget"])]
        paid = set(original.loc[original.role.isin(["gradient_train", "validation"]), "sample_id"])
        assert paid == set(usage["portfolio_revealed_ids"])
        assert len(paid) == usage["portfolio_actual_budget"]
        assert usage["actual_budget"] == len(usage["gradient_train_ids"]) + len(usage["validation_ids"])
        expected_train = donor_training_rows(original, usage["column"], usage["protocol"])
        for column, ids in usage["joint_train_ids"].items():
            assert set(ids) == set(expected_train.loc[expected_train.column.eq(column), "sample_id"])
        assert usage["test_labels_used_for_fit_or_selection"] == 0
        assert usage["donor_validation_labels_used"] == 0
        assert usage["lambda_zero_equals_independent_affine"]


def test_complete_metrics_and_stop_decision_do_not_hide_stronger_controls():
    study = Path("studies/transfer/residual_diagnostics")
    metrics = pd.read_csv(study/"all_metrics.csv")
    assert len(metrics) == 960
    key = ["column", "protocol", "seed", "planned_budget", "method"]
    assert not metrics.duplicated(key).any()
    assert (metrics.groupby("method").size() == 120).all()
    names = [f"{target}_{metric}" for target in ["V1", "V2"] for metric in ["r2", "rmse", "mae"]]
    assert np.isfinite(metrics[names + ["normalized_rmse", "combined_normalized_rmse"]]).all().all()
    pairs = pd.read_csv(study/"paired_aulc.csv")
    shared = pairs.loc[pairs.method.eq("shared_column_affine") & pairs.protocol.eq("compound")]
    assert len(shared.loc[shared.reference.eq("affine") & shared.stable_material]) == 3
    assert not shared.loc[shared.reference.eq("scale_only"), "stable_material"].any()
    decision = json.loads((study/"decision.json").read_text())
    assert decision["decision"] == "NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA"
    assert decision["additional_methods_after_test"] == 0
    assert decision["adaptive_readout_tested"] is False

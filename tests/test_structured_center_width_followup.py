from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.qgeognn_al.transfer.controlled_conditional_extension import (
    center_width_inverse,
    center_width_transform,
    fit_scalar_conditional,
    project_physical,
)
from src.qgeognn_al.transfer.structured_center_width import fit_structured_center_width


def _data(seed: int = 20260909):
    rng = np.random.default_rng(seed)
    source = np.sort(rng.uniform(2.0, 25.0, (120, 2)), axis=1)
    ea = rng.uniform(0.0, 1.0, len(source))
    truth = source * (2.0 + 0.3 * ea[:, None]) + np.asarray([1.0, 2.0])
    return source, truth, ea


def _historical_m3(source, truth, ea, mass_ratio=6.25, penalty=0.1):
    center_source, width_source = center_width_transform(source)
    center_truth, width_truth = center_width_transform(truth)
    center_fit = fit_scalar_conditional(
        center_source, center_truth, ea, float(np.std(center_source)), mass_ratio, penalty
    )
    width_fit = fit_scalar_conditional(
        width_source, width_truth, ea, float(np.std(width_source)), mass_ratio, penalty
    )
    center_prediction = center_fit.predict(np.repeat(center_source, 2, axis=1), ea)[:, 0]
    width_prediction = width_fit.predict(np.repeat(width_source, 2, axis=1), ea)[:, 0]
    return project_physical(center_width_inverse(center_prediction, width_prediction))[0]


def test_historical_m3_behavior_is_unchanged():
    source, truth, ea = _data()
    expected = _historical_m3(source, truth, ea)
    actual = fit_structured_center_width(
        source, truth, ea, 6.25, 0.1, "M3_CENTER_WIDTH"
    ).predict(source, ea)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_both_zero_extra_candidates_exactly_match_m3():
    source, truth, ea = _data()
    baseline = fit_structured_center_width(
        source, truth, ea, 6.25, 0.1, "M3_CENTER_WIDTH"
    ).predict(source, ea)
    for method in ("CENTER_MAGNITUDE", "CENTER_CONDITIONED_WIDTH"):
        candidate = fit_structured_center_width(
            source, truth, ea, 6.25, 0.1, method, zero_extra=True
        )
        np.testing.assert_allclose(candidate.predict(source, ea), baseline, atol=1e-10, rtol=0.0)


def test_center_scale_is_fit_only_and_parameter_counts_are_correct():
    source, truth, ea = _data()
    fit_index = np.arange(80)
    validation = source[80:].copy()
    validation[:, :] = validation * 1000.0
    combined = np.vstack([source[fit_index], validation])
    fit = fit_structured_center_width(
        combined[fit_index], truth[fit_index], ea[fit_index], 6.25, 0.1, "CENTER_MAGNITUDE"
    )
    expected = float(np.std(center_width_transform(source[fit_index])[0]))
    assert fit.center_scale == expected
    assert fit.center_scale != float(np.std(center_width_transform(combined)[0]))
    assert fit_structured_center_width(source, truth, ea, 6.25, 0.1, "M3_CENTER_WIDTH").free_coefficients == 6
    assert fit.free_coefficients == 7
    assert fit_structured_center_width(
        source, truth, ea, 6.25, 0.1, "CENTER_CONDITIONED_WIDTH"
    ).free_coefficients == 7


def test_regularization_priors_and_predictions_are_finite_and_deterministic():
    source, truth, ea = _data()
    for method in ("M3_CENTER_WIDTH", "CENTER_MAGNITUDE", "CENTER_CONDITIONED_WIDTH"):
        first = fit_structured_center_width(source, truth, ea, 10.0, 1.0, method)
        second = fit_structured_center_width(source, truth, ea, 10.0, 1.0, method)
        prediction = first.predict(source, ea)
        assert np.isfinite(prediction).all()
        np.testing.assert_array_equal(prediction, second.predict(source, ea))
        assert first.center_fit.audit()["regularization"].startswith("SSE/n")
        assert first.width_fit.audit()["regularization"].startswith("SSE/n")


def test_transform_projection_group_isolation_and_variance_identities():
    source, truth, ea = _data()
    center, width = center_width_transform(source)
    np.testing.assert_allclose(center_width_inverse(center, width), source, atol=1e-14)
    prediction = fit_structured_center_width(
        source, truth, ea, 6.25, 0.1, "CENTER_CONDITIONED_WIDTH"
    ).predict(source, ea)
    assert np.all(prediction[:, 0] >= 0)
    assert np.all(prediction[:, 1] >= prediction[:, 0])
    center_truth, width_truth = center_width_transform(truth)
    center_prediction, width_prediction = center_width_transform(prediction)
    ec = center_prediction[:, 0] - center_truth[:, 0]
    ew = width_prediction[:, 0] - width_truth[:, 0]
    covariance = np.cov(ec, ew, ddof=0)[0, 1]
    np.testing.assert_allclose(
        np.var(prediction[:, 0] - truth[:, 0]), np.var(ec) + 0.25 * np.var(ew) - covariance,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.var(prediction[:, 1] - truth[:, 1]), np.var(ec) + 0.25 * np.var(ew) + covariance,
        atol=1e-12,
    )
    groups = np.repeat(np.arange(20), 6)
    for train, valid in GroupKFold(5).split(groups, groups=groups):
        assert set(groups[train]).isdisjoint(groups[valid])


def test_frozen_identities_and_outer_blind_artifacts():
    root = Path(__file__).resolve().parents[1]
    study = root / "studies/transfer/structured_center_width_followup"
    decision_path = study / "promotion_decision.json"
    if not decision_path.exists():
        return
    decision = json.loads(decision_path.read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    assert decision["outer_test_truth_read"] is False
    assert decision["outer_stage_executed"] is False
    assert protocol["runner_outer_stage"] == "not implemented"
    assert not list(study.glob("outer*"))
    schedule = pd.read_csv(root / protocol["gradient_train_identity_source"])
    assert sorted(schedule.outer_seed.unique()) == sorted(protocol["seeds"])

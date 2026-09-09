from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.qgeognn_al.transfer.conditional_scaling import fit_conditional
from src.qgeognn_al.transfer.controlled_conditional_extension import (
    center_width_inverse,
    center_width_transform,
    endpoint_magnitude,
    fit_conditional_nested,
    fit_scalar_conditional,
    project_physical,
)


def _synthetic():
    rng = np.random.default_rng(2026)
    source = rng.uniform(1.0, 20.0, (120, 2))
    ea = rng.uniform(0.0, 1.0, len(source))
    truth = source * (2.0 + 0.2 * ea[:, None]) + 0.7
    return source, truth, ea


def test_nested_zero_extra_matches_legacy_conditional_exactly():
    source, truth, ea = _synthetic()
    scales = np.asarray([7.8796590346317394, 16.076509553562932])
    for penalty in (0.0, 0.1, 1.0):
        legacy = fit_conditional(source, truth, ea, scales, 6.25, penalty).predict(source, ea)
        extension = fit_conditional_nested(
            source, truth, ea, scales, 6.25, penalty,
            extra=np.zeros((len(source), 2, 1)), extra_names=("extra",),
        )
        corrected = extension.predict(source, ea, np.zeros((len(source), 2, 1)))
        assert extension.coefficients.shape == (2, 4)
        np.testing.assert_allclose(corrected, legacy, atol=1e-10, rtol=0.0)


def test_endpoint_specific_magnitude_and_width_effects_are_finite():
    source, truth, ea = _synthetic()
    scales = np.asarray([7.0, 16.0])
    magnitude = endpoint_magnitude(source, scales)
    assert magnitude.shape == source.shape
    width = np.repeat((source[:, 1] - source[:, 0])[:, None], 2, axis=1)
    for extra in (magnitude[:, :, None], width[:, :, None], np.column_stack([ea, ea])[:, :, None]):
        fit = fit_conditional_nested(source, truth, ea, scales, 2.0, 0.1, extra=extra, extra_names=("x",))
        assert np.isfinite(fit.predict(source, ea, extra)).all()


def test_penalty_matches_sse_over_n_and_conditional_priors():
    source, truth, ea = _synthetic()
    scales = np.asarray([7.0, 16.0])
    penalty = 0.1
    fit = fit_conditional_nested(source, truth, ea, scales, 2.0, penalty)
    u = source / scales
    raw = u * (ea - ea.mean())[:, None]
    standardized = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    for target in range(2):
        design = np.column_stack([u[:, target], np.ones(len(u)), standardized[:, target]])
        labels = truth[:, target] / (2.0 * scales[target])
        expected = np.linalg.solve(
            design.T @ design / len(u) + penalty * np.eye(3),
            design.T @ labels / len(u) + penalty * np.asarray([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(fit.coefficients[target], expected, atol=1e-12, rtol=0.0)


def test_m3_has_six_free_coefficients_and_is_deterministic():
    source, truth, ea = _synthetic()
    center, width = center_width_transform(source)
    center_truth, width_truth = center_width_transform(truth)
    fits = []
    for _ in range(2):
        center_fit = fit_scalar_conditional(center, center_truth, ea, float(center.std()), 2.0, 0.1)
        width_fit = fit_scalar_conditional(width, width_truth, ea, float(width.std()), 2.0, 0.1)
        fits.append(np.r_[center_fit.coefficients[0], width_fit.coefficients[0]])
    assert fits[0].shape == (6,)
    np.testing.assert_array_equal(fits[0], fits[1])


def test_center_width_inverse_and_projection_contract():
    values = np.asarray([[2.0, 5.0], [4.0, 9.0]])
    center, width = center_width_transform(values)
    np.testing.assert_allclose(center_width_inverse(center[:, 0], width[:, 0]), values)
    projected, audit = project_physical(np.asarray([[3.0, 1.0], [-1.0, 2.0]]))
    assert np.all(projected[:, 1] >= projected[:, 0])
    assert np.all(projected[:, 0] >= 0)
    assert audit["projection_frequency"] == 1.0


def test_compound_group_kfold_isolation():
    groups = np.repeat(np.arange(10), 4)
    for train, valid in GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups):
        assert set(groups[train]).isdisjoint(set(groups[valid]))


def test_controlled_study_artifact_is_test_blind():
    root = Path(__file__).resolve().parents[1]
    decision = json.loads((root / "studies/transfer/controlled_lightweight_transfer_audit/candidate_gate_decisions.json").read_text())
    protocol = json.loads((root / "studies/transfer/controlled_lightweight_transfer_audit/protocol.json").read_text())
    assert decision["test_truth_used_for_fit_or_selection"] is False
    assert decision["outer_test_truth_authorized"] is False
    assert protocol["test_truth_used_for_fit_or_selection"] is False
    assert [item["candidate"] for item in decision["sequential_authorization"]] == [
        "M2_MAG", "M2_WIDTH", "M2_LOADING_SOLVENT",
    ]


def test_loading_solvent_is_supported_binary_indicator_and_identities_match_parent():
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "studies/transfer/filtered_full_data_benchmark"
    for column in ("25g", "40g"):
        features = pd.read_csv(benchmark / f"filtered_features_{column}.csv")
        assert set(features["loading solvent"].astype(str)) == {"DCM", "PE"}
        indicator = features["loading solvent"].astype(str).eq("DCM").astype(int)
        assert set(indicator) == {0, 1}
    schedule = pd.read_csv(benchmark / "split_manifest.csv")
    controlled_protocol = json.loads(
        (root / "studies/transfer/controlled_lightweight_transfer_audit/protocol.json").read_text()
    )
    assert sorted(schedule.outer_seed.unique().tolist()) == sorted(controlled_protocol["seeds"])
    assert set(schedule.loc[schedule.role.eq("gradient_train"), "sample_id"]).issubset(
        set(pd.concat([
            pd.read_csv(benchmark / "filtered_features_25g.csv")[["sample_id"]],
            pd.read_csv(benchmark / "filtered_features_40g.csv")[["sample_id"]],
        ]).sample_id)
    )

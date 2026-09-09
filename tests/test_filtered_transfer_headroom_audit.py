"""Contracts for the filtered transfer headroom audit primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.qgeognn_al.transfer.center_width import fit_center_width
from src.qgeognn_al.transfer.varying_coefficient import fit_varying_coefficient


def test_varying_coefficient_recovers_declared_low_capacity_signal() -> None:
    rng = np.random.default_rng(17)
    source = rng.uniform(1.0, 20.0, size=(160, 2))
    ea = rng.uniform(0.0, 1.0, size=len(source))
    magnitude = np.log1p(source).mean(axis=1)
    effects = np.column_stack([ea, magnitude])
    truth = source * (2.0 + 0.35 * ea[:, None] + 0.15 * magnitude[:, None]) + 1.5
    fit = fit_varying_coefficient(source, truth, effects, [10.0, 10.0], 1.0, 0.0,
                                  effect_names=("EA_fraction", "source_magnitude"))
    prediction = fit.predict(source, effects)
    np.testing.assert_allclose(prediction, truth, atol=1e-9)
    assert fit.audit()["free_coefficients"] == 8


def test_varying_coefficient_penalty_and_normalization_are_finite() -> None:
    source = np.column_stack([np.arange(1.0, 31.0), np.arange(2.0, 62.0, 2.0)])
    effects = np.column_stack([np.linspace(0.0, 1.0, len(source)), np.log1p(source).mean(axis=1)])
    truth = source * 2.0
    fit = fit_varying_coefficient(source, truth, effects, [10.0, 20.0], 2.0, 1.0,
                                  effect_names=("EA_fraction", "source_magnitude"))
    assert np.isfinite(fit.predict(source, effects)).all()
    with pytest.raises(ValueError):
        fit_varying_coefficient(source, truth, effects, [0.0, 20.0], 2.0, 1.0,
                                effect_names=("EA_fraction", "source_magnitude"))


def test_center_width_projection_enforces_physical_order() -> None:
    rng = np.random.default_rng(22)
    source = rng.uniform(2.0, 20.0, size=(100, 2))
    source.sort(axis=1)
    ea = rng.uniform(0.0, 1.0, size=len(source))
    truth = np.column_stack([source.mean(axis=1) * 2.0, source.mean(axis=1) * 2.0 + 1.0])
    fit = fit_center_width(source, truth, ea, 2.0, 0.1)
    prediction, audit = fit.predict(source, ea)
    assert np.isfinite(prediction).all()
    assert np.all(prediction[:, 0] >= 0.0)
    assert np.all(prediction[:, 1] >= prediction[:, 0])
    assert 0.0 <= audit["projection_frequency"] <= 1.0


def test_compound_group_kfold_has_no_group_overlap() -> None:
    groups = np.repeat(np.arange(12), 3)
    for train, valid in GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups):
        assert not set(groups[train]) & set(groups[valid])


def test_frozen_protocol_and_gate_are_test_blind() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "studies/transfer/filtered_transfer_headroom_audit/protocol.json").read_text())
    decision = json.loads((root / "studies/transfer/filtered_transfer_headroom_audit/promotion_decision.json").read_text())
    assert protocol["test_truth_used_for_fit_or_selection"] is False
    assert protocol["promotion_gate"]["selection_scope"] == "gradient_train_only_nested_groupkfold"
    assert decision["test_truth_used_for_fit_or_selection"] is False
    assert decision["outer_test_truth_authorized"] is False
    assert decision["promoted"] == []

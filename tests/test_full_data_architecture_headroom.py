from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.qgeognn_al.transfer.architecture_headroom import (
    TinyLatentAdapter, fit_hierarchical_center_width, fit_hierarchical_scalar,
    fit_latent_ridge, tiny_adapter_parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/transfer/full_data_architecture_headroom"
BASELINE = ROOT / "studies/transfer/full_data_baseline_finalization"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"


def _synthetic(seed=20260910):
    rng = np.random.default_rng(seed)
    columns = np.repeat(["25g", "40g"], 80)
    source = np.sort(rng.uniform(1, 15, (160, 2)), axis=1)
    ea = rng.uniform(0, 1, len(source))
    ratio = np.where(columns == "25g", 6.25, 10.0)
    center = source.mean(1) * ratio * (1.1 + 0.08 * ea + np.where(columns == "25g", .02, -.02))
    width = (source[:, 1] - source[:, 0]) * ratio * (.9 + .04 * ea)
    truth = np.column_stack([center - width / 2, center + width / 2])
    return source, truth, ea, columns


def test_hierarchical_mass_anchor_and_zero_deviation_reduction():
    source, truth, ea, columns = _synthetic()
    fit = fit_hierarchical_center_width(source, truth, ea, columns, ("25g", "40g"),
                                        mass_ratios={"25g": 6.25, "40g": 10.0},
                                        base_penalty=.1, delta_penalty=1e20)
    assert fit.mass_ratios == {"25g": 6.25, "40g": 10.0}
    assert np.max(np.abs(fit.center.deviations)) < 1e-12
    assert np.max(np.abs(fit.width.deviations)) < 1e-12
    assert np.isfinite(fit.predict(source, ea, columns)).all()


def test_hierarchical_scalar_exactly_uses_declared_mass_normalized_target():
    source, truth, ea, columns = _synthetic()
    center = truth.mean(1)
    ratios = np.where(columns == "25g", 6.25, 10.0)
    fit = fit_hierarchical_scalar(source.mean(1), center / ratios, ea, columns,
                                  ("25g", "40g"), base_penalty=0, delta_penalty=.1)
    prediction = fit.predict(source.mean(1), ea, columns) * ratios
    assert np.sqrt(np.mean((prediction - center) ** 2)) < 0.1


def test_ridge_normalization_is_fit_only_and_deterministic():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(40, 128)); residual = latent[:, :2] + .01
    first = fit_latent_ridge(latent[:30], residual[:30], 1.0)
    second = fit_latent_ridge(latent[:30], residual[:30], 1.0)
    np.testing.assert_array_equal(first.predict_residual(latent[30:]), second.predict_residual(latent[30:]))
    np.testing.assert_allclose(first.feature_mean, latent[:30].mean(0))
    assert not np.allclose(first.feature_mean, latent.mean(0))


def test_tiny_adapter_is_zero_equivalent_and_below_parameter_cap():
    torch.manual_seed(42)
    adapter = TinyLatentAdapter(128)
    output = adapter(torch.randn(17, 128))
    assert torch.count_nonzero(output).item() == 0
    assert tiny_adapter_parameter_count(128) == 2098
    assert tiny_adapter_parameter_count(128) < 5000


def test_inherited_full_ids_source_hash_and_latent_audit():
    if not (STUDY / "protocol.json").exists():
        return
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    current = pd.read_csv(STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")
    parent = pd.read_csv(BASELINE / "FULL_DATA_LABEL_ACCOUNTING.csv")
    pd.testing.assert_frame_equal(current, parent)
    split_columns = ["column", "protocol", "outer_seed", "sample_id", "role"]
    current_split = pd.read_csv(STUDY / "split_manifest.csv")[split_columns].sort_values(split_columns).reset_index(drop=True)
    parent_split = pd.read_csv(BASELINE / "split_manifest.csv")[split_columns].sort_values(split_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(current_split, parent_split)
    audit = json.loads((STUDY / "latent_representation_audit.json").read_text())
    assert audit["dimension"] == 128
    assert audit["condition_information_fused"] is True
    assert audit["relative_to_pooling"] == "after graph pooling"
    assert audit["relative_to_prediction_head"].startswith("immediately before")


def test_inner_cv_and_prediction_freeze_contracts():
    if not (STUDY / "prediction_freeze_manifest.json").exists():
        return
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert protocol["inner_cv"]["group"] == "canonical_smiles"
    assert protocol["outer_test_used_for_selection"] is False
    frozen = json.loads((STUDY / "prediction_freeze_manifest.json").read_text())
    assert frozen["status"] == "FROZEN_BEFORE_TEST_EVALUATION"
    assert frozen["contexts"] == 20
    for relative, digest in frozen["files"].items():
        assert hashlib.sha256((STUDY / relative).read_bytes()).hexdigest() == digest
    rerun = json.loads((STUDY / "deterministic_rerun.json").read_text())
    assert rerun["status"] == "PASS"
    assert rerun["prediction_freeze_manifest_byte_identical"] is True


def test_actual_contexts_are_isolated_finite_and_deterministic():
    if not (STUDY / "prediction_freeze_manifest.json").exists():
        return
    for path in STUDY.glob("runtime/contexts/*/*/seed_*/fit_audit.json"):
        audit = json.loads(path.read_text())
        assert audit["contract"]["test_labels_used_for_fit_normalization_or_selection"] == 0
        assert audit["contract"]["outer_validation_used_for_selection"] is False
        assert audit["methods"]["M3_CENTER_WIDTH_FULL"]["equivalence_max_abs_difference"] < 1e-10
        assert audit["methods"]["M3_LATENT_TINY_ADAPTER"]["qgeognn_trainable_parameters"] == 0
        assert audit["methods"]["M3_LATENT_TINY_ADAPTER"]["trainable_parameters"] < 5000
        frame = pd.read_csv(path.with_name("predictions_blind.csv.gz"))
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()


def test_scoring_only_exists_after_freeze_and_decision_is_closed_set():
    if not (STUDY / "decision.json").exists():
        return
    assert (STUDY / "prediction_freeze_manifest.json").exists()
    decision = json.loads((STUDY / "decision.json").read_text())
    assert decision["status"] in {
        "HIERARCHICAL_CENTER_WIDTH_PROMOTED", "LATENT_RESIDUAL_RIDGE_PROMOTED",
        "LATENT_TINY_ADAPTER_PROMOTED", "M3_CENTER_WIDTH_FULL_RETAINED_AFTER_ARCHITECTURE_HEADROOM_STUDY",
    }
    assert decision["prediction_freeze_verified_before_test_evaluation"] is True
    assert decision["active_learning_started"] is False

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.qgeognn_al.transfer.joint_hierarchical_latent import (
    endpoint_scales, fit_direct_latent, fit_hierarchical_cw_v2, fit_joint_hier_latent,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/transfer/full_data_joint_hierarchical_latent"
PARENT = ROOT / "studies/transfer/full_data_baseline_finalization"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"


def _synthetic(seed=42):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["25g", "40g"], 45)
    source = np.sort(rng.uniform(1, 15, (90, 2)), axis=1)
    ea = rng.uniform(0, 1, len(source))
    latent = rng.normal(size=(len(source), 128))
    ratio = np.where(labels == "25g", 6.25, 10.0)
    center = ratio * (1.05 * source.mean(1) + .1 * source.mean(1) * (ea - ea.mean()))
    width = ratio * (.85 * np.diff(source, axis=1)[:, 0] + .05 * latent[:, 0])
    truth = np.column_stack([center - width / 2, center + width / 2])
    return source, truth, ea, latent, labels


def test_separate_center_width_lambda_and_endpoint_scales_are_exact():
    source, truth, ea, _, labels = _synthetic()
    scales = endpoint_scales(truth[:70])
    fit = fit_hierarchical_cw_v2(source[:70], truth[:70], ea[:70], labels[:70], ("25g", "40g"),
                                 {"25g": 6.25, "40g": 10.0}, .1, 100, scales=scales)
    assert fit.center_lambda == .1 and fit.width_lambda == 100
    np.testing.assert_array_equal(fit.scales, scales)
    assert np.isfinite(fit.predict(source[70:], ea[70:], labels[70:])).all()


def test_joint_beta_zero_is_exact_hierarchical_reduction():
    source, truth, ea, latent, labels = _synthetic()
    kwargs = dict(columns=("25g", "40g"), mass_ratios={"25g": 6.25, "40g": 10.0})
    hier = fit_hierarchical_cw_v2(source, truth, ea, labels, center_lambda=1, width_lambda=10, **kwargs)
    joint = fit_joint_hier_latent(source, truth, ea, latent, labels, delta_lambda_center=1,
                                  delta_lambda_width=10, latent_lambda_center=None,
                                  latent_lambda_width=None, **kwargs)
    np.testing.assert_allclose(hier.predict_raw(source, ea, labels),
                               joint.predict_raw(source, ea, latent, labels), atol=1e-12, rtol=0)


def test_direct_is_single_stage_deterministic_and_uses_no_anchor_prediction():
    source, truth, ea, latent, labels = _synthetic()
    args = (source[:70], truth[:70], ea[:70], latent[:70], labels[:70], {"25g": 6.25, "40g": 10.0})
    first = fit_direct_latent(*args, kind="ridge", strength=10)
    second = fit_direct_latent(*args, kind="ridge", strength=10)
    np.testing.assert_array_equal(first.predict(source[70:], ea[70:], latent[70:], labels[70:]),
                                  second.predict(source[70:], ea[70:], latent[70:], labels[70:]))
    assert first.feature_mean.shape == (131,)  # C_source, W_source, EA, 128D h


def test_pca_is_fitted_only_on_supplied_training_latent():
    source, truth, ea, latent, labels = _synthetic()
    fit = fit_joint_hier_latent(source[:70], truth[:70], ea[:70], latent[:70], labels[:70],
        ("25g", "40g"), {"25g": 6.25, "40g": 10.0}, delta_lambda_center=1,
        delta_lambda_width=1, latent_lambda_center=100, latent_lambda_width=100, pca_components=16)
    np.testing.assert_allclose(fit.pca.mean_, latent[:70].mean(0), atol=1e-12, rtol=0)
    assert not np.allclose(fit.pca.mean_, latent.mean(0))


def test_actual_population_splits_source_and_freeze_contracts():
    if not (STUDY / "split_manifest.csv").exists():
        return
    keys = ["column", "protocol", "outer_seed", "sample_id", "role"]
    current = pd.read_csv(STUDY / "split_manifest.csv")[keys].sort_values(keys).reset_index(drop=True)
    parent = pd.read_csv(PARENT / "split_manifest.csv")[keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(current, parent)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    manifest = json.loads((STUDY / "prediction_freeze_manifest.json").read_text())
    assert manifest["status"] == "FROZEN_BEFORE_TEST_EVALUATION" and manifest["contexts"] == 20
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((STUDY / relative).read_bytes()).hexdigest() == digest
    for path in STUDY.glob("runtime/contexts/*/compound/seed_*/fit_audit.json"):
        audit = json.loads(path.read_text())
        assert audit["test_truth_used"] is False and audit["compound_group_leakage"] == 0


def test_declared_grids_methods_and_closed_decision_state():
    if not (STUDY / "decision.json").exists():
        return
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if "delta_grid" not in protocol:  # incomplete interrupted artifact from before this study implementation
        return
    assert protocol["delta_grid"] == [.1, 1, 10, 100]
    assert protocol["latent_grid"] == [1, 10, 100, 1000]
    assert protocol["ridge_grid"] == [.1, 1, 10, 100, 1000]
    assert protocol["pls_grid"] == [2, 4, 8, 16]
    decision = json.loads((STUDY / "decision.json").read_text())
    assert decision["status"] in {"JOINT_HIERARCHICAL_LATENT_PROMOTED", "HIER_CW_V2_PROMOTED",
                                  "CONTROLLED_SHALLOW_ADAPTATION_PROMOTED", "HIER_CW_25G_40G_RETAINED"}
    assert decision["active_learning_started"] is False

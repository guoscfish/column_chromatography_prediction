import inspect

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.fused_features import (
    FUSED_DIMENSION,
    fuse_gradient_latent_features,
    representation_diagnostics,
)
from src.qgeognn_al.active_learning_v2.maxdet import conditional_gradient_maxdet
from src.qgeognn_al.active_learning_v2.cache import array_hash
from src.qgeognn_al.active_learning_v2.fused_maxdet_runner import _cache_features, _load_feature_cache


def test_fusion_shape_block_normalization_and_kernel_equivalence():
    rng = np.random.default_rng(12)
    gradient = rng.normal(size=(9, 512))
    latent = rng.normal(size=(9, 128))
    result = fuse_gradient_latent_features(gradient, latent, 4)
    assert result.features.shape == (9, FUSED_DIMENSION)
    assert result.audit["gradient_labeled_mean_norm_sq"] == pytest.approx(1.0)
    assert result.audit["latent_labeled_mean_norm_sq"] == pytest.approx(1.0)
    assert result.audit["fused_labeled_mean_norm_sq"] == pytest.approx(1.0)
    g = gradient / result.gradient_scale
    h = latent / result.latent_scale
    expected = 0.5 * (g @ g.T) + 0.5 * (h @ h.T)
    np.testing.assert_allclose(result.features @ result.features.T, expected, rtol=2e-6, atol=2e-6)


def test_labeled_only_scales_and_deterministic_selection():
    rng = np.random.default_rng(8)
    gradient = rng.normal(size=(10, 512))
    latent = rng.normal(size=(10, 128))
    first = fuse_gradient_latent_features(gradient, latent, 4)
    changed = gradient.copy(); changed[4:] *= 17
    changed_latent = latent.copy(); changed_latent[4:] *= 19
    second = fuse_gradient_latent_features(changed, changed_latent, 4)
    assert second.gradient_scale == pytest.approx(first.gradient_scale)
    assert second.latent_scale == pytest.approx(first.latent_scale)
    one = conditional_gradient_maxdet(first.features, np.arange(4), np.arange(4, 10), 3)
    two = conditional_gradient_maxdet(first.features.copy(), np.arange(4), np.arange(4, 10), 3)
    assert np.array_equal(one.selected_positions, two.selected_positions)


def test_zero_block_energy_and_invalid_dimensions_fail_loudly():
    with pytest.raises(ValueError, match="gradient block"):
        fuse_gradient_latent_features(np.zeros((3, 512)), np.ones((3, 128)), 2)
    with pytest.raises(ValueError, match="latent block"):
        fuse_gradient_latent_features(np.ones((3, 512)), np.zeros((3, 128)), 2)
    with pytest.raises(ValueError, match="expected 512D"):
        fuse_gradient_latent_features(np.ones((3, 8)), np.ones((3, 128)), 2)


def test_feature_constructor_and_selector_do_not_accept_truth_inputs():
    assert not set(inspect.signature(fuse_gradient_latent_features).parameters) & {"truth", "labels", "targets", "test_truth", "validation_truth"}
    assert not set(inspect.signature(conditional_gradient_maxdet).parameters) & {"truth", "labels", "targets", "test_truth", "pool_truth"}


def test_feature_cache_resume_validates_contract_content_and_order(tmp_path):
    features = np.arange(24, dtype=np.float32).reshape(4, 6)
    indices = np.array([8, 3, 5, 1])
    base = {"checkpoint_sha256": "model-a", "alpha": 0.5}
    contract = {**base, "feature_sha256": array_hash(features)}
    path = tmp_path / "features.npz"
    _cache_features(path, contract, features, indices)

    loaded = _load_feature_cache(path, base, indices, 6)
    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], features)
    assert loaded[1] == contract

    with pytest.raises(RuntimeError, match="incompatible"):
        _load_feature_cache(path, {**base, "alpha": 0.25}, indices, 6)
    with pytest.raises(RuntimeError, match="order/shape"):
        _load_feature_cache(path, base, indices[::-1], 6)


def test_representation_diagnostics_feature_spectrum_matches_gram_definition():
    rng = np.random.default_rng(91)
    features = rng.normal(size=(17, 7))
    result = representation_diagnostics(features)

    gram = features @ features.T
    eigenvalues = np.maximum(np.linalg.eigvalsh((gram + gram.T) / 2.0), 0.0)
    probabilities = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    expected_rank = np.exp(-np.sum(probabilities * np.log(probabilities)))
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    kernel = normalized @ normalized.T
    expected_corr = np.abs(kernel[~np.eye(len(features), dtype=bool)]).mean()

    assert result["effective_rank"] == pytest.approx(expected_rank, rel=1e-12)
    assert result["pairwise_kernel_abs_corr"] == pytest.approx(expected_corr, rel=1e-12)

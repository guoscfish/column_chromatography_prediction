import numpy as np
import pytest
from scipy.linalg import cho_solve

from src.qgeognn_al.active_learning_v2.epig_models import (
    BayesianLinearSurrogate, GaussianProcessSurrogate, gaussian_probability_metrics,
    qualify_model,
)


def data(seed=37, n=32):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = np.column_stack([2 * np.sin(x[:, 0]) + .2 * rng.normal(size=n),
                         3 * np.sin(x[:, 0]) + .4 * rng.normal(size=n)])
    return x, y


@pytest.mark.parametrize("kernel", ["rbf", "matern32"])
def test_exact_gp_matches_independent_dense_conditioning(kernel):
    x, y = data()
    model = GaussianProcessSurrogate(kernel=kernel, maxiter=15).fit(x[:22], y[:22], output_scales=[2, 3])
    posterior = model.posterior(x[22:], row_ids=np.arange(100, 110))
    tx = model.transform_.transform(x[22:])
    for output, fitted in enumerate(model.models_):
        kstar = model._kernel(tx, model.train_x_, fitted["theta"])
        direct = model._kernel(tx, tx, fitted["theta"]) - kstar @ cho_solve(fitted["factor"], kstar.T)
        physical = direct * model.scales_[output] ** 2
        got = posterior.cov(np.arange(10), np.arange(10))[output::2, output::2]
        np.testing.assert_allclose(got, physical, atol=1e-12)
    dense = posterior.cov(np.arange(10), np.arange(10), noisy=True)
    assert np.linalg.eigvalsh(dense).min() > 0
    np.testing.assert_allclose(dense[::2, 1::2], 0)
    np.testing.assert_allclose(posterior.latent_diag(np.arange(10)),
                               np.array([dense[2*i:2*i+2, 2*i:2*i+2] - posterior.noise_cov for i in range(10)]), atol=1e-12)
    assert posterior.row_ids.tolist() == list(range(100, 110))


@pytest.mark.parametrize("factory", [GaussianProcessSurrogate, BayesianLinearSurrogate])
def test_physical_scaling_preserves_joint_model(factory):
    x, y = data(n=25)
    first = factory(maxiter=15).fit(x[:18], y[:18], output_scales=[2, 3]).posterior(x[18:])
    scale = np.array([7, .2])
    second = factory(maxiter=15).fit(x[:18], y[:18] * scale, output_scales=np.array([2, 3]) * scale).posterior(x[18:])
    np.testing.assert_allclose(second.mean, first.mean * scale, rtol=1e-5, atol=1e-6)
    factors = np.tile(scale, len(first.x))
    np.testing.assert_allclose(second.cov(np.arange(7), np.arange(7), noisy=True),
                               first.cov(np.arange(7), np.arange(7), noisy=True) * np.outer(factors, factors), rtol=1e-5, atol=1e-6)


def test_matrix_normal_blr_has_real_cross_output_dependence():
    x, y = data(n=40)
    model = BayesianLinearSurrogate(coregionalized=True, maxiter=30).fit(x, y)
    posterior = model.posterior(x[:6])
    dense = posterior.cov(np.arange(6), np.arange(6), noisy=True)
    assert np.abs(dense[0, 3]) > 1e-7
    assert np.abs(posterior.noise_cov[0, 1]) > 1e-7
    assert np.linalg.eigvalsh(dense).min() > 0
    independent = BayesianLinearSurrogate(coregionalized=False).fit(x, y).posterior(x[:6])
    np.testing.assert_allclose(independent.cov(np.arange(6), np.arange(6))[::2, 1::2], 0)
    np.testing.assert_allclose(posterior.latent_block(1, 2), posterior.latent_block(2, 1).T)


def test_coregionalized_blr_matches_direct_dense_multioutput_gaussian():
    x, y = data(n=18)
    model = BayesianLinearSurrogate(coregionalized=True, maxiter=30).fit(x[:12], y[:12], output_scales=[2, 3])
    p = model.posterior(x[12:])
    train = model._phi(model.transform_.transform(x[:12]))
    test = model._phi(model.transform_.transform(x[12:]))
    prior_variance = 1 / model.audit_["prior_precision"]
    yy = np.kron(train @ train.T * prior_variance, model.physical_B_) + np.kron(np.eye(12), model.noise_cov_)
    ty = np.kron(test @ train.T * prior_variance, model.physical_B_)
    tt = np.kron(test @ test.T * prior_variance, model.physical_B_)
    expected_covariance = tt - ty @ np.linalg.solve(yy, ty.T)
    expected_mean = ty @ np.linalg.solve(yy, (y[:12] - model.center_ * model.scales_).ravel())
    expected_mean = expected_mean.reshape(6, 2) + model.center_ * model.scales_
    np.testing.assert_allclose(p.cov(np.arange(6), np.arange(6)), expected_covariance, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(p.mean, expected_mean, rtol=1e-9, atol=1e-10)


def test_base_mean_and_explicit_noise_semantics():
    x, y = data()
    base = y + 100
    model = BayesianLinearSurrogate(maxiter=15).fit(x[:24], y[:24], mean_train=base[:24], output_scales=[2, 3])
    p = model.posterior(x[24:], mean=base[24:])
    np.testing.assert_allclose(p.mean, y[24:], atol=1e-10)
    a, b = np.array([1, 3]), np.array([3, 2])
    diff = p.cov(a, b, noisy=True) - p.cov(a, b)
    np.testing.assert_allclose(diff[2:4, :2], p.noise_cov)
    np.testing.assert_allclose(diff[:2], 0)
    assert np.all(np.diag(p.noise_cov) > 0)


def test_oof_train_positions_disjoint_and_finite_joint_scores():
    x, y = data(n=30)
    result = qualify_model(lambda: BayesianLinearSurrogate(maxiter=20), x, y, n_splits=3, seed=10)
    seen = []
    for fold in result["folds"]:
        assert not set(fold["train_positions"]) & set(fold["held_out_positions"])
        assert fold["minimum_predictive_eigenvalue"] > 0
        assert np.isfinite(fold["full_joint_nll_per_row"])
        assert fold["full_joint_nll_per_row"] == pytest.approx(2 * fold["full_joint_nll_per_scalar"])
        assert np.isfinite(fold["diagonal_nll_per_scalar"])
        assert fold["whitened_residual_rms"] >= 0
        seen.extend(fold["held_out_positions"])
    assert sorted(seen) == list(range(30))
    assert np.isfinite(result["metrics"]["joint_nll"])
    assert result["oof_mean"].shape == y.shape


def test_gaussian_probability_metrics_known_value_and_invalid_covariance():
    y = np.zeros((8, 2))
    metrics = gaussian_probability_metrics(y, y, np.tile(np.eye(2), (8, 1, 1)))
    assert metrics["joint_nll"] == pytest.approx(np.log(2 * np.pi))
    assert metrics["coverage80"] == [1, 1]
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_probability_metrics(y, y, np.zeros((8, 2, 2)))


def test_validation_rejects_bad_inputs():
    x, y = data()
    with pytest.raises(ValueError):
        GaussianProcessSurrogate(kernel="invented")
    with pytest.raises(ValueError):
        GaussianProcessSurrogate().fit(x, y, output_scales=[0, 1])
    p = BayesianLinearSurrogate().fit(x, y).posterior(x[:3])
    with pytest.raises(ValueError):
        p.cov(np.array([-1]), np.array([0]))
    with pytest.raises(ValueError):
        BayesianLinearSurrogate().fit(x, y).posterior(x[:3], row_ids=[1, 1, 2])


def test_oof_fold_resume_skips_completed_fits_and_audits_interrupted_attempt(tmp_path):
    x, y = data(n=18)
    calls = []
    def interrupted_factory():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interrupted fold")
        return BayesianLinearSurrogate(maxiter=10)
    contract = {"family": "blr", "maxiter": 10}
    with pytest.raises(RuntimeError, match="simulated"):
        qualify_model(interrupted_factory, x, y, cache_directory=tmp_path, cache_contract=contract)
    assert (tmp_path / "fold_0_complete.json").exists()
    assert not (tmp_path / "fold_1_complete.json").exists()
    result = qualify_model(interrupted_factory, x, y, cache_directory=tmp_path, cache_contract=contract)
    assert len(calls) == 4  # 3 actual successful folds plus the failed attempt.
    assert result["fit_attempt_count"] == 4
    def must_not_fit():
        raise AssertionError("completed OOF fold was refit")
    replay = qualify_model(must_not_fit, x, y, cache_directory=tmp_path, cache_contract=contract)
    np.testing.assert_array_equal(replay["oof_mean"], result["oof_mean"])
    np.testing.assert_array_equal(replay["oof_covariance_blocks"], result["oof_covariance_blocks"])
    assert replay["metrics"] == result["metrics"]


def test_oof_cache_rejects_input_model_and_saved_output_drift(tmp_path):
    x, y = data(n=18)
    factory = lambda: BayesianLinearSurrogate(maxiter=10)
    contract = {"family": "blr", "maxiter": 10}
    qualify_model(factory, x, y, cache_directory=tmp_path, cache_contract=contract)
    with pytest.raises(RuntimeError, match="contract drift"):
        qualify_model(factory, x + .01, y, cache_directory=tmp_path, cache_contract=contract)
    with pytest.raises(RuntimeError, match="contract drift"):
        qualify_model(factory, x, y, cache_directory=tmp_path, cache_contract={"family": "other"})
    path = tmp_path / "fold_0.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="cache drift"):
        qualify_model(factory, x, y, cache_directory=tmp_path, cache_contract=contract)
    with pytest.raises(ValueError, match="explicit"):
        qualify_model(factory, x, y, cache_directory=tmp_path)

"""Synthetic-only tests; no repository source targets are opened."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.epig import score_gaussian_acquisition, select_gaussian_batch
from src.qgeognn_al.active_learning_v2.epig_models import BayesianLinearSurrogate, qualify_model


SPEC = importlib.util.spec_from_file_location("epig_stage2_nested_test", Path(__file__).with_name("nested_calibration.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONTRACT = {"test_only": "synthetic", "model_options": {"max_components": 3, "maxiter": 15, "restarts": 1}}


def synthetic(n=30):
    rng = np.random.default_rng(673)
    x = rng.normal(size=(n, 4))
    y = np.column_stack([2*x[:, 0] + rng.normal(size=n),
                         3*x[:, 0] + .4*x[:, 1] + rng.normal(size=n)])
    return x, y


@pytest.mark.parametrize("family", MODULE.FAMILIES)
def test_all_registered_families_produce_valid_two_output_posterior(family):
    x, y = synthetic()
    model = MODULE.model_factory(family, CONTRACT).fit(x[:20], y[:20])
    posterior = MODULE.CalibratedPosterior(model.posterior(x[20:]), [.7, 1.2])
    assert posterior.mean.shape == (10, 2)
    assert np.linalg.eigvalsh(posterior.cov(np.arange(10), np.arange(10), noisy=True)).min() > 0


def test_predefined_temperatures_match_analytic_formulas():
    y = np.array([[1, 2], [-1, -4], [3, 6], [-2, 2]], dtype=float)
    mean = np.zeros_like(y)
    cov = np.tile(np.diag([1., 4.]), (4, 1, 1))
    actual = MODULE.estimate_temperatures(y, mean, cov)
    z = y / np.array([1, 2])
    np.testing.assert_array_equal(actual["identity"], [1, 1])
    np.testing.assert_allclose(actual["nll"], np.sqrt(np.mean(z*z, axis=0)))
    np.testing.assert_allclose(actual["central80"], np.quantile(np.abs(z), .8, axis=0) / MODULE.ndtri(.9))
    with pytest.raises(ValueError, match="nonpositive"):
        MODULE.estimate_temperatures(np.zeros_like(y), mean, cov)


def test_adapter_scales_every_covariance_block_but_keeps_mean():
    x, y = synthetic()
    p = BayesianLinearSurrogate(coregionalized=True, maxiter=15).fit(x[:20], y[:20]).posterior(x[20:])
    t = np.array([.3, 2.7])
    q = MODULE.CalibratedPosterior(p, width_temperature=t)
    idx = np.arange(10)
    tiled = np.tile(t, len(idx))
    np.testing.assert_array_equal(q.mean, p.mean)
    np.testing.assert_allclose(q.cov(idx, idx, noisy=True), p.cov(idx, idx, noisy=True)*np.outer(tiled, tiled))
    np.testing.assert_allclose(q.latent_diag(idx), p.latent_diag(idx)*np.outer(t, t))
    np.testing.assert_allclose(q.noise_blocks(idx), p.noise_blocks(idx)*np.outer(t, t))
    np.testing.assert_allclose(q.predict()[1], p.predict()[1]*t*t)
    assert np.linalg.eigvalsh(q.cov(idx, idx, noisy=True)).min() > 0
    with pytest.raises(ValueError):
        MODULE.CalibratedPosterior(p, [0, 1])


def test_full_output_epig_scores_and_conditional_batch_are_invariant():
    x, y = synthetic()
    p = BayesianLinearSurrogate(coregionalized=True, maxiter=15).fit(x[:12], y[:12]).posterior(x)
    q = MODULE.CalibratedPosterior(p, [.37, 2.1])
    candidates, refs = np.arange(12, 30), np.arange(30)
    np.testing.assert_allclose(score_gaussian_acquisition(p, candidates, refs),
                               score_gaussian_acquisition(q, candidates, refs), rtol=1e-11, atol=1e-12)
    first = select_gaussian_batch(p, candidates, refs, batch_size=5)
    second = select_gaussian_batch(q, candidates, refs, batch_size=5)
    assert first["selected_positions"] == second["selected_positions"]
    np.testing.assert_allclose([v["conditional_score"] for v in first["trace"]], [v["conditional_score"] for v in second["trace"]], rtol=1e-10, atol=1e-12)
    # General endpoint-specific scaling does change trace-risk IVR weights.
    assert not np.allclose(score_gaussian_acquisition(p, candidates, refs, objective="ivr"),
                           score_gaussian_acquisition(q, candidates, refs, objective="ivr"))


def test_nested_identity_matches_stage1_oof_and_replay_never_refits(tmp_path):
    x, y = synthetic()
    calls = []
    result = MODULE.run_nested_family(157, "blr_independent", x, y, tmp_path, CONTRACT, before_fit=calls.append)
    assert result["fit_attempt_count"] == len(calls) == 12
    assert sum(v["phase"] == "inner_oof" for v in calls) == 9
    assert sum(v["phase"] == "outer_or_full_fit" for v in calls) == 3
    baseline = qualify_model(lambda: BayesianLinearSurrogate(max_components=3, maxiter=15), x, y, seed=157+910007)
    identity = result["candidates"]["identity"]
    np.testing.assert_allclose(identity["oof_mean"], baseline["oof_mean"])
    np.testing.assert_allclose(identity["oof_covariance_blocks"], baseline["oof_covariance_blocks"])
    assert identity["metrics"] == baseline["metrics"]
    for temperature in result["temperatures"]:
        assert temperature["inner_seed"] == 157+910007+1009*(temperature["fold"]+1)
    def no_fit(_):
        raise AssertionError("cache replay attempted a model fit")
    replay = MODULE.run_nested_family(157, "blr_independent", x, y, tmp_path, CONTRACT, before_fit=no_fit)
    np.testing.assert_array_equal(replay["candidates"]["nll"]["oof_mean"], result["candidates"]["nll"]["oof_mean"])
    assert replay["fit_attempt_count"] == 12


def test_outer_held_out_truth_cannot_change_its_temperature_or_predictions(tmp_path):
    x, y = synthetic()
    original = MODULE.run_nested_family(157, "blr_independent", x, y, tmp_path/"first", CONTRACT)
    outer_held = np.array(original["candidates"]["identity"]["folds"][0]["held_out_positions"])
    changed = y.copy()
    changed[outer_held] += 50
    altered = MODULE.run_nested_family(157, "blr_independent", x, changed, tmp_path/"second", CONTRACT)
    assert original["temperatures"][0] == altered["temperatures"][0]
    for scheme in MODULE.SCHEMES:
        np.testing.assert_array_equal(original["candidates"][scheme]["oof_mean"][outer_held], altered["candidates"][scheme]["oof_mean"][outer_held])
        np.testing.assert_array_equal(original["candidates"][scheme]["oof_covariance_blocks"][outer_held], altered["candidates"][scheme]["oof_covariance_blocks"][outer_held])
        assert original["candidates"][scheme]["folds"][0]["full_joint_nll_per_scalar"] != altered["candidates"][scheme]["folds"][0]["full_joint_nll_per_scalar"]


def test_nested_interruption_reuses_completed_inner_and_outer_fits(tmp_path):
    x, y = synthetic()
    calls = []
    def interrupt(event):
        calls.append(event)
        if len(calls) == 6:
            raise RuntimeError("synthetic interruption")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        MODULE.run_nested_family(157, "blr_independent", x, y, tmp_path, CONTRACT, before_fit=interrupt)
    assert (tmp_path/"fold_0/complete.json").exists()
    result = MODULE.run_nested_family(157, "blr_independent", x, y, tmp_path, CONTRACT, before_fit=calls.append)
    assert result["fit_attempt_count"] == len(calls) == 13
    with pytest.raises(RuntimeError, match="contract drift"):
        MODULE.run_nested_family(157, "blr_independent", x+.1, y, tmp_path, CONTRACT)
    path = tmp_path/"fold_0/oof_arrays.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="output drift"):
        MODULE.run_nested_family(157, "blr_independent", x, y, tmp_path, CONTRACT)


def test_final_helper_fits_four_models_and_resumes(tmp_path):
    x, y = synthetic()
    calls = []
    full = MODULE.fit_calibrated_full(157, "blr_independent", "nll", x, y, tmp_path, CONTRACT, before_fit=calls.append)
    assert len(calls) == full["fit_attempt_count"] == 4
    q = MODULE.CalibratedPosterior(full["model"].posterior(x), full["temperature"])
    assert q.mean.shape == (30, 2)
    replay = MODULE.fit_calibrated_full(157, "blr_independent", "nll", x, y, tmp_path, CONTRACT,
                                      before_fit=lambda _: pytest.fail("replayed completed full fit"))
    np.testing.assert_array_equal(replay["temperature"], full["temperature"])
    with pytest.raises(RuntimeError, match="contract drift"):
        MODULE.fit_calibrated_full(157, "blr_independent", "central80", x, y, tmp_path, CONTRACT)

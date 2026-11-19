"""Analytic and independent dense numerical checks for Gaussian EPIG."""

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.epig import (
    GaussianAcquisitionState, gaussian_pair_information,
    score_gaussian_acquisition, select_gaussian_batch,
)


class DensePosterior:
    def __init__(self, covariance, noise, n=8, q=2):
        self.k = np.asarray(covariance, dtype=np.float64)
        self.noise_cov = np.asarray(noise, dtype=np.float64)
        self.row_ids = np.array([f"train_{i}" for i in range(n)])
        self.n_outputs = q
        self.mean = np.zeros((n, q))

    def cov(self, left, right, noisy=False):
        li = (np.asarray(left)[:, None] * self.n_outputs + np.arange(self.n_outputs)).reshape(-1)
        ri = (np.asarray(right)[:, None] * self.n_outputs + np.arange(self.n_outputs)).reshape(-1)
        value = self.k[np.ix_(li, ri)].copy()
        if noisy:
            for i, a in enumerate(left):
                for j, b in enumerate(right):
                    if a == b:
                        value[i*self.n_outputs:(i+1)*self.n_outputs,
                              j*self.n_outputs:(j+1)*self.n_outputs] += self.noise_cov
        return value

    def latent_diag(self, positions):
        return np.stack([self.cov([i], [i]) for i in positions])


def random_posterior(seed=17, n=8, q=2):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n*q, 7))
    return DensePosterior(features @ features.T / 7 + np.eye(n*q)*0.02,
                          np.diag(np.linspace(0.3, 0.8, q)), n, q)


def dense_conditioned(posterior, left, right, selected):
    value = posterior.cov(left, right)
    if len(selected):
        value -= posterior.cov(left, selected) @ np.linalg.solve(
            posterior.cov(selected, selected, noisy=True), posterior.cov(selected, right))
    return value


def dense_scores(posterior, candidates, refs, selected=(), objective="epig", weights=None):
    weights = np.ones(len(refs))/len(refs) if weights is None else np.asarray(weights)/np.sum(weights)
    results = []
    for candidate in candidates:
        if candidate in selected:
            results.append(-np.inf)
            continue
        sx = dense_conditioned(posterior, [candidate], [candidate], selected) + posterior.noise_cov
        values = []
        for target in refs:
            st = dense_conditioned(posterior, [target], [target], selected) + posterior.noise_cov
            cross = dense_conditioned(posterior, [target], [candidate], selected)
            after = st - cross @ np.linalg.solve(sx, cross.T)
            value = (0.5*(np.linalg.slogdet(st)[1]-np.linalg.slogdet(after)[1])
                     if objective == "epig" else np.trace(st-after))
            values.append(value)
        results.append(np.dot(weights, values))
    return np.asarray(results)


def test_scalar_analytic_formula_and_independent_noise():
    information, reduction = gaussian_pair_information([[3.]], [[5.]], [[2.]])
    assert information == pytest.approx(-0.5*np.log1p(-4/15), abs=1e-14)
    assert reduction == pytest.approx(4/3, abs=1e-14)
    zero, no_reduction = gaussian_pair_information([[3.]], [[5.]], [[0.]])
    assert zero == no_reduction == 0
    # Same input has two independently noisy future labels, not self-MI.
    p = DensePosterior([[2.]], [[0.5]], n=1, q=1)
    score = score_gaussian_acquisition(p, [0], [0])[0]
    assert score == pytest.approx(-0.5*np.log1p(-(2/2.5)**2), abs=1e-14)


@pytest.mark.parametrize("candidate_q", [1, 2, 3])
@pytest.mark.parametrize("target_q", [1, 2, 3])
def test_rectangular_output_blocks_match_dense_logdet(candidate_q, target_q):
    rng = np.random.default_rng(candidate_q*10+target_q)
    features = rng.normal(size=(candidate_q+target_q, 7))
    covariance = features @ features.T + np.eye(candidate_q+target_q)*0.4
    sx = covariance[:candidate_q, :candidate_q]
    st = covariance[candidate_q:, candidate_q:]
    cross = covariance[candidate_q:, :candidate_q]
    info, reduction = gaussian_pair_information(sx, st, cross)
    conditional = st-cross@np.linalg.solve(sx,cross.T)
    assert info == pytest.approx(.5*(np.linalg.slogdet(st)[1]-np.linalg.slogdet(conditional)[1]), rel=1e-12, abs=1e-13)
    assert reduction == pytest.approx(np.trace(st-conditional), rel=1e-12, abs=1e-13)


def test_log1p_preserves_tiny_information_and_near_singular_valid_pair():
    tiny, _ = gaussian_pair_information([[1.]], [[1.]], [[1e-10]])
    assert tiny > 0
    assert tiny == pytest.approx(5e-21, rel=1e-14)
    rho = 1-1e-10
    large, _ = gaussian_pair_information([[1.]], [[1.]], [[rho]])
    assert large == pytest.approx(-.5*np.log1p(-rho*rho), rel=1e-14)


@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("objective", ["epig", "ivr"])
def test_all_scores_and_greedy_choices_match_direct_dense_conditioning(q, objective):
    p = random_posterior(q=q)
    candidates, refs = np.arange(2, 8), np.arange(8)
    weights = np.arange(1, 9)
    result = select_gaussian_batch(p, candidates, refs, objective=objective,
                                   batch_size=4, labeled_positions=[0, 1],
                                   reference_weights=weights, chunk_size=2)
    chosen = []
    for step, record in enumerate(result["trace"]):
        direct = dense_scores(p, candidates, refs, chosen, objective, weights)
        assert record["candidate_index"] == int(np.argmax(direct))
        assert record["conditional_score"] == pytest.approx(float(np.max(direct)), rel=1e-10, abs=1e-12)
        chosen.append(record["row_position"])
    np.testing.assert_allclose(result["initial_scores"], dense_scores(p, candidates, refs, objective=objective, weights=weights), rtol=1e-10, atol=1e-12)
    assert len(set(result["selected_row_ids"])) == 4
    if objective == "epig":
        # Sum of conditional gains equals direct batch MI with each target.
        joint = []
        sbb = p.cov(chosen, chosen, noisy=True)
        for target in refs:
            st = p.cov([target], [target]) + p.noise_cov
            cross = p.cov([target], chosen)
            joint.append(0.5*(np.linalg.slogdet(st)[1]-np.linalg.slogdet(st-cross@np.linalg.solve(sbb,cross.T))[1]))
        assert result["audit"]["conditional_batch_information_nats"] == pytest.approx(np.dot(weights/weights.sum(), joint), rel=1e-10)


def test_joint_output_dependence_is_not_sum_of_marginal_epig():
    p = random_posterior(q=2)
    joint = score_gaussian_acquisition(p, [2, 3, 4], [5, 6, 7])
    independent_sum = sum(score_gaussian_acquisition(p, [2, 3, 4], [5, 6, 7], output_indices=[output]) for output in [0, 1])
    assert not np.allclose(joint, independent_sum, atol=1e-4)
    for output in [0, 1]:
        mask = np.arange(output, 16, 2)
        scalar = DensePosterior(p.k[np.ix_(mask, mask)], p.noise_cov[output:output+1, output:output+1], n=8, q=1)
        np.testing.assert_allclose(score_gaussian_acquisition(p, [2,3,4], [5,6,7], output_indices=[output]),
                                   score_gaussian_acquisition(scalar, [2,3,4], [5,6,7]), atol=1e-13)


def test_physical_unit_transform_preserves_information_and_rescales_ivr():
    p = random_posterior()
    transform = np.array([[5., 1.], [0., 0.25]])
    full = np.kron(np.eye(8), transform)
    transformed = DensePosterior(full @ p.k @ full.T, transform @ p.noise_cov @ transform.T)
    first = select_gaussian_batch(p, np.arange(2,8), np.arange(8), batch_size=3)
    second = select_gaussian_batch(transformed, np.arange(2,8), np.arange(8), batch_size=3)
    assert first["selected_positions"] == second["selected_positions"]
    np.testing.assert_allclose(first["initial_scores"], second["initial_scores"], rtol=1e-12, atol=1e-13)
    scaled = DensePosterior(p.k*100, p.noise_cov*100)
    np.testing.assert_allclose(score_gaussian_acquisition(scaled, [2,3], [4,5], objective="ivr"),
                               score_gaussian_acquisition(p, [2,3], [4,5], objective="ivr")*100, rtol=1e-12)


def test_pointwise_and_conditional_batches_are_distinct_and_ties_stable():
    # Candidates 0,1 are near duplicates; 2 covers another target direction.
    features = np.array([[1.,0.], [1.,0.], [0.,0.9], [1.,0.], [0.,1.]])
    p = DensePosterior(features @ features.T, [[0.1]], n=5, q=1)
    pointwise = select_gaussian_batch(p, [0,1,2], [3,4], batch_size=2, batch_mode="pointwise_topk")
    greedy = select_gaussian_batch(p, [0,1,2], [3,4], batch_size=2)
    assert pointwise["selected_positions"] == [0,1]
    assert greedy["selected_positions"] == [0,2]
    assert greedy["audit"]["conditional_batch_information_nats"] > pointwise["audit"]["conditional_batch_information_nats"]
    assert pointwise["method"] != greedy["method"]


def test_preconditioned_scores_and_weighted_distribution():
    p = random_posterior()
    actual = score_gaussian_acquisition(p, [2,3,4,5], [0,1,6,7], selected_positions=[3], reference_weights=[0,1,3,0])
    np.testing.assert_allclose(actual, dense_scores(p, [2,3,4,5], [0,1,6,7], [3], weights=[0,1,3,0]), rtol=1e-12, atol=1e-12)
    # Repeated reference draws equal their explicit empirical masses.
    np.testing.assert_allclose(score_gaussian_acquisition(p, [2,3], [4,4,5]),
                               score_gaussian_acquisition(p, [2,3], [4,5], reference_weights=[2,1]), atol=1e-13)


@pytest.mark.parametrize("kwargs", [
    {"candidate_positions": [2,2]}, {"candidate_positions": [-1,2]},
    {"candidate_positions": [2,8]}, {"candidate_positions": [2.0,3.0]},
    {"candidate_positions": [1,2], "labeled_positions": [1]},
    {"batch_size": 0}, {"batch_size": 8}, {"batch_mode": "top_magic"},
    {"reference_weights": [1,-1,1]}, {"reference_weights": [0,0,0]},
    {"output_indices": [0,0]}, {"output_indices": [2]},
    {"max_cache_bytes": 1},
])
def test_invalid_ids_weights_modes_and_budget_are_rejected(kwargs):
    arguments = dict(candidate_positions=[2,3,4], reference_positions=[0,1,7], batch_size=2)
    arguments.update(kwargs)
    with pytest.raises((ValueError, MemoryError)):
        select_gaussian_batch(random_posterior(), **arguments)


@pytest.mark.parametrize("sxx,stt,cross", [
    ([[0.]], [[1.]], [[0.]]), ([[1.]], [[1.]], [[2.]]),
    ([[1.]], [[1.]], [[1.]]), ([[1.]], [[np.nan]], [[0.]]),
    ([[1.,1.],[0.,1.]], np.eye(2), np.zeros((2,2))),
])
def test_invalid_gaussian_is_rejected_without_silent_jitter(sxx,stt,cross):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        gaussian_pair_information(sxx,stt,cross)


def test_negative_noise_and_negative_latent_are_rejected():
    p = random_posterior()
    p.noise_cov[0,0] = -0.1
    with pytest.raises(np.linalg.LinAlgError):
        GaussianAcquisitionState(p, [2,3], [4,5])
    p = random_posterior()
    p.k[4,4] = -1
    with pytest.raises(np.linalg.LinAlgError):
        GaussianAcquisitionState(p, [2,3], [4,5])


def test_chunk_size_does_not_change_scores_and_fallback_avoids_dense_candidates():
    p = random_posterior()
    np.testing.assert_allclose(score_gaussian_acquisition(p, [2,3,4], [5,6,7], chunk_size=1),
                               score_gaussian_acquisition(p, [2,3,4], [5,6,7], chunk_size=32), atol=1e-13)
    class NoDiag:
        row_ids, n_outputs, noise_cov = p.row_ids, p.n_outputs, p.noise_cov
        def cov(self, left, right, noisy=False):
            assert not (len(left) > 1 and np.array_equal(left, right))
            return p.cov(left, right, noisy)
    np.testing.assert_allclose(score_gaussian_acquisition(NoDiag(), [2,3,4], [5,6,7]),
                               score_gaussian_acquisition(p, [2,3,4], [5,6,7]), atol=1e-13)

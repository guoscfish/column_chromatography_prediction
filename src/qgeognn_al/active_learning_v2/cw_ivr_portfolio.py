"""Label-free, shared-context incremental LCMD and conditional IVR adapters.

Historical batch selectors remain byte-for-byte untouched. These incremental
adapters preserve their arithmetic, stable ordering, and objective definitions.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .lcmd import _squared_distances, lcmd_tp_select
from .ivr import conditional_batch_ivr


class CoverageState:
    def __init__(self, pool, train):
        self.pool = np.asarray(pool, dtype=np.float64)
        train = np.asarray(train, dtype=np.float64)
        if self.pool.ndim != 2 or train.ndim != 2 or not len(train) or self.pool.shape[1] != train.shape[1]:
            raise ValueError('aligned pool and nonempty training centers required')
        if not np.isfinite(self.pool).all() or not np.isfinite(train).all():
            raise ValueError('finite features required')
        distances = _squared_distances(self.pool, train)
        self.closest = distances.argmin(axis=1)
        self.minimum = distances[np.arange(len(self.pool)), self.closest]
        self.available = np.ones(len(self.pool), dtype=bool)
        self.center_count = len(train)
        self.selected = []

    def propose(self):
        scores = np.bincount(self.closest[self.available], weights=self.minimum[self.available], minlength=self.center_count)
        cluster = int(np.argmax(scores))
        candidates = np.flatnonzero(self.available & (self.closest == cluster))
        if not len(candidates):
            raise RuntimeError('LCMD selected an empty largest cluster')
        point = int(candidates[np.argmax(self.minimum[candidates])])
        return point, {'largest_cluster_score': float(scores[cluster]),
                       'selected_from_center': cluster,
                       'nearest_center_sq_distance_before_selection': float(self.minimum[point])}

    def condition(self, point):
        if not self.available[point]:
            raise ValueError('duplicate batch location')
        self.available[point] = False
        distance = _squared_distances(self.pool, self.pool[point:point+1])[:, 0]
        improved = self.available & (distance < self.minimum)
        self.closest[improved] = self.center_count
        self.minimum[improved] = distance[improved]
        self.closest[point], self.minimum[point] = self.center_count, 0.0
        self.center_count += 1
        self.selected.append(int(point))


class IVRState:
    def __init__(self, features, labeled_positions, candidate_positions):
        raw = np.asarray(features, dtype=np.float64)
        self.labeled = np.asarray(labeled_positions, dtype=int)
        self.candidates = np.asarray(candidate_positions, dtype=int)
        if raw.ndim != 2 or not np.isfinite(raw).all() or not len(raw):
            raise ValueError('finite nonempty features required')
        joined = np.r_[self.labeled, self.candidates]
        if len(joined) != len(raw) or sorted(joined.tolist()) != list(range(len(raw))):
            raise ValueError('L/U must partition the reference universe')
        scale = float(np.mean(np.einsum('ij,ij->i', raw, raw)))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError('positive feature scale required')
        self.x = raw / np.sqrt(scale)
        precision = np.eye(raw.shape[1]) + self.x[self.labeled].T @ self.x[self.labeled]
        covariance = cho_solve(cho_factor(precision, lower=True), np.eye(raw.shape[1]))
        moment = self.x.T @ self.x / len(raw)
        self.q = self.x @ covariance
        self.h = self.q @ moment
        self.available = np.ones(len(self.candidates), dtype=bool)
        self.selected = []

    def scores(self):
        variance = np.einsum('ij,ij->i', self.q, self.x)
        numerator = np.einsum('ij,ij->i', self.q, self.h)
        if not np.isfinite(variance).all() or not np.isfinite(numerator).all() or min(variance) < -1e-9 or min(numerator) < -1e-9:
            raise RuntimeError('conditional covariance lost numerical validity')
        scores = np.maximum(numerator[self.candidates], 0) / (1 + np.maximum(variance[self.candidates], 0))
        scores[~self.available] = -np.inf
        return scores, variance

    def propose(self):
        scores, _ = self.scores()
        return int(np.argmax(scores))

    def condition(self, point):
        if not self.available[point]:
            raise ValueError('duplicate batch location')
        scores, variance = self.scores()
        position = int(self.candidates[point])
        before, reduction = float(variance.mean()), float(scores[point])
        denominator = 1 + float(variance[position])
        v, hm = self.q[position].copy(), self.h[position].copy()
        cross = self.q @ self.x[position]
        self.q -= np.outer(cross, v) / denominator
        self.h -= np.outer(cross, hm) / denominator
        after = float(np.einsum('ij,ij->', self.q, self.x) / len(self.x))
        if not np.isclose(before - after, reduction, rtol=1e-7, atol=1e-11):
            raise RuntimeError('IVR marginal reduction mismatch')
        self.available[point] = False
        self.selected.append(int(point))
        return {'integrated_variance_before': before, 'selected_marginal_variance_reduction': reduction,
                'integrated_variance_after': after, 'conditional_variance': float(variance[position])}


def _norm_diagnostics(features):
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    gram = x.T @ x
    return {'gradient_norm_mean': float(norms.mean()), 'gradient_norm_std': float(norms.std()),
            'gradient_norm_p50': float(np.quantile(norms, .5)), 'gradient_norm_p90': float(np.quantile(norms, .9)),
            'gradient_norm_p95': float(np.quantile(norms, .95)), 'gradient_norm_max': float(norms.max()),
            'gradient_effective_rank': float(np.trace(gram)**2 / np.square(gram).sum())}


def select_portfolio(cw_features, ivr_features, labeled_count):
    """Exactly 32 alternating turns, CW first; both states observe every pick.

    Matrices share ordered L_t+U_t rows (the complete fixed outer universe).
    Independent proposals are diagnostics only; no proposed labels are read.
    """
    cw, raw = np.asarray(cw_features), np.asarray(ivr_features)
    n = int(labeled_count)
    if cw.shape != raw.shape or not 0 < n <= len(cw) - 32:
        raise ValueError('aligned L+U banks with at least 32 candidates required')
    coverage = CoverageState(cw[n:], cw[:n])
    ivr = IVRState(raw, np.arange(n), np.arange(n, len(raw)))
    initial_distances = np.sqrt(coverage.minimum.copy())
    trace, selected = [], []
    for step in range(32):
        cw_pick, cw_diag = coverage.propose()
        ivr_pick = ivr.propose()
        expert = 'cw' if step % 2 == 0 else 'ivr'
        point = cw_pick if expert == 'cw' else ivr_pick
        nearest = float(coverage.minimum[point])
        row = {'portfolio_step': step, 'expert': expert, 'pool_position': point,
               'shared_context_count': step, 'coverage_center_count': coverage.center_count,
               'cw_proposal_pool_position': cw_pick, 'ivr_proposal_pool_position': ivr_pick,
               'proposals_agree': cw_pick == ivr_pick,
               'proposal_cw_sq_distance': float(_squared_distances(cw[n+cw_pick:n+cw_pick+1], cw[n+ivr_pick:n+ivr_pick+1])[0, 0]),
               'nearest_center_sq_distance_before_selection': nearest,
               **(cw_diag if expert == 'cw' else {}), **ivr.condition(point)}
        coverage.condition(point)
        if coverage.selected != ivr.selected:
            raise RuntimeError('experts did not observe the same complete shared batch')
        selected.append(point)
        trace.append(row)
    independent_cw = lcmd_tp_select(cw[n:], cw[:n], 16).selected_pool_positions.tolist()
    independent_ivr = conditional_batch_ivr(raw, np.arange(n), np.arange(n, len(raw)), 16)['selected_candidate_positions']
    overlap = len(set(independent_cw) & set(independent_ivr))
    final_distances = np.sqrt(coverage.minimum)
    # Geometric baseline only: independent union has 32-overlap locations, no backfill.
    independent_coverage = CoverageState(cw[n:], cw[:n])
    independent_variance = IVRState(raw, np.arange(n), np.arange(n, len(raw)))
    for p in dict.fromkeys(independent_cw + independent_ivr):
        independent_coverage.condition(p)
        independent_variance.condition(p)
    before, after = trace[0]['integrated_variance_before'], trace[-1]['integrated_variance_after']
    diagnostics = {**_norm_diagnostics(raw),
        'coverage_nearest_distance_mean': float(initial_distances.mean()),
        'coverage_nearest_distance_p90': float(np.quantile(initial_distances, .9)),
        'coverage_after_mean': float(final_distances.mean()), 'coverage_after_p90': float(np.quantile(final_distances, .9)),
        'ivr_relative_reducible_variance': (before-after)/before,
        'ivr_integrated_variance_before': before, 'ivr_integrated_variance_after': after,
        'independent_proposal_overlap_count': overlap, 'independent_union_size': 32-overlap,
        'independent_union_coverage_mean': float(np.sqrt(independent_coverage.minimum).mean()),
        'independent_union_integrated_variance': float(independent_variance.scores()[1].mean()),
        'shared_proposal_agreement_steps': sum(r['proposals_agree'] for r in trace),
        'cw_turns': 16, 'ivr_turns': 16, 'batch_size': 32, 'unique_ids': len(set(selected)),
        'independent_cw_positions': independent_cw, 'independent_ivr_positions': independent_ivr}
    return {'selected_pool_positions': selected, 'trace': trace, 'diagnostics': diagnostics}

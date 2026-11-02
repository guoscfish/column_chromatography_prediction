"""Independent representation/selector contracts; historical kernels stay unchanged."""
from __future__ import annotations
import numpy as np
from .gradient_transforms import raw_output_transform, center_width_transform, output_whitening_transform
from .lcmd import lcmd_tp_select
from .maxdet import conditional_gradient_maxdet
from .ivr import conditional_batch_ivr

SEEDS = (157, 6101)
BUDGETS = tuple(range(333, 526, 32))
ARMS = {'cw_maxdet': ('cw', 'maxdet'), 'cw_ivr': ('cw', 'ivr'),
        'whitening_lcmd': ('whitening', 'lcmd')}
PROJECTION_SEEDS = (17, 101, 1009)
DIMENSIONS = (256, 512, 1024)


def transforms(targets, scales):
    raw, ra = raw_output_transform(scales)
    cw, ca = center_width_transform(targets)
    white, wa = output_whitening_transform(targets, scales)
    q = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
    eig, vec = np.linalg.eigh(cw.T @ cw)
    matched = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    return {'raw': raw, 'rotation': q @ raw, 'cw': cw, 'whitening': white,
            'matched': matched}, {'raw': ra, 'cw': ca, 'whitening': wa}


def select(features, labeled_count, selector, batch_size=32):
    """Rows must be ordered L_t (acquisition order), U_t (canonical order)."""
    x = np.asarray(features)
    n = int(labeled_count)
    if x.ndim != 2 or not 0 < n < len(x):
        raise ValueError('nonempty ordered L/U matrix required')
    l, u = np.arange(n), np.arange(n, len(x))
    if selector == 'lcmd':
        result = lcmd_tp_select(x[n:], x[:n], batch_size)
        positions, trace = result.selected_pool_positions, result.trace
    elif selector == 'maxdet':
        result = conditional_gradient_maxdet(x, l, u, batch_size)
        positions, trace = result.selected_candidate_positions, result.trace
    elif selector == 'ivr':
        result = conditional_batch_ivr(x, l, u, batch_size)
        positions, trace = result['selected_candidate_positions'], result['trace']
    else:
        raise ValueError('unknown selector')
    positions = np.asarray(positions, dtype=int)
    if len(positions) != batch_size or len(set(positions)) != batch_size or np.any(positions < 0) or np.any(positions >= len(u)):
        raise RuntimeError('invalid selected pool positions')
    return positions, list(trace)

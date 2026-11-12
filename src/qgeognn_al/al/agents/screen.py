"""Frozen screen-v1 numerical proposal and independent RNG streams.

The exact original combiner uses the one authoritative migrated kernels. It
never changes the separate Free32 complete-pool profiles into this shortlist.
"""
import numpy as np
from ..acquisitions.lcmd import lcmd_tp_select
from ..acquisitions.ivr import conditional_batch_ivr
from ..acquisitions.maxdet import conditional_gradient_maxdet

def rng(seed, round_index, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(round_index), int(stream), 1616]))

def shortlist(cw, raw, labeled_count, seed, round_index):
    """Condition feature-only proposals on pending CW16; retain all reference rows."""
    cw, raw = np.asarray(cw), np.asarray(raw)
    n = int(labeled_count)
    if cw.shape != raw.shape or not 0 < n <= len(cw) - 144:
        raise ValueError('aligned L+U banks with at least 144 U rows required')
    if not np.isfinite(cw).all() or not np.isfinite(raw).all():
        raise ValueError('nonfinite feature bank')
    pending = lcmd_tp_select(cw[n:], cw[:n], 16).selected_pool_positions + n
    centers = np.r_[np.arange(n), pending]
    remaining = np.array([i for i in range(n, len(cw)) if i not in set(pending)])
    proposals = {
        'cw': remaining[lcmd_tp_select(cw[remaining], cw[centers], 40).selected_pool_positions].tolist(),
        'ivr': remaining[conditional_batch_ivr(raw, centers, remaining, 40)['selected_candidate_positions']].tolist(),
        'maxdet': conditional_gradient_maxdet(raw, centers, remaining, 40).selected_positions.tolist(),
        'random': rng(seed, round_index, 1).choice(remaining, 8, replace=False).tolist(),
    }
    union = list(dict.fromkeys(p for values in proposals.values() for p in values))
    rest = [p for p in remaining if p not in set(union)]
    fill = rng(seed, round_index, 2).choice(rest, 128 - len(union), replace=False).tolist()
    union += fill
    # Independent reproducible presentation, control and failure-fill streams.
    presentation = rng(seed, round_index, 3).permutation(union).tolist()
    return {'pending_positions': pending.tolist(), 'candidate_positions': presentation,
            'proposals': proposals, 'random_backfill': fill,
            'random16_positions': rng(seed, round_index, 4).choice(presentation, 16, replace=False).tolist(),
            'fallback_positions': rng(seed, round_index, 5).permutation(presentation).tolist()}

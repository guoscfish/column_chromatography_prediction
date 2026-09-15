"""Matched acquisition interface: inputs contain no targets or label store."""

import numpy as np

from .coverage import coreset_tp_select
from .lcmd import lcmd_tp_select
from .protocol import RANDOM_CONTROLS, random_control_positions
from .uncertainty import ensemble_uncertainty_select


def acquire_batches(*, gradients, representations, ensemble_predictions, l0_count,
                    scales, outer_seed, batch_size):
    if batch_size not in (16, 32, 64):
        raise ValueError("only frozen B=16/32/64 are allowed")
    pool_size = len(gradients) - l0_count
    arms = {f"random_control_{i}": random_control_positions(pool_size, batch_size, outer_seed, i)
            for i in range(RANDOM_CONTROLS)}
    arms["lcmd"] = lcmd_tp_select(gradients[l0_count:], gradients[:l0_count], batch_size).selected_pool_positions
    if batch_size == 32:
        if len(representations) != len(gradients) or ensemble_predictions.shape[1] != pool_size:
            raise ValueError("acquisition inputs must share ordered L0/U0 identities")
        arms["coreset"] = coreset_tp_select(representations[l0_count:], representations[:l0_count], batch_size)
        arms["uncertainty"] = ensemble_uncertainty_select(ensemble_predictions, scales, batch_size)[0]
    for positions in arms.values():
        if len(positions) != batch_size or len(np.unique(positions)) != batch_size:
            raise RuntimeError("acquisition size/uniqueness violation")
        if np.any(positions < 0) or np.any(positions >= pool_size):
            raise RuntimeError("acquisition escaped U0")
    return arms

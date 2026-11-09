"""Explicit effective-seed random sampling and frozen control seed derivation."""
import numpy as np


def random_select(pool_size: int, batch_size: int, effective_seed: int):
    if type(pool_size) is not int or type(batch_size) is not int or not 1 <= batch_size <= pool_size:
        raise ValueError("legal pool and batch sizes required")
    if type(effective_seed) is not int or effective_seed < 0:
        raise ValueError("effective_seed must be a nonnegative integer")
    return np.random.default_rng(effective_seed).choice(pool_size, size=batch_size, replace=False)


def random_control_seed(outer_seed: int, control: int):
    if type(outer_seed) is not int or outer_seed < 0 or type(control) is not int or not 0 <= control < 5:
        raise ValueError("registered outer seed and control index required")
    return outer_seed * 1_000_003 + 50_000 + control


def random_control_positions(pool_size, batch, outer_seed, control):
    return random_select(pool_size, batch, random_control_seed(outer_seed, control))


def historical_random_trajectory_seed(outer_seed: int):
    if type(outer_seed) is not int or outer_seed < 0:
        raise ValueError('registered nonnegative outer seed required')
    return outer_seed * 1_000_003 + 900_001


def fixed_random_trajectory_batch(original_u0_ids, current_u_ids, effective_seed, round_index, batch_size, selection_count=None):
    """Take a fixed-size slice of one U0 permutation, never redraw from U_t.

    ``round_index`` is zero-based; ``batch_size`` belongs to the frozen protocol
    and must stay fixed across its trajectory. The historical profile uses 32.
    Authorization must equal the exact remaining suffix identity set.
    """
    from ...infra.contracts import sample_ids
    original = sample_ids(original_u0_ids, name='original U0')
    current = sample_ids(current_u_ids, name='current U_t')
    if not original or type(effective_seed) is not int or effective_seed < 0:
        raise ValueError('nonempty U0 and explicit effective seed required')
    if type(round_index) is not int or round_index < 0 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('registered nonnegative round and positive fixed batch size required')
    order = np.random.default_rng(effective_seed).permutation(len(original))
    trajectory = tuple(original[int(position)] for position in order)
    count = batch_size if selection_count is None else selection_count
    if type(count) is not int or not 1 <= count <= batch_size:
        raise ValueError('selection count must be a legal full or final partial batch')
    start = round_index * batch_size; stop = start + count
    if stop > len(trajectory):
        raise ValueError('fixed trajectory cannot supply a complete registered batch')
    if set(current) != set(trajectory[start:]):
        raise ValueError('current U_t is not the exact remaining frozen random trajectory')
    return trajectory[start:stop]

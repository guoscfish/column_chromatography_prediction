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

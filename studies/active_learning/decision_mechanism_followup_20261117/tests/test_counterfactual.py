"""Exact LCMD semantics and counterfactual admission tests without labels."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('counterfactual_unit', HERE / 'counterfactual.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
lcmd = module.load_module('counterfactual_lcmd_unit', ROOT / 'src/qgeognn_al/active_learning_v2/lcmd.py')


def test_lcmd_is_largest_cluster_not_farthest_first():
    pool = np.asarray(list(range(1, 11)) + [115], dtype=np.float32)[:, None]
    result = lcmd.lcmd_tp_select(pool, np.asarray([[0], [100]], dtype=np.float32), 2)
    assert result.selected_pool_positions[0] == 9
    assert np.argmax(result.initial_nearest_sq_distance) == 10


def test_stable_tie_order_and_zero_vectors():
    result = lcmd.lcmd_tp_select(np.asarray([[1.], [-1.]]), np.asarray([[0.]]), 2)
    assert result.selected_pool_positions.tolist() == [0, 1]
    result = lcmd.lcmd_tp_select(np.zeros((4, 3), np.float32), np.zeros((2, 3), np.float32), 4)
    assert result.selected_pool_positions.tolist() == [0, 1, 2, 3]


def test_32_prefix_is_same_as_16_and_float64_calculation():
    rng = np.random.default_rng(20261117)
    pool, train = rng.normal(size=(80, 12)).astype(np.float32), rng.normal(size=(7, 12)).astype(np.float32)
    a = lcmd.lcmd_tp_select(pool, train, 16)
    b = lcmd.lcmd_tp_select(pool, train, 32)
    assert np.array_equal(a.selected_pool_positions, b.selected_pool_positions[:16])
    assert b.initial_nearest_sq_distance.dtype == np.float64


def test_ordered_prefix_required_and_gain_monotone():
    assert module.audit_prefix(['a', 'b', 'c'], ['a', 'b']) == (True, None)
    assert module.audit_prefix(['b', 'a', 'c'], ['a', 'b']) == (False, 0)
    before = np.asarray([1., .4])
    assert np.isclose(module.gain(before, np.asarray([[.5, .2], [.6, .1]])), .55)


def test_reported_rationale_is_transparent_rule_only():
    flags = module.reason_flags('Matched sample volume and PE/EA solvent-response; transfer to unobserved scaffold.')
    assert flags['sample_volume'] and flags['pe'] and flags['structure_transfer']


def test_independent_scalar_trace_with_cluster_sums_and_ties():
    def scalar_reference(pool, train, count):
        centers = [list(row) for row in train]
        available = list(range(len(pool)))
        trace = []
        for _ in range(count):
            assigned = {}
            distances = {}
            for i in available:
                all_distances = [sum((float(a) - float(b)) ** 2 for a, b in zip(pool[i], center)) for center in centers]
                minimum = min(all_distances)
                assigned[i] = all_distances.index(minimum)
                distances[i] = minimum
            sums = [sum(distances[i] for i in available if assigned[i] == k) for k in range(len(centers))]
            chosen_cluster = sums.index(max(sums))
            eligible = [i for i in available if assigned[i] == chosen_cluster]
            maximum = max(distances[i] for i in eligible)
            chosen = next(i for i in eligible if distances[i] == maximum)
            trace.append((chosen, chosen_cluster, sums[chosen_cluster], distances[chosen]))
            centers.append(list(pool[chosen]))
            available.remove(chosen)
        return trace

    # Integer inputs make direct scalar sums and norm-dot calculations exact.
    rng = np.random.default_rng(117)
    for _ in range(20):
        pool, train = rng.integers(-7, 8, size=(19, 4)), rng.integers(-7, 8, size=(3, 4))
        reference = scalar_reference(pool, train, 9)
        actual = lcmd.lcmd_tp_select(pool.astype(np.float32), train.astype(np.float32), 9)
        for expected, observed in zip(reference, actual.trace):
            assert expected == (observed['pool_position'], observed['selected_from_center'],
                                observed['largest_cluster_score'], observed['nearest_center_sq_distance_before_selection'])

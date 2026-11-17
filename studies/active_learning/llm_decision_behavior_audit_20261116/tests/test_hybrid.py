"""Unit checks for explicit source ownership and monotone coverage attribution."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('hybrid_text', Path(__file__).parents[1] / 'hybrid_text.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_explicit_source_and_order_independence():
    n, l = list(range(16)), list(range(16, 32))
    actual_l, actual_n = module.split_halves({'batch_ids': list(reversed(n + l)), 'pending_ids': n},
                                            {'choices': [{'id': x} for x in l]})
    assert actual_l == l and actual_n == n


def test_source_overlap_rejected():
    with pytest.raises(AssertionError):
        module.split_halves({'batch_ids': list(range(32)), 'pending_ids': list(range(16))},
                            {'choices': [{'id': x} for x in range(15, 31)]})


def test_monotonic_coverage_order_average():
    gains = module.marginal_coverage(np.array([1., .5]), np.array([.2, .4]), np.array([.3, .1]))
    assert np.isclose(gains['union_gain'], .6)
    assert np.isclose(gains['llm_order_average'] + gains['numerical_order_average'], .6)
    assert gains['llm_after_numerical'] <= gains['llm_standalone']
    assert gains['numerical_after_llm'] <= gains['numerical_standalone']


def test_transparent_rationale_rules():
    flags = module.text_flags('Explore new molecules, match a control, avoid duplicates.')
    assert flags['exploration_terms'] >= 1
    assert flags['local_control_terms'] >= 1
    assert flags['avoid_redundancy_terms'] >= 1

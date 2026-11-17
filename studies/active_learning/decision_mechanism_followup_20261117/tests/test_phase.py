"""Exact observed-grid integration, influence accounting and budget checks."""
import importlib.util
from pathlib import Path
import math
import pytest

spec = importlib.util.spec_from_file_location('phase_protocol', Path(__file__).parents[1] / 'phase_protocol.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nonmonotone_loss_is_not_best_so_far():
    points = {333: 1., 365: .2, 397: .5}
    assert math.isclose(module.segment(points, 333, 397), .475)
    assert not math.isclose(module.segment(points, 333, 397), .4)


def test_missing_horizon_is_rejected():
    with pytest.raises(ValueError):
        module.segment({333: 1., 397: .5}, 333, 397)


def test_phase_area_additivity():
    points = {x: .4 + .001 * ((x - 333) // 32 % 4) for x in range(333, 1454, 32)}
    full = module.integrate(points, 333, 1453)
    assert math.isclose(full, sum(module.integrate(points, a, b) for a, b in module.PHASES))


def test_local_omission_keeps_constant_difference():
    points = {x: -.02 for x in range(333, 1454, 32)}
    result = module.omission_sensitivity(points)
    assert len(result) == 34
    assert all(math.isclose(r['remaining_observed_interval_delta_AULC'], -.02) for r in result)


def test_expansion_cost_excludes_reused_initial_fit():
    cost = module.planned_cost(1453)
    assert cost['effective_llm_decisions'] == cost['new_qgeognn_fits'] == 105
    assert cost['reusable_initial_fits'] == 3
    assert cost['newly_acquired_row_events'] == 3360
    assert module.planned_cost(1485)['new_qgeognn_fits'] == 108
    with pytest.raises(ValueError):
        module.planned_cost(1454)

import numpy as np
import pandas as pd

from scripts.analyze_4g_error_landscape import _nearest_distance, _parse_ea_fraction, _safe_qcut


def test_parse_ea_fraction_uses_the_second_ratio_component():
    values = _parse_ea_fraction(pd.Series(["40/1", "1/1", "0/1"]))
    np.testing.assert_allclose(values.to_numpy(), [1 / 41, 0.5, 1.0])


def test_nearest_distance_is_zero_for_a_training_point():
    train = np.array([[0.0, 0.0], [2.0, 0.0]])
    query = np.array([[0.0, 0.0], [1.0, 0.0]])
    distances = _nearest_distance(query, train)
    np.testing.assert_allclose(distances[0], 0.0)
    assert distances[1] > 0


def test_safe_qcut_handles_constant_inputs_without_raising():
    labels = _safe_qcut(pd.Series([1.0, 1.0, 1.0]))
    assert labels.tolist() == ["all", "all", "all"]

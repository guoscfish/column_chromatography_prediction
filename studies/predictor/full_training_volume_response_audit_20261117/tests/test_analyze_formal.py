"""Small hand-computed response checks, with synthetic labels only."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT))
SPEC = importlib.util.spec_from_file_location("formal_response_analysis", OUT / "analyze_formal.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def pair():
    return pd.DataFrame(dict(row_seed=[42], axis=["V/ul"], pair_id=["pair"], series_id=["series"],
        molecule_id=["molecule"], low_id=["a"], high_id=["b"], scoring_allowed=[True],
        label_authorized=[True], global_quarantine=[False], exposure=["train_train"]))


def test_physical_delta_join_and_flat_definition():
    labels = pd.DataFrame(dict(sample_id=["b", "a"], V1_ml=[5., 1.], V2_ml=[8., 2.]))
    predictions = pd.DataFrame(dict(sample_id=["a", "b"], V1_q50=[2., 2.], V2_q50=[3., 9.]))
    scored = analysis.score_pairs(pair(), labels, predictions, 42)
    assert scored.delta_true.tolist() == [4., 6.]
    assert scored.delta_pred.tolist() == [0., 6.]
    measure = analysis.metrics(scored.delta_true, scored.delta_pred, .5)
    assert measure["observed_abs_mean"] == 5.
    assert measure["predicted_abs_mean"] == 3.
    assert measure["difference_mae"] == 2.
    assert measure["flat_fraction"] == .5
    assert measure["direction_accuracy"] == .5


def test_equal_models_does_not_pool_pairs_or_confuse_ratios():
    frame = pd.DataFrame(dict(row_seed=[42, 525, 1101], axis=["V/ul"]*3,
        observed_abs_mean=[1., 1., 10.], predicted_abs_mean=[1., 0., 1.],
        amplitude_ratio=[1., 0., .1], n_pairs=[1, 1, 100]))
    result = analysis.equal_model_summary(frame, ["axis"]).iloc[0]
    assert result.n_models == 3
    assert np.isclose(result.amplitude_ratio_mean, 1.1 / 3)
    assert np.isclose(result.ratio_of_equal_model_amplitude_means, 2. / 12.)
    assert result.amplitude_ratio_mean != result.ratio_of_equal_model_amplitude_means


def test_combined_nrmse_is_official_rms_not_arithmetic_mean():
    assert np.isclose(analysis.combined_nrmse([.3, .4]), np.sqrt(.125))
    assert not np.isclose(analysis.combined_nrmse([.3, .4]), .35)
    with pytest.raises(ValueError):
        analysis.combined_nrmse([.3])


def test_quarantine_and_missing_prediction_hard_fail():
    q = sorted(analysis.QUARANTINED_IDS)[0]
    labels = pd.DataFrame(dict(sample_id=["a", "b"], V1_ml=[1., 2.], V2_ml=[3., 4.]))
    pred = pd.DataFrame(dict(sample_id=["a", q], V1_q50=[1., 2.], V2_q50=[3., 4.]))
    with pytest.raises(PermissionError):
        analysis.score_pairs(pair(), labels, pred, 42)
    pred.loc[1, "sample_id"] = "missing"
    with pytest.raises(ValueError):
        analysis.score_pairs(pair(), labels, pred, 42)

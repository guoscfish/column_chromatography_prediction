import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2.efficiency_reporting import METRICS, crossings, evaluate


def test_rebound_and_r2_direction_are_distinguished():
    result = crossings([333, 365, 397, 429], [1., .5, .9, .4], .6)
    assert result["interpolated_labels"] == pytest.approx(358.6)
    assert result["first_observed_labels"] == 365
    assert result["sustained_labels"] == 429
    assert crossings([333, 365, 397], [.2, .8, .3], .7, higher=True)["sustained_labels"] is np.nan
    assert crossings([333, 365], [1., .9], .5)["status"] == "CENSORED"


def synthetic():
    rows, full = [], []
    for seed in (1, 2):
        full.append({"outer_seed": seed, **{m: .9 if m.endswith("R2") else .2 for m in METRICS}})
        for method, errors in (("random", [1., .8, .6]), ("lcmd", [1., .4, .5])):
            for budget, error in zip((333, 365, 397), errors):
                rows.append({"outer_seed": seed, "method": method, "active_label_count": budget,
                             **{m: 1-error if m.endswith("R2") else error for m in METRICS}})
    return pd.DataFrame(rows), pd.DataFrame(full)


def test_endpoint_aulc_gap_and_empirical_saving():
    curves, full = synthetic()
    tables = evaluate(curves, full)
    summary = tables["per_seed_metrics"].query("method == 'lcmd' and metric == 'V1_RMSE'").iloc[0]
    assert summary.aulc == pytest.approx(.575)
    assert summary.gap_closed == pytest.approx(.625)
    saving = tables["label_saving"].query("scope == 'seed' and method == 'lcmd' and metric == 'V1_RMSE' and target == 'Random@397' and crossing == 'first_observed_labels'").iloc[0]
    assert saving.saved_labels == 32
    assert saving.incremental_saving == .5
    assert tables["paired_effects"].query("metric == 'V1_R2' and statistic == 'aulc'").iloc[0].directional_wins == 2


def test_missing_duplicate_and_initial_mismatch_fail_closed():
    curves, full = synthetic()
    with pytest.raises(ValueError, match="schedule"):
        evaluate(curves.iloc[:-1], full)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(pd.concat([curves, curves.iloc[:1]]), full)
    curves.loc[0, "V1_R2"] = -.1
    with pytest.raises(ValueError, match="initial"):
        evaluate(curves, full)


def test_nonpositive_gap_is_undefined_and_censor_is_not_imputed():
    curves, full = synthetic()
    full["V1_RMSE"] = 1.1
    tables = evaluate(curves, full)
    assert tables["labels_to_target"].query("metric == 'V1_RMSE' and target == 'N80'").status.eq("NONPOSITIVE_FULL_GAP").all()
    assert tables["label_saving"].query("target == 'N95'").saved_labels.isna().all()

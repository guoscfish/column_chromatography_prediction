"""Independent arithmetic checks: gain definitions, pooled tail and descriptive counts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import summarize_phase2 as report
import run_phase2 as r


def synthetic_results():
    rows=[]
    for seed in r.SEEDS:
        for c in r.TARGETS:
            for fold in range(1,6):
                for arm in r.old.ARMS:
                    # Mean relative gains positive, ratio-of-means negative, by construction.
                    a=[1.,10.,1.,10.,1.][fold-1]
                    b=[.5,12.,.5,12.,.5][fold-1]
                    rows.append(dict(outer_seed=seed,column=c,inner_fold=fold,arm=arm,
                        combined_normalized_rmse=a if arm=='A1' else b,
                        inner_validation_count=1,V1_rmse=1.,V2_rmse=2.,V1_mae=1.,V2_mae=2.,
                        Center_rmse=1.,Width_rmse=2.,V1_max_abs_error=3.,V2_max_abs_error=4.,
                        Center_max_abs_error=3.,Width_max_abs_error=4.))
    return pd.DataFrame(rows)


def synthetic_tails():
    rows=[]
    for seed in r.SEEDS:
        for c in r.TARGETS:
            for arm in r.old.ARMS:
                for e in ['V1','V2']:
                    rows.extend([dict(outer_seed=seed,column=c,arm=arm,endpoint=e,tail_sse=9.,tail_count=1,total_sse=9.),
                                 dict(outer_seed=seed,column=c,arm=arm,endpoint=e,tail_sse=7.,tail_count=7,total_sse=7.)])
    return pd.DataFrame(rows)


def test_gain_definitions_can_have_opposite_signs_and_seed_win_uses_means():
    paired,methods,seeds,summary,*_=report.compute_statistics(synthetic_results(),synthetic_tails())
    expected_A=(.5-.2+.5-.2+.5)/5*100
    expected_B=(1-25.5/23)*100
    np.testing.assert_allclose(seeds.mean_paired_fold_gain_percent,expected_A)
    np.testing.assert_allclose(seeds.gain_of_means_percent,expected_B)
    assert not seeds.seed_win.any()
    assert summary.seed_wins.eq(0).all() and summary.fold_wins.eq(15).all()


def test_tail_pools_sse_and_count_within_seed_before_seed_summary():
    *_,pooled,pairs,summary=report.compute_statistics(synthetic_results(),synthetic_tails())
    # sqrt((9+7)/(1+7)) != mean(3,1); zeros are not invented for empty folds.
    np.testing.assert_allclose(pooled.tail_rmse,np.sqrt(2))
    assert pooled.tail_count.eq(8).all()
    np.testing.assert_allclose(summary.mean_tail_rmse,np.sqrt(2))


def test_incomplete_or_duplicate_target_observations_rejected():
    result=synthetic_results()
    with pytest.raises(RuntimeError):report.compute_statistics(result.iloc[:-1],synthetic_tails())
    duplicate=pd.concat([result.iloc[:-1],result.iloc[[0]]],ignore_index=True)
    with pytest.raises(RuntimeError):report.compute_statistics(duplicate,synthetic_tails())

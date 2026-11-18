"""Verify crossed-model weighting, paired signs, and response thresholds."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('matched_summary_unit', Path(__file__).parents[1] / 'matched_summary.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mean_AL_campaigns_then_models_not_pair_pool_or_flat_cross():
    rows, pairs = [], []
    for formal, al, n, value in [(42, 157, 100, 0.), (42, 887, 1, 10.), (525, 157, 1, 20.)]:
        row = {'formal_seed': formal, 'al_seed': al, 'axis': 'V/ul', 'exposure': 'both_source_train',
               'method': 'Fully_trained', 'target': 'V1', 'actual_threshold_ml': .5, 'n_pairs': n}
        row.update({metric: value for metric in module.METRICS})
        rows.append(row)
        for index in range(n):
            pairs.append({'formal_seed': formal, 'al_seed': al, 'axis': 'V/ul', 'exposure': 'both_source_train',
                          'method': 'Fully_trained', 'target': 'V1', 'pair_key': f'{al}:{index}',
                          'molecule': 'm', 'series_key': 's'})
    out = module.hierarchical_methods(pd.DataFrame(rows), pd.DataFrame(pairs))
    final = out[out.aggregation_level == 'equal_formal_after_equal_AL_campaigns'].iloc[0]
    assert np.isclose(final.difference_mae, 12.5)
    assert not np.isclose(final.difference_mae, 10.)
    assert final.n_formal_models == 2


def test_paired_improvement_signs_and_identical_truth_cohort():
    rows = []
    for method, mae in [('AL_L525', 4.), ('AL_L1005', 3.), ('AL_L1485', 2.), ('Fully_trained', 1.)]:
        rows.append({'formal_seed': 42, 'al_seed': 157, 'axis': 'V/ul', 'exposure': 'both_source_train',
                     'method': method, 'target': 'V1', 'actual_threshold_ml': .5, 'n_pairs': 2, 'n_meaningful': 2,
                     'observed_abs_mean': 5., 'predicted_abs_mean': 1., 'difference_mae': mae,
                     'direction_accuracy': .5, 'flat_fraction': .5})
    out = module.paired_improvements(pd.DataFrame(rows))
    row = out[(out.aggregation_level == 'formal_AL_campaign') & (out.comparison == 'Full_vs_AL525')].iloc[0]
    assert row.mae_improvement == 3.
    assert row.mae_relative_improvement == .75


def test_meaningful_strict_threshold_and_flat_can_coexist_with_right_sign():
    out = module.numbers([.5, 1., -1.], [.001, .09, -.2], .5)
    assert out['n_meaningful'] == 2
    assert out['direction_accuracy'] == 1.
    assert out['flat_fraction'] == .5
    none = module.numbers([.1], [.01], .5)
    assert np.isnan(none['direction_accuracy']) and np.isnan(none['flat_fraction'])


def test_count_labels_distinguish_overall_from_missing_scientific_data():
    pairs = pd.DataFrame([{'formal_seed': 42, 'al_seed': 157, 'axis': 'V/ul',
                           'exposure': 'both_source_train', 'pair_key': 'p', 'series_key': 's', 'molecule': 'm'}])
    counts = module.denominators(pairs)
    assert counts.exposure.notna().all()
    assert 'all_exposures' in set(counts.exposure)

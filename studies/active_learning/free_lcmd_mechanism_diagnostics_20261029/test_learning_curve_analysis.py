"""Lightweight scientific-contract tests; no training or truth access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('learning_curve_analysis',HERE/'learning_curve_analysis.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_normalized_aulc_uses_budget_span_and_observed_values():
    # Trapezoid is (0.9+0.7)/2 for first interval, (0.7+0.6)/2 for second.
    assert mod.normalized_aulc([333,365,397],[0.9,0.7,0.6],333,397)==pytest.approx(.725)
    with pytest.raises(ValueError):mod.normalized_aulc([333,365],[.9,.7],333,397)
    with pytest.raises(ValueError):mod.normalized_aulc([365,333],[.7,.9],333,365)


def test_nonmonotonic_stage_retains_worsening_and_net_gain():
    s=mod.stage_metrics([333,365,397],[.8,.6,.65],333,397,.4)
    assert s['endpoint_gain']==pytest.approx(.15)
    assert s['worsening_steps']==1 and s['worsening_fraction']==.5
    assert s['total_worsening_magnitude']==pytest.approx(.05)
    assert s['total_variation']==pytest.approx(.25)
    assert s['gain_per_32_labels']==pytest.approx(.075)
    assert s['fraction_of_start_gap_to_full_closed']==pytest.approx(.375)
    with pytest.raises(ValueError):mod.stage_metrics([333,397],[.8,.65],333,397,.4)


def test_full_reference_is_not_forced_to_be_a_lower_bound():
    s=mod.stage_metrics([333,365],[.3,.31],333,365,.35)
    assert s['start_gap_to_full_reference']<0
    assert s['fraction_of_start_gap_to_full_closed'] is None
    assert s['endpoint_gain']<0


def test_threshold_first_hit_is_observed_and_can_be_transient():
    labels=[333,365,397,429];errors=[.6,.39,.43,.38]
    hit=mod.threshold_hit(labels,errors,.4)
    assert hit['first_observed_labels']==365 and not hit['censored']
    assert hit['sustained_from_labels']==429 and not hit['sustained_censored']
    # A final-only sustained hit is descriptive, not evidence of future stability.
    assert hit['sustained_from_labels']==labels[-1]
    miss=mod.threshold_hit(labels,errors,.3)
    assert miss['first_observed_labels'] is None and miss['censored']
    assert miss['censor_labels']==429 and miss['sustained_from_labels'] is None


def test_threshold_sustained_censored_despite_a_first_hit():
    hit=mod.threshold_hit([333,365,397],[.6,.39,.43],.4)
    assert hit['first_observed_labels']==365
    assert hit['sustained_from_labels'] is None and hit['sustained_censored']


def test_crossings_ignore_initial_tie_and_record_observed_brackets():
    c=mod.crossings([333,365,397,429,461],[.8,.6,.4,.5,.3],[.8,.5,.45,.5,.4])
    assert c['crossing_count_excluding_equal_initialization']==1
    assert c['events'][0]['left_observed_labels']==365
    assert c['events'][0]['right_observed_labels']==397
    assert c['first_observed_free_better_labels']==397
    assert c['free_better_at_every_remaining_observed_fit_from']==461
    assert c['equal_fits']==2


def test_frozen_snapshot_reproduces_published_endpoint_and_aulc():
    curves=json.loads((mod.ROOT/mod.CURVES).read_text())
    audit=json.loads((mod.ROOT/mod.AUDIT).read_text())
    for method in mod.METHODS:
        values=[];ends=[]
        for seed in mod.SEEDS:
            rows=curves[str(seed)][method]
            assert [r['labels'] for r in rows]==mod.EXPECTED_LABELS
            values.append(mod.normalized_aulc([r['labels'] for r in rows],
                          [r['nrmse'] for r in rows],333,1485))
            ends.append(rows[-1]['nrmse'])
        published=audit['common_budget_comparison']
        assert np.mean(values)==pytest.approx(published['normalized_aulc_333_to1485'][method]['mean'],abs=1e-12)
        assert np.mean(ends)==pytest.approx(published['endpoint_validation_nrmse'][method]['mean'],abs=1e-12)


def test_six_stage_weighted_aulc_equals_full_curve():
    curves=json.loads((mod.ROOT/mod.CURVES).read_text())
    for seed in mod.SEEDS:
        for method in mod.METHODS:
            y=[r['nrmse'] for r in curves[str(seed)][method]]
            parts=sum((b-a)*mod.normalized_aulc(mod.EXPECTED_LABELS,y,a,b) for a,b in mod.STAGES[:6])/1152
            assert parts==pytest.approx(mod.normalized_aulc(mod.EXPECTED_LABELS,y,333,1485),abs=1e-12)


def test_midstage_gain_is_exactly_gap_closure_but_aulc_has_different_ranking():
    curves=json.loads((mod.ROOT/mod.CURVES).read_text())
    ref=json.loads((mod.ROOT/mod.REFERENCE).read_text())
    references={int(r['seed']):r['full3330'] for r in ref['rows']}
    rounds,stages,summary=mod.compute(curves,references)
    assert len(rounds)==370 and len(stages)==100
    for r in summary['653_to1005_gap_closure_by_seed']:
        assert r['start_free_minus_lcmd']>0
        assert r['free_gain']>r['lcmd_gain']
        assert r['identity_residual']==pytest.approx(0,abs=1e-12)
    middle=summary['stages']['653-1005']
    assert middle[mod.FREE]['normalized_aulc']['mean']>middle['lcmd']['normalized_aulc']['mean']
    assert middle['paired_free_minus_lcmd']['end_nrmse']['mean']<0

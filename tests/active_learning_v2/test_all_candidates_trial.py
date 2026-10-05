"""Full-pool delivery preserves all candidates and rejects illegal direct plans."""
import copy

import pytest

from scripts.studies.run_qgeognn_v2_row_llm_all_candidates_trial import (
    DIRECT_PROMPT, decode_table, encode_catalog, validate_direct)
from src.qgeognn_al.active_learning_v2.scientist_selector import ReadOnlyCatalog
from src.qgeognn_al.active_learning_v2.protocol import stable_hash


def make_catalog():
    def row(i, observed=False):
        data = {'id': str(i), 'smiles': 'CCO', 'conditions': {'PE/EA': '10/1',
                'Density g/ml': 0.99, 'V/ul': 100.0, 'loading solvent': 'PE',
                'Volume of loading solvent/ul': 200}}
        if observed:
            data.update(V1_ml=5.304999828338623, V2_ml=9.0649995803833, record_kind='initial_L333')
        else:
            data.update(pred_V1_ml=7.494732856750488, pred_V2_ml=15.13082504272461,
                        cw_distance_percentile=0.61144)
        return data
    return ReadOnlyCatalog(candidates=[row(i) for i in range(600)],
                           observed=[row(600, True)], pending=[], salt='test')


def packet(catalog):
    p = {'selection_count': 32, 'round': 0, **encode_catalog(catalog)}
    p['packet_sha256'] = stable_hash(p)
    return p


def selection(p):
    return {'type': 'selection', 'packet_sha256': p['packet_sha256'],
            'choices': [{'id': str(i), 'reason': 'test condition contrast', 'hypothesis_id': None}
                        for i in range(568, 600)], 'hypotheses': [],
            'batch_strategy': 'condition contrasts', 'batch_rationale': 'learn response differences',
            'feedback_interpretation': ''}


def test_lossless_delivery_includes_candidates_beyond_retrieval_budget():
    c = make_catalog()
    p = packet(c)
    assert len(p['candidates']['rows']) == 600
    assert len(p['molecules']) == 1
    for name in ('candidates', 'observed', 'pending'):
        assert {r['id']: r for r in decode_table(p, name)} == getattr(c, name)
    assert 'READ-ONLY JSON RELAY' not in DIRECT_PROMPT
    assert validate_direct(selection(p), c, p)['choices'][-1]['id'] == '599'
    assert not c.queries and not c.viewed


@pytest.mark.parametrize('change', [
    lambda r: r['choices'][0].update(id='600'),
    lambda r: r['choices'][0].update(id='unknown'),
    lambda r: r['choices'][0].update(id='599'),
    lambda r: r['choices'].pop(),
    lambda r: r.update(packet_sha256='foreign'),
    lambda r: r.update(type='query'),
    lambda r: r['choices'][0].update(hypothesis_id='missing'),
])
def test_direct_selection_rejects_invalid_batch(change):
    c = make_catalog()
    p = packet(c)
    result = selection(p)
    change(result)
    with pytest.raises(ValueError):
        validate_direct(result, c, p)


def test_hidden_truth_in_candidate_is_rejected_before_delivery():
    c = make_catalog()
    contaminated = copy.deepcopy(c.candidates['0'])
    contaminated = {k: contaminated[k] for k in ('id', 'smiles', 'conditions', 'pred_V1_ml', 'pred_V2_ml')}
    contaminated['V1_ml'] = 123
    with pytest.raises(ValueError, match='unauthorized'):
        ReadOnlyCatalog(candidates=[contaminated], observed=[], pending=[], salt='test')


def test_candidate_cannot_be_cited_as_measured_hypothesis_evidence():
    c = make_catalog()
    p = packet(c)
    result = selection(p)
    result['hypotheses'] = [{'id': 'H1', 'content': 'condition effect', 'basis': 'mixed',
        'supporting_observed_ids': ['599'], 'contradicting_observed_ids': [],
        'confidence': 'medium', 'status': 'proposed',
        'why_it_matters_for_model_learning': 'improve generalization',
        'next_discriminating_question': 'does changing loading alter retention?'}]
    with pytest.raises(ValueError, match='unobserved evidence'):
        validate_direct(result, c, p)
    result['hypotheses'][0]['supporting_observed_ids'] = ['600']
    assert validate_direct(result, c, p)['hypotheses'][0]['supporting_observed_ids'] == ['600']

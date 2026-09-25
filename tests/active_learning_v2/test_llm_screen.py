import json

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.llm_screen import shortlist, validate_response
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select


def test_shortlist_keeps_cw_prefix_excludes_pending_and_is_reproducible():
    generator = np.random.default_rng(12)
    cw, raw = generator.normal(size=(190, 12)), generator.normal(size=(190, 12))
    selection = shortlist(cw, raw, 30, 157, 0)
    assert selection == shortlist(cw, raw, 30, 157, 0)
    prefix = lcmd_tp_select(cw[30:], cw[:30], 32).selected_pool_positions[:16] + 30
    assert selection['pending_positions'] == prefix.tolist()
    candidates = selection['candidate_positions']
    assert len(candidates) == len(set(candidates)) == 128
    assert not set(candidates) & set(prefix)
    assert min(candidates) >= 30
    assert len(set(selection['random16_positions'])) == 16
    assert set(selection['random16_positions']) <= set(candidates)
    assert set(selection['fallback_positions']) == set(candidates)


def test_malformed_response_cannot_escape_candidate_universe():
    candidates = [f'x{i}' for i in range(128)]
    response = {'packet_sha256': 'abc', 'choices': [
        {'id': 'x0', 'reason': 'valid'}, {'id': 'x0', 'reason': 'duplicate'},
        {'id': 'hidden', 'reason': 'not eligible'}, {'id': 'x1', 'reason': ''}]}
    result = validate_response(response, candidates, list(reversed(candidates)), 'abc')
    assert result['response_failed'] and result['fallback_count'] == 15
    ids = [v['id'] for v in result['choices']]
    assert ids[0] == 'x0' and ids[1] == 'x127'
    assert len(ids) == len(set(ids)) == 16 and set(ids) <= set(candidates)
    with pytest.raises(ValueError, match='bound'):
        validate_response(response, candidates, candidates, 'other-packet')


def test_valid_sixteen_need_no_fallback():
    ids = [f'x{i}' for i in range(128)]
    response = {'packet_sha256': 'a', 'choices': [{'id': v, 'reason': 'complementary'} for v in ids[:16]]}
    result = validate_response(response, ids, ids[::-1], 'a')
    assert result['fallback_count'] == 0 and not result['response_failed']


@pytest.mark.parametrize('response', ['not JSON', '[1, 2]', 'null', '42'])
def test_unparseable_or_nonobject_response_uses_frozen_fallback(response):
    ids = [f'x{i}' for i in range(128)]
    result = validate_response(response, ids, ids[::-1], 'a')
    assert result['response_failed'] and result['fallback_count'] == 16
    assert [v['id'] for v in result['choices']] == ids[::-1][:16]


def test_raw_json_response_preserves_valid_choices():
    ids = [f'x{i}' for i in range(128)]
    response = json.dumps({'packet_sha256': 'a', 'choices': [
        {'id': value, 'reason': 'complementary'} for value in ids[:16]]})
    result = validate_response(response, ids, ids[::-1], 'a')
    assert [v['id'] for v in result['choices']] == ids[:16]
    assert not result['response_failed'] and result['fallback_count'] == 0


def test_incomplete_report_cannot_open_label_context(tmp_path, monkeypatch):
    from src.qgeognn_al.active_learning_v2 import llm_screen

    monkeypatch.setattr(llm_screen, 'STUDY', tmp_path)
    monkeypatch.setattr(llm_screen, 'validate', lambda: None)

    def forbidden_context(seed):
        pytest.fail('test-label context opened before all trajectories froze')

    monkeypatch.setattr(llm_screen, 'Context', forbidden_context)
    with pytest.raises(FileNotFoundError):
        llm_screen.report()
    assert not (tmp_path / 'global_pre_test_freeze.json').exists()

"""Offline correction continuation checks: no provider calls or training."""
import copy
from contextlib import nullcontext
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/studies/run_qgeognn_v2_row_llm_nocw_resume.py'
    spec = importlib.util.spec_from_file_location('nocw_resume_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'STUDY', tmp_path)
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'grants': [{'seed': 887, 'round': 3, 'new_answer_limit': 6},
                                           {'seed': 6101, 'round': 8, 'new_answer_limit': 6}]}))
    monkeypatch.setattr(m, 'POLICY', policy)
    monkeypatch.setattr(m.s, 'prompt', lambda variant: 'same frozen prompt')
    monkeypatch.setattr(m.s, 'audit_no_cw', lambda obj: None)
    return m


def setup_selection(m, monkeypatch):
    catalog = SimpleNamespace(pending=[], candidates=list(range(32)), observed=list(range(333)))
    encoded = {'candidates': {'columns': ['id'], 'rows': [[str(i)] for i in range(32)]}}
    monkeypatch.setattr(m.s, 'encode_catalog', lambda c: encoded)
    packet = {**encoded, 'selection_count': 32, 'candidate_count': 32, 'observed_count': 333, 'active_label_count': 333}
    packet['packet_sha256'] = m.s.stable_hash(packet)
    config = {'selection_attempts': 3, 'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh'}

    def check(answer, *args):
        if not answer['valid']:
            raise ValueError('illegal supplied candidate ID')
        return answer
    monkeypatch.setattr(m.s, 'validate_plan', check)
    return packet, catalog, config


@pytest.mark.parametrize('accepted_turn', [3, 5, None])
def test_only_append_three_answers_preserving_original_prefix(adapter, tmp_path, monkeypatch, accepted_turn):
    m = adapter
    packet, catalog, config = setup_selection(m, monkeypatch)
    directory = tmp_path / 'selections/seed_887/llm32_nocw/round_03'
    directory.mkdir(parents=True)
    calls = []

    def request(messages, actual_config, d, turn, call):
        saved = d / f'turn_{turn:02d}.json'
        if saved.exists():
            return json.loads(saved.read_text())
        calls.append((turn, copy.deepcopy(actual_config), copy.deepcopy(messages)))
        return {'answer': json.dumps({'valid': turn == accepted_turn}), 'success': True}
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    with pytest.raises(RuntimeError, match='bounded corrections'):
        m.ORIGINAL_RUN_SELECTION(packet, catalog, directory, 'no_cw', config, None)
    prefix = {p.name: p.read_bytes() for p in directory.iterdir()}
    calls.clear()
    if accepted_turn is None:
        with pytest.raises(RuntimeError, match='six-answer limit'):
            m.run_selection(packet, catalog, directory, 'no_cw', config, None)
        assert not (directory / 'selection.json').exists()
    else:
        assert m.run_selection(packet, catalog, directory, 'no_cw', config, None)['valid']
    assert [x[0] for x in calls] == list(range(3, 6 if accepted_turn is None else accepted_turn+1))
    assert all(x[1] == config for x in calls)  # No request config/hash rewriting.
    assert all((directory / name).read_bytes() == value for name, value in prefix.items())
    assert not (directory / 'request_06.json').exists()
    assert all(json.loads(x[2][1]['content']) == packet for x in calls)


def test_other_rounds_keep_three_answer_limit(adapter, tmp_path, monkeypatch):
    packet, catalog, config = setup_selection(adapter, monkeypatch)
    called = []
    monkeypatch.setattr(adapter, 'ORIGINAL_RUN_SELECTION', lambda *a: called.append(a) or 'original')
    for seed, r in [(887, 4), (6101, 7), (157, 3), (2357, 8), (12203, 10)]:
        assert adapter.run_selection(packet, catalog, tmp_path / f'selections/seed_{seed}/llm32_nocw/round_{r:02d}', 'no_cw', config, None) == 'original'
    assert len(called) == 5


def test_complete_catalog_binding_still_checked(adapter, tmp_path, monkeypatch):
    packet, catalog, config = setup_selection(adapter, monkeypatch)
    packet['candidate_count'] = 31
    with pytest.raises(ValueError, match='count mismatch'):
        adapter.run_selection(packet, catalog, tmp_path / 'selections/seed_887/llm32_nocw/round_03', 'no_cw', config, None)


def test_resume_worker_skips_completed_rounds(adapter, tmp_path, monkeypatch):
    m = adapter
    monkeypatch.setattr(m, 'activate', lambda seed: None)
    monkeypatch.setattr(m, 'validate', lambda: None)
    monkeypatch.setattr(m, 'resume_round', lambda seed: 3)
    monkeypatch.setattr(m.stream, 'unresolved', lambda seed: [])
    monkeypatch.setattr(m.base, 'credential_audit', lambda: {})
    monkeypatch.setattr(m.base, 'KEY_FILES', {887: tmp_path/'mock.key'})
    monkeypatch.setattr(m.base, 'BASE_EXCLUSIVE_LOCK', lambda path: nullcontext())
    monkeypatch.setattr(m.base, 'status_record', lambda *a, **kw: None)
    monkeypatch.setattr(m.s, 'require_authorization', lambda: None)
    monkeypatch.setattr(m.s, 'BUDGETS', (333, 365, 397, 429, 461, 493))
    calls = []
    monkeypatch.setattr(m.s, 'select', lambda seed, method, r, **kw: calls.append(('select', method, r)))
    monkeypatch.setattr(m.s, 'advance', lambda seed, method, r, **kw: calls.append(('fit', method, r)))
    m.worker(887)
    assert calls == [('select', 'llm32_nocw', 3), ('fit', 'llm32_nocw', 3),
                     ('select', 'llm32_nocw', 4), ('fit', 'llm32_nocw', 4)]


def test_active_seed_prevents_launch_before_controller_signal(adapter, monkeypatch):
    m = adapter
    monkeypatch.setattr(m, 'activate', lambda: None)
    monkeypatch.setattr(m, 'validate', lambda: None)
    monkeypatch.setattr(m.s, 'require_authorization', lambda: None)
    monkeypatch.setattr(m.base, 'credential_audit', lambda: {})
    monkeypatch.setattr(m.base, 'BASE_EXCLUSIVE_LOCK', lambda path: nullcontext())
    monkeypatch.setattr(m.retry, 'seed_busy', lambda seed: True)
    monkeypatch.setattr(m.os, 'kill', lambda *a: pytest.fail('inflight worker/controller interrupted'))
    with pytest.raises(RuntimeError, match='active or ambiguous'):
        m.launch()


@pytest.mark.parametrize('seed,round_index', [(887, 3), (6101, 8)])
def test_real_failed_prefix_replays_exactly_without_paid_call(adapter, tmp_path, monkeypatch, seed, round_index):
    m = adapter
    source = m.s.ROOT / ('studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/'
                        f'five_seed_nocw_to1005_20261029/selections/seed_{seed}/llm32_nocw/round_{round_index:02d}')
    directory = tmp_path / f'selections/seed_{seed}/llm32_nocw/round_{round_index:02d}'
    shutil.copytree(source, directory)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    from src.qgeognn_al.active_learning_v2 import nocw_inputs
    monkeypatch.setattr(m.s, 'prompt', m.base.prompt)
    monkeypatch.setattr(m.s, 'validate_plan', m.base.validate_plan)
    monkeypatch.setattr(m.s, 'audit_no_cw', nocw_inputs.audit_no_cw)
    def request(messages, config, d, turn, call):
        if turn == 3:
            raise LookupError('offline boundary before first new answer')
        return json.loads((d / f'turn_{turn:02d}.json').read_text())
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    packet = json.loads((directory / 'input.json').read_text())
    catalog = m.s.ReadOnlyCatalog(**json.loads((directory / 'catalog.json').read_text()))
    config = json.loads((directory / 'request_00.json').read_text())['config']
    with pytest.raises(LookupError, match='offline boundary'):
        m.run_selection(packet, catalog, directory, 'no_cw', config, None)
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    request3 = json.loads((directory / 'request_03.json').read_text())
    assert request3['config'] == config
    assert json.loads(request3['messages'][1]['content']) == packet


def test_validate_cli_initializes_five_seed_context(adapter, monkeypatch):
    seen = []
    monkeypatch.setattr(adapter.stream, 'activate', lambda: seen.append('activate'))
    monkeypatch.setattr(adapter, 'validate', lambda: seen.append('validate') or {'status': 'VALID'})
    monkeypatch.setattr(adapter.sys, 'argv', ['resume', 'validate'])
    adapter.main()
    assert seen == ['activate', 'validate']

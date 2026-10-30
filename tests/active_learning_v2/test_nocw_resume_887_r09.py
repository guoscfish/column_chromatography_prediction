"""Offline checks for the authorized seed887 round09 grant; no provider calls."""
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/studies/run_qgeognn_v2_row_llm_nocw_resume_887_r09.py'
    spec = importlib.util.spec_from_file_location('nocw_resume887r09_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'STUDY', tmp_path)
    monkeypatch.setattr(m.prior, 'STUDY', tmp_path)
    old = tmp_path / 'original_grants.json'
    old.write_text(json.dumps({'grants': [{'seed': 887, 'round': 3, 'new_answer_limit': 6},
                                        {'seed': 6101, 'round': 8, 'new_answer_limit': 6}]}))
    monkeypatch.setattr(m.prior, 'POLICY', old)
    chain = [(m.predecessor, 6101, 11), (m.predecessor.predecessor, 6101, 10),
             (m.predecessor.predecessor.predecessor, 2357, 10), (m, 887, 9)]
    for module, seed, round_index in chain:
        policy = tmp_path / f'policy_{seed}_{round_index}.json'
        policy.write_text(json.dumps({'grant': {'seed': seed, 'round': round_index, 'new_answer_limit': 6}}))
        monkeypatch.setattr(module, 'STUDY', tmp_path)
        monkeypatch.setattr(module, 'POLICY', policy)
    monkeypatch.setattr(m.prior, 'answer_limit', m.answer_limit)
    monkeypatch.setattr(m.s, 'prompt', lambda variant: 'same frozen prompt')
    monkeypatch.setattr(m.s, 'audit_no_cw', lambda messages: None)
    return m


@pytest.mark.parametrize('seed,round_index,limit', [
    (887,9,6), (887,10,3), (887,3,6), (887,4,3), (6101,8,6), (6101,10,6),
    (6101,11,6), (6101,12,3), (2357,10,6), (2357,15,3), (157,9,3), (12203,9,3),
])
def test_six_grants_are_narrow_and_pending_round15_stays_three(adapter, seed, round_index, limit):
    directory = adapter.STUDY / f'selections/seed_{seed}/llm32_nocw/round_{round_index:02d}'
    assert adapter.answer_limit(directory, {'selection_attempts': 3}) == limit


@pytest.mark.parametrize('accepted_turn', [3,5,None])
def test_only_three_new_answers_and_one_hint_without_rewriting_prefix(adapter, monkeypatch, accepted_turn):
    m = adapter
    directory = m.failed_directory()
    directory.mkdir(parents=True)
    catalog = SimpleNamespace(pending=[], candidates=list(range(32)), observed=list(range(621)))
    encoded = {'candidates': {'columns': ['id'], 'rows': [[str(i)] for i in range(32)]}}
    monkeypatch.setattr(m.s, 'encode_catalog', lambda c: encoded)
    packet = {**encoded, 'selection_count': 32, 'candidate_count': 32,
              'observed_count': 621, 'active_label_count': 621}
    packet['packet_sha256'] = m.s.stable_hash(packet)
    config = {'selection_attempts': 3, 'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh'}
    calls = []
    def request(messages, actual_config, directory, turn, call):
        saved = directory / f'turn_{turn:02d}.json'
        if saved.exists():
            return json.loads(saved.read_text())
        calls.append((turn, copy.deepcopy(messages), copy.deepcopy(actual_config)))
        return {'answer': json.dumps({'valid': turn == accepted_turn})}
    def check(answer, *args):
        if not answer['valid']:
            raise ValueError('selection packet binding mismatch')
        return answer
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    monkeypatch.setattr(m.s, 'validate_plan', check)
    with pytest.raises(RuntimeError, match='bounded corrections'):
        m.prior.ORIGINAL_RUN_SELECTION(packet, catalog, directory, 'no_cw', config, None)
    prefix = {p.name: p.read_bytes() for p in directory.iterdir()}
    calls.clear()
    if accepted_turn is None:
        with pytest.raises(RuntimeError, match='six-answer limit'):
            m.run_selection(packet, catalog, directory, 'no_cw', config, None)
    else:
        assert m.run_selection(packet, catalog, directory, 'no_cw', config, None)['valid']
    assert [c[0] for c in calls] == list(range(3, 6 if accepted_turn is None else accepted_turn + 1))
    assert all(c[2] == config and json.loads(c[1][1]['content']) == packet for c in calls)
    assert all(sum(msg['content'] == m.packet_hash_hint(packet) for msg in c[1]) == 1 for c in calls)
    assert all((directory / name).read_bytes() == value for name, value in prefix.items())
    assert not (directory / 'request_06.json').exists()


def real_prefix(adapter, monkeypatch, seed, round_index):
    m = adapter
    source = m.s.ROOT / ('studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/'
                         f'five_seed_nocw_to1005_20261029/selections/seed_{seed}/llm32_nocw/round_{round_index:02d}')
    directory = m.STUDY / f'selections/seed_{seed}/llm32_nocw/round_{round_index:02d}'
    shutil.copytree(source, directory)
    from src.qgeognn_al.active_learning_v2 import nocw_inputs
    monkeypatch.setattr(m.s, 'prompt', m.base.prompt)
    monkeypatch.setattr(m.s, 'validate_plan', m.base.validate_plan)
    monkeypatch.setattr(m.s, 'audit_no_cw', nocw_inputs.audit_no_cw)
    return directory


@pytest.mark.parametrize('seed,round_index', [(887,9), (6101,11)])
def test_real_new_and_previous_hash_hint_prefixes_replay_without_provider(adapter, monkeypatch, seed, round_index):
    m = adapter
    directory = real_prefix(m, monkeypatch, seed, round_index)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    def request(messages, config, directory, turn, call):
        if turn == 3:
            raise LookupError('offline boundary before any new paid answer')
        return json.loads((directory / f'turn_{turn:02d}.json').read_text())
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    packet = json.loads((directory / 'input.json').read_text())
    catalog = m.s.ReadOnlyCatalog(**json.loads((directory / 'catalog.json').read_text()))
    config = json.loads((directory / 'request_00.json').read_text())['config']
    with pytest.raises(LookupError, match='offline boundary'):
        m.run_selection(packet, catalog, directory, 'no_cw', config, None)
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    new = json.loads((directory / 'request_03.json').read_text())
    assert new['config'] == config and json.loads(new['messages'][1]['content']) == packet
    assert new['messages'][-1] == {'role': 'user', 'content': m.packet_hash_hint(packet)}


def test_original_63_character_answer_is_still_rejected_after_reminder(adapter, monkeypatch):
    m = adapter
    directory = real_prefix(m, monkeypatch, 887, 9)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    malformed = json.loads((directory / 'turn_02.json').read_text())
    calls = []
    def request(messages, config, directory, turn, call):
        if turn < 3:
            return json.loads((directory / f'turn_{turn:02d}.json').read_text())
        calls.append(turn)
        return malformed
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    packet = json.loads((directory / 'input.json').read_text())
    catalog = m.s.ReadOnlyCatalog(**json.loads((directory / 'catalog.json').read_text()))
    config = json.loads((directory / 'request_00.json').read_text())['config']
    with pytest.raises(RuntimeError, match='six-answer limit'):
        m.run_selection(packet, catalog, directory, 'no_cw', config, None)
    assert calls == [3,4,5] and not (directory / 'selection.json').exists()
    assert not (directory / 'request_06.json').exists()
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    assert len(json.loads(malformed['answer'])['packet_sha256']) == 63


@pytest.mark.parametrize('busy,unresolved', [(True,[]), (False,['started'])])
def test_active_or_unresolved_target_blocks_launch_before_any_signal(adapter, monkeypatch, busy, unresolved):
    m = adapter
    monkeypatch.setattr(m, 'activate', lambda: None)
    monkeypatch.setattr(m, 'validate', lambda: None)
    monkeypatch.setattr(m.s, 'require_authorization', lambda: None)
    monkeypatch.setattr(m.base, 'credential_audit', lambda: {})
    monkeypatch.setattr(m.base, 'BASE_EXCLUSIVE_LOCK', lambda p: nullcontext())
    monkeypatch.setattr(m.retry, 'seed_busy', lambda seed: busy)
    monkeypatch.setattr(m.stream, 'unresolved', lambda seed: unresolved)
    monkeypatch.setattr(m.os, 'kill', lambda *a: pytest.fail('inflight worker interrupted'))
    with pytest.raises(RuntimeError, match='active or ambiguous'):
        m.launch()


def test_worker_reuses_completed_fits_and_runs_only_authorized_seed_nocw(adapter, monkeypatch):
    m = adapter
    monkeypatch.setattr(m, 'activate', lambda seed: None)
    monkeypatch.setattr(m, 'validate', lambda: None)
    monkeypatch.setattr(m.prior, 'resume_round', lambda seed: 9)
    monkeypatch.setattr(m.stream, 'unresolved', lambda seed: [])
    monkeypatch.setattr(m.s, 'require_authorization', lambda: None)
    monkeypatch.setattr(m.base, 'credential_audit', lambda: {})
    monkeypatch.setattr(m.base, 'KEY_FILES', {887: m.STUDY / 'mock.key'})
    monkeypatch.setattr(m.base, 'BASE_EXCLUSIVE_LOCK', lambda p: nullcontext())
    monkeypatch.setattr(m.base, 'status_record', lambda *a, **kw: None)
    monkeypatch.setattr(m.s, 'BUDGETS', tuple(333 + 32 * r for r in range(12)))
    calls = []
    monkeypatch.setattr(m.s, 'select', lambda seed, method, r, **kw: calls.append(('select',seed,method,r)))
    monkeypatch.setattr(m.s, 'advance', lambda seed, method, r, **kw: calls.append(('fit',seed,method,r)))
    m.worker()
    assert calls == [(kind,887,'llm32_nocw',r) for r in (9,10) for kind in ('select','fit')]


def test_controller_recognizes_all_six_worker_entrypoints(adapter, monkeypatch):
    m = adapter
    entries = (m.stream.SCRIPT, m.prior.SCRIPT, m.predecessor.predecessor.predecessor.SCRIPT,
               m.predecessor.predecessor.SCRIPT, m.predecessor.SCRIPT, m.SCRIPT)
    for entry in entries:
        monkeypatch.setattr(m.stream, 'process_info', lambda pid: f'Ss python {entry} worker --seed 6101 --allow-paid-llm')
        m.existing_worker_identity(6101, 123)
    monkeypatch.setattr(m.stream, 'process_info', lambda pid: 'Ss unrelated worker --seed 6101')
    with pytest.raises(RuntimeError, match='identity mismatch'):
        m.existing_worker_identity(6101, 123)


def test_other_rounds_delegate_and_no_new_hint_is_added(adapter, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter, 'BASE_RUN_SELECTION', lambda *args: calls.append(args) or 'delegated')
    for seed, round_index in [(887,3),(887,10),(6101,11),(6101,12),(2357,15),(157,9)]:
        directory = adapter.STUDY / f'selections/seed_{seed}/llm32_nocw/round_{round_index:02d}'
        assert adapter.run_selection({}, None, directory, 'no_cw', {}, None) == 'delegated'
    assert len(calls) == 6


def test_hint_requires_complete_original_hash(adapter):
    with pytest.raises(ValueError, match='complete'):
        adapter.packet_hash_hint({'packet_sha256': 'a' * 63})
    value = 'a' * 63 + '2'
    assert value in adapter.packet_hash_hint({'packet_sha256': value})

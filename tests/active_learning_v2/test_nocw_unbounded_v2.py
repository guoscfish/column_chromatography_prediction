"""Offline checks for continuous correction authorization; never call a provider."""
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
    source = Path(__file__).resolve().parents[2] / 'scripts/studies/run_qgeognn_v2_row_llm_nocw_unbounded_v2.py'
    spec = importlib.util.spec_from_file_location('nocw_unbounded_test', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'STUDY', tmp_path)
    monkeypatch.setattr(m, 'POLICY', tmp_path / 'unbounded_policy.json')
    monkeypatch.setattr(m.s, 'prompt', lambda variant: 'same frozen prompt')
    monkeypatch.setattr(m.s, 'audit_no_cw', lambda messages: None)
    return m


def fake_packet(m, monkeypatch):
    catalog = SimpleNamespace(pending=[], candidates=list(range(32)), observed=list(range(973)))
    encoded = {'candidates': {'columns': ['id'], 'rows': [[str(i)] for i in range(32)]}}
    monkeypatch.setattr(m.s, 'encode_catalog', lambda c: encoded)
    packet = {**encoded, 'selection_count': 32, 'candidate_count': 32,
              'observed_count': 973, 'active_label_count': 973}
    packet['packet_sha256'] = m.s.stable_hash(packet)
    return catalog, packet, {'selection_attempts': 3, 'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh'}


@pytest.mark.parametrize('seed,rnd', [(157,17), (887,13), (2357,15), (6101,20),
                                    (157,18), (887,14), (2357,16), (12203,20)])
def test_no_answer_cap_even_beyond_99_and_no_hint_outside_three_explicit_rounds(adapter, monkeypatch, seed, rnd):
    m = adapter
    directory = m.directory_for(seed, rnd)
    directory.mkdir(parents=True)
    catalog, packet, config = fake_packet(m, monkeypatch)
    calls = []
    def request(messages, actual_config, directory, turn, call):
        calls.append(turn)
        assert actual_config == config and json.loads(messages[1]['content']) == packet
        expected_hints = int((seed, rnd) in m.HASH_HINT_ROUNDS and turn >= 3)
        assert sum(msg['content'] == m.packet_hash_hint(packet) for msg in messages) == expected_hints
        return {'answer': json.dumps({'valid': turn == 102})}
    def check(result, *args):
        if not result['valid']:
            raise ValueError('must select distinct legal supplied candidate IDs')
        return result
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    monkeypatch.setattr(m.s, 'validate_plan', check)
    assert m.run_selection(packet, catalog, directory, 'no_cw', config, None)['valid']
    assert calls == list(range(103))
    assert (directory / 'request_102.json').exists()
    assert len(list(directory.glob('rejected_*.json'))) == 102
    assert not (directory / 'request_103.json').exists()


def real_prefix(m, monkeypatch, seed, rnd):
    study = m.s.ROOT / ('studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/'
                        'revisions/five_seed_nocw_to1005_20261029')
    directory = m.directory_for(seed, rnd)
    directory.mkdir(parents=True)
    if seed in m.prepared_draft.RESUME and m.prepared_draft.RESUME[seed] == rnd:
        policy = json.loads((study / f'progress/failure_seed_{seed}_round{rnd:02d}_audit.json').read_text())
        names = policy['preserved_prefix']
    else:
        policies = {
            (887,3): 'correction_resume_amendment_20261030_v2.json',
            (6101,8): 'correction_resume_amendment_20261030_v2.json',
            (2357,10): 'correction_resume_seed2357_round10_20261030.json',
            (6101,10): 'correction_resume_seed6101_round10_20261030.json',
            (6101,11): 'correction_resume_seed6101_round11_20261030.json',
            (887,9): 'correction_resume_seed887_round09_20261030.json',
        }
        policy = json.loads((study / policies[(seed,rnd)]).read_text())
        grants = policy.get('grants', [policy.get('grant')])
        names = next(g['preserved_prefix'] for g in grants if (g['seed'],g['round']) == (seed,rnd))
    for name, digest in names.items():
        source = study / name
        assert m.s.sha256_file(source) == digest
        shutil.copyfile(source, directory / source.name)
    from src.qgeognn_al.active_learning_v2 import nocw_inputs
    monkeypatch.setattr(m.s, 'prompt', m.base.prompt)
    monkeypatch.setattr(m.s, 'validate_plan', m.base.validate_plan)
    monkeypatch.setattr(m.s, 'audit_no_cw', nocw_inputs.audit_no_cw)
    packet = json.loads((directory / 'input.json').read_text())
    catalog = m.s.ReadOnlyCatalog(**json.loads((directory / 'catalog.json').read_text()))
    config = json.loads((directory / 'request_00.json').read_text())['config']
    return directory, packet, catalog, config


@pytest.mark.parametrize('seed,rnd', [(157,17), (887,13), (2357,15), (6101,20),
                                    (887,3), (6101,8), (2357,10), (6101,10), (6101,11), (887,9)])
def test_every_original_failure_and_six_previous_grants_replay_without_mutation(adapter, monkeypatch, seed, rnd):
    m = adapter
    directory, packet, catalog, config = real_prefix(m, monkeypatch, seed, rnd)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    def request(messages, actual_config, directory, turn, call):
        if turn == 3:
            raise LookupError('offline boundary before new paid answer')
        return json.loads((directory / f'turn_{turn:02d}.json').read_text())
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    with pytest.raises(LookupError, match='offline boundary'):
        m.run_selection(packet, catalog, directory, 'no_cw', config, None)
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    request3 = json.loads((directory / 'request_03.json').read_text())
    assert request3['config'] == config and json.loads(request3['messages'][1]['content']) == packet
    assert sum(msg['content'] == m.packet_hash_hint(packet) for msg in request3['messages']) == int((seed,rnd) in m.HASH_HINT_ROUNDS)


@pytest.mark.parametrize('bad_turn,expected', [(0,'binding mismatch'), (1,'distinct legal'), (2,'binding mismatch')])
def test_wrong_64_character_hashes_and_illegal_ids_stay_rejected_without_cap(adapter, monkeypatch, bad_turn, expected):
    m = adapter
    directory, packet, catalog, config = real_prefix(m, monkeypatch, 6101, 20)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    bad = json.loads((directory / f'turn_{bad_turn:02d}.json').read_text())
    assert len(json.loads(bad['answer'])['packet_sha256']) == 64
    calls = []
    def request(messages, actual_config, directory, turn, call):
        if turn < 3:
            return json.loads((directory / f'turn_{turn:02d}.json').read_text())
        if turn == 8:
            raise LookupError('offline boundary')
        calls.append(turn)
        assert actual_config == config and json.loads(messages[1]['content']) == packet
        return copy.deepcopy(bad)
    monkeypatch.setattr(m.s, 'request_with_retries', request)
    with pytest.raises(LookupError, match='offline boundary'):
        m.run_selection(packet, catalog, directory, 'no_cw', config, None)
    assert calls == [3,4,5,6,7] and not (directory / 'selection.json').exists()
    assert expected in json.loads((directory / 'rejected_07.json').read_text())['error']
    assert all((directory / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('seed', [157,887,2357])
def test_worker_reuses_completed_fits_and_runs_only_remaining_nocw_rounds(adapter, monkeypatch, seed):
    m = adapter
    monkeypatch.setattr(m, 'activate', lambda seed: None)
    monkeypatch.setattr(m, 'validate', lambda: None)
    monkeypatch.setattr(m.prior, 'resume_round', lambda seed: m.RESUME[seed])
    monkeypatch.setattr(m.stream, 'unresolved', lambda seed: [])
    monkeypatch.setattr(m.s, 'require_authorization', lambda: None)
    monkeypatch.setattr(m.base, 'credential_audit', lambda: {})
    monkeypatch.setattr(m.base, 'KEY_FILES', {seed: m.STUDY / 'mock.key'})
    monkeypatch.setattr(m.base, 'BASE_EXCLUSIVE_LOCK', lambda p: nullcontext())
    monkeypatch.setattr(m.base, 'status_record', lambda *a, **kw: None)
    calls = []
    monkeypatch.setattr(m.s, 'select', lambda seed, method, r, **kw: calls.append(('select',seed,method,r)))
    monkeypatch.setattr(m.s, 'advance', lambda seed, method, r, **kw: calls.append(('fit',seed,method,r)))
    m.POLICY.write_text('{}')
    m.worker(seed)
    assert calls == [(kind,seed,'llm32_nocw',r) for r in range(m.RESUME[seed],21) for kind in ('select','fit')]
    receipt = json.loads((m.STUDY / f'progress/resume_unbounded_v2_prefix_seed_{seed}.json').read_text())
    assert receipt['completed_fits_audited_and_skipped'] == m.RESUME[seed]
    assert receipt['new_model_calls_during_audit'] == 0


@pytest.mark.parametrize('seed', [6101,12203])
def test_completed_seed_is_never_redispatched(adapter, seed):
    with pytest.raises(ValueError, match='completed seed'):
        adapter.worker(seed)


@pytest.mark.parametrize('busy,unresolved', [(True,[]), (False,['started'])])
def test_active_or_ambiguous_target_blocks_launch_before_signalling_controller(adapter, monkeypatch, busy, unresolved):
    m = adapter
    monkeypatch.setattr(m, 'activate', lambda: None)
    monkeypatch.setattr(m, 'validate', lambda: None)
    monkeypatch.setattr(m.s, 'require_authorization', lambda: None)
    monkeypatch.setattr(m.base, 'credential_audit', lambda: {})
    monkeypatch.setattr(m.base, 'BASE_EXCLUSIVE_LOCK', lambda p: nullcontext())
    monkeypatch.setattr(m.retry, 'seed_busy', lambda seed: busy)
    monkeypatch.setattr(m.stream, 'unresolved', lambda seed: unresolved)
    monkeypatch.setattr(m.os, 'kill', lambda *a: pytest.fail('inflight process interrupted'))
    with pytest.raises(RuntimeError, match='active or ambiguous'):
        m.launch()


def test_protocol_failure_propagates_without_request_retry(adapter, monkeypatch):
    m = adapter
    d = m.directory_for(887,14)
    d.mkdir(parents=True)
    catalog, packet, config = fake_packet(m, monkeypatch)
    calls = []
    def fail(*args):
        calls.append(1)
        raise m.stream.FatalTransportStop('wrong served model')
    monkeypatch.setattr(m.s, 'request_with_retries', fail)
    with pytest.raises(m.stream.FatalTransportStop, match='wrong served model'):
        m.run_selection(packet,catalog,d,'no_cw',config,None)
    assert len(calls) == 1 and not (d/'request_01.json').exists()


def test_failed_worker_receipts_prevent_automatic_redispatch(adapter, monkeypatch):
    m = adapter
    (m.STUDY/'progress').mkdir()
    m.POLICY.write_text('{}')
    for seed in m.base.SEEDS:
        (m.STUDY/f'progress/seed_{seed}.json').write_text(json.dumps({'seed':seed,'phase':'COMPLETE' if seed in (6101,12203) else 'FAILED_CLOSED'}))
        if seed not in (6101,12203):
            (m.STUDY/f'progress/resume_unbounded_v2_worker_seed_{seed}.json').write_text('{}')
    monkeypatch.setattr(m,'activate',lambda:None)
    monkeypatch.setattr(m,'validate',lambda:None)
    monkeypatch.setattr(m.s,'require_authorization',lambda:None)
    monkeypatch.setattr(m.base,'BASE_EXCLUSIVE_LOCK',lambda p:nullcontext())
    monkeypatch.setattr(m.retry,'seed_busy',lambda seed:False)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('abnormal worker restarted'))
    def stop(*a):
        raise LookupError('one supervisor iteration')
    monkeypatch.setattr(m.time,'sleep',stop)
    with pytest.raises(LookupError):
        m.supervise()
    rows=json.loads((m.STUDY/'progress/stream_supervisor.json').read_text())['workers']
    assert [r['supervision'] for r in rows] == ['NEEDS_RECONCILIATION']*3+['COMPLETE']*2

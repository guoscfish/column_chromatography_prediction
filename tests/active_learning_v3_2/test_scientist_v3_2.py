"""Synthetic-only tests. No real model calls, candidate reveals or test evaluation."""
import copy
import json
from pathlib import Path
import subprocess
import pytest
from src.qgeognn_al.active_learning_v3_2 import catalog as catalogs, memory, schema, selector, transport, protocol, study
from src.qgeognn_al.active_learning_v3_2.artifacts import once, read, file_hash, verify_audit


def row(i, observed=False, smiles='CCO', pe='1/1', load=1):
    result = {'id': str(i), 'smiles': smiles, 'conditions': {'PE/EA': pe, 'Density g/ml': 1.,
        'V/ul': load, 'loading solvent': 'EA', 'Volume of loading solvent/ul': 100.}}
    if observed:
        result.update(V1_ml=1., V2_ml=10., record_kind='initial_L333')
    else:
        result.update(pred_V1_ml=2., pred_V2_ml=20., cw_distance_percentile=.5)
        for t, value in (('V1', 2.), ('V2', 20.)):
            result.update({f'pred_{t}_q10_ml': value/2, f'pred_{t}_q50_ml': value, f'pred_{t}_q90_ml': value*2})
    return result


def raw(hybrid=False):
    return {'candidates': [row(i, pe=f'{i%3}/1', load=i%4+1) for i in range(1, 101)],
            'observed': [row('o', True)], 'pending': [row(f'p{i}') for i in range(16)] if hybrid else [],
            'salt': 'synthetic', 'target_scales': [1., 100.]}


def setup(hybrid=False, **kwargs):
    method = ('cw16_synthetic' if hybrid else study.METHODS[0])
    ledger = memory.new_ledger(157, method)
    c = catalogs.ReadOnlyCatalog(**raw(hybrid), ledger=ledger, **kwargs)
    p = selector.make_packet(c, ledger, seed=157, method=method, round_index=0, selection_count=16 if hybrid else 32)
    return c, p, ledger


def query(args=None, operation='search_candidates', state=None):
    return {'type': 'query', 'queries': [{'operation': operation, 'args': args or {}}],
            'working_state': state or schema.empty_working_state()}


def plan(c, p):
    return {'type': 'selection', 'packet_sha256': p['packet_sha256'],
        'choices': [{'id': i, 'reason': 'A synthetic matched contrast.', 'hypothesis_id': None,
                     'scientific_role': 'matched_control'} for i in sorted(c.viewed)[:p['selection_count']]],
        'hypothesis_updates': [], 'feedback_interpretation': '' if p['round'] == 0 else 'Update from measured controls.',
        'batch_strategy': 'Test local contrasts.', 'batch_rationale': 'Resolve local response curves.',
        'open_scientific_questions': []}


def scripted(c, p, answers=None):
    calls = []
    def call(messages, config):
        calls.append(copy.deepcopy(messages))
        index = len(calls)-1
        response = answers[index] if answers and index < len(answers) else (
            query({'offset': index*24}) if index < 2 else plan(c, p))
        return json.dumps(response), {'mock': True, 'usage': {'input_tokens': 123, 'output_tokens': 20,
                                                            'output_tokens_details': {'reasoning_tokens': 10}}}
    return call, calls


def config():
    return transport.settings('responses', base_url='https://example.invalid')


def belief(statement='Loading affects retention', status='proposed', evidence=()):
    return {'statement': statement, 'status': status, 'confidence': 'low', 'basis': 'mixed',
            'supporting_observed_ids': list(evidence), 'contradicting_observed_ids': [],
            'why_it_matters_for_model_learning': 'Discriminate a local response curve.',
            'next_discriminating_question': 'Does loading alter the response?'}


def update(ident='H0001', action='create', value=None):
    return {'id': ident, 'action': action, 'reason': 'Measured contrast.',
            'belief': None if action == 'retain' else (value or belief())}


@pytest.mark.parametrize('group', ['candidates', 'pending'])
@pytest.mark.parametrize('field', ['V1_ml', 'V2_ml', 'true_V1_ml', 'test_error', 'premeasurement_error_V2_ml'])
def test_candidate_pending_truth_rejected(group, field):
    data = raw(True)
    data[group][0][field] = 100
    with pytest.raises(ValueError, match='unauthorized'):
        catalogs.ReadOnlyCatalog(**data)


@pytest.mark.parametrize('operation', catalogs.CONVENIENCE)
def test_convenience_boundary_budget_and_determinism(operation):
    data = raw(True)
    a = catalogs.ReadOnlyCatalog(**data, view_budget=3, query_budget=2)
    b = catalogs.ReadOnlyCatalog(**data, view_budget=3, query_budget=2)
    args = {'reference_id': 'o', 'limit': 24}
    if operation == 'get_matched_loading_contrasts':
        args['solvent_tolerance'] = 1
    if operation == 'get_matched_solvent_contrasts':
        args['loading_relative_tolerance'] = 1
    first = a.query(operation, args)
    assert first == b.query(operation, args)
    assert len(a.viewed) <= 3
    for r in first['matches']:
        if r['catalog_role'] != 'observed':
            assert not ({'V1_ml', 'V2_ml'} & r.keys())
        if r['catalog_role'] == 'candidate':
            assert r['id'] in a.viewed
    a.query(operation, {**args, 'offset': 24})
    assert len(a.viewed) <= 3
    with pytest.raises(ValueError, match='budget'):
        a.query(operation, args)


def test_query_errors_do_not_consume_scientific_budget():
    c = catalogs.ReadOnlyCatalog(**raw(), query_budget=3)
    for op, args in [('get', {'id':['1']}), ('search_candidates', {'descriptor':'molecule_novelty'}), ('search_candidates', {'filters':[{'field':'bad','op':'eq','value':1}]})]:
        with pytest.raises(ValueError): c.query(op, args)
    assert len(c.queries) == 0 and len(c.invalid_queries) == 3
    assert c.query('get', {'ids':['1']})['normalized_query_alias'] is True


def test_series_sorting_and_matched_controls():
    c = catalogs.ReadOnlyCatalog(**raw(True))
    rows = c.query('get_condition_series', {'reference_id': 'o'})['matches']
    keys = [(catalogs.pe_fraction(r['conditions']['PE/EA']), r['loading_amount_mg']) for r in rows]
    assert keys == sorted(keys)
    loading = c.query('get_matched_loading_contrasts', {'reference_id': 'o'})['matches']
    assert loading and all(r['conditions']['PE/EA'] == '1/1' and r['loading_amount_mg'] != 1 for r in loading)
    solvent = c.query('get_matched_solvent_contrasts', {'reference_id': 'o'})['matches']
    assert solvent and all(r['loading_amount_mg'] == 1 and r['conditions']['PE/EA'] != '1/1' for r in solvent)
    assert catalogs.pe_fraction('0/1') == 0


def test_public_spans_and_pool_summary_ignore_measured_truth():
    c, p, _ = setup()
    assert 'V1_quantile_width_ml' not in json.dumps(p)
    assert 'predicted_q10_q90_span_V1_ml' in json.dumps(p)
    summary = c.pool_summary()
    c.observed['o']['V1_ml'] = 1e99
    assert c.pool_summary() == summary
    assert summary['candidate_count'] == 100 and summary['distinct_molecule_count'] == 1
    assert c.query('search_candidates', {'sort_by': 'predicted_q10_q90_span_V1_ml'})['matches']


def test_transcript_compaction_audit_completeness_and_explicit_retention(tmp_path):
    c, p, ledger = setup()
    # Old query result marker cannot originate in packet or persisted working state accidentally.
    original = c.query
    number = 0
    def query_with_marker(operation, args):
        nonlocal number
        result = original(operation, args)
        result['evidence_marker'] = f'unique_result_{number}'
        number += 1
        return result
    c.query = query_with_marker
    state = schema.empty_working_state()
    state['open_questions'] = ['Explicitly retained unique_result_0']
    answers = [query({'offset': 0}), query({'offset': 24}, state=state), query({'offset': 48})]
    call, calls = scripted(c, p, answers)
    transport.run_selector(p, c, ledger, tmp_path, config(), call=call)
    assert len(calls) == 4 and all(len(m) == 2 for m in calls)
    assert 'unique_result_0' in json.dumps(calls[1])
    assert 'unique_result_0' in json.dumps(calls[2])  # only explicitly retained state
    assert 'unique_result_1' in json.dumps(calls[2])
    assert 'unique_result_0' not in json.dumps(calls[3])
    assert 'unique_result_1' not in json.dumps(calls[3])
    assert 'unique_result_2' in json.dumps(calls[3])
    events = verify_audit(tmp_path/'audit.jsonl')
    assert [e['event_type'] for e in events].count('request') == 4
    assert [e['event_type'] for e in events].count('response') == 4
    assert [e['event_type'] for e in events].count('query') == 3
    assert all(f'unique_result_{i}' in (tmp_path/'audit.jsonl').read_text() for i in range(3))
    assert read(tmp_path/'turn_03.json')['context_budget']['reasoning_tokens'] == 10
    for event in events:
        assert all(k in event for k in ('seed', 'method', 'round', 'turn', 'request_sha256',
                                       'sequence', 'timestamp', 'artifact_sha256'))


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(next_query_intent='x'*12001),
    lambda s: s.update(active_hypothesis_ids=['H9999']),
    lambda s: s.update(shortlist=[{'id': 'illegal', 'why_still_interesting': 'why', 'scientific_role': 'coverage'}]),
    lambda s: s.update(key_evidence=[{'observed_id': '1', 'relevance': 'why', 'direction': 'supports'}]),
    lambda s: s.update(extra='transcript'),
])
def test_working_state_rejects_illegal_or_oversized(mutation):
    c, _, ledger = setup()
    s = schema.empty_working_state()
    mutation(s)
    with pytest.raises(ValueError):
        schema.validate_working_state(s, c, ledger['hypotheses'])


def test_total_working_state_bound_even_when_individual_fields_valid():
    c, _, _ = setup()
    s = schema.empty_working_state()
    for k in ('focus_questions', 'rejected_directions', 'open_questions'):
        s[k] = ['q'*400]*6
    s['shortlist'] = [{'id': i, 'why_still_interesting': 'x'*180, 'scientific_role': 'coverage'} for i in c.initial_cards() for i in [i['id']]]
    s['next_query_intent'] = 'x'*600
    assert len(schema.dumps(s)) > 12000
    with pytest.raises(ValueError, match='12000'):
        schema.validate_working_state(s, c, [])


def test_persistent_ledger_identity_and_history():
    ledger = memory.new_ledger(157, 'free')
    ledger = memory.merge_ledger(ledger, [update(value=belief(evidence=['o']))], 0, ['o'])
    assert memory.available_ids(ledger)[0] == 'H0002'
    ledger = memory.merge_ledger(ledger, [update(action='support', value=belief(status='supported', evidence=['o']))], 1, ['o'])
    ledger = memory.merge_ledger(ledger, [update(action='revise', value=belief('Conditional loading effect', 'weakened'))], 2, ['o'])
    h = ledger['hypotheses']['H0001']
    assert h['created_round'] == 0 and h['last_updated_round'] == 2 and len(h['history']) == 3
    assert h['history'][0]['belief']['statement'] == 'Loading affects retention'
    assert all('history' not in h for h in memory.ledger_context(ledger))
    for bad in [update('H0001'), update('H9999'), update(action='support', value=belief('Changed identity', 'supported')),
                update('H0002', value=belief(evidence=['candidate']))]:
        with pytest.raises(ValueError):
            memory.merge_ledger(ledger, [bad], 3, ['o'])
    ledger = memory.merge_ledger(ledger, [update(action='reject', value=belief('Conditional loading effect', 'rejected'))], 3, ['o'])
    assert memory.ledger_context(ledger) == []
    assert 'H0001' in ledger['hypotheses']


def test_balanced_replay_normalization_determinism_and_contradiction():
    data = raw()
    data['observed'] = []
    errors = [(0.01, 500), (8, 0), (-9, 0), (.01, .01), (.02, .02), (3, 30), (-3, -30),
              (1, 100), (2, 0), (-1, 0), (0, 1), (.1, .1), (0, 900), (0, 800), (0, 700)]
    for i, (a, b) in enumerate(errors):
        data['observed'].append({**row(f'o{i}', True), 'premeasurement_error_V1_ml': a,
            'premeasurement_error_V2_ml': b, 'premeasurement_prediction_V1_ml': 1+a,
            'premeasurement_prediction_V2_ml': 10+b})
    c = catalogs.ReadOnlyCatalog(**data)
    ledger = memory.new_ledger(157, 'free')
    h = belief(); h['contradicting_observed_ids'] = ['o5']
    ledger = memory.merge_ledger(ledger, [update(value=h)], 0, c.observed)
    records = memory.normalized_outcomes(c.observed, [1, 100])
    bank = memory.balanced_replay(c.observed, records, ledger, ['o10', 'o11'])
    assert bank == memory.balanced_replay(dict(reversed(list(c.observed.items()))), list(reversed(records)), ledger, ['o11', 'o10'])
    assert len(bank) == 12 and len({r['id'] for r in bank}) == 12
    assert {'contradiction', 'success', 'high_normalized_error'} <= {r['category'] for r in bank}
    assert 'o3' in {r['id'] for r in bank} and 'o2' in {r['id'] for r in bank}
    by_id = {r['id']: r for r in records}
    assert by_id['o1']['combined_normalized_error'] > by_id['o0']['combined_normalized_error']
    assert by_id['o0']['normalized_error_V2'] == 5
    digest = memory.measurement_digest(records, ['o10', 'o11'])
    assert digest['measurement_count'] == 2


@pytest.mark.parametrize('exception', [RuntimeError('Bearer sk-secret'),
    subprocess.CalledProcessError(1, ['codex'], stderr='api_key=secret'),
    transport.TransportFailure('cli_exit', exit_code=1, cli_version='codex-cli test')])
def test_failure_receipt_sanitized_stop_no_retry(tmp_path, exception):
    c, p, ledger = setup()
    calls = []
    def fail(*args):
        calls.append(1)
        raise exception
    with pytest.raises(RuntimeError, match='STOP'):
        transport.run_selector(p, c, ledger, tmp_path, config(), call=fail)
    receipt = read(tmp_path/'failure_00.json')
    assert not receipt['automatic_retry'] and not receipt['protocol_changed']
    assert 'secret' not in (tmp_path/'failure_00.json').read_text()
    assert 'secret' not in (tmp_path/'audit.jsonl').read_text()
    with pytest.raises(RuntimeError, match='already started'):
        transport.run_selector(p, c, ledger, tmp_path, config(), call=fail)
    assert len(calls) == 1 and not (tmp_path/'selection.json').exists()


def test_invalid_json_and_query_error_feedback_are_bounded(tmp_path):
    c, p, ledger = setup()
    calls = []
    def call(messages, cfg):
        calls.append(messages)
        n = len(calls)
        if n == 1:
            return '{bad', {}
        answer = query(operation='invalid') if n == 2 else query({'offset': (n-3)*24}) if n < 5 else plan(c, p)
        return json.dumps(answer), {}
    transport.run_selector(p, c, ledger, tmp_path, config(), call=call)
    assert 'response_validation' in json.dumps(calls[1])
    assert 'query_validation' in json.dumps(calls[2])
    assert len(c.queries) == 2


def test_context_overflow_fails_without_call_and_reports_component(tmp_path):
    c, p, ledger = setup()
    p['scientific_memory']['hypothesis_ledger'] = ['x'*(schema.CONTEXT_HARD_CHARS+1)]
    with pytest.raises(transport.ContextBudgetError) as error:
        transport.run_selector(p, c, ledger, tmp_path, config(), call=lambda *args: pytest.fail('no call'))
    assert error.value.audit['largest_component'] == 'hypothesis_ledger'
    assert read(tmp_path/'context_failure_00.json')['input_char_count'] > 100000


def fake_study(tmp_path, hybrid=False):
    method = ('cw16_synthetic' if hybrid else study.METHODS[0])
    data = raw(hybrid)
    checkpoint = tmp_path/'initial_checkpoint'
    checkpoint.write_text('synthetic checkpoint')
    once(tmp_path/f'initial/seed_157/{method}.json', data)
    cfg = config()
    once(tmp_path/'protocol.json', {'transport': cfg, 'fingerprint': protocol.fingerprint(cfg),
         'input_hashes': {str(checkpoint): file_hash(checkpoint)},
         'initial': {'157': {'checkpoint': str(checkpoint)}}})
    (tmp_path/'selector_prompt.txt').write_text(selector.SYSTEM_PROMPT)
    return method


@pytest.mark.parametrize('hybrid', [False])
@pytest.mark.parametrize('transient_failure', [False, True])
def test_stage_select_lock_state_and_idempotent_freeze(tmp_path, monkeypatch, hybrid, transient_failure):
    method = fake_study(tmp_path, hybrid)
    assert study.state(tmp_path, 157, method)['next_action'] == 'stage'
    result = study.stage(157, method, root=tmp_path)
    assert result['new_labels_revealed'] == result['llm_calls'] == result['fits'] == 0
    d = study.directory(tmp_path, 157, method, 0)
    p = read(d/'input.json')
    n = 0
    failed = []
    monkeypatch.setattr(transport.time, 'sleep', lambda _: None)
    def call(messages, cfg):
        nonlocal n
        if transient_failure and not failed:
            failed.append(True)
            raise api_status_error(503)
        n += 1
        if n <= 2:
            return json.dumps(query({'offset': (n-1)*24})), {}
        latest = json.loads(messages[-1]['content'])['latest_query_results']
        initial_ids = [r['id'] for r in p['initial_cards']]
        current_ids = [r['id'] for r in latest[0]['result']['matches']]
        selected = list(dict.fromkeys(initial_ids+current_ids))[:p['selection_count']]
        c = type('Viewed', (), {'viewed': selected})()
        return json.dumps(plan(c, p)), {}
    assert study.select(157, method, root=tmp_path, call=call)['batch_count'] == 32
    assert protocol.validate(tmp_path)['locked']
    assert study.state(tmp_path, 157, method)['next_action'] == 'advance'
    assert not (d/'measurement.json').exists()
    assert study.select(157, method, root=tmp_path, call=lambda *a: pytest.fail('no repeat'))['status'] == 'ALREADY_FROZEN'
    (d/'selection.json').write_text((d/'selection.json').read_text()+' ')
    with pytest.raises(RuntimeError, match='drift'):
        study.audit_batch(tmp_path, 157, method, 0)


@pytest.mark.parametrize('mutation', ['source', 'prompt', 'schema', 'memory', 'transport', 'retry_policy', 'deleted_lock', 'reregister'])
def test_protocol_lock_fail_closed(tmp_path, monkeypatch, mutation):
    fake_study(tmp_path)
    protocol.lock_protocol(tmp_path)
    if mutation == 'source':
        monkeypatch.setattr(protocol, 'code_hashes', lambda: {'changed.py': 'changed'})
    elif mutation == 'prompt':
        monkeypatch.setattr(protocol, 'SYSTEM_PROMPT', 'different')
    elif mutation == 'schema':
        monkeypatch.setitem(protocol.QUERY_SCHEMA, 'limit', 99)
    elif mutation == 'memory':
        monkeypatch.setitem(schema.MEMORY_POLICY, 'replay_size', 99)
    elif mutation == 'deleted_lock':
        d = study.directory(tmp_path, 157, study.METHODS[0], 0)
        once(d/'batch_freeze.json', {})
        (tmp_path/'protocol.lock.json').unlink()
    else:
        p = read(tmp_path/'protocol.json')
        if mutation == 'retry_policy':
            p['transport']['retry_policy']['max_retries'] = 0
        else:
            p['transport']['model'] = 'changed'
        if mutation == 'reregister':
            p['fingerprint'] = protocol.fingerprint(p['transport'])
        (tmp_path/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(RuntimeError, match='STOP STUDY'):
        protocol.validate(tmp_path)


def test_no_selector_validation_test_metrics_or_previous_ids(tmp_path):
    method = fake_study(tmp_path)
    study.stage(157, method, root=tmp_path)
    p = read(study.directory(tmp_path, 157, method, 0)/'input.json')
    serialized = json.dumps(p)
    for forbidden in ('validation_error', 'test_error', 'test_metrics', 'validation_labels', 'previous_selected_ids',
                      'previous_batch_rationale', 'recent_outcomes', 'previous_feedback_interpretation'):
        assert forbidden not in serialized
    assert 'prior_interpretation_before_latest_measurements' in serialized


def test_foreign_ledger_rejected():
    c, _, ledger = setup()
    with pytest.raises(ValueError, match='foreign'):
        selector.make_packet(c, ledger, seed=6101, method=ledger['method'], round_index=0, selection_count=32)


def select_synthetic(root, method, r=0, hypothesis_action='create'):
    study.stage(157, method, r, root=root)
    d = study.directory(root, 157, method, r)
    packet = read(d/'input.json')
    candidate_ids = [r['id'] for r in read(d/'catalog.json')['candidates']]
    count = 0
    def call(messages, cfg):
        nonlocal count
        count += 1
        if count < 3:
            return json.dumps(query({'ids': candidate_ids[(count-1)*24:count*24]}, 'get_candidates')), {}
        fake = type('Viewed', (), {'viewed': candidate_ids[:32]})()
        result = plan(fake, packet)
        result['hypothesis_updates'] = [update(action=hypothesis_action,
            value=belief(status='supported' if hypothesis_action == 'support' else 'proposed', evidence=['o']))]
        result['choices'][0]['hypothesis_id'] = 'H0001'
        return json.dumps(result), {}
    return study.select(157, method, r, root=root, call=call)


@pytest.mark.parametrize('fit_fault', [None, 'initialization_hash', 'validation_ids_hash', 'test_labels_used_for_fit_or_checkpoint_selection'])
def test_mock_measure_fit_next_round_stable_ledger_and_canonical_state(tmp_path, monkeypatch, fit_fault):
    import numpy as np
    import pandas as pd
    from src.qgeognn_al.active_learning_v3_2 import predictor
    from src.qgeognn_al.active_learning_v2.protocol import ids_hash
    method = fake_study(tmp_path)
    # Add immutable synthetic initial training provenance before protocol lock.
    initial_audit = tmp_path/'initial_fit.json'
    once(initial_audit, {'initialization_hash': 'same-init', 'validation_ids_hash': 'fixed-validation', 'test_labels_used_for_fit_or_checkpoint_selection': 0})
    frozen = read(tmp_path/'protocol.json')
    frozen['initial']['157']['checkpoint_audit'] = str(initial_audit)
    (tmp_path/'protocol.json').write_text(json.dumps(frozen))
    select_synthetic(tmp_path, method)
    calls = []
    class Store:
        def freeze_acquisitions(self, ids):
            self.frozen = ids
        def reveal(self, ids, purpose):
            if purpose == 'after_acquisition_fit':
                assert set(ids) <= set(self.frozen)
                assert (study.directory(tmp_path, 157, method, 0)/'batch_freeze.json').exists()
            calls.append(purpose)
            self.audit = [{'purpose': purpose, 'acquisitions_frozen': True, 'requested_ids_hash': ids_hash(ids)}]
            return np.array([[4., 40.]]*len(ids))
    class Context:
        def __init__(self, root, seed, method):
            self.seed, self.runtime = seed, root/f'runtime/seed_{seed}'
            self.data = pd.DataFrame({'sample_id': ['o', *map(str, range(1, 101)), 'v']})
            self.roles = {'l0': np.array([0]), 'u0': np.arange(1, 101), 'validation': np.array([101])}
            self.l0_truth = np.array([[1., 10.]])
        def ids(self, indices):
            return self.data.iloc[list(indices)].sample_id.tolist()
        def new_store(self):
            return Store()
        def fit(self, method, r, labeled, truth, target):
            calls.append('fit')
            assert len(labeled) == len(truth) == 33 and r == 1
            target.mkdir(parents=True)
            (target/'best.pt').write_bytes(b'synthetic model')
            (target/'predictions.csv.gz').write_bytes(b'synthetic predictions, no test truth')
            once(target/'fit_audit.json', {'train_ids_hash': ids_hash(self.ids(labeled)),
                 'train_rows': len(labeled), 'initialization_hash': 'same-init', 'validation_ids_hash': 'fixed-validation', 'test_labels_used_for_fit_or_checkpoint_selection': 0})
            if fit_fault:
                path = target/'fit_audit.json'
                value = read(path); value[fit_fault] = 'wrong'
                path.write_text(json.dumps(value))
    monkeypatch.setattr(predictor, 'Context', Context)
    # This advance is entirely synthetic: Store and fit cannot access real labels/model.
    if fit_fault:
        with pytest.raises(RuntimeError, match='drift'):
            study.advance(157, method, root=tmp_path)
        assert not (study.directory(tmp_path,157,method,0)/'training_complete.json').exists()
        return
    assert study.advance(157, method, root=tmp_path)['active_label_count'] == 33
    assert calls == ['after_acquisition_fit', 'initial_fit', 'fit']
    state = study.state(tmp_path, 157, method)
    assert state['round'] == 1 and state['next_action'] == 'stage'
    assert state['ledger']['hypotheses']['H0001']['created_round'] == 0
    assert not study.directory(tmp_path,157,method,1).exists()
    assert read(study.directory(tmp_path, 157, method, 0)/'training_complete.json')['test_truth_access_count'] == 0
    measurement = study.directory(tmp_path, 157, method, 0)/'measurement.json'
    bad = read(measurement); bad['records'][0]['error_V1_ml'] += 1
    measurement.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match='mismatch|drift'):
        study.state(tmp_path, 157, method)


def test_cli_exit_adapter_preserves_category_without_stderr(tmp_path, monkeypatch):
    home = tmp_path/'personal'
    (home/'.codex').mkdir(parents=True)
    (home/'.codex/config.toml').write_text('model_provider="p"\n[model_providers.p]\nbase_url="https://example.invalid"\nwire_api="responses"\n')
    executable = tmp_path/'fake_codex'
    executable.write_text('synthetic executable')
    cfg = {**config(), 'backend': 'codex_cli', 'cli': {'executable': str(executable),
             'sha256': file_hash(executable), 'version': 'codex-cli synthetic'}}
    monkeypatch.setattr(transport.Path, 'home', lambda: home)
    def fail(command, **kwargs):
        assert kwargs['env']['CODEX_HOME'] != str(home/'.codex')
        assert not (Path(kwargs['env']['CODEX_HOME'])/'sessions').exists()
        return subprocess.CompletedProcess(command, 1, '', 'Authorization: Bearer sk-secret')
    monkeypatch.setattr(transport.subprocess, 'run', fail)
    with pytest.raises(transport.TransportFailure) as error:
        transport.cli_call([], cfg)
    receipt = transport.failure_receipt(error.value, 'request-hash', cfg)
    assert receipt['exit_code'] == 1 and receipt['error_category'] == 'cli_exit'
    assert 'sk-secret' not in json.dumps(receipt)


def test_audit_tamper_and_query_fanout_bound(tmp_path):
    c, p, ledger = setup()
    bad = query(); bad['queries'] *= 3
    answers = [bad, query({'offset': 0}), query({'offset': 24})]
    call, calls = scripted(c, p, answers)
    transport.run_selector(p, c, ledger, tmp_path, config(), call=call)
    assert 'oversized' in json.dumps(calls[1])
    assert len(c.queries) == 2
    audit = tmp_path/'audit.jsonl'
    audit.write_text(audit.read_text().replace('A synthetic matched contrast.', 'tampered', 1))
    with pytest.raises(RuntimeError, match='audit hash'):
        verify_audit(audit)


def test_initial_catalog_missing_active_contract_uses_only_exact_archived_seal(tmp_path):
    source = tmp_path/'selections/seed_6101/free_llm32_scientist_v2/round_00'
    once(source/'catalog.json', raw())
    archive = tmp_path/'revisions/preselection/selections/seed_6101/free_llm32_scientist_v2/round_00/contract.json'
    once(archive, {'seed': 6101, 'method': 'free_llm32_scientist_v2', 'round': 0,
                  'catalog_sha256': file_hash(source/'catalog.json')})
    assert study.initial_contract(source, tmp_path, 6101, 'free_llm32_scientist_v2') == archive
    assert not (source/'contract.json').exists()
    (source/'catalog.json').write_text('{}')
    with pytest.raises(RuntimeError, match='no immutable provenance'):
        study.initial_contract(source, tmp_path, 6101, 'free_llm32_scientist_v2')


def api_status_error(status, code=None):
    import httpx
    from openai import APIStatusError
    return APIStatusError('Authorization: Bearer sk-secret',
        response=httpx.Response(status, request=httpx.Request('POST', 'https://example.invalid')),
        body={'code': code, 'message': 'sk-secret'})


@pytest.mark.parametrize('status', [408, 409, 429, 500, 502, 503, 504, 599])
def test_transient_status_retries_same_request_without_replaying_queries(tmp_path, monkeypatch, status):
    c, p, ledger = setup()
    successful, calls = scripted(c, p)
    attempts, delays = [], []
    def call(messages, cfg):
        attempts.append(copy.deepcopy(messages))
        if len(attempts) <= 2:
            raise api_status_error(status)
        return successful(messages, cfg)
    monkeypatch.setattr(transport.time, 'sleep', delays.append)
    transport.run_selector(p, c, ledger, tmp_path, config(), call=call)
    assert attempts[0] == attempts[1] == attempts[2]
    assert delays == [2, 4] and len(attempts) == 5 and len(calls) == 3
    assert len(c.queries) == 2
    events = verify_audit(tmp_path/'audit.jsonl')
    starts = [e for e in events if e['event_type'] == 'transport_attempt']
    assert len(starts) == 5 and len({e['request_sha256'] for e in starts[:3]}) == 1
    assert len([e for e in events if e['event_type'] == 'response']) == 3
    assert read(tmp_path/'turn_00.json')['provenance']['transport_attempts'] == 3
    assert not list(tmp_path.glob('failure_*.json'))
    assert 'sk-secret' not in (tmp_path/'audit.jsonl').read_text()


@pytest.mark.parametrize('kind', ['connection', 'timeout', 'server'])
def test_transient_retry_exhaustion_is_bounded_and_audited(tmp_path, monkeypatch, kind):
    import httpx
    from openai import APIConnectionError, APITimeoutError
    request = httpx.Request('POST', 'https://example.invalid')
    error = {'connection': APIConnectionError(message='sk-secret', request=request),
             'timeout': APITimeoutError(request=request), 'server': api_status_error(503)}[kind]
    c, p, ledger = setup()
    attempts, delays = [], []
    def fail(*args):
        attempts.append(1)
        raise error
    monkeypatch.setattr(transport.time, 'sleep', delays.append)
    with pytest.raises(RuntimeError, match='STOP'):
        transport.run_selector(p, c, ledger, tmp_path, config(), call=fail)
    assert len(attempts) == 4 and delays == [2, 4, 8]
    assert len(c.queries) == 0 and not (tmp_path/'selection.json').exists()
    receipt = read(tmp_path/'failure_00.json')
    assert receipt['attempts'] == 4 and receipt['retries_performed'] == 3
    assert receipt['automatic_retry'] and not receipt['will_retry']
    assert len(list(tmp_path.glob('transport_attempt_*.json'))) == 4
    events = verify_audit(tmp_path/'audit.jsonl')
    assert events[-1]['event_type'] == 'transport_failure'
    assert 'sk-secret' not in (tmp_path/'audit.jsonl').read_text()
    with pytest.raises(RuntimeError, match='already started'):
        transport.run_selector(p, c, ledger, tmp_path, config(), call=fail)
    assert len(attempts) == 4


@pytest.mark.parametrize('status,code', [(400, None), (401, None), (403, None), (404, None),
    (422, None), (429, 'insufficient_quota'), (429, 'billing_hard_limit_reached')])
def test_permanent_api_errors_do_not_retry(tmp_path, monkeypatch, status, code):
    c, p, ledger = setup()
    calls = []
    def fail(*args):
        calls.append(1)
        raise api_status_error(status, code)
    monkeypatch.setattr(transport.time, 'sleep', lambda _: pytest.fail('no sleep'))
    with pytest.raises(RuntimeError, match='STOP'):
        transport.run_selector(p, c, ledger, tmp_path, config(), call=fail)
    assert len(calls) == 1
    receipt = read(tmp_path/'failure_00.json')
    assert receipt['http_status'] == status and not receipt['retryable']
    assert 'sk-secret' not in (tmp_path/'audit.jsonl').read_text()


def test_protocol_guard_rechecked_before_retry(tmp_path, monkeypatch):
    c, p, ledger = setup()
    changed, calls = [], []
    def guard():
        if changed:
            raise RuntimeError('STOP STUDY: source drift')
    def fail(*args):
        calls.append(1)
        raise api_status_error(503)
    monkeypatch.setattr(transport.time, 'sleep', lambda _: changed.append(True))
    with pytest.raises(RuntimeError, match='source drift'):
        transport.run_selector(p, c, ledger, tmp_path, config(), call=fail, guard=guard)
    assert len(calls) == 1 and not (tmp_path/'selection.json').exists()


def test_retry_configuration_and_legacy_no_retry(tmp_path, monkeypatch):
    for value in (-1, 4, True, 1.5):
        with pytest.raises(ValueError, match='transport_retries'):
            transport.settings(transport_retries=value)
    with pytest.raises(ValueError, match='only for Responses'):
        transport.settings('codex_cli', transport_retries=1)
    assert transport.settings()['retry_policy']['max_transport_calls_per_selection'] == 120
    assert study.STUDY.name == 'qgeognn_v2_row_llm_scientist_v3_2'
    c, p, ledger = setup()
    cfg = config(); cfg.pop('retry_policy'); cfg['automatic_retry'] = False
    calls = []
    def fail(*args):
        calls.append(1)
        raise api_status_error(503)
    monkeypatch.setattr(transport.time, 'sleep', lambda _: pytest.fail('no sleep'))
    with pytest.raises(RuntimeError, match='STOP'):
        transport.run_selector(p, c, ledger, tmp_path, cfg, call=fail)
    assert len(calls) == 1


def test_responses_sdk_retries_disabled_and_model_parameters_preserved(monkeypatch):
    from types import SimpleNamespace
    import openai
    constructor, requests = [], []
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return iter([SimpleNamespace(type='response.completed')])
        def get_final_response(self):
            return SimpleNamespace(model='gpt-6-sol', status='completed', output=[], output_text='{}', id='synthetic', usage=None)
    def stream(**kwargs):
        requests.append(kwargs)
        return Stream()
    def client(**kwargs):
        constructor.append(kwargs)
        return SimpleNamespace(responses=SimpleNamespace(stream=stream))
    monkeypatch.setattr(openai, 'OpenAI', client)
    monkeypatch.setenv('SCIENTIST_API_KEY', 'synthetic-test-key')
    answer, provenance = transport.responses_call([], config())
    assert answer == '{}' and provenance['served_model'] == 'gpt-6-sol'
    assert constructor[0]['max_retries'] == 0 and constructor[0]['timeout'] == 600
    assert requests[0] == {'model': 'gpt-6-sol', 'input': [], 'tools': [], 'store': False,
                           'reasoning': {'effort': 'high'}, 'text': {'format': {'type':'json_schema', 'name':'scientist_response', 'strict':True, 'schema':config()['response_schema']}}}

@pytest.mark.parametrize('args', [
    {'filters':[{'field':'molecule_novelty','op':'eq','value':0}]},
    {'filters':[{'field':'condition_novelty','op':'eq','value':1}]},
    {'filters':[{'field':'max_similarity_to_observed','op':'gte','value':.9}]},
    {'filters':[{'field':'cw_distance_percentile','op':'lte','value':.5}]},
    {'similar_to':'o','filters':[{'field':'similarity','op':'gte','value':.9}]},
])
def test_flat_filter_regressions(args):
    c, p, _ = setup()
    result = c.query('search_candidates', args)
    assert result['matches']
    assert p['query_api_schema']['json_schema']
    import jsonschema
    jsonschema.validate({'operation':'search_candidates','args':args}, catalogs.QUERY_SCHEMA['json_schema'])


def test_get_rows_roles_pagination_and_truth_boundary():
    c = catalogs.ReadOnlyCatalog(**raw(True))
    ids = ['1','o','p0']
    first = c.query('get', {'ids':ids,'limit':2})
    second = c.query('get_rows', {'ids':ids,'limit':2,'offset':first['next_offset']})
    assert [r['catalog_role'] for r in first['matches']+second['matches']] == ['candidate','observed','pending']
    assert first['normalized_query_alias'] and not second['normalized_query_alias']
    assert 'V1_ml' not in first['matches'][0] and 'V1_ml' in first['matches'][1]
    assert 'V1_ml' not in second['matches'][0]
    for op in ('search_candidates','search_pending'):
        with pytest.raises(ValueError, match='observed-only'):
            c.query(op, {'filters':[{'field':'V1_ml','op':'gte','value':0}]})


@pytest.mark.parametrize('text', ['{} {}', '[{}]', 'prefix {"x":{}} suffix {}', 'prefix {"a":1, "a":2}', 'prefix {"x":NaN}', '{broken {"type":"selection"}', 'prefix {"x": 1} {broken'])
def test_json_recovery_rejects_ambiguous_or_malformed(text):
    from src.qgeognn_al.active_learning_v3_2.response_contract import decode_json
    with pytest.raises(ValueError): decode_json(text)


def test_json_recovery_audited_and_strict_selection(tmp_path):
    from src.qgeognn_al.active_learning_v3_2.response_contract import decode_json
    c,p,ledger = setup()
    count = 0
    def call(messages, cfg):
        nonlocal count
        count += 1
        v = query({'offset':(count-1)*24}) if count < 3 else plan(c,p)
        return 'I will inspect candidates. ' + json.dumps(v), {}
    result, _ = transport.run_selector(p,c,ledger,tmp_path,config(),call=call)
    assert len(result['choices']) == 32
    assert sum(e['event_type']=='json_recovery' for e in verify_audit(tmp_path/'audit.jsonl')) == 3
    bad = {**result, 'choices':result['choices'][:-1]}
    with pytest.raises(ValueError): selector.validate_selection(decode_json(json.dumps(bad))[0],c,p,ledger)


@pytest.mark.parametrize('malformed_at', [None, 23])
def test_24_accepted_queries_have_reserved_finalization(tmp_path, malformed_at):
    c,p,ledger = setup()
    calls=[]
    injected=False
    def call(messages,cfg):
        nonlocal injected
        payload=json.loads(messages[-1]['content'])
        ctx=payload['compact_round_context']
        calls.append(ctx['interaction_state']['mode'])
        if malformed_at is not None and len(c.queries)==malformed_at and not injected:
            injected=True
            return json.dumps(query({'descriptor':'molecule_novelty'})), {}
        if len(c.queries)<24:
            return json.dumps(query({'offset':24*(len(c.queries)%4)})), {}
        assert calls[-1]=='FINALIZE_ONLY'
        assert set(r['id'] for r in ctx['viewed_candidate_inventory']) == c.viewed
        return json.dumps(plan(c,p)), {}
    transport.run_selector(p,c,ledger,tmp_path,config(),call=call)
    assert len(c.queries)==24 and len(calls)==25+(malformed_at is not None)
    assert read(tmp_path/'interaction_summary.json')['finalization_mode_calls']==1


def test_25th_query_prompts_finalization_instead_of_dropping_trajectory(tmp_path):
    c,p,ledger=setup()
    count=0
    def call(messages,cfg):
        nonlocal count
        count+=1
        return json.dumps(query({'offset':24*(count%4)}) if count<=25 else plan(c,p)),{}
    transport.run_selector(p,c,ledger,tmp_path,config(),call=call)
    assert count==26 and len(c.queries)==24


def test_cannot_retry_forever(tmp_path):
    c,p,ledger=setup()
    calls=[]
    def call(*args):
        calls.append(1)
        return json.dumps(query(operation='bad')),{}
    with pytest.raises(RuntimeError,match='repair budget'):
        transport.run_selector(p,c,ledger,tmp_path,config(),call=call)
    assert len(calls)==5 and not c.queries


def test_hypothesis_review_requires_direct_evidence_or_two_rounds():
    c,p,ledger=setup()
    ledger=memory.merge_ledger(ledger,[update()],0,c.observed)
    assert not memory.hypotheses_requiring_review(ledger,1)
    assert memory.hypotheses_requiring_review(ledger,2)[0]['stale_two_rounds']
    direct=memory.hypotheses_requiring_review(ledger,1,['o'],[{'id':'o','hypothesis_id':'H0001'}])
    assert direct[0]['latest_direct_evidence']
    p=selector.make_packet(c,ledger,seed=157,method=ledger['method'],round_index=2,selection_count=32)
    c.query('search_candidates',{'offset':0});c.query('search_candidates',{'offset':24})
    value=plan(c,p)
    with pytest.raises(ValueError,match='review required'): selector.validate_selection(value,c,p,ledger)
    value['hypothesis_updates']=[update(action='retain')]
    _,merged=selector.validate_selection(value,c,p,ledger)
    assert merged['hypotheses']['H0001']['last_updated_round']==2
    assert len(merged['hypotheses']['H0001']['history'])==2


def test_diagnostic_is_informational_even_for_concentrated_batch():
    c,p,ledger=setup()
    c.query('search_candidates',{});c.query('search_candidates',{'offset':24})
    value=plan(c,p)
    result,_=selector.validate_selection(value,c,p,ledger)
    diagnostic=catalogs.batch_diagnostic(result['choices'],c)
    assert diagnostic['unique_molecules']==1 and diagnostic['largest_same_molecule_fraction']==1
    assert diagnostic['targeted_follow_up_count']==32 and diagnostic['broader_exploration_count']==0
    assert diagnostic['mean_pairwise_structural_similarity']==1


def test_v31_artifacts_and_scientific_sources_untouched():
    paths=['studies/active_learning/qgeognn_v2_row_llm_scientist_v3_1','src/qgeognn_al/active_learning_v3',
           'src/qgeognn_al/active_learning_v2','src/qgeognn_al/models','src/qgeognn_al/training']
    # The frozen V3.2 release must retain the original V3.1 scientific sources.
    # Main now also contains later V2 transport work and independent studies.
    assert subprocess.check_output(
        ['git','diff','8a8a917','a66a753','--',*paths],text=True)==''
    original = subprocess.check_output(
        ['git','ls-tree','-r','--name-only','8a8a917','--',*paths],text=True).splitlines()
    evolved = {'src/qgeognn_al/active_learning_v2/scientist_study.py',
               'src/qgeognn_al/active_learning_v2/scientist_transport.py'}
    protected = [path for path in original if path not in evolved]
    assert subprocess.check_output(['git','diff','8a8a917','--',*protected],text=True)==''



def test_six_rounds_only_registered_free_seed157():
    assert study.SEEDS==(157,) and study.METHODS==('free_llm32_scientist_v3_2',) and study.BUDGETS==(333,365,397,429,461,493,525)
    for seed,method,r in [(157,study.METHODS[0],6),(6101,study.METHODS[0],0),(157,'cw16_llm16_scientist_v3_2',0)]:
        with pytest.raises(ValueError): study.directory(Path('/unused'),seed,method,r)


def test_structured_wire_contract_nullable_optionals():
    from src.qgeognn_al.active_learning_v3_2.response_contract import decode_wire, RESPONSE_SCHEMA
    c,p,ledger=setup()
    args={k:None for k in catalogs.OPERATIONS['search_candidates']['properties']}
    args['filters']=[{'field':'molecule_novelty','op':'eq','value':0}]
    value={'response':query(args)}
    decoded=decode_wire(value)
    assert decoded['queries'][0]['args']=={'filters':args['filters']}
    value['response']['unexpected']='not allowed'
    with pytest.raises(Exception): decode_wire(value)

@pytest.mark.parametrize('event_type', ['response.failed','response.incomplete','error'])
def test_stream_failure_never_accepts_partial_json(monkeypatch,event_type):
    from types import SimpleNamespace
    import openai
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):return iter([SimpleNamespace(type=event_type)])
        def get_final_response(self):pytest.fail('must not accept partial response')
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs:Stream())))
    monkeypatch.setenv('SCIENTIST_API_KEY','synthetic-key')
    with pytest.raises(transport.TransportFailure,match='incomplete_or_failed'):
        transport.responses_call([],config())

@pytest.mark.parametrize('fail_round',[None,2])
def test_continuous_loop_exactly_six_rounds_or_stops(tmp_path,fail_round):
    import importlib.util
    from types import SimpleNamespace
    path=Path('scripts/studies/run_qgeognn_v3_2_loop.py')
    spec=importlib.util.spec_from_file_location('scientist_v32_loop_test',path)
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    once(tmp_path/'protocol.json',{'budgets':[333,365,397,429,461,493,525]})
    current={'round':0,'active_label_count':333,'next_action':'stage'}
    calls=[]
    def state(*args):return dict(current)
    def action(name):
        def perform(seed,method,r,root):
            assert seed==157 and method==study.METHODS[0] and 0<=r<6
            calls.append((name,r))
            if name=='select' and r==fail_round:raise RuntimeError('synthetic STOP')
            if name=='stage':current['next_action']='select'
            elif name=='select':current['next_action']='advance'
            else:
                current.update(round=r+1,active_label_count=365+32*r,next_action='complete_no_test_evaluation' if r==5 else 'stage')
            return {'status':'synthetic'}
        return perform
    api=SimpleNamespace(state=state,stage=action('stage'),select=action('select'),advance=action('advance'))
    result=controller.run_loop(tmp_path,157,study.METHODS[0],api=api,validate=lambda _:None)
    if fail_round is None:
        assert result['status']=='COMPLETED' and result['active_label_count']==525
        assert [r for name,r in calls if name=='advance']==list(range(6))
        assert len(calls)==18
    else:
        assert result['status']=='STOPPED' and current['active_label_count']==397
        assert not any(r>2 for _,r in calls)
    with pytest.raises(RuntimeError,match='already started'):
        controller.run_loop(tmp_path,157,study.METHODS[0],api=api,validate=lambda _:None)

@pytest.mark.parametrize('kind',['remote_protocol','read_error','read_timeout','missing_completion'])
def test_stream_disconnect_retries_are_bounded_and_keep_identical_request(tmp_path,monkeypatch,kind):
    import httpx
    errors={'remote_protocol':httpx.RemoteProtocolError('synthetic disconnect'),
            'read_error':httpx.ReadError('synthetic disconnect'),
            'read_timeout':httpx.ReadTimeout('synthetic timeout'),
            'missing_completion':transport.TransportFailure('stream_disconnected_before_completion')}
    c,p,ledger=setup(); calls=[];delays=[]
    monkeypatch.setattr(transport.time,'sleep',delays.append)
    def call(messages,cfg):
        calls.append(messages)
        raise errors[kind]
    with pytest.raises(RuntimeError,match='STOP: transport failure'):
        transport.run_selector(p,c,ledger,tmp_path,config(),call=call)
    assert len(calls)==4 and all(m==calls[0] for m in calls)
    assert delays==[2,4,8] and not c.queries
    receipt=read(tmp_path/'failure_00.json')
    assert receipt['attempts']==4 and receipt['retryable'] and receipt['exception_type']


def test_missing_stream_completion_is_transient_but_partial_content_not_used(monkeypatch):
    from types import SimpleNamespace
    import openai
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):return iter([])
        def get_final_response(self):raise RuntimeError("Didn't receive a `response.completed` event.")
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs:Stream())))
    monkeypatch.setenv('SCIENTIST_API_KEY','synthetic')
    with pytest.raises(transport.TransportFailure) as error:
        transport.responses_call([],config())
    assert transport.transport_error(error.value)['retryable']

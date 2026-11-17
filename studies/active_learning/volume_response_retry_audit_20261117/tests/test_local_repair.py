"""Conservative local-edit invariants on small label-free plans."""
import copy
import importlib.util
from pathlib import Path

path = Path(__file__).parents[1] / 'local_repair.py'
spec = importlib.util.spec_from_file_location('local_repair_unit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {'type': 'selection', 'packet_sha256': 'fixed', 'batch_strategy': 'Original strategy',
            'batch_rationale': 'Original scientific rationale', 'feedback_interpretation': '',
            'choices': [{'id': 'a', 'reason': 'Keep this science.', 'hypothesis_id': 'H1'},
                        {'id': 'bad', 'reason': 'Do not invent its identity.', 'hypothesis_id': 'H1'}],
            'hypotheses': [{'id': 'H1', 'content': 'Original hypothesis', 'basis': 'observed_data',
                            'confidence': 'medium', 'status': 'unresolved', 'supporting_observed_ids': ['obs'],
                            'contradicting_observed_ids': [], 'why_it_matters_for_model_learning': 'Original mechanism',
                            'next_discriminating_question': 'Original question'}]}


def test_preserve_science_and_leave_vacancy_without_prefix_fallback():
    plan = fixture()
    original = copy.deepcopy(plan)
    result = module.repair_plan(plan, {'a', 'prefix_first'}, {'obs'}, set(), {'packet_sha256': 'fixed', 'selection_count': 2, 'round': 0})
    assert plan == original
    assert result['draft']['choices'] == [original['choices'][0]]
    assert result['draft']['hypotheses'] == original['hypotheses']
    assert result['open_slots'] == [1]
    assert 'prefix_first' not in [choice['id'] for choice in result['draft']['choices']]


def test_observed_or_pending_never_selected_and_duplicates_keep_first():
    plan = fixture()
    plan['choices'] += [copy.deepcopy(plan['choices'][0]), {'id': 'pending', 'reason': 'Pending', 'hypothesis_id': None},
                        {'id': 'obs', 'reason': 'Observed', 'hypothesis_id': None}]
    result = module.repair_plan(plan, {'a', 'pending'}, {'obs'}, {'pending'}, {'packet_sha256': 'fixed', 'selection_count': 5, 'round': 0})
    assert [c['id'] for c in result['draft']['choices']] == ['a']
    assert len(result['open_slots']) == 4


def test_do_not_fix_unobserved_evidence_or_modify_hypothesis():
    plan = fixture()
    plan['hypotheses'][0]['supporting_observed_ids'] = ['a']
    result = module.repair_plan(plan, {'a'}, {'obs'}, set(), {'packet_sha256': 'fixed', 'selection_count': 2, 'round': 0})
    assert result['draft']['hypotheses'][0]['supporting_observed_ids'] == ['a']
    assert any(issue['code'] == 'unobserved_evidence_reference' for issue in result['issues'])


def test_only_exact_whitespace_identity_normalization():
    plan = fixture()
    plan['choices'][0]['id'] = ' a '
    result = module.repair_plan(plan, {'a', 'b'}, {'obs'}, set(), {'packet_sha256': 'fixed', 'selection_count': 2, 'round': 0})
    assert result['draft']['choices'][0]['id'] == 'a'
    assert result['draft']['choices'][0]['reason'] == plan['choices'][0]['reason']
    assert len(result['deterministic_fixes']) == 1
    assert result['open_slots'] == [1]


def test_hypothesis_binding_and_length_remain_blocked():
    plan = fixture()
    plan['packet_sha256'] = 'wrong'
    plan['choices'][0]['hypothesis_id'] = 'unknown'
    plan['batch_rationale'] = 'x' * 30001
    result = module.repair_plan(plan, {'a', 'bad'}, {'obs'}, set(), {'packet_sha256': 'fixed', 'selection_count': 2, 'round': 0})
    codes = {issue['code'] for issue in result['issues']}
    assert {'packet_binding_unresolved', 'invalid_hypothesis_reference', 'plan_over_30000_no_auto_truncation'} <= codes
    assert result['draft']['packet_sha256'] == 'wrong'
    assert len(result['draft']['batch_rationale']) == 30001


def test_verified_binding_field_only_preserves_scientific_content():
    plan = fixture()
    plan['packet_sha256'] = 'wrong'
    before = copy.deepcopy(plan)
    result = module.repair_plan(plan, {'a', 'bad'}, {'obs'}, set(),
                                {'packet_sha256': 'fixed', 'selection_count': 2, 'round': 0},
                                binding_proof={'verified': True, 'request_hash_exact': True,
                                               'request_user_packet_equal': True, 'packet_self_binding_exact': True})
    assert result['draft']['packet_sha256'] == 'fixed'
    assert result['draft']['choices'] == before['choices']
    assert result['draft']['hypotheses'] == before['hypotheses']
    assert plan == before
    assert result['deterministic_fixes'][0]['old_value'] == 'wrong'


def test_binding_proof_requires_exact_saved_request(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    packet = {'selection_count': 2, 'round': 0}
    packet['packet_sha256'] = module.protocol_hash(packet)
    import json
    request = {'messages': [{'role': 'system', 'content': 'Fixed'},
                            {'role': 'user', 'content': json.dumps(packet)}], 'config': {'model': 'fixed'}}
    (tmp_path / 'request_00.json').write_text(json.dumps(request))
    turn = {'request_sha256': module.protocol_hash(request)}
    proof = module.binding_context_proof(tmp_path, 'turn_00.json', turn, packet)
    assert proof['verified']
    turn['request_sha256'] = 'wrong'
    assert not module.binding_context_proof(tmp_path, 'turn_00.json', turn, packet)['verified']


def test_historical_draft_ledger_keeps_original_science_and_every_slot():
    import json
    import hashlib
    out = path.parent
    records = [json.loads(line) for line in (out / 'local_repair_drafts.jsonl').read_text().splitlines()]
    assert len(records) == 127
    for record in records:
        receipt = json.loads((module.ROOT / record['source']).read_text())
        original = json.loads(receipt['answer']) if isinstance(receipt['answer'], str) else receipt['answer']
        draft = record['draft']
        assert draft['hypotheses'] == original['hypotheses']
        for key in set(original) - {'choices', 'packet_sha256'}:
            assert draft[key] == original[key]
        assert len(draft['choices']) + len(record['unresolved_original_choices']) == 32
        assert len(record['open_slots']) == len(record['unresolved_original_choices'])
        for choice in draft['choices']:
            source = next(value for value in original['choices'] if value['id'] == choice['id'])
            assert choice == source
        for blocked in record['unresolved_original_choices']:
            assert blocked['original_choice'] == original['choices'][blocked['original_order']]
    manifest = json.loads((out / 'local_repair_inputs.json').read_text())
    for source in manifest['sources']:
        assert hashlib.sha256((module.ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']

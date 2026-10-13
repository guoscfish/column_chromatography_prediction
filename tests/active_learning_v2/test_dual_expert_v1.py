import json

import pytest

from src.qgeognn_al.active_learning_v2.dual_expert_selector import (
    CHEMISTRY_SYSTEM_PROMPT, ML_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT,
    validate_expert_memo, validate_planner_selection, trajectory_memory,
)
from src.qgeognn_al.active_learning_v2.scientist_selector import ReadOnlyCatalog
from src.qgeognn_al.active_learning_v2.dual_expert_transport import run_agent, settings


def _row(i, observed=False):
    value = {"id": str(i), "smiles": "CCO" if i % 2 else "CCN",
             "conditions": {"PE/EA": 10 + i, "Density g/ml": 1.0, "V/ul": 10 + i,
                            "loading solvent": "DCM", "Volume of loading solvent/ul": 100},
             "pred_V1_ml": 10.0 + i, "pred_V2_ml": 20.0 + i,
             "pred_V1_q10_ml": 9.0 + i, "pred_V1_q50_ml": 10.0 + i, "pred_V1_q90_ml": 11.0 + i,
             "pred_V2_q10_ml": 19.0 + i, "pred_V2_q50_ml": 20.0 + i, "pred_V2_q90_ml": 21.0 + i,
             "cw_distance_percentile": i / 100}
    if observed:
        value.update({"V1_ml": 10.0, "V2_ml": 20.0, "record_kind": "initial_L333"})
    return value


def _catalog():
    return ReadOnlyCatalog(candidates=[_row(i) for i in range(40)], observed=[_row(100, True)], pending=[], salt="test")


def _planner_packet():
    return {"packet_sha256": "packet", "selection_count": 32, "round": 0}


def test_role_prompts_are_distinct_and_free_of_fixed_allocation():
    assert CHEMISTRY_SYSTEM_PROMPT != ML_SYSTEM_PROMPT != PLANNER_SYSTEM_PROMPT
    for prompt in (CHEMISTRY_SYSTEM_PROMPT, ML_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT):
        assert "validation/test" in prompt
        assert "quota" in prompt.lower()
    assert "16" not in PLANNER_SYSTEM_PROMPT
    assert "final batch" in CHEMISTRY_SYSTEM_PROMPT


def test_expert_memo_cannot_be_final_selection():
    with pytest.raises(ValueError, match="cannot contain final selection"):
        validate_expert_memo({"type": "memo", "role": "chemistry_scientist", "choices": []}, "chemistry_scientist")


def test_planner_can_reject_expert_recommendation_and_choose_current_pool():
    catalog = _catalog()
    catalog.initial_cards()
    first = catalog.query("search_candidates", {"limit": 24})
    catalog.query("search_candidates", {"offset": first["next_offset"], "limit": 24})
    choices = [{"id": str(i), "reason": "planner found a more informative contrast", "knowledge_basis": [],
                "related_hypothesis_id": None} for i in range(8, 40)]
    response = {"type": "selection", "packet_sha256": "packet",
                "expert_synthesis": {"chemistry_takeaways": "chemistry suggested ids 0-7",
                    "ml_takeaways": "ML suggested ids 0-7", "agreements": "limited", "disagreements": "planner differs",
                    "planner_resolution": "select ids 8-39 after checking the pool"},
                "batch_strategy": "contrastive coverage", "choices": choices, "hypotheses": [],
                "batch_rationale": "The planner independently selected a legal contrast set.",
                "expected_learning_value": "generalization", "feedback_interpretation": ""}
    assert len(validate_planner_selection(response, catalog, _planner_packet())["choices"]) == 32


def test_expert_and_planner_transcripts_are_independent(tmp_path):
    catalog = _catalog()
    catalog.initial_cards()
    calls = []

    def expert_call(messages, config):
        calls.append(messages[0]["content"])
        return json.dumps({"type": "memo", "role": "chemistry_scientist", "state_summary": "observed only"}), {"mock": True}

    memo = run_agent("chemistry_scientist", {"packet_sha256": "x"}, catalog,
                     tmp_path / "chemistry", settings(backend="responses"), call=expert_call)
    assert memo["role"] == "chemistry_scientist"
    with pytest.raises(ValueError):
        run_agent("ml_scientist", {"packet_sha256": "x"}, catalog, tmp_path / "ml",
                  settings(backend="responses"), call=lambda *_: (json.dumps(memo), {"mock": True}))
    assert (tmp_path / "chemistry" / "memo.json").exists()
    assert not (tmp_path / "ml" / "memo.json").exists()
    assert calls[0] == CHEMISTRY_SYSTEM_PROMPT


def test_memory_rejects_foreign_seed_and_has_no_future_label_field():
    with pytest.raises(ValueError, match="foreign"):
        trajectory_memory([{"seed": 6101, "method": "other", "records": []}], 157, "free_llm32_dual_expert_v1")
    memory = trajectory_memory([], 157, "free_llm32_dual_expert_v1")
    assert "future_labels" not in json.dumps(memory)


def _full_packet(catalog, r=0):
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import encode_catalog
    from src.qgeognn_al.active_learning_v2.protocol import stable_hash
    value = {"seed": 157, "method": "free_llm32_dual_expert_v1", "round": r,
             "selection_count": 32, "delivery": "all_candidates_one_request",
             "candidate_count": len(catalog.candidates), "observed_count": len(catalog.observed),
             **encode_catalog(catalog)}
    value["packet_sha256"] = stable_hash(value)
    return value


def _direct_plan(packet):
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import decode_table
    ids = [row['id'] for row in decode_table(packet, 'candidates')][:32]
    return {"type": "selection", "packet_sha256": packet['packet_sha256'],
            "expert_synthesis": {key: 'independent advice assessed' for key in (
                'chemistry_takeaways', 'ml_takeaways', 'agreements', 'disagreements', 'planner_resolution')},
            "batch_strategy": 'condition contrasts', "batch_rationale": 'test residual structure',
            "expected_learning_value": 'improve accuracy', "feedback_interpretation": 'measured feedback assessed',
            "choices": [{"id": i, "reason": 'test condition response', "hypothesis_id": None,
                         "knowledge_basis": ['machine_learning']} for i in ids], "hypotheses": []}


def test_full_pool_roles_replay_and_repair_invalid_json(tmp_path):
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    catalog = _catalog()
    packet = study.role_packet(_full_packet(catalog), 'scientific_planner')
    attempts = []
    def call(messages, config):
        attempts.append(messages)
        return ('not JSON' if len(attempts) == 1 else json.dumps(_direct_plan(packet))), {"mock": True}
    result = run_agent('scientific_planner', packet, catalog, tmp_path, settings(), call=call)
    assert len(attempts) == 2 and len(result['choices']) == 32
    assert not catalog.queries
    assert catalog.viewed == set(catalog.candidates)
    assert run_agent('scientific_planner', packet, _catalog(), tmp_path, settings(),
                     call=lambda *_: pytest.fail('replay must make no calls')) == result
    packet['candidate_count'] -= 1
    with pytest.raises(ValueError, match='counts'):
        run_agent('scientific_planner', packet, _catalog(), tmp_path, settings(), call=call)


def test_two_cycles_freeze_reveal_fit_and_forward_all_roles(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    from src.qgeognn_al.active_learning_v2.dialog_bridge import once, read
    from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore

    monkeypatch.setattr(study, 'STUDY', tmp_path)
    monkeypatch.setattr(study, 'validate', lambda: {})
    once(tmp_path / 'protocol.json', {'transport': settings()})
    source = tmp_path / 'labels.csv'
    n = 418
    pd.DataFrame({'sample_id': [str(i) for i in range(n)],
                  'V1_ml': np.arange(n, dtype=float), 'V2_ml': np.arange(n, dtype=float)+2}).to_csv(source, index=False)
    partition = pd.DataFrame({'sample_id': [str(i) for i in range(n)],
                              'role': ['l0']*333+['u0']*80+['validation']*3+['test']*2})
    fits = []
    accesses = []
    class Context:
        def __init__(self, seed):
            self.seed = seed
            self.runtime = tmp_path / f'runtime/seed_{seed}'
            self.roles = {'l0': np.arange(333), 'u0': np.arange(333, 413),
                          'validation': np.arange(413, 416), 'test': np.arange(416, 418)}
            self.l0_truth = np.array([[i, i+2] for i in range(333)], dtype=float)
        def ids(self, indices):
            return [str(i) for i in indices]
        def new_store(self):
            store = RestrictedLabelStore(source, partition)
            original = store.reveal
            def reveal(ids, purpose):
                accesses.append((list(ids), purpose))
                return original(ids, purpose)
            store.reveal = reveal
            return store
        def fit(self, method, r, labeled, truth, runtime):
            assert np.array_equal(truth, np.array([[i, i+2] for i in labeled]))
            fits.append((r, len(labeled), self.validation_truth.copy()))
    monkeypatch.setattr(study, 'Context', Context)
    monkeypatch.setattr(study, 'prediction_freeze', lambda *_: None)
    def stage(seed, method, r):
        ctx = Context(seed)
        labeled, unlabeled, truth, history = study.state(ctx, method, r)
        raw = {'candidates': [_row(int(i)) for i in unlabeled], 'pending': [],
               'observed': [{**_row(int(i), True), 'V1_ml': float(v[0]), 'V2_ml': float(v[1])}
                            for i, v in zip(labeled, truth)], 'salt': f'test:{r}'}
        packet = _full_packet(ReadOnlyCatalog(**raw), r)
        packet['memory'] = trajectory_memory(history, seed, method)
        directory = study.selection_directory(seed, method, r)
        once(directory / 'input.json', packet)
        once(directory / 'catalog.json', raw)
        once(directory / 'contract.json', {'U_t_ids': ctx.ids(unlabeled)})
    monkeypatch.setattr(study, 'stage', stage)
    seen = []
    def call(messages, config):
        packet = json.loads(messages[1]['content'])
        seen.append(packet)
        if packet['role'] == 'scientific_planner':
            answer = _direct_plan(packet)
        else:
            assert 'chemistry_memo' not in packet and 'ml_memo' not in packet
            answer = {'type': 'memo', 'role': packet['role'], 'packet_sha256': packet['packet_sha256'],
                      'state_summary': 'independent analysis'}
        return json.dumps(answer), {'mock': True}
    for r in range(2):
        study.select(157, study.METHOD, r, call=call)
        assert not (study.selection_directory(157, study.METHOD, r) / 'feedback.json').exists()
        assert study.advance(157, study.METHOD, r)['active_label_count'] == 365+32*r
    assert [fit[:2] for fit in fits] == [(1, 365), (2, 397)]
    assert len(seen) == 6
    first = read(study.selection_directory(157, study.METHOD, 0) / 'selection.json')
    for packet in seen[3:]:
        assert packet['memory']['previous_plan'] == first
        assert len(packet['memory']['recent_outcomes']) == 32
        assert len(packet['memory']['previous_selected_ids']) == 32
        assert packet['memory']['chemistry']['role'] == 'chemistry_scientist'
        assert packet['memory']['machine_learning']['role'] == 'ml_scientist'
        assert packet['observed_count'] == 365 and packet['candidate_count'] == 48
        assert set(c['id'] for c in first['choices']).isdisjoint(
            row[0] for row in packet['candidates']['rows'])
    assert all(purpose != 'final_test_evaluation' for _, purpose in accesses)
    directory = study.selection_directory(157, study.METHOD, 0)
    study.select(157, study.METHOD, 0, call=lambda *_: pytest.fail('frozen selection must replay'))
    raw = read(directory / 'selection.json')
    raw['batch_rationale'] = 'tampered'
    (directory / 'selection.json').write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match='drift'):
        study.advance(157, study.METHOD, 0)


def test_planner_synthesis_accepts_text_points_without_changing_selection():
    catalog = _catalog()
    packet = _full_packet(catalog)
    response = _direct_plan(packet)
    response['expert_synthesis']['agreements'] = ['matched contrasts are informative', 'prediction is not truth']
    response['expert_synthesis']['disagreements'] = ['coverage versus focused response curves']
    result = validate_planner_selection(response, catalog, packet)
    assert result['expert_synthesis'] == response['expert_synthesis']
    assert [c['id'] for c in result['choices']] == [c['id'] for c in response['choices']]


@pytest.mark.parametrize('invalid', ['', ' ', [], [''], ['valid', None], {'point': 'text'}, 123])
def test_planner_synthesis_rejects_missing_or_nontext_content(invalid):
    catalog = _catalog()
    packet = _full_packet(catalog)
    response = _direct_plan(packet)
    response['expert_synthesis']['agreements'] = invalid
    with pytest.raises(ValueError, match='each must be nonempty text'):
        validate_planner_selection(response, catalog, packet)


def test_sol61_xhigh_settings_are_local_to_dual_expert():
    from src.qgeognn_al.active_learning_v2.scientist_transport import settings as original_settings
    config = settings()
    assert config['model'] == 'gpt-6.1-sol'
    assert config['reasoning_effort'] == 'xhigh'
    assert original_settings()['model'] == 'gpt-6-astra'
    assert original_settings()['reasoning_effort'] == 'high'
    with pytest.raises(ValueError, match='unsupported reasoning'):
        settings(reasoning_effort='invalid')


def test_inherited_round_uses_original_transport_new_round_uses_sol61(tmp_path, monkeypatch):
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    from src.qgeognn_al.active_learning_v2.dialog_bridge import once
    from src.qgeognn_al.active_learning_v2.scientist_transport import settings as original_settings
    monkeypatch.setattr(study, 'STUDY', tmp_path)
    old_dir='selections/seed_157/free_llm32_dual_expert_v1/round_01'
    once(tmp_path/'protocol.json', {'transport':settings(), 'inheritance':{
        'selection_directories':[old_dir],'transport':original_settings()}})
    assert study.transport_for(tmp_path/old_dir)['model']=='gpt-6-astra'
    assert study.transport_for(tmp_path/old_dir)['reasoning_effort']=='high'
    current=tmp_path/'selections/seed_157/free_llm32_dual_expert_v1/round_02'
    assert study.transport_for(current)['model']=='gpt-6.1-sol'
    assert study.transport_for(current)['reasoning_effort']=='xhigh'


def test_inherited_artifacts_copy_exact_bytes_and_reject_drift(tmp_path, monkeypatch):
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    from src.qgeognn_al.artifacts import sha256_file
    parent=tmp_path/'parent';parent.mkdir()
    source=parent/'freeze.json';source.write_text('{ "ids": ["1", "2"] }\n')
    target=tmp_path/'continuation';monkeypatch.setattr(study,'STUDY',target)
    inheritance={'copied_files':{'selections/freeze.json':str(source)},
                 'source_hashes':{str(source):sha256_file(source)}}
    study.inherit_files(inheritance)
    assert (target/'selections/freeze.json').read_bytes()==source.read_bytes()
    study.inherit_files(inheritance)
    (target/'selections/freeze.json').write_text('tampered')
    with pytest.raises(RuntimeError,match='destination drift'):
        study.inherit_files(inheritance)
    source.write_text('parent changed')
    with pytest.raises(RuntimeError,match='source drift'):
        study.inherit_files(inheritance)


def test_continuation_runs_only_three_new_cycles(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    from src.qgeognn_al.active_learning_v2.dialog_bridge import once
    monkeypatch.setattr(study,'STUDY',tmp_path)
    inheritance={'seed':157,'method':study.METHOD,'start_round':2}
    calls=[]
    monkeypatch.setattr(study,'make_inheritance',lambda *a:inheritance)
    monkeypatch.setattr(study,'inherit_files',lambda *a:None)
    def prepare(config,inheritance=None):
        once(tmp_path/'protocol.json',{'transport':config,'inheritance':inheritance})
    monkeypatch.setattr(study,'prepare',prepare)
    monkeypatch.setattr(study,'check_transport',lambda *a:None)
    monkeypatch.setattr(study,'Context',lambda seed:object())
    monkeypatch.setattr(study,'state',lambda context,method,r:calls.append(('state',r)))
    monkeypatch.setattr(study,'select',lambda seed,method,r:calls.append(('select',r)))
    monkeypatch.setattr(study,'advance',lambda seed,method,r:calls.append(('fit',r)))
    monkeypatch.setattr(study,'status_phase',lambda *a,**k:nullcontext())
    monkeypatch.setattr(study,'report',lambda seed,method,rounds:{'rounds':rounds})
    assert study.run(157,study.METHOD,5,settings(),start_round=2,parent_revision='parent')=={'rounds':5}
    assert calls==[('state',2),('select',2),('fit',2),('select',3),('fit',3),('select',4),('fit',4)]
    scope=__import__('json').loads((tmp_path/'run_scope_seed_157_to_493.json').read_text())
    assert scope['additional_rounds']==3 and scope['start_round']==2


def test_memo_correction_reports_complete_contract(tmp_path):
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    packet=study.role_packet(_full_packet(_catalog()),'ml_scientist')
    calls=[]
    def call(messages, config):
        calls.append(messages)
        response={'type':'memo','role':'ml_scientist','packet_sha256':packet['packet_sha256'],
                  'state_summary': {} if len(calls)==1 else 'valid analysis'}
        return json.dumps(response), {'mock':True}
    run_agent('ml_scientist',packet,_catalog(),tmp_path,settings(),call=call)
    correction=json.loads(calls[1][-1]['content'])['instruction']
    assert 'state_summary must be a string' in correction
    assert '9,500 characters' in correction
    assert 'no hypothesis_updates' in correction


def test_extra_format_attempt_is_bounded_and_can_replay(tmp_path):
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    packet=study.role_packet(_full_packet(_catalog()),'ml_scientist')
    calls=[]
    def call(messages,config):
        calls.append(messages)
        answer={'type':'memo','role':'ml_scientist','packet_sha256':packet['packet_sha256'],
                'state_summary': {} if len(calls)<4 else 'valid shortened memo'}
        return json.dumps(answer),{'mock':True}
    result=run_agent('ml_scientist',packet,_catalog(),tmp_path,settings(),call=call,extra_format_attempts=1)
    assert len(calls)==4 and 'MEMO FORMAT' in calls[-1][-1]['content']
    assert run_agent('ml_scientist',packet,_catalog(),tmp_path,settings(),
        call=lambda *_:pytest.fail('replay must not call provider'),extra_format_attempts=1)==result
    with pytest.raises(ValueError,match='at most one'):
        run_agent('ml_scientist',packet,_catalog(),tmp_path,settings(),extra_format_attempts=2)


def test_inherited_transport_map_preserves_mixed_model_prefix(tmp_path, monkeypatch):
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    from src.qgeognn_al.active_learning_v2.dialog_bridge import once
    from src.qgeognn_al.active_learning_v2.scientist_transport import settings as original
    monkeypatch.setattr(study,'STUDY',tmp_path)
    dirs=[f'selections/seed_157/{study.METHOD}/round_{r:02d}' for r in range(3)]
    once(tmp_path/'protocol.json',{'transport':settings(),'inheritance':{
        'selection_directories':dirs,'transport':settings(),
        'transports':{dirs[0]:original(),dirs[1]:original(),dirs[2]:settings()}}})
    assert [study.transport_for(tmp_path/d)['model'] for d in dirs]==['gpt-6-astra','gpt-6-astra','gpt-6.1-sol']


def test_unobserved_evidence_diagnostic_names_hypothesis_field_and_id():
    catalog=_catalog();packet=_full_packet(catalog);plan=_direct_plan(packet)
    plan['hypotheses']=[{'id':'H1','content':'prediction-only contrast','why_it_matters_for_model_learning':'coverage',
        'next_discriminating_question':'measure it','basis':'model_behavior','confidence':'low','status':'proposed',
        'supporting_observed_ids':['0'],'contradicting_observed_ids':[]}]
    with pytest.raises(ValueError,match='hypothesis H1 cites unobserved evidence in supporting_observed_ids') as caught:
        validate_planner_selection(plan,catalog,packet)
    assert "['0']" in str(caught.value)


def test_dual_budget_extension_does_not_change_original_scientist_budget():
    from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
    from src.qgeognn_al.active_learning_v2 import scientist_study as original
    assert study.BUDGETS==tuple(range(333,622,32))
    assert original.BUDGETS[-1]==525
    assert study.BUDGETS[4]==461 and study.BUDGETS[9]==621
    assert study.selection_directory(157,study.METHOD,8).name=='round_08'
    with pytest.raises(ValueError): study.selection_directory(157,study.METHOD,9)

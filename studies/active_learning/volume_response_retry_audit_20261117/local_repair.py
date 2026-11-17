#!/usr/bin/env python3
"""Conservative local draft repair on historical proposals; no API or label access."""
from __future__ import annotations
import ast
import copy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'studies/active_learning/decision_mechanism_followup_20261117'
SOURCE = ROOT / 'src/qgeognn_al/active_learning_v2'
INPUTS = {}


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protocol_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    assert raw == path.read_bytes(), ('unstable source', str(path))
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, ('source hash mismatch', str(path))
    INPUTS[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(raw)}
    return raw


def js(path, expected=None):
    return json.loads(read(path, expected))


def ids_from_table(packet, table):
    data = packet.get(table, {})
    columns = data.get('columns', [])
    if not data.get('rows'):
        return []
    assert 'id' in columns
    index = columns.index('id')
    return [row[index] for row in data['rows']]


def binding_context_proof(directory, proposal_source, turn, packet):
    """Authorize copying a binding field only from an exactly matched saved request."""
    request_name = Path(proposal_source).name.replace('turn_', 'request_')
    path = directory / request_name
    if not path.exists():
        return {'verified': False, 'reason': 'no_saved_request_for_exact_response'}
    request = js(path)
    try:
        exact_request = protocol_hash({'messages': request['messages'], 'config': request['config']}) == turn['request_sha256']
        user_packet = json.loads(request['messages'][1]['content'])
        same_packet = request['messages'][1]['role'] == 'user' and user_packet == packet
        without_binding = {key: value for key, value in packet.items() if key != 'packet_sha256'}
        self_consistent = protocol_hash(without_binding) == packet['packet_sha256']
        return {'verified': bool(exact_request and same_packet and self_consistent),
                'request_hash_exact': exact_request, 'request_user_packet_equal': same_packet,
                'packet_self_binding_exact': self_consistent,
                'request_source': str(path.relative_to(ROOT)),
                'request_source_sha256': INPUTS[str(path.relative_to(ROOT))]['sha256'],
                'request_sha256': turn['request_sha256']}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {'verified': False, 'reason': 'saved_request_structure_not_exactly_reconstructable'}


def exact_validators():
    """Use the C1-C3 audited route, including the frozen CW wrapper."""
    path = OUT / 'retry_validators.py'
    read(path)
    spec = importlib.util.spec_from_file_location('local_repair_actual_validators', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for method in ['historical_free_mixed_model', 'new_free_cw_sol61_xhigh', 'nocw']:
        function, sources = module.load_validator(method)
        for source in sources:
            read(source)
        result[method] = function
    return result


def structural_science_issues(plan, observed, packet, strict_nocw=False):
    """Enumerate every visible issue without deleting scientific content."""
    issues = []

    def add(code, location, category):
        issues.append({'code': code, 'location': location, 'category': category})

    if plan.get('type') != 'selection':
        add('wrong_or_missing_type', 'type', 'schema_or_intent_review')
    if plan.get('packet_sha256') != packet['packet_sha256']:
        add('packet_binding_unresolved', 'packet_sha256', 'identity_review')
    hypotheses = plan.get('hypotheses')
    if not isinstance(hypotheses, list) or len(hypotheses) > 8:
        add('hypothesis_container_or_count', 'hypotheses', 'scientific_review')
        hypotheses = []
    seen = set()
    for index, hypothesis in enumerate(hypotheses):
        prefix = f'hypotheses[{index}]'
        if not isinstance(hypothesis, dict):
            add('hypothesis_not_object', prefix, 'scientific_review')
            continue
        for field in ['id', 'content', 'why_it_matters_for_model_learning', 'next_discriminating_question']:
            if not isinstance(hypothesis.get(field), str) or not hypothesis[field].strip():
                add('missing_hypothesis_text', prefix + '.' + field, 'scientific_review')
        hid = hypothesis.get('id')
        if isinstance(hid, str):
            if hid in seen:
                add('duplicate_hypothesis_id', prefix + '.id', 'scientific_review')
            seen.add(hid)
        if hypothesis.get('basis') not in ['chemical_prior', 'observed_data', 'model_behavior', 'mixed']:
            add('invalid_basis', prefix + '.basis', 'scientific_review')
        if hypothesis.get('confidence') not in ['low', 'medium', 'high']:
            add('invalid_confidence', prefix + '.confidence', 'scientific_review')
        if hypothesis.get('status') not in ['proposed', 'supported', 'weakened', 'rejected', 'unresolved']:
            add('invalid_status', prefix + '.status', 'scientific_review')
        for field in ['supporting_observed_ids', 'contradicting_observed_ids']:
            evidence = hypothesis.get(field)
            if not isinstance(evidence, list) or any(not isinstance(sid, str) for sid in evidence):
                add('invalid_evidence_container', prefix + '.' + field, 'scientific_evidence_review')
            elif not set(evidence) <= observed:
                add('unobserved_evidence_reference', prefix + '.' + field, 'scientific_evidence_review')
        if strict_nocw:
            allowed = {'id', 'content', 'basis', 'supporting_observed_ids', 'contradicting_observed_ids', 'confidence', 'status', 'why_it_matters_for_model_learning', 'next_discriminating_question'}
            if set(hypothesis) - allowed:
                add('extra_hypothesis_fields_no_auto_deletion', prefix, 'schema_review')
    for index, choice in enumerate(plan.get('choices', [])):
        prefix = f'choices[{index}]'
        if not isinstance(choice, dict):
            add('choice_not_object', prefix, 'scientific_review')
            continue
        if not isinstance(choice.get('reason'), str) or not choice['reason'].strip():
            add('missing_choice_reason', prefix + '.reason', 'scientific_review')
        if 'hypothesis_id' not in choice or (choice['hypothesis_id'] is not None and choice['hypothesis_id'] not in seen):
            add('invalid_hypothesis_reference', prefix + '.hypothesis_id', 'scientific_review')
        if strict_nocw and set(choice) - {'id', 'reason', 'hypothesis_id'}:
            add('extra_choice_fields_no_auto_deletion', prefix, 'schema_review')
    for field in ['batch_strategy', 'batch_rationale', 'feedback_interpretation']:
        value = plan.get(field)
        if not isinstance(value, str) or ((field != 'feedback_interpretation' or packet['round'] > 0) and not value.strip()):
            add('missing_batch_text', field, 'scientific_review')
    if len(str(plan)) > 30000:
        add('plan_over_30000_no_auto_truncation', 'plan', 'scientific_compression_review')
    if strict_nocw and set(plan) - {'type', 'packet_sha256', 'batch_strategy', 'choices', 'hypotheses', 'feedback_interpretation', 'batch_rationale'}:
        add('extra_plan_fields_no_auto_deletion', 'plan', 'schema_review')
    return issues


def repair_plan(plan, legal, observed, pending, packet, strict_nocw=False, binding_proof=None):
    """Keep original legal choices; never fill vacancies or edit science fields."""
    original = copy.deepcopy(plan)
    choices = plan.get('choices')
    if not isinstance(choices, list):
        return {'unrecoverable_structure': True, 'draft': copy.deepcopy(plan), 'candidate_rows': [],
                'issues': [{'code': 'choices_not_list', 'category': 'unrecoverable_structure', 'location': 'choices'}]}
    eligible = set(legal) - set(pending)
    seen, kept, candidates, fixes = set(), [], [], []
    for index, choice in enumerate(choices):
        row = {'original_order': index, 'retained': False, 'deterministic_fix': '', 'original_choice_sha256': stable(choice)}
        if not isinstance(choice, dict) or not isinstance(choice.get('id'), str):
            row.update(original_id=choice.get('id') if isinstance(choice, dict) else None, action='unrecoverable_choice_identity')
            candidates.append(row)
            continue
        sid, modified = choice['id'], copy.deepcopy(choice)
        row['original_id'] = sid
        if sid not in eligible and sid.strip() in eligible:
            modified['id'] = sid.strip()
            sid = modified['id']
            row['deterministic_fix'] = 'strip_id_outer_whitespace_exact_unique_pool_match'
            fixes.append({'original_order': index, 'field': 'id', 'operation': row['deterministic_fix']})
        row['draft_id'] = sid
        if sid in pending:
            row['action'] = 'pending_id_excluded_need_replacement'
        elif sid in observed:
            row['action'] = 'already_observed_id_need_replacement'
        elif sid not in eligible:
            row['action'] = 'unknown_or_unsupplied_id_need_identity_judgment'
        elif sid in seen:
            row['action'] = 'duplicate_occurrence_excluded_first_legal_choice_kept'
        else:
            seen.add(sid)
            kept.append(modified)
            row.update(retained=True, action='preserve_original_legal_unique_choice', draft_order=len(kept) - 1,
                       reason_unchanged=modified.get('reason') == choice.get('reason'),
                       hypothesis_reference_unchanged=modified.get('hypothesis_id') == choice.get('hypothesis_id'),
                       retained_choice_sha256=stable(modified))
        candidates.append(row)
    draft = copy.deepcopy(plan)
    draft['choices'] = kept
    proof_complete = binding_proof and all(binding_proof.get(key) is True for key in
                                          ['verified', 'request_hash_exact', 'request_user_packet_equal', 'packet_self_binding_exact'])
    if draft.get('packet_sha256') != packet['packet_sha256'] and proof_complete:
        fixes.append({'field': 'packet_sha256', 'operation': 'restore_exact_hash_from_verified_saved_request',
                      'old_value': draft.get('packet_sha256'), 'new_value': packet['packet_sha256'],
                      'context_proof': copy.deepcopy(binding_proof)})
        draft['packet_sha256'] = packet['packet_sha256']
    issues = structural_science_issues(draft, set(observed), packet, strict_nocw)
    hypotheses = draft.get('hypotheses', [])
    hypothesis_index = {value.get('id'): index for index, value in enumerate(hypotheses) if isinstance(value, dict) and isinstance(value.get('id'), str)} if isinstance(hypotheses, list) else {}
    for row in candidates:
        if not row['retained']:
            row['science_gate_status'] = 'ID_invalid_or_duplicate_no_execution'
            continue
        choice = kept[row['draft_order']]
        index = hypothesis_index.get(choice.get('hypothesis_id'))
        locations = [f"choices[{row['draft_order']}]" ]
        if index is not None:
            locations.append(f'hypotheses[{index}]')
        linked = [issue['code'] for issue in issues if any(issue['location'].startswith(location) for location in locations)]
        row['linked_visible_issue_codes'] = '|'.join(linked)
        row['science_gate_status'] = 'linked_schema_or_evidence_issue_open' if linked else 'no_linked_structural_issue_detected_semantics_unchecked'
    assert original == plan
    assert stable(draft.get('hypotheses')) == stable(original.get('hypotheses'))
    assert all(row.get('reason_unchanged', True) and row.get('hypothesis_reference_unchanged', True) for row in candidates)
    return {'unrecoverable_structure': False, 'draft': draft, 'candidate_rows': candidates,
            'issues': issues, 'deterministic_fixes': fixes,
            'open_slots': list(range(len(kept), packet['selection_count'])),
            'hypotheses_unchanged': stable(draft.get('hypotheses')) == stable(original.get('hypotheses')),
            'original_plan_sha256': stable(original), 'draft_plan_sha256': stable(draft),
            'scientific_acceptance': 'not_established_by_deterministic_schema_and_membership_checks'}


def validation_error(plan, catalog, packet, method, validators):
    try:
        validators[method](plan, catalog, packet)
        return ''
    except (ValueError, TypeError, KeyError) as exc:
        return str(exc)


def main():
    validators = exact_validators()
    cases = pd.read_csv(io.BytesIO(read(OLD / 'llm_prefix_cases.csv')))
    assert len(cases) == 127
    rows, candidate_rows, issue_rows, drafts, binding_rows = [], [], [], [], []
    for case in cases.to_dict('records'):
        identity = {key: case[key] for key in ['method', 'seed', 'round', 'L_before', 'phase']}
        directory = (ROOT / case['source']).parent
        freeze = js(directory / 'batch_freeze.json', case['source_sha256'])
        packet = js(directory / 'input.json', freeze['artifact_hashes']['input.json'])
        contract = js(directory / 'contract.json', freeze['artifact_hashes']['contract.json'])
        turn = js(ROOT / case['first_complete_proposal_source'], case['first_complete_proposal_sha256'])
        plan = turn['answer']
        plan = json.loads(plan) if isinstance(plan, str) else plan
        assert isinstance(plan, dict) and len(plan['choices']) == 32
        assert [choice['id'] for choice in plan['choices']] == json.loads(case['first_proposal_choice_ID_order'])
        pool = ids_from_table(packet, 'candidates')
        observed = set(ids_from_table(packet, 'observed'))
        pending = set(ids_from_table(packet, 'pending'))
        assert set(pool) <= set(contract['U_t_ids']) and observed == set(contract['L_t_ids'])
        assert not (set(pool) & observed) and not (set(pool) & pending)
        catalog = SimpleNamespace(candidates=dict.fromkeys(pool), observed=dict.fromkeys(observed))
        raw_error = validation_error(plan, catalog, packet, case['method'], validators)
        actual = js(directory / 'selection.json', freeze['artifact_hashes']['selection.json'])
        final_error = validation_error(actual, catalog, packet, case['method'], validators)
        binding_proof = None
        if plan.get('packet_sha256') != packet['packet_sha256']:
            binding_proof = binding_context_proof(directory, case['first_complete_proposal_source'], turn, packet)
            binding_rows.append({**identity, 'original_wrong_value': plan.get('packet_sha256'),
                                 'known_request_packet_sha256': packet['packet_sha256'], **binding_proof})
        result = repair_plan(plan, set(pool), observed, pending, packet,
                             case['method'] != 'historical_free_mixed_model', binding_proof)
        draft_error = validation_error(result['draft'], catalog, packet, case['method'], validators)
        legal_before = {choice['id'] for choice in plan['choices'] if choice['id'] in set(pool) - pending}
        kept = {choice['id'] for choice in result['draft']['choices']}
        assert legal_before <= kept
        assert len(legal_before) == int(case['first_unique_legal_choices_n'])
        final_ids = {choice['id'] for choice in actual['choices']}
        categories = {issue['category'] for issue in result['issues']}
        science_open = any('scientific' in category for category in categories)
        slots = len(result['open_slots'])
        ready = not slots and not draft_error
        resolution = 'passes_historical_structural_gate' if ready else ('replacement_and_science_review' if slots and science_open else 'targeted_replacement_only' if slots else 'science_or_schema_review_only')
        rows.append({**identity, 'first_proposal_source': case['first_complete_proposal_source'],
                     'original_proposal_n': len(plan['choices']), 'original_unique_legal_n': len(legal_before),
                     'preserved_unique_legal_n': len(kept), 'budget_retention_fraction': len(kept) / 32,
                     'legal_choice_preservation_fraction': len(legal_before & kept) / len(legal_before) if legal_before else None,
                     'historical_final_retained_original_legal_n': len(legal_before & final_ids),
                     'vacant_slots': slots, 'deterministic_schema_fix_n': len(result['deterministic_fixes']),
                     'unresolved_issue_n': len(result['issues']), 'science_review_needed': science_open,
                     'case_resolution': resolution, 'raw_historical_validator_error': raw_error,
                     'local_draft_validator_error': draft_error, 'final_historical_validator_error': final_error,
                     'local_draft_passes_historical_gate': ready, 'hypotheses_unchanged': result['hypotheses_unchanged'],
                     'binding_restored_from_verified_request': bool(binding_proof and binding_proof.get('verified')),
                     'original_plan_sha256': result['original_plan_sha256'], 'draft_plan_sha256': result['draft_plan_sha256'],
                     'no_api': True, 'historical_selection_written': False})
        candidate_rows += [{**identity, **row} for row in result['candidate_rows']]
        issue_rows += [{**identity, **row} for row in result['issues']]
        drafts.append({**identity, 'source': case['first_complete_proposal_source'],
                       'original_plan_sha256': result['original_plan_sha256'], 'draft_plan_sha256': result['draft_plan_sha256'],
                       'draft': result['draft'], 'open_slots': result['open_slots'],
                       'unresolved_original_choices': [{'original_order': row['original_order'], 'action': row['action'],
                                                        'original_choice': copy.deepcopy(plan['choices'][row['original_order']])}
                                                       for row in result['candidate_rows'] if not row['retained']],
                       'deterministic_fixes': result['deterministic_fixes'],
                       'issues': result['issues'], 'scientific_acceptance': result['scientific_acceptance'],
                       'replacement_instruction': 'Fill only the listed vacancies or resolve flagged scientific evidence; preserve all retained choices. No catalog-prefix fallback.'})
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'local_repair_cases.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(OUT / 'local_repair_candidates.csv', index=False)
    pd.DataFrame(issue_rows, columns=list(identity) + ['code', 'location', 'category']).to_csv(OUT / 'local_repair_unresolved.csv', index=False)
    pd.DataFrame(binding_rows).to_csv(OUT / 'local_repair_binding_proofs.csv', index=False)
    (OUT / 'local_repair_drafts.jsonl').write_text('\n'.join(json.dumps(value, ensure_ascii=False) for value in drafts) + '\n')
    frame.groupby('method').agg(cases=('round', 'size'), original_legal=('original_unique_legal_n', 'sum'),
                              retained=('preserved_unique_legal_n', 'sum'), vacancies=('vacant_slots', 'sum'),
                              historical_retained=('historical_final_retained_original_legal_n', 'sum'),
                              science_review=('science_review_needed', 'sum'),
                              structural_ready=('local_draft_passes_historical_gate', 'sum')).reset_index().to_csv(OUT / 'local_repair_summary.csv', index=False)
    verification = {'cases': len(frame), 'original_choices': len(candidate_rows),
                    'original_unique_legal': int(frame.original_unique_legal_n.sum()),
                    'retained_original_unique_legal': int(frame.preserved_unique_legal_n.sum()),
                    'historical_final_retained_original_legal': int(frame.historical_final_retained_original_legal_n.sum()),
                    'vacancies': int(frame.vacant_slots.sum()), 'deterministic_schema_fixes': int(frame.deterministic_schema_fix_n.sum()),
                    'science_review_cases': int(frame.science_review_needed.sum()),
                    'ready_after_local_draft': int(frame.local_draft_passes_historical_gate.sum()),
                    'final_historical_validator_failures': int(frame.final_historical_validator_error.ne('').sum()),
                    'all_reason_and_hypothesis_text_preserved': True,
                    'historical_sources_modified': False, 'new_llm_calls': 0, 'new_training': 0,
                    'gate_caveat': 'Observed-ID membership and historical schema validation do not establish scientific evidence truth or Row improvement.'}
    (OUT / 'local_repair_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'local_repair_inputs.json').write_text(json.dumps({'sources': list(INPUTS.values())}, ensure_ascii=False, indent=2) + '\n')
    write_design(frame, verification)
    print(json.dumps(verification, ensure_ascii=False))


def write_design(frame, verification):
    near_complete = frame[frame.original_unique_legal_n == 31]
    table = frame.groupby('method').agg(cases=('round', 'size'), preserved=('preserved_unique_legal_n', 'sum'),
                                       vacant=('vacant_slots', 'sum'), historical_retained=('historical_final_retained_original_legal_n', 'sum'),
                                       science_review=('science_review_needed', 'sum'), structural_ready=('local_draft_passes_historical_gate', 'sum')).reset_index()
    header = '| ' + ' | '.join(table.columns) + ' |\n|' + '|'.join(['---'] * len(table.columns)) + '|\n'
    contents = '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in table.to_numpy())
    (OUT / 'LOCAL_REPAIR_DESIGN.md').write_text(f'''# 离线局部纠错：保留方案，分开处理ID与科学证据

本轮实际dry-run了127个最终严格选择展示表前32行的历史批次。每例都从已保存首份可解析的32项proposal读取原始科学选择，核对turn SHA、freeze、packet和当轮H/U/pending，未调用API、未更改selector或历史selection、未训练。

## 实测保留能力

{header}{contents}

共4064个原选择项，原合法唯一ID为{verification['original_unique_legal']}，原型保留{verification['retained_original_unique_legal']}，即所有原本合法唯一选择均保留，保留预算比例{verification['retained_original_unique_legal']/4064:.2%}。历史最终前缀方案仅保留其中{verification['historical_final_retained_original_legal']}个。原型留下{verification['vacancies']}个明确空额，不用表前32或其他候选自动补位。

以上都是每case先唯一化、再跨case累计的选择事件，不是全仓库不同候选ID的总数。特别是首方案已有31个合法ID的{len(near_complete)}例，原型保留{int(near_complete.preserved_unique_legal_n.sum())}个，历史最终只保留{int(near_complete.historical_final_retained_original_legal_n.sum())}个；只留下{int(near_complete.vacant_slots.sum())}个ID空额。不过其中{int(near_complete.science_review_needed.sum())}例仍需科学证据复核，不能把它们都称为“只补一个ID即可”。

必须区分：**ID保留不等于科学证据已被接受**。{verification['science_review_cases']}例仍有可确定定位的未观测证据引用问题；历史validator在本地草稿后完整通过{verification['ready_after_local_draft']}例。本次确定性字段规范化发生{verification['deterministic_schema_fixes']}次，全部是有严格原请求证明的根packet_sha恢复，证据内容保持不变；没有为了提高“修复率”擅自删除证据、截短方案或添加理由。两个原本32个ID都合法的案例在恢复binding后仍有未观测证据问题，故不能宣称完全自动恢复。

127例的首份可解析方案均有32项，未出现整个choice容器无法恢复的案例；这不等于所有首次API响应可恢复：本样本以“历史首份可解析32项proposal”为条件，不能外推到JSON损坏、截断或缺日志的人群。非法ID的原始科学身份依然不能凭空恢复。125例需要补ID，另2例ID完整但科学证据仍需复核。

## 原型算法与不变式

1. 当轮可保留集合为实际U与展示候选的交集，再排除pending和已测H。候选顺序完全不用于补位。
2. 按原proposal顺序保留合法ID的首次出现，重复项单独标注。每个保留choice的reason、hypothesis引用和全部其他字段逐值不变；全部hypotheses、批次解释逐值不变。
3. 支持两种确定性工程规范化：ID首尾空白剥除且精确匹配唯一当轮候选；或原保存request的messages/config哈希与该turn精确匹配、user packet逐值等于当轮input、packet内部绑定自校验均通过时，复制已知正确的根packet_sha。后一操作保留原wrong值和全部证明哈希；不能据此证明模型内部使用了正确上下文。缺任何一项证明就不重绑定。不会编辑近似ID、纠正拼写、猜分子、删除候选型证据、把basis改成prior，或生成新的实验假设。
4. 逐个列出缺额、无效ID、错误证据归属、hypothesis引用、缺少文本、超长方案和NoCW字段边界。需要新科学判断的项保持阻断。完整草稿是审查材料，不是可执行selection。
5. 使用C1–C3共用的真实函数纯AST接口，执行Historical direct、New frozen CW wrapper及NoCW五seed wrapper；没有引入未运行paired协议额外prose-ID禁令。最终已接受的127份历史selection作为validator兼容性对照，失败数{verification['final_historical_validator_failures']}。精确运行版本映射还应与C1–C3的transport审计一起读。

## 有效性层级

- **ID层**：合法、唯一、在当轮未标注/展示集合、非pending，可确定保留。
- **schema与证据成员资格层**：原字段、hypothesis引用、观测证据ID集合等可以自动检查；正确引用一个已观测ID仍不能证明其支持了该假设。
- **科学内容层**：理由是否与分子/条件对应、假设是否真正被证据支持、冲突证据如何解释，需要人工或未来授权LLM重新判断。当前不执行Judge，也不以通过schema冒充科学正确。

## 能减少什么，尚不能说明什么

dry-run说明无需丢弃大量原本合法的候选即可定位待修项；若未来只允许填空与定点证据修订，可以在协议上避免整批重选。然而这不是已观察的模型纠错成功率：本轮没有调用LLM，不能证明它会遵守局部编辑指令、费用必然下降、前缀率必然降低或Row性能改善。旧历史中“首方案→前缀最终”与本原型是确定性保真对照，不是学习收益反事实。

草稿choices为保留项按原相对顺序压紧的审查列表，`open_slots`表示还差多少个列表位置；`unresolved_original_choices`另存每个被排除项的原位置、完整reason/hypothesis引用与错误原因，防止把未知ID背后的科学意图一并丢弃。候选级`science_gate_status`仅标出是否发现关联的schema/证据成员资格问题，未发现问题仍不等于科学语义已证实。若新的科学复核确需撤销某个合法ID，应作为显式科学修订记录，不能伪装成确定性格式修复。

复现：`python -B studies/active_learning/volume_response_retry_audit_20261117/local_repair.py`。候选级操作见`local_repair_candidates.csv`；待补槽位及不变科学文本见`local_repair_drafts.jsonl`；问题定位见`local_repair_unresolved.csv`。所有输出只在本研究目录。
''')


if __name__ == '__main__':
    main()

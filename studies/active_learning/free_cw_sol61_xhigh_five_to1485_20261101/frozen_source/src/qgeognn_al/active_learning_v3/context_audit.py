"""Read-only V2 reference comparison; never executes a selector or opens a label store."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from ..active_learning_v2.protocol import stable_hash
from .artifacts import file_hash, once, read
from .catalog import ReadOnlyCatalog
from .memory import new_ledger, merge_ledger
from .schema import dumps, empty_working_state
from .selector import make_packet
from .transport import build_request, SYSTEM_PROMPT


def compare(v2_root, output):
    v2_root, output = Path(v2_root), Path(output)
    d = v2_root/'selections/seed_157/free_llm32_scientist_v2/round_03'
    packet, raw = read(d/'input.json'), read(d/'catalog.json')
    old_system = (v2_root/'selector_prompt.txt').read_text()
    old_config = read(v2_root/'protocol.json')['transport']
    messages = [{'role': 'system', 'content': old_system},
                {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]
    trace_sizes, queries, final = [], [], None
    for path in sorted(d.glob('turn_??.json')):
        receipt = read(path)
        binding = stable_hash({'messages': messages, 'config': old_config})
        if binding != receipt['request_sha256']:
            raise RuntimeError('read-only V2 replay request binding mismatch')
        trace_sizes.append({'turn': int(path.stem[-2:]), 'serialized_message_chars': len(json.dumps(messages, ensure_ascii=False)),
                            'reported_provider_usage': receipt['provenance'].get('usage')})
        answer = json.loads(receipt['answer'])
        if answer.get('type') == 'selection':
            final = receipt
            break
        q = read(d/f'query_{path.stem[-2:]}.json')
        queries.append(q)
        feedback = {'query_results': q['results'], 'remaining_answers': 28-int(path.stem[-2:])-1}
        if any(q.get('errors', [])):
            feedback['query_errors'] = q['errors']
        messages += [{'role': 'assistant', 'content': receipt['answer']},
                     {'role': 'user', 'content': json.dumps(feedback, ensure_ascii=False)}]
    if final is None:
        raise RuntimeError('reference round is not completed')
    # This is a counterfactual representation, NOT a continuation/migration of V2 beliefs.
    scales = read(v2_root/'runtime/seed_157/context.json')['preprocessing']['target_scales']
    target_scales = [scales[t] for t in ('V1', 'V2')]
    c = ReadOnlyCatalog(**raw, target_scales=target_scales)
    method = 'free_llm32_scientist_v3'
    ledger = new_ledger(157, method)
    mapping = {}
    for h in packet['memory']['previous_hypotheses']:
        ident = f'H{ledger["next_id"]:04d}'
        mapping[h['id']] = ident
        belief = {k: h[k] for k in ('status', 'confidence', 'basis', 'supporting_observed_ids',
                    'contradicting_observed_ids', 'why_it_matters_for_model_learning', 'next_discriminating_question')}
        belief['statement'] = h['content']
        ledger = merge_ledger(ledger, [{'id': ident, 'action': 'create',
            'reason': 'Comparison-only encoding of one prior V2 belief; no trajectory migration.', 'belief': belief}], 2, c.observed)
    c.ledger = copy.deepcopy(ledger)
    p = make_packet(c, ledger, seed=157, method=method, round_index=3, selection_count=32,
        latest_ids=[r['candidate_id'] for r in packet['memory']['recent_outcomes']],
        prior_interpretation=packet['memory']['previous_feedback_interpretation'])
    working = empty_working_state()
    working['focus_questions'] = ['Read-only architecture comparison; no new scientific decision is being made.']
    working['active_hypothesis_ids'] = list(ledger['hypotheses'])
    working['shortlist'] = [{'id': row['id'], 'why_still_interesting': 'Reference-only viewed candidate; scientific value not reassessed.',
                             'scientific_role': 'other'} for row in p['initial_cards']]
    evidence = sorted({i for h in ledger['hypotheses'].values() for i in h['supporting_observed_ids']+h['contradicting_observed_ids']})[:16]
    working['key_evidence'] = [{'observed_id': i, 'relevance': 'Prior hypothesis evidence retained by ID; full observed facts remain queryable.',
                              'direction': 'control'} for i in evidence]
    # All four results of V2's final query answer retained for a conservative size comparison.
    # V3 real relay permits only two queries/answer. Nothing in this comparison selects IDs.
    latest = [{'query': q, 'result': res} for q, res in zip(queries[-1]['queries'], queries[-1]['results'])]
    def rename(value):
        if isinstance(value, list):
            return [rename(v) for v in value]
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k == 'historical_abs_error_ml':
                    continue
                for t in ('V1', 'V2'):
                    if k == f'{t}_quantile_width_ml':
                        k = f'predicted_q10_q90_span_{t}_ml'
                result[k] = rename(v)
            return result
        return value
    latest = rename(latest)
    v3_messages, size = build_request(p, working, latest)
    old_len = lambda value: len(json.dumps(value, ensure_ascii=False))
    rows = []
    def add(name, before, after):
        rows.append({'component': name, 'v2_chars': before, 'v3_chars': after,
            'v2_tokens_estimate_chars_div_4': (before+3)//4, 'v3_tokens_estimate_chars_div_4': (after+3)//4,
            'reduction_percent': round(100*(1-after/before), 2) if before else None})
    add('system_prompt', len(old_system), len(SYSTEM_PROMPT))
    add('packet_or_compact_context', old_len(packet), len(dumps(p)))
    add('accumulated_query_conversation_or_latest_result', old_len(messages[2:]), len(dumps(latest)))
    add('working_state', 0, len(dumps(working)))
    add('previous_selected_ids', old_len(packet['memory']['previous_selected_ids']), 0)
    add('recent_outcome_reason_text_only', sum(len(r['reason']) for r in packet['memory']['recent_outcomes']), 0)
    add('previous_batch_rationale', old_len(packet['memory']['previous_batch_rationale']), 0)
    add('hypotheses_or_current_ledger', old_len(packet['memory']['previous_hypotheses']), len(dumps(p['scientific_memory']['hypothesis_ledger'])))
    add('high_error_examples_or_balanced_replay', old_len(packet['memory']['high_error_measured_examples']), len(dumps(p['scientific_memory']['balanced_replay_bank'])))
    add('whole_serialized_request', old_len(messages), len(dumps(v3_messages)))
    output.mkdir(parents=True, exist_ok=True)
    once(output/'context_size_comparison.json', {'reference': str(d), 'reference_round': 3,
        'label_access': 'existing observed catalog only; no new truth or label-store access',
        'v2_request_hashes_verified': True, 'v2_calls': trace_sizes, 'components': rows,
        'v3_context_budget': size, 'v3_request_at_full_working_state_bound_chars': size['input_char_count']+12000-len(dumps(working)),
        'projection_limits': ['V3 is a static counterfactual, not measured LLM behavior or equivalent scientific conclusions.',
            'Existing V2 prior beliefs are encoded for comparison only, not used to initialize a clean V3 study.',
            'Working state is an explicit synthetic example, not a second-LLM summary.',
            'Final V2 query group has four results; retained in full here despite V3 two-query turn cap.',
            'Characters/4 is a rough estimate, not a reliable tokenizer or CLI provider input count.'],
        'hypothesis_mapping_comparison_only': mapping,
        'reference_hashes': {p.name: file_hash(p) for p in [d/'input.json', d/'catalog.json', *sorted(d.glob('turn_??.json')), *sorted(d.glob('query_??.json'))]}})
    once(output/'v3_reference_context.json', {'packet': p, 'working_state': working, 'latest_query_results': latest})
    return rows

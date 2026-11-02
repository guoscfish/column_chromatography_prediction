#!/usr/bin/env python3
"""One-shot full-candidate trial using an existing label-safe round packet.

This intentionally does not modify the frozen query-based study or reveal labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2.dialog_bridge import once, read
from src.qgeognn_al.active_learning_v2.protocol import stable_hash
from src.qgeognn_al.active_learning_v2.scientist_selector import (
    CONDITIONS, MAX_HYPOTHESES, ReadOnlyCatalog, SYSTEM_PROMPT)
from src.qgeognn_al.active_learning_v2.scientist_transport import responses_call, settings
from src.qgeognn_al.active_learning_v2.scientist_study import exclusive_lock, status_phase

BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions'
DEFAULT_SOURCE = BASE / 'token4research_astra_high_stream_trial_20260930/selections/seed_157/free_llm32_scientist_v2/round_00'
DEFAULT_OUTPUT = BASE / 'token4research_astra_high_all_candidates_trial_20260930'

DIRECT_PROMPT = SYSTEM_PROMPT.split('READ-ONLY JSON RELAY')[0].replace(
    'Older measured records remain queryable.', 'All measured records are included below.') + '''FULL CATALOG DELIVERY
All eligible candidates and all currently observed records are supplied in this one
request. No retrieval, querying, paging, tools, or additional data are available.
Return a final selection directly. Consider the entire supplied candidate pool.
The packet uses a lossless column-oriented encoding: each table has columns and rows;
zip each row with columns. molecule_id references the molecules dictionary containing
canonical SMILES and descriptors. The conditions array follows condition_columns.
No candidates have been shortlisted, omitted or numerically rounded. Candidate row
order uses the same reproducible salted ID hash as the query-based experiment.
All supplied candidate IDs are eligible. Observed IDs are measured evidence only.
Return exactly one JSON object without Markdown fences or surrounding commentary.
Keep the entire scientific plan concise, below 30,000 characters.

FINAL SCIENTIFIC PLAN
''' + SYSTEM_PROMPT.split('FINAL SCIENTIFIC PLAN\n', 1)[1].replace(
    'distinct legal, viewed candidate IDs', 'distinct legal supplied candidate IDs')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_catalog(catalog):
    """Factor repeated molecular metadata without dropping any row or precision."""
    molecules = {}
    keys = {}
    for row in sorted(catalog.references.values(), key=lambda r: (r['smiles'], r['id'])):
        if row['smiles'] not in keys:
            key = f'm{len(keys)}'
            keys[row['smiles']] = key
            molecules[key] = {'smiles': row['smiles'], 'descriptors': row['descriptors']}
        elif molecules[keys[row['smiles']]]['descriptors'] != row['descriptors']:
            raise ValueError('inconsistent molecular metadata')
    tables = {}
    for name in ('candidates', 'observed', 'pending'):
        records = sorted(getattr(catalog, name).values(), key=catalog.key)
        fields = sorted(set().union(*(set(r) for r in records)) - {'id', 'smiles', 'descriptors', 'conditions'})
        tables[name] = {'columns': ['id', 'molecule_id', 'conditions', *fields],
                        'rows': [[r['id'], keys[r['smiles']], [r['conditions'][c] for c in CONDITIONS],
                                  *[r.get(f) for f in fields]] for r in records]}
    return {'condition_columns': list(CONDITIONS), 'molecules': molecules, **tables}


def decode_table(packet, name):
    rows = []
    for values in packet[name]['rows']:
        if len(values) != len(packet[name]['columns']):
            raise ValueError('table width mismatch')
        row = dict(zip(packet[name]['columns'], values))
        row.update(packet['molecules'][row.pop('molecule_id')])
        row['conditions'] = dict(zip(packet['condition_columns'], row['conditions']))
        rows.append(row)
    return rows


def validate_direct(response, catalog, packet):
    """Direct-delivery validation has no retrieval requirement or fabricated query."""
    if not isinstance(response, dict) or response.get('type') != 'selection' or response.get('packet_sha256') != packet['packet_sha256']:
        raise ValueError('selection packet binding mismatch')
    choices, hypotheses = response.get('choices'), response.get('hypotheses')
    if not isinstance(choices, list) or len(choices) != packet['selection_count'] or not all(isinstance(c, dict) for c in choices):
        raise ValueError('wrong selection count or choice structure')
    ids = [c.get('id') for c in choices]
    supplied = {r['id'] for r in decode_table(packet, 'candidates')}
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or not set(ids) <= supplied & set(catalog.candidates):
        raise ValueError('must select distinct legal supplied candidate IDs')
    if not isinstance(hypotheses, list) or len(hypotheses) > MAX_HYPOTHESES:
        raise ValueError('hypotheses must contain zero to eight entries')
    seen = set()
    for h in hypotheses:
        if not isinstance(h, dict):
            raise ValueError('hypothesis must be an object')
        for field in ('id', 'content', 'why_it_matters_for_model_learning', 'next_discriminating_question'):
            if not isinstance(h.get(field), str) or not h[field].strip():
                raise ValueError('incomplete hypothesis')
        if h['id'] in seen:
            raise ValueError('duplicate hypothesis ID')
        seen.add(h['id'])
        if h.get('basis') not in ('chemical_prior', 'observed_data', 'model_behavior', 'mixed'):
            raise ValueError('invalid hypothesis basis')
        if h.get('confidence') not in ('low', 'medium', 'high') or h.get('status') not in ('proposed', 'supported', 'weakened', 'rejected', 'unresolved'):
            raise ValueError('invalid hypothesis confidence/status')
        for field in ('supporting_observed_ids', 'contradicting_observed_ids'):
            evidence = h.get(field)
            if not isinstance(evidence, list) or any(not isinstance(i, str) for i in evidence) or not set(evidence) <= set(catalog.observed):
                raise ValueError('hypothesis cites unobserved evidence')
    for c in choices:
        if not isinstance(c.get('reason'), str) or not c['reason'].strip():
            raise ValueError('each choice needs a reason')
        if 'hypothesis_id' not in c or (c['hypothesis_id'] is not None and c['hypothesis_id'] not in seen):
            raise ValueError('invalid hypothesis reference')
    for field in ('batch_strategy', 'batch_rationale', 'feedback_interpretation'):
        if not isinstance(response.get(field), str) or ((field != 'feedback_interpretation' or packet['round'] > 0) and not response[field].strip()):
            raise ValueError(f'missing {field}')
    if len(str(response)) > 30000:
        raise ValueError('scientific plan exceeds memory size bound')
    return copy.deepcopy(response)


def prepare(source, output):
    contract, original, raw = [read(source / n) for n in ('contract.json', 'input.json', 'catalog.json')]
    if contract['input_sha256'] != digest(source / 'input.json') or contract['catalog_sha256'] != digest(source / 'catalog.json'):
        raise ValueError('source round artifacts changed')
    catalog = ReadOnlyCatalog(**raw)
    if set(catalog.candidates) | set(catalog.pending) != set(contract['U_t_ids']) or set(catalog.observed) != set(contract['L_t_ids']):
        raise ValueError('source pool or measured-record identity mismatch')
    packet = {k: original[k] for k in ('seed', 'method', 'round', 'active_label_count', 'selection_count', 'objective', 'memory')}
    packet.update(delivery='all_candidates_one_request', validation_test_records=0,
                  candidate_count=len(catalog.candidates), observed_count=len(catalog.observed),
                  **encode_catalog(catalog))
    for name in ('candidates', 'observed', 'pending'):
        decoded = {r['id']: r for r in decode_table(packet, name)}
        expected = getattr(catalog, name)
        if set(decoded) != set(expected) or any({k: v for k, v in decoded[i].items() if v is not None} !=
                                              {k: v for k, v in expected[i].items() if v is not None} for i in expected):
            raise ValueError('lossless catalog encoding failed')
    packet['packet_sha256'] = stable_hash(packet)
    config = settings(backend='responses')
    messages = [{'role': 'system', 'content': DIRECT_PROMPT},
                {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}]
    import tiktoken
    encoding = tiktoken.get_encoding('o200k_base')
    estimated_tokens = sum(len(encoding.encode(m['content'])) for m in messages)
    protocol = {'mode': 'all_candidates_one_request', 'transport': config, 'query_budget': 0,
                'model_call_budget': 1, 'source_directory': str(source),
                'source_hashes': {n: digest(source/n) for n in ('contract.json', 'input.json', 'catalog.json')},
                'script_sha256': digest(__file__), 'prompt_sha256': stable_hash(DIRECT_PROMPT),
                'candidate_count': len(catalog.candidates), 'observed_count': len(catalog.observed),
                'estimated_input_tokens_o200k_base': estimated_tokens,
                'token_estimate_is_not_provider_token_count': True,
                'request_sha256': stable_hash({'messages': messages, 'config': config}),
                'all_records_sent': True, 'numeric_rounding': False, 'new_labels_revealed': 0}
    output.mkdir(parents=True, exist_ok=True)
    once(output/'protocol.json', protocol)
    once(output/'input.json', packet)
    once(output/'request.json', {'messages': messages, 'config': config})
    return packet, catalog, config, messages, protocol


def run(source, output, live=False):
    with exclusive_lock(output/'execution.lock'):
        packet, catalog, config, messages, protocol = prepare(source, output)
        print(json.dumps({'status': 'PREPARED', 'candidates': len(catalog.candidates),
                          'observed': len(catalog.observed), 'estimated_input_tokens': protocol['estimated_input_tokens_o200k_base']}, ensure_ascii=False), flush=True)
        if not live:
            return
        receipt_path = output/'response.json'
        if not receipt_path.exists():
            if (output/'request_started.json').exists():
                raise RuntimeError('request already attempted; inspect artifacts before any explicit retry in a new output directory')
            once(output/'request_started.json', {'request_sha256': protocol['request_sha256'], 'started_at_unix': time.time()})
            started = time.monotonic()
            try:
                with status_phase('one-shot all-candidate selection', 30):
                    answer, provenance = responses_call(messages, config)
                once(receipt_path, {'request_sha256': protocol['request_sha256'], 'answer': answer,
                                    'provenance': provenance, 'elapsed_seconds': time.monotonic()-started})
            except Exception as error:
                once(output/'failure.json', {'error_type': type(error).__name__,
                    'http_status': getattr(error, 'status_code', None), 'elapsed_seconds': time.monotonic()-started})
                raise RuntimeError('full-pool request failed; see failure.json') from None
        receipt = read(receipt_path)
        if receipt['request_sha256'] != protocol['request_sha256']:
            raise ValueError('response request mismatch')
        try:
            result = validate_direct(json.loads(receipt['answer']), catalog, packet)
        except (ValueError, TypeError, KeyError) as error:
            once(output/'validation_failure.json', {'error_type': type(error).__name__, 'message': str(error)[:500]})
            raise
        once(output/'selection.json', result)
        batch = list(catalog.pending) + [c['id'] for c in result['choices']]
        if len(batch) != 32 or len(set(batch)) != 32:
            raise ValueError('invalid batch')
        once(output/'batch_freeze.json', {'batch_ids': batch, 'pending_ids': list(catalog.pending),
            'supplied_candidate_ids': sorted(catalog.candidates), 'new_batch_labels_revealed': False,
            'premeasurement_predictions': [{'candidate_id': i, 'pred_V1_ml': catalog.references[i]['pred_V1_ml'],
                                           'pred_V2_ml': catalog.references[i]['pred_V2_ml']} for i in batch],
            'artifact_hashes': {n: digest(output/n) for n in ('protocol.json','input.json','request.json','response.json','selection.json')}})
        # Comparison is read only after selection, never supplied to the model.
        baseline_path = source/'batch_freeze.json'
        overlap = len(set(batch) & set(read(baseline_path)['batch_ids'])) if baseline_path.exists() else None
        summary = {'status': 'BATCH_FROZEN_BEFORE_REVEAL', 'model': receipt['provenance']['served_model'],
            'reasoning_effort': 'high', 'eligible_candidates': len(catalog.candidates),
            'supplied_candidates': len(catalog.candidates), 'supplied_observed': len(catalog.observed),
            'selected_count': len(batch), 'model_calls': 1, 'queries': 0,
            'elapsed_seconds': receipt['elapsed_seconds'], 'usage': receipt['provenance']['usage'],
            'native_tool_calls': receipt['provenance']['native_tool_calls'], 'batch_audit': 'passed',
            'overlap_with_query_selection': overlap, 'new_labels_revealed': 0, 'new_training_runs': 0,
            'accuracy_comparison': 'not evaluated; both candidate delivery and observed-record exposure differ'}
        once(output/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--live', action='store_true', help='send one actual streaming model request')
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.live)

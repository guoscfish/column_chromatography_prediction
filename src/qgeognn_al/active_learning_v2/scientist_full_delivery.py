"""Lossless full-catalog delivery and direct scientific-plan validation."""
from __future__ import annotations
import copy
from .scientist_selector import CONDITIONS, MAX_HYPOTHESES, SYSTEM_PROMPT

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


DIRECT_PROMPT += """
EVIDENCE ID CHECK BEFORE SUBMITTING
The observed table contains the ONLY IDs permitted in supporting_observed_ids and
contradicting_observed_ids. Candidate-table IDs are NOT measured evidence, even if
their predictions look convincing. For a hypothesis based only on predictions or
chemical prior, use empty observed-evidence lists. Check each cited ID against the
observed table before returning your JSON. Do not describe a predicted error as a
measured error.
"""


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
                raise ValueError(f'hypothesis {h["id"]} cites unobserved evidence in {field}: {evidence}; only observed-table IDs are allowed')
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


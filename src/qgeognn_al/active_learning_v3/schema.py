"""Frozen bounds and strict scientist response contracts. No data access."""
from __future__ import annotations
import copy
import json

VERSION = 'llm_scientist_v3_1'
QUERY_BUDGET = 24
VIEW_BUDGET = 480
MODEL_CALL_BUDGET = 28
QUERIES_PER_TURN = 2
PER_QUERY_LIMIT = 24
WORKING_STATE_CHARS = 12000
CONTEXT_HARD_CHARS = 100000
MAX_ACTIVE_HYPOTHESES = 12
NEW_HYPOTHESES_PER_ROUND = 4
ROLES = ('failure_repair', 'hypothesis_test', 'condition_curve', 'matched_control',
         'structural_exploration', 'coverage', 'other')
WORKING_SCHEMA = {'focus_questions': 6, 'active_hypothesis_ids': 12, 'shortlist': 48,
                  'key_evidence': 16, 'rejected_directions': 6, 'open_questions': 6,
                  'next_query_intent': 600}
MEMORY_POLICY = {'replay_size': 12, 'error': 'signed prediction-minus-truth / fixed L333 scale',
                 'combined': 'RMS of two normalized errors', 'active_hypotheses': 12,
                 'history_in_context': False, 'previous_selected_ids_in_context': False}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def exact(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f'{name}: unexpected/missing fields')


def text(value, limit=600, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError('invalid or oversized text')


def items(value, limit, name):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f'{name}: oversized or invalid list')


def ids(value, allowed, limit, name):
    items(value, limit, name)
    if any(not isinstance(i, str) for i in value) or len(set(value)) != len(value) or not set(value) <= set(allowed):
        raise ValueError(f'{name}: illegal or duplicate ID')


def empty_working_state():
    return {k: '' if k == 'next_query_intent' else [] for k in WORKING_SCHEMA}


def validate_working_state(value, catalog, hypothesis_ids):
    exact(value, WORKING_SCHEMA, 'working_state')
    if len(dumps(value)) > WORKING_STATE_CHARS:
        raise ValueError('working_state exceeds 12000 characters')
    for field in ('focus_questions', 'rejected_directions', 'open_questions'):
        items(value[field], WORKING_SCHEMA[field], field)
        for item in value[field]:
            text(item, 400)
    ids(value['active_hypothesis_ids'], hypothesis_ids, 12, 'hypothesis reference')
    items(value['shortlist'], 48, 'shortlist')
    seen = []
    for row in value['shortlist']:
        exact(row, ('id', 'why_still_interesting', 'scientific_role'), 'shortlist')
        seen.append(row['id'])
        text(row['why_still_interesting'], 180)
        if row['scientific_role'] not in ROLES:
            raise ValueError('illegal scientific_role')
    ids(seen, set(catalog.candidates) & catalog.viewed, 48, 'shortlist')
    items(value['key_evidence'], 16, 'key_evidence')
    seen = []
    for row in value['key_evidence']:
        exact(row, ('observed_id', 'relevance', 'direction'), 'evidence')
        seen.append(row['observed_id'])
        text(row['relevance'], 240)
        if row['direction'] not in ('supports', 'contradicts', 'control'):
            raise ValueError('illegal evidence direction')
    ids(seen, catalog.observed, 16, 'evidence')
    text(value['next_query_intent'], 600, empty=True)
    return copy.deepcopy(value)

"""Deterministic belief ledger and balanced observed evidence; no chemical conclusions."""
from __future__ import annotations
import copy
import math
import numpy as np
from .schema import (dumps, exact, text, ids, items, MAX_ACTIVE_HYPOTHESES,
                     NEW_HYPOTHESES_PER_ROUND)

BELIEF_FIELDS = ('statement', 'status', 'confidence', 'basis', 'supporting_observed_ids',
                 'contradicting_observed_ids', 'why_it_matters_for_model_learning',
                 'next_discriminating_question')
ACTIONS = ('retain', 'revise', 'weaken', 'support', 'reject', 'create')


def new_ledger(seed, method):
    return {'seed': seed, 'method': method, 'next_id': 1, 'hypotheses': {}}


def available_ids(ledger):
    return [f'H{i:04d}' for i in range(ledger['next_id'], ledger['next_id'] + NEW_HYPOTHESES_PER_ROUND)]


def merge_ledger(ledger, updates, round_index, observed_ids):
    """Host owns identity. Revision changes belief, never creation identity/history."""
    result = copy.deepcopy(ledger)
    items(updates, MAX_ACTIVE_HYPOTHESES + NEW_HYPOTHESES_PER_ROUND, 'hypothesis_updates')
    seen, issued = set(), available_ids(ledger)
    for update in updates:
        exact(update, ('id', 'action', 'reason', 'belief'), 'hypothesis update')
        ident, action = update['id'], update['action']
        if not isinstance(ident, str) or ident in seen or action not in ACTIONS:
            raise ValueError('invalid hypothesis update identity/action')
        seen.add(ident)
        text(update['reason'], 400)
        old = result['hypotheses'].get(ident)
        if action == 'create':
            if old is not None or ident not in issued:
                raise ValueError('hypothesis ID not host-issued')
        elif old is None:
            raise ValueError('unknown persistent hypothesis ID')
        if action == 'retain':
            if update['belief'] is not None:
                raise ValueError('retain cannot replace belief')
            belief = {k: old[k] for k in BELIEF_FIELDS}
        else:
            belief = update['belief']
            exact(belief, BELIEF_FIELDS, 'belief')
            for field in ('statement', 'why_it_matters_for_model_learning', 'next_discriminating_question'):
                text(belief[field], 600)
            if belief['basis'] not in ('chemical_prior', 'observed_data', 'model_behavior', 'mixed'):
                raise ValueError('invalid basis')
            if belief['confidence'] not in ('low', 'medium', 'high') or belief['status'] not in (
                    'proposed', 'supported', 'weakened', 'rejected', 'unresolved'):
                raise ValueError('invalid confidence/status')
            for field in ('supporting_observed_ids', 'contradicting_observed_ids'):
                ids(belief[field], observed_ids, 12, field)
            required = {'weaken': 'weakened', 'support': 'supported', 'reject': 'rejected'}
            if action in required and belief['status'] != required[action]:
                raise ValueError('action/status mismatch')
            if old and action != 'revise' and belief['statement'] != old['statement']:
                raise ValueError('statement change requires explicit revise action')
        if old and round_index < old['last_updated_round']:
            raise ValueError('hypothesis round regressed')
        history = copy.deepcopy(old['history']) if old else []
        history.append({'round': round_index, 'action': action, 'reason': update['reason'],
                        'belief': copy.deepcopy(belief)})
        result['hypotheses'][ident] = {'id': ident, **copy.deepcopy(belief),
            'created_round': old['created_round'] if old else round_index,
            'last_updated_round': round_index, 'history': history}
        if action == 'create':
            result['next_id'] = max(result['next_id'], int(ident[1:]) + 1)
    if sum(h['status'] != 'rejected' for h in result['hypotheses'].values()) > MAX_ACTIVE_HYPOTHESES:
        raise ValueError('active hypothesis bound exceeded; explicitly retire beliefs')
    return result


def ledger_context(ledger):
    # Rejected beliefs stay queryable in full; their IDs remain in packet inventory.
    return [{k: copy.deepcopy(v) for k, v in h.items() if k != 'history'}
            for _, h in sorted(ledger['hypotheses'].items()) if h['status'] != 'rejected']


def normalized_outcomes(observed, scales):
    if len(scales) != 2 or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('fixed L333 scales must be finite and positive')
    records = []
    for ident, row in sorted(observed.items()):
        if 'premeasurement_error_V1_ml' not in row:
            continue
        errors = [float(row[f'premeasurement_error_{t}_ml']) for t in ('V1', 'V2')]
        if not all(math.isfinite(e) for e in errors):
            raise ValueError('nonfinite observed residual')
        norm = [e / s for e, s in zip(errors, scales)]
        records.append({'id': ident, 'raw_error_V1_ml': errors[0], 'raw_error_V2_ml': errors[1],
            'normalized_error_V1': norm[0], 'normalized_error_V2': norm[1],
            'combined_normalized_error': math.sqrt(sum(e*e for e in norm)/2)})
    return records


def quantiles(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {'count': len(values), **(dict(zip(('min', 'q10', 'q25', 'median', 'q75', 'q90', 'max'),
            np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist())) if values else {})}


def measurement_digest(records, latest_ids, measured_count=None):
    latest = [r for r in records if r['id'] in set(latest_ids)]
    return {'measurement_count': len(latest_ids) if measured_count is None else measured_count,
        'residual_count': len(latest), 'error_timing': 'acquisition-time prediction minus measurement',
        'normalized_errors': {t: quantiles(r[f'normalized_error_{t}'] for r in latest) for t in ('V1', 'V2')},
        'combined_normalized_error': quantiles(r['combined_normalized_error'] for r in latest),
        'direction_counts': {t: {'over': sum(r[f'normalized_error_{t}'] > 0 for r in latest),
            'under': sum(r[f'normalized_error_{t}'] < 0 for r in latest),
            'equal': sum(r[f'normalized_error_{t}'] == 0 for r in latest)} for t in ('V1', 'V2')},
        'key_ids': [r['id'] for r in sorted(latest, key=lambda r: (-r['combined_normalized_error'], r['id']))[:3]]}


def balanced_replay(observed, records, ledger, latest_ids):
    """Two slots/category with deterministic backfill. Roles are memory, not batch quotas."""
    by_id = {r['id']: r for r in records}
    contradictions = {i for h in ledger['hypotheses'].values() if h['status'] != 'rejected'
                      for i in h['contradicting_observed_ids']}
    failure = sorted(records, key=lambda r: (-r['combined_normalized_error'], r['id']))
    success = sorted(records, key=lambda r: (r['combined_normalized_error'], r['id']))
    over = sorted((r for r in records if max(r['normalized_error_V1'], r['normalized_error_V2']) > 0),
                  key=lambda r: (-max(r['normalized_error_V1'], r['normalized_error_V2']), r['id']))
    under = sorted((r for r in records if min(r['normalized_error_V1'], r['normalized_error_V2']) < 0),
                   key=lambda r: (min(r['normalized_error_V1'], r['normalized_error_V2']), r['id']))
    recent = sorted((r for r in records if r['id'] in set(latest_ids)), key=lambda r: (r['combined_normalized_error'], r['id']))
    # Spread across the recent error distribution, rather than only selecting its hardest cases.
    if recent:
        recent = [recent[i] for i in dict.fromkeys([len(recent)//2, 0, len(recent)-1, *range(len(recent))])]
    groups = [('contradiction', [{'id': i, **by_id.get(i, {})} for i in sorted(contradictions) if i in observed]),
              ('success', success), ('high_normalized_error', failure), ('overprediction', over),
              ('underprediction', under), ('recent_representative', recent)]
    selected = {}
    for category, rows in groups:
        count = 0
        for row in rows:
            if row['id'] not in selected and count < 2:
                selected[row['id']] = {'id': row['id'], 'category': category, 'residuals': by_id.get(row['id']),
                                       'observation': copy.deepcopy(observed[row['id']])}
                count += 1
    for row in recent + success + failure:
        if len(selected) >= 12:
            break
        selected.setdefault(row['id'], {'id': row['id'], 'category': 'backfill', 'residuals': row,
                                         'observation': copy.deepcopy(observed[row['id']])})
    return list(selected.values())


def scientific_memory(catalog, ledger, scales, latest_ids=(), prior_interpretation='', open_questions=()):
    records = normalized_outcomes(catalog.observed, scales)
    text(prior_interpretation, 1200, empty=True)
    items(list(open_questions), 6, 'open scientific questions')
    for question in open_questions:
        text(question, 400)
    return {'hypothesis_ledger': ledger_context(ledger),
            'latest_measurements_digest': measurement_digest(records, latest_ids),
            'balanced_replay_bank': balanced_replay(catalog.observed, records, ledger, latest_ids),
            'open_scientific_questions': list(open_questions),
            'prior_interpretation_before_latest_measurements': prior_interpretation}

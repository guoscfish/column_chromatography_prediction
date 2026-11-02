"""V3 deterministic views over authorized catalog rows. No label-store access."""
from __future__ import annotations
from collections import Counter
import copy
import math
from rdkit import DataStructs
from ..active_learning_v2.scientist_selector import ReadOnlyCatalog as V2Catalog, SIGNALS, DESCRIPTORS
from .memory import normalized_outcomes, quantiles
from .schema import QUERY_BUDGET, VIEW_BUDGET, PER_QUERY_LIMIT, dumps

SPAN_FIELDS = tuple(f'predicted_q10_q90_span_{t}_ml' for t in ('V1', 'V2'))
ERROR_FIELDS = ('raw_error_V1_ml', 'raw_error_V2_ml', 'normalized_error_V1',
                'normalized_error_V2', 'combined_normalized_error')
CONVENIENCE = ('get_condition_series', 'get_nearest_analogs',
               'get_matched_loading_contrasts', 'get_matched_solvent_contrasts')
QUERY_SCHEMA = {'ordinary': ['get_candidates', 'get_observed', 'get_pending',
                            'search_candidates', 'search_observed', 'search_pending'],
                'convenience': list(CONVENIENCE), 'ledger': ['get_hypotheses'],
                'limit': PER_QUERY_LIMIT, 'convenience_args': ['reference_id', 'offset', 'limit',
                    'k', 'min_tanimoto', 'solvent_tolerance', 'loading_relative_tolerance'],
                'solvent_tolerance': 'absolute PE fraction difference',
                'loading_relative_tolerance': 'absolute difference / max(absolute loadings)',
                'failed_queries_consume_budget': True}


def pe_fraction(value):
    if isinstance(value, str) and '/' in value:
        a, b = map(float, value.split('/'))
        if a < 0 or b < 0 or a+b <= 0:
            raise ValueError('invalid PE/EA ratio')
        return a/(a+b)
    value = float(value)  # canonical features store PE/EA as a numeric ratio
    if not math.isfinite(value) or value < 0:
        raise ValueError('invalid PE/EA ratio')
    return value/(1+value)


class ReadOnlyCatalog(V2Catalog):
    def __init__(self, candidates, observed, pending, *, salt, target_scales,
                 ledger=None, query_budget=QUERY_BUDGET, view_budget=VIEW_BUDGET):
        # Reject nested payloads in X and nonfinite numbers before inherited descriptors.
        for row in candidates + observed + pending:
            if not isinstance(row.get('id'), str) or not isinstance(row.get('smiles'), str):
                raise ValueError('catalog ID/SMILES must be strings')
            for value in row.get('conditions', {}).values():
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise ValueError('invalid condition scalar')
                if isinstance(value, str) and len(value) > 100:
                    raise ValueError('oversized condition')
            pe_fraction(row['conditions']['PE/EA'])
            for key, value in row.items():
                if key.endswith('_ml') or key == 'cw_distance_percentile':
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError('nonfinite/non-numeric catalog value')
        super().__init__(candidates, observed, pending, salt=salt,
                         query_budget=query_budget, view_budget=view_budget)
        self.target_scales = list(target_scales)
        self.ledger = copy.deepcopy(ledger or {'hypotheses': {}})
        for row in self.references.values():
            for t in ('V1', 'V2'):
                old = f'{t}_quantile_width_ml'
                if old in row:
                    row[f'predicted_q10_q90_span_{t}_ml'] = row.pop(old)
            row.pop('historical_abs_error_ml', None)
        for record in normalized_outcomes(self.observed, target_scales):
            self.observed[record['id']].update({k: v for k, v in record.items() if k != 'id'})

    def _check_signal(self, field, rows, args):
        if field in (*ERROR_FIELDS, 'V1_ml', 'V2_ml'):
            if rows is not self.observed:
                raise ValueError('measured endpoints/residuals are observed-only')
            return
        if field in ('historical_abs_error_ml', 'V1_quantile_width_ml', 'V2_quantile_width_ml'):
            raise ValueError('use V3 normalized error / predicted span fields')
        if field in SPAN_FIELDS:
            return
        return super()._check_signal(field, rows, args)

    def query(self, operation, args):
        if len(self.queries) >= self.query_budget:
            raise ValueError('query budget exhausted')
        before = len(self.queries)
        try:
            if not isinstance(operation, str):
                raise ValueError('operation must be a string')
            if operation in CONVENIENCE:
                response = self._convenience(operation, args)
            elif operation == 'get_hypotheses':
                if not isinstance(args, dict) or set(args) != {'ids'} or not isinstance(args['ids'], list) or len(args['ids']) > 2:
                    raise ValueError('get_hypotheses requires at most two IDs')
                if any(not isinstance(i, str) or i not in self.ledger['hypotheses'] for i in args['ids']):
                    raise ValueError('unknown hypothesis ID')
                # Current full belief only. Historical revisions remain audit-only.
                response = {'matches': [{k: copy.deepcopy(v) for k, v in self.ledger['hypotheses'][i].items()
                                         if k != 'history'} for i in dict.fromkeys(args['ids'])]}
            else:
                return super().query(operation, args)
            response.update(remaining_queries=self.query_budget-before-1,
                            remaining_new_candidate_views=self.view_budget-len(self.viewed))
            self.queries.append(copy.deepcopy({'operation': operation, 'args': args, 'response': response}))
            return response
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            if len(self.queries) == before:
                self.queries.append({'operation': operation, 'args': copy.deepcopy(args),
                                     'error': type(error).__name__})
            raise ValueError(str(error)) from error

    def _convenience(self, operation, args):
        allowed = {'reference_id', 'offset', 'limit'}
        if operation == 'get_nearest_analogs':
            allowed |= {'k', 'min_tanimoto'}
        if operation == 'get_matched_loading_contrasts':
            allowed |= {'solvent_tolerance'}
        if operation == 'get_matched_solvent_contrasts':
            allowed |= {'loading_relative_tolerance'}
        if not isinstance(args, dict) or set(args)-allowed or not isinstance(args.get('reference_id'), str):
            raise ValueError('invalid convenience query arguments')
        reference = self.references.get(args['reference_id'])
        if reference is None:
            raise ValueError('unknown reference ID')
        offset, limit = args.get('offset', 0), args.get('limit', args.get('k', 24))
        if ('k' in args and 'limit' in args and args['k'] != args['limit']) or type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 24:
            raise ValueError('invalid pagination/k')
        solvent_tol = args.get('solvent_tolerance', 0.0)
        load_tol = args.get('loading_relative_tolerance', 0.0)
        threshold = args.get('min_tanimoto', 0.0)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
               for v in (solvent_tol, load_tol, threshold)):
            raise ValueError('invalid threshold/tolerance')
        rows = []
        for row in self.references.values():
            if operation == 'get_nearest_analogs':
                sim = DataStructs.TanimotoSimilarity(self.fps[reference['smiles']], self.fps[row['smiles']])
                if row['id'] == reference['id'] or sim < threshold:
                    continue
                row = {**row, 'similarity': sim}
            elif row['smiles'] != reference['smiles']:
                continue
            if operation.startswith('get_matched_'):
                if row['id'] == reference['id']:
                    continue
                # Control carrier solvent and volume; disclose actual conditions for all matches.
                if any(row['conditions'][k] != reference['conditions'][k] for k in
                       ('loading solvent', 'Volume of loading solvent/ul')):
                    continue
                delta_pe = abs(pe_fraction(row['conditions']['PE/EA'])-pe_fraction(reference['conditions']['PE/EA']))
                a, b = row['loading_amount_mg'], reference['loading_amount_mg']
                delta_load = abs(a-b)/max(abs(a), abs(b), 1e-12)
                if operation == 'get_matched_loading_contrasts' and (delta_pe > solvent_tol or delta_load <= 1e-12):
                    continue
                if operation == 'get_matched_solvent_contrasts' and (delta_load > load_tol or delta_pe <= 1e-12):
                    continue
            kind = 'observed' if row['id'] in self.observed else 'pending' if row['id'] in self.pending else 'candidate'
            rows.append({**row, 'catalog_role': kind})
        if operation == 'get_nearest_analogs':
            rows.sort(key=lambda r: (-r['similarity'], r['id']))
        else:
            rows.sort(key=lambda r: (pe_fraction(r['conditions']['PE/EA']), r['loading_amount_mg'],
                                    str(r['conditions']['loading solvent']), r['id']))
        total, accepted, limited = len(rows), [], False
        for row in rows[offset:offset+limit]:
            if row['id'] in self.candidates:
                if row['id'] not in self.viewed and len(self.viewed) >= self.view_budget:
                    limited = True
                    continue
                self.viewed.add(row['id'])
            accepted.append(copy.deepcopy(row))
        return {'reference_id': reference['id'], 'matches': accepted, 'total_matches': total,
                'returned_count': len(accepted), 'next_offset': offset+limit if offset+limit < total else None,
                'view_budget_limited': limited}

    def pool_summary(self):
        # Candidates + pending comprise U_t. Measured rows contribute only novelty/similarity X.
        rows = list(self.candidates.values()) + list(self.pending.values())
        counts = Counter(r['smiles'] for r in rows)
        fields = ('cw_distance_percentile', 'max_similarity_to_observed', 'pred_V1_q50_ml',
                  'pred_V2_q50_ml', *SPAN_FIELDS)
        return {'candidate_count': len(self.candidates), 'pending_count': len(self.pending),
                'pool_count': len(rows), 'distinct_molecule_count': len(counts),
                'molecule_condition_count_distribution': quantiles(counts.values()),
                'novel_molecule_fraction': (sum(all(r['molecule_novelty'] for r in rows if r['smiles'] == s)
                                                for s in counts)/len(counts)) if counts else 0,
                'frequencies': {f: dict(sorted(Counter(str(r['conditions'][f]) for r in rows).items()))
                                for f in ('PE/EA', 'loading solvent')},
                'distributions': {f: quantiles(r.get(f) for r in rows) for f in fields},
                'source': 'U_t features/current predictions only; no target labels'}

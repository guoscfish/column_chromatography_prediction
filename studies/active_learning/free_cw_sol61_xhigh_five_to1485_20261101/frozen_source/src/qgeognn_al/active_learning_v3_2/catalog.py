"""Small, explicit query API; successful scientific queries alone consume budget."""
from collections import Counter
import copy
import math
from rdkit import DataStructs
from ..active_learning_v3.catalog import ReadOnlyCatalog as V3Catalog, CONVENIENCE, ERROR_FIELDS, SPAN_FIELDS, pe_fraction
from ..active_learning_v2.scientist_selector import DESCRIPTORS, SIGNALS, CONDITIONS
from .schema import QUERY_BUDGET, VIEW_BUDGET, PER_QUERY_LIMIT
from ..active_learning_v3.memory import quantiles

FIELDS = tuple(f for f in (*DESCRIPTORS, *SIGNALS, *SPAN_FIELDS) if f not in
               ('historical_abs_error_ml', 'V1_quantile_width_ml', 'V2_quantile_width_ml'))
FILTER_SCHEMA = {'type': 'object', 'properties': {'field': {'enum': list(FIELDS + ERROR_FIELDS + ('V1_ml', 'V2_ml'))},
    'op': {'enum': ['eq', 'gte', 'lte']}, 'value': {'type': 'number'}},
    'required': ['field', 'op', 'value'], 'additionalProperties': False}
PAGE = {'offset': {'type': 'integer', 'minimum': 0}, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 24}}

def obj(properties, required=()):
    return {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}

OPERATIONS = {'get_rows': obj({'ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}, **PAGE}, ['ids'])}
for name in ('search_candidates', 'search_observed', 'search_pending'):
    OPERATIONS[name] = obj({**PAGE, 'filters': {'type': 'array', 'items': FILTER_SCHEMA},
        'smiles': {'type': 'string'}, 'same_molecule_as': {'type': 'string'}, 'similar_to': {'type': 'string'},
        'min_tanimoto': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'conditions': obj({k: {'type': ['string', 'number']} for k in CONDITIONS}),
        'sort_by': {'enum': list(FIELDS + ERROR_FIELDS + ('V1_ml', 'V2_ml'))}, 'order': {'enum': ['asc', 'desc']}})
for name in CONVENIENCE:
    extra = ({'min_tanimoto': {'type': 'number', 'minimum': 0, 'maximum': 1}} if name == 'get_nearest_analogs' else
             {'solvent_tolerance': {'type': 'number', 'minimum': 0, 'maximum': 1}} if name == 'get_matched_loading_contrasts' else
             {'loading_relative_tolerance': {'type': 'number', 'minimum': 0, 'maximum': 1}} if name == 'get_matched_solvent_contrasts' else {})
    OPERATIONS[name] = obj({'reference_id': {'type': 'string'}, **PAGE, **extra}, ['reference_id'])
OPERATIONS['get_hypotheses'] = obj({'ids': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2}}, ['ids'])
QUERY_SCHEMA = {'json_schema': {'anyOf': [obj({'operation': {'const': op}, 'args': spec}, ['operation', 'args']) for op, spec in OPERATIONS.items()]},
    'accepted_scientific_query_budget': 24, 'validation_repair_budget': 4, 'failed_queries_consume_budget': False,
    'alias': {'get': 'get_rows (only IDs and pagination accepted; audited)'},
    'filter_semantics': 'AND of flat field/op/value filters; molecule_novelty and condition_novelty are 0/1 signals; CW is cw_distance_percentile in [0,1]; similarity requires similar_to; measured endpoints/residuals observed-only',
    'examples': [{'operation': 'get_rows', 'args': {'ids': ['<viewed ID>']}},
                 {'operation': 'search_candidates', 'args': {'filters': [{'field': 'molecule_novelty', 'op': 'eq', 'value': 1}, {'field': 'max_similarity_to_observed', 'op': 'lte', 'value': .6}], 'limit': 24}},
                 {'operation': 'get_condition_series', 'args': {'reference_id': '<known ID>', 'offset': 0, 'limit': 24}}]}

class ReadOnlyCatalog(V3Catalog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_queries = []
        for row in self.observed.values():
            if 'premeasurement_error_V1_ml' in row:
                row['residual_timing'] = 'historical measurement-time residual; not current model error after retraining'

    def query(self, operation, args):
        original, original_args = operation, copy.deepcopy(args)
        before_views = set(self.viewed)
        try:
            if len(self.queries) >= self.query_budget:
                raise ValueError('query budget exhausted: no more queries are available; return selection only')
            if not isinstance(operation, str) or not isinstance(args, dict):
                raise ValueError('query needs string operation and object args')
            if operation == 'get':
                operation = 'get_rows'
            if operation in ('get_candidates', 'get_observed', 'get_pending'):
                # Exact legacy retrieval remains available, without ambiguous alias repair.
                role = operation[4:]
                allowed_ids = getattr(self, role)
                if any(i not in allowed_ids for i in args.get('ids', [])):
                    raise ValueError('ID outside requested catalog role')
                schema = OPERATIONS['get_rows']
            else:
                schema = OPERATIONS.get(operation)
            if schema is None:
                raise ValueError('unsupported operation; use query_api_schema')
            import jsonschema
            try:
                jsonschema.validate(args, schema)
            except jsonschema.ValidationError as error:
                raise ValueError('invalid query args: ' + error.message[:240]) from None
            if operation in CONVENIENCE:
                response = self._convenience(operation, args)
            elif operation == 'get_hypotheses':
                if any(i not in self.ledger['hypotheses'] for i in args['ids']):
                    raise ValueError('unknown hypothesis ID')
                response = {'matches': [{k: copy.deepcopy(v) for k, v in self.ledger['hypotheses'][i].items() if k != 'history'} for i in dict.fromkeys(args['ids'])]}
            else:
                if operation.startswith('get_'):
                    if any(i not in self.references for i in args['ids']):
                        raise ValueError('unknown row ID')
                    rows = [self.references[i] for i in dict.fromkeys(args['ids'])]
                else:
                    table = getattr(self, operation[7:])
                    filtering = {k: v for k, v in args.items() if k != 'filters'}
                    for name in ('similar_to', 'same_molecule_as'):
                        if name in args and args[name] not in self.references:
                            raise ValueError('unknown reference ID')
                    if 'min_tanimoto' in args and 'similar_to' not in args:
                        raise ValueError('min_tanimoto requires similar_to')
                    rows = self._filtered(table, filtering)
                    for spec in args.get('filters', []):
                        field, op, value = spec['field'], spec['op'], spec['value']
                        if not math.isfinite(value):
                            raise ValueError('nonfinite filter')
                        self._check_signal(field, table, args)
                        def matches(row):
                            v = row['descriptors'].get(field) if field in DESCRIPTORS else row.get(field)
                            return v is not None and {'eq': lambda: v == value, 'gte': lambda: v >= value, 'lte': lambda: v <= value}[op]()
                        rows = [r for r in rows if matches(r)]
                offset, limit = args.get('offset', 0), args.get('limit', 24)
                found, limited = [], False
                for row in rows[offset:offset+limit]:
                    if row['id'] in self.candidates:
                        if row['id'] not in self.viewed and len(self.viewed) >= self.view_budget:
                            limited = True
                            continue
                        self.viewed.add(row['id'])
                    role = 'observed' if row['id'] in self.observed else 'pending' if row['id'] in self.pending else 'candidate'
                    found.append({**copy.deepcopy(row), 'catalog_role': role})
                response = {'matches': found, 'total_matches': len(rows), 'returned_count': len(found),
                    'next_offset': offset+limit if offset+limit < len(rows) else None, 'view_budget_limited': limited}
            response.update(remaining_queries=self.query_budget-len(self.queries)-1,
                remaining_new_candidate_views=self.view_budget-len(self.viewed), normalized_query_alias=original == 'get')
            self.queries.append(copy.deepcopy({'operation': original, 'args': original_args, 'response': response}))
            return response
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.viewed = before_views
            self.invalid_queries.append({'operation': original, 'args': original_args, 'message': str(error)})
            raise ValueError(str(error)) from error


def batch_diagnostic(choices, catalog):
    rows = [catalog.candidates[c['id']] for c in choices]
    molecules = Counter(r['smiles'] for r in rows)
    roles = Counter(c['scientific_role'] for c in choices)
    similarities = [r['max_similarity_to_observed'] for r in rows]
    pairs = [DataStructs.TanimotoSimilarity(catalog.fps[a['smiles']], catalog.fps[b['smiles']])
             for i, a in enumerate(rows) for b in rows[i+1:]]
    return {'selected_count': len(rows), 'unique_molecules': len(molecules),
        'largest_same_molecule_count': max(molecules.values(), default=0),
        'largest_same_molecule_fraction': max(molecules.values(), default=0)/max(1, len(rows)),
        'molecule_counts': dict(sorted(molecules.items())),
        'molecule_count_histogram': dict(sorted(Counter(molecules.values()).items())),
        'hypothesis_linked_counts': dict(Counter(c['hypothesis_id'] or 'unlinked' for c in choices)),
        'scientific_role_counts': dict(roles), 'selected_CW_percentiles': quantiles(r['cw_distance_percentile'] for r in rows),
        'molecule_novelty_count': sum(r['molecule_novelty'] for r in rows),
        'mean_similarity_to_observed': sum(similarities)/max(1, len(rows)),
        'max_similarity_to_observed': max(similarities, default=None),
        'mean_pairwise_structural_similarity': sum(pairs)/len(pairs) if pairs else None,
        'targeted_follow_up_count': sum(roles[r] for r in ('failure_repair','hypothesis_test','condition_curve','matched_control')),
        'broader_exploration_count': sum(roles[r] for r in ('structural_exploration','coverage')),
        'other_count': roles['other'], 'role_counts_are_metadata_only': True, 'enforced_quotas': None}

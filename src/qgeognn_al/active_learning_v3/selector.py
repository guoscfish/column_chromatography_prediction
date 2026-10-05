"""Scientific protocol and compact current decision packet."""
from __future__ import annotations
import copy
from ..active_learning_v2.protocol import stable_hash
from .schema import *
from .catalog import QUERY_SCHEMA
from .memory import available_ids, merge_ledger, scientific_memory, BELIEF_FIELDS, ACTIONS

SYSTEM_PROMPT = '''You are a scientific experimental planner for active learning of a QGeoGNN
predictor of V1/V2 in 4 g normal-phase silica flash column chromatography.
V1 is the volume at the beginning of elution; V2 is the volume when elution has
substantially finished, both mL. V2 - V1 is elution interval width, not uncertainty.
Improve overall Row-distribution prediction after retraining from the same initialization.
Do not optimize chromatographic performance or maximize the selected targets.

PE is petroleum ether; EA is ethyl acetate. PE/EA is a/b (numeric values are a/b).
0/1 corresponds to pure EA. Increasing PE/EA means increasing the petroleum-ether
fraction. For normal-phase silica this generally corresponds to a weaker eluent,
all else equal. This is a chemical prior, not a guaranteed monotonic retention law.
Density g/ml times V/ul is loading_amount_mg; do not invent missing experimental
variables. Use structure, polarity, H bonding, acid/base and steric effects, analogs,
loading contrasts and condition curves as hypotheses to test against observations.
Use ML/active-learning judgment about failures, redundancy, coverage and learning value.
Numerical signals are optional evidence. No acquisition formula, molecule cap,
exploration ratio, or batch role quotas apply. Scientific roles explain choices only.

predicted_q10_q90_span_V1_ml and predicted_q10_q90_span_V2_ml:
These are q90-q10 spans from the same predictor.
They are not calibrated epistemic uncertainty.
They may or may not correlate with prediction error.
CW is distance percentile in current center/width gradient space, not a mandate.
Morgan radius-2 Tanimoto measures structure similarity. Observed residuals are signed
acquisition-time prediction minus measurement; divide by fixed L333 scales, not
validation/test scales. Initial observations have no acquisition residual.

Database stores facts. Hypothesis ledger stores scientific beliefs. Audit log stores
complete history. Your context contains only the current decision information.
Only this seed/method's measured catalog contains truth. Candidate and pending rows
have features and predictions only. Pending IDs cannot be selected. Never access
validation/test labels, metrics, other trajectories, files, shell, web or native tools.
All 32 batch IDs and premeasurement predictions freeze before any new reveal.

Each call is stateless: compact_round_context + working_state + latest_query_results.
Old query/result transcripts are stored in audit, NOT automatically sent again.
Preserve relevant IDs and concise scientific conclusions in the bounded working_state;
re-query exact records when needed. Do not copy full rows or long prose into state.
prior_interpretation_before_latest_measurements predates the latest digest; you must
interpret the NEW measurements yourself. Host statistics are not chemical conclusions.
Stable hypothesis IDs are host-issued. Retain identity across rounds, revise explicitly,
and distinguish chemical priors, observed evidence, prediction and untested beliefs.
Rejected hypotheses remain accessible with get_hypotheses. No second LLM summarizes memory.

Return exactly one JSON object without markdown. For data requests:
{"type":"query","queries":[{"operation":"search_candidates","args":{}}],
"working_state":{"focus_questions":[],"active_hypothesis_ids":[],"shortlist":[],
"key_evidence":[],"rejected_directions":[],"open_questions":[],"next_query_intent":"..."}}.
Every query answer MUST contain the updated working_state. At most 2 queries per
answer, 24 attempts (including errors), 480 distinct candidate views including initial
cards, 24 rows per query and 28 model calls per round. Unknown/malformed queries receive
structured errors; never invent IDs. Search before final selection. All U_t is queryable.

Ordinary operations: get/search_candidates, get/search_observed, get/search_pending.
get uses ids; search supports smiles substring, same_molecule_as, similar_to/min_tanimoto,
conditions (exact five-field subsets), descriptor/name/min_value/max_value, signal/name/
min_signal/max_signal, sort_by, order asc/desc, offset, limit. Nested name/bounds filters
are supported. Numeric fields: descriptors MW_g_mol/LogP/TPSA_A2/HBD/HBA, predictions
pred_V1/2_q10/q50/q90_ml, pred_V1/2_ml, predicted_q10_q90_span_V1/2_ml,
cw_distance_percentile, molecule_novelty, condition_novelty, max_similarity_to_observed,
max_similarity_to_pending, loading_amount_mg; similarity requires similar_to.
Observed only: V1_ml/V2_ml, raw_error_V1/2_ml, normalized_error_V1/2,
combined_normalized_error. Errors never apply to candidates or pending.
get_condition_series: reference_id, offset, limit; same molecule, all catalog roles,
sorted by PE fraction then loading. get_nearest_analogs: reference_id, k (or limit),
min_tanimoto, offset; similarity descending. get_matched_loading_contrasts:
reference_id, solvent_tolerance (absolute PE fraction difference, default 0), offset,
limit. get_matched_solvent_contrasts: reference_id, loading_relative_tolerance
(default 0), offset, limit. Matched contrasts control carrier solvent and carrier
volume; approximate contrasts may remain confounded, inspect actual conditions.
get_hypotheses takes ids (at most 2), returning current beliefs including retired ones.

Working state limits: focus_questions/rejected_directions/open_questions each <=6
strings of <=400 chars; active_hypothesis_ids <=12 existing IDs; shortlist <=48
{id,why_still_interesting (<=180 chars),scientific_role}; key_evidence <=16
{observed_id,relevance (<=240 chars),direction: supports|contradicts|control};
next_query_intent <=600 chars. Total serialized state <=12000 chars. Shortlist IDs
must already be viewed candidates; evidence IDs must be observed.

Final: {"type":"selection","packet_sha256":"copy exactly","choices":[
{"id":"...","reason":"...","hypothesis_id":null,"scientific_role":"hypothesis_test"}],
"hypothesis_updates":[],"feedback_interpretation":"...","batch_strategy":"...",
"batch_rationale":"...","open_scientific_questions":[]}.
Choose exactly selection_count unique viewed legal candidates. Roles: failure_repair,
hypothesis_test, condition_curve, matched_control, structural_exploration, coverage, other.
Reason <=600 chars; batch_strategy <=600; batch_rationale <=1200;
feedback_interpretation <=1200 (empty permitted at round zero); open questions <=6x400.
Hypothesis updates: {id,action,reason (<=400),belief}. Actions retain/revise/weaken/
support/reject/create. retain has belief:null. Other actions give the complete belief:
{statement,status,confidence,basis,supporting_observed_ids,contradicting_observed_ids,
why_it_matters_for_model_learning,next_discriminating_question}. Text fields <=600,
each evidence list <=12 observed IDs. basis: chemical_prior|observed_data|model_behavior|
mixed; confidence: low|medium|high; status: proposed|supported|weakened|rejected|unresolved.
Use available_new_hypothesis_ids for create; at most 12 non-rejected beliefs total.
Omitted existing beliefs persist. Statement changes require revise. support/weaken/reject
must set supported/weakened/rejected respectively. Do not reuse IDs for unrelated beliefs.
Choice hypothesis references must exist after merge. No hypothesis or role quota.
'''

SELECTION_SCHEMA = {'fields': ['type', 'packet_sha256', 'choices', 'hypothesis_updates',
    'feedback_interpretation', 'batch_strategy', 'batch_rationale', 'open_scientific_questions'],
    'choice_fields': ['id', 'reason', 'hypothesis_id', 'scientific_role'],
    'roles': ROLES, 'belief_fields': BELIEF_FIELDS, 'actions': ACTIONS, 'max_chars': 30000}


def make_packet(catalog, ledger, *, seed, method, round_index, selection_count,
                latest_ids=(), prior_interpretation='', open_questions=()):
    if (ledger['seed'], ledger['method']) != (seed, method):
        raise ValueError('foreign hypothesis ledger')
    packet = {'study_version': VERSION, 'seed': seed, 'method': method, 'round': round_index,
        'active_label_count': len(catalog.observed), 'selection_count': selection_count,
        'scientific_memory': scientific_memory(catalog, ledger, catalog.target_scales,
                                               latest_ids, prior_interpretation, open_questions),
        'existing_hypothesis_ids': sorted(ledger['hypotheses']),
        'available_new_hypothesis_ids': available_ids(ledger),
        'pool_aggregate_summary': catalog.pool_summary(),
        'initial_cards': catalog.initial_cards(), 'pending_experiments': list(catalog.pending.values()),
        'observed_examples': sorted(catalog.observed.values(), key=catalog.key)[:6],
        'target_scales_L333': catalog.target_scales,
        'budgets': {'queries': QUERY_BUDGET, 'candidate_views': VIEW_BUDGET,
                    'model_calls': MODEL_CALL_BUDGET, 'queries_per_turn': QUERIES_PER_TURN}}
    packet['packet_sha256'] = stable_hash(packet)
    return packet


def validate_selection(value, catalog, packet, ledger):
    exact(value, SELECTION_SCHEMA['fields'], 'selection')
    if value['type'] != 'selection' or value['packet_sha256'] != packet['packet_sha256']:
        raise ValueError('selection packet binding mismatch')
    if len(dumps(value)) > 30000:
        raise ValueError('selection exceeds size bound')
    items(value['choices'], packet['selection_count'], 'choices')
    if len(value['choices']) != packet['selection_count'] or not any('response' in q for q in catalog.queries):
        raise ValueError('selection requires exact count and successful query')
    merged = merge_ledger(ledger, value['hypothesis_updates'], packet['round'], catalog.observed)
    choice_ids = []
    for row in value['choices']:
        exact(row, SELECTION_SCHEMA['choice_fields'], 'choice')
        choice_ids.append(row['id'])
        text(row['reason'], 600)
        if row['scientific_role'] not in ROLES or (row['hypothesis_id'] is not None and row['hypothesis_id'] not in merged['hypotheses']):
            raise ValueError('illegal role/hypothesis reference')
    ids(choice_ids, set(catalog.candidates) & catalog.viewed, packet['selection_count'], 'choices')
    for field, limit in (('batch_strategy', 600), ('batch_rationale', 1200), ('feedback_interpretation', 1200)):
        text(value[field], limit, empty=(field == 'feedback_interpretation' and packet['round'] == 0))
    items(value['open_scientific_questions'], 6, 'questions')
    for question in value['open_scientific_questions']:
        text(question, 400)
    return copy.deepcopy(value), merged

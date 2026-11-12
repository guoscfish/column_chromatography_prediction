"""Profile-specific recorded-response validation, no provider calls or state mutation."""
from __future__ import annotations
import copy
import json
import math
import re
from types import SimpleNamespace
from ...infra.contracts import content_hash as stable_hash, sample_ids
from .catalog import CONDITIONS, DESCRIPTORS, PREDICTIONS, OBSERVED_FIELDS, decode_table
from .profiles import get_profile

MAX_HYPOTHESES = 8
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

def validate_screen_response(response, candidate_ids, fallback_ids, packet_hash):
    """One attempt, retain valid choices, deterministically fill invalid/missing IDs."""
    if len(candidate_ids) != 128 or len(set(candidate_ids)) != 128 or set(fallback_ids) != set(candidate_ids):
        raise ValueError('invalid candidate or fallback universe')
    errors, accepted = [], []
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = None
            errors.append('invalid_json')
    if not isinstance(response, dict):
        errors.append('response_not_object')
        response = {'packet_sha256': packet_hash}
    if response.get('packet_sha256') != packet_hash:
        raise ValueError('response is not bound to this frozen packet')
    choices = response.get('choices', [])
    if not isinstance(choices, list):
        choices = []
        errors.append('choices_not_list')
    if len(choices) != 16:
        errors.append('wrong_choice_count')
    for item in choices[:16]:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str)
                or item['id'] not in candidate_ids or item['id'] in [v['id'] for v in accepted]
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
            errors.append('invalid_duplicate_or_missing_reason')
            continue
        accepted.append({'id': item['id'], 'reason': item['reason'], 'origin': 'llm'})
    valid = len(accepted)
    for value in fallback_ids:
        if len(accepted) == 16:
            break
        if value not in [v['id'] for v in accepted]:
            accepted.append({'id': value, 'reason': 'Preregistered random fallback', 'origin': 'fallback'})
    return {'choices': accepted, 'errors': errors, 'fallback_count': 16-valid,
            'response_failed': bool(errors), 'packet_sha256': packet_hash}

CW_FIELD = "cw_distance_percentile"

CW_PATTERN = re.compile(
    r"\bcw\b|cw[_ -]?(?:distance|rank|score|coverage|percentile)|\blcmd\b|"
    r"(?:center[ /_-]*(?:and[ /_-]*)?width|gradient|geometric)[\s_-]*"
    r"(?:gradient[\s_-]*)?(?:distance|coverage|rank|percentile|score)|"
    r"梯度.{0,12}(?:距离|覆盖|排名|分位|评分)|中心.{0,6}宽度|"
    r"(?:distance|coverage|rank|percentile|score)[\s_-]*(?:in[\s_-]*)?(?:gradient|cw)", re.I)

def audit_no_cw(value):
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    # Actual messages can contain a JSON string nested inside content (raw
    # answers and corrections). Decode Unicode escapes before scanning those
    # strings too, so ASCII-escaped Chinese/English cannot bypass this boundary.
    def visible_strings(item):
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for key, child in item.items():
                yield str(key)
                yield from visible_strings(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                yield from visible_strings(child)
    for text in (serialized, *visible_strings(value)):
        for _ in range(4):
            decoded = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1],16)), text)
            if decoded == text:
                break
            text = decoded
        match = CW_PATTERN.search(text)
        if match:
            raise ValueError(f"NoCW visible content rejected: {match.group(0)}")
    return {"passed": True, "serialized_sha256": stable_hash(value),
            "serialized_characters": len(serialized)}

BASE = {"id", "smiles", "conditions", "descriptors"}

PRED = {"pred_V1_ml", "pred_V2_ml", *PREDICTIONS}

FEEDBACK_FIELDS = {"candidate_id", "source", "pred_V1_ml", "pred_V2_ml", "true_V1_ml",
                   "true_V2_ml", "error_V1_ml", "error_V2_ml", "reason", "hypothesis_id"}

HYPOTHESIS_FIELDS = {"id", "content", "basis", "supporting_observed_ids", "contradicting_observed_ids",
                     "confidence", "status", "why_it_matters_for_model_learning", "next_discriminating_question"}

PLAN_FIELDS = {"type", "packet_sha256", "batch_strategy", "choices", "hypotheses",
               "feedback_interpretation", "batch_rationale"}

HISTORY_FIELDS = {"seed", "method", "round", "batch_freeze_sha256", "records", "hypotheses",
                  "feedback_interpretation", "batch_rationale"}

def _only(value, fields, kind):
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"unauthorized {kind} fields")

def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric field required")

def validate_card(row, observed):
    if not isinstance(row["id"], str) or not isinstance(row["smiles"], str):
        raise ValueError("string identity/SMILES required")
    conditions = row["conditions"]
    ratio = conditions["PE/EA"]
    if not isinstance(ratio, str) or not re.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", ratio):
        raise ValueError("invalid PE/EA ratio")
    if conditions["loading solvent"] not in ("PE", "EA", "DCM"):
        raise ValueError("unregistered loading solvent")
    for k in ("Density g/ml", "V/ul", "Volume of loading solvent/ul"):
        _finite(conditions[k])
    if "descriptors" in row:
        _only(row["descriptors"], set(DESCRIPTORS), "descriptors")
        for value in row["descriptors"].values():
            _finite(value)
    for k in (set(OBSERVED_FIELDS)-{"record_kind"} if observed else PRED | {CW_FIELD}):
        if k in row:
            _finite(row[k])
    if observed and (not {"V1_ml", "V2_ml", "record_kind"} <= set(row) or
            row["record_kind"] not in ("initial_L333", "acquired_with_premeasurement_prediction")):
        raise ValueError("measured record schema mismatch")

def _table_catalog(packet):
    mappings = {name: {r['id']: r for r in decode_table(packet, name)} for name in ('candidates','observed','pending')}
    for name in mappings:
        if len(mappings[name]) != len(packet[name]['rows']):
            raise ValueError('duplicate encoded IDs')
    groups = list(mappings.values())
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError('encoded candidate/observed/pending overlap')
    return SimpleNamespace(**mappings, references={key:row for m in groups for key,row in m.items()})


def validate_packet(profile, packet):
    p = get_profile(profile)
    if p.delivery == 'screened_128':
        from .prompts import prompt
        allowed_top = {'prompt','labeled_count','pending_CW16','candidates','units',
                       'observed_outcome_values_in_packet','validation_or_test_records_in_packet','packet_sha256'}
        if set(packet) != allowed_top or packet['prompt'] != prompt(p):
            raise ValueError('unauthorized screen packet fields or Prompt drift')
        if (type(packet['labeled_count']) is not int or packet['labeled_count'] not in (333,365,397)
                or not re.fullmatch('[0-9a-f]{64}', packet.get('packet_sha256',''))
                or packet['units'] != {'pred_V1_V2':'mL','MW':'g/mol','TPSA':'angstrom^2'}):
            raise ValueError('screen frozen budget, units or packet binding mismatch')
        if (len(packet.get('candidates', [])) != 128 or len(packet.get('pending_CW16', [])) != 16
                or packet.get('observed_outcome_values_in_packet') != 0
                or packet.get('validation_or_test_records_in_packet') != 0):
            raise ValueError('screen packet counts/truth barrier mismatch')
        candidate_ids = sample_ids([r['id'] for r in packet['candidates']], name='screen candidates')
        pending_ids = sample_ids([r['id'] for r in packet['pending_CW16']], name='screen pending')
        if set(candidate_ids) & set(pending_ids):
            raise ValueError('screen candidates overlap pending experiments')
        allowed = {'id','smiles','conditions','descriptors','pred_V1_V2','cw_distance_percentile',
                   'max_Tanimoto_to_Lt','same_molecule_rows_in_Lt'}
        for row in packet['candidates']+packet['pending_CW16']:
            if set(row)!=allowed or not isinstance(row['smiles'],str) or set(row['conditions'])!=set(CONDITIONS):
                raise ValueError('unauthorized screen card fields')
            if set(row['descriptors'])!={'MW','LogP','TPSA','HBD','HBA'}:
                raise ValueError('unauthorized screen descriptors')
            for value in row['descriptors'].values():
                _finite(value)
            if not isinstance(row['pred_V1_V2'],list) or len(row['pred_V1_V2'])!=2:
                raise ValueError('screen prediction layout mismatch')
            for value in row['pred_V1_V2']+[row['cw_distance_percentile'],row['max_Tanimoto_to_Lt']]:
                _finite(value)
            validate_card({'id':row['id'],'smiles':row['smiles'],'conditions':row['conditions']},False)
            count=row['same_molecule_rows_in_Lt']
            if type(count) is not int or not 0<=count<=packet['labeled_count']:
                raise ValueError('screen observed molecule count mismatch')
        return None
    allowed = {'study_version','seed','method','round','active_label_count','selection_count','objective',
               'memory','observed_count','validation_test_records','delivery','candidate_count',
               'condition_columns','molecules','candidates','observed','pending','packet_sha256'}
    if set(packet)-allowed or packet.get('delivery') != p.delivery or packet.get('validation_test_records') != 0:
        raise ValueError('unauthorized full packet fields or truth barrier')
    if packet.get('selection_count') != p.selection_count:
        raise ValueError('profile selection count mismatch')
    if packet.get('packet_sha256') != stable_hash({k:v for k,v in packet.items() if k!='packet_sha256'}):
        raise ValueError('packet binding mismatch')
    if packet.get('condition_columns') != list(CONDITIONS):
        raise ValueError('condition schema mismatch')
    signal = {'V1_quantile_width_ml','V2_quantile_width_ml','condition_novelty','loading_amount_mg',
              'max_similarity_to_observed','max_similarity_to_pending','molecule_novelty','historical_abs_error_ml'}
    for name in ('candidates','pending','observed'):
        fields = {'id','molecule_id','conditions'} | signal | (set(OBSERVED_FIELDS) if name=='observed' else PRED|{CW_FIELD})
        if set(packet[name]['columns'])-fields or len(set(packet[name]['columns']))!=len(packet[name]['columns']):
            raise ValueError('unauthorized encoded table column')
    for molecule in packet['molecules'].values():
        if set(molecule)!={'smiles','descriptors'} or not isinstance(molecule['smiles'],str):
            raise ValueError('unauthorized molecule fields')
        if set(molecule['descriptors'])!=set(DESCRIPTORS):
            raise ValueError('unauthorized molecular descriptors')
        for value in molecule['descriptors'].values():
            _finite(value)
    catalog = _table_catalog(packet)
    for name in ('candidates','pending','observed'):
        for row in getattr(catalog,name).values():
            validate_card({k:v for k,v in row.items() if v is not None},name=='observed')
    memory=packet['memory']
    expected_memory={'previous_hypotheses','previous_feedback_interpretation','previous_batch_rationale',
                     'previous_selected_ids','recent_outcomes','high_error_measured_examples','memory_policy'}
    if set(memory)!=expected_memory or not set(memory['previous_selected_ids'])<=set(catalog.observed):
        raise ValueError('memory includes unknown fields or unobserved IDs')
    for record in memory['recent_outcomes']+memory['high_error_measured_examples']:
        _only(record,FEEDBACK_FIELDS,'visible measured feedback')
        if record['candidate_id'] not in catalog.observed:
            raise ValueError('feedback contains an unobserved ID')
    for hypothesis in memory['previous_hypotheses']:
        _only(hypothesis,HYPOTHESIS_FIELDS,'visible hypothesis')
        for field in ('supporting_observed_ids','contradicting_observed_ids'):
            if not set(hypothesis[field])<=set(catalog.observed):
                raise ValueError('hypothesis memory cites unobserved evidence')
    if (packet.get('candidate_count') != len(catalog.candidates)
            or packet.get('observed_count') != len(catalog.observed)
            or packet.get('active_label_count') != len(catalog.observed)
            or len(catalog.pending) != p.pending_count):
        raise ValueError('full packet pool/observed/pending count mismatch')
    if not p.cw_visible:
        audit_no_cw(packet)
    return catalog


def validate_response(profile, raw, packet, *, fallback_ids=(), pending_ids=()):
    p = get_profile(profile)
    catalog = validate_packet(p, packet)
    if p.delivery == 'screened_128':
        result = validate_screen_response(raw, [r['id'] for r in packet['candidates']],
                                          list(fallback_ids), packet['packet_sha256'])
        return tuple(r['id'] for r in packet['pending_CW16']) + tuple(r['id'] for r in result['choices'])
    result = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    validate_direct(result, catalog, packet)
    if p.name.startswith('free32_'):
        _only(result, PLAN_FIELDS, 'plan')
        for h in result['hypotheses']:
            _only(h, HYPOTHESIS_FIELDS, 'hypothesis')
        for choice in result['choices']:
            _only(choice, {'id','reason','hypothesis_id'}, 'choice')
    if p.name.endswith('_paired_v1'):
        prose = [result[k] for k in ('batch_strategy','batch_rationale','feedback_interpretation')]
        prose += [c['reason'] for c in result['choices']]
        prose += [h[k] for h in result['hypotheses'] for k in ('content','why_it_matters_for_model_learning','next_discriminating_question')]
        for row_id in catalog.references:
            if any(re.search(r'(?<![\w])'+re.escape(row_id)+r'(?![\w])', text) for text in prose):
                raise ValueError('row IDs in prose are forbidden; use typed observed evidence lists')
    if not p.cw_visible:
        audit_no_cw(result)
    pending = sample_ids(pending_ids, name='ordered numerical pending')
    if len(pending)!=p.pending_count or set(pending)!=set(catalog.pending):
        raise ValueError('explicit numerical pending order required; encoded table order is different')
    return pending + tuple(c['id'] for c in result['choices'])


def packet_candidate_ids(profile, packet):
    """Visible candidate order; numerical pending remains a separate ordered input."""
    p = get_profile(profile)
    catalog = validate_packet(p, packet)
    if p.delivery == 'screened_128':
        return tuple(row['id'] for row in packet['candidates'])
    return tuple(catalog.candidates)

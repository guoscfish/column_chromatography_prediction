"""Strict structured wire schema and deterministic, non-generative JSON recovery."""
import copy
import json
from .catalog import QUERY_SCHEMA, obj
from .schema import ROLES, WORKING_SCHEMA
from .memory import BELIEF_FIELDS, ACTIONS

STR = {'type': 'string'}
def array(item):
    return {'type': 'array', 'items': item}

def strict(schema):
    """Responses requires all properties required; optional wire fields use null.

    Only documented optional query fields may be null; decoding omits those fields.
    This is a registered wire representation, never recovery of scientific content.
    """
    s = copy.deepcopy(schema)
    if 'const' in s:
        s['enum'] = [s.pop('const')]
        s['type'] = 'string'
    if 'enum' in s and 'type' not in s:
        s['type'] = 'string'
    if s.get('type') == 'object':
        required = s.get('required', [])
        s['properties'] = {k: strict(v) if k in required else {'anyOf': [strict(v), {'type':'null'}]}
                           for k, v in s['properties'].items()}
        s['required'] = list(s['properties'])
        s['additionalProperties'] = False
    if 'items' in s:
        s['items'] = strict(s['items'])
    if 'anyOf' in s:
        s['anyOf'] = [strict(v) for v in s['anyOf']]
    return s

WORKING = obj({k: STR if k == 'next_query_intent' else array(STR) for k in WORKING_SCHEMA}, WORKING_SCHEMA)
WORKING['properties']['shortlist'] = array(obj({'id': STR, 'why_still_interesting': STR, 'scientific_role': {'type':'string','enum':list(ROLES)}}, ['id','why_still_interesting','scientific_role']))
WORKING['properties']['key_evidence'] = array(obj({'observed_id':STR, 'relevance':STR, 'direction':{'type':'string','enum':['supports','contradicts','control']}}, ['observed_id','relevance','direction']))
BELIEF = obj({k: array(STR) if k.endswith('_ids') else STR for k in BELIEF_FIELDS}, BELIEF_FIELDS)
CHOICE = obj({'id':STR, 'reason':STR, 'hypothesis_id':{'type':['string','null']}, 'scientific_role':{'type':'string','enum':list(ROLES)}}, ['id','reason','hypothesis_id','scientific_role'])
UPDATE = obj({'id':STR,'action':{'type':'string','enum':list(ACTIONS)},'reason':STR,'belief':{'anyOf':[BELIEF, {'type':'null'}]}}, ['id','action','reason','belief'])
SELECTION = obj({'type':{'const':'selection'},'packet_sha256':STR,'choices':array(CHOICE),'hypothesis_updates':array(UPDATE),
    'feedback_interpretation':STR,'batch_strategy':STR,'batch_rationale':STR,'open_scientific_questions':array(STR)},
    ['type','packet_sha256','choices','hypothesis_updates','feedback_interpretation','batch_strategy','batch_rationale','open_scientific_questions'])
QUERY = obj({'type':{'const':'query'},'queries':array(QUERY_SCHEMA['json_schema']),'working_state':WORKING}, ['type','queries','working_state'])
RESPONSE_SCHEMA = strict(obj({'response': {'anyOf':[QUERY, SELECTION]}}, ['response']))

def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result

def _constant(value):
    raise ValueError('nonfinite JSON constant')

DECODER = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)

def decode_json(answer):
    """Allow exactly one top-level object, optionally surrounded by plain prose.

    Scan complete top-level containers, respecting nested braces/quoted strings.
    Unbalanced containers, arrays, duplicate keys and multiple objects fail closed.
    """
    if not isinstance(answer, str):
        raise ValueError('expected JSON text')
    try:
        value = DECODER.decode(answer.strip())
    except ValueError:
        starts, stack, quoted, escaped, begin = [], [], False, False, None
        for i, char in enumerate(answer):
            if not stack:
                if char in '{[':
                    stack.append(char); begin = i; quoted = False
                elif char in '}]':
                    raise ValueError('unmatched JSON delimiter')
                continue
            if quoted:
                if escaped: escaped = False
                elif char == '\\': escaped = True
                elif char == '"': quoted = False
                continue
            if char == '"': quoted = True
            elif char in '{[': stack.append(char)
            elif char in '}]':
                if (stack.pop(), char) not in (('{','}'),('[',']')):
                    raise ValueError('mismatched JSON delimiter')
                if not stack: starts.append(answer[begin:i+1])
        if stack or len(starts) != 1:
            raise ValueError('recovery requires exactly one complete top-level JSON object')
        value = DECODER.decode(starts[0])
        recovered = True
    else:
        recovered = False
    if not isinstance(value, dict):
        raise ValueError('expected top-level JSON object')
    return value, recovered

def decode_wire(value):
    """Validate exact structured envelope before omitting nullable optional args."""
    import jsonschema
    jsonschema.validate(value, RESPONSE_SCHEMA)
    result = copy.deepcopy(value['response'])
    if result['type'] == 'query':
        for query in result['queries']:
            query['args'] = {k:v for k,v in query['args'].items() if v is not None}
            if 'conditions' in query['args']:
                query['args']['conditions'] = {k:v for k,v in query['args']['conditions'].items() if v is not None}
    return result

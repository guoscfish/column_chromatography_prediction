"""Load only audited pure validation ASTs; never import experiment runners."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]


def source_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validator(method):
    source=ROOT/'src/qgeognn_al/active_learning_v2'
    ns={'copy':copy,'json':json,'re':re,'MAX_HYPOTHESES':8,
        'stable_hash':lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    base=source/'scientist_full_delivery.py'
    tree=ast.parse(base.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'decode_table','validate_direct'}]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(base),'exec'),ns)
    if method=='historical_free_mixed_model':
        return ns['validate_direct'],[base]
    inputs=source/'nocw_inputs.py';tree=ast.parse(inputs.read_text())
    assignments={'CW_PATTERN','HYPOTHESIS_FIELDS','PLAN_FIELDS'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_only','audit_no_cw'} or
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in n.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(inputs),'exec'),ns)
    ns['views']=SimpleNamespace(**{k:ns[k] for k in ['_only','PLAN_FIELDS','HYPOTHESIS_FIELDS','audit_no_cw']})
    if method=='nocw':
        wrapper=ROOT/'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py';variant='no_cw'
    elif method=='new_free_cw_sol61_xhigh':
        wrapper=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py';variant='cw_aware'
    else:raise ValueError('Unknown historical method')
    tree=ast.parse(wrapper.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_plan')
    function.body=[n for n in function.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(wrapper),'exec'),ns)
    def validate(plan,catalog,packet):return ns['validate_plan'](plan,catalog,packet,variant)
    return validate,[base,inputs,wrapper]


def catalog_identity(packet):
    def ids(name):
        table=packet[name];column=table['columns'].index('id')
        return {row[column]:{} for row in table['rows']}
    return SimpleNamespace(candidates=ids('candidates'),observed=ids('observed'),pending=ids('pending'))


def parse_answer(raw):
    try:return json.loads(raw) if isinstance(raw,str) else raw,None
    except (json.JSONDecodeError,TypeError) as error:return None,type(error).__name__


def id_diagnostics(plan,pool,history,pending=()):
    if not isinstance(plan,dict) or not isinstance(plan.get('choices'),list):
        return {'choices_parseable':False,'choice_ids':None,'choices_n':None,'unique_legal_n':None}
    choices=plan['choices']
    if not all(isinstance(x,dict) and isinstance(x.get('id'),str) for x in choices):
        return {'choices_parseable':False,'choice_ids':None,'choices_n':len(choices),'unique_legal_n':None}
    ids=[x['id'] for x in choices];p=set(pool)-set(pending);h=set(history)
    return {'choices_parseable':True,'choice_ids':ids,'choices_n':len(ids),'unique_ids_n':len(set(ids)),
            'duplicate_occurrences':len(ids)-len(set(ids)),'unique_legal_n':len(set(ids)&p),
            'unknown_occurrences':sum(i not in p and i not in h and i not in set(pending) for i in ids),
            'observed_occurrences':sum(i in h for i in ids),'pending_occurrences':sum(i in set(pending) for i in ids)}


def error_category(error):
    text=str(error).lower()
    if 'json' in text or 'expecting' in text or 'invalid control character' in text:return 'format_json'
    if 'packet' in text and ('binding' in text or 'hash' in text):return 'packet_binding'
    if 'distinct legal' in text or 'candidate' in text and ('supplied' in text or 'viewed' in text):return 'candidate_identity'
    if 'unobserved evidence' in text or 'cites unobserved' in text or 'observed-table' in text and 'only' in text:return 'evidence_membership'
    if 'wrong selection count' in text:return 'selection_count_or_structure'
    if 'unknown' in text and 'field' in text or 'unauthorized' in text:return 'field_whitelist'
    if 'cw' in text and ('forbidden' in text or 'visible' in text):return 'nocw_forbidden_content'
    if 'hypothesis' in text or 'hypotheses' in text:return 'hypothesis_schema'
    if 'reason' in text or 'rationale' in text or 'scientific plan' in text or 'feedback_interpretation' in text:
        return 'text_or_length_schema'
    if not text or text=='none':return 'none'
    return 'other_recorded_failure'

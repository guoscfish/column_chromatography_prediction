"""Deterministic authorized packets and correction transcripts for frozen profiles.

Inputs are explicit per-trajectory public catalogs and already measured feedback.
No truth-store, model, filesystem, transport or alternate experiment is consulted.
"""
from __future__ import annotations
import copy
import json
from ...infra.contracts import content_hash as stable_hash, sample_ids
from .profiles import get_profile
from .prompts import prompt
from .catalog import FullCatalog, encode_catalog, scientific_memory, CONDITIONS, OBSERVED_FIELDS
from .validation import (BASE, PRED, CW_FIELD, HISTORY_FIELDS, FEEDBACK_FIELDS,
                         HYPOTHESIS_FIELDS, _only, validate_card, audit_no_cw,
                         validate_packet)


def _history(profile, history, seed, round_index):
    if len(history) != round_index:
        raise ValueError('incomplete trajectory history')
    for index, item in enumerate(history):
        _only(item, HISTORY_FIELDS, 'history')
        if (item['seed'], item['method'], item['round']) != (seed, profile.trajectory_method, index):
            raise ValueError('foreign trajectory history')
        if len(item['records']) != profile.batch_size:
            raise ValueError('incomplete feedback batch')
        for record in item['records']:
            _only(record, FEEDBACK_FIELDS, 'feedback')
            allowed = {'LLM'} if profile.pending_count == 0 else {'CW', 'LLM'}
            if record['source'] not in allowed:
                raise ValueError('foreign selection source')
        for hypothesis in item['hypotheses']:
            _only(hypothesis, HYPOTHESIS_FIELDS, 'hypothesis')
    if not profile.cw_visible:
        # Private method metadata does not cross the visible NoCW boundary.
        audit_no_cw([{k:v for k,v in h.items() if k!='method'} for h in history])


def build_full_packet(profile, raw_catalog, *, seed, round_index, history,
                      candidate_ids, observed_ids, catalog_view='raw_with_cw'):
    """Build the complete pool; callers must supply its independent authorization.

    candidate_ids is the complete authorized U_t, including any numerical pending
    IDs. observed_ids is L_t. No numerical shortlist or hidden pool truncation is
    introduced. Preserve raw_catalog['pending'] order separately for freezing.
    """
    p = get_profile(profile)
    if catalog_view not in ('raw_with_cw','source_visible'):
        raise ValueError('unknown catalog input stage')
    if p.delivery != 'all_candidates_one_request':
        raise ValueError('screen v1 requires its independent 128-candidate constructor')
    if type(seed) is not int or seed < 0 or type(round_index) is not int or round_index < 0:
        raise ValueError('explicit nonnegative seed/round required')
    authorized = sample_ids(candidate_ids, name='complete authorized U_t')
    observed = sample_ids(observed_ids, name='authorized measured L_t')
    if set(authorized) & set(observed):
        raise ValueError('candidate/observed authorization overlap')
    _only(raw_catalog, {'candidates','observed','pending','salt'}, 'catalog')
    raw = copy.deepcopy(raw_catalog)
    if catalog_view=='source_visible' and not p.cw_visible:
        audit_no_cw(raw)
    if len(raw['pending']) != p.pending_count:
        raise ValueError('profile pending count mismatch')
    if (set(row['id'] for row in raw['candidates']+raw['pending']) != set(authorized)
            or set(row['id'] for row in raw['observed']) != set(observed)):
        raise ValueError('catalog must cover complete authorized U_t and L_t exactly')
    _history(p, history, seed, round_index)
    for name in ('candidates','observed','pending'):
        for row in raw[name]:
            measured = name=='observed'
            allowed = BASE | (set(OBSERVED_FIELDS) if measured else PRED | {CW_FIELD})
            _only(row, allowed, name)
            required=PRED | ({CW_FIELD} if p.cw_visible or catalog_view=='raw_with_cw' else set())
            if not measured and not required <= set(row):
                raise ValueError('candidate prediction/CW schema incomplete')
            if set(row['conditions']) != set(CONDITIONS):
                raise ValueError('condition schema mismatch')
            validate_card(row, measured)
            if not p.cw_visible:
                row.pop(CW_FIELD, None)
    if p.name.endswith('_paired_v1'):
        version='llm32_nocw_paired_v1'
        salt=f'{version}:{seed}:paired:{round_index}'
        visible_method=None
    elif p.name in {'free32_cwaware','free32_nocw'}:
        version='llm_scientist_v2'
        visible_method='free_llm32_scientist_v2'
        salt=f'{version}:{seed}:{visible_method}:{round_index}'
    else:
        version='llm_scientist_v2'
        visible_method='cw16_llm16_scientist_v2'
        salt=f'{version}:{seed}:{visible_method}:{round_index}'
    raw['salt']=salt
    catalog=FullCatalog(**raw)
    memory=scientific_memory(history, seed, p.trajectory_method)
    if p.name.endswith('_paired_v1'):
        memory['memory_policy']=memory['memory_policy'].replace('older records queryable', 'older records included in observed table')
    packet={'study_version':version,'seed':seed,'round':round_index,
            'active_label_count':len(catalog.observed),'selection_count':p.selection_count,
            'objective':'overall Row V1/V2 predictive accuracy after retraining',
            'memory':memory,'validation_test_records':0,
            'delivery':'all_candidates_one_request','candidate_count':len(catalog.candidates),
            'observed_count':len(catalog.observed),**encode_catalog(catalog)}
    if p.name in {'free32_cwaware','free32_nocw'}:
        # The actual five-seed overrides append method to the paired packet;
        # do not reorder it into Scientist V2's earlier position in messages.
        packet['method']=visible_method
    elif visible_method is not None:
        packet={'study_version':version,'seed':seed,'method':visible_method,'round':round_index,
                'active_label_count':len(catalog.observed),'selection_count':p.selection_count,
                'objective':'overall Row V1/V2 predictive accuracy after retraining',
                'memory':memory,'observed_count':len(catalog.observed),'validation_test_records':0,
                'delivery':'all_candidates_one_request','candidate_count':len(catalog.candidates),
                **encode_catalog(catalog)}
    packet['packet_sha256']=stable_hash(packet)
    validate_packet(p, packet)
    if not p.cw_visible:
        audit_no_cw({'prompt':prompt(p),'catalog':raw,'packet':packet})
    return packet


def build_profile_input(profile, raw_catalog, **kwargs):
    return {'packet':build_full_packet(profile, raw_catalog, **kwargs),
            'pending_ids':[row['id'] for row in raw_catalog['pending']]}


def build_messages(profile, packet):
    p=get_profile(profile)
    validate_packet(p, packet)
    messages=[{'role':'system','content':prompt(p)},
              {'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
    if not p.cw_visible:
        audit_no_cw(messages)
    return messages


def correction_messages(profile, raw_answer, error):
    """Append only protocol-specific correction, preserving the original answer.

    Screen v1 has one attempt and its own deterministic fallback; no correction
    dialogue is fabricated for that historical protocol.
    """
    p=get_profile(profile)
    if p.delivery=='screened_128':
        raise ValueError('screen v1 is a one-attempt/fallback protocol')
    if p.name in {'cw16_llm16_scientist_v2','free16_cwgradient_lcmd16'}:
        instruction=('Return a corrected final JSON selection from the SAME complete catalog. No queries. '
                     'Only observed-table IDs may be cited as measured evidence; use empty evidence lists for prediction-only hypotheses.')
        feedback={'error':str(error)[:1000],'instruction':instruction}
        content=json.dumps(feedback,ensure_ascii=False)
    else:
        instruction=('Return corrected final JSON from the SAME complete pool. Exactly 32 distinct candidate IDs. '
                     'Measured evidence belongs only in typed observed-ID lists. No queries or tools.')
        content=json.dumps({'error':str(error)[:1000],'instruction':instruction})
    additions=[{'role':'assistant','content':raw_answer},{'role':'user','content':content}]
    if not p.cw_visible:
        audit_no_cw(additions)
    return additions


def replay_recorded_case(case):
    """Verify original bytes and replay the selection layer of a golden record.

    This result never qualifies training, reveal, metrics or a complete AL
    trajectory. Original turns/pool/Prompt must be present for each claimed
    comparison; missing data is not replaced by a synthetic transcript.
    """
    import hashlib
    from .validation import validate_response, validate_screen_response
    p=get_profile(case['profile'])
    files=case['files']
    for name, record in files.items():
        if hashlib.sha256(record['text'].encode()).hexdigest()!=record['sha256']:
            raise ValueError(f'recorded source bytes changed: {name}')
    if 'prompt' in case and hashlib.sha256(case['prompt']['text'].encode()).hexdigest()!=case['prompt']['sha256']:
        raise ValueError('recorded Prompt bytes changed')
    def load(name):
        return json.loads(files[name]['text'])
    frozen=load('batch_freeze.json')
    checked_artifacts=0
    for name,digest in frozen.get('artifact_hashes',{}).items():
        if name in files:
            if digest!=files[name]['sha256']:
                raise ValueError('original freeze artifact binding mismatch')
            checked_artifacts+=1
    if p.delivery=='screened_128':
        packet=load('packet.json')
        packet['packet_sha256']=files['packet.json']['sha256']
        proposal=load('proposal.json')
        position_ids=dict(zip(proposal['candidate_positions'],[r['id'] for r in packet['candidates']]))
        fallback=[position_ids[i] for i in proposal['fallback_positions']]
        raw=files['response.json']['text']
        supplement=validate_screen_response(raw,[r['id'] for r in packet['candidates']],fallback,packet['packet_sha256'])
        if supplement!=load('validated_response.json'):
            raise ValueError('screen validator/fallback diff')
        ids=validate_response(p,raw,packet,fallback_ids=fallback)
        prompt_equal=packet['prompt']==prompt(p)
        if not prompt_equal:
            raise ValueError('screen frozen Prompt diff')
        request_hashes=[]
        packet_sha=packet['packet_sha256']
        turns=1
        corrections=0
        message_evidence='UNKNOWN_ORIGINAL_OUTGOING_TRANSCRIPT; frozen packet and response binding verified'
    else:
        packet=load('input.json')
        contract=load('contract.json')
        if contract['input_sha256']!=files['input.json']['sha256']:
            raise ValueError('source input/contract hash diff')
        from .catalog import decode_table
        if (set(contract['L_t_ids'])!={r['id'] for r in decode_table(packet,'observed')}
                or set(contract['U_t_ids'])!={r['id'] for name in ('candidates','pending') for r in decode_table(packet,name)}):
            raise ValueError('source L/U authorization differs from recorded packet')
        if 'catalog.json' in files and contract['catalog_sha256']!=files['catalog.json']['sha256']:
            raise ValueError('source catalog/contract hash diff')
        private = case.get('source_kind') == 'private_snapshot'
        original_requests=private and 'request_00.json' in files
        if original_requests:
            original_request=load('request_00.json')
            if original_request['messages'][0]!={'role':'system','content':prompt(p)}:
                raise ValueError('source profile differs from private recorded Prompt')
            transport_config=original_request['config']
        else:
            if case['prompt']['text']!=prompt(p):
                raise ValueError('frozen Prompt artifact diff')
            transport_config=case['transport']
        messages=build_messages(p,packet)
        turnnames=sorted(name for name in files if name.startswith('turn_') and name.endswith('.json'))
        request_hashes=[]
        ids=None
        corrections=0
        for index,name in enumerate(turnnames):
            turn=load(name)
            if original_requests:
                original_request=load(f'request_{index:02d}.json')
                if original_request!={'messages':messages,'config':transport_config}:
                    raise ValueError('private original messages/config differ from source profile builder')
                if not p.cw_visible:
                    audit_no_cw(original_request['messages'])
            request_hash=stable_hash({'messages':messages,'config':transport_config})
            if turn['request_sha256']!=request_hash:
                raise ValueError('recorded request/messages/config hash diff')
            request_hashes.append(request_hash)
            try:
                ids=validate_response(p,turn['answer'],packet,pending_ids=frozen['pending_ids'])
            except (ValueError,TypeError,KeyError) as error:
                correction=correction_messages(p,turn['answer'],error)
                rejected_name=f'rejected_turn_{index:02d}.json'
                if rejected_name not in files:rejected_name=f'rejected_{index:02d}.json'
                rejected=load(rejected_name)
                if json.loads(correction[1]['content'])!=rejected:
                    raise ValueError('recorded correction transcript diff')
                corrections+=1
                messages+=correction
            else:
                if json.loads(turn['answer'])!=load('selection.json'):
                    raise ValueError('recorded accepted JSON selection diff')
                break
        if ids is None:
            raise ValueError('recording contains no accepted response')
        packet_sha=packet['packet_sha256']
        turns=len(request_hashes)
        message_evidence='EXACT_MESSAGES_CONFIG_REQUEST_HASH_MATCH'
    if tuple(frozen['batch_ids'])!=ids or len(ids)!=32 or len(set(ids))!=32:
        raise ValueError('ordered selected IDs differ from original freeze')
    return {'status':'EXACT_REPLAY_PASS','scope':'selection layer only; NOT a fit/reveal/metric trajectory replay',
            'source_kind':case.get('source_kind','tracked_git'),'profile':p.name,'seed':case['seed'],'round':case['round'],'budget':case['budget'],
            'packet_sha256':packet_sha,'input_file_sha256':files['packet.json' if p.delivery=='screened_128' else 'input.json']['sha256'],
            'checkpoint_sha256':None if p.delivery=='screened_128' else contract['checkpoint_sha256'],
            'batch_freeze_sha256':files['batch_freeze.json']['sha256'],'selected_ids':list(ids),'selected_ids_sha256':stable_hash(list(ids)),
            'id_diff_count':0,'messages':message_evidence,'request_hashes':request_hashes,
            'turns':turns,'corrections':corrections,'source_artifact_hashes_checked':checked_artifacts,
            'prediction_diff':'NOT_MEASURED; existing premeasurement predictions preserved as original bytes',
            'fit_reveal_metrics':'NOT_VERIFIED_IN_SELECTION_REPLAY',
            'new_training':False,'new_real_labels':0,'provider_calls':0}


def rebuild_recorded_packet(case):
    """Independently reconstruct a recorded first/second full-pool input.

    At round1 the frozen memory contains the complete sole prior batch, so the
    scientific_memory input is recoverable without guessing missing history.
    For later rounds a full external feedback history is required instead.
    """
    p=get_profile(case['profile'])
    if p.delivery!='all_candidates_one_request':
        return {'status':'NOT_APPLICABLE','scope':'screen has a different numerical/rounded card constructor'}
    if case['round'] not in (0,1):
        return {'status':'BLOCKED_MISSING_FULL_HISTORY','scope':'later memory does not contain every earlier raw feedback batch'}
    files=case['files'];packet=json.loads(files['input.json']['text'])
    raw=json.loads(files['catalog.json']['text']);contract=json.loads(files['contract.json']['text'])
    memory=packet['memory']
    history=[] if case['round']==0 else [{'seed':case['seed'],'method':p.trajectory_method,'round':0,
        'records':memory['recent_outcomes'],'hypotheses':memory['previous_hypotheses'],
        'feedback_interpretation':memory['previous_feedback_interpretation'],
        'batch_rationale':memory['previous_batch_rationale']}]
    rebuilt=build_full_packet(p,raw,seed=case['seed'],round_index=case['round'],history=history,
        candidate_ids=contract['U_t_ids'],observed_ids=contract['L_t_ids'],
        catalog_view='source_visible' if not p.cw_visible else 'raw_with_cw')
    compact=lambda value:json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if compact(rebuilt)!=compact(packet):
        raise ValueError('packet builder differs from original raw catalog/feedback or field order')
    return {'status':'EXACT_REPLAY_PASS','scope':'source-visible full catalog + recoverable complete round0/1 memory -> original packet/messages',
            'packet_sha256':rebuilt['packet_sha256'],'catalog_input_sha256':files['catalog.json']['sha256'],
            'candidate_count':rebuilt['candidate_count'],'observed_count':rebuilt['observed_count'],
            'id_order_and_numeric_values_exact':True,'message_serialization_exact':True,
            'history_origin':'none at round0; entire sole earlier batch recovered from original frozen round1 memory'}

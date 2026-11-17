#!/usr/bin/env python3
"""Audit frozen request messages, failure records and proposal conservation offline."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from retry_validators import load_validator,catalog_identity,parse_answer,id_diagnostics,error_category

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OLD=ROOT/'studies/active_learning/decision_mechanism_followup_20261117'
SNAP=ROOT/'studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz'
FREV=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions'
CW=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
METHODS=['historical_free_mixed_model','new_free_cw_sol61_xhigh','nocw']
FILES={}


def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def read(path,expected=None):
    path=Path(path);raw=path.read_bytes();assert raw==path.read_bytes(),('unstable source',rel(path))
    digest=sha(raw)
    if expected:assert digest==expected,('frozen source mismatch',rel(path))
    key=rel(path)
    if key in FILES:assert FILES[key]['sha256']==digest
    FILES[key]={'path':key,'sha256':digest,'bytes':len(raw)}
    return raw
def load(path,expected=None):return json.loads(read(path,expected))
def phase(budget):return '333-525' if budget<525 else '525-813' if budget<813 else '813-1005' if budget<1005 else '1005-1485'
def ordinal(path):return int(re.search(r'_(\d+)',Path(path).stem).group(1))


def safe_error(value):
    text=str(value)
    text=re.sub(r'https?://\S+','<endpoint-redacted>',text)
    text=re.sub(r'(?i)bearer\s+\S+','Bearer <redacted>',text)
    text=re.sub(r'\bsk-[A-Za-z0-9_-]{12,}','<key-redacted>',text)
    return text[:1200]


def safe_config(config):
    endpoint=str(config.get('base_url',''))
    host=urlsplit(endpoint).hostname or ''
    return {'config_sha256':stable(config),'backend':config.get('backend',''),
            'requested_model':config.get('model',''),'reasoning_effort':config.get('reasoning_effort',''),
            'endpoint_sha256':sha(endpoint.encode()),'endpoint_host_sha256':sha(host.encode()),
            'max_output_tokens':config.get('max_output_tokens'),
            'nominal_selection_attempts':config.get('selection_attempts'),
            'nominal_transport_attempts':config.get('transport_attempts'),
            'sampling_keys':json.dumps(sorted(config.get('sampling_parameters',{}))),
            'config_server_context_keys':json.dumps(sorted(k for k in config if k in {'previous_response_id','conversation','conversation_id'})),
            'credential_values_exported':False}


def compare_message_history(previous,current,answer,rejection):
    additions=current[len(previous):]
    assistant=any(m.get('role')=='assistant' and m.get('content')==answer for m in additions)
    error=False
    for message in additions:
        if message.get('role')=='user':
            try:error |= json.loads(message.get('content',''))==rejection
            except (ValueError,TypeError):pass
    return {'complete_previous_prefix':current[:len(previous)]==previous,
            'full_assistant_answer':assistant,'same_error_object':error,
            'appended_roles':[m.get('role') for m in additions]}


def source_inventory():
    paths=[ROOT/'src/qgeognn_al/active_learning_v2'/name for name in
           ['scientist_study.py','scientist_transport.py','scientist_full_delivery.py','nocw_study.py','nocw_inputs.py']]
    paths += [ROOT/'scripts/studies'/name for name in [
        'run_qgeognn_v2_row_llm_nocw_five.py','run_qgeognn_v2_row_llm_nocw_stream.py',
        'run_qgeognn_v2_row_llm_nocw_resume.py','run_qgeognn_v2_row_llm_nocw_unbounded_v2.py',
        'run_qgeognn_v2_row_llm_nocw_unbounded_v3.py','run_free_cw_sol61_five_20261101.py',
        'run_free_cw_parallel_20261101.py','run_free_cw_persistent_20261102.py',
        'run_nocw_cw_paired_to1485_20261112.py','repair_nocw_history_replay_20261115.py',
        'resume_nocw_upstream_reconciled_20261114.py','resume_nocw_auth_retry_20261115.py',
        'run_nocw_request_limit3_20261115.py','repair_nocw_sse_frames_20261116.py']]
    paths += [CW/'concurrency_v2/frozen_runner.py',CW/'persistent_20261102_v2/frozen_runner.py',
              CW/'frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py',
              CW/'frozen_source/scripts/studies/run_qgeognn_v2_row_llm_nocw_stream.py',
              CW/'frozen_source/src/qgeognn_al/active_learning_v2/nocw_study.py',
              CW/'frozen_source/src/qgeognn_al/active_learning_v2/scientist_transport.py']
    rows=[]
    names={'activate','activate_worker','select','run_selection','run_full_selector','run_selector',
           'request_with_retries','_call_with_retries','live_call','live_stream','responses_call','body_for'}
    for path in paths:
        if not path.exists():continue
        raw=read(path);tree=ast.parse(raw.decode())
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                text=ast.get_source_segment(raw.decode(),node) or ''
                rows.append({'path':rel(path),'sha256':sha(raw),'function':node.name,'line':node.lineno,
                             'source_end_line':node.end_lineno,
                             'mentions_previous_response_id':bool(re.search(r'\bprevious_response_id\b',text)),
                             'mentions_conversation_argument':bool(re.search(r'[\"\']conversation[\"\']\s*:',text)),
                             'has_unbounded_count_loop':'itertools.count(' in text or 'in count()' in text,
                             'mentions_messages_as_input':bool(re.search(r'input\s*=\s*messages|[\"\']input[\"\']\s*:\s*messages',text))})
    pd.DataFrame(rows).to_csv(OUT/'retry_controller_sources.csv',index=False)
    policies=[CW/'persistent_20261102_v2/authorization.json',CW/'parallel_execution_amendment.json',
              CW/'protocol.lock.json',CW/'source_snapshot.json']
    nr=ROOT/'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029'
    policies += [nr/name for name in ['protocol.lock.json','stream_transport_amendment.json',
                 'persistent_retry_amendment.json','unbounded_correction_amendment_20261031_v2.json',
                 'unbounded_correction_amendment_20261031_v3.json']]
    for path in policies:
        if path.exists():read(path)


def historical_candidates():
    configs={};prompts={}
    for path in sorted(FREV.glob('*/protocol.json')):
        value=load(path);config=value.get('transport',value.get('config',{}).get('transport'))
        if isinstance(config,dict) and 'model' in config:
            key=stable(config);configs.setdefault(key,{'config':config,'sources':[]})['sources'].append(rel(path))
    for path in sorted(FREV.glob('*/selector_prompt.txt')):
        text=read(path).decode();prompts.setdefault(sha(text.encode()),{'prompt':text,'sources':[]})['sources'].append(rel(path))
    return list(configs.values()),list(prompts.values())


def reconstruct_historical(packet,turn0,configs,prompts):
    matches=[]
    served=str(turn0.get('provenance',{}).get('served_model',''))
    for candidate in configs:
        config=candidate['config']
        if served and not (served==config.get('model') or served.startswith(str(config.get('model'))+'-')):continue
        for prompt in prompts:
            messages=[{'role':'system','content':prompt['prompt']},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
            if stable({'messages':messages,'config':config})==turn0.get('request_sha256'):
                matches.append((messages,config,candidate['sources'],prompt['sources']))
    if len(matches)==1:return matches[0]
    return None


def error_record(directory,index,hashes):
    for name in [f'rejected_{index:02d}.json',f'rejected_turn_{index:02d}.json']:
        path=directory/name
        if path.exists():
            value=load(path,hashes.get(name));text=value.get('error',value.get('message',value.get('error_type','')))
            return value,safe_error(text),rel(path)
    return None,'',''


def body_check(messages,config,method):
    if method=='historical_free_mixed_model':
        # Exact invocation is audited statically; legacy SDK defaults are not fabricated.
        return {'client_input_explicit_full_messages':True,'client_previous_response_id_supplied':False,
                'client_conversation_supplied':False,'body_profile':'legacy SDK responses.stream explicit input',
                'generation_cap_in_client':None,'body_sha256':''}
    path=CW/'frozen_source/scripts/studies/run_qgeognn_v2_row_llm_nocw_stream.py' if method=='new_free_cw_sol61_xhigh' else ROOT/'scripts/studies/run_qgeognn_v2_row_llm_nocw_stream.py'
    tree=ast.parse(read(path).decode());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='body_for')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    body=namespace['body_for'](messages,config)
    return {'client_input_explicit_full_messages':body['input']==messages,
            'client_previous_response_id_supplied':'previous_response_id' in body,
            'client_conversation_supplied':any(k in body for k in ['conversation','conversation_id']),
            'body_profile':'frozen body_for full messages, streamed HTTP responses',
            'generation_cap_in_client':body.get('max_output_tokens'),'body_sha256':stable(body),
            'body_keys':json.dumps(sorted(body))}


def main(args):
    source_inventory();configs,prompts=historical_candidates()
    snapshot=json.loads(lzma.decompress(read(args.snapshot)))
    states=[s for s in snapshot['states'] if s['method'] in METHODS]
    oldprefix=pd.read_csv(io.BytesIO(read(OLD/'llm_display_prefix_batches.csv'))).set_index(['method','seed','round'])
    validators={}
    for method in METHODS:
        validator,paths=load_validator(method);validators[method]=validator
        for path in paths:read(path)
    batches=[];proposals=[];contexts=[];messages_rows=[];attempt_rows=[];response_rows=[];config_rows={};binding=[]
    for number,state in enumerate(states):
        key={k:state[k] for k in ['method','seed','round','L_before','L_after']};key['phase']=phase(state['L_before'])
        method=state['method'];directory=(ROOT/state['source']).parent
        frozen=load(directory/'batch_freeze.json',state['source_sha256']);hashes=frozen['artifact_hashes']
        packet=load(directory/'input.json',hashes['input.json']);final=load(directory/'selection.json',hashes['selection.json'])
        column=packet['candidates']['columns'].index('id');pool=[v[column] for v in packet['candidates']['rows']]
        assert set(pool)==set(state['unlabeled_ids']) and not packet['pending']['rows']
        selected=[v['id'] for v in final['choices']];assert selected==frozen['batch_ids']==state['selected_ids']
        isprefix=selected==pool[:32]
        assert isprefix==bool(oldprefix.loc[(method,state['seed'],state['round'])].ordered_first32)
        catalog=catalog_identity(packet)
        names=sorted([name for name in hashes if re.fullmatch(r'turn_\d+\.json',Path(name).name)],key=ordinal)
        if not names:names=sorted([p.name for p in directory.glob('turn_*.json') if re.fullmatch(r'turn_\d+\.json',p.name)],key=ordinal)
        receipts={ordinal(name):load(directory/name,hashes.get(name)) for name in names}
        historical_reconstruction=reconstruct_historical(packet,receipts[min(receipts)],configs,prompts) if method==METHODS[0] else None
        reconstructed_messages=copy.deepcopy(historical_reconstruction[0]) if historical_reconstruction else None
        historical_config=historical_reconstruction[1] if historical_reconstruction else None
        previous_messages=None;previous_answer=None;previous_rejection=None
        accepted_indices=[];proposal_rows=[];request_rows=[];hash_failures=0
        for index,receipt in sorted(receipts.items()):
            answer=receipt.get('answer','');plan,parse_error=parse_answer(answer)
            diagnostics=id_diagnostics(plan,pool,state['labeled_ids'])
            rejection,recorded_error,rejection_source=error_record(directory,index,hashes)
            validation_error=''
            if parse_error:validation_error='answer must be one JSON object without Markdown'
            else:
                try:validators[method](plan,catalog,packet)
                except (ValueError,TypeError,KeyError) as error:validation_error=safe_error(error)
            accepted=plan==final
            if accepted:accepted_indices.append(index)
            legal=set(diagnostics.get('choice_ids') or [])&set(pool)
            usage=receipt.get('provenance',{}).get('usage') or {}
            if isinstance(usage,list):usage={}
            usage_details=usage.get('output_tokens_details',{}) or {}
            prow={**key,'turn':index,'receipt_source':rel(directory/f'turn_{index:02d}.json'),
                  'receipt_sha256':FILES[rel(directory/f'turn_{index:02d}.json')]['sha256'],
                  'answer_sha256':sha(str(answer).encode()),'answer_characters':len(str(answer)),
                  'answer_utf8_bytes':len(str(answer).encode()),'JSON_parse_error':parse_error or '',
                  **{k:v for k,v in diagnostics.items() if k!='choice_ids'},
                  'choice_ID_order':json.dumps(diagnostics.get('choice_ids')),
                  'unique_legal_retained_in_final_n':len(legal&set(selected)),
                  'legal_retention_in_final':len(legal&set(selected))/len(legal) if legal else np.nan,
                  'recorded_error':recorded_error,'recorded_error_category':error_category(recorded_error),
                  'rejection_source':rejection_source,'recomputed_validation_error':validation_error,
                  'recomputed_error_category':error_category(validation_error),
                  'recorded_recomputed_error_exact':recorded_error==validation_error if rejection is not None else None,
                  'accepted_exact_match':accepted,'ordered_first32':diagnostics.get('choice_ids')==pool[:32],
                  'request_sha256':receipt.get('request_sha256',''),'served_model':receipt.get('provenance',{}).get('served_model',''),
                  'input_tokens':usage.get('input_tokens'),'output_tokens':usage.get('output_tokens'),
                  'reasoning_tokens':usage_details.get('reasoning_tokens')}
            proposals.append(prow);proposal_rows.append(prow)
            request_path=directory/f'request_{index:02d}.json';request=None;config=None;message_list=None;grade='unavailable'
            if request_path.exists():
                request=load(request_path,hashes.get(request_path.name));message_list=request['messages'];config=request['config'];grade='stored_request'
            elif reconstructed_messages is not None:
                message_list=copy.deepcopy(reconstructed_messages);config=historical_config;request={'messages':message_list,'config':config};grade='reconstructed_hash_candidate'
            if request is not None:
                matches=stable(request)==receipt.get('request_sha256');hash_failures+=not matches
                if grade.startswith('reconstructed') and matches:grade='reconstructed_exact_hash'
                safe=safe_config(config);config_rows[safe['config_sha256']]=safe
                base_packet_equal=False
                try:base_packet_equal=json.loads(message_list[1]['content'])==packet
                except (ValueError,KeyError,IndexError,TypeError):pass
                prefix_retained=message_list[:len(previous_messages)]==previous_messages if previous_messages is not None else None
                additions=message_list[len(previous_messages):] if previous_messages is not None else []
                copied_assistant=any(m.get('role')=='assistant' and m.get('content')==previous_answer for m in additions) if previous_answer is not None else None
                error_matches=False if previous_rejection is not None else None
                for message in additions:
                    if message.get('role')=='user' and previous_rejection is not None:
                        try:error_matches |= json.loads(message.get('content',''))==previous_rejection
                        except (ValueError,TypeError):pass
                context={**key,'turn':index,'context_grade':grade,'request_file_present':request_path.exists(),
                         'request_hash_matches_receipt':matches,'messages_n':len(message_list),
                         'roles':json.dumps([m.get('role') for m in message_list]),
                         'messages_characters':sum(len(str(m.get('content',''))) for m in message_list),
                         'messages_utf8_bytes':sum(len(str(m.get('content','')).encode()) for m in message_list),
                         'original_packet_semantic_equal':base_packet_equal,
                         'original_system_preserved':message_list[0]==(previous_messages[0] if previous_messages else message_list[0]),
                         'previous_messages_complete_prefix':prefix_retained,'previous_assistant_full_answer_appended':copied_assistant,
                         'previous_error_object_appended':error_matches,'appended_messages_n':len(additions),
                         'extra_top_request_keys':json.dumps(sorted(set(request)-{'messages','config'})),
                         **safe,**body_check(message_list,config,method)}
                contexts.append(context);request_rows.append(context)
                for position,message in enumerate(message_list):
                    content=str(message.get('content',''));kind='system' if position==0 else 'original_packet' if position==1 else 'assistant_raw_answer' if message.get('role')=='assistant' else 'correction_or_user_hint'
                    messages_rows.append({**key,'turn':index,'message_position':position,'role':message.get('role'),
                                          'kind':kind,'content_characters':len(content),'content_utf8_bytes':len(content.encode()),
                                          'content_sha256':sha(content.encode()),'message_keys':json.dumps(sorted(message)),
                                          'raw_content_exported':False})
                if index==0 and method=='nocw' and state['seed'] in (157,6101,12203) and (state['seed'],state['round']) in [(157,18),(6101,11),(6101,20),(12203,31)]:
                    binding.append({**key,'request_hash_matches_receipt':matches,'packet_semantic_equal':base_packet_equal,
                                    'packet_self_binding_valid':stable({k:v for k,v in packet.items() if k!='packet_sha256'})==packet['packet_sha256'],
                                    'response_binding_is_wrong':isinstance(plan,dict) and plan.get('packet_sha256')!=packet['packet_sha256'],
                                    'request_sha256':receipt.get('request_sha256'),'request_source':rel(request_path)})
                previous_messages=message_list
            if reconstructed_messages is not None and rejection is not None:
                reconstructed_messages += [{'role':'assistant','content':answer},
                    {'role':'user','content':json.dumps(rejection,ensure_ascii=False)}]
            previous_answer=answer;previous_rejection=rejection
            # Inspect only metadata in durable server responses, never publish reasoning or packet content.
            for path in sorted(directory.glob(f'stream_{index:02d}_*.response.json')):
                saved=load(path);response=saved.get('response',{})
                response_rows.append({**key,'turn':index,'source':rel(path),
                    'request_hash_matches_turn':saved.get('request_sha256')==receipt.get('request_sha256'),
                    'server_previous_response_id_present':'previous_response_id' in response,
                    'server_previous_response_id_nonnull':response.get('previous_response_id') is not None,
                    'server_conversation_present':'conversation' in response,'server_conversation_nonnull':response.get('conversation') is not None,
                    'server_truncation':str(response.get('truncation','UNRECORDED')),
                    'server_status':response.get('status',''),'server_model':response.get('model',''),
                    'server_max_output_tokens':response.get('max_output_tokens')})
            for path in sorted(directory.glob(f'attempt_{index:02d}_*.json')):
                if path.name.endswith('.started.json'):continue
                attempt=load(path);start_path=path.with_name(path.stem+'.started.json');started=load(start_path) if start_path.exists() else {}
                attempt_rows.append({**key,'turn':index,'source':rel(path),'attempt':int(re.search(r'_(\d+)\.json$',path.name).group(1)),
                    'recorded_success':attempt.get('success'),'request_hash_matches_turn':attempt.get('request_sha256')==receipt.get('request_sha256'),
                    'error_type':attempt.get('error_type',''),'http_status':attempt.get('http_status'),
                    'error_text_sha256':stable(attempt.get('reason',attempt.get('message',''))),
                    'started_utc':started.get('started_utc',''),'ended_utc':attempt.get('ended_utc',''),
                    'persistent_amendment_sha256':started.get('persistent_amendment_sha256',attempt.get('persistent_amendment_sha256','')),
                    'retry_amendment_sha256':started.get('retry_amendment_sha256',attempt.get('retry_amendment_sha256','')),
                    'effective_transport_limit':started.get('effective_transport_limit','UNRECORDED'),
                    'recovered_local_completion':attempt.get('recovered_local_completion',False)})
            legacy_failures=directory/'retries'/f'turn_{index:02d}'
            for path in sorted(legacy_failures.glob('attempt_*.json')):
                attempt=load(path);attempt_rows.append({**key,'turn':index,'source':rel(path),'attempt':attempt.get('attempt'),
                    'recorded_success':False,'request_hash_matches_turn':attempt.get('request_sha256')==receipt.get('request_sha256'),
                    'error_type':attempt.get('error_type',''),'error_text_sha256':stable(attempt.get('message','')),
                    'http_status':attempt.get('http_status'),'legacy_failure_only_ledger':True})
        first=proposal_rows[0]
        first_failure=next((p for p in proposal_rows if p['recorded_error']),None)
        first_request=request_rows[0] if request_rows else {}
        last_request=request_rows[-1] if request_rows else {}
        batch={**key,'prefix_final':isprefix,'source':state['source'],'saved_turn_receipts_n':len(proposal_rows),
               'accepted_receipt_exact_matches':len(accepted_indices),'accepted_turn':accepted_indices[0] if len(accepted_indices)==1 else None,
               'first_receipt_source':first['receipt_source'],'first_answer_sha256':first['answer_sha256'],
               'first_choices_parseable':first['choices_parseable'],'first_choice_ID_order':first['choice_ID_order'],
               'first_choices_n':first.get('choices_n'),'first_unique_legal_n':first.get('unique_legal_n'),
               'first_unknown_occurrences':first.get('unknown_occurrences'),'first_duplicate_occurrences':first.get('duplicate_occurrences'),
               'first_legal_retained_n':first['unique_legal_retained_in_final_n'],'first_legal_retention':first['legal_retention_in_final'],
               'first_failure_category':first_failure['recorded_error_category'] if first_failure else 'none',
               'first_failure_error':first_failure['recorded_error'] if first_failure else '',
               'first_failure_source':first_failure['rejection_source'] if first_failure else '',
               'first_served_model':first['served_model'],'first_answer_characters':first['answer_characters'],
               'first_input_tokens':first['input_tokens'],'first_output_tokens':first['output_tokens'],
               'first_reasoning_tokens':first['reasoning_tokens'],
               'request_contexts_available_n':len(request_rows),'request_hash_failures_n':hash_failures,
               'all_original_packets_present':all(r['original_packet_semantic_equal'] for r in request_rows) if request_rows else None,
               'all_prior_histories_retained':all(r['previous_messages_complete_prefix'] is not False for r in request_rows) if request_rows else None,
               'all_previous_assistant_answers_appended':all(r['previous_assistant_full_answer_appended'] is not False for r in request_rows) if request_rows else None,
               'first_message_characters':first_request.get('messages_characters'),
               'accepted_message_characters':last_request.get('messages_characters'),
               'context_growth_characters':last_request.get('messages_characters',0)-first_request.get('messages_characters',0) if request_rows else None,
               'requested_model':first_request.get('requested_model',''),'reasoning_effort':first_request.get('reasoning_effort',''),
               'config_sha256':first_request.get('config_sha256',''),'endpoint_host_sha256':first_request.get('endpoint_host_sha256',''),
               'nominal_selection_attempts':first_request.get('nominal_selection_attempts'),
               'historical_reconstruction_grade':'exact_first_hash' if historical_reconstruction else 'not_reconstructed' if method==METHODS[0] else 'stored_requests'}
        batches.append(batch)
        if number%50==0:print('retry audit states',number,'/',len(states),flush=True)
    outputs={'retry_case_batches.csv':batches,'retry_proposals.csv':proposals,'retry_request_contexts.csv':contexts,
             'retry_message_inventory.csv':messages_rows,'retry_transport_attempts.csv':attempt_rows,
             'retry_server_response_metadata.csv':response_rows,'retry_transport_configs.csv':list(config_rows.values()),
             'packet_binding_context_checks.csv':binding}
    for name,rows in outputs.items():pd.DataFrame(rows).to_csv(OUT/name,index=False)
    frame=pd.DataFrame(batches)
    frame.groupby(['method','first_failure_category']).agg(batches=('round','size'),prefix_batches=('prefix_final','sum'),
        first_legal_mean=('first_unique_legal_n','mean'),mean_legal_retention=('first_legal_retention','mean')).reset_index().to_csv(OUT/'prefix_by_first_failure.csv',index=False)
    frame.groupby(['method','saved_turn_receipts_n']).agg(batches=('round','size'),prefix_batches=('prefix_final','sum')).reset_index().to_csv(OUT/'prefix_by_correction_count.csv',index=False)
    summary={'status':'COMPUTED_PENDING_REVIEW','batches':len(batches),'prefix_batches':int(frame.prefix_final.sum()),
             'proposals':len(proposals),'request_contexts':len(contexts),'network_attempt_records':len(attempt_rows),
             'server_response_records':len(response_rows),'new_API_calls':0,'new_LLM_calls':0,'new_training':0,
             'test_truth_reads':0,'unrevealed_truth_reads':0,'credential_files_opened':0,'endpoint_values_exported':False,
             'sources':list(FILES.values())}
    (OUT/'retry_audit_inputs.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,default=SNAP)
    main(parser.parse_args())

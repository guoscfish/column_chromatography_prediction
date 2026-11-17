#!/usr/bin/env python3
"""Reconcile explicit extra-authorized historical messages and summarize audits."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
import retry_context_audit as audit

OUT=audit.OUT
ROOT=audit.ROOT


def file_sha(path):return audit.sha(Path(path).read_bytes())


def reconcile():
    context=pd.read_csv(OUT/'retry_request_contexts.csv')
    batch=pd.read_csv(OUT/'retry_case_batches.csv')
    inventory=pd.read_csv(OUT/'retry_message_inventory.csv')
    before={name:file_sha(OUT/name) for name in ['retry_request_contexts.csv','retry_case_batches.csv','retry_message_inventory.csv']}
    configs,prompts=audit.historical_candidates()
    proof=[]
    failed=context[(context.method=='historical_free_mixed_model')&(~context.request_hash_matches_receipt)]
    for row in failed.itertuples():
        selected=batch[(batch.method==row.method)&(batch.seed==row.seed)&(batch['round']==row.round)].iloc[0]
        directory=(ROOT/selected.source).parent
        packet=audit.load(directory/'input.json');first=audit.load(directory/'turn_00.json')
        messages,config,config_sources,prompt_sources=audit.reconstruct_historical(packet,first,configs,prompts)
        for turn in range(row.turn):
            receipt=audit.load(directory/f'turn_{turn:02d}.json')
            rejection=audit.load(directory/f'rejected_turn_{turn:02d}.json')
            messages += [{'role':'assistant','content':receipt['answer']},{'role':'user','content':json.dumps(rejection,ensure_ascii=False)}]
        extra=audit.load(directory/'extra_correction_request.json')
        authorization=audit.load(directory/'authorized_extra_correction.json')
        messages.append({'role':'user','content':json.dumps(extra['feedback'],ensure_ascii=False)})
        receipt=audit.load(directory/f'turn_{row.turn:02d}.json')
        reconstructed_hash=audit.stable({'messages':messages,'config':config})
        assert reconstructed_hash==extra['request_sha256']==receipt['request_sha256']
        assert row.turn==3 and len(messages)==9
        mask=(context.method==row.method)&(context.seed==row.seed)&(context['round']==row.round)&(context.turn==row.turn)
        context.loc[mask,'context_grade']='reconstructed_exact_hash_extra_authorization'
        context.loc[mask,'request_hash_matches_receipt']=True
        context.loc[mask,'messages_n']=9
        context.loc[mask,'roles']=json.dumps([m['role'] for m in messages])
        context.loc[mask,'messages_characters']=sum(len(str(m['content'])) for m in messages)
        context.loc[mask,'messages_utf8_bytes']=sum(len(str(m['content']).encode()) for m in messages)
        context.loc[mask,'appended_messages_n']=3
        remove=(inventory.method==row.method)&(inventory.seed==row.seed)&(inventory['round']==row.round)&(inventory.turn==row.turn)
        inventory=inventory[~remove]
        fresh=[]
        for index,message in enumerate(messages):
            content=str(message['content'])
            fresh.append({key:getattr(row,key) for key in ['method','seed','round','L_before','L_after','phase','turn']}|
                         {'message_position':index,'role':message['role'],
                          'kind':'authorized_extra_user_feedback' if index==8 else 'system' if index==0 else 'original_packet' if index==1 else 'assistant_raw_answer' if message['role']=='assistant' else 'correction_or_user_hint',
                          'content_characters':len(content),'content_utf8_bytes':len(content.encode()),
                          'content_sha256':audit.sha(content.encode()),'message_keys':json.dumps(sorted(message)),
                          'raw_content_exported':False})
        inventory=pd.concat([inventory,pd.DataFrame(fresh)],ignore_index=True)
        bmask=(batch.method==row.method)&(batch.seed==row.seed)&(batch['round']==row.round)
        batch.loc[bmask,'request_hash_failures_n']=0
        batch.loc[bmask,'accepted_message_characters']=sum(len(str(m['content'])) for m in messages)
        batch.loc[bmask,'context_growth_characters']=float(batch.loc[bmask,'accepted_message_characters'].iloc[0])-float(batch.loc[bmask,'first_message_characters'].iloc[0])
        batch.loc[bmask,'historical_reconstruction_grade']='exact_hash_with_authorized_extra'
        proof.append({'method':row.method,'seed':row.seed,'round':row.round,'turn':row.turn,
                      'messages_n':9,'source':audit.rel(directory/'extra_correction_request.json'),
                      'authorization_source':audit.rel(directory/'authorized_extra_correction.json'),
                      'extra_feedback_sha256':audit.stable(extra['feedback']),'request_sha256':reconstructed_hash,
                      'exact_turn_and_saved_extra_hash_match':True,
                      'message_action':'append additional user feedback after ordinary assistant+error pair; no replacement',
                      'config_sources':json.dumps(config_sources),'prompt_sources':json.dumps(prompt_sources)})
    if proof:
        context.to_csv(OUT/'retry_request_contexts.csv',index=False)
        batch.to_csv(OUT/'retry_case_batches.csv',index=False)
        inventory.sort_values(['method','seed','round','turn','message_position']).to_csv(OUT/'retry_message_inventory.csv',index=False)
        pd.DataFrame(proof).to_csv(OUT/'retry_historical_extra_contexts.csv',index=False)
        (OUT/'retry_reconciliation_record.json').write_text(json.dumps({'status':'PASS_EXACT_HISTORICAL_EXTRA_RECONSTRUCTION',
            'corrected_derived_contexts':len(proof),'historical_sources_modified':False,'before_output_sha256':before,
            'after_output_sha256':{name:file_sha(OUT/name) for name in before}},indent=2)+'\n')
    return batch,context


def summarize(batch,context):
    proposal=pd.read_csv(OUT/'retry_proposals.csv')
    attempts=pd.read_csv(OUT/'retry_transport_attempts.csv',low_memory=False)
    responses=pd.read_csv(OUT/'retry_server_response_metadata.csv')
    responses['client_journal_whitelist_excludes_continuation_and_truncation']=True
    responses['server_context_and_truncation_observable']=False
    responses.to_csv(OUT/'retry_server_response_metadata.csv',index=False)
    # A summary partition is observational; it is never interpreted as a randomized comparison.
    batch['audit_group']=np.where(batch.first_failure_category=='none','no_recorded_failure',np.where(batch.prefix_final,'failure_then_prefix','failure_without_prefix'))
    conservation=batch.groupby(['method','audit_group']).agg(batches=('round','size'),
        original_unique_legal=('first_unique_legal_n','sum'),retained_legal=('first_legal_retained_n','sum'),
        median_per_batch_retention=('first_legal_retention','median'),mean_per_batch_retention=('first_legal_retention','mean'),
        mean_context_growth_characters=('context_growth_characters','mean')).reset_index()
    conservation['weighted_legal_retention']=conservation.retained_legal/conservation.original_unique_legal
    conservation.to_csv(OUT/'retry_plan_conservation.csv',index=False)
    exact=batch[batch.accepted_receipt_exact_matches==1]
    strict=exact.groupby(['method','audit_group']).agg(batches=('round','size'),
        original_unique_legal=('first_unique_legal_n','sum'),retained_legal=('first_legal_retained_n','sum')).reset_index()
    strict['weighted_legal_retention']=strict.retained_legal/strict.original_unique_legal
    strict.to_csv(OUT/'retry_plan_conservation_strict.csv',index=False)
    pd.DataFrame([{'scope':'all_recorded_Free32_batches','batches':len(batch),'prefix_batches':int(batch.prefix_final.sum()),
                   'prefix_fraction':batch.prefix_final.mean(),'manual_transcription_batches':int((batch.accepted_receipt_exact_matches!=1).sum())},
                  {'scope':'exact_raw_LLM_final_answer_only','batches':len(exact),'prefix_batches':int(exact.prefix_final.sum()),
                   'prefix_fraction':exact.prefix_final.mean(),'manual_transcription_batches':0}]).to_csv(OUT/'retry_cohort_scope.csv',index=False)
    error=batch.groupby(['method','first_failure_category']).agg(batches=('round','size'),prefix=('prefix_final','sum')).reset_index()
    error['prefix_fraction']=error.prefix/error.batches
    error.to_csv(OUT/'retry_first_error_frequencies.csv',index=False)
    first=proposal.sort_values('turn').drop_duplicates(['method','seed','round'])
    first=batch[['method','seed','round','prefix_final','first_failure_category']].merge(
        first[['method','seed','round','unknown_occurrences','observed_occurrences','duplicate_occurrences']],
        on=['method','seed','round'],validate='one_to_one')
    first=first[first.first_failure_category=='candidate_identity'].copy()
    for name in ['unknown','observed','duplicate']:
        first[name+'_ID_case']=first[name+'_occurrences']>0
    first.groupby(['method','prefix_final']).agg(cases=('round','size'),
        unknown_ID_cases=('unknown_ID_case','sum'),observed_ID_cases=('observed_ID_case','sum'),
        duplicate_ID_cases=('duplicate_ID_case','sum')).reset_index().to_csv(OUT/'retry_ID_error_subtypes.csv',index=False)
    rejected=batch[batch.first_failure_category!='none'].copy()
    rejected['correction_count_bin']=pd.cut(rejected.saved_turn_receipts_n,bins=[1,2,3,5,1000],labels=['2 answers','3 answers','4-5 answers','6+ answers'])
    rejected.groupby(['method','correction_count_bin'],observed=True).agg(batches=('round','size'),prefix=('prefix_final','sum'),
        mean_first_input_tokens=('first_input_tokens','mean'),mean_context_growth=('context_growth_characters','mean')).reset_index().to_csv(OUT/'retry_h3_correction_count.csv',index=False)
    # Length quartiles are defined within method and only among failed-first-answer batches.
    rejected['first_input_length_quartile']=rejected.groupby('method').first_message_characters.transform(lambda x:pd.qcut(x.rank(method='first'),4,labels=False)+1)
    rejected.groupby(['method','first_input_length_quartile']).agg(batches=('round','size'),prefix=('prefix_final','sum'),
        min_chars=('first_message_characters','min'),max_chars=('first_message_characters','max')).reset_index().to_csv(OUT/'retry_h3_initial_length.csv',index=False)
    requested=batch.groupby(['method','requested_model','reasoning_effort'],dropna=False).agg(batches=('round','size'),prefix=('prefix_final','sum')).reset_index()
    requested.to_csv(OUT/'retry_h5_model_config.csv',index=False)
    markers=attempts.copy()
    markers['persistent_marker']=markers.get('persistent_amendment_sha256',pd.Series(index=markers.index,dtype=str)).notna()
    policy=markers.groupby(['method','seed','round']).agg(attempt_records=('turn','size'),
        persistent_marked_attempts=('persistent_marker','sum'),max_recorded_transport_attempt_index=('attempt','max')).reset_index()
    policy=policy.merge(batch[['method','seed','round','saved_turn_receipts_n','prefix_final','nominal_selection_attempts']],on=['method','seed','round'],validate='one_to_one',how='right')
    policy['has_actual_persistent_marker']=policy.persistent_marked_attempts.fillna(0)>0
    policy['observed_answers_exceed_nominal_config']=policy.saved_turn_receipts_n>policy.nominal_selection_attempts
    policy.to_csv(OUT/'retry_policy_observability.csv',index=False)
    t=attempts.copy()
    t['ended_date_shanghai']=pd.to_datetime(t.get('ended_utc'),errors='coerce',utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
    t.groupby(['method','ended_date_shanghai'],dropna=False).agg(records=('turn','size'),
        recorded_success=('recorded_success','sum')).reset_index().to_csv(OUT/'retry_h5_attempt_dates.csv',index=False)
    attempts.groupby(['method','error_type','http_status'],dropna=False).size().reset_index(name='recorded_attempts').to_csv(OUT/'retry_transport_failure_summary.csv',index=False)
    base='src/qgeognn_al/active_learning_v2/scientist_full_delivery.py'
    frozen='studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/frozen_source/'+base
    validator_same=file_sha(ROOT/base)==file_sha(ROOT/frozen)
    assert validator_same
    contracts=[]
    for source in batch[batch.method=='historical_free_mixed_model'].source.unique():
        protocol=(ROOT/source).parent.parents[3]/'protocol.json'
        if not protocol.exists():continue
        value=audit.load(protocol)
        for name in ['scientist_study.py','scientist_transport.py','scientist_full_delivery.py']:
            key='src/qgeognn_al/active_learning_v2/'+name
            if key not in value.get('code_hashes',{}):continue
            raw=audit.read(ROOT/key)
            contracts.append({'cohort':'Historical','protocol_source':audit.rel(protocol),'code_source':key,
                              'recorded_sha256':value['code_hashes'][key],'actual_sha256':audit.sha(raw),
                              'source_contract_matches':value['code_hashes'][key]==audit.sha(raw)})
    contracts=pd.DataFrame(contracts).drop_duplicates()
    assert contracts.source_contract_matches.all()
    historical_contract_n=len(contracts)
    extra_contracts=[]
    snapshot_path=audit.CW/'source_snapshot.json';source_snapshot=audit.load(snapshot_path)
    for key in ['src/qgeognn_al/active_learning_v2/nocw_study.py',
                'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py',
                'scripts/studies/run_qgeognn_v2_row_llm_nocw_stream.py']:
        path=audit.CW/'frozen_source'/key;raw=audit.read(path)
        extra_contracts.append({'cohort':'NewCW_frozen_source','protocol_source':audit.rel(snapshot_path),
            'code_source':audit.rel(path),'recorded_sha256':source_snapshot['frozen_hashes'][key],
            'actual_sha256':audit.sha(raw),'source_contract_matches':source_snapshot['frozen_hashes'][key]==audit.sha(raw)})
    nocw_protocol=ROOT/'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029/protocol.lock.json'
    nocw=audit.load(nocw_protocol)
    for key in ['src/qgeognn_al/active_learning_v2/nocw_study.py',
                'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py',
                'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py']:
        raw=audit.read(ROOT/key)
        extra_contracts.append({'cohort':'NoCW_initial_protocol','protocol_source':audit.rel(nocw_protocol),
            'code_source':key,'recorded_sha256':nocw['code_hashes'][key],'actual_sha256':audit.sha(raw),
            'source_contract_matches':nocw['code_hashes'][key]==audit.sha(raw)})
    contracts=pd.concat([contracts,pd.DataFrame(extra_contracts)],ignore_index=True)
    assert contracts.source_contract_matches.all()
    contracts.to_csv(OUT/'retry_source_contract_checks.csv',index=False)
    exceptions=batch[batch.accepted_receipt_exact_matches!=1]
    assert len(exceptions)==1
    exception=exceptions.iloc[0]
    directory=(ROOT/exception.source).parent
    manual=audit.load(directory/'authorized_id_transcription_correction.json')
    raw=audit.read(directory/manual['source'],manual['source_sha256'])
    original=json.loads(json.loads(raw)['answer'])
    repaired=json.loads(json.dumps(original))
    changed=0
    for choice in repaired['choices']:
        if choice['id']==manual['old_id']:
            choice['id']=manual['new_id'];changed+=1
    accepted=audit.load(directory/'selection.json')
    assert changed==1 and repaired==accepted
    manual_proof={'status':'PASS_EXPLICIT_AUTHORIZED_ONE_ID_TRANSCRIPTION',
                  'method':exception.method,'seed':int(exception.seed),'round':int(exception['round']),
                  'prefix_case':bool(exception.prefix_final),'source':audit.rel(directory/'authorized_id_transcription_correction.json'),
                  'original_receipt_sha256':manual['source_sha256'],'changed_choice_IDs':changed,
                  'all_other_fields_exactly_preserved':True,'original_legal_IDs_retained':int(exception.first_legal_retained_n),
                  'new_API_calls_for_recorded_correction':manual['new_llm_calls'],
                  'scope':'Included in 540 recorded Free32 batches; excluded from 539 exact raw-final-answer cohort.'}
    (OUT/'retry_manual_selection_exception.json').write_text(json.dumps(manual_proof,ensure_ascii=False,indent=2)+'\n')
    failures=proposal[proposal.rejection_source.notna()]
    status={'status':'PASS_WITH_EXPLICIT_HISTORICAL_MANUAL_SELECTION_EXCEPTION',
        'batches':len(batch),'prefix_batches':int(batch.prefix_final.sum()),'logical_proposals':len(proposal),
        'request_contexts':len(context),'stored_request_contexts':int(context.request_file_present.sum()),
        'historically_reconstructed_contexts':int((~context.request_file_present).sum()),
        'all_request_hashes_match':bool(context.request_hash_matches_receipt.all()),
        'all_original_packets_semantically_present':bool(context.original_packet_semantic_equal.all()),
        'continued_contexts':int(context.previous_messages_complete_prefix.notna().sum()),
        'all_continued_histories_preserved':bool(context.previous_messages_complete_prefix.dropna().all()),
        'all_prior_full_assistant_answers_appended':bool(context.previous_assistant_full_answer_appended.dropna().all()),
        'all_recorded_prior_error_objects_appended':bool(context.previous_error_object_appended.dropna().all()),
        'explicit_previous_response_id_requests':int(context.client_previous_response_id_supplied.sum()),
        'explicit_conversation_requests':int(context.client_conversation_supplied.sum()),
        'recorded_failures_recomputed':len(failures),'all_recorded_failure_messages_exact':bool(failures.recorded_recomputed_error_exact.all()),
        'accepted_exact_receipt_matches':int((batch.accepted_receipt_exact_matches==1).sum()),
        'strict_raw_final_answer_cohort':len(exact),'strict_prefix_fraction':float(exact.prefix_final.mean()),
        'manual_exception_is_prefix':bool(batch[batch.accepted_receipt_exact_matches!=1].prefix_final.any()),
        'known_manual_selection_exception':'Historical seed157 round5: existing authorized ID transcription; no uniquely matching final raw LLM answer.',
        'final_validator_source_current_equals_new_frozen':validator_same,
        'network_attempt_records':len(attempts),'network_attempt_hash_mismatches':int((~attempts.request_hash_matches_turn).sum()),
        'server_response_records':len(responses),'server_nonnull_previous_response_id':int(responses.server_previous_response_id_nonnull.sum()),
        'server_nonnull_conversation':int(responses.server_conversation_nonnull.sum()),
        'server_truncation_observability':responses.server_truncation.value_counts().to_dict(),
        'server_context_fields_observable':False,
        'server_metadata_limit':'Saved responses are client whitelist(id/model/status/usage/output); zero continuation counts mean unrecorded, not proof of absence or no server truncation.',
        'distinct_endpoint_host_hashes':int(context.endpoint_host_sha256.nunique()),
        'prefix_original_unique_legal':int(batch[batch.prefix_final].first_unique_legal_n.sum()),
        'prefix_historical_retained_legal':int(batch[batch.prefix_final].first_legal_retained_n.sum()),
        'failure_nonprefix_original_unique_legal':int(batch[(~batch.prefix_final)&(batch.first_failure_category!='none')].first_unique_legal_n.sum()),
        'failure_nonprefix_retained_legal':int(batch[(~batch.prefix_final)&(batch.first_failure_category!='none')].first_legal_retained_n.sum()),
        'strict_failure_nonprefix_original_unique_legal':int(exact[(~exact.prefix_final)&(exact.first_failure_category!='none')].first_unique_legal_n.sum()),
        'strict_failure_nonprefix_retained_legal':int(exact[(~exact.prefix_final)&(exact.first_failure_category!='none')].first_legal_retained_n.sum()),
        'historical_source_contract_checks':historical_contract_n,'source_contract_checks_total':len(contracts),
        'all_source_contracts_match':bool(contracts.source_contract_matches.all()),
        'new_API_calls':0,'credential_files_opened':0,'endpoint_values_exported':False,
        'limits':'Client history/body evidence does not establish provider internal context, weights, model identity beyond returned fields, causal prefix mechanisms, or billed request counts.'}
    assert status['batches']==540 and status['prefix_batches']==127 and status['logical_proposals']==1120
    assert status['all_request_hashes_match'] and status['all_original_packets_semantically_present']
    assert status['all_continued_histories_preserved'] and status['all_prior_full_assistant_answers_appended']
    assert status['all_recorded_prior_error_objects_appended'] and status['all_recorded_failure_messages_exact']
    assert status['network_attempt_hash_mismatches']==0
    (OUT/'retry_audit_verification.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    manifest=json.loads((OUT/'retry_audit_inputs.json').read_text());sources={x['path']:x for x in manifest['sources']}
    sources.update(audit.FILES);manifest['sources']=list(sources.values());manifest['status']=status['status']
    (OUT/'retry_audit_inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False))


def main():
    batch,context=reconcile();summarize(batch,context)


if __name__=='__main__':main()

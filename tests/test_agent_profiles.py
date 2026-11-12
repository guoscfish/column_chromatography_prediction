"""Frozen scientific profiles, source-visible boundaries and actual recorded plans."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from src.qgeognn_al.infra.contracts import content_hash
from src.qgeognn_al.al.agents.profiles import get_profile, profile_names
from src.qgeognn_al.al.agents.prompts import prompt, CW_DESCRIPTION, COMMON_PROMPT, DIRECT_PROMPT
from src.qgeognn_al.al.agents.validation import (audit_no_cw, validate_response,
    validate_packet, validate_screen_response, packet_candidate_ids)
from src.qgeognn_al.al.agents.replay_packets import (build_full_packet, build_messages,
    correction_messages, replay_recorded_case)
from src.qgeognn_al.al.agents.recorded import replay_fixture
from src.qgeognn_al.al.agents.screen import rng, shortlist

FIXTURE=Path(__file__).parent/'fixtures/llm_recorded_replay.json.gz'

@pytest.fixture(scope='module')
def recorded():
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))


def raw_catalog(count=180, pending=0):
    conditions={'PE/EA':'10/1','Density g/ml':1.0,'V/ul':10.0,'loading solvent':'EA',
                'Volume of loading solvent/ul':10.0}
    rows=[]
    for i in range(count):
        row={'id':f'U{i:04d}','smiles':'CC','conditions':conditions.copy(),
             'pred_V1_ml':2.0,'pred_V2_ml':5.0,'cw_distance_percentile':i/count}
        row.update({f'pred_{t}_{q}_ml':float(j+(t=='V2')*3) for t in ('V1','V2')
                    for j,q in enumerate(('q10','q50','q90'),1)})
        rows.append(row)
    observed={'id':'L0000','smiles':'CC','conditions':conditions.copy(),
              'V1_ml':2.0,'V2_ml':5.0,'record_kind':'initial_L333'}
    return {'candidates':rows[pending:],'pending':rows[:pending],'observed':[observed],'salt':'unused'}


def build(name='free32_cwaware',count=180):
    p=get_profile(name);raw=raw_catalog(count,p.pending_count)
    packet=build_full_packet(p,raw,seed=157,round_index=0,history=[],
        candidate_ids=[r['id'] for r in raw['candidates']+raw['pending']],observed_ids=['L0000'])
    return raw,packet


def plan(packet):
    ids=packet_candidate_ids('free32_cwaware',packet)
    return {'type':'selection','packet_sha256':packet['packet_sha256'],'batch_strategy':'learn',
            'choices':[{'id':i,'reason':'test a response relationship','hypothesis_id':None} for i in ids[:32]],
            'hypotheses':[],'feedback_interpretation':'','batch_rationale':'learn the joint response'}


def test_explicit_profiles_and_prompt_source_identity():
    assert len(profile_names())==7
    assert get_profile('free32_cwaware').selection_count==32
    assert get_profile('cw16_llm16_screen_v1').delivery=='screened_128'
    assert get_profile('cw16_llm16_scientist_v2').delivery=='all_candidates_one_request'
    assert get_profile('free16_cwgradient_lcmd16').trajectory_method=='cw16_llm16_scientist_v2'
    assert content_hash(DIRECT_PROMPT)=='3fea9e22b2b325117f68e22b06ab0030767de6edddf4017df27128a4308840c4'
    assert prompt('free32_nocw')==prompt('free32_cwaware').replace(CW_DESCRIPTION,'')
    assert prompt('free32_nocw_paired_v1')==COMMON_PROMPT.replace(CW_DESCRIPTION,'')
    assert prompt('free32_nocw')!=prompt('free32_nocw_paired_v1')
    with pytest.raises(ValueError):get_profile('free')


@pytest.mark.parametrize('index',range(8))
def test_actual_recorded_rounds_exact_selection_and_correction(recorded,index):
    result=replay_recorded_case(recorded['cases'][index])
    assert result['status']=='EXACT_REPLAY_PASS' and result['id_diff_count']==0
    assert result['provider_calls']==result['new_real_labels']==0
    assert result['new_training'] is False
    if result['profile']=='cw16_llm16_scientist_v2':
        assert result['messages']=='EXACT_MESSAGES_CONFIG_REQUEST_HASH_MATCH'
        assert result['corrections']==result['round']


@pytest.mark.parametrize('round_index',[0,1])
def test_full_catalog_builder_matches_real_packet_and_outgoing_messages(recorded,round_index):
    case=next(c for c in recorded['cases'] if c['profile']=='cw16_llm16_scientist_v2' and c['round']==round_index)
    raw=json.loads(case['files']['catalog.json']['text']);expected=json.loads(case['files']['input.json']['text'])
    memory=expected['memory']
    # At round1 the frozen memory contains the entire sole earlier feedback
    # batch; this is not a reconstruction of later missing runtime histories.
    history=[] if round_index==0 else [{'seed':157,'method':expected['method'],'round':0,
        'records':memory['recent_outcomes'],'hypotheses':memory['previous_hypotheses'],
        'feedback_interpretation':memory['previous_feedback_interpretation'],
        'batch_rationale':memory['previous_batch_rationale']}]
    actual=build_full_packet(case['profile'],raw,seed=157,round_index=round_index,history=history,
        candidate_ids=[r['id'] for r in raw['candidates']+raw['pending']],observed_ids=[r['id'] for r in raw['observed']])
    assert actual==expected
    assert json.dumps(actual,ensure_ascii=False,separators=(',',':'))==json.dumps(expected,ensure_ascii=False,separators=(',',':'))
    assert build_messages(case['profile'],actual)==build_messages(case['profile'],expected)


def test_free_complete_pool_authorization_and_unknown_fields_fail_closed():
    raw,packet=build()
    assert len(packet_candidate_ids('free32_cwaware',packet))==180
    chosen=validate_response('free32_cwaware',plan(packet),packet)
    assert len(chosen)==32 and len(set(chosen))==32
    with pytest.raises(ValueError,match='complete authorized'):
        build_full_packet('free32_cwaware',raw,seed=157,round_index=0,history=[],
            candidate_ids=[r['id'] for r in raw['candidates']][:-1],observed_ids=['L0000'])
    raw['candidates'][0]['true_V1_ml']=123
    with pytest.raises(ValueError,match='unauthorized'):
        build_full_packet('free32_cwaware',raw,seed=157,round_index=0,history=[],
            candidate_ids=[r['id'] for r in raw['candidates']],observed_ids=['L0000'])


@pytest.mark.parametrize('leak',[{'cw_distance_percentile':.2},{'reason':'LCMD coverage'},
    {'error':'中心宽度梯度距离'},{'history':[{'memo':'CW distance'}]}])
def test_nocw_scans_keys_memory_errors_and_corrections(leak):
    _,packet=build('free32_nocw')
    audit_no_cw(build_messages('free32_nocw',packet))
    with pytest.raises(ValueError,match='NoCW visible'):audit_no_cw(leak)
    with pytest.raises(ValueError,match='NoCW visible'):
        correction_messages('free32_nocw',json.dumps(leak),'invalid plan')


def test_hybrid_pending_order_is_explicit_and_not_encoded_table_order(recorded):
    case=next(c for c in recorded['cases'] if c['profile']=='cw16_llm16_scientist_v2')
    packet=json.loads(case['files']['input.json']['text']);frozen=json.loads(case['files']['batch_freeze.json']['text'])
    raw=case['files']['selection.json']['text']
    with pytest.raises(ValueError,match='pending order'):
        validate_response(case['profile'],raw,packet)
    ids=validate_response(case['profile'],raw,packet,pending_ids=frozen['pending_ids'])
    assert ids==tuple(frozen['batch_ids'])


@pytest.mark.parametrize('key',['hidden_truth','validation_labels','correction_transcript'])
def test_screen_unknown_top_level_fields_cannot_cross_provider_boundary(recorded,key):
    case=recorded['cases'][0];packet=json.loads(case['files']['packet.json']['text'])
    packet['packet_sha256']=case['files']['packet.json']['sha256'];packet[key]={'U1':99}
    with pytest.raises(ValueError,match='unauthorized screen packet'):
        build_messages(case['profile'],packet)


def test_screen_one_attempt_fallback_retains_legal_order():
    ids=[f'U{i}' for i in range(128)];fallback=ids[::-1];binding='a'*64
    response={'packet_sha256':binding,'choices':[{'id':'U8','reason':'observe new response'},
        {'id':'U8','reason':'duplicate'},{'id':'foreign','reason':'not allowed'}]}
    result=validate_screen_response(response,ids,fallback,binding)
    assert result['choices'][0]['id']=='U8'
    assert [r['id'] for r in result['choices'][1:]]==fallback[:15]
    assert result['fallback_count']==15
    with pytest.raises(ValueError,match='one-attempt'):
        correction_messages('cw16_llm16_screen_v1','bad','invalid')


def test_screen_original_independent_rngs_and_proposal_composition():
    seed,rnd=157,1
    assert np.array_equal(rng(seed,rnd,3).permutation(128),np.random.default_rng(np.random.SeedSequence([seed,rnd,3,1616])).permutation(128))
    features=np.random.default_rng(9).normal(size=(152,4))
    result=shortlist(features,features*.5,8,seed,rnd)
    assert len(result['pending_positions'])==16
    assert len(result['candidate_positions'])==128==len(set(result['candidate_positions']))
    assert not set(result['pending_positions'])&set(result['candidate_positions'])
    assert {k:len(v) for k,v in result['proposals'].items()}=={'cw':40,'ivr':40,'maxdet':40,'random':8}
    assert set(result['fallback_positions'])==set(result['candidate_positions'])
    assert result==shortlist(features,features*.5,8,seed,rnd)


def test_replay_entry_and_missing_real_profiles_are_explicitly_blocked():
    assert replay_fixture(FIXTURE)['cases_replayed']==8
    blocked=replay_fixture(FIXTURE,'free32_nocw')
    assert blocked['status']=='BLOCKED_MISSING_FIXTURE' and blocked['cases_replayed']==0
    alias=replay_fixture(FIXTURE,'free16_cwgradient_lcmd16')
    assert alias['cases_replayed']==2 and alias['profiles']==['free16_cwgradient_lcmd16']


def _private_format_from_public_case(case,tmp_path):
    import hashlib
    result=copy.deepcopy(case)
    result['source_kind']='private_snapshot'
    private={}
    for name,record in case['files'].items():
        path=tmp_path/name;path.write_text(record['text']);path.chmod(0o444)
        private[name]={'source_kind':'private_snapshot','resource':{
            'path':str(path),'root':str(tmp_path),'sha256':record['sha256']}}
    freeze=case['files']['batch_freeze.json']
    result['files']=private
    result['tracked_freeze_anchor']={'source_commit':get_profile(case['profile']).source_commit,
        'path':freeze['path'],'sha256':freeze['sha256'],'git_blob':freeze['git_blob']}
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':2,'source_kind':'private_snapshot',
        'source_commit':get_profile(case['profile']).source_commit,'cases':[result]}))
    manifest.chmod(0o444)
    return result,manifest


def test_private_manifest_format_replays_without_fabricating_git_identity(recorded,tmp_path):
    from src.qgeognn_al.al.agents.recorded import _private_case
    source=next(c for c in recorded['cases'] if c['profile']=='cw16_llm16_scientist_v2')
    case,path=_private_format_from_public_case(source,tmp_path)
    hydrated=_private_case(case,tmp_path)
    assert all('git_blob' not in f and f['source_kind']=='private_snapshot' for f in hydrated['files'].values())
    path.chmod(0o644)
    with pytest.raises(ValueError,match='independent read-only'):
        replay_fixture(path)
    path.chmod(0o444)
    import os
    alias=tmp_path/'manifest_alias.json';os.link(path,alias)
    with pytest.raises(ValueError,match='independent read-only'):
        replay_fixture(path)
    alias.unlink()
    result=replay_fixture(path)
    assert result['status']=='EXACT_REPLAY_PASS' and result['source_kind']=='private_snapshot'
    assert 'turn_00.json' in result['results'][0]['freeze_anchor_coverage']['direct_artifact_files']
    assert result['cases_replayed']==1 and result['fit_reveal_metrics']=='NOT_VERIFIED_IN_SELECTION_REPLAY'


def test_private_manifest_resource_tamper_is_rejected_before_replay(recorded,tmp_path):
    source=next(c for c in recorded['cases'] if c['profile']=='cw16_llm16_scientist_v2')
    _,manifest=_private_format_from_public_case(source,tmp_path)
    file=tmp_path/'turn_00.json';file.chmod(0o644);file.write_text('{}');file.chmod(0o444)
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        replay_fixture(manifest)


def test_private_manifest_cannot_open_undeclared_or_outside_root_files(recorded,tmp_path):
    from src.qgeognn_al.al.agents.recorded import _private_case
    source=next(c for c in recorded['cases'] if c['profile']=='cw16_llm16_scientist_v2')
    case,_=_private_format_from_public_case(source,tmp_path)
    escaped=copy.deepcopy(case);escaped['files']['input.json']['resource']['root']=str(tmp_path.parent)
    with pytest.raises(ValueError,match='own manifest root'):_private_case(escaped,tmp_path)
    unauthorized=copy.deepcopy(case);unauthorized['files']={'credentials.json':case['files']['input.json']}
    with pytest.raises(ValueError,match='literal independent'):_private_case(unauthorized,tmp_path)


def test_private_manifest_cannot_redeclare_hash_to_bypass_frozen_artifact(recorded,tmp_path):
    import hashlib
    source=next(c for c in recorded['cases'] if c['profile']=='cw16_llm16_scientist_v2')
    _,manifest=_private_format_from_public_case(source,tmp_path)
    file=tmp_path/'turn_00.json'
    changed=json.loads(file.read_text());changed['provenance']['owned_test_mutation']=True
    file.chmod(0o644);file.write_text(json.dumps(changed));file.chmod(0o444)
    # Updating a mutable external manifest to match tampered bytes must not
    # bypass the separately pinned original freeze's artifact digest.
    declaration=json.loads(manifest.read_text())
    declaration['cases'][0]['files']['turn_00.json']['resource']['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.chmod(0o644);manifest.write_text(json.dumps(declaration));manifest.chmod(0o444)
    with pytest.raises(ValueError,match='tracked freeze artifact hash'):
        replay_fixture(manifest)

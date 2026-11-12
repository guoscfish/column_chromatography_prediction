"""Run-owned Row-to-agent packets: complete pools, feedback and truth boundaries."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.al.agents.catalog import decode_table
from src.qgeognn_al.al.agents.profiles import get_profile
from src.qgeognn_al.al.agents.screen import shortlist
from src.qgeognn_al.al.agents.validation import audit_no_cw, packet_candidate_ids
from src.qgeognn_al.al.acquisitions.lcmd import _squared_distances
from src.qgeognn_al.al.workflow.agent_data import RowAgentPacketBuilder
from src.qgeognn_al.al.workflow.state import read_envelope, write_envelope
from src.qgeognn_al.infra.contracts import content_hash

FULL_PROFILES=('free32_cwaware','free32_nocw','cw16_llm16_scientist_v2',
               'free16_cwgradient_lcmd16','free32_cwaware_paired_v1','free32_nocw_paired_v1')


@pytest.fixture
def row_inputs():
    ids=tuple(f'L{i}' for i in range(333))+tuple(f'U{i}' for i in range(192))
    frame=pd.DataFrame({'sample_id':ids,'canonical_smiles':['CCO' if i%3 else 'CCN' for i in range(len(ids))],
        'PE/EA':['10/1']*len(ids),'Density g/ml':[.7]*len(ids),'V/ul':[12.]*len(ids),
        'loading solvent':['PE']*len(ids),'Volume of loading solvent/ul':[20.]*len(ids)})
    return SimpleNamespace(ids=ids,positions={i:n for n,i in enumerate(ids)},frame=frame,
        raw=np.random.default_rng(157).normal(size=(len(ids),512)),
        cw=np.random.default_rng(158).normal(size=(len(ids),512)),
        predictions=np.random.default_rng(159).normal(size=(len(ids),6))+np.arange(6),scope='c'*64)


def fit_receipt(directory, public, scope):
    write_envelope(directory/f'fit_{public["round"]:04d}.json',{'scope':scope,
        'labeled_ids_hash':content_hash(public['labeled_ids']), 'fit':{'seed':157,
        'checkpoint_hash':public['checkpoint_hash'],'labeled_ids_hash':content_hash(public['labeled_ids'])}})


def builder_fixture(tmp_path, row, profile):
    data=SimpleNamespace(feature_frame=row.frame.copy(deep=True),l0_ids=row.ids[:333],u0_ids=row.ids[333:],
        source_sha256='1'*64,partition_hash='2'*64,graph_cache_sha256='3'*64,seed=157)
    backend=SimpleNamespace(contract_hash='5'*64,_model=lambda fit:object())
    builder=RowAgentPacketBuilder(profile,data,backend,tmp_path)
    # Owned label-free feature/prediction arrays stand in for inference. These
    # cases qualify the packet boundary, never real model/gradient equivalence.
    builder._predict_current=lambda model,candidates:np.array([row.predictions[row.positions[i]] for i in candidates])
    def banks(model, public):
        order=[row.positions[i] for i in public['labeled_ids']+public['candidate_ids']]
        return row.cw[order],row.raw[order]
    builder._feature_banks=banks
    public={'seed':157,'round':0,'labeled_ids':list(row.ids[:333]),'candidate_ids':list(row.ids[333:]),
        'observed_labels':{i:[1.,2.] for i in row.ids[:333]},'checkpoint_hash':'4'*64}
    fit_receipt(tmp_path,public,row.scope)
    return builder,public


def measured_next_round(builder, public, directory, scope):
    supplied=builder(public);packet=supplied['packet'];pending=supplied['pending_ids']
    candidates=packet_candidate_ids(builder.profile,packet)
    choices=[{'id':i,'reason':'owned fixture','hypothesis_id':None} for i in candidates[:builder.profile.selection_count]]
    plan={'type':'selection','packet_sha256':packet['packet_sha256'],'batch_strategy':'owned fixture',
        'batch_rationale':'owned fixture','feedback_interpretation':'','hypotheses':[],'choices':choices}
    chosen=list(pending)+[c['id'] for c in choices]
    predictions=[{'candidate_id':i,'pred_V1_ml':supplied['premeasurement_predictions'][i][0],
        'pred_V2_ml':supplied['premeasurement_predictions'][i][1]} for i in chosen]
    frozen=write_envelope(directory/'round_0000.freeze.json',{'scope':scope,'outer_seed':157,'round':0,
        'acquisition':builder.profile.name,'selected_ids':chosen,'budget':32,
        'labeled_ids_hash':content_hash(public['labeled_ids']),'candidate_ids_hash':content_hash(public['candidate_ids']),
        'audit':{'agent_record':{'profile':builder.profile.name,'pending_ids':pending,
            'accepted_response':plan,'premeasurement_predictions':predictions}}})
    labels={i:[10.+j,20.+j] for j,i in enumerate(chosen)}
    write_envelope(directory/'round_0000.labels.json',{'scope':scope,'freeze_hash':frozen['sha256'],'labels':labels})
    next_public={**public,'round':1,'labeled_ids':public['labeled_ids']+chosen,
        'candidate_ids':[i for i in public['candidate_ids'] if i not in set(chosen)],
        'observed_labels':{**public['observed_labels'],**labels},'checkpoint_hash':'6'*64}
    fit_receipt(directory,next_public,scope)
    return supplied,next_public,chosen,predictions,labels


@pytest.mark.parametrize('profile',FULL_PROFILES)
def test_full_profile_complete_pool_and_only_frozen_measured_feedback(tmp_path,row_inputs,profile):
    b,public=builder_fixture(tmp_path,row_inputs,profile)
    first,next_public,chosen,predictions,labels=measured_next_round(b,public,tmp_path,row_inputs.scope)
    packet=first['packet'];candidate_ids=packet_candidate_ids(profile,packet)
    assert len(candidate_ids)==192-b.profile.pending_count
    assert set(candidate_ids+tuple(first['pending_ids']))==set(public['candidate_ids'])
    assert set(first['premeasurement_predictions'])==set(public['candidate_ids'])
    assert len(decode_table(packet,'observed'))==333
    second=b(next_public)['packet'];observed=decode_table(second,'observed')
    acquired=[r for r in observed if r['record_kind']=='acquired_with_premeasurement_prediction']
    assert len(acquired)==32 and len(observed)==365
    assert [r['candidate_id'] for r in second['memory']['recent_outcomes']]==chosen
    assert second['memory']['previous_selected_ids']==chosen
    assert len(second['memory']['high_error_measured_examples'])==8
    for card in acquired:
        pred=next(p for p in predictions if p['candidate_id']==card['id'])
        assert card['V1_ml']==labels[card['id']][0]
        assert card['premeasurement_prediction_V1_ml']==pred['pred_V1_ml']
        assert card['premeasurement_error_V2_ml']==pred['pred_V2_ml']-labels[card['id']][1]
    if not b.profile.cw_visible:
        assert audit_no_cw(packet)['passed'] and audit_no_cw(second)['passed']
        assert all('cw_distance_percentile' not in r for r in decode_table(second,'candidates'))


@pytest.mark.parametrize('corruption',['candidate_order','label_value','extra_validation','feature_input','missing_feedback'])
def test_builder_fails_closed_on_foreign_or_unproven_context(tmp_path,row_inputs,corruption):
    b,public=builder_fixture(tmp_path,row_inputs,'free32_cwaware')
    _,current,chosen,_,_=measured_next_round(b,public,tmp_path,row_inputs.scope)
    current=deepcopy(current)
    if corruption=='candidate_order':current['candidate_ids'].reverse()
    elif corruption=='label_value':current['observed_labels'][chosen[0]][0]+=1
    elif corruption=='extra_validation':current['validation_labels']={'forbidden':[1.,2.]}
    elif corruption=='feature_input':b.dataset.feature_frame.loc[0,'Density g/ml']+=1
    else:
        path=tmp_path/'round_0000.freeze.json';payload=read_envelope(path)['payload'];payload['audit']={}
        frozen=write_envelope(path,payload)
        receipt=read_envelope(tmp_path/'round_0000.labels.json')['payload'];receipt['freeze_hash']=frozen['sha256']
        write_envelope(tmp_path/'round_0000.labels.json',receipt)
    with pytest.raises(ValueError):b(current)


def test_screen_uses_128_candidates_16_pending_no_measured_values_or_memory(tmp_path,row_inputs):
    b,public=builder_fixture(tmp_path,row_inputs,'cw16_llm16_screen_v1');result=b(public);packet=result['packet']
    assert len(packet['candidates'])==128 and len(packet['pending_CW16'])==16
    assert set(result['fallback_ids'])==set(r['id'] for r in packet['candidates'])
    assert 'memory' not in packet and packet['observed_outcome_values_in_packet']==0
    card=packet['candidates'][0];raw=row_inputs.predictions[row_inputs.positions[card['id']]]
    assert card['pred_V1_V2']==[round(float(raw[1]),3),round(float(raw[4]),3)]
    assert result['premeasurement_predictions'][card['id']]==[float(raw[1]),float(raw[4])]
    assert set(result['premeasurement_predictions'])==set(public['candidate_ids'])


@pytest.mark.historical_archive
def test_screen_packet_matches_pinned_original_pure_constructor(tmp_path,row_inputs):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import Descriptors,rdFingerprintGenerator
    from tests.support import historical_reference as legacy
    b,public=builder_fixture(tmp_path,row_inputs,'cw16_llm16_screen_v1');actual=b(public)['packet']
    old=legacy.load('src/qgeognn_al/active_learning_v2/llm_screen.py',['make_packet'],injected={
        'pd':pd,'Chem':Chem,'DataStructs':DataStructs,'Descriptors':Descriptors,'rdFingerprintGenerator':rdFingerprintGenerator,
        'PROMPT':actual['prompt'],'load_predictor_checkpoint':lambda path:object(),
        'predict_outputs':lambda model,atom,angle,indices:(row_inputs.predictions[indices],indices),
        '_squared_distances':_squared_distances})
    assert old.__source_sha256__==get_profile('cw16_llm16_screen_v1').source_sha256
    proposal=shortlist(row_inputs.cw,row_inputs.raw,333,157,0)
    expected=old.make_packet(SimpleNamespace(data=row_inputs.frame,atom=[],angle=[]),
        np.arange(len(row_inputs.ids)),333,None,row_inputs.cw,proposal)
    assert {k:v for k,v in actual.items() if k!='packet_sha256'}==expected
    assert actual['packet_sha256']==hashlib.sha256((json.dumps(expected,indent=2)+'\n').encode()).hexdigest()

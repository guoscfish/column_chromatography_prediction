"""Provider-free integration of real source plans, correction and durable resume."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.qgeognn_al.infra.contracts import content_hash
from src.qgeognn_al.al.agents.catalog import decode_table
from src.qgeognn_al.al.agents.service import AgentSelector
from src.qgeognn_al.al.agents.transport import RecordedTransport, ProviderConfig, request_hash
from src.qgeognn_al.al.agents.replay_packets import build_messages

FIXTURE=Path(__file__).parent/'fixtures/llm_recorded_replay.json.gz'

@pytest.fixture(scope='module')
def cases():
    data=json.loads(gzip.decompress(FIXTURE.read_bytes()))
    return [c for c in data['cases'] if c['profile']=='cw16_llm16_scientist_v2']


def setup_case(case,tmp_path):
    files=case['files'];packet=json.loads(files['input.json']['text']);frozen=json.loads(files['batch_freeze.json']['text'])
    cfg=case['transport'];config=ProviderConfig(model=cfg['model'],reasoning_effort=cfg['reasoning_effort'],base_url=cfg['base_url'])
    messages=build_messages(case['profile'],packet);recordings={}
    for name in sorted(n for n in files if n.startswith('turn_') and n.endswith('.json')):
        saved=json.loads(files[name]['text'])
        recordings[request_hash(messages,config)]={'answer':saved['answer'],
            'provenance':{**saved['provenance'],'legacy_request_sha256':saved['request_sha256']}}
        index=int(name[5:7]);rejected=f'rejected_turn_{index:02d}.json'
        if rejected in files:
            messages += [{'role':'assistant','content':saved['answer']},
                         {'role':'user','content':json.dumps(json.loads(files[rejected]['text']),ensure_ascii=False)}]
    transport=RecordedTransport(recordings)
    supplied={'packet':packet,'pending_ids':frozen['pending_ids']}
    selector=AgentSelector(case['profile'],packet_supplier=lambda public:copy.deepcopy(supplied),
        packet_contract_hash=content_hash({'source_input_sha256':files['input.json']['sha256']}),
        transport=transport,provider_config=config,directory=tmp_path/'requests')
    observed=decode_table(packet,'observed')
    candidates=decode_table(packet,'candidates')+decode_table(packet,'pending')
    snapshot=SimpleNamespace(train_ids=tuple(row['id'] for row in observed),
        candidate_ids=tuple(row['id'] for row in candidates),
        predictor=SimpleNamespace(model_checkpoint_hash='a'*64),sha256='b'*64)
    state={'round':case['round'],'labels':{row['id']:[row['V1_ml'],row['V2_ml']] for row in observed}}
    spec=SimpleNamespace(seed=case['seed'],acquisition_seed=case['seed']+7)
    return selector,transport,supplied,snapshot,state,spec,frozen


@pytest.mark.parametrize('round_index',[0,1])
def test_real_recorded_plan_correction_and_durable_resume(cases,tmp_path,round_index):
    case=next(c for c in cases if c['round']==round_index)
    selector,transport,supplied,snapshot,state,spec,frozen=setup_case(case,tmp_path)
    first=selector.select(snapshot=snapshot,state=state,budget=32,spec=spec)
    assert first['selected_ids']==frozen['batch_ids']
    assert transport.calls==round_index+1
    assert first['audit']['provider_invoked'] is False
    assert all(p['source_kind']=='recorded' and p['paid_request'] is False for p in first['audit']['response_provenance'])
    calls=transport.calls
    second=selector.select(snapshot=snapshot,state=state,budget=32,spec=spec)
    assert first==second and transport.calls==calls


def test_full_pool_cannot_be_silently_reduced_to128(cases,tmp_path):
    selector,transport,supplied,snapshot,state,spec,_=setup_case(cases[0],tmp_path)
    packet=supplied['packet'];packet['candidates']['rows']=packet['candidates']['rows'][:128]
    packet['candidate_count']=128
    packet['packet_sha256']=content_hash({k:v for k,v in packet.items() if k!='packet_sha256'})
    with pytest.raises(ValueError,match='every authorized U'):
        selector.select(snapshot=snapshot,state=state,budget=32,spec=spec)
    assert transport.calls==0 and not (tmp_path/'requests').exists()


def test_observed_values_are_independently_authorized_before_provider(cases,tmp_path):
    selector,transport,_,snapshot,state,spec,_=setup_case(cases[0],tmp_path)
    state['labels'][snapshot.train_ids[0]][0]+=1
    with pytest.raises(ValueError,match='authorized measured feedback'):
        selector.select(snapshot=snapshot,state=state,budget=32,spec=spec)
    assert transport.calls==0


def test_foreign_seed_and_round_never_reach_transport(cases,tmp_path):
    selector,transport,_,snapshot,state,spec,_=setup_case(cases[0],tmp_path)
    spec.seed=6101
    with pytest.raises(ValueError,match='another seed or round'):
        selector.select(snapshot=snapshot,state=state,budget=32,spec=spec)
    assert transport.calls==0

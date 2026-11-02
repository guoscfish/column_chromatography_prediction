"""Offline recovery tests; no network, labels, signals, or training."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.fixture
def adapter(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location("persistent_free_test",root/"scripts/studies/run_free_cw_persistent_20261102.py")
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    p.load_base()
    for name in ('STUDY','SEEDS','VERSION','ARMS','METHODS','BUDGETS','initial_paths','Context',
        'code_hashes','once','exclusive_lock','input_view','prompt','validate_plan',
        'request_with_retries','live_call','run_selection'):
        monkeypatch.setattr(p.m.s,name,getattr(p.m.s,name))
    for module,name in ((p.m.views,'ARMS'),(p.m.views,'VERSION'),(p.m.base,'STUDY'),(p.m.base,'VERSION'),
        (p.m.shortseq,'fit_from_same_initialization'),(p.m.fitrunner,'train_source')):
        monkeypatch.setattr(module,name,getattr(module,name))
    p.activate_worker(157)
    p.test_config=p.m.read(p.STUDY/"config.json")["transport"]
    monkeypatch.setattr(p,"validate",lambda:{"offline":True})
    monkeypatch.setattr(p.time,"sleep",lambda *args:None)
    monkeypatch.setattr(p.m.base,"status_record",lambda *args,**kwargs:None)
    for module in (p,p.a,p.m):monkeypatch.setattr(module,"STUDY",tmp_path)
    monkeypatch.setattr(p,"EXT",tmp_path/"persistent")
    monkeypatch.setattr(p,"POLICY",tmp_path/"policy.json")
    p.POLICY.write_text('{}')
    (tmp_path/'locks').mkdir()
    return p


def test_transport_exceeds_old_cap_without_reset_and_reuses_success(adapter,tmp_path):
    import httpx
    p=adapter;c=p.test_config;d=tmp_path/"round_00";calls=[]
    def call(*args):
        calls.append(args)
        if len(calls)<=12:raise httpx.RemoteProtocolError("offline transient")
        return '{}',{"mock":True}
    result=p.request_with_retries([],c,d,0,call)
    assert len(calls)==13 and result['success']
    assert len(list(d.glob('attempt_*.json')))==26
    again=p.request_with_retries([],c,d,0,lambda *a:pytest.fail('duplicate paid request'))
    assert again==result and (d/'attempt_00_12.json').exists()
    assert [p.backoff(i) for i in range(9)]==[2,10,60,120,300,600,600,600,600]
    assert p.m.stream.body_for([],c)=={"model":"gpt-6.1-sol","input":[],"reasoning":{"effort":"xhigh"},
        "tools":[],"store":False,"max_output_tokens":32768,"stream":True}


def test_legacy_receipts_and_higher_attempt_indices_retained(adapter,tmp_path):
    p=adapter;c=p.test_config;d=tmp_path/'round_00';h=p.request_hash([],c)
    for i in range(101):
        p.a.once(d/f'attempt_00_{i:02d}.json',{"request_sha256":h,"success":False,
            "fatal_protocol_error":i==7,"reason":"provider HTTP 522" if i==7 else "offline"})
    before=(d/'attempt_00_07.json').read_bytes()
    assert p.request_with_retries([],c,d,0,lambda *a:('{}',{}))['success']
    assert (d/'attempt_00_101.json').exists() and (d/'attempt_00_07.json').read_bytes()==before
    assert list((p.EXT/'reclassified').rglob('attempt_00_07.json'))


@pytest.mark.parametrize('reason',['provider HTTP 401','provider HTTP 402','wrong-model response','unsupported reasoning effort'])
def test_configuration_failures_never_retried_or_downgraded(adapter,tmp_path,reason):
    p=adapter;c=p.test_config;calls=[]
    def call(*args):
        calls.append(1);raise p.m.FatalProtocolError(reason)
    for _ in range(2):
        with pytest.raises(p.m.FatalProtocolError):p.request_with_retries([],c,tmp_path/'round_00',0,call)
    assert len(calls)==1


def test_durable_response_recovered_and_live_owner_preserved(adapter,tmp_path,monkeypatch):
    p=adapter;c=p.test_config;d=tmp_path/'round_00';h=p.request_hash([],c)
    p.a.once(d/'attempt_00_00.started.json',{'request_sha256':h})
    p.a.once(d/'stream_00_00.json',{'pid':1234})
    monkeypatch.setattr(p,'process_command',lambda pid:'alive')
    with pytest.raises(RuntimeError,match='owner remains alive'):
        p.request_with_retries([],c,d,0,lambda *a:pytest.fail('duplicate active request'))
    p.a.once(d/'stream_00_00.completed.json',{'request_sha256':h,'answer':'{}',
        'provenance':{'served_model':'gpt-6.1-sol','usage':{}}})
    saved=p.request_with_retries([],c,d,0,lambda *a:pytest.fail('durable response resent'))
    assert saved['recovered_local_completion'] and saved['answer']=='{}'


def test_dead_owner_without_response_accounted_then_retried(adapter,tmp_path,monkeypatch):
    p=adapter;c=p.test_config;d=tmp_path/'round_00';h=p.request_hash([],c)
    p.a.once(d/'attempt_00_00.started.json',{'request_sha256':h})
    before=(d/'attempt_00_00.started.json').read_bytes()
    monkeypatch.setattr(p,'process_command',lambda pid:'')
    assert p.request_with_retries([],c,d,0,lambda *a:('{}',{}))['success']
    assert (d/'attempt_00_00.started.json').read_bytes()==before
    assert p.m.read(d/'attempt_00_00.json')['provider_outcome'].startswith('unconfirmed')
    assert (d/'attempt_00_01.json').exists()


def test_answer_corrections_exceed_eight_and_keep_original_messages(adapter,tmp_path,monkeypatch):
    p=adapter;c=p.test_config
    catalog=SimpleNamespace(candidates=list(range(40)),observed=list(range(333)),pending=[])
    packet={'selection_count':32,'candidate_count':40,'observed_count':333,'active_label_count':333,
        'candidate_data':[[i,0.12345] for i in range(40)]}
    packet['packet_sha256']=p.m.s.stable_hash(packet)
    monkeypatch.setattr(p.m.s,'encode_catalog',lambda catalog:{'candidate_data':packet['candidate_data']})
    def validator(result,*args):
        if result.get('valid'):return result
        raise ValueError('offline invalid selection')
    monkeypatch.setattr(p.m.s,'validate_plan',validator)
    def execute(limit,d,old=False):
        count=[]
        def call(*args):
            count.append(1);return json.dumps({'valid':len(count)>=limit}),{}
        if old:
            monkeypatch.setattr(p.m.s,'request_with_retries',p.request_with_retries)
            p.OLD_RUN_SELECTION(packet,catalog,d,'cw_aware',c,call)
        else:p.run_selection(packet,catalog,d,'cw_aware',c,call)
        return len(count)
    old=tmp_path/'old/round_00';new=tmp_path/'new/round_00'
    assert execute(8,old,True)==8 and execute(12,new)==12
    for i in range(8):
        assert p.m.read(old/f'request_{i:02d}.json')==p.m.read(new/f'request_{i:02d}.json')
    assert execute(12,new)==0
    assert (new/'selection.json').exists() and len(list(new.glob('rejected_*.json')))==11


def test_healthy_stream_not_paused_by_monitor(adapter,tmp_path,monkeypatch):
    p=adapter;pid=1234
    monkeypatch.setattr(p,'process_command',lambda pid:f'Ss {p.ENTRY} worker --seed 157')
    p.a.once(tmp_path/'progress/stream_seed_157.json',{'pid':pid,'phase':'IN_PROGRESS','last_line_utc':p.m.now()})
    monkeypatch.setattr(p.os,'kill',lambda *a:pytest.fail('healthy request signaled'))
    assert not p.retire_worker(157,pid,'offline watchdog',idle_limit=660)


@pytest.mark.parametrize('durable',[False,True])
def test_stale_watchdog_preserves_durable_completion(adapter,tmp_path,monkeypatch,durable):
    p=adapter;pid=1234;signals=[]
    monkeypatch.setattr(p,'process_command',lambda pid:f'T {p.ENTRY} worker --seed 157')
    monkeypatch.setattr(p.os,'kill',lambda pid,sig:signals.append(sig))
    d=tmp_path/f'selections/seed_157/{p.m.ARM}/round_00'
    p.a.once(d/'attempt_00_08.started.json',{'request_sha256':'offline'})
    started=(d/'attempt_00_08.started.json').read_bytes()
    p.a.once(tmp_path/'progress/seed_157.json',{'pid':pid,'phase':'SELECTING'})
    p.a.once(tmp_path/'progress/stream_seed_157.json',{'pid':pid,'phase':'IN_PROGRESS','turn':0,'attempt':8,
        'last_line_utc':'2020-01-01T00:00:00+00:00'})
    if durable:p.a.once(d/'stream_00_08.response.json',{'offline':'durable'})
    assert p.retire_worker(157,pid,'offline watchdog',idle_limit=660)==(not durable)
    assert (p.signal.SIGTERM in signals)==(not durable)
    assert (d/'attempt_00_08.started.json').read_bytes()==started
    assert (d/'attempt_00_08.json').exists()==(not durable)


def test_fatal_state_not_restarted_but_exhausted_legacy_is(adapter):
    p=adapter
    assert p.transient_stop({'phase':'FAILED_CLOSED','reason':'bounded transport attempt budget exhausted; no fallback'})
    assert not p.transient_stop({'phase':'FAILED_CLOSED','reason':'wrong model'})
    assert not p.transient_stop({'phase':'FAILED_CLOSED','reason':'provider HTTP 401'})

import importlib.util
from pathlib import Path
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/qgeognn_al/resources.py').exists())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
fixture=load('immediate_fixture',ROOT/'tests/active_learning_v2/test_free_cw_persistent.py')
adapter=fixture.adapter
now=load('immediate_overlay',Path(__file__).with_name('controller_overlay.py'))
recovery=load('prior_403_immediate_test',now.PRIOR)
def test_immediate_retry_preserves_input_success_and_counters(adapter,tmp_path,monkeypatch):
    p=adapter;recovery.install(p);now.install(p);sleeps=[];calls=[]
    monkeypatch.setattr(p.time,'sleep',lambda delay:sleeps.append(delay))
    messages=[{'role':'user','content':'complete original input'}]
    def call(m,c):
        calls.append((m,c))
        if len(calls)<4:raise p.m.FatalProtocolError('provider HTTP 403')
        return '{}',{}
    d=tmp_path/'round_09';result=p.request_with_retries(messages,p.test_config,d,0,call)
    assert len(calls)==4 and all(x==(messages,p.test_config) for x in calls)
    assert sleeps==[0,0,0] and p.backoff(100000)==0
    assert p.m.read(d/'attempt_00_02.json')['next_backoff_seconds']==0
    assert p.request_with_retries(messages,p.test_config,d,0,lambda *a:pytest.fail('duplicate'))==result
@pytest.mark.parametrize('reason',['provider HTTP 401','provider HTTP 402','wrong-model response','unsupported reasoning effort'])
def test_other_fatal_unchanged(adapter,tmp_path,reason):
    p=adapter;recovery.install(p);now.install(p)
    def call(*args):raise p.m.FatalProtocolError(reason)
    with pytest.raises(p.m.FatalProtocolError):p.request_with_retries([],p.test_config,tmp_path/'round_09',0,call)

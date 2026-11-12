"""Transport recovery and credential isolation without any network requests."""
from types import SimpleNamespace
import pytest
from src.qgeognn_al.al.agents.transport import (ProviderConfig, RecordedTransport,
    ResponsesTransport, DurableTransport, AmbiguousRequest, request_hash)

CONFIG=ProviderConfig('frozen-model','high','https://example.invalid/v1')
MESSAGES=[{'role':'system','content':'one pinned prompt'},{'role':'user','content':'complete packet'}]


def test_recorded_binding_and_successful_resume_never_redispatch(tmp_path):
    key=request_hash(MESSAGES,CONFIG)
    recorded=RecordedTransport({key:{'answer':'{"selected_ids":["u"]}'}})
    durable=DurableTransport(recorded,tmp_path/'session')
    first=durable.call(MESSAGES,CONFIG,turn=0)
    assert DurableTransport(recorded,tmp_path/'session').call(MESSAGES,CONFIG,turn=0)==first
    assert recorded.calls==1
    with pytest.raises(ValueError,match='changed'):
        durable.call([*MESSAGES,{'role':'user','content':'other round'}],CONFIG,turn=0)
    with pytest.raises(ValueError,match='quota'):
        durable.call(MESSAGES,CONFIG,turn=3)


def test_unknown_outcome_cannot_spend_again_on_resume(tmp_path):
    class Failed:
        kind='mock';paid=False;calls=0
        def call(self,*args):self.calls+=1;raise TimeoutError('unknown completion')
    transport=Failed();directory=tmp_path/'request'
    with pytest.raises(TimeoutError):DurableTransport(transport,directory).call(MESSAGES,CONFIG,turn=0)
    with pytest.raises(AmbiguousRequest):DurableTransport(transport,directory).call(MESSAGES,CONFIG,turn=0)
    assert transport.calls==1


def test_provider_gate_precedes_credentials_and_client(monkeypatch):
    def fail(**kwargs):pytest.fail('client must not be constructed')
    monkeypatch.delenv('SCIENTIST_API_KEY',raising=False)
    with pytest.raises(PermissionError):ResponsesTransport(client_factory=fail).call(MESSAGES,CONFIG)


def test_sdk_stream_shape_model_and_native_tool_audit_with_mock(monkeypatch):
    calls=[]
    response=SimpleNamespace(status='completed',model='frozen-model-20261112',
        output=[SimpleNamespace(type='message')],output_text='{"ok":true}',id='recorded-id',usage=None)
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get_final_response(self):return response
    class Client:
        def __init__(self,**kwargs):calls.append(kwargs);self.responses=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def stream(self,**kwargs):calls.append(kwargs);return Stream()
    monkeypatch.setenv('SCIENTIST_API_KEY','owned-mock-secret')
    transport=ResponsesTransport(allow_provider=True,client_factory=Client)
    answer,provenance=transport.call(MESSAGES,CONFIG)
    assert answer==response.output_text and provenance['native_tool_calls']==0
    assert calls[0]['max_retries']==0
    assert calls[1]=={'model':CONFIG.model,'input':MESSAGES,'reasoning':{'effort':'high'},'tools':[],'store':False}
    response.output=[SimpleNamespace(type='function_call')]
    with pytest.raises(RuntimeError):transport.call(MESSAGES,CONFIG)
    response.output=[];response.model='different-model'
    with pytest.raises(RuntimeError):transport.call(MESSAGES,CONFIG)


def test_no_ambient_fallback_or_unsafe_url():
    for url in ('http://example.invalid','https://key:secret@example.invalid/v1','https://example.invalid/?key=x'):
        with pytest.raises(ValueError):ProviderConfig('x','high',url)
    with pytest.raises(ValueError):request_hash([{'role':'tool','content':'hidden'}],CONFIG)

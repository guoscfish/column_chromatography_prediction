"""Offline authorization boundary and immutable attempt accounting checks."""
import importlib.util
from pathlib import Path
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/qgeognn_al/resources.py').exists())
spec = importlib.util.spec_from_file_location('fixture403', ROOT/'tests/active_learning_v2/test_free_cw_persistent.py')
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
adapter = fixture.adapter
spec = importlib.util.spec_from_file_location('overlay403', Path(__file__).with_name('controller_overlay.py'))
overlay = importlib.util.module_from_spec(spec); spec.loader.exec_module(overlay)


def test_preserve_failed_receipt_and_reuse_success(adapter, tmp_path):
    p = adapter; overlay.install(p); d = tmp_path/'round_09'
    saved = {'request_sha256':p.request_hash([],p.test_config),'success':False,
             'fatal_protocol_error':True,'reason':overlay.REASON}
    p.a.once(d/'attempt_00_00.json',saved); before=(d/'attempt_00_00.json').read_bytes()
    calls=[]
    def call(*args):calls.append(args);return '{}',{}
    result=p.request_with_retries([],p.test_config,d,0,call)
    assert result['success'] and len(calls)==1
    assert (d/'attempt_00_00.json').read_bytes()==before
    assert (d/'attempt_00_01.json').exists()
    assert p.request_with_retries([],p.test_config,d,0,lambda *a:pytest.fail('duplicate'))==result


def test_new_403_keeps_status_backoff_and_full_input(adapter,tmp_path):
    p=adapter;overlay.install(p);calls=[];messages=[{'role':'user','content':'complete untruncated input'}]
    def call(msg,config):
        calls.append((msg,config))
        if len(calls)==1:raise p.m.FatalProtocolError(overlay.REASON)
        return '{}',{}
    d=tmp_path/'round_09';p.request_with_retries(messages,p.test_config,d,0,call)
    failed=p.m.read(d/'attempt_00_00.json')
    assert failed['http_status']==403 and failed['next_backoff_seconds']==2
    assert len(calls)==2 and calls[0]==calls[1]==(messages,p.test_config)
    assert p.transient_stop({'phase':'FAILED_CLOSED','reason':overlay.REASON})
    assert [p.backoff(i) for i in [5,8,1000]]==[600,600,600]


@pytest.mark.parametrize('reason',['provider HTTP 401','provider HTTP 402','provider HTTP 403 other',
    'provider HTTP 404','wrong-model response','unsupported reasoning effort','unknown failure'])
def test_other_fatal_remains_fatal(adapter,tmp_path,reason):
    p=adapter;overlay.install(p);calls=[]
    def call(*args):calls.append(1);raise p.m.FatalProtocolError(reason)
    with pytest.raises(p.m.FatalProtocolError):p.request_with_retries([],p.test_config,tmp_path/'round_09',0,call)
    assert len(calls)==1


def test_old_h2_handling_preserved(adapter):
    p=adapter
    shim=overlay.module('h2shim_offline',overlay.PREVIOUS/'upstream_h2_worker.py')
    shim.install(p);overlay.install(p)
    assert p.legacy_transient({'fatal_protocol_error':True,'reason':shim.REASON})
    assert p.legacy_transient({'fatal_protocol_error':True,'reason':overlay.REASON})
    assert p.a.transient_http_reason('provider HTTP 522')==522

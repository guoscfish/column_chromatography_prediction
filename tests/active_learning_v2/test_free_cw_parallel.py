"""Offline checks for five-key scheduling and bounded transport recovery."""
import importlib.util
from pathlib import Path
import json
import pytest


@pytest.fixture
def adapter(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    path=root/"scripts/studies/run_free_cw_parallel_20261101.py"
    spec=importlib.util.spec_from_file_location("parallel_free_test",path)
    a=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    a.load_base()
    a.activate(157)
    monkeypatch.setattr(a,"validate",lambda:{"offline":True})
    monkeypatch.setattr(a.time,"sleep",lambda *args:None)
    monkeypatch.setattr(a.m.base,"status_record",lambda *args,**kwargs:None)
    return a


def config(adapter):
    return adapter.m.read(adapter.STUDY/"config.json")["transport"]


def test_five_existing_keys_are_distinct_without_disclosure(adapter):
    audit=adapter.m.credential_audit()
    assert audit["distinct_keys"]==5 and not audit["key_values_logged"]
    assert len(set(audit["key_aliases"].values()))==5


def test_same_frozen_model_body_and_bounded_caps(adapter,tmp_path):
    c=config(adapter)
    assert adapter.m.stream.body_for([],c)=={"model":"gpt-6.1-sol","input":[],"reasoning":{"effort":"xhigh"},
        "tools":[],"store":False,"max_output_tokens":32768,"stream":True}
    assert adapter.attempt_cap(tmp_path/"round_00",0,c)==8
    assert adapter.attempt_cap(tmp_path/"round_00",1,c)==4
    assert adapter.attempt_cap(tmp_path/"round_01",0,c)==4
    assert adapter.transient_http_reason("provider HTTP 522")==522
    assert adapter.transient_http_reason("provider HTTP 401") is None
    assert adapter.transient_http_reason("served model differs") is None


def test_old_522_receipt_preserved_then_success_reused(adapter,tmp_path,monkeypatch):
    c=config(adapter)
    monkeypatch.setattr(adapter,"STUDY",tmp_path)
    monkeypatch.setattr(adapter,"EXT",tmp_path/"extension")
    messages=[]; d=tmp_path/"round_00"
    h=adapter.m.s.stable_hash({"messages":messages,"config":c})
    for j in range(4):
        adapter.once(d/f"attempt_00_{j:02d}.json",{"request_sha256":h,"success":False,"fatal_protocol_error":j==3,
            "reason":"provider HTTP 522" if j==3 else "network", "elapsed_seconds":1})
    before=(d/"attempt_00_03.json").read_bytes()
    calls=[]
    def call(*args):
        calls.append(1)
        return "{}",{"mock":True}
    result=adapter.request_with_retries(messages,c,d,0,call)
    assert result["success"] and len(calls)==1
    assert (d/"attempt_00_03.json").read_bytes()==before
    again=adapter.request_with_retries(messages,c,d,0,lambda *a:pytest.fail("duplicate success"))
    assert again==result and (d/"attempt_00_04.json").exists()


def test_ambiguous_request_not_resent(adapter,tmp_path):
    c=config(adapter);d=tmp_path/"round_00"
    adapter.once(d/"attempt_00_00.started.json",{"request_sha256":adapter.m.s.stable_hash({"messages":[],"config":c})})
    with pytest.raises(RuntimeError,match="ambiguous"):
        adapter.request_with_retries([],c,d,0,lambda *args:pytest.fail("in-flight request repeated"))


def test_gateway_retries_finite_across_restarts(adapter,tmp_path):
    c=config(adapter); calls=[];d=tmp_path/"round_00"
    def fail(*args):
        calls.append(1)
        raise adapter.m.FatalProtocolError("provider HTTP 522")
    for _ in range(2):
        with pytest.raises(RuntimeError,match="bounded transport"):
            adapter.request_with_retries([],c,d,0,fail)
    assert len(calls)==8


def test_model_or_credential_failures_never_retried(adapter,tmp_path):
    c=config(adapter);calls=[]
    def fail(*args):
        calls.append(1)
        raise adapter.m.FatalProtocolError("provider HTTP 401")
    for _ in range(2):
        with pytest.raises(adapter.m.FatalProtocolError,match="401"):
            adapter.request_with_retries([],c,tmp_path/"round_00",0,fail)
    assert len(calls)==1

"""Safety and durable replay checks for the authorized CW-visible rerun."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    if root.name=="frozen_source":
        root=root.parents[3]
    path=root/"scripts/studies/run_free_cw_sol61_five_20261101.py"
    spec=importlib.util.spec_from_file_location("new_free_cw_adapter_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.import_frozen()
    for name in ("STUDY","SEEDS","VERSION","ARMS","METHODS","BUDGETS","initial_paths","Context",
        "code_hashes","once","exclusive_lock","input_view","prompt","validate_plan","request_with_retries","live_call"):
        monkeypatch.setattr(module.s,name,getattr(module.s,name))
    monkeypatch.setattr(module.views,"ARMS",dict(module.views.ARMS))
    monkeypatch.setattr(module.views,"VERSION",module.views.VERSION)
    monkeypatch.setattr(module.base,"STUDY",module.base.STUDY)
    monkeypatch.setattr(module.base,"VERSION",module.base.VERSION)
    monkeypatch.setattr(module.shortseq,"fit_from_same_initialization",module.shortseq.fit_from_same_initialization)
    monkeypatch.setattr(module.fitrunner,"train_source",module.fitrunner.train_source)
    module.activate()
    return module


def test_frozen_initial_protocol_and_packet_diffs(adapter):
    assert adapter.s.validate()["status"]=="VALID"
    assert adapter.BUDGETS[-1]==1485 and len(adapter.BUDGETS)==37
    for seed in adapter.SEEDS:
        diff=adapter.read(adapter.STUDY/f"offline_checks/packet_diff_seed_{seed}.json")
        assert diff["passed"] and len(diff["changes"])==2997
        assert diff["non_CW_content_and_order_identical"]
    assert "CW distance is a percentile" in adapter.prompt("cw_aware")
    assert "Do not cite row IDs in free prose" not in adapter.prompt("cw_aware")


def test_cached_success_is_reused_without_call(adapter,tmp_path):
    config=adapter.read(adapter.STUDY/"config.json")["transport"]
    messages=[{"role":"user","content":"offline"}]
    calls=[]
    def call(*args):
        calls.append(1)
        return "{}",{"mock":True}
    first=adapter.request_with_retries(messages,config,tmp_path,0,call)
    second=adapter.request_with_retries(messages,config,tmp_path,0,lambda *a:pytest.fail("duplicate call"))
    assert first==second and len(calls)==1


def test_ambiguous_request_never_replayed(adapter,tmp_path):
    config=adapter.read(adapter.STUDY/"config.json")["transport"]
    messages=[]
    adapter.once(tmp_path/"attempt_00_00.started.json",{"request_sha256":adapter.s.stable_hash({"messages":messages,"config":config})})
    with pytest.raises(RuntimeError,match="ambiguous interrupted"):
        adapter.request_with_retries(messages,config,tmp_path,0,lambda *a:pytest.fail("unexpected paid call"))


def test_durable_stream_completion_recovered_without_call(adapter,tmp_path,monkeypatch):
    monkeypatch.setattr(adapter,"STUDY",tmp_path)
    config={"model":"gpt-6.1-sol","reasoning_effort":"xhigh","transport_attempts":4}
    messages=[]
    request_hash=adapter.s.stable_hash({"messages":messages,"config":config})
    d=tmp_path/"round"
    adapter.once(d/"attempt_00_00.started.json",{"request_sha256":request_hash})
    adapter.once(d/"stream_00_00.response.json",{"request_sha256":request_hash,"request_id":"req-offline",
        "response":{"id":"resp-offline","model":"gpt-6.1-sol","status":"completed","usage":{"total_tokens":1},
            "output":[{"type":"message","content":[{"type":"output_text","text":"{}"}]}]}})
    result=adapter.request_with_retries(messages,config,d,0,lambda *a:pytest.fail("duplicate call"))
    assert result["recovered_local_completion"] and result["provenance"]["request_id"]=="req-offline"


def test_wrong_or_changed_model_stops_without_retry(adapter,tmp_path,monkeypatch):
    monkeypatch.setattr(adapter,"STUDY",tmp_path)
    config={"model":"gpt-6.1-sol","reasoning_effort":"xhigh"}
    response={"id":"offline","model":"gpt-6-astra","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"{}"}]}]}
    with pytest.raises(adapter.FatalProtocolError,match="wrong-model"):
        adapter.response_result(response,config)
    response["model"]="gpt-6.1-sol"
    adapter.response_result(response,config)
    response["model"]="gpt-6.1-sol-other"
    with pytest.raises(adapter.FatalProtocolError,match="locked"):
        adapter.response_result(response,config)


def test_transient_transport_has_fixed_persistent_bound(adapter,tmp_path,monkeypatch):
    import httpx
    monkeypatch.setattr(adapter.time,"sleep",lambda *a:None)
    monkeypatch.setattr(adapter.base,"status_record",lambda *a,**k:None)
    config={"model":"gpt-6.1-sol","reasoning_effort":"xhigh","transport_attempts":4,"bounded_backoff_seconds":[2,10,60]}
    calls=[]
    def fail(*a):
        calls.append(1)
        raise httpx.ConnectError("offline")
    for _ in range(2):
        with pytest.raises(RuntimeError,match="bounded transport"):
            adapter.request_with_retries([],config,tmp_path,0,fail)
    assert len(calls)==4


def test_completed_training_receipt_skips_retraining(adapter,tmp_path,monkeypatch):
    import numpy as np
    import torch
    from src.qgeognn_al.active_learning_v2.gradient_features import state_dict_hash
    model=torch.nn.Linear(1,1)
    checkpoint=tmp_path/"best.pt"
    checkpoint.write_bytes(b"offline-finished-checkpoint")
    adapter.once(tmp_path/"fit_intent.json",{"offline":True})
    intent={"initialization_state_hash":state_dict_hash(model),"train_indices":[0],"validation_indices":[1],
        "preprocessing":{},"training_config":{},"fit_intent_sha256":adapter.sha(tmp_path/"fit_intent.json")}
    history=[{"epoch":1,"validation_selection_score":1.0}]
    adapter.once(tmp_path/"training_complete.json",{"intent":intent,"history":history,"best_epoch":1,
        "checkpoint_sha256":adapter.sha(checkpoint),"completed_utc":"offline"})
    monkeypatch.setattr(adapter,"ORIGINAL_TRAIN",lambda *a,**k:pytest.fail("finished fit retrained"),raising=False)
    assert adapter.durable_train_source(model,[],[],np.array([0]),np.array([1]),{},{},checkpoint)==(history,1)


def test_raw_nonmonotone_aulc_and_sample_std():
    path=Path(__file__).resolve().parents[2]/"scripts/studies/report_free_cw_sol61_five_20261101.py"
    spec=importlib.util.spec_from_file_location("free_cw_report_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points=[{"budget":333+32*j,"validation_selection_score":v} for j,v in enumerate([3.,1.,5.])]
    assert module.aulc(points)==2.5
    assert module.summarize([1.,3.])["seed_std_ddof1"]==pytest.approx(2**.5)


def test_api_body_matches_nocw_and_no_fallback(adapter):
    config=adapter.read(adapter.STUDY/"config.json")["transport"]
    body=adapter.stream.body_for([],config)
    assert body=={"model":"gpt-6.1-sol","input":[],"reasoning":{"effort":"xhigh"},
        "tools":[],"store":False,"max_output_tokens":32768,"stream":True}
    assert config["selection_attempts"]==8 and config["transport_attempts"]==4

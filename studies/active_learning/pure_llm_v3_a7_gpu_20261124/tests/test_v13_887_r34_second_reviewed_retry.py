from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor as gpu
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import is_sealed_relay_failure


def test_w13_extends_the_exact_two_closed_scopes_without_resetting_science():
    previous = gpu.SCOPES[887, 12]
    lane = gpu.SCOPES[887, 13]
    assert lane.archive[:-1] == previous.archive
    assert lane.payload_sha256 == previous.payload_sha256 == (
        "eb9170128722472e627a9ae16fa23692c1fc19c3ed6156513bde1c98e0c45a91"
    )
    assert (lane.seed,lane.round,lane.original_start_round,lane.replay_start_round) == (887,34,34,34)
    assert lane.cooldown_field == "not_before_unix" and lane.not_before_unix == 1795735622.4724555
    assert lane.recovery_budget == {"maximum_logical_requests":1,"maximum_physical_attempts":2}
    assert lane.budget == {"first_logical_requests":2,"maximum_logical_requests":4,"maximum_physical_attempts":8}
    assert lane.required_sources == previous.required_sources
    assert {"boundary.json","round_34/source.json","round_34/boundary_memory.json","round_34/boundary_user_message.json",
            "round_34/authoritative_packet.json","round_34/turn_00/request.json"} == set(dict(lane.fitted_prefix_files))
    review = lane.archive[-1]
    assert review.identity == gpu._887_R34_W12
    assert review.replay_files and len(review.replay_files) == 3
    assert len(review.attempts) == 2
    assert [item.attempt_id for item in review.attempts] == [
        "c69c1ba0999b411f805fa72b57b8ce86","8367b4506c584f0ab07ce0fa907546c7",
    ]
    for item in review.attempts:
        progress = item.progress
        assert progress["provider_error_code"] == "upstream_error" and progress["sse_event_count"] == 41
        assert "response.output_text.delta" not in progress["sse_milestones"]
        assert "response.completed" not in progress["sse_milestones"]
        assert is_sealed_relay_failure({"error":item.failure,"provider_status":"failed",
            "response_id":progress["conflicting_response_id"],"provider_request_id":progress["provider_request_id"],
            "served_model":progress["model"],"usage":{}})
        assert not is_sealed_relay_failure({"error":item.failure,"provider_status":"failed",
            "response_id":progress["response_id"],"provider_request_id":progress["provider_request_id"],
            "served_model":progress["model"],"usage":{}})


def test_w13_does_not_inherit_r33s_full_import_budget():
    lane = gpu.SCOPES[887, 13]
    parent = SimpleNamespace(digest=lane.archive[-1].identity.contract, profile={})
    assert gpu._reviewed_r33_parent_budget(parent,lane) is None
    parent.profile={"full_imported_boundary_round":33}
    with pytest.raises(PermissionError,match="no reviewed successor"):
        gpu._reviewed_r33_parent_budget(parent,lane)
    assert replace(lane, boundary_physical_attempts=1).budget["maximum_physical_attempts"] == 7


def test_real_w12_two_slot_parent_uses_same_imported_round_and_full_original_budget(tmp_path, monkeypatch):
    lane = gpu.SCOPES[887, 13]
    parent_scope = gpu.SCOPES[887, 12]

    @dataclass
    class Parent:
        digest: str
        predecessor: dict
        profile: dict
        transport: object
        source_sha256: dict | None = None
        rounds: int = 36
        run_directory: str = ""

    @dataclass
    class State:
        round: int

    parent = Parent(
        lane.archive[-1].identity.contract,
        {"recovery_budget":parent_scope.recovery_budget,"execution_budget":parent_scope.budget,
         "cache_directory":str(tmp_path),"public_resource_receipt_sha256":"original_cache",
         "bridge_wait_seconds":7800},
        {"resume_boundaries":{"887":{"round":34}},"sealed_relay_retry_seconds":180},
        SimpleNamespace(logical_replies=3,transport_attempts=2),
    )
    request = {"request_id":lane.archive[-1].request_id,"payload_sha256":lane.payload_sha256,
               "binding":{name:"1"*64 for name in ("checkpoint_sha256","packet_sha256","wire_packet_sha256",
                                                   "base_messages_sha256",gpu.retry.ALIAS)},
               "payload":{"input":[{"role":"system","content":"normal"},
                                  {"role":"user","content":"same complete prior request"}]}}
    old_run=tmp_path/lane.archive[-1].identity.name
    old_release=tmp_path/"immutable_old_release"
    run=tmp_path/lane.run_name
    new_release=tmp_path/"independent_new_release"
    old_active=tmp_path/"round_34"
    records=[
        {"run":old_run,"directory":tmp_path,"active":old_active,"contract":parent,"approval":{},
         "request":request,"receipts":[],"terminal":{"not_before_unix":lane.cooldown_unix}},
        {"run":tmp_path/lane.archive[0].identity.name,"receipts":[]},
    ]
    pins = dict(lane.fitted_prefix_files)
    boundary=(State(34),{"memory":"same"},34)
    stage = SimpleNamespace(binding=request["binding"],messages=request["payload"]["input"])
    seen=[]
    def standard_boundary(root,directory,active,contract,approval,scope,cache):
        seen.append((root,active,scope.original_start_round,scope.round))
        return boundary,{"sealed":"prefix"}
    monkeypatch.setattr(gpu,"_archives",lambda *args:records)
    monkeypatch.setattr(gpu,"file_digest",lambda p:pins.get(p.relative_to(tmp_path).as_posix(),"original_cache"))
    monkeypatch.setattr(gpu.retry,"_boundary",standard_boundary)
    monkeypatch.setattr(gpu.seed_runner,"_scope",lambda *args:{})
    monkeypatch.setattr(gpu.seed_runner,"_stage",lambda *args:stage)
    monkeypatch.setattr(gpu.seed_runner,"pin_boundary_user_message",lambda staged,*args:staged)
    monkeypatch.setattr(gpu,"make_request",lambda *args:request)
    monkeypatch.setattr(gpu.retry,"_compare_science",lambda candidate,archived,host:candidate==archived and host is None)
    monkeypatch.setattr(gpu,"_source_manifest",lambda *args:{"source":"exact"})
    child,next_state,_,outgoing = gpu.construct(new_release,old_release,old_run,run,lane)
    assert seen == [(old_release,old_active,34,34)]
    assert next_state == boundary and outgoing is request
    assert child.predecessor["recovery_budget"] == parent_scope.recovery_budget
    assert child.predecessor["execution_budget"] == parent_scope.budget
    assert child.profile["resume_boundaries"]["887"]["round"] == 34
    wrong={**parent.predecessor,"execution_budget":lane.execution_budget(1)}
    records[0]["contract"]=replace(parent,predecessor=wrong)
    with pytest.raises(PermissionError,match="retired physical budget"):
        gpu.construct(new_release,old_release,old_run,run,lane)

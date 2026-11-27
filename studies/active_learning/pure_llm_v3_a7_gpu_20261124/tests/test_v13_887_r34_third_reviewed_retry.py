from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor as gpu
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import is_sealed_relay_failure


def test_w14_extends_the_exact_three_closed_scopes_without_resetting_science():
    previous = gpu.SCOPES[887, 13]
    lane = gpu.SCOPES[887, 14]
    assert lane.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in lane.archive] == [2, 2, 2]
    assert lane.payload_sha256 == previous.payload_sha256 == (
        "eb9170128722472e627a9ae16fa23692c1fc19c3ed6156513bde1c98e0c45a91"
    )
    assert (lane.seed,lane.round,lane.original_start_round,lane.replay_start_round) == (887,34,34,34)
    assert lane.cooldown_field == "not_before_unix" and lane.not_before_unix == 1795736713.3270583
    assert lane.recovery_budget == {"maximum_logical_requests":1,"maximum_physical_attempts":2}
    assert lane.budget == {"first_logical_requests":2,"maximum_logical_requests":4,"maximum_physical_attempts":8}
    assert lane.required_sources == previous.required_sources
    assert {"boundary.json","round_34/source.json","round_34/boundary_memory.json","round_34/boundary_user_message.json",
            "round_34/authoritative_packet.json","round_34/turn_00/request.json"} == set(dict(lane.fitted_prefix_files))
    review = lane.archive[-1]
    assert review.identity == gpu._887_R34_W13
    assert review.replay_files and len(review.replay_files) == 3
    assert len(review.attempts) == 2
    assert [item.attempt_id for item in review.attempts] == [
        "8df7bf378f1342f2952ca941e7f7d443","6f0af1c51ba64c9dbd0dc74e16f93fb7",
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


def test_w14_does_not_inherit_r33s_full_import_budget():
    lane = gpu.SCOPES[887, 14]
    parent = SimpleNamespace(digest=lane.archive[-1].identity.contract, profile={})
    assert gpu._reviewed_r33_parent_budget(parent,lane) is None
    parent.profile={"full_imported_boundary_round":33}
    with pytest.raises(PermissionError,match="no reviewed successor"):
        gpu._reviewed_r33_parent_budget(parent,lane)
    assert replace(lane, boundary_physical_attempts=1).budget["maximum_physical_attempts"] == 7


def test_real_w13_two_slot_parent_uses_same_imported_round_and_full_original_budget(tmp_path, monkeypatch):
    lane = gpu.SCOPES[887, 14]
    parent_scope = gpu.SCOPES[887, 13]

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
    ] + [{"run":tmp_path/old.identity.name, "receipts":[]} for old in reversed(lane.archive[:-1])]
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

def test_w17_keeps_the_original_r34_payload_and_six_authenticated_failed_scopes():
    previous = gpu.SCOPES[887, 16]
    current = gpu.SCOPES[887, 17]
    latest = current.archive[-1]
    assert current.archive[:-1] == previous.archive
    assert sum(len(review.attempts) for review in current.archive) == 12
    assert current.payload_sha256 == previous.payload_sha256
    common = ("round_34/source.json", "round_34/boundary_memory.json",
              "round_34/boundary_user_message.json", "round_34/authoritative_packet.json")
    prior_files, current_files = dict(previous.fitted_prefix_files), dict(current.fitted_prefix_files)
    assert all(prior_files[name] == current_files[name] for name in common)
    assert current_files["boundary.json"] == "8e63b57d68e3d34743accd36c74e239632bb8a873492975043ce5c2e2169da44"
    assert current_files["round_34/turn_00/request.json"] == "2db15b66c6890ae0d7776665265b56427bc908703a32b7baa9b4fd7546d81936"
    assert current.recovery_budget == previous.recovery_budget
    assert current.budget == previous.budget
    assert current.not_before_unix == 1795740736.874102
    assert [attempt.attempt_id for attempt in latest.attempts] == [
        "697b22a230a54c5698a1a80139ae01f0",
        "1e1b11df64084833b167671f6f3b2894",
    ]
    assert all(attempt.progress["provider_error_code"] == "upstream_error"
               and attempt.progress["sse_event_count"] == 41 for attempt in latest.attempts)
    assert latest.replay_files == {
        "attempt_00.replay_authorization.json": "a65f6cde4c4b8cad9eb3c32dea881c28876fde575827cf1176e6703d3972d9e6",
        "attempt_00.replay.terminal.json": "44b1c197e0626b55cd58367dfc49a4fc16c8b3dc33fb2483c618ac49933f3c81",
        "attempt_01.replay_link.json": "c1027bfdf0f757573b1b45ee07e9805843ca86e3e1cd9f8c0868c5a6ea67e546",
    }

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor as gpu


@dataclass
class Parent:
    digest: str
    predecessor: dict
    profile: dict
    transport: object
    rounds: int
    source_sha256: dict | None = None
    run_directory: str = ""


@dataclass
class State:
    round: int


@pytest.fixture
def predecessor():
    return Parent(
        gpu._887_R33_CONTINUATION.contract,
        {"dispatcher_module":"dispatch_v13_887_r33_continuation",
         **{key:value for key,value in gpu._887_R33_REVIEWED_PARENT.items() if key.endswith("budget")}},
        {"full_imported_boundary_round":33, "replay_eligible_from_round":33, "sealed_relay_retry_seconds":180,
         "resume_boundaries":{"887":{"round":33}}},
        SimpleNamespace(logical_replies=3, transport_attempts=2),
        36,
    )


def test_exact_r34_slots_and_same_body():
    lane = gpu.SCOPES[887, 12]
    assert (lane.seed,lane.round,lane.original_start_round,lane.replay_start_round) == (887,34,33,34)
    assert lane.recovery_budget == {"maximum_logical_requests":1,"maximum_physical_attempts":2}
    assert lane.budget == {"first_logical_requests":2,"maximum_logical_requests":4,"maximum_physical_attempts":8}
    assert lane.not_before_unix == 1795734464.15188
    review, = lane.archive
    first,last = review.attempts
    assert first.identity == last.identity == gpu._887_R33_CONTINUATION
    assert first.request_id == last.request_id == "7825e8ef5b9181aeb184569692dadbe056e2dd63c9b33e00402529c4b5119e52"
    assert lane.payload_sha256 == "eb9170128722472e627a9ae16fa23692c1fc19c3ed6156513bde1c98e0c45a91"
    assert [item.terminal_file_sha256 for item in review.attempts] == [
        "d3a769a01f8a051033ec04170123c00d0f0d48ebe3ecae90206abee50ed8391b",
        "8d40cc02f116d35e031a9d9f27a3d60875c59f58cd7877fd88155db544ac94dc",
    ]
    for item in review.attempts:
        assert (item.failure,item.progress["provider_error_code"],item.progress["conflicting_event_type"]) == (
            "response_id_mismatch","upstream_error","response.failed",
        )
        assert "response.output_text.delta" not in item.progress["sse_milestones"]
        assert "response.completed" not in item.progress["sse_milestones"]
    assert dict(lane.required_sources)[(gpu.STUDY / "a7_gpu/seed_runner.py").as_posix()].startswith("417f4125")


def test_full_round_origin_is_only_enabled_by_exact_parent(predecessor):
    lane=gpu.SCOPES[887,12]
    assert gpu._reviewed_r33_parent_budget(predecessor, lane) == {
        "round":33,
        "recovery_budget":{"maximum_logical_requests":3,"maximum_physical_attempts":2},
        "execution_budget":{"first_logical_requests":3,"maximum_logical_requests":9,"maximum_physical_attempts":18},
    }
    for foreign in (replace(lane, wave=13),replace(lane, round=33),replace(lane, original_start_round=32)):
        with pytest.raises(PermissionError, match="no reviewed successor"):
            gpu._reviewed_r33_parent_budget(predecessor, foreign)
    for wrong in (None, True, 34):
        changed=replace(predecessor,profile={**predecessor.profile,"full_imported_boundary_round":wrong})
        with pytest.raises(PermissionError, match="normal boundary"):
            gpu._reviewed_r33_parent_budget(changed,lane)
    normal=replace(predecessor,digest="3"*64,profile={})
    assert gpu._reviewed_r33_parent_budget(normal,gpu.SCOPES[887,11]) is None


def test_w12_rebinds_verified_state_and_renews_only_r34_slots(predecessor, tmp_path, monkeypatch):
    lane=gpu.SCOPES[887,12]
    predecessor.predecessor.update(cache_directory=str(tmp_path),public_resource_receipt_sha256="sealed",
                                 bridge_wait_seconds=4200)
    old=tmp_path/lane.archive[-1].identity.name
    run=tmp_path/lane.run_name
    release=tmp_path/"release"
    old_release=tmp_path/"old_release"
    proof={"state_sha256":"state","memory_sha256":"memory"}
    boundary=(State(34),{},34)
    binding={field:"a"*64 for field in ("checkpoint_sha256","packet_sha256","wire_packet_sha256","base_messages_sha256",gpu.retry.ALIAS)}
    user={"role":"user","content":"exact prior input"}
    request={"request_id":lane.archive[-1].request_id,"payload_sha256":lane.payload_sha256,"binding":binding,
             "payload":{"input":[{"role":"system","content":"normal"},user]}}
    staged=SimpleNamespace(binding=binding,messages=request["payload"]["input"])
    records=[{"run":old,"directory":tmp_path,"active":tmp_path,"contract":predecessor,"approval":{},
              "request":request,"receipts":[],"terminal":{"not_before_unix":lane.cooldown_unix}}]
    pinned=dict(lane.fitted_prefix_files)
    monkeypatch.setattr(gpu,"_archives",lambda *_:records)
    monkeypatch.setattr(gpu,"file_digest",lambda p:pinned.get(p.relative_to(tmp_path).as_posix(),"sealed"))
    monkeypatch.setattr(gpu,"_reviewed_boundary",lambda *_:(boundary,proof))
    monkeypatch.setattr(gpu.seed_runner,"_scope",lambda *_:{})
    monkeypatch.setattr(gpu.seed_runner,"_stage",lambda *_:staged)
    monkeypatch.setattr(gpu.seed_runner,"pin_boundary_user_message",lambda a,*_:a)
    monkeypatch.setattr(gpu,"make_request",lambda *_:request)
    monkeypatch.setattr(gpu.retry,"_compare_science",lambda plan,old,host:plan==old and host is None)
    monkeypatch.setattr(gpu,"_source_manifest",lambda *_:{"source":"sha"})
    new, kept, _, body = gpu.construct(release,old_release,old,run,lane)
    assert kept == boundary and body is request
    assert new.profile["replay_eligible_from_round"] == 34
    assert "full_imported_boundary_round" not in new.profile
    assert new.profile["resume_boundaries"]["887"]["user_message_sha256"] == digest(user)
    assert new.predecessor["reviewed_normal_parent_budget"] == gpu._887_R33_REVIEWED_PARENT
    assert new.predecessor["recovery_budget"] == lane.recovery_budget
    assert new.predecessor["execution_budget"] == lane.budget
    wrong={**predecessor.predecessor,"recovery_budget":{"maximum_logical_requests":1,"maximum_physical_attempts":2}}
    records[0]["contract"]=replace(predecessor,predecessor=wrong)
    with pytest.raises(PermissionError, match="retired physical budget"):
        gpu.construct(release,old_release,old,run,lane)

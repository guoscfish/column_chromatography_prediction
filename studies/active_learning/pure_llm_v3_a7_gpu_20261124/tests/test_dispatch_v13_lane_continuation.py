"""Verify both lane continuations require a sealed, correctly bound CUDA handover."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM, ProviderReply
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import bridge_endpoint
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import SCHEMA
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_lane_continuation as follow
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_dispatch_v13_lane_extension import (
    lane_case, staged,  # noqa: F401
)


@pytest.fixture
def handover(lane_case, staged, monkeypatch):
    lane = lane_case.lane
    parent = lane_case.scope["parent_contract_sha256"]
    parent_auth = lane_case.scope["parent_authorization_sha256"]
    monkeypatch.setattr(follow, "PARENT_CONTRACT", parent)
    monkeypatch.setattr(follow, "PARENT_AUTHORIZATION", parent_auth)
    extension_sha = lane_case.args.contract_sha256
    extension_auth = lane_case.args.authorization_sha256
    spool = staged.spools / extension_sha
    spool.mkdir(mode=0o700)
    once(spool / "run_binding.json", {"contract_sha256": extension_sha, "authorization_sha256": extension_auth, "mock": False})
    job = lane_case.job
    reconstruction = {}
    if lane.seed == 887:
        request = job["request"]
        binding = request["binding"]
        host_hash = follow.AUTHORITATIVE_MESSAGE_HASH
        rebuilt = "7" * 64
        reconstruction = {"authoritative_message_reconstruction": {
            "archived_sha256": binding[host_hash], "rebuilt_sha256": rebuilt,
        }}
        job = staged.make(binding["a7_formal"], lane_case.transport, job["attempt_id"], wire=request["payload"]["input"],
                          state={**binding, host_hash: rebuilt})
    identity = job["attempt_id"]
    filename = f"{identity}.json"
    once(spool / "jobs" / filename, job)
    once(spool / follow.SELECTION, {"attempt_id": identity, "job_sha256": job["job_sha256"]})
    usage = {"input_tokens": 20, "output_tokens": 5}
    reply = ProviderReply("{}", "resp_fixture", lane_case.transport.served_model, usage)
    raw = {"id": reply.response_id, "model": reply.served_model, "status": "completed", "usage": usage,
           "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
    result = make_result(job, reply=reply, credential_id="key_01", model_identity=staged.model, raw_response=raw)
    once(spool / "results" / filename, result)
    once(spool / "delivered" / filename, {"attempt_id": identity, "result_sha256": result["result_sha256"]})
    once(spool / "provider_model_lock.json", staged.model)
    once(spool / "attempts" / identity / "raw_provider_response.json", raw)
    once(spool / "attempts" / identity / "provider_capture_receipt.json", {
        "attempt_id": identity, "job_sha256": job["job_sha256"], "metadata": {"raw_response_sha256": digest(raw)},
    })
    initial = lane.round_index + 1
    remaining = {6101: 34, 887: 29}[lane.seed]
    budget = {"first_logical_requests": remaining, "maximum_logical_requests": remaining * 3,
              "maximum_physical_attempts": remaining * 6}
    boundary = {"round": initial, "state_sha256": "1" * 64, "memory_sha256": "2" * 64}
    checkpoint, transition = "3" * 64, "4" * 64
    marker = {"contract_sha256": extension_sha, "seed": lane.seed, "round": initial,
              "active_labels": {6101: 397, 887: 557}[lane.seed], "checkpoint_sha256": checkpoint,
              "state_sha256": boundary["state_sha256"], "memory_sha256": boundary["memory_sha256"],
              "transition_sha256": transition, "original_future_budget": budget}
    run = lane_case.run.with_name(f"{follow.PARENT_RUN}-{lane.seed}-continuation-01")
    predecessor = {"dispatcher_module": follow.MODULE, "parent_run_directory": str(run.with_name(follow.PARENT_RUN)),
                   "parent_contract_sha256": parent, "parent_authorization_sha256": parent_auth,
                   "parent_seed_pause_sha256": "8" * 64, "extension_run_directory": str(lane_case.run),
                   "extension_contract_sha256": extension_sha, "extension_authorization_sha256": extension_auth,
                   "seed": lane.seed, "round": initial, "execution_budget": budget, "bridge_wait_seconds": 6000,
                   "checkpoint_sha256": checkpoint, "extension_transition_sha256": transition,
                   "extension_completion_sha256": digest(marker), "extension_fit_audit_sha256": "5" * 64,
                   "extension_attempt_id": identity, "extension_job_sha256": job["job_sha256"],
                   "extension_result_sha256": result["result_sha256"], "model_identity": staged.model,
                   **reconstruction}
    transport = replace(lane_case.transport, logical_replies=3, transport_attempts=2)
    extension_profile = bridge_endpoint.exchange(lane_case.run, "info")["contract"]["profile"]
    profile = {**extension_profile, "target_labels": 1485, "transport": asdict(transport),
               "resume_boundaries": {str(lane.seed): boundary}}
    record = {"profile": profile, "predecessor": predecessor, "run_directory": str(run), "budget": budget,
              "source_sha256": {path.as_posix(): file_digest(follow.ROOT / path) for path in follow.SOURCES},
              "api_execution": follow.SSH_API, "gpu_slots": [0]}
    info = {"schema": SCHEMA, "contract": record, "contract_sha256": digest(record), "authorization_sha256": "6" * 64,
            "mock": False, "paused": False}
    args = SimpleNamespace(seed=lane.seed, extension_contract_sha256=extension_sha, extension_authorization_sha256=extension_auth,
                           contract_sha256=info["contract_sha256"], authorization_sha256=info["authorization_sha256"],
                           ssh_host="offline", remote_root="/fixture", remote_run_dir=str(run), remote_python="/python")
    courier = SimpleNamespace(info=lambda: info)
    pilot = replace(transport, wire_mode=RESPONSES_STREAM)
    monkeypatch.setattr(follow, "SSHCourier", lambda *_: courier)
    monkeypatch.setattr(follow, "verify_local_source", lambda *_, **__: None)
    monkeypatch.setattr(follow, "sdk_status", lambda _: {"matches": True})
    monkeypatch.setattr(follow, "_approved_prompt_sha", lambda _: staged.binding["prompt_sha256"])
    monkeypatch.setattr(follow, "_pool", lambda _: (staged.pool, None, pilot))
    return SimpleNamespace(args=args, courier=courier, info=info, spool=spool, transport=transport, identity=identity)


def test_exact_boundary_and_pool_after_delivered_extension(handover, lane_case, staged, monkeypatch):
    def check(channel, courier, *, watch, workers):
        assert courier is handover.courier and watch and workers == 1
        assert channel.path == staged.spools / handover.args.contract_sha256
        assert channel.pool.directory == staged.pool.directory and channel.pool.config.parallelism == 3
        assert channel.transport == handover.transport
        request = lane_case.job["request"]
        proof = handover.info["contract"]["predecessor"]
        messages = deepcopy(request["payload"]["input"])
        packet = json.loads(messages[1]["content"])
        packet["round"] = proof["round"]
        messages[1]["content"] = dumps(packet)
        state = {**request["binding"], "round": proof["round"], "checkpoint_sha256": proof["checkpoint_sha256"],
                 "wire_packet_sha256": digest(packet), "base_messages_sha256": digest(messages)}
        scope = {"contract_sha256": handover.args.contract_sha256, "authorization_sha256": handover.args.authorization_sha256,
                 "seed": lane_case.lane.seed, "mock": False}
        job = staged.make(scope, handover.transport, "8" * 32, wire=messages, state=state)
        assert channel.validate(job)[0]["binding"]["checkpoint_sha256"] == proof["checkpoint_sha256"]
        bad = staged.make(scope, handover.transport, "8" * 32, wire=messages, state={**state, "checkpoint_sha256": "0" * 64})
        with pytest.raises(PermissionError, match="verified checkpoint"):
            channel.validate(bad)
        return {"verified": True}

    monkeypatch.setattr(follow, "run_dispatcher", check)
    assert follow.run(handover.args) == {"verified": True}


def test_rejects_mismatched_extension_authorization_before_opening_pool(handover, monkeypatch):
    handover.args.extension_authorization_sha256 = "9" * 64
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("a different authorization opened the shared keys"))
    with pytest.raises(PermissionError, match="authorized lane"):
        follow.run(handover.args)


def test_missing_acknowledgment_prevents_continuation(handover, monkeypatch):
    (handover.spool / "delivered" / f"{handover.identity}.json").unlink()
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("an undelivered response opened the shared keys"))
    with pytest.raises(FileNotFoundError):
        follow.run(handover.args)


def test_no_implicit_exception_for_a_host_message_digest(handover, lane_case, monkeypatch):
    info = handover.info
    scope = info["contract"]["predecessor"]
    key = "authoritative_message_reconstruction"
    if lane_case.lane.seed == 887:
        scope.pop(key)
    else:
        scope[key] = {"archived_sha256": lane_case.original["binding"][follow.AUTHORITATIVE_MESSAGE_HASH],
                      "rebuilt_sha256": "7" * 64}
    info["contract_sha256"] = handover.args.contract_sha256 = digest(info["contract"])
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("an unauthorized digest exception opened the shared keys"))
    with pytest.raises(PermissionError, match="signed seed 887 scope|exact original payload"):
        follow.run(handover.args)

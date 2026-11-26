"""Verify the continuation can only use the sealed L397 handover."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM, ProviderReply
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import SCHEMA
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prepared_ipv4_continue_2357 as follow
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_dispatch_prepared_ipv4_recovery_2357 import (
    staged,  # noqa: F401
)


@pytest.fixture
def handover(staged, monkeypatch):
    extension_sha = staged.args.contract_sha256
    extension_auth = staged.args.authorization_sha256
    monkeypatch.setattr(follow, "EXTENSION_CONTRACT", extension_sha)
    monkeypatch.setattr(follow, "EXTENSION_AUTHORIZATION", extension_auth)
    spool = staged.spools / extension_sha
    spool.mkdir(mode=0o700)
    once(spool / "run_binding.json", {"contract_sha256": extension_sha, "authorization_sha256": extension_auth, "mock": False})
    job = staged.job
    identity = staged.identity
    filename = f"{identity}.json"
    once(spool / "jobs" / filename, job)
    once(spool / follow.SELECTION, {"attempt_id": identity, "job_sha256": job["job_sha256"]})
    usage = {"input_tokens": 20, "output_tokens": 5}
    reply = ProviderReply("{}", "resp_fixture", staged.transport.served_model, usage)
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
    boundary = {"round": follow.START_ROUND, "state_sha256": "1" * 64, "memory_sha256": "2" * 64}
    checkpoint = "3" * 64
    transition = "4" * 64
    marker = {"contract_sha256": extension_sha, "seed": follow.SEED, "round": follow.START_ROUND,
              "active_labels": 397, "checkpoint_sha256": checkpoint, "state_sha256": boundary["state_sha256"],
              "memory_sha256": boundary["memory_sha256"], "transition_sha256": transition,
              "original_future_budget": follow.BUDGET}
    run = staged.run.with_name(follow.RUN)
    predecessor = {"dispatcher_module": follow.MODULE, "parent_run_directory": str(run.with_name(follow.PARENT_RUN)),
                   "parent_contract_sha256": follow.PARENT_CONTRACT, "parent_authorization_sha256": follow.PARENT_AUTHORIZATION,
                   "extension_run_directory": str(run.with_name(follow.EXTENSION_RUN)),
                   "extension_contract_sha256": extension_sha, "extension_authorization_sha256": extension_auth,
                   "seed": follow.SEED, "round": follow.START_ROUND, "execution_budget": follow.BUDGET,
                   "checkpoint_sha256": checkpoint, "extension_transition_sha256": transition,
                   "extension_completion_sha256": digest(marker), "extension_fit_audit_sha256": "5" * 64,
                   "extension_attempt_id": identity, "extension_job_sha256": job["job_sha256"],
                   "extension_result_sha256": result["result_sha256"], "model_identity": staged.model}
    transport = replace(staged.transport, logical_replies=3, transport_attempts=2)
    profile = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333,
               "target_labels": 1485, "batch_size": 32, "api_parallelism": 1, "store": False,
               "confirmation_seeds": [follow.SEED], "qualification_seed": follow.SEED, "requires_initial_qualification": False,
               "wire_representation": follow.FORMAL_WIRE_REPRESENTATION, "resume_boundaries": {str(follow.SEED): boundary},
               "transport": asdict(transport), "request_timeouts": asdict(follow.TIMEOUTS)}
    record = {"profile": profile, "predecessor": predecessor, "run_directory": str(run), "budget": follow.BUDGET,
              "source_sha256": {path.as_posix(): file_digest(follow.ROOT / path) for path in follow.SOURCES},
              "api_execution": follow.SSH_API, "gpu_slots": [0]}
    info = {"schema": SCHEMA, "contract": record, "contract_sha256": digest(record), "authorization_sha256": "6" * 64,
            "mock": False, "paused": False}
    args = SimpleNamespace(contract_sha256=info["contract_sha256"], authorization_sha256=info["authorization_sha256"],
                           ssh_host="offline", remote_root="/fixture", remote_run_dir=str(run), remote_python="/python")
    courier = SimpleNamespace(info=lambda: info)
    pilot = replace(transport, timeout_seconds=1800, wire_mode=RESPONSES_STREAM)
    monkeypatch.setattr(follow, "SSHCourier", lambda *_: courier)
    monkeypatch.setattr(follow, "verify_local_source", lambda *_, **__: None)
    monkeypatch.setattr(follow, "sdk_status", lambda _: {"matches": True})
    monkeypatch.setattr(follow, "_approved_prompt_sha", lambda _: staged.binding["prompt_sha256"])
    monkeypatch.setattr(follow, "_pool", lambda _: (staged.pool, None, pilot))
    return SimpleNamespace(args=args, courier=courier, info=info, spool=spool, transport=transport)


def test_verified_handover_uses_existing_global_pool_and_one_worker(handover, staged, monkeypatch):
    def check(channel, courier, *, watch, workers):
        assert courier is handover.courier and watch and workers == 1
        assert channel.path == staged.spools / handover.args.contract_sha256
        assert channel.pool.directory == staged.pool.directory and channel.pool.config.parallelism == 3
        assert channel.transport == handover.transport
        proof = handover.info["contract"]["predecessor"]
        messages = deepcopy(staged.job["request"]["payload"]["input"])
        packet = json.loads(messages[1]["content"])
        packet["round"] = follow.START_ROUND
        messages[1]["content"] = dumps(packet)
        state = {**staged.binding, "round": follow.START_ROUND, "checkpoint_sha256": proof["checkpoint_sha256"],
                 "wire_packet_sha256": digest(packet), "base_messages_sha256": digest(messages)}
        scope = {"contract_sha256": handover.args.contract_sha256, "authorization_sha256": handover.args.authorization_sha256,
                 "seed": follow.SEED, "mock": False}
        job = staged.make(scope, handover.transport, "8" * 32, wire=messages, state=state)
        assert channel.validate(job)[0]["binding"]["checkpoint_sha256"] == proof["checkpoint_sha256"]
        bad = staged.make(scope, handover.transport, "8" * 32, wire=messages, state={**state, "checkpoint_sha256": "0" * 64})
        with pytest.raises(PermissionError, match="verified checkpoint"):
            channel.validate(bad)
        return {"verified": True}

    monkeypatch.setattr(follow, "run_dispatcher", check)
    assert follow.run(handover.args) == {"verified": True}


def test_invalid_transition_marker_never_opens_key_pool(handover, monkeypatch):
    info = handover.info
    info["contract"]["predecessor"]["extension_transition_sha256"] = "f" * 64
    info["contract_sha256"] = handover.args.contract_sha256 = digest(info["contract"])
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("a forged handover opened the shared keys"))
    with pytest.raises(PermissionError, match="completion marker"):
        follow.run(handover.args)


def test_missing_local_delivery_never_opens_key_pool(handover, staged, monkeypatch):
    (handover.spool / "delivered" / f"{staged.identity}.json").unlink()
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("an undelivered extension opened the shared keys"))
    with pytest.raises(FileNotFoundError):
        follow.run(handover.args)

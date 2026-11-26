"""Verify the Mac handoff from 2357's exact delivered R2 response."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import ProviderReply, RESPONSES_STREAM
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import SCHEMA
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_2357_r2_continuation as follow
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_dispatch_v13_2357_r2_sse_extension import (
    r2, staged,  # noqa: F401
)


@pytest.fixture
def handoff(r2, staged, monkeypatch):
    contract, auth = r2.args.contract_sha256, r2.args.authorization_sha256
    job = read(next((r2.run / "bridge/outbox").glob("*.json")))
    identity = job["attempt_id"]
    name = f"{identity}.json"
    spool = staged.spools / contract
    spool.mkdir(mode=0o700)
    once(spool / "run_binding.json", {"contract_sha256": contract, "authorization_sha256": auth, "mock": False})
    once(spool / "jobs" / name, job)
    once(spool / follow.SELECTION, {"attempt_id": identity, "job_sha256": job["job_sha256"]})
    once(spool / "provider_model_lock.json", staged.model)
    usage = {"input_tokens": 20, "output_tokens": 5}
    reply = ProviderReply("{}", "resp_fixture", r2.transport.served_model, usage)
    raw = {"id": reply.response_id, "model": reply.served_model, "status": "completed", "usage": usage,
           "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
    result = make_result(job, reply=reply, credential_id="key_01", model_identity=staged.model, raw_response=raw)
    once(spool / "results" / name, result)
    ack = spool / "delivered" / name
    once(ack, {"attempt_id": identity, "result_sha256": result["result_sha256"]})
    attempt = spool / "attempts" / identity
    once(attempt / "raw_provider_response.json", raw)
    once(attempt / "provider_capture_receipt.json", {
        "attempt_id": identity, "job_sha256": job["job_sha256"], "metadata": {"raw_response_sha256": digest(raw)},
    })
    transport = replace(r2.transport, logical_replies=3, transport_attempts=2)
    run = r2.run.with_name(follow.RUN)
    boundary = {"round": 3, "state_sha256": "1" * 64, "memory_sha256": "2" * 64}
    marker = {"contract_sha256": contract, "seed": 2357, "round": 3, "active_labels": 429,
              "checkpoint_sha256": "3" * 64, "state_sha256": "1" * 64, "memory_sha256": "2" * 64,
              "transition_sha256": "4" * 64, "original_future_budget": follow.BUDGET}
    prior = {"dispatcher_module": follow.MODULE, "parent_run_directory": str(run.with_name(follow.PARENT_RUN)),
             "parent_contract_sha256": follow.PARENT_CONTRACT, "parent_authorization_sha256": follow.PARENT_AUTHORIZATION,
             "parent_seed_pause_sha256": "5" * 64, "extension_run_directory": str(r2.run),
             "extension_contract_sha256": contract, "extension_authorization_sha256": auth,
             "seed": 2357, "round": 3, "execution_budget": follow.BUDGET, "bridge_wait_seconds": 7800,
             "checkpoint_sha256": marker["checkpoint_sha256"], "extension_transition_sha256": marker["transition_sha256"],
             "extension_completion_sha256": digest(marker), "extension_fit_audit_sha256": "6" * 64,
             "extension_attempt_id": identity, "extension_job_sha256": job["job_sha256"],
             "extension_result_sha256": result["result_sha256"], "model_identity": staged.model}
    profile = {**read(r2.run / "formal_contract.json")["profile"], "target_labels": 1485, "transport": asdict(transport),
               "resume_boundaries": {"2357": boundary}}
    record = {"profile": profile, "predecessor": prior, "run_directory": str(run), "budget": follow.BUDGET,
              "source_sha256": {path.as_posix(): file_digest(follow.ROOT / path) for path in follow.SOURCES},
              "api_execution": follow.SSH_API, "gpu_slots": [0]}
    info = {"schema": SCHEMA, "contract": record, "contract_sha256": digest(record), "authorization_sha256": "7" * 64,
            "mock": False, "paused": False}
    args = SimpleNamespace(contract_sha256=info["contract_sha256"], authorization_sha256=info["authorization_sha256"],
                           extension_contract_sha256=contract, extension_authorization_sha256=auth,
                           ssh_host="offline", remote_root="/fixture", remote_run_dir=str(run), remote_python="/python")
    courier = SimpleNamespace(info=lambda: info)
    pilot = replace(transport, wire_mode=RESPONSES_STREAM)
    monkeypatch.setattr(follow, "PARENT_PAYLOAD_SHA", r2.original["payload_sha256"])
    monkeypatch.setattr(follow, "SSHCourier", lambda *_: courier)
    monkeypatch.setattr(follow, "verify_local_source", lambda *_, **__: None)
    monkeypatch.setattr(follow, "sdk_status", lambda _: {"matches": True})
    monkeypatch.setattr(follow, "_approved_prompt_sha", lambda _: staged.binding["prompt_sha256"])
    monkeypatch.setattr(follow, "_pool", lambda _: (staged.pool, None, pilot))
    return SimpleNamespace(args=args, info=info, ack=ack, transport=transport, courier=courier)


def test_handoff_uses_shared_pool_and_approved_idle_policy(handoff, staged, monkeypatch):
    def check(channel, courier, *, watch, workers):
        assert courier is handoff.courier and watch and workers == 1
        assert channel.pool.directory == staged.pool.directory and channel.pool.config.parallelism == 3
        assert channel.transport == handoff.transport and channel.sse_inactivity_seconds == 900
        assert channel.info["contract"]["predecessor"]["checkpoint_sha256"] == "3" * 64
        return "verified"

    monkeypatch.setattr(follow, "run_dispatcher", check)
    assert follow.run(handoff.args) == "verified"


def test_missing_delivery_prevents_continuation(handoff, monkeypatch):
    handoff.ack.unlink()
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("missing receipt opened the Key Pool"))
    with pytest.raises(FileNotFoundError):
        follow.run(handoff.args)


def test_modified_resume_boundary_does_not_reach_pool(handoff, monkeypatch):
    handoff.info["contract"]["profile"]["resume_boundaries"]["2357"]["memory_sha256"] = "8" * 64
    handoff.info["contract_sha256"] = handoff.args.contract_sha256 = digest(handoff.info["contract"])
    monkeypatch.setattr(follow, "_pool", lambda _: pytest.fail("invalid handover opened the Key Pool"))
    with pytest.raises(PermissionError, match="sealed L429"):
        follow.run(handoff.args)

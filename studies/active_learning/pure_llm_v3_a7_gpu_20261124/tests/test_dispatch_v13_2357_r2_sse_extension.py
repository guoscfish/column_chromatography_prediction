"""Verify the second 2357 one-shot's exact custody and inactivity policy without API calls."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM, TransportFailure
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import bridge_endpoint, dispatcher
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_2357_r2_sse_extension as extension
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_dispatch_prepared_ipv4_recovery_2357 import (
    staged,  # noqa: F401
)


@pytest.fixture
def r2(staged, monkeypatch):
    parent = staged.spools / extension.PARENT_CONTRACT
    parent.mkdir(mode=0o700)
    once(parent / "run_binding.json", {"contract_sha256": extension.PARENT_CONTRACT,
                                       "authorization_sha256": extension.PARENT_AUTHORIZATION, "mock": False})
    once(parent / "provider_model_lock.json", staged.model)
    original_transport = replace(staged.transport, logical_replies=3, transport_attempts=2)
    messages = deepcopy(staged.original["payload"]["input"])
    packet = json.loads(messages[1]["content"])
    packet["round"] = extension.ROUND
    messages[1]["content"] = dumps(packet)
    state = {**staged.binding, "round": extension.ROUND, "wire_packet_sha256": digest(packet),
             "base_messages_sha256": digest(messages)}
    grant = {"seed": extension.SEED, "mock": False, "contract_sha256": extension.PARENT_CONTRACT,
             "authorization_sha256": extension.PARENT_AUTHORIZATION}
    evidence = []
    for index, identity in enumerate(extension.PARENT_ATTEMPTS):
        job = staged.make(grant, original_transport, identity, ordinal=index, wire=messages, state=state)
        posted = index == 0
        error = "timeout" if posted else "connect_error"
        phase = "ReadTimeout" if posted else "direct_ipv4_preflight_failed"
        result = make_result(job, failure=TransportFailure(error, ambiguous=posted, retryable=not posted),
                             provider_dispatched=posted, credential_id="key_01")
        once(parent / "jobs" / f"{identity}.json", job)
        once(parent / "results" / f"{identity}.json", result)
        once(parent / "delivered" / f"{identity}.json", {"attempt_id": identity, "result_sha256": result["result_sha256"]})
        once(parent / "attempts" / identity / "provider_progress.json", {
            "attempt_id": identity, "job_sha256": job["job_sha256"], "error_type": phase,
            "http_status": 200 if posted else None,
        })
        evidence.append({"attempt": index, "attempt_id": identity, "job_sha256": job["job_sha256"], "error": error,
                         "provider_dispatched": posted, "ambiguous": posted, "result_sha256": result["result_sha256"],
                         "terminal_sha256": "1" * 64})
    original = job["request"]
    monkeypatch.setattr(extension, "PARENT_REQUEST", original["request_id"])
    monkeypatch.setattr(extension, "PARENT_PAYLOAD_SHA", original["payload_sha256"])
    run = staged.run.with_name(extension.RUN)
    scope = {"dispatcher_module": extension.MODULE, "parent_run_directory": str(run.with_name(extension.PARENT_RUN)),
             "parent_contract_sha256": extension.PARENT_CONTRACT, "parent_authorization_sha256": extension.PARENT_AUTHORIZATION,
             "seed": extension.SEED, "round": extension.ROUND, "logical_index": 0,
             "prior_request_id": original["request_id"], "payload_sha256": original["payload_sha256"],
             "prior_attempts": evidence, "model_identity": staged.model, "bridge_wait_seconds": 7800,
             "parent_seed_pause_sha256": "2" * 64, "prior_replay_authorization_sha256": "3" * 64,
             "prior_replay_link_sha256": "4" * 64, **{key: state[key] for key in extension.BOUND_HASHES}}
    transport = staged.transport
    profile = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
               "initial_labels": 333, "target_labels": 429, "batch_size": 32, "api_parallelism": 1,
               "store": False, "confirmation_seeds": [extension.SEED], "qualification_seed": extension.SEED,
               "requires_initial_qualification": False, "wire_representation": extension.FORMAL_WIRE_REPRESENTATION,
               "sse_inactivity_seconds": 900, "request_timeouts": asdict(extension.TIMEOUTS), "transport": asdict(transport),
               "resume_boundaries": {str(extension.SEED): {"round": extension.ROUND}}}
    record = {"profile": profile, "predecessor": scope, "run_directory": str(run), "budget": extension.BUDGET,
              "source_sha256": {path.as_posix(): file_digest(extension.ROOT / path) for path in extension.SOURCES},
              "api_execution": extension.SSH_API, "gpu_slots": [0]}
    contract = digest(record)
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": contract, "authorized_by": "fixture",
                "authorization_reference": "test", "authorize_paid_api": True, "authorize_label_reveal": True,
                "authorize_cuda_training": True}
    auth = digest(approval)
    grant = {"seed": extension.SEED, "mock": False, "contract_sha256": contract, "authorization_sha256": auth}
    identity = "7" * 32
    job = staged.make(grant, transport, identity, wire=messages, state=state)
    turn = run / f"seed_{extension.SEED}/round_{extension.ROUND:02d}/turn_00"
    once(run / "formal_contract.json", record)
    once(run / "authorization.json", approval)
    once(run / "bridge/outbox" / f"{identity}.json", job)
    once(turn.parent / AUTHORITATIVE_PACKET_FILE, {"synthetic": "scientific packet"})
    once(turn / "request.json", job["request"])
    once(turn / "attempt_00.started.json", job["started"])
    once(turn / "formal_request_authorization.json", {
        "scope": "paid_selection_only", "formal_run": grant, "request_id": job["request_id"],
    })
    pilot = replace(original_transport, timeout_seconds=1800, wire_mode=RESPONSES_STREAM)
    monkeypatch.setattr(extension, "verify_local_source", lambda *_, **__: None)
    monkeypatch.setattr(extension, "sdk_status", lambda _: {"matches": True})
    monkeypatch.setattr(extension, "_approved_prompt_sha", lambda _: state["prompt_sha256"])
    monkeypatch.setattr(extension, "_pool", lambda _: (staged.pool, None, pilot))
    args = SimpleNamespace(contract_sha256=contract, authorization_sha256=auth, wait_seconds=10, poll=0.01,
                           ssh_host="offline", remote_root="/fixture", remote_run_dir=str(run), remote_python="/python")
    return SimpleNamespace(args=args, parent=parent, run=run, scope=scope, original=original, transport=transport)


def test_one_mock_submission_uses_effective_sse_limit_and_recovers_after_delivery_interruption(r2, staged, monkeypatch):
    submitted, deliveries = [], []

    class LocalCourier:
        def __init__(self, *_):
            pass

        def info(self):
            return bridge_endpoint.exchange(r2.run, "info")

        def list(self):
            return bridge_endpoint.exchange(r2.run, "list")

        def get(self, identity):
            return bridge_endpoint.exchange(r2.run, "get", identity)

        def claim(self, identity, dispatcher_id):
            return bridge_endpoint.exchange(r2.run, "claim", identity, dispatcher_id=dispatcher_id)

        def put(self, identity, dispatcher_id, result):
            deliveries.append(result)
            if len(deliveries) == 1:
                raise InterruptedError("fixture stops after locally sealing the result")
            return bridge_endpoint.exchange(r2.run, "put", identity, result, dispatcher_id)

    def mock_provider(request, config, *, timeouts, sse_inactivity_seconds, **_):
        submitted.append(request)
        lease = LEASE.get()
        assert lease.pool.directory == staged.pool.directory and lease.pool.config.parallelism == 3
        assert request["payload"] == r2.original["payload"] and config == r2.transport
        assert (sse_inactivity_seconds, timeouts.read_idle_seconds, timeouts.total_seconds) == (900, 1800, 3600)
        lease.dispatched()
        lease.outcome("timeout")
        raise TransportFailure("timeout", ambiguous=True)

    monkeypatch.setattr(extension, "SSHCourier", LocalCourier)
    monkeypatch.setattr(dispatcher, "call_responses", mock_provider)
    with pytest.raises(InterruptedError):
        extension.run(r2.args)
    final = extension.run(r2.args)
    assert final["delivered"] and final["provider_dispatched"] and final["failure"] == "timeout"
    assert len(submitted) == 1 and deliveries[0] == deliveries[1]


def test_recovery_refuses_a_replay_with_posted_evidence_or_original_saved_text(r2):
    original, _, _ = extension._parent(r2.parent, r2.scope)
    assert original == r2.original
    changed = deepcopy(r2.scope)
    changed["prior_attempts"][1]["provider_dispatched"] = True
    with pytest.raises(PermissionError, match="no-POST replay"):
        extension._parent(r2.parent, changed)
    once(r2.parent / "attempts" / extension.PARENT_ATTEMPTS[0] / "raw_provider_response.json", {"status": "completed"})
    with pytest.raises(PermissionError, match="sealed Mac receipts"):
        extension._parent(r2.parent, r2.scope)


def test_recovery_needs_exact_effective_idle_and_replay_hash(r2):
    info = bridge_endpoint.exchange(r2.run, "info")
    assert extension._approved_scope(info, str(r2.run))["bridge_wait_seconds"] == 7800
    info["contract"]["predecessor"].pop("prior_replay_link_sha256")
    with pytest.raises(PermissionError, match="one-attempt"):
        extension._approved_scope(info, str(r2.run))

"""Check the lane-specific one-shot dispatch and receipt recovery offline."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM, TransportFailure
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import bridge_endpoint, dispatcher
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_lane_extension as extension
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_dispatch_prepared_ipv4_recovery_2357 import (
    staged,  # noqa: F401
)


@pytest.fixture(params=(6101, 887))
def lane_case(staged, request, monkeypatch):
    return _lane_case(staged, request.param, monkeypatch)


def _lane_case(staged, seed, monkeypatch):
    lane = extension.LANES[seed]
    parent = staged.parent.with_name("d" * 64)
    parent.mkdir(mode=0o700)
    monkeypatch.setattr(extension, "PARENT_CONTRACT", parent.name)
    parent_auth = "e" * 64
    monkeypatch.setattr(extension, "PARENT_AUTHORIZATION", parent_auth)
    once(parent / "run_binding.json", {"contract_sha256": parent.name, "authorization_sha256": parent_auth, "mock": False})
    once(parent / "provider_model_lock.json", staged.model)
    original_transport = replace(staged.transport, logical_replies=3, transport_attempts=2)
    transport = replace(staged.transport, timeout_seconds=1800)
    messages = deepcopy(staged.original["payload"]["input"])
    packet = json.loads(messages[1]["content"])
    packet.update(seed=lane.seed, round=lane.round_index)
    messages[1]["content"] = dumps(packet)
    state = {**staged.binding, "seed": lane.seed, "round": lane.round_index, "wire_packet_sha256": digest(packet),
             "base_messages_sha256": digest(messages)}
    evidence = []
    for index, identity in enumerate(lane.prior_attempts):
        grant = {"seed": lane.seed, "mock": False, "contract_sha256": parent.name, "authorization_sha256": parent_auth}
        job = staged.make(grant, original_transport, identity, ordinal=index, wire=messages, state=state)
        posted = index == 1
        error = lane.posted_error if posted else "connect_error"
        phase = lane.posted_transport_errors[0] if posted else "direct_ipv4_preflight_failed"
        stopped = TransportFailure(error, ambiguous=posted, retryable=not posted)
        result = make_result(job, failure=stopped, provider_dispatched=posted, credential_id="key_01")
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
    run = staged.run.with_name(lane.name)
    original = job["request"]
    scope = {"dispatcher_module": extension.MODULE, "parent_run_directory": str(run.with_name(extension.PARENT_RUN)),
             "parent_contract_sha256": parent.name, "parent_authorization_sha256": parent_auth,
             "bridge_wait_seconds": 6000,
             "seed": lane.seed, "round": lane.round_index, "logical_index": 0, "prior_request_id": original["request_id"],
             "payload_sha256": original["payload_sha256"], "prior_attempts": evidence, "model_identity": staged.model,
             **{key: state[key] for key in extension.BOUND_HASHES}}
    profile = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
               "initial_labels": 333, "target_labels": 333 + (lane.round_index + 1) * 32,
               "batch_size": 32, "api_parallelism": 1, "store": False, "confirmation_seeds": [lane.seed],
               "qualification_seed": lane.seed, "requires_initial_qualification": False,
               "wire_representation": extension.FORMAL_WIRE_REPRESENTATION, "request_timeouts": asdict(extension.TIMEOUTS),
               "transport": asdict(transport), "resume_boundaries": {str(lane.seed): {"round": lane.round_index}}}
    record = {"profile": profile, "predecessor": scope, "run_directory": str(run), "budget": extension.BUDGET,
              "source_sha256": {path.as_posix(): file_digest(extension.ROOT / path) for path in extension.SOURCES},
              "api_execution": extension.SSH_API, "gpu_slots": [0]}
    contract = digest(record)
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": contract, "authorized_by": "fixture",
                "authorization_reference": "test", "authorize_paid_api": True, "authorize_label_reveal": True,
                "authorize_cuda_training": True}
    auth = digest(approval)
    grant = {"seed": lane.seed, "mock": False, "contract_sha256": contract, "authorization_sha256": auth}
    identity = "7" * 32
    job = staged.make(grant, transport, identity, wire=messages, state=state)
    turn = run / f"seed_{lane.seed}/round_{lane.round_index:02d}/turn_00"
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
    args = SimpleNamespace(seed=lane.seed, contract_sha256=contract, authorization_sha256=auth, wait_seconds=10, poll=0.01,
                           ssh_host="offline", remote_root="/fixture", remote_run_dir=str(run), remote_python="/python")
    return SimpleNamespace(args=args, parent=parent, lane=lane, run=run, scope=scope,
                           original=original, job=job, transport=transport)


def test_lane_allows_one_mock_post_and_only_redelivers_on_restart(lane_case, staged, monkeypatch):
    requests, deliveries = [], []

    class LocalCourier:
        def __init__(self, *_):
            pass

        def info(self):
            return bridge_endpoint.exchange(lane_case.run, "info")

        def list(self):
            return bridge_endpoint.exchange(lane_case.run, "list")

        def get(self, identity):
            return bridge_endpoint.exchange(lane_case.run, "get", identity)

        def claim(self, identity, dispatcher_id):
            return bridge_endpoint.exchange(lane_case.run, "claim", identity, dispatcher_id=dispatcher_id)

        def put(self, identity, dispatcher_id, result):
            deliveries.append(result)
            if len(deliveries) == 1:
                raise InterruptedError("simulated interruption after local result seal")
            return bridge_endpoint.exchange(lane_case.run, "put", identity, result, dispatcher_id)

    def mock_provider(request, config, *, timeouts, **_):
        requests.append(request)
        lease = LEASE.get()
        assert lease.pool.directory == staged.pool.directory and lease.pool.config.parallelism == 3
        assert request["payload"] == lane_case.original["payload"] and config == lane_case.transport
        assert (timeouts.read_idle_seconds, timeouts.total_seconds) == (900, 1800)
        lease.dispatched()
        lease.outcome("timeout")
        raise TransportFailure("timeout", ambiguous=True)

    monkeypatch.setattr(extension, "SSHCourier", LocalCourier)
    monkeypatch.setattr(dispatcher, "call_responses", mock_provider)
    with pytest.raises(InterruptedError):
        extension.run(lane_case.args)
    result = extension.run(lane_case.args)
    assert result["delivered"] and result["provider_dispatched"] and result["failure"] == "timeout"
    assert len(requests) == 1 and deliveries[0] == deliveries[1]


def test_prior_attempt_scope_distinguishes_preflight_from_post(lane_case):
    original, _, _ = extension._parent(lane_case.parent, lane_case.lane, lane_case.scope)
    assert original == lane_case.original
    scope = deepcopy(lane_case.scope)
    scope["prior_attempts"][0]["provider_dispatched"] = True
    with pytest.raises(PermissionError, match="first preflight"):
        extension._parent(lane_case.parent, lane_case.lane, scope)


def test_scope_requires_full_mixed_queue_wait(lane_case):
    info = bridge_endpoint.exchange(lane_case.run, "info")
    assert extension._approved_scope(info, str(lane_case.run), lane_case.lane)["bridge_wait_seconds"] == 6000
    info["contract"]["predecessor"]["bridge_wait_seconds"] = 4200
    with pytest.raises(PermissionError, match="lane and timeouts"):
        extension._approved_scope(info, str(lane_case.run), lane_case.lane)


def test_157_lane_uses_progress_timeout_and_mixed_queue_wait():
    timeouts, inactivity, bridge_wait = extension._lane_timing(extension.LANES[157])
    assert (timeouts.read_idle_seconds, timeouts.total_seconds, inactivity, bridge_wait) == (1800, 3600, 900, 7800)


def test_default_wait_covers_an_original_v13_request_and_the_new_request(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "dispatch_v13_lane_extension", "--execute-paid", "--seed", "6101", "--remote-root", "/fixture",
        "--remote-run-dir", "/fixture/recovery", "--contract-sha256", "a" * 64, "--authorization-sha256", "b" * 64,
    ])
    monkeypatch.setattr(extension.os, "umask", lambda _: None)
    monkeypatch.setattr(extension, "run", lambda args: {"wait": args.wait_seconds})
    extension.main()
    assert json.loads(capsys.readouterr().out) == {"wait": 6000}


def test_only_887_can_use_exact_signed_host_reconstruction(lane_case, staged):
    host_hash = extension.AUTHORITATIVE_MESSAGE_HASH
    request = lane_case.job["request"]
    source = lane_case.original
    pair = {"archived_sha256": source["binding"][host_hash], "rebuilt_sha256": "7" * 64}
    messages = request["payload"]["input"]
    state = {**request["binding"], host_hash: pair["rebuilt_sha256"]}

    def job_for(binding, wire=messages):
        return staged.make(binding["a7_formal"], lane_case.transport, lane_case.job["attempt_id"], wire=wire, state=binding)

    def validate(job, reconstruction):
        return extension._validate(SimpleNamespace(validate=verify_job), job, lane_case.lane, source,
                                   state["prompt_sha256"], reconstruction)

    reconstructed = job_for(state)
    with pytest.raises(PermissionError):
        validate(reconstructed, None)
    if lane_case.lane.seed == 6101:
        with pytest.raises(PermissionError):
            validate(reconstructed, pair)
        return
    assert validate(reconstructed, pair) == reconstructed
    with pytest.raises(PermissionError):
        validate(reconstructed, {**pair, "rebuilt_sha256": "8" * 64})
    with pytest.raises(PermissionError):
        validate(job_for({**state, "checkpoint_sha256": "8" * 64}), pair)
    packet = json.loads(messages[1]["content"])
    packet["unexpected_payload_change"] = 1
    changed = [messages[0], {"role": "user", "content": dumps(packet)}]
    with pytest.raises(PermissionError):
        validate(job_for({**state, "wire_packet_sha256": digest(packet), "base_messages_sha256": digest(changed)}, changed), pair)


def test_157_must_have_a_sealed_timeout_and_no_saved_response(staged, monkeypatch):
    case = _lane_case(staged, 157, monkeypatch)
    lane = case.lane
    identity = lane.prior_attempts[1]
    result = case.parent / "results" / f"{identity}.json"
    parked = result.with_suffix(".not-yet-sealed")
    result.rename(parked)
    with pytest.raises(FileNotFoundError):
        extension._parent(case.parent, lane, case.scope)
    parked.rename(result)
    saved = case.parent / "attempts" / identity
    progress_file = saved / "provider_progress.json"
    progress = json.loads(progress_file.read_bytes())
    progress_file.write_text(dumps({**progress, "error_type": "ReadTimeout"}))
    original, _, _ = extension._parent(case.parent, lane, case.scope)
    assert original == case.original
    once(saved / "raw_provider_response.json", {"status": "completed"})
    with pytest.raises(PermissionError, match="unresolved posted attempt"):
        extension._parent(case.parent, lane, case.scope)

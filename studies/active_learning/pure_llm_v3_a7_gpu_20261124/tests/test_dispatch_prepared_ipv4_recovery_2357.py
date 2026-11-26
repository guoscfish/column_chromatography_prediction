"""Exercise the exact one-shot recovery handoff with local fixtures."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportFailure, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import bridge_endpoint, dispatcher
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE, FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE, KeyPool, PoolConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prepared_ipv4_recovery_2357 as recovery


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    repo = recovery.ROOT
    seed, round_index = recovery.SEED, recovery.ROUND
    run = tmp_path / recovery.RUN
    spools = tmp_path / ".config/purellm/spool"
    parent = spools / recovery.PARENT_CONTRACT
    parent.mkdir(parents=True, mode=0o700)
    parent_auth = "d" * 64
    once(parent / "run_binding.json", {"contract_sha256": recovery.PARENT_CONTRACT,
                                      "authorization_sha256": parent_auth, "mock": False})
    model = {"provider": "token4research", "requested_model": "gpt-6.2-astra", "served_model": "gpt-6.2-astra",
             "reported_model_version": None, "requested_reasoning_effort": "xhigh", "reported_reasoning_effort": "xhigh"}
    once(parent / "provider_model_lock.json", model)
    pilot = recovery.TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                                     "https://token4research.cn/v1", 1800, 32768, transport_attempts=2, sdk_version="2.28.0")
    old = replace(pilot, wire_mode=recovery.RESPONSES_STREAM_IPV4, timeout_seconds=3600)
    transport = replace(old, logical_replies=1, transport_attempts=1)
    representation = recovery.FORMAL_WIRE_REPRESENTATION
    authoritative = {"synthetic": "scientific packet"}
    prompt = "Synthetic A7 task."
    packet = {"seed": seed, "round": round_index, "layout": "B", "compression": representation["compression"],
              "memory": {"mode": "M2"}, "selection_count": 32, "wire_schema": representation["schema"],
              "source_compression": "lossless", "feature_precision": representation["feature_precision"],
              "crossing_guard": True, "derived": {}}
    messages = [{"role": "system", "content": prompt + FORMAT_SUFFIX}, {"role": "user", "content": dumps(packet)}]
    binding = {"seed": seed, "round": round_index, "arm": "A7", "checkpoint_sha256": "c" * 64,
               "packet_sha256": digest(authoritative), "mapping_sha256": "a" * 64, "state_sha256": "b" * 64,
               "wire_packet_sha256": digest(packet), "wire_representation": representation,
               "authoritative_base_messages_sha256": "f" * 64, "base_messages_sha256": digest(messages),
               "prompt_sha256": digest(prompt), "split_sha256": "e" * 64, "schema_version": SCHEMA_VERSION,
               "prompt_version": "v3"}

    def grant(contract, auth):
        return {"seed": seed, "mock": False, "contract_sha256": contract, "authorization_sha256": auth}

    def make(scope, config, identity, ordinal=0, wire=messages, state=binding):
        request = make_request({**state, "a7_formal": scope}, 0, wire, config)
        started = {"attempt_id": identity, "request_id": request["request_id"], "attempt": ordinal,
                   "started_unix": 1234, "transport": "mac_bridge"}
        return make_job(request, config, started)

    prior = []
    for ordinal, identity in enumerate(recovery.PARENT_ATTEMPTS):
        job = make(grant(recovery.PARENT_CONTRACT, parent_auth), old, identity, ordinal)
        result = make_result(job, failure=TransportFailure("timeout", ambiguous=True),
                             provider_dispatched=True, credential_id="key_01")
        once(parent / "jobs" / f"{identity}.json", job)
        once(parent / "results" / f"{identity}.json", result)
        once(parent / "delivered" / f"{identity}.json", {"attempt_id": identity, "result_sha256": result["result_sha256"]})
        prior.append({"attempt": ordinal, "attempt_id": identity, "job_sha256": job["job_sha256"],
                      "result_sha256": result["result_sha256"], "terminal_sha256": "1" * 64})
    original = job["request"]
    scope = {"dispatcher_module": recovery.MODULE, "parent_run_directory": str(run.with_name(recovery.PARENT)),
             "parent_contract_sha256": recovery.PARENT_CONTRACT, "parent_authorization_sha256": parent_auth,
             "seed": seed, "round": round_index, "logical_index": 0, "prior_request_id": original["request_id"],
             "payload_sha256": original["payload_sha256"], "prior_attempts": prior, "model_identity": model,
             **{key: binding[key] for key in recovery.BOUND_HASHES}}
    profile = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333,
               "target_labels": 397, "batch_size": 32, "api_parallelism": 1, "store": False, "confirmation_seeds": [seed],
               "qualification_seed": seed, "requires_initial_qualification": False, "wire_representation": representation,
               "resume_boundaries": {str(seed): {"round": round_index}}, "transport": asdict(transport),
               "request_timeouts": asdict(recovery.TIMEOUTS)}
    record = {"profile": profile, "predecessor": scope, "run_directory": str(run), "budget": recovery.BUDGET,
              "source_sha256": {path.as_posix(): file_digest(repo / path) for path in recovery.SOURCES},
              "api_execution": recovery.SSH_API, "gpu_slots": [0]}
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": digest(record), "authorized_by": "fixture",
                "authorization_reference": "test", "authorize_paid_api": True, "authorize_label_reveal": True,
                "authorize_cuda_training": True}
    auth = digest(approval)
    contract = digest(record)
    new_scope = grant(contract, auth)
    attempt = "9" * 32
    job = make(new_scope, transport, attempt)
    request = job["request"]
    turn = run / f"seed_{seed}/round_{round_index:02d}/turn_00"
    once(run / "formal_contract.json", record)
    once(run / "authorization.json", approval)
    once(run / "bridge/outbox" / f"{attempt}.json", job)
    once(turn.parent / AUTHORITATIVE_PACKET_FILE, authoritative)
    once(turn / "request.json", request)
    once(turn / "attempt_00.started.json", job["started"])
    once(turn / "formal_request_authorization.json", {"scope": "paid_selection_only", "formal_run": new_scope,
                                                       "request_id": request["request_id"]})
    paths = {}
    for index in range(1, 4):
        key = tmp_path / f"key_0{index}"
        key.touch(mode=0o600)
        key.write_text("synthetic-fixture-value")
        paths[key.name] = key
    pool = KeyPool(tmp_path / "shared-pool", PoolConfig(3, paths))
    pool.prepare()
    monkeypatch.setattr(recovery, "verify_local_source", lambda *_, **__: None)
    monkeypatch.setattr(recovery, "_approved_prompt_sha", lambda _: binding["prompt_sha256"])
    monkeypatch.setattr(recovery, "sdk_status", lambda _: {"matches": True})
    monkeypatch.setattr(recovery, "_pool", lambda path: (pool, None, pilot))
    args = SimpleNamespace(contract_sha256=contract, authorization_sha256=auth, wait_seconds=10, poll=0.01,
                           ssh_host="offline", remote_root=str(repo), remote_run_dir=str(run), remote_python="/python")
    return SimpleNamespace(args=args, spools=spools, parent=parent, scope=scope, run=run, identity=attempt, pool=pool,
                           original=original, job=job, transport=transport, make=make, binding=binding, model=model)


def test_one_mock_post_survives_delivery_interruption_and_restart(staged, monkeypatch):
    paid = []
    delivery = []

    class LocalCourier:
        def __init__(self, *_):
            pass

        def info(self):
            return bridge_endpoint.exchange(staged.run, "info")

        def list(self):
            return bridge_endpoint.exchange(staged.run, "list")

        def get(self, identity):
            return bridge_endpoint.exchange(staged.run, "get", identity)

        def claim(self, identity, dispatcher_id):
            return bridge_endpoint.exchange(staged.run, "claim", identity, dispatcher_id=dispatcher_id)

        def put(self, identity, dispatcher_id, result):
            delivery.append(result)
            if len(delivery) == 1:
                raise InterruptedError("simulated dispatcher interruption after local seal")
            return bridge_endpoint.exchange(staged.run, "put", identity, result, dispatcher_id)

    def mock_provider(request, config, *, capture_response, capture_receipt, **_):
        paid.append(request)
        assert request["payload"] == staged.original["payload"] and config == staged.transport
        lease = LEASE.get()
        assert lease.pool.directory == staged.pool.directory
        assert lease.pool.config.parallelism == 3
        lease.dispatched()
        usage = {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}
        raw = {"id": "resp_fixture", "model": config.served_model, "status": "completed", "usage": usage,
               "reasoning": {"effort": "xhigh"},
               "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        capture_receipt({"raw_response_sha256": digest(raw), "provider_request_id": "req_fixture",
                         "model_version": None, "reported_reasoning_effort": "xhigh"})
        capture_response(raw)
        lease.outcome("completed", usage=usage, response_complete=True)
        return ProviderReply("{}", "resp_fixture", config.served_model, usage, provider_request_id="req_fixture")

    monkeypatch.setattr(recovery, "SSHCourier", LocalCourier)
    monkeypatch.setattr(dispatcher, "call_responses", mock_provider)
    with pytest.raises(InterruptedError):
        recovery.run(staged.args)
    rebound = recovery.run(staged.args)
    assert rebound["delivered"] and rebound["outcome"] == "completed"
    assert len(paid) == 1 and delivery[0] == delivery[1]
    assert not staged.pool.status(readonly=True)["queue"]


def test_refuses_changed_hash_and_existing_original_completion(staged):
    with pytest.raises(PermissionError, match="original sealed timeouts"):
        recovery._parent_request(staged.parent, {**staged.scope, "payload_sha256": "0" * 64})
    path = staged.parent / "attempts" / recovery.PARENT_ATTEMPTS[1] / "raw_provider_response.json"
    once(path, {"status": "completed"})
    with pytest.raises(PermissionError, match="no captured completion"):
        recovery._parent_request(staged.parent, staged.scope)


def test_refuses_another_valid_bridge_job_with_changed_checkpoint(staged):
    original = staged.job["request"]
    changed = staged.make(original["binding"]["a7_formal"], staged.transport, staged.identity,
                          state={**staged.binding, "checkpoint_sha256": "0" * 64})
    fake = SimpleNamespace(validate=lambda job: verify_job(job))
    with pytest.raises(PermissionError, match="exact original"):
        recovery._validate(fake, changed, staged.original, staged.binding["prompt_sha256"])

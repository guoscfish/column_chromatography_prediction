"""GPU-to-Mac handoff with synthetic replies and no live provider or model fitting."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportStop, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgeCourier, make_job, make_result, verify_job, verify_result,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import bridge_endpoint, scheduler
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import Dispatcher, run_dispatcher, spool_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import SSH_API
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.seed_runner import (
    qualification_status, round_directory, seed_directory,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import ssh_courier
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_seed_runner import (
    MockProvider, SyntheticBackend, execute, formal, metadata,
)


def make_sample_job(formal, seed=157, *, mock=True, round_index=0):
    contract, approval = formal
    scope = {"contract_sha256": contract.digest, "authorization_sha256": digest(approval),
             "seed": seed, "mock": mock}
    packet = {"seed": seed, "round": round_index, "layout": "B", "compression": "lossless",
              "selection_count": 32, "memory": {"mode": "M2"}}
    messages = [{"role": "system", "content": "Synthetic A7 task."},
                {"role": "user", "content": dumps(packet)}]
    binding = {"seed": seed, "round": round_index, "arm": "A7", "checkpoint_sha256": "d" * 64,
               "packet_sha256": digest(packet), "mapping_sha256": "a" * 64, "state_sha256": "b" * 64,
               "base_messages_sha256": digest(messages), "prompt_sha256": digest(messages[0]["content"]),
               "split_sha256": "e" * 64, "schema_version": SCHEMA_VERSION, "prompt_version": "v3", "a7_formal": scope}
    request = make_request(binding, 0, messages, contract.transport)
    started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": f"{seed:032x}",
               "started_unix": 1234.0, "transport": "mac_bridge"}
    return make_job(request, contract.transport, started), request, started


def mocked_result(job, reply, config, *, raw_response=None):
    lock = verify_reported_model(config, reply, {"model_version": "mock-v1",
                                               "reported_reasoning_effort": "ultra"})
    return make_result(job, reply=reply, credential_id="test-slot", model_identity=lock,
                       raw_response=raw_response)


def test_handoff_binds_exact_request_and_rejects_cross_seed_response(formal):
    contract, approval = formal
    job, request, started = make_sample_job(formal)
    checked_request, checked_config, checked_started = verify_job(
        job, authorization_sha256=digest(approval), contract_sha256=contract.digest,
    )
    assert (checked_request, checked_config, checked_started) == (request, contract.transport, started)
    assert checked_request["payload_sha256"] == digest(request["payload"])
    with pytest.raises((ValueError, PermissionError, TransportStop)):
        verify_job(job, authorization_sha256="0" * 64, contract_sha256=contract.digest)
    reply = ProviderReply("{}", "mock-complete", contract.transport.served_model, {"input_tokens": 10})
    result = mocked_result(job, reply, contract.transport)
    returned = verify_result(result, job)
    assert (returned.response_id, returned.text) == (reply.response_id, reply.text)
    foreign, _, _ = make_sample_job(formal, seed=887)
    with pytest.raises((ValueError, PermissionError, TransportStop)):
        verify_result(result, foreign)


def test_ssh_uses_strict_host_keys_and_submits_result_over_stdin(monkeypatch):
    exchanges = []

    def run(argv, **options):
        exchanges.append((argv, options))
        return SimpleNamespace(stdout='{"stored":true}')

    monkeypatch.setattr(ssh_courier.subprocess, "run", run)
    link = ssh_courier.SSHCourier("eit-hpc", "/gpfs/release", "/gpfs/runs/a7", timeout=7)
    assert link.put("a" * 32, "b" * 32, {"text": "Synthetic private output"}) == {"stored": True}
    [(argv, options)] = exchanges
    assert "--" in argv and "StrictHostKeyChecking=yes" in argv and "BatchMode=yes" in argv
    assert all("Synthetic private output" not in arg for arg in argv)
    assert "Synthetic private output" in options["input"]
    assert options["timeout"] == 7
    with pytest.raises(ValueError, match="attempt"):
        link.get("../../unauthorized")
    assert len(exchanges) == 1


def test_gpfs_endpoint_checks_staged_authority_and_durable_dispatcher_claim(tmp_path, formal):
    run = (tmp_path / "synthetic_gpfs").resolve()
    base_contract, _ = formal
    contract = replace(base_contract, profile={**base_contract.profile, "transport": asdict(base_contract.transport)},
                       run_directory=str(run), api_execution=SSH_API, predecessor={"synthetic": True})
    approval = {**formal[1], "contract_sha256": contract.digest}
    job, request, started = make_sample_job((contract, approval), mock=False)
    identity = started["attempt_id"]
    once(run / "formal_contract.json", contract.record())
    once(run / "authorization.json", approval)
    once(run / "bridge/outbox" / f"{identity}.json", job)
    turn = run / "seed_157/round_00/turn_00"
    once(turn / "request.json", request)
    once(turn / "attempt_00.started.json", started)
    with pytest.raises((ValueError, FileNotFoundError)):
        bridge_endpoint.exchange(run, "get", identity)
    once(turn / "formal_request_authorization.json", {
        "scope": "paid_selection_only", "formal_run": request["binding"]["a7_formal"],
        "request_id": request["request_id"],
    })
    info = ssh_courier.verify_info(bridge_endpoint.exchange(run, "info"), expected_contract=contract.digest)
    inventory = ssh_courier.verify_list(bridge_endpoint.exchange(run, "list"), info)
    assert inventory["attempt_ids"] == [identity]
    (run / "pause.request").touch()
    with pytest.raises(PermissionError, match="paused"):
        bridge_endpoint.exchange(run, "claim", identity, dispatcher_id="b" * 32)
    (run / "pause.request").unlink()
    assert bridge_endpoint.exchange(run, "claim", identity, dispatcher_id="b" * 32) == job
    with pytest.raises((PermissionError, ValueError)):
        bridge_endpoint.exchange(run, "claim", identity, dispatcher_id="c" * 32)
    reply = ProviderReply("{}", "fixture", contract.transport.served_model, {"output_tokens": 1})
    with pytest.raises(ValueError, match="invalid completed provider response"):
        mocked_result(job, reply, contract.transport)
    raw = {"id": reply.response_id, "model": reply.served_model, "status": "completed", "usage": reply.usage,
           "output": [{"type": "message", "content": [{"type": "output_text", "text": reply.text}]}]}
    result = mocked_result(job, reply, contract.transport, raw_response=raw)
    with pytest.raises((PermissionError, ValueError)):
        bridge_endpoint.exchange(run, "put", identity, result, dispatcher_id="c" * 32)
    first = bridge_endpoint.exchange(run, "put", identity, result, dispatcher_id="b" * 32)
    assert first == {"attempt_id": identity, "result_sha256": result["result_sha256"]}
    assert bridge_endpoint.exchange(run, "put", identity, result, dispatcher_id="b" * 32) == first
    assert bridge_endpoint.exchange(run, "list")["attempt_ids"] == []


def test_one_gpu_pilot_starts_three_workers_and_caps_bridge_rounds(tmp_path, formal, monkeypatch):
    base, approval = formal
    run, lease_root = tmp_path / "pilot", tmp_path / "leases"
    lease_root.mkdir()
    profile = {**base.profile, "confirmation_seeds": [157, 887, 2357],
               "initial_labels": 333, "target_labels": 653, "requires_initial_qualification": False,
               "transport": asdict(base.transport)}
    contract = replace(base, profile=profile, gpu_slots=(0,), lease_root=str(lease_root),
                       api_execution=SSH_API, run_directory=str(run), predecessor={"synthetic": True})
    approval = {**approval, "contract_sha256": contract.digest}
    started = []

    def start(_ctx, _root, run, _contract, _approval, _pool, seed, gpu, stop, _generation):
        started.append((seed, gpu, stop))

        def join():
            assert started == [(157, 0, 10), (887, 0, 10), (2357, 0, 10)]
            once(run / "workers" / f"seed_{seed}.json", {"status": "complete"})

        return SimpleNamespace(exitcode=0, join=join)

    monkeypatch.setattr(scheduler, "_start", start)
    assert scheduler._run(tmp_path, run, contract, approval, None, resume=False) == 0
    assert not (run / "qualification.json").exists()
    job, _, marker = make_sample_job((contract, approval), seed=2357, mock=False, round_index=10)
    once(run / "bridge/outbox" / f"{marker['attempt_id']}.json", job)
    with pytest.raises(ValueError, match="outside the authorized run"):
        bridge_endpoint.exchange(run, "get", marker["attempt_id"])


@pytest.mark.parametrize("upload_reached_gpfs", [False, True])
def test_mac_dispatcher_reuses_completed_response_after_failed_delivery(tmp_path, formal, upload_reached_gpfs):
    original = formal[0]
    contract = replace(original, profile={**original.profile, "transport": asdict(original.transport)})
    approval = {**formal[1], "contract_sha256": contract.digest}
    job, request, started = make_sample_job((contract, approval))
    identity = started["attempt_id"]
    info = {"schema": ssh_courier.SCHEMA, "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "contract": contract.record(), "mock": True, "paused": False}
    emitted = []

    def use_mock(actual, config):
        assert actual == request
        emitted.append(actual["request_id"])
        return ProviderReply("{}", "mock-provider-completed", config.served_model, {"output_tokens": 1})

    spool = tmp_path / "mac_private_spool"

    class FakeSSH:
        uploaded = None
        claimed_by = None
        disconnect = True

        def info(self):
            return info

        def list(self):
            return {key: value for key, value in info.items() if key in ssh_courier.LIST_FIELDS} | {
                "attempt_ids": [] if self.uploaded else [identity],
            }

        def get(self, attempt_id):
            assert attempt_id == identity
            return job

        def claim(self, attempt_id, dispatcher_id):
            assert attempt_id == identity and self.claimed_by in (None, dispatcher_id)
            self.claimed_by = dispatcher_id
            return job

        def put(self, attempt_id, dispatcher_id, result):
            assert attempt_id == identity and dispatcher_id == self.claimed_by
            assert read(spool / "results" / f"{identity}.json") == result
            assert self.uploaded is None or self.uploaded == result
            if self.disconnect:
                self.disconnect = False
                if upload_reached_gpfs:
                    self.uploaded = result
                raise ssh_courier.CourierUnavailable("synthetic SSH interruption")
            self.uploaded = result
            return {"attempt_id": attempt_id, "result_sha256": result["result_sha256"]}

    courier = FakeSSH()
    dispatcher = Dispatcher(spool, info=info, contract_sha256=contract.digest, transport=contract.transport, call=use_mock)
    with pytest.raises(ssh_courier.CourierUnavailable, match="synthetic SSH"):
        run_dispatcher(dispatcher, courier, watch=False, workers=1)
    sealed = read(spool / "results" / f"{identity}.json")
    assert dispatcher.undelivered() == [identity] and len(emitted) == 1

    restarted = Dispatcher(spool, info=info, contract_sha256=contract.digest, transport=contract.transport, call=use_mock)
    assert restarted.dispatcher_id == dispatcher.dispatcher_id
    final = run_dispatcher(restarted, courier, watch=False, workers=1)
    assert final["delivered_results"] == 1
    assert restarted.undelivered() == [] and len(emitted) == 1
    assert courier.uploaded == sealed and spool_status(spool)["open_or_ambiguous_count"] == 0


def test_mac_started_without_response_is_never_dispatched_again(tmp_path, formal):
    original = formal[0]
    contract = replace(original, profile={**original.profile, "transport": asdict(original.transport)})
    approval = {**formal[1], "contract_sha256": contract.digest}
    job, _, started = make_sample_job((contract, approval))
    identity = started["attempt_id"]
    info = {"schema": ssh_courier.SCHEMA, "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "contract": contract.record(), "mock": True, "paused": False}
    calls = []

    def interrupted_provider(_request, _config):
        calls.append(identity)
        raise RuntimeError("synthetic Mac process failed during its provider call")

    settings = {"info": info, "contract_sha256": contract.digest,
                "transport": contract.transport, "call": interrupted_provider}
    spool = tmp_path / "mac_private_spool"
    first = Dispatcher(spool, **settings)
    first.accept(job)
    with pytest.raises(RuntimeError, match="failed during"):
        first.dispatch(identity, before_call=lambda *_: None)
    second = Dispatcher(spool, **settings)
    assert second.pending() == []
    assert second.dispatch(identity, before_call=lambda *_: None) == "open_or_ambiguous"
    assert second.undelivered() == [] and calls == [identity]
    assert spool_status(spool)["open_or_ambiguous_count"] == 1


@pytest.mark.parametrize("repair_first_response", [False, True])
def test_gpu_wait_resumes_exact_mac_receipts_and_freezes_before_fit(
    tmp_path, formal, metadata, repair_first_response,
):
    contract, approval = formal
    run = tmp_path / "gpu_traj"
    bridge = run / "bridge"
    backend = SyntheticBackend(157, metadata)
    mac_model = MockProvider(invalid_first=repair_first_response)
    gpu_busy = False

    @contextmanager
    def fit_slot():
        nonlocal gpu_busy
        assert not gpu_busy
        gpu_busy = True
        try:
            yield
        finally:
            gpu_busy = False

    courier = BridgeCourier(bridge, authorization_sha256=digest(approval), contract_sha256=contract.digest,
                            model_lock=run / "provider_model_lock.json", wait_seconds=0.04, poll_seconds=0.01)

    def managed_attempt(request, config, started):
        assert not gpu_busy
        return courier(request, config, started)

    def no_gpu_provider(*args):
        pytest.fail("the GPU tried to call the provider directly")

    opts = {"fit_slot": fit_slot, "transport_options": {"managed_attempt": managed_attempt}}
    seed_dir = seed_directory(run, backend.seed)
    logical_replies = 2 if repair_first_response else 1
    for index in (0, 1):
        current = round_directory(seed_dir, index)
        attempts = {}
        for logical in range(logical_replies):
            turn = current / f"turn_{logical:02d}"
            with pytest.raises(TransportStop):
                execute(run, formal, backend, no_gpu_provider, max_rounds=index + 1, **opts)
            started_path = turn / "attempt_00.started.json"
            identity = read(started_path)
            attempts[started_path] = file_digest(started_path)
            assert not gpu_busy and (backend.reveal_calls, backend.fit_calls) == (index, index)
            assert not (current / "batch_freeze.json").exists()
            assert not list(turn.glob("*.terminal.json"))
            job_file = bridge / "outbox" / f"{identity['attempt_id']}.json"
            job = read(job_file)
            request, config, started = verify_job(job, authorization_sha256=digest(approval),
                                                   contract_sha256=contract.digest)
            assert request == read(turn / "request.json") and started == identity
            assert (request["binding"]["round"], request["logical_index"]) == (index, logical)
            assert read(turn / "formal_request_authorization.json")["request_id"] == request["request_id"]
            response = mac_model(request, config)
            once(bridge / "results" / job_file.name, mocked_result(job, response, config))
        step = execute(run, formal, backend, no_gpu_provider, max_rounds=index + 1, **opts)
        assert step.completed_rounds == index + 1
        assert all(file_digest(path) == sha for path, sha in attempts.items())
        freeze = read(current / "batch_freeze.json")
        assert len(freeze["batch_ids"]) == 32
        assert freeze["accepted_logical_index"] == logical_replies - 1
        assert read(round_directory(seed_dir, index + 1) / "transition.json")["selection_freeze_sha256"] == (
            file_digest(current / "batch_freeze.json")
        )
        assert (backend.reveal_calls, backend.fit_calls) == (index + 1, index + 1)
    assert qualification_status(run, contract, approval, mock=True)["status"] == "qualified"
    assert (step.status, step.active_labels) == ("complete", 66)
    assert len(mac_model.requests) == 2 * logical_replies
    assert execute(run, formal, backend, no_gpu_provider, **opts).status == "complete"
    assert (backend.reveal_calls, backend.optimizer_runs) == (2, 2)

"""Offline tests for Mac attempt custody and recovery."""

from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace
import time

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, ProviderReply, TransportFailure,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import dispatcher, provider, ssh_courier
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import KeyPool, PoolConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_distributed_a7 import make_sample_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_seed_runner import formal  # noqa: F401


@pytest.fixture
def staged(tmp_path, formal):
    base, approval = formal
    contract = replace(base, profile={**base.profile, "transport": asdict(base.transport)})
    approval = {**approval, "contract_sha256": contract.digest}
    info = {"schema": ssh_courier.SCHEMA, "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "contract": contract.record(), "mock": True, "paused": False}
    job, _, started = make_sample_job((contract, approval))

    def forbidden(*_):
        pytest.fail("a recovered or ambiguous journal must not call a provider")

    mac = dispatcher.Dispatcher(tmp_path / "spool", info=info, contract_sha256=contract.digest,
                                transport=contract.transport, call=forbidden)
    return mac, job, started["attempt_id"]


def _started(mac, job, identity):
    mac.accept(job)
    attempt = mac.path / "attempts" / identity
    once(attempt / "started.json", {
        "attempt_id": identity, "job_sha256": job["job_sha256"],
        "credential_id": "synthetic-key", "credential_lease_ticket": "test-ticket",
        "started_utc": "2026-11-25T00:00:00+00:00",
    })
    attempt.chmod(0o700)
    return attempt


@pytest.fixture
def readiness_run(tmp_path, formal, monkeypatch):
    base, approval = formal
    timeouts = RequestTimeouts()
    transport = replace(base.transport, wire_mode=RESPONSES_STREAM_IPV4, timeout_seconds=timeouts.total_seconds)
    contract = replace(base, transport=transport, profile={**base.profile, "transport": asdict(transport),
        "request_timeouts": asdict(timeouts), "sse_inactivity_seconds": 900,
        "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3})
    approval = {**approval, "contract_sha256": contract.digest}
    info = {"schema": ssh_courier.SCHEMA, "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "contract": contract.record(), "mock": False, "paused": False}
    job, request, started = make_sample_job((contract, approval), mock=False)
    identity = started["attempt_id"]
    spool = tmp_path / "spool"
    key = tmp_path / "fixture-key"
    key.write_text("synthetic-key")
    key.chmod(0o600)
    pool = KeyPool(spool / "key_pool", PoolConfig(1, {"test-slot": key}))

    def reopen():
        return dispatcher.Dispatcher(spool, info=info, contract_sha256=contract.digest,
                                     transport=transport, mock=False, pool=pool)

    mac = reopen()
    pool.prepare()
    assert mac.accept(job) == identity
    clock = SimpleNamespace(now=10_000)
    monkeypatch.setattr(dispatcher, "time", SimpleNamespace(time=lambda: clock.now, monotonic=time.monotonic))
    return SimpleNamespace(mac=mac, reopen=reopen, job=job, request=request, identity=identity, clock=clock,
                           attempt=spool / "attempts" / identity)


def _probe_failure(lease):
    lease.outcome("connect_error")
    raise provider._IPv4PreflightFailure(retry_probe=True)


def _watch_inventory(run):
    return {key: value for key, value in run.mac.info.items() if key != "contract"} | {"attempt_ids": [run.identity]}


def test_watch_preserves_queue_on_ssh_fault_but_fences_on_authenticated_pause(readiness_run, monkeypatch):
    run = readiness_run
    steps = iter(("ready", "info_outage", "list_outage", "paused_upload_outage", "info_outage", "ready"))
    observed, submitted = [], []
    phase = None

    class EndWatch(Exception):
        pass

    def info():
        nonlocal phase
        phase = next(steps)
        if phase == "info_outage":
            raise ssh_courier.CourierUnavailable("test SSH info outage")
        return {**run.mac.info, "paused": phase == "paused_upload_outage"}

    def inventory():
        if phase == "list_outage":
            raise ssh_courier.CourierUnavailable("test SSH inventory outage")
        return _watch_inventory(run)

    def deliver(*_):
        if phase == "paused_upload_outage":
            raise ssh_courier.CourierUnavailable("test SSH delivery outage")

    def submit(function, identity, **options):
        assert function == run.mac.dispatch and identity == run.identity
        submitted.append(options)
        return Future()

    def check_waiting(_poll):
        assert len(submitted) == 1
        observed.append((phase, submitted[0]["pause"]()))
        if phase == "paused_upload_outage":
            with pytest.raises(dispatcher.RemotePaused):
                submitted[0]["before_call"](run.identity, run.job)
        if len(observed) == 6:
            raise EndWatch

    courier = SimpleNamespace(info=info, list=inventory,
                              retry_exchange=lambda *_a, **_kw: pytest.fail("paused waiter cannot claim"))
    monkeypatch.setattr(dispatcher, "deliver", deliver)
    monkeypatch.setattr(dispatcher, "ThreadPoolExecutor", lambda **_: nullcontext(SimpleNamespace(submit=submit)))
    monkeypatch.setattr(dispatcher.time, "sleep", check_waiting, raising=False)

    with pytest.raises(EndWatch):
        dispatcher.run_dispatcher(run.mac, courier, watch=True, workers=1)
    assert observed == [
        ("ready", False), ("info_outage", False), ("list_outage", False),
        ("paused_upload_outage", True), ("info_outage", True), ("ready", False),
    ]


def test_readiness_requires_a_second_fresh_claim_before_provider_post(readiness_run, monkeypatch):
    run = readiness_run
    claims = []

    def claim(action, *, attempt_id, dispatcher_id):
        assert action == "claim" and (attempt_id, dispatcher_id) == (run.identity, run.mac.dispatcher_id)
        claims.append(attempt_id)
        if len(claims) == 2:
            raise ssh_courier.CourierUnavailable("test SSH outage immediately before provider POST")
        return run.job

    def call(_request, _config, *, before_provider_dispatch, **_):
        assert len(claims) == 1
        before_provider_dispatch()
        pytest.fail("a POST requires both independently authenticated claims")

    courier = SimpleNamespace(info=lambda: run.mac.info, list=lambda: _watch_inventory(run), retry_exchange=claim)
    monkeypatch.setattr(dispatcher, "call_responses", call)
    status = dispatcher.run_dispatcher(run.mac, courier, watch=False, workers=1)
    assert claims == [run.identity, run.identity]
    assert status["round"]["deferred"] == 1
    assert not (run.attempt / "started.json").exists()
    assert run.mac.pending() == [run.identity]
    assert run.mac.pool.status(readonly=True)["keys"][0]["requests"] == 0


def test_readiness_waits_without_start_then_same_attempt_completes_after_restart(readiness_run, monkeypatch):
    run = readiness_run
    starts, validations = [], []
    start_file = run.attempt / "started.json"
    result_path = run.mac.path / "results" / f"{run.identity}.json"
    usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

    def before_call(identity, job):
        assert (identity, job) == (run.identity, run.job)
        assert not start_file.exists()
        validations.append(identity)

    def call(request, config, *, ipv4_preflight_retry, before_provider_dispatch, model_lock, capture_response,
             capture_receipt, **_):
        assert request == run.request and ipv4_preflight_retry is True
        lease = dispatcher.LEASE.get()
        assert not start_file.exists() and not dispatcher._lease_dispatched(lease)
        windows = sorted(run.attempt.glob("readiness_*.json"))
        starts.append(read(windows[-1]))
        if len(starts) == 1:
            _probe_failure(lease)
        before_provider_dispatch()
        adopted = read(start_file)
        assert adopted["attempt_id"] == run.identity and adopted["job_sha256"] == run.job["job_sha256"]
        assert adopted["credential_lease_ticket"] == lease.ticket
        assert not dispatcher._lease_dispatched(lease)
        lease.dispatched()
        reply = ProviderReply("{}", "resp-fixture", config.served_model, usage)
        declared = {"model_version": None, "reported_reasoning_effort": None}
        raw = {"id": reply.response_id, "model": reply.served_model, "status": "completed", "usage": usage,
               "output": [{"type": "message", "content": [{"type": "output_text", "text": reply.text}]}]}
        once(model_lock, provider.verify_reported_model(config, reply, declared))
        capture_receipt({"raw_response_sha256": digest(raw), "provider_request_id": None, **declared})
        capture_response(raw)
        lease.outcome("completed", usage=usage, response_complete=True)
        return reply

    monkeypatch.setattr(dispatcher, "call_responses", call)
    assert run.mac.dispatch(run.identity, before_call=before_call) == "waiting_for_route"
    assert not start_file.exists() and not result_path.exists()
    assert run.mac.pool.status(readonly=True)["keys"][0]["requests"] == 0
    assert run.mac.pool.status(readonly=True)["queue"] == []
    assert run.mac.dispatch(run.identity, before_call=before_call) == "waiting_for_route"
    assert len(starts) == 1
    run.clock.now += 16
    resumed = run.reopen()
    assert resumed.pending() == [run.identity] and resumed.dispatcher_id == run.mac.dispatcher_id
    assert resumed.dispatch(run.identity, before_call=before_call) == "sealed"
    assert [entry["window"] for entry in starts] == [1, 2]
    assert all(entry["attempt_id"] == run.identity and entry["job_sha256"] == run.job["job_sha256"]
               for entry in starts)
    assert len(validations) == 3
    result = resumed.result(run.identity)
    assert result["provider_dispatched"] is True and result["request_id"] == run.request["request_id"]
    assert result["outcome"]["reply"]["usage"] == usage
    assert resumed.pool.status(readonly=True)["keys"][0]["requests"] == 1
    assert resumed.dispatch(run.identity, before_call=lambda *_: pytest.fail("sealed attempt must not be sent")) == "sealed"


@pytest.mark.parametrize("crash_during_last_probe", [False, True])
def test_readiness_window_budget_survives_a_restart(readiness_run, monkeypatch, crash_during_last_probe):
    run = readiness_run
    probes = []

    def call(request, _config, **_):
        assert request == run.request and not (run.attempt / "started.json").exists()
        probes.append(request["request_id"])
        if crash_during_last_probe and len(probes) == 3:
            raise RuntimeError("interrupted no-key probe")
        _probe_failure(dispatcher.LEASE.get())

    monkeypatch.setattr(dispatcher, "call_responses", call)
    for window in (1, 2):
        assert run.reopen().dispatch(run.identity, before_call=lambda *_: None) == "waiting_for_route"
        assert len(probes) == window and not (run.attempt / "started.json").exists()
        run.clock.now += 16
    if crash_during_last_probe:
        with pytest.raises(RuntimeError, match="interrupted no-key probe"):
            run.reopen().dispatch(run.identity, before_call=lambda *_: None)
        assert not (run.attempt / "started.json").exists()
    resumed = run.reopen()
    def validate_without_key(identity, job):
        assert (identity, job) == (run.identity, run.job)
        assert dispatcher.LEASE.get() is None

    validate_with_key = (lambda *_: pytest.fail("archiving exhausted probes does not acquire a Key")
                         ) if crash_during_last_probe else lambda *_: None
    assert resumed.dispatch(run.identity, before_call=validate_with_key,
                            before_no_post=validate_without_key) == "sealed"
    assert len(probes) == 3 and len(list(run.attempt.glob("readiness_*.json"))) == 3
    assert not (run.attempt / "started.json").exists()
    result = resumed.result(run.identity)
    assert result["provider_dispatched"] is False and result["outcome"]["failure"]["code"] == "connect_error"
    assert result["outcome"]["failure"]["ambiguous"] is False
    assert resumed.pending() == [] and resumed.pool.status(readonly=True)["keys"][0]["requests"] == 0


def test_readiness_never_requeues_an_interrupted_model_post(readiness_run, monkeypatch):
    run = readiness_run

    def call(_request, _config, *, before_provider_dispatch, **_):
        before_provider_dispatch()
        dispatcher.LEASE.get().dispatched()
        raise RuntimeError("interrupted after possible provider dispatch")

    monkeypatch.setattr(dispatcher, "call_responses", call)
    with pytest.raises(RuntimeError, match="interrupted after possible provider dispatch"):
        run.mac.dispatch(run.identity, before_call=lambda *_: None)
    resumed = run.reopen()
    assert (run.attempt / "started.json").is_file() and resumed.pending() == []
    assert resumed.dispatch(run.identity, before_call=lambda *_: pytest.fail("ambiguous attempt must stay closed")) == "open_or_ambiguous"
    assert resumed.pool.status(readonly=True)["keys"][0]["requests"] == 1


def test_restart_recovers_complete_raw_response_without_provider_or_new_key(staged):
    mac, job, identity = staged
    attempt = _started(mac, job, identity)
    raw = {
        "id": "response-fixture", "model": mac.transport.served_model, "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "reasoning": {"effort": "ultra"},
    }
    once(attempt / "provider_capture_receipt.json", {
        "attempt_id": identity, "job_sha256": job["job_sha256"],
        "metadata": {"raw_response_sha256": digest(raw), "provider_request_id": "req-fixture",
                     "model_version": "test-serve-1", "reported_reasoning_effort": "ultra"},
    })
    once(attempt / "raw_provider_response.json", raw)

    assert mac.recover_captured(exclude=[identity]) == 0
    assert mac.recover_captured() == 1
    assert mac.recover_captured() == 0
    sealed = mac.result(identity)
    assert sealed["raw_response"] == raw
    assert sealed["provider_dispatched"] and sealed["credential_id"] == "synthetic-key"
    assert sealed["model_identity"]["reported_model_version"] == "test-serve-1"
    reply = sealed["outcome"]["reply"]
    assert reply["provider_request_id"] == "req-fixture"
    assert reply["usage"] == raw["usage"] and reply["text"] == "{}"
    assert not mac.pending() and mac.undelivered() == [identity]


def test_unknown_or_unbound_started_is_never_resent(staged):
    mac, job, identity = staged
    attempt = _started(mac, job, identity)
    once(attempt / "raw_provider_response.json", {"status": "completed"})
    assert mac.recover_captured() == 0  # No durable exact-attempt capture receipt.
    assert mac.dispatch(identity, before_call=lambda *_: pytest.fail("already started")) == "open_or_ambiguous"
    assert mac.pending() == []
    assert dispatcher.spool_status(mac.path)["open_or_ambiguous_count"] == 1


def test_stream_progress_keeps_early_ids_after_ambiguous_failure(tmp_path, formal, monkeypatch):
    base, approval = formal
    timeouts = RequestTimeouts()
    contract = replace(base, profile={**base.profile, "transport": asdict(base.transport),
                                      "request_timeouts": asdict(timeouts)})
    approval = {**approval, "contract_sha256": contract.digest}
    info = {"schema": ssh_courier.SCHEMA, "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "contract": contract.record(), "mock": False, "paused": False}
    job, _, started = make_sample_job((contract, approval), mock=False)
    key = tmp_path / "fixture-key"
    key.write_text("synthetic-key")
    key.chmod(0o600)
    spool = tmp_path / "spool"
    pool = KeyPool(spool / "key_pool", PoolConfig(1, {"test-slot": key}))
    mac = dispatcher.Dispatcher(spool, info=info, contract_sha256=contract.digest, transport=contract.transport,
                                mock=False, pool=pool)
    pool.prepare()
    attempt = mac.accept(job)

    def interrupt(_request, _config, *, progress, **kwargs):
        assert kwargs["timeouts"] == timeouts
        lease = dispatcher.LEASE.get()
        lease.dispatched()
        progress({"phase": "http_phase", "elapsed_seconds": 1, "network_phase": "tls", "network_state": "complete",
                  "phase_duration_seconds": .5, "explicit_proxy": False})
        progress({"phase": "http_headers", "elapsed_seconds": 2, "http_status": 200,
                  "provider_request_id": "req-fixture", "authorization": "synthetic-key"})
        progress({"phase": "sse_event", "elapsed_seconds": 3, "event_type": "response.created",
                  "response_id": "resp-fixture", "text": "synthetic unpublished text", "model": "sk-fake-key",
                  "sse_event_count": 1})
        live, = dispatcher.spool_status(spool)["provider_progress"]
        assert live["event_type"] == "response.created" and live["sealed"] is False
        progress({"phase": "sse_event", "elapsed_seconds": 4, "event_type": "response.in_progress",
                  "sse_event_count": 2, "sse_interval_seconds": 1})
        progress({"phase": "transport_failure", "elapsed_seconds": 8, "error_type": "RemoteProtocolError"})
        lease.outcome("sse_frame_error")
        raise TransportFailure("sse_frame_error", ambiguous=True)

    monkeypatch.setattr(dispatcher, "call_responses", interrupt)
    assert attempt == started["attempt_id"]
    assert mac.dispatch(attempt, before_call=lambda *_: None) == "sealed"
    trace, = dispatcher.spool_status(spool)["provider_progress"]
    assert (trace["response_id"], trace["provider_request_id"], trace["error_type"], trace["http_status"],
            trace["sealed"]) == ("resp-fixture", "req-fixture", "RemoteProtocolError", 200, True)
    assert trace["network_phases"]["tls"] == {"state": "complete", "duration_seconds": .5, "observed_seconds": 1}
    assert trace["explicit_proxy"] is False
    assert [trace[field] for field in ("first_sse_seconds", "last_sse_seconds", "sse_event_count")] == [3, 4, 2]
    assert trace["sse_milestones"]["response.in_progress"] == {"elapsed_seconds": 4, "interval_seconds": 1}
    progress_file = spool / "attempts" / attempt / dispatcher.PROGRESS_FILENAME
    assert progress_file.stat().st_mode & 0o777 == 0o600
    assert trace.keys().isdisjoint({"authorization", "model", "text"})
    assert mac.result(attempt)["outcome"]["failure"]["ambiguous"] is True
    assert mac.dispatch(attempt, before_call=lambda *_: pytest.fail("attempt must not be resent")) == "sealed"


def test_readonly_key_pool_status_does_not_prune_or_open_credentials(tmp_path, monkeypatch):
    key_path = tmp_path / "absent-key"
    pool = KeyPool(tmp_path / "key_pool", PoolConfig(1, {"alias_only": key_path}))
    pool.prepare()

    def forbidden(*_):
        pytest.fail("readonly status attempted to prune or open a Key")

    monkeypatch.setattr(pool, "_prune", forbidden)
    monkeypatch.setattr(PoolConfig, "status", forbidden)
    info = pool.status(readonly=True)
    assert info["initialized"] and info["queue"] == []
    assert [(row["id"], row["status"], row["requests"]) for row in info["keys"]] == [
        ("alias_only", "disabled", 0),
    ]
    assert not key_path.exists()


def test_receipt_ack_cannot_change_completed_result(staged):
    mac, job, identity = staged
    _started(mac, job, identity)
    receipt = {"result_sha256": "a" * 64}
    with pytest.raises(ValueError, match="exact sealed result"):
        mac.mark_delivered(identity, receipt, {"attempt_id": identity, "result_sha256": "b" * 64})
    assert not (mac.path / "delivered" / f"{identity}.json").exists()


def test_paid_source_gate_checks_the_approved_provider_file():
    root = dispatcher.ROOT
    sources = {
        path.relative_to(root).as_posix(): file_digest(path)
        for directory in (root / dispatcher.STUDY / "a7_gpu", root / dispatcher.OFFLINE / "pure_llm_v3")
        for path in directory.glob("*.py")
    }
    authority = {"contract": {"source_sha256": sources}}
    dispatcher.verify_local_source(authority)
    sources[(dispatcher.STUDY / "a7_gpu/provider.py").as_posix()] = "0" * 64
    with pytest.raises(PermissionError, match="provider.py"):
        dispatcher.verify_local_source(authority)

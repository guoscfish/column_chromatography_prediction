"""Offline tests for Mac attempt custody and recovery."""

from dataclasses import asdict, replace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportFailure
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import dispatcher, ssh_courier
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

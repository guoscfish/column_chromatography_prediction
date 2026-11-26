import json
import shlex
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_6101_cached_r2 as dispatch


INFO = {"contract_sha256": "a" * 64, "authorization_sha256": "b" * 64}
PROOF = {"cached_import": {"provider_post": False}}
PENDING = {"status": "pending", **INFO}
CHECKPOINT = "c" * 64
COMPLETE = {"status": "complete", **INFO, "seed": 6101, "round": 3, "active_labels": 429,
            "provider_post": False, "cached_import_sha256": dispatch.digest(PROOF["cached_import"]),
            "checkpoint_sha256": CHECKPOINT, "transition_file_sha256": "d" * 64}


def test_requires_complete_handoff_in_exact_authorized_scope():
    assert dispatch._verified_checkpoint(PENDING, INFO, PROOF) is None
    assert dispatch._verified_checkpoint(COMPLETE, INFO, PROOF) == CHECKPOINT
    for changed in ({"authorization_sha256": "e" * 64}, {"active_labels": 397}, {"round": 2},
                    {"cached_import_sha256": "e" * 64}, {"provider_post": 0}, {"provider_post": True},
                    {"transition_file_sha256": None}):
        with pytest.raises(PermissionError):
            dispatch._verified_checkpoint({**COMPLETE, **changed}, INFO, PROOF)
    with pytest.raises(PermissionError):
        dispatch._verified_checkpoint({**PENDING, "contract_sha256": "e" * 64}, INFO, PROOF)


def test_pending_and_transient_ssh_are_safe_before_fit(monkeypatch):
    for observation, wait in ((PENDING, 0), (PENDING, 2)):
        clock = iter((10, 12))
        monkeypatch.setattr(dispatch, "monotonic", lambda: next(clock))
        monkeypatch.setattr(dispatch, "_read_handoff", lambda *args: observation)
        if wait == 0:
            assert dispatch._wait_handoff(None, INFO, PROOF, wait) is None
        else:
            with pytest.raises(dispatch.CourierUnavailable):
                dispatch._wait_handoff(None, INFO, PROOF, wait)

    observations = iter((PENDING, dispatch.CourierUnavailable("SSH interrupted"), COMPLETE))

    def read(*args):
        observation = next(observations)
        if isinstance(observation, Exception):
            raise observation
        return observation

    monkeypatch.setattr(dispatch, "monotonic", lambda: 10)
    monkeypatch.setattr(dispatch, "sleep", lambda delay: None)
    monkeypatch.setattr(dispatch, "_read_handoff", read)
    assert dispatch._wait_handoff(None, INFO, PROOF, 5) == CHECKPOINT


def test_only_exact_fit_checkpoint_can_start_round_three(monkeypatch):
    worker = dispatch.FutureDispatcher.__new__(dispatch.FutureDispatcher)
    worker.handoff_checkpoint = CHECKPOINT
    with pytest.raises(PermissionError, match="local cached"):
        worker._check_boundary({}, {})

    request = {"binding": {"round": 3, "checkpoint_sha256": CHECKPOINT}, "request_id": "new"}
    started = {"attempt_id": "new"}
    transport = object()
    monkeypatch.setattr(dispatch.RelayRetryDispatcher, "validate", lambda *_: (request, transport, started))
    assert worker.validate({}) == (request, transport, started)
    request["binding"]["checkpoint_sha256"] = "different"
    with pytest.raises(PermissionError):
        worker.validate({})
    request["binding"]["round"] = 4
    assert worker.validate({}) == (request, transport, started)
    started["attempt_id"] = dispatch.gpu.ATTEMPT
    with pytest.raises(PermissionError):
        worker.validate({})


def test_handoff_uses_strict_quoted_read_only_ssh(monkeypatch):
    def ssh(argv, **kwargs):
        assert argv[:-1] == ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            "-o", "StrictHostKeyChecking=yes", "--", "eit-hpc"]
        assert shlex.split(argv[-1]) == ["cd", "/shared/release", "&&", "exec", "/shared/venv/bin/python",
                                       "-B", "-c", dispatch.HANDOFF_CHECK, "/shared/run", *INFO.values()]
        assert kwargs == {"input": "", "capture_output": True, "text": True, "check": True, "timeout": 20}
        return SimpleNamespace(stdout=json.dumps(COMPLETE))

    courier = dispatch.SSHCourier("eit-hpc", "/shared/release", "/shared/run", "/shared/venv/bin/python")
    monkeypatch.setattr(dispatch.subprocess, "run", ssh)
    assert dispatch._read_handoff(courier, INFO) == COMPLETE

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(255, "ssh", stderr="redacted")

    monkeypatch.setattr(dispatch.subprocess, "run", unavailable)
    with pytest.raises(dispatch.CourierUnavailable, match="did not complete"):
        dispatch._read_handoff(courier, INFO)


def test_no_dispatcher_created_before_handoff(monkeypatch):
    transport = dispatch.TransportConfig(provider="token4research", base_url="https://token4research.cn/v1",
                                         max_output_tokens=32768, requested_model="gpt-6.2-astra", served_model="gpt-6.2-astra",
                                         reasoning_effort="xhigh", sdk_version="2.28.0",
                                         logical_replies=3, transport_attempts=2,
                                         timeout_seconds=dispatch.resilient.TIMEOUTS.total_seconds,
                                         wire_mode=dispatch.RESPONSES_STREAM_IPV4)
    info = {**INFO, "contract": {"profile": {"transport": asdict(transport)}}}
    args = SimpleNamespace(ssh_host="eit-hpc", remote_root="/shared/release", remote_run_dir="/shared/run",
                           remote_python="/shared/venv/bin/python", verify_only=True, handoff_wait=10, **INFO)
    monkeypatch.setattr(dispatch, "SSHCourier", lambda *a: SimpleNamespace(retry_exchange=lambda name: info))
    monkeypatch.setattr(dispatch, "verify_info", lambda info, **kw: info)
    monkeypatch.setattr(dispatch, "_scope", lambda *a: PROOF)
    monkeypatch.setattr(dispatch, "_archive", lambda *a: transport)
    monkeypatch.setattr(dispatch.resilient, "_pool", lambda *a: (None, None, transport))
    monkeypatch.setattr(dispatch, "sdk_status", lambda transport: {"matches": True})
    monkeypatch.setattr(dispatch, "_wait_handoff", lambda *a: None)
    monkeypatch.setattr(dispatch, "FutureDispatcher", lambda *a, **kw: pytest.fail("attempted dispatch before L429"))
    assert dispatch.run(args)["l429_handoff_verified"] is False
    args.verify_only = False
    with pytest.raises(PermissionError, match="requires a verified L429"):
        dispatch.run(args)

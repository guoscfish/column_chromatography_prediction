"""Offline tests for one bounded, authenticated SSH read or claim."""

from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import ssh_courier


@pytest.fixture
def courier(monkeypatch):
    clock = SimpleNamespace(now=0, delays=[])

    def sleep(delay):
        clock.delays.append(delay)
        clock.now += delay

    monkeypatch.setattr(ssh_courier, "monotonic", lambda: clock.now)
    monkeypatch.setattr(ssh_courier, "sleep", sleep)
    return ssh_courier.SSHCourier("fixture-host", "/release", "/run"), clock


def test_claim_retries_same_ids_after_missing_ssh_ack(courier, monkeypatch):
    client, clock = courier
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise ssh_courier.subprocess.CalledProcessError(255, command, stderr="do not log endpoint output")
        return SimpleNamespace(stdout='{"job":"fixture"}')

    monkeypatch.setattr(ssh_courier.subprocess, "run", run)
    attempt, dispatcher = "a" * 32, "b" * 32
    assert client.retry_exchange("claim", attempt_id=attempt, dispatcher_id=dispatcher) == {"job": "fixture"}
    assert len(calls) == 2 and calls[0] == calls[1]
    command, arguments = calls[0]
    assert "StrictHostKeyChecking=yes" in command
    assert f"--attempt-id {attempt} --dispatcher-id {dispatcher}" in command[-1]
    assert arguments["input"] == "" and arguments["timeout"] == 8
    assert clock.delays == [.25]


def test_readiness_ssh_checks_share_one_monotonic_deadline(courier, monkeypatch):
    client, clock = courier
    timeouts = []

    def run(command, *, timeout, **_kwargs):
        timeouts.append(timeout)
        clock.now += timeout
        raise ssh_courier.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(ssh_courier.subprocess, "run", run)
    with pytest.raises(ssh_courier.CourierUnavailable, match="retry budget"):
        client.retry_exchange("info")
    assert timeouts == [8, 8, 3] and clock.delays == [.25, .75]
    assert clock.now == 20


def test_late_ssh_ack_cannot_open_dispatch(courier, monkeypatch):
    client, clock = courier

    def run(*_args, **_kwargs):
        clock.now = 21
        return SimpleNamespace(stdout='{"job":"too late"}')

    monkeypatch.setattr(ssh_courier.subprocess, "run", run)
    with pytest.raises(ssh_courier.CourierUnavailable, match="retry budget"):
        client.retry_exchange("claim", attempt_id="a" * 32, dispatcher_id="b" * 32)
    assert clock.delays == []


def test_retry_does_not_repeat_authenticated_rejection(courier, monkeypatch):
    client, clock = courier
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        raise ssh_courier.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(ssh_courier.subprocess, "run", run)
    with pytest.raises(PermissionError, match="refused"):
        client.retry_exchange("claim", attempt_id="a" * 32, dispatcher_id="b" * 32)
    assert len(calls) == 1 and clock.delays == []


def test_old_exchange_and_result_upload_do_not_use_new_retries(courier, monkeypatch):
    client, clock = courier
    timeouts = []

    def run(command, *, timeout, **_kwargs):
        timeouts.append(timeout)
        raise ssh_courier.subprocess.CalledProcessError(255, command)

    monkeypatch.setattr(ssh_courier.subprocess, "run", run)
    with pytest.raises(ValueError, match="read or an exact claim"):
        client.retry_exchange("put", attempt_id="a" * 32, dispatcher_id="b" * 32)
    with pytest.raises(ssh_courier.CourierUnavailable):
        client.info()
    assert timeouts == [20] and clock.delays == []

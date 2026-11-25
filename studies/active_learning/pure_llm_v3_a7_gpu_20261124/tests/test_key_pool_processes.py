"""Exercise physical leases and SDK wiring without network traffic or real credentials."""

from functools import partial
from importlib.metadata import version
import multiprocessing
import os
from queue import Empty
import secrets
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportConfig, TransportStop, make_request, round_lock, run_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE, KeyPool, PoolConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import call_responses


def _transport(**changes):
    fields = {
        "provider": "synthetic", "requested_model": "synthetic-model", "served_model": "synthetic-model",
        "reasoning_effort": "ultra", "base_url": "https://example.invalid/v1", "timeout_seconds": 4,
        "max_output_tokens": 128, "transport_attempts": 2, "logical_replies": 2,
        "base_delay_seconds": 1, "max_delay_seconds": 10, "rate_limit_threshold": 3,
        "sdk_version": version("openai"),
    }
    return TransportConfig(**{**fields, **changes})


def _request(seed, config, *, logical_index=0):
    return make_request({"seed": seed, "round": 0}, logical_index,
                        [{"role": "user", "content": "Synthetic dispatch test."}], config)


def _pool(tmp_path, count=3, *, clock=time.time):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    paths, values = {}, {}
    for number in range(count):
        key_id = f"slot-{number + 1}"
        path = private / f"{key_id}.test"
        value = secrets.token_urlsafe(32)
        with open(path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
            stream.write(value)
        paths[key_id], values[key_id] = path, value
    config = PoolConfig(count, paths)
    pool = KeyPool(tmp_path / "runtime" / "api", config, clock=clock, poll=0.01)
    pool.prepare()
    return pool, values


def _hold_one_attempt(directory, pool_config, index, gate, release, notifications):
    pool = KeyPool(directory, pool_config, poll=0.01)
    config = _transport()
    if not gate.wait(15):
        raise RuntimeError("test worker did not receive its start signal")
    with pool.attempt(_request(index, config), config) as key_id:
        lease = LEASE.get()
        lease.dispatched()
        notifications.put((index, os.getpid(), key_id))
        if not release.wait(15):
            raise RuntimeError("test worker did not receive its release signal")
        lease.outcome("completed", usage={"input_tokens": 1}, response_complete=True)


def _wait_queue(pool, busy, waiting, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        queue = pool.status()["queue"]
        if sum(row["state"] == "busy" for row in queue) == busy and (
            sum(row["state"] == "queued" for row in queue) == waiting
        ):
            return queue
        time.sleep(0.02)
    pytest.fail("expected cross-process credential occupancy did not stabilize")


def _close(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(5)


def test_five_spawned_workers_respect_three_distinct_leases_and_fifo(tmp_path):
    pool, _ = _pool(tmp_path)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    releases = [context.Event() for _ in range(5)]
    notices = context.Queue()
    processes = [
        context.Process(target=_hold_one_attempt,
                        args=(pool.directory, pool.config, index, start, releases[index], notices))
        for index in range(5)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        start.set()
        first = [notices.get(timeout=15) for _ in range(3)]
        assert len({key for _, _, key in first}) == 3
        assert len({pid for _, pid, _ in first}) == 3
        queue = _wait_queue(pool, busy=3, waiting=2)
        waiting = [row["seed"] for row in queue if row["state"] == "queued"]
        with pytest.raises(Empty):
            notices.get(timeout=0.15)

        releases[first[0][0]].set()
        fourth = notices.get(timeout=8)
        assert fourth[0] == waiting[0]
        assert fourth[2] == first[0][2]  # The only credential released so far.
        _wait_queue(pool, busy=3, waiting=1)

        releases[first[1][0]].set()
        fifth = notices.get(timeout=8)
        assert fifth[0] == waiting[1]
        assert fifth[2] == first[1][2]
        _wait_queue(pool, busy=3, waiting=0)
        for release in releases:
            release.set()
        for process in processes:
            process.join(8)
        assert [p.exitcode for p in processes] == [0] * 5
        final = pool.status()
        assert final["queue"] == []
        assert sum(key["completed"] for key in final["keys"]) == 5
    finally:
        _close(started)
        notices.close()
        notices.join_thread()


def _exit_after_dispatch(pool_dir, pool_config, request_dir, config, request):
    pool = KeyPool(pool_dir, pool_config, poll=0.01)

    def exit_abruptly(_request, _config):
        LEASE.get().dispatched()
        os._exit(23)

    with round_lock(request_dir):
        run_request(request_dir, request, config, exit_abruptly, attempt_slot=pool.attempt)


def test_abrupt_worker_exit_frees_key_and_does_not_replay_ambiguous_request(tmp_path):
    pool, _ = _pool(tmp_path, count=1)
    config = _transport()
    request = _request(157, config)
    directory = tmp_path / "runtime" / "seed_157"
    processes = multiprocessing.get_context("spawn")
    worker = processes.Process(target=_exit_after_dispatch,
                               args=(pool.directory, pool.config, directory, config, request))
    worker.start()
    try:
        worker.join(15)
        assert worker.exitcode == 23
        attempts = directory / "turn_00"
        started = read(attempts / "attempt_00.started.json")
        assert started["request_id"] == request["request_id"]
        assert started["credential_id"] == "slot-1"
        assert not (attempts / "attempt_00.terminal.json").exists()

        state = pool.status()
        assert state["queue"] == []
        assert state["keys"][0]["status"] == "available"
        with pool.tx() as connection:
            abandoned = connection.execute(
                "SELECT state,dispatched,outcome FROM queue WHERE request_id=?", (request["request_id"],),
            ).fetchone()
        assert tuple(abandoned) == ("abandoned", 1, "ambiguous_worker_exit")

        def forbidden_send(*_args):
            pytest.fail("the ambiguous attempt was automatically replayed")

        with round_lock(directory), pytest.raises(TransportStop, match="requires reconciliation"):
            run_request(directory, request, config, forbidden_send, attempt_slot=pool.attempt)

        def complete(_request, live_config):
            lease = LEASE.get()
            lease.dispatched()
            lease.outcome("completed", response_complete=True)
            return ProviderReply("{}", "another-seed", live_config.served_model, {})

        another = tmp_path / "runtime" / "seed_887"
        with round_lock(another):
            answer = run_request(another, _request(887, config), config, complete, attempt_slot=pool.attempt)
        assert answer["credential_id"] == "slot-1"
        assert answer["response_id"] == "another-seed"
        assert not (attempts / "attempt_00.terminal.json").exists()
    finally:
        _close([worker])


class _Stream:
    def __init__(self, config):
        self.response = SimpleNamespace(headers=httpx.Headers({"x-request-id": "mock-provider-request"}))
        final = SimpleNamespace(
            id="mock-response", model=config.served_model, status="completed", output_text="{}",
            reasoning=SimpleNamespace(effort="ultra"), model_extra={"model_version": "mock-version-1"},
            usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12),
        )
        self.events = [SimpleNamespace(type="response.completed", response=final)]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.events)


@pytest.mark.parametrize("second_attempt_succeeds", [True, False])
def test_credential_429_rotates_real_transport_mock_sdk_without_resetting_budget(
        tmp_path, monkeypatch, second_attempt_succeeds):
    clock = [100.0]
    pool, values = _pool(tmp_path, clock=lambda: clock[0])
    config = _transport()
    runtime = tmp_path / "runtime"
    rounds = runtime / "round_00"
    model_lock = runtime / "model_lock.json"
    request = _request(157, config, logical_index=1)
    sent = []

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url, max_retries, timeout):
            self.key_id = next((key for key, value in values.items() if value == api_key), None)
            assert self.key_id is not None
            assert base_url == config.base_url and max_retries == 0 and timeout == config.timeout_seconds
            self.responses = SimpleNamespace(create=self.create)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create(self, **payload):
            sent.append((self.key_id, payload.pop("extra_headers")["X-Client-Request-Id"], payload))
            if len(sent) == 1 or not second_attempt_succeeds:
                response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/v1/responses"),
                                          headers={"x-ratelimit-scope": "credential", "retry-after": "1"})
                raise openai.RateLimitError("synthetic credential limit", response=response, body={})
            return _Stream(config)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-synthetic-environment-placeholder")

    def advance_clock(delay):
        clock[0] += delay

    def send_or_resume():
        with round_lock(rounds):
            return run_request(
                rounds, request, config, partial(call_responses, model_lock=model_lock), attempt_slot=pool.attempt,
                now=lambda: clock[0], sleep=advance_clock, jitter=lambda: 0.0,
            )

    if second_attempt_succeeds:
        for _ in range(2):  # Resume must reuse the completed response.
            assert send_or_resume()["response_id"] == "mock-response"
        assert read(model_lock)["reported_model_version"] == "mock-version-1"
    else:
        for _ in range(2):  # The other available key cannot reset the request budget on restart.
            with pytest.raises(TransportStop, match="http_429"):
                send_or_resume()
        assert not model_lock.exists()

    assert [(key, identity) for key, identity, _ in sent] == [
        ("slot-1", request["request_id"]), ("slot-2", request["request_id"]),
    ]
    assert all(payload == request["payload"] for _, _, payload in sent)
    assert os.environ["OPENAI_API_KEY"] == "unrelated-synthetic-environment-placeholder"
    turn = rounds / "turn_01"
    receipts = [read(turn / f"attempt_{i:02d}.terminal.json") for i in range(2)]
    assert [item["credential_id"] for item in receipts] == ["slot-1", "slot-2"]
    assert [item["status"] for item in receipts] == [
        "retryable", "completed" if second_attempt_succeeds else "stopped",
    ]
    assert not (turn / "attempt_02.started.json").exists()
    states = {key["id"]: key for key in pool.status()["keys"]}
    assert states["slot-1"]["status"] == "cooldown"
    assert states["slot-3"]["requests"] == 0 and states["slot-3"]["status"] == "available"
    assert sum(key["corrective_requests"] for key in states.values()) == 2

    credentials = [value.encode() for value in values.values()]
    leaks = []
    for path in runtime.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if any(secret in data for secret in credentials):
                leaks.append(path.relative_to(runtime).as_posix())
    assert not leaks, "synthetic raw credential material was present in a runtime artifact"

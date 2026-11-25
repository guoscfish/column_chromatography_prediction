from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE, KeyPool, PoolConfig


def _pool(root, clock):
    keys = {}
    for index in range(3):
        key_id = f"key_{index + 1}"
        path = root / key_id
        with path.open("w") as handle:
            handle.write("synthetic-key")
        path.chmod(0o600)
        keys[key_id] = path
    pool = KeyPool(root / "pool", PoolConfig(3, keys), clock=clock, poll=0.01)
    pool.prepare()
    return pool


def _request(index):
    return {"binding": {"seed": 157 + index, "round": 0}, "logical_index": 0, "request_id": str(index)}


def test_five_waiting_requests_share_three_disjoint_keys(tmp_path):
    clock = lambda: 100
    pool = _pool(tmp_path, clock)
    transport = SimpleNamespace(max_delay_seconds=10)
    released = Event()
    three_active = Event()
    guard = Lock()
    active = set()
    concurrency = []

    def run(index):
        with pool.attempt(_request(index), transport) as key:
            lease = LEASE.get()
            lease.dispatched()
            with guard:
                assert key not in active
                active.add(key)
                concurrency.append(len(active))
                if len(active) == 3:
                    three_active.set()
            assert released.wait(timeout=6)
            with guard:
                active.remove(key)
            lease.outcome("completed", usage={"output_tokens": 3}, response_complete=True)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run, i) for i in range(5)]
        try:
            assert three_active.wait(timeout=6)
        finally:
            released.set()
        for future in futures:
            future.result(timeout=8)
    status = pool.status()
    assert max(concurrency) == 3
    assert sum(k["requests"] for k in status["keys"]) == 5
    assert sum(k["completed"] for k in status["keys"]) == 5
    assert sum(k["usage"]["output_tokens"] for k in status["keys"]) == 15


def test_credential_failure_and_shared_limit_have_different_scopes(tmp_path):
    now = [100]
    pool = _pool(tmp_path, lambda: now[0])
    transport = SimpleNamespace(max_delay_seconds=4)
    with pool.attempt(_request(0), transport) as denied_key:
        lease = LEASE.get()
        lease.dispatched()
        lease.outcome("http_403")
    with pool.attempt(_request(1), transport) as limited_key:
        assert limited_key != denied_key
        lease = LEASE.get()
        lease.dispatched()
        lease.outcome("http_429", scope="shared", retry_after=10)
    with pytest.raises(TransportStop, match="shared provider cooldown"):
        with pool.attempt(_request(2), transport):
            pytest.fail("another credential must not bypass a shared rate limit")
    now[0] += 10
    with pool.attempt(_request(3), transport) as key:
        assert key != denied_key
    states = {key["id"]: key["status"] for key in pool.status()["keys"]}
    assert states[denied_key] == "disabled"

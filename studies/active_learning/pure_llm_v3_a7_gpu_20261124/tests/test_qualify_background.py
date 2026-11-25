from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import KeyPool, PoolConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import qualify_background as probe


class FakeError(Exception):
    pass


class FakeStatusError(FakeError):
    def __init__(self, status, *, code, param=None):
        self.status_code = status
        self.body = {"error": {"code": code, "param": param}}
        self.response = SimpleNamespace(headers={"x-request-id": "request_test"})


class FakeResponse:
    def __init__(self, status):
        self.body = {"id": "resp_synthetic", "model": probe.MODEL, "reasoning": {"effort": probe.EFFORT},
                     "background": True, "store": False, "status": status}
        self.headers = {"x-request-id": "request_test"}
        if status == "completed":
            self.body.update(
                usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
                       "output_tokens_details": {"reasoning_tokens": 10}},
                output=[{"type": "message", "content": [
                    {"type": "output_text", "text": '{"candidate_id":"SYN-A","difference":3}'}
                ]}],
            )

    def parse(self):
        return self

    def model_dump(self, **kwargs):
        return self.body


@pytest.fixture
def ledger_pool(tmp_path):
    ledger = tmp_path / "probe"
    ledger.mkdir(mode=0o700)
    files = {}
    for name in ("key_01", "key_02"):
        path = tmp_path / name
        path.write_text("synthetic-only-" + name)
        path.chmod(0o600)
        files[name] = path
    pool = KeyPool(tmp_path / "pool", PoolConfig(2, files))
    pool.prepare()
    transport = TransportConfig("token4research", probe.MODEL, probe.MODEL, probe.EFFORT,
                                probe.BASE_URL, 90, 4096, sdk_version=probe.SDK_VERSION)
    probe.plan(ledger)
    return ledger, pool, transport


def fake_client(monkeypatch, post, get):
    calls = []

    @contextmanager
    def client(key, limit):
        def create(**kwargs):
            calls.append(("POST", key, kwargs))
            return post()

        def retrieve(response_id):
            calls.append(("GET", key, response_id))
            return get()

        sdk = SimpleNamespace(APIStatusError=FakeStatusError, OpenAIError=FakeError)
        resources = SimpleNamespace(create=create, retrieve=retrieve)
        yield SimpleNamespace(responses=SimpleNamespace(with_raw_response=resources)), sdk

    monkeypatch.setattr(probe, "_client", client)
    return calls


def test_durable_one_post_same_id_and_credential_retrieval(ledger_pool, monkeypatch):
    ledger, pool, config = ledger_pool
    phases = iter(("in_progress", "completed"))
    calls = fake_client(monkeypatch, lambda: FakeResponse("queued"), lambda: FakeResponse(next(phases)))
    with pytest.raises(PermissionError):
        probe.submit(ledger, pool, config, paid=False)
    assert calls == []
    first = probe.submit(ledger, pool, config, paid=True)
    assert first["submission"]["status"] == "queued"
    assert probe.submit(ledger, pool, config, paid=True) == first
    assert probe.poll(ledger, pool, config)["result"]["status"] == "in_progress"
    done = probe.poll(ledger, pool, config)
    assert done["same_id_retrieval_completed"] and done["result"]["synthetic_answer_ok"]
    assert done["result"]["usage"]["reasoning_tokens"] == 10
    assert probe.poll(ledger, pool, config) == done
    assert [(method, key) for method, key, _ in calls] == [
        ("POST", "synthetic-only-key_01"), ("GET", "synthetic-only-key_01"), ("GET", "synthetic-only-key_01"),
    ]
    args = calls[0][2]
    assert (args["background"], args["store"], args["stream"], args["max_output_tokens"]) == (True, False, False, 4096)
    assert args["reasoning"] == {"effort": "xhigh"}
    assert calls[1][2] == calls[2][2] == "resp_synthetic"
    assert pool.status()["queue"] == []
    assert not any("synthetic-only-" in f.read_text() for f in ledger.rglob("*.json"))


def test_interruption_before_response_id_never_resubmits(ledger_pool, monkeypatch):
    ledger, pool, config = ledger_pool

    def lost_process():
        raise RuntimeError("simulated process loss after submit")

    calls = fake_client(monkeypatch, lost_process, lambda: pytest.fail("must not look up an unknown ID"))
    with pytest.raises(RuntimeError):
        probe.submit(ledger, pool, config, paid=True)
    assert probe.status(ledger)["phase"] == "submission_unknown"
    assert probe.submit(ledger, pool, config, paid=True)["phase"] == "submission_unknown"
    assert probe.poll(ledger, pool, config)["phase"] == "submission_unknown"
    assert [request[0] for request in calls] == ["POST"]


def test_background_rejection_is_sanitized_and_sealed(ledger_pool, monkeypatch):
    ledger, pool, config = ledger_pool

    def rejected():
        raise FakeStatusError(400, code="unsupported_parameter", param="background")

    calls = fake_client(monkeypatch, rejected, lambda: pytest.fail("no durable response ID"))
    rejected_state = probe.submit(ledger, pool, config, paid=True)
    assert rejected_state["phase"] == "submission_failed"
    assert rejected_state["result"]["http_status"] == 400
    assert rejected_state["result"]["provider_param"] == "background"
    assert not pool.status()["limits"]["shared_disabled"]
    assert probe.poll(ledger, pool, config) == rejected_state
    assert probe.submit(ledger, pool, config, paid=True) == rejected_state
    assert len(calls) == 1


def test_mismatched_retrieval_identity_is_never_a_completion(ledger_pool, monkeypatch):
    ledger, pool, config = ledger_pool

    def foreign():
        result = FakeResponse("completed")
        result.body["id"] = "resp_different"
        return result

    calls = fake_client(monkeypatch, lambda: FakeResponse("queued"), foreign)
    probe.submit(ledger, pool, config, paid=True)
    resumed = probe.poll(ledger, pool, config)
    assert not resumed["same_id_retrieval_completed"]
    assert resumed["result"] == {"error": "ValueError"}
    assert calls[-1][2] == "resp_synthetic"

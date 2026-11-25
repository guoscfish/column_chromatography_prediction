import asyncio
from contextlib import contextmanager
import json
import threading
import time
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    TransportConfig, TransportFailure, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import provider
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts


def _event(kind, identity="resp_stream", *, status="in_progress"):
    fields = {"id": identity, "object": "response", "status": status, "model": "mock-model", "output": []}
    if status == "completed":
        fields.update(reasoning={"effort": "max"}, usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
        fields["output"] = [{"id": "msg_stream", "type": "message", "role": "assistant", "status": "completed",
                              "content": [{"type": "output_text", "text": "synthetic answer", "annotations": []}]}]
    return ("data: " + json.dumps({"type": kind, "response": fields}) + "\n\n").encode()


@pytest.fixture
def mock_provider(tmp_path, monkeypatch):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    original_client = openai.OpenAI
    original_key = "test-key-canary"
    config = TransportConfig("offline", "mock-model", "mock-model", "max", "https://example.invalid/v1",
                             15, 4096, sdk_version=openai.__version__)
    request = make_request({"seed": 157, "round": 0}, 0, [], config)

    @contextmanager
    def fake_key():
        yield original_key, "fake"

    monkeypatch.setattr(provider, "api_key", fake_key)

    def call(source, *, close=None, request_id="req_stream"):
        class MockStream(httpx.SyncByteStream):
            def __iter__(self):
                yield from source()

            def close(self):
                if close is not None:
                    close()

        def handler(_request):
            return httpx.Response(200, stream=MockStream(), headers={
                "content-type": "text/event-stream", "x-request-id": request_id, "authorization": original_key,
            })

        def openai_client(**kwargs):
            assert kwargs["max_retries"] == 0
            return original_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(openai, "OpenAI", openai_client)
        progress, receipt, raw, outcomes = [], [], [], []
        token = provider.LEASE.set(SimpleNamespace(path=None, dispatched=lambda: None,
                                                   outcome=lambda code, **details: outcomes.append(code)))
        try:
            try:
                result = provider.call_responses(request, config, model_lock=tmp_path / "model-lock.json",
                                                 capture_response=raw.append, capture_receipt=receipt.append,
                                                 progress=progress.append)
            except TransportFailure as error:
                result = error
        finally:
            provider.LEASE.reset(token)
        return SimpleNamespace(result=result, progress=progress, receipt=receipt, raw=raw, outcomes=outcomes,
                               secret=original_key)

    return call


def test_terminal_is_captured_without_reading_broken_tail(mock_provider):
    import httpx

    def body():
        yield _event("response.created")
        yield _event("response.completed", status="completed")
        pytest.fail("read after the definitive provider completion")

    def close():
        raise httpx.RemoteProtocolError("mock cleanup error after completion")

    check = mock_provider(body, close=close)
    assert check.result.text == "synthetic answer"
    assert check.result.response_id == "resp_stream"
    assert check.raw[0]["status"] == "completed"
    assert check.receipt[0]["raw_response_sha256"] == digest(check.raw[0])
    assert check.outcomes == ["completed"]
    assert [row["phase"] for row in check.progress][:3] == ["request_dispatched", "http_headers", "sse_event"]
    assert check.progress[1]["provider_request_id"] == "req_stream"
    assert check.progress[2]["response_id"] == "resp_stream"


@pytest.mark.parametrize(("exception_name", "failure_code"), [
    ("RemoteProtocolError", "sse_frame_error"),
    ("ReadTimeout", "timeout"),
])
def test_interrupted_stream_reports_ids_and_distinguishes_transport_cause(mock_provider, exception_name, failure_code):
    import httpx

    def body():
        yield _event("response.created")
        raise getattr(httpx, exception_name)("truncated fixture")

    check = mock_provider(body)
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == (failure_code, True, False)
    assert check.progress[1]["http_status"] == 200
    assert check.progress[1]["provider_request_id"] == "req_stream"
    assert check.progress[2]["response_id"] == "resp_stream"
    assert check.progress[-1]["error_type"] == exception_name
    assert not check.raw and not check.receipt
    assert check.secret not in json.dumps(check.progress)


def test_completion_must_agree_with_previously_reported_response_id(mock_provider):
    def body():
        yield _event("response.created")
        yield _event("response.completed", "resp_different", status="completed")

    check = mock_provider(body, request_id="test-key-canary")
    assert check.result.code == "response_id_mismatch" and check.result.ambiguous
    assert not check.raw and not check.receipt
    assert check.secret not in json.dumps(check.progress)
    assert "resp_different" not in json.dumps(check.progress)


def test_sse_error_retains_only_structured_failure_fields(mock_provider):
    def body():
        yield _event("response.created")
        error = {"type": "error", "sequence_number": 1, "code": "server_error",
                 "param": "test-key-canary", "message": "full provider message has sensitive content"}
        yield ("data: " + json.dumps(error) + "\n\n").encode()

    check = mock_provider(body)
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("error", True, False)
    event = next(row for row in check.progress if row.get("event_type") == "error")
    assert event["provider_error_code"] == "server_error"
    assert "provider_error_param" not in event and "message" not in event
    assert check.secret not in json.dumps(check.progress)
    assert not check.raw and not check.receipt


def test_overdue_terminal_remains_ambiguous_without_auto_recovery_capture():
    event = SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_late", status="completed"))

    class LateStream:
        response = SimpleNamespace(headers={})

        def __iter__(self):
            yield event

    captured = []
    with pytest.raises(TransportFailure, match="timeout") as exc:
        provider._collect(LateStream(), deadline=0, capture_response=captured.append)
    assert exc.value.ambiguous and not exc.value.retryable
    assert not captured


@pytest.fixture
def async_mock_provider(tmp_path, monkeypatch):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    original_http_client = httpx.AsyncClient
    secret = "test-key-canary"
    config = TransportConfig("offline", "mock-model", "mock-model", "max", "https://example.invalid/v1",
                             15, 4096, sdk_version=openai.__version__)
    request = make_request({"seed": 157, "round": 0}, 0, [], config)
    policy = RequestTimeouts(.1, .2, .9, 1, 1)

    @contextmanager
    def fake_key():
        yield secret, "fixture"

    monkeypatch.setattr(provider, "api_key", fake_key)

    def call(source, *, proxy=False, resolver=None, close=None):
        progress, receipt, raw, outcomes, requests = [], [], [], [], []

        class MockStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for part in source():
                    yield part

            async def aclose(self):
                if close is not None:
                    await close()

        async def handler(item):
            requests.append(item.method)
            assert item.extensions["timeout"] == {"connect": .1, "read": .9, "write": .2, "pool": .1}
            if resolver is not None:
                await asyncio.get_running_loop().run_in_executor(None, resolver)
            trace = item.extensions["trace"]
            network = "proxy" if proxy else "connection"
            for phase in ("connection.connect_tcp", f"{network}.start_tls", "http11.send_request_body"):
                for state in ("started", "complete"):
                    await trace(f"{phase}.{state}", {"authorization": secret, "request": item})
            return httpx.Response(200, stream=MockStream(), headers={
                "content-type": "text/event-stream", "x-request-id": "req_stream", "authorization": secret,
            })

        class MockClient(original_http_client):
            def __init__(self, **kwargs):
                assert kwargs["follow_redirects"] is False
                super().__init__(**kwargs, transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "AsyncClient", MockClient)
        token = provider.LEASE.set(SimpleNamespace(path=None, dispatched=lambda: None,
                                                   outcome=lambda code, **details: outcomes.append(code)))
        started = time.monotonic()
        try:
            try:
                result = provider.call_responses(request, config, model_lock=tmp_path / "async-lock.json",
                                                 capture_response=raw.append, capture_receipt=receipt.append,
                                                 progress=progress.append, timeouts=policy)
            except TransportFailure as error:
                result = error
        finally:
            provider.LEASE.reset(token)
        return SimpleNamespace(result=result, progress=progress, receipt=receipt, raw=raw, outcomes=outcomes,
                               elapsed=time.monotonic() - started, requests=requests)

    return call


def test_hard_total_interrupts_blocked_sse_before_idle_timeout(async_mock_provider):
    cancelled = []

    async def body():
        await asyncio.sleep(.3)
        yield _event("response.created")
        try:
            await asyncio.sleep(20)
        finally:
            cancelled.append(True)

    check = async_mock_provider(body)
    assert len(check.requests) == 1 and check.requests[0] == "POST"
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("timeout", True, False)
    assert check.progress[-1]["error_type"] == "hard_wall_deadline"
    assert check.outcomes == ["timeout"] and cancelled == [True]
    assert .9 <= check.elapsed < 3
    assert not check.raw and not check.receipt
    assert any(row.get("response_id") == "resp_stream" for row in check.progress)


def test_async_completion_records_proxy_milestones_and_usage(async_mock_provider):
    async def body():
        yield _event("response.created")
        yield _event("response.completed", status="completed")
        pytest.fail("read beyond the definitive terminal event")

    check = async_mock_provider(body, proxy=True)
    assert check.result.response_id == "resp_stream"
    assert check.receipt[0]["raw_response_sha256"] == digest(check.raw[0])
    assert check.outcomes == ["completed"]
    assert [row["network_phase"] for row in check.progress if row["phase"] == "http_phase"] == [
        "dns_tcp", "tls", "send_body",
    ]
    assert any(row.get("explicit_proxy") is True for row in check.progress)
    assert any(row.get("sse_event_count") == 2 and row["sse_interval_seconds"] >= 0 for row in check.progress)
    assert check.progress[-1]["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert "test-key-canary" not in json.dumps(check.progress)


def test_hard_total_does_not_drain_a_blocked_resolver_executor(async_mock_provider):
    started = threading.Event()
    release = threading.Event()

    def blocked_resolver():
        started.set()
        release.wait(5)

    async def body():
        yield _event("response.created")
        pytest.fail("received an SSE event before connection established")

    try:
        check = async_mock_provider(body, resolver=blocked_resolver)
        assert started.is_set()
        assert check.requests == ["POST"]
        assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("timeout", True, False)
        assert check.progress[-1]["error_type"] == "hard_wall_deadline"
        assert .9 <= check.elapsed < 3
        assert not check.raw and not check.receipt
    finally:
        release.set()


def test_predeadline_completion_survives_stalled_stream_teardown(async_mock_provider):
    async def body():
        yield _event("response.completed", status="completed")

    async def close():
        await asyncio.sleep(20)

    check = async_mock_provider(body, close=close)
    assert check.result.response_id == "resp_stream"
    assert check.receipt[0]["raw_response_sha256"] == digest(check.raw[0])
    assert check.outcomes == ["completed"] and check.requests == ["POST"]
    assert check.progress[-1]["error_type"] == "hard_wall_deadline"
    assert .9 <= check.elapsed < 3

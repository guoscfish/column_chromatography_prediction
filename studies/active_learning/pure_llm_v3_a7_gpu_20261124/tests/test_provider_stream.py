import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
import ssl
import threading
import time
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_JSON_IPV4, RESPONSES_STREAM_IPV4, ResponseMessagePhaseError, TransportConfig, TransportFailure,
    make_request, responses_call,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import dispatcher, provider
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_provider_websocket import run_ws  # noqa: F401


def _frame(kind, **fields):
    return ("data: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


def _event(kind, identity="resp_stream", *, status="in_progress", **event_fields):
    fields = {"id": identity, "object": "response", "status": status, "model": "mock-model", "output": []}
    if status == "completed":
        fields.update(reasoning={"effort": "max"}, usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
        fields["output"] = [{"id": "msg_stream", "type": "message", "role": "assistant", "status": "completed",
                              "content": [{"type": "output_text", "text": "synthetic answer", "annotations": []}]}]
    return _frame(kind, response=fields, **event_fields)


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


@pytest.mark.parametrize(("different_id", "reported_id"), [
    ("resp_different", "resp_different"),
    ("test-key-canary", None),
    ("contains whitespace", None),
])
def test_completion_must_agree_with_previously_reported_response_id(mock_provider, different_id, reported_id):
    def body():
        yield _event("response.created")
        yield _event("response.completed", different_id, status="completed")

    check = mock_provider(body, request_id="test-key-canary")
    assert check.result.code == "response_id_mismatch" and check.result.ambiguous
    assert not check.raw and not check.receipt
    assert check.secret not in json.dumps(check.progress)
    mismatch = next(row for row in check.progress
                    if row["phase"] == "sse_event" and row.get("error_type") == "response_id_mismatch")
    assert mismatch["response_id"] == "resp_stream"
    assert mismatch.get("conflicting_response_id") == reported_id
    assert mismatch["conflicting_event_type"] == "response.completed"
    assert mismatch["response_id_matches_transport_alias"] is False


def test_response_id_matching_transport_alias_still_requires_reconciliation(mock_provider):
    request_id = "00000001-0002-4000-8000-000000000003"
    alias = "resp_00000001000240008000000000000003"

    def body():
        yield _event("response.created")
        yield _event("response.completed", alias, status="completed")

    check = mock_provider(body, request_id=request_id)
    assert check.result.code == "response_id_mismatch" and check.result.ambiguous
    assert not check.raw and not check.receipt
    mismatch = next(row for row in check.progress if row.get("conflicting_response_id") == alias)
    assert mismatch["conflicting_event_type"] == "response.completed"
    assert mismatch["response_id_matches_transport_alias"] is True


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

    def call(source, *, proxy=False, resolver=None, close=None, before_response=None, with_progress=True,
             fail_connect_tcp=False, ipv4=False, peer="192.0.2.1", http_version=b"HTTP/1.1", with_timeouts=True,
             preflight_status=401, preflight_peer="192.0.2.1", preflight_delay=0, preflight_error=None,
             sse_inactivity_seconds=None, json_response=False, http_status=200, media_type=None, with_capture=True,
             ipv4_preflight_retry=False, preflight_sequence=(), before_provider_dispatch=None):
        progress, receipt, raw, outcomes, requests, network_options, dispatches = [], [], [], [], [], [], []
        request_order, probe_calls = [], []
        ipv4 = ipv4 or json_response
        selected = (replace(config, wire_mode=RESPONSES_JSON_IPV4, requested_model="gpt-6.2-astra",
                            served_model="gpt-6.2-astra", reasoning_effort="xhigh", timeout_seconds=policy.total_seconds)
                    if json_response else replace(config, wire_mode=RESPONSES_STREAM_IPV4) if ipv4 else config)
        signed_request = make_request(request["binding"], request["logical_index"], request["payload"]["input"], selected)
        if json_response:
            same_sse = make_request(request["binding"], request["logical_index"], request["payload"]["input"],
                                    replace(selected, wire_mode=RESPONSES_STREAM_IPV4))
            assert signed_request["payload"] == {**same_sse["payload"], "stream": False}
            assert signed_request["payload_sha256"] != same_sse["payload_sha256"]
        else:
            assert signed_request["payload"] == request["payload"]
            assert signed_request["payload_sha256"] == request["payload_sha256"]
        if ipv4:
            assert signed_request["request_id"] != request["request_id"]

        class MockStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for part in source():
                    yield part

            async def aclose(self):
                if close is not None:
                    await close()

        async def handler(item):
            assert item.extensions["timeout"] == {"connect": .1, "read": .9, "write": .2, "pool": .1}
            if item.method == "GET":
                assert ipv4 and item.url.path == "/v1/models"
                assert "authorization" not in item.headers and "cookie" not in item.headers
                trace = item.extensions.get("trace")
                assert (trace is not None) == (ipv4_preflight_retry and with_progress)
                probe_number = len(probe_calls)
                probe_calls.append(True)
                probe = preflight_sequence[probe_number] if preflight_sequence else {}
                if trace is not None:
                    for name in ("connect_tcp.started", "connect_tcp.complete", "start_tls.started"):
                        await trace(f"connection.{name}", {"authorization": secret})
                try:
                    if delay := probe.get("delay", preflight_delay):
                        await asyncio.sleep(delay)
                    if error := probe.get("error", preflight_error):
                        raise error
                except (Exception, asyncio.CancelledError):
                    if trace is not None:
                        await trace("connection.start_tls.failed", {"authorization": secret})
                    raise
                if trace is not None:
                    await trace("connection.start_tls.complete", {})
                return httpx.Response(probe.get("status", preflight_status), stream=probe.get("stream"), headers={
                    "Set-Cookie": "preflight=ignored; Path=/",
                    "Retry-After": "4",
                }, extensions={
                    "network_stream": SimpleNamespace(get_extra_info=lambda _: (preflight_peer, 443)),
                    "http_version": b"HTTP/1.1",
                })
            assert "cookie" not in item.headers
            if resolver is not None:
                await asyncio.get_running_loop().run_in_executor(None, resolver)
            trace = item.extensions["trace"]
            if before_response is not None:
                await before_response(trace, item)
            network = "proxy" if proxy else "connection"
            for phase in ("connection.connect_tcp", f"{network}.start_tls", "http11.send_request_body"):
                for state in ("started", "complete"):
                    await trace(f"{phase}.{state}", {"authorization": secret, "request": item})
            extensions = {}
            if ipv4:
                extensions = {"network_stream": SimpleNamespace(get_extra_info=lambda _: (peer, 443)),
                              "http_version": http_version}
            return httpx.Response(http_status, stream=MockStream(), headers={
                "content-type": media_type or ("application/json; charset=utf-8" if json_response else "text/event-stream"),
                "x-request-id": "req_stream", "authorization": secret,
            }, extensions=extensions)

        def direct_transport(**kwargs):
            network_options.append(kwargs)
            return httpx.MockTransport(handler)

        if ipv4:
            monkeypatch.setattr(httpx, "AsyncHTTPTransport", direct_transport)

        class MockClient(original_http_client):
            def __init__(self, **kwargs):
                assert kwargs["follow_redirects"] is False
                if ipv4:
                    assert kwargs["trust_env"] is False
                    super().__init__(**kwargs)
                    return
                transport = httpx.MockTransport(handler)
                if fail_connect_tcp:
                    httpcore = pytest.importorskip("httpcore")

                    class BlockedTCP(httpcore.AsyncNetworkBackend):
                        async def connect_tcp(self, host, port, timeout, **_kwargs):
                            assert (host, port, timeout) == ("example.invalid", 443, .1)
                            raise httpcore.ConnectTimeout(secret)

                    # Exercise the real HTTPcore trace and HTTPX/SDK exception wrappers.
                    transport = httpx.AsyncHTTPTransport()
                    transport._pool._network_backend = BlockedTCP()
                super().__init__(**kwargs, transport=transport)

            async def send(self, request, **kwargs):
                item = request
                requests.append(item.method)
                request_order.append(item.method)
                assert item.url.host == "example.invalid"
                if item.method == "POST":
                    assert json.loads(item.content) == signed_request["payload"]
                    assert item.headers["x-client-request-id"] == signed_request["request_id"]
                return await super().send(item, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", MockClient)

        def mark_dispatched():
            request_order.append("lease_dispatched")
            dispatches.append(True)

        def guard():
            request_order.append("provider_guard")
            before_provider_dispatch()

        token = provider.LEASE.set(SimpleNamespace(path=None, dispatched=mark_dispatched,
                                                   outcome=lambda code, **details: outcomes.append((code, details))))
        started = time.monotonic()
        try:
            try:
                result = provider.call_responses(signed_request, selected, model_lock=tmp_path / "async-lock.json",
                                                 capture_response=raw.append if with_capture else None,
                                                 capture_receipt=receipt.append if with_capture else None,
                                                 progress=progress.append if with_progress else None,
                                                 timeouts=policy if with_timeouts else None,
                                                 sse_inactivity_seconds=sse_inactivity_seconds,
                                                 ipv4_preflight_retry=ipv4_preflight_retry,
                                                 before_provider_dispatch=guard if before_provider_dispatch else None)
            except TransportFailure as error:
                result = error
            except PermissionError as error:
                result = error
        finally:
            provider.LEASE.reset(token)
        return SimpleNamespace(result=result, progress=progress, receipt=receipt, raw=raw,
                               outcomes=[code for code, _ in outcomes], outcome_details=outcomes,
                               elapsed=time.monotonic() - started, requests=requests,
                               network_options=network_options, dispatches=dispatches, config=selected,
                               request=signed_request, request_order=request_order)

    return call


def test_http_200_nested_sse_error_preserves_only_allowlisted_diagnostics(async_mock_provider, tmp_path):
    async def body():
        yield _event("response.created")
        yield _frame("error", error={"code": "server_error", "type": "test-key-canary", "status": 502,
                                      "param": "reasoning.effort", "message": "unpublished response text",
                                      "headers": {"authorization": "test-key-canary"}})

    check = async_mock_provider(body, ipv4=True)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert (check.result.code, check.result.retryable, check.result.ambiguous) == ("provider_stream_error", False, True)
    assert not check.raw and not check.receipt
    assert check.progress[-1]["error_type"] == "APIError"
    attempt = tmp_path / "local-attempt"
    attempt.mkdir()
    save = dispatcher._transport_progress(attempt, {"job_sha256": "a" * 64, "request": check.request})
    for update in check.progress:
        save(update)
    snapshot = json.loads((attempt / dispatcher.PROGRESS_FILENAME).read_text())
    assert {key: value for key, value in snapshot.items() if key.startswith("provider_error_")} == {
        "provider_error_code": "server_error", "provider_error_type": "unrecognized",
        "provider_error_param": "reasoning.effort", "provider_error_status": 502,
    }
    assert snapshot["http_status"] == 200 and snapshot["response_id"] == "resp_stream"
    saved = json.dumps([snapshot, check.progress])
    assert "test-key-canary" not in saved and "unpublished response text" not in saved and "authorization" not in saved


@pytest.mark.parametrize("status", [429, 429.0, None])
def test_only_explicit_nested_429_applies_shared_hold_without_replay(async_mock_provider, status):
    async def body():
        yield _event("response.created")
        yield _frame("error", error={"code": "rate_limit_exceeded", "type": "rate_limit_error", "status": status,
                                      "retry_after": 19, "rate_limit_scope": "credential"})

    check = async_mock_provider(body, ipv4=True)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert check.result.ambiguous and not check.result.retryable and not check.raw and not check.receipt
    code, details = check.outcome_details[0]
    assert code == check.result.code == ("http_429" if type(status) is int else "provider_stream_error")
    assert details["scope"] == "shared" and details["retry_after"] == (19 if type(status) is int else None)


def test_generic_sdk_error_does_not_claim_a_provider_error_frame(async_mock_provider):
    import openai

    async def body():
        yield _event("response.created")
        raise openai.OpenAIError("test-key-canary")

    check = async_mock_provider(body, ipv4=True)
    assert check.requests == ["GET", "POST"] and check.result.code == "sdk_error" and check.result.ambiguous
    assert check.progress[-1]["error_type"] == "OpenAIError"
    assert not any(key.startswith("provider_error_") for key in check.progress[-1])
    assert "test-key-canary" not in json.dumps(check.progress)


def _json_payload(**changes):
    return {
        "id": "resp_json", "object": "response", "status": "completed", "model": "gpt-6.2-astra",
        "store": False, "background": False, "reasoning": {"effort": "xhigh"},
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                  "output_tokens_details": {"reasoning_tokens": 1}, "input_tokens_details": {"cached_tokens": 2}},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "synthetic answer"}]}],
        "model_version": "astra-version", "vendor_field": {"record_exactly": True},
        **changes,
    }


def _json_body(payload):
    async def body():
        content = json.dumps(payload).encode()
        yield content[:40]
        yield content[40:]
    return body


def _message(text, phase=None, *, status="completed", role="assistant"):
    item = {"type": "message", "role": role, "status": status, "id": f"msg_{phase or 'legacy'}",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    if phase is not None:
        item["phase"] = phase
    return item


def _sse_completed_output(output):
    return _frame("response.completed", response=_json_payload(id="resp_stream", model="mock-model",
                                                               output=output, reasoning={"effort": "max"}))


@pytest.mark.parametrize("wire", ["sse", "json", "websocket"])
def test_only_explicit_completed_final_enters_reply_and_recovery(request, tmp_path, wire):
    preview, final = '{"preview":true}', '{"choices":["candidate-fixture"]}'
    output = [_message(preview, "commentary"), _message(final, "final_answer")]

    if wire == "websocket":
        payload = _json_payload(output=output)

        async def frames():
            yield {"type": "response.completed", "response": payload}

        check = request.getfixturevalue("run_ws")(frames)
        reply, receipts = check.reply, check.receipts
    else:
        async def sse():
            yield _sse_completed_output(output)

        fixture = request.getfixturevalue("async_mock_provider")
        if wire == "json":
            check = fixture(_json_body(_json_payload(output=output)), json_response=True)
        else:
            check = fixture(sse, ipv4=True)
        reply, receipts = check.result, check.receipt
    assert reply.text == final
    assert check.raw[0]["output"] == output
    assert receipts[0]["raw_response_sha256"] == digest(check.raw[0])
    recovered, _ = provider.recover_completed_response(check.raw[0], check.config,
        model_lock=tmp_path / "recovered.json", metadata=receipts[0])
    assert recovered == reply


def test_multiple_untagged_messages_keep_legacy_concatenation(mock_provider):
    def body():
        yield _sse_completed_output([_message("legacy one"), _message(" and two")])

    check = mock_provider(body)
    assert check.result.text == "legacy one and two"


@pytest.mark.parametrize("output", [
    [_message("commentary only", "commentary")],
    [_message("first", "final_answer"), _message("second", "final_answer")],
    [_message("untagged"), _message("tagged", "final_answer")],
    [_message("unknown", "unexpected"), _message("tagged", "final_answer")],
    [_message("unfinished", "final_answer", status="in_progress")],
    [_message("wrong role", "final_answer", role="user")],
    [_message("  ", "final_answer")],
])
def test_ambiguous_terminal_phases_keep_raw_but_never_return_an_answer(async_mock_provider, tmp_path, output):
    async def sse():
        yield _sse_completed_output(output)

    for check in (async_mock_provider(sse, ipv4=True),
                  async_mock_provider(_json_body(_json_payload(output=output)), json_response=True)):
        assert isinstance(check.result, ResponseMessagePhaseError)
        assert not check.result.retryable and not check.result.ambiguous
        assert check.requests == ["GET", "POST"]
        assert check.raw[0]["output"] == output
        assert check.receipt[0]["raw_response_sha256"] == digest(check.raw[0])
        with pytest.raises(ResponseMessagePhaseError):
            provider.recover_completed_response(check.raw[0], check.config,
                model_lock=tmp_path / "unused.json", metadata=check.receipt[0])


def test_ipv4_json_captures_original_response_and_recovers_without_post(async_mock_provider, tmp_path):
    raw = _json_payload()
    check = async_mock_provider(_json_body(raw), json_response=True)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert check.result.text == "synthetic answer" and check.result.provider_request_id == "req_stream"
    assert check.outcomes == ["completed"] and check.raw == [raw]
    assert check.receipt == [{"raw_response_sha256": digest(raw), "provider_request_id": "req_stream",
                              "model_version": "astra-version", "reported_reasoning_effort": "xhigh"}]
    assert check.result.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                                  "reasoning_tokens": 1, "cached_tokens": 2}
    assert any(row["phase"] == "http_headers" for row in check.progress)
    assert all(row["phase"] != "sse_event" for row in check.progress)
    assert check.progress[-1]["phase"] == "transport_completed"
    assert "test-key-canary" not in json.dumps([check.progress, check.raw, check.receipt])
    restored, identity = provider.recover_completed_response(
        check.raw[0], check.config, model_lock=tmp_path / "recovered.json", metadata=check.receipt[0])
    assert restored == check.result and identity["reported_reasoning_effort"] == "xhigh"
    assert check.requests == ["GET", "POST"]
    with pytest.raises(TransportFailure, match="json_ipv4_dispatcher_required"):
        responses_call(check.request, check.config)


@pytest.mark.parametrize(("payload", "error"), [
    (_json_payload(status="in_progress"), "incomplete_response"),
    (_json_payload(store=True), "json_response_mode_mismatch"),
    (_json_payload(usage=None), "invalid_usage"),
    (_json_payload(id=""), "missing_response_identity"),
    (_json_payload(reasoning={"effort": "test-key-canary"}), "json_credential_echo"),
])
def test_ipv4_json_rejects_unsafe_or_nonterminal_envelopes(async_mock_provider, payload, error):
    check = async_mock_provider(_json_body(payload), json_response=True)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert check.result.code == error and not check.result.retryable
    assert not check.raw and not check.receipt
    assert "test-key-canary" not in json.dumps(check.progress)


@pytest.mark.parametrize(("overrides", "code"), [
    ({"model": "other-model"}, "served_model_mismatch"),
    ({"reasoning": {"effort": "max"}}, "served_reasoning_effort_mismatch"),
])
def test_ipv4_json_never_delivers_an_unapproved_model_or_effort(async_mock_provider, overrides, code):
    check = async_mock_provider(_json_body(_json_payload(**overrides)), json_response=True)
    assert check.requests == ["GET", "POST"] and check.result.code == code
    assert check.receipt[0]["raw_response_sha256"] == digest(check.raw[0])


def test_ipv4_json_only_starts_with_local_capture_and_bound_timeouts(async_mock_provider):
    check = async_mock_provider(None, json_response=True, with_capture=False)
    assert not check.requests and check.result.code == "json_ipv4_capture_required"
    check = async_mock_provider(None, json_response=True, with_timeouts=False)
    assert not check.requests and check.result.code == "direct_ipv4_timeouts_required"
    with pytest.raises(ValueError, match="SSE inactivity"):
        async_mock_provider(None, json_response=True, sse_inactivity_seconds=.2)


def test_ipv4_json_missing_complete_body_has_no_retry_or_capture(async_mock_provider):
    async def body():
        yield b'{"object":"response", "id":"resp_json"'
        await asyncio.sleep(20)

    check = async_mock_provider(body, json_response=True)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("timeout", True, False)
    assert check.progress[-1]["error_type"] == "hard_wall_deadline"
    assert not check.raw and not check.receipt and check.elapsed < 3


def test_ipv4_json_cannot_take_sse_or_retry_http_500(async_mock_provider):
    async def sse():
        pytest.fail("JSON transport must not parse an SSE body")
        yield b""

    check = async_mock_provider(sse, json_response=True, media_type="text/event-stream")
    assert check.requests == ["GET", "POST"] and check.result.code == "json_response_content_type_mismatch"
    assert not check.raw and not check.receipt
    check = async_mock_provider(_json_body({"error": {"code": "server_error"}}), json_response=True, http_status=500)
    assert check.requests == ["GET", "POST"] and check.result.code == "http_500"
    assert check.result.ambiguous and not check.result.retryable and not check.raw and not check.receipt


def test_ipv4_transport_preserves_wire_payload_and_proves_direct_http11(async_mock_provider):
    async def body():
        yield _event("response.completed", status="completed")

    check = async_mock_provider(body, ipv4=True)
    assert check.result.response_id == "resp_stream"
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    preflight = next(row for row in check.progress if row.get("network_phase") == "direct_ipv4_preflight")
    assert preflight["network_state"] == "complete" and "http_status" not in preflight
    assert check.network_options == [{
        "local_address": "0.0.0.0", "verify": True, "http1": True, "http2": False, "trust_env": False, "retries": 0,
    }]
    route = next(row for row in check.progress if row.get("network_phase") == "direct_ipv4_http11")
    assert route["network_state"] == "complete" and route["explicit_proxy"] is False
    assert "192.0.2.1" not in json.dumps(check.progress)


@pytest.mark.parametrize("route", [{"peer": "::1"}, {"http_version": b"HTTP/2"}])
def test_ipv4_transport_stops_on_route_mismatch(async_mock_provider, route):
    async def body():
        pytest.fail("unexpected route must not consume response content")
        yield b""

    check = async_mock_provider(body, ipv4=True, **route)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == (
        "direct_ipv4_route_mismatch", True, False,
    )
    assert not check.raw and not check.receipt


@pytest.mark.parametrize(("preflight", "error_code"), [
    ({"preflight_status": 404}, "connect_error"),
    ({"preflight_peer": "::1"}, "connect_error"),
    ({"preflight_status": 429}, "http_429"),
])
def test_ipv4_failed_preflight_never_reserves_or_sends_a_post(async_mock_provider, preflight, error_code):
    check = async_mock_provider(None, ipv4=True, **preflight)
    assert check.requests == ["GET"] and not check.dispatches
    assert (check.result.code, check.result.retryable, check.result.ambiguous) == (error_code, True, False)
    assert check.result.retry_after == (4 if error_code == "http_429" else None)
    assert check.progress[-1]["error_type"] == "direct_ipv4_preflight_failed"
    assert not check.raw and not check.receipt


def test_ipv4_tls_preflight_exception_is_safe_before_post(async_mock_provider):
    import httpx

    check = async_mock_provider(None, ipv4=True, preflight_error=httpx.ConnectTimeout("synthetic timeout"))
    assert check.requests == ["GET"] and not check.dispatches
    assert (check.result.code, check.result.retryable, check.result.ambiguous) == ("connect_error", True, False)


def test_ipv4_preflight_has_independent_deadline(async_mock_provider, monkeypatch):
    monkeypatch.setattr(provider, "DIRECT_IPV4_PREFLIGHT_SECONDS", .05)
    check = async_mock_provider(None, ipv4=True, preflight_delay=20)
    assert .05 <= check.elapsed < .75
    assert check.requests == ["GET"] and not check.dispatches
    assert (check.result.code, check.result.retryable, check.result.ambiguous) == ("connect_error", True, False)


@pytest.fixture
def short_retry_preflights(monkeypatch):
    monkeypatch.setattr(provider, "DIRECT_IPV4_RETRY_PROBE_SECONDS", .05)
    monkeypatch.setattr(provider, "DIRECT_IPV4_RETRY_DELAY_SECONDS", .01)


def test_opt_in_ipv4_retries_late_tls_before_guard_and_sends_one_post(async_mock_provider, short_retry_preflights):
    async def body():
        yield _event("response.completed", status="completed")

    check = async_mock_provider(body, ipv4=True, ipv4_preflight_retry=True, before_provider_dispatch=lambda: None,
                                preflight_sequence=[{"delay": 20}, {}])
    assert check.requests == ["GET", "GET", "POST"] and check.dispatches == [True]
    assert check.request_order == ["GET", "GET", "provider_guard", "lease_dispatched", "POST"]
    assert check.result.response_id == "resp_stream"
    phases = {row.get("network_phase"): row.get("network_state") for row in check.progress}
    assert phases["direct_ipv4_preflight_1_dns_tcp"] == "complete"
    assert phases["direct_ipv4_preflight_1_tls"] == "failed"
    assert phases["direct_ipv4_preflight_1"] == "failed"
    assert phases["direct_ipv4_preflight_2_tls"] == "complete"
    assert phases["direct_ipv4_preflight_2"] == "complete"
    assert "test-key-canary" not in json.dumps(check.progress)


def test_opt_in_ipv4_failed_probes_keep_physical_attempt_available(async_mock_provider, short_retry_preflights):
    check = async_mock_provider(None, ipv4=True, ipv4_preflight_retry=True, preflight_delay=20,
                                before_provider_dispatch=lambda: pytest.fail("guard ran before route readiness"))
    assert check.request_order == ["GET", "GET"] and not check.dispatches
    assert isinstance(check.result, provider._IPv4PreflightFailure) and check.result.retry_probe
    assert check.result.code == "connect_error" and not check.result.ambiguous
    assert .1 <= check.elapsed < .75 and not check.raw and not check.receipt


def test_opt_in_ipv4_keeps_shared_429_when_rejected_body_cannot_close_cleanly(async_mock_provider, short_retry_preflights):
    import httpx

    class RateLimitedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("must respect 429 headers without reading their body")
            yield b""

        async def aclose(self):
            raise httpx.ReadError("synthetic rate-limit response cleanup")

    check = async_mock_provider(None, ipv4=True, ipv4_preflight_retry=True,
                                preflight_sequence=[{"status": 429, "stream": RateLimitedBody()}])
    assert check.requests == ["GET"] and not check.dispatches
    assert (check.result.code, check.result.retry_after, check.result.retry_probe) == ("http_429", 4, False)
    assert check.outcome_details[0][1]["scope"] == "shared"


@pytest.mark.parametrize(("options", "code"), [
    ({"preflight_status": 403}, "ipv4_preflight_http_403"),
    ({"preflight_status": 404}, "connect_error"),
    ({"preflight_peer": "::1"}, "connect_error"),
])
def test_opt_in_ipv4_refuses_denied_or_wrong_route_in_one_probe(async_mock_provider, short_retry_preflights, options, code):
    check = async_mock_provider(None, ipv4=True, ipv4_preflight_retry=True, **options)
    assert check.requests == ["GET"] and not check.dispatches
    assert check.result.code == code and not check.result.retry_probe
    if code == "ipv4_preflight_http_403":
        assert not check.result.retryable


def test_opt_in_ipv4_does_not_retry_a_certificate_error(async_mock_provider, short_retry_preflights):
    import httpx

    error = httpx.ConnectError("synthetic certificate failure")
    error.__cause__ = ssl.SSLCertVerificationError("certificate untrusted")
    check = async_mock_provider(None, ipv4=True, ipv4_preflight_retry=True, preflight_error=error)
    assert check.requests == ["GET"] and not check.dispatches
    assert check.result.code == "connect_error" and not check.result.retry_probe


def test_opt_in_ipv4_dispatch_guard_failure_preserves_original_error(async_mock_provider, short_retry_preflights):
    cause = PermissionError("coordinator revoked the staged attempt")

    def revoked():
        raise cause

    check = async_mock_provider(None, ipv4=True, ipv4_preflight_retry=True, before_provider_dispatch=revoked)
    assert check.result is cause and check.request_order == ["GET", "provider_guard"]
    assert not check.dispatches and not check.outcomes


def test_ipv4_dispatch_guard_and_retry_require_explicit_direct_sse(async_mock_provider):
    with pytest.raises(ValueError, match="explicit IPv4 preflight retry policy"):
        async_mock_provider(None, ipv4=True, before_provider_dispatch=lambda: None)
    with pytest.raises(ValueError, match="explicit direct SSE policy"):
        async_mock_provider(None, json_response=True, ipv4_preflight_retry=True)


def test_opt_in_ipv4_does_not_retry_a_post_failure(async_mock_provider, short_retry_preflights):
    import httpx

    async def body():
        yield _event("response.created")
        raise httpx.ReadTimeout("synthetic timeout after submission")

    check = async_mock_provider(body, ipv4=True, ipv4_preflight_retry=True, preflight_sequence=[{"delay": 20}, {}])
    assert check.requests == ["GET", "GET", "POST"] and check.dispatches == [True]
    assert not isinstance(check.result, provider._IPv4PreflightFailure)
    assert check.result.code == "timeout" and check.result.ambiguous and not check.result.retryable
    assert not check.raw and not check.receipt


def test_ipv4_transport_refuses_missing_bound_timeouts_before_post(async_mock_provider):
    check = async_mock_provider(None, ipv4=True, with_timeouts=False)
    assert not check.requests and not check.network_options
    assert check.result.code == "direct_ipv4_timeouts_required"


def test_sse_inactivity_expires_while_waiting_for_first_event(async_mock_provider):
    async def body():
        await asyncio.sleep(5)
        pytest.fail("provider wait was not cancelled")
        yield b""

    check = async_mock_provider(body, ipv4=True, sse_inactivity_seconds=.12)
    assert check.requests == ["GET", "POST"] and check.dispatches == [True]
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("timeout", True, False)
    assert check.progress[-1]["error_type"] == "sse_inactivity_timeout"
    assert "sse_activity_seconds" not in check.progress[-1]
    assert not check.raw and not check.receipt


def test_sse_inactivity_ignores_repeated_and_nonprogress_frames(async_mock_provider):
    async def body():
        yield _event("response.created", sequence_number=0)
        yield _event("response.in_progress", sequence_number=1)
        yield _frame("response.output_item.done", sequence_number=2, output_index=0,
                     item={"id": "rs_stream", "type": "reasoning", "summary": []})
        delta = {"item_id": "msg_stream", "output_index": 1, "content_index": 0}
        yield _frame("response.output_text.delta", sequence_number=3, delta="draft", **delta)
        for sequence in range(4, 18):
            await asyncio.sleep(.025)
            yield b": relay keepalive\n\n"
            yield _frame("heartbeat", sequence_number=sequence)
            yield _event("response.in_progress", sequence_number=sequence)
            yield _frame("response.output_text.delta", sequence_number=sequence, delta="", **delta)
            yield _frame("response.output_text.delta", sequence_number=3, delta="draft", **delta)

    check = async_mock_provider(body, ipv4=True, sse_inactivity_seconds=.12)
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("timeout", True, False)
    assert check.progress[-1]["error_type"] == "sse_inactivity_timeout"
    assert any(row.get("event_type") == "response.output_item.done" for row in check.progress)
    first_text = next(row for row in check.progress if row.get("event_type") == "response.output_text.delta")
    last_frame = [row for row in check.progress if row["phase"] == "sse_event"][-1]
    assert first_text["elapsed_seconds"] < last_frame["elapsed_seconds"]
    assert first_text["elapsed_seconds"] == last_frame["sse_activity_seconds"]
    assert check.progress[-1]["sse_activity_seconds"] == first_text["elapsed_seconds"]
    assert not check.raw and not check.receipt


def test_meaningful_sse_progress_renews_deadline_and_terminal_is_captured(async_mock_provider):
    async def body():
        yield _event("response.created", sequence_number=0)
        await asyncio.sleep(.08)
        yield _frame("response.reasoning_summary_text.delta", sequence_number=1, item_id="rs_stream",
                     output_index=0, summary_index=0, delta="reasoning progress")
        await asyncio.sleep(.08)
        yield _frame("response.output_text.delta", sequence_number=2, item_id="msg_stream",
                     output_index=1, content_index=0, delta="synthetic answer")
        await asyncio.sleep(.08)
        yield _event("response.completed", status="completed", sequence_number=3)
        pytest.fail("read after the terminal frame")

    check = async_mock_provider(body, ipv4=True, sse_inactivity_seconds=.18)
    assert check.result.text == "synthetic answer"
    assert check.raw[0]["status"] == "completed"
    assert check.receipt[0]["raw_response_sha256"] == digest(check.raw[0])
    assert check.outcomes == ["completed"]


@pytest.mark.parametrize("duration", [0, True, float("inf"), float("nan"), 2])
def test_sse_inactivity_requires_a_valid_async_duration(async_mock_provider, duration):
    with pytest.raises(ValueError, match="SSE inactivity"):
        async_mock_provider(None, sse_inactivity_seconds=duration)


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


@pytest.mark.parametrize("with_progress", (False, True))
def test_traced_tcp_connect_timeout_is_a_bounded_retry_without_progress_dependency(async_mock_provider,
                                                                                  with_progress):
    check = async_mock_provider(None, with_progress=with_progress, fail_connect_tcp=True)
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("connect_error", False, True)
    assert check.outcomes == ["connect_error"] and check.requests == ["POST"]
    assert not check.raw and not check.receipt
    if with_progress:
        assert check.progress[-2]["network_phase"] == "dns_tcp" and check.progress[-2]["network_state"] == "failed"
        assert check.progress[-1]["error_type"] == "connect_timeout_before_post"
    else:
        assert not check.progress
    assert "test-key-canary" not in json.dumps(check.progress)
    assert "test-key-canary" not in str(check.result)


def test_proxy_connect_tunnel_timeout_before_provider_post_is_retryable(async_mock_provider):
    httpx = pytest.importorskip("httpx")
    httpcore = pytest.importorskip("httpcore")

    async def fail(trace, _item):
        tunnel = {"request": SimpleNamespace(method=b"CONNECT")}
        for stage in ("send_request_headers", "send_request_body", "receive_response_headers"):
            await trace(f"http11.{stage}.started", tunnel)
            await trace(f"http11.{stage}.complete", {})
        cause = httpcore.ConnectTimeout("proxy opened a tunnel but upstream TLS timed out")
        await trace("proxy.start_tls.started", {})
        await trace("proxy.start_tls.failed", {"exception": cause})
        raise httpx.ConnectTimeout(str(cause)) from cause

    check = async_mock_provider(None, before_response=fail)
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == ("connect_error", False, True)
    assert check.outcomes == ["connect_error"] and check.requests == ["POST"]
    assert any(row.get("explicit_proxy") is True for row in check.progress)


@pytest.mark.parametrize(("exception_name", "evidence", "code"), (
    ("ConnectTimeout", "post_started", "timeout"),
    ("ConnectTimeout", "unknown_method", "timeout"),
    ("ConnectTimeout", "send_without_start", "timeout"),
    ("ConnectTimeout", "no_trace", "timeout"),
    ("ConnectTimeout", "unrelated_failure", "timeout"),
    ("ReadTimeout", "traced", "timeout"),
    ("WriteTimeout", "traced", "timeout"),
    ("RemoteProtocolError", "traced", "connection_interrupted"),
))
def test_only_causally_bound_unsent_connect_timeout_is_retryable(async_mock_provider, exception_name, evidence, code):
    httpx = pytest.importorskip("httpx")
    httpcore = pytest.importorskip("httpcore")

    async def fail(trace, item):
        cause = httpcore.ConnectTimeout("offline connect timeout")
        if evidence != "no_trace":
            await trace("connection.connect_tcp.started", {})
            await trace("connection.connect_tcp.failed", {"exception": cause})
        if evidence == "post_started":
            await trace("http11.send_request_headers.started", {"request": item})
        if evidence == "unknown_method":
            await trace("http11.send_request_body.started", {})
        if evidence == "send_without_start":
            await trace("http11.send_request_headers.failed", {})
        if evidence == "unrelated_failure":
            cause = httpcore.ConnectTimeout("a different failed connection")
        raise getattr(httpx, exception_name)("offline failure") from cause

    check = async_mock_provider(None, before_response=fail)
    assert (check.result.code, check.result.ambiguous, check.result.retryable) == (code, True, False)
    assert check.outcomes == [code] and check.requests == ["POST"]
    assert not check.raw and not check.receipt

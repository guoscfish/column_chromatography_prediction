import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
import socket
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, RESPONSES_WEBSOCKET_IPV4, TransportConfig, TransportFailure, make_request, responses_call,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import provider
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts


def event(kind, identity="resp_exact", **changes):
    response = {"id": identity, "object": "response", "model": "gpt-6.2-astra", "status": "in_progress", "output": []}
    if kind == "response.completed":
        response.update(status="completed", store=False, background=False, reasoning={"effort": "xhigh"},
                        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                               "output_tokens_details": {"reasoning_tokens": 1}},
                        output=[{"type": "message", "id": "msg_exact", "role": "assistant", "status": "completed",
                                 "content": [{"type": "output_text", "text": "synthetic", "annotations": []}]}],
                        relay_extension={"carry": "exact"})
    return {"type": kind, "response": {**response, **changes}, "request_id": "req_model"}


@pytest.fixture
def run_ws(tmp_path, monkeypatch):
    openai = pytest.importorskip("openai")
    client = pytest.importorskip("websockets.asyncio.client")
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    secret = "test-ws-key-canary"
    config = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                             "https://token4research.cn/v1", 1, 32768, wire_mode=RESPONSES_WEBSOCKET_IPV4,
                             sdk_version=openai.__version__)
    messages = [{"role": "system", "content": "synthetic scope"}, {"role": "user", "content": "synthetic input"}]
    request = make_request({"seed": 6101, "round": 1}, 0, messages, config)
    assert request["payload"] == make_request(request["binding"], 0, messages,
                                               replace(config, wire_mode=RESPONSES_STREAM_IPV4))["payload"]

    @contextmanager
    def fake_key():
        yield secret, "fixture"

    monkeypatch.setattr(provider, "api_key", fake_key)

    def call(events, *, idle=.4, status=101, close_error=None, send_delay=0, with_capture=True):
        actions, frames, progress, raw, receipts, outcomes = [], [], [], [], [], []
        fake_tcp = SimpleNamespace(family=socket.AF_INET, close=lambda: actions.append("socket_close"))

        async def fake_tcp_open(host, port):
            assert (host, port) == ("token4research.cn", 443)
            actions.append("tcp")
            return fake_tcp

        monkeypatch.setattr(provider, "_connect_ipv4_socket", fake_tcp_open)

        class Wire:
            response = Response(101, "Switching Protocols", Headers({"x-request-id": "req_upgrade"}))

            async def send(self, message):
                actions.append("single_send")
                frames.append(json.loads(message))
                if send_delay:
                    await asyncio.sleep(send_delay)

            async def recv(self, *, decode):
                assert decode is False
                assert self.events is not None
                frame = await self.events.__anext__()
                return json.dumps(frame).encode()

            async def close(self, **_options):
                actions.append("websocket_close")
                if close_error:
                    raise close_error

        async def connect(uri, *, additional_headers, **options):
            actions.append("upgrade")
            assert uri == "wss://token4research.cn/v1/responses"
            assert additional_headers == {"Authorization": f"Bearer {secret}",
                                          "X-Client-Request-Id": request["request_id"]}
            assert options["proxy"] is None and options["sock"] is fake_tcp
            assert options["server_hostname"] == "token4research.cn" and options["ssl"] is True
            assert not options["logger"].isEnabledFor(10)
            if status != 101:
                raise InvalidStatus(Response(status, "fixture failure", Headers({"Retry-After": "4"})))
            wire = Wire()
            wire.events = events() if events is not None else None
            return wire

        monkeypatch.setattr(client, "connect", connect)

        def note(mark, storage, item):
            actions.append(mark)
            storage.append(item)

        token = provider.LEASE.set(SimpleNamespace(path=None, dispatched=lambda: actions.append("reserved"),
            outcome=lambda code, **details: outcomes.append((code, details))))
        try:
            try:
                reply = provider.call_responses(
                    request, config, model_lock=tmp_path / "model.json", progress=progress.append,
                    capture_response=(lambda data: note("raw", raw, data)) if with_capture else None,
                    capture_receipt=(lambda data: note("receipt", receipts, data)) if with_capture else None,
                    timeouts=RequestTimeouts(.2, .2, .9, 1, 1), sse_inactivity_seconds=idle,
                )
            except TransportFailure as error:
                reply = error
        finally:
            provider.LEASE.reset(token)
        assert secret not in json.dumps([progress, raw, receipts])
        return SimpleNamespace(reply=reply, actions=actions, frames=frames, progress=progress, raw=raw,
                               receipts=receipts, outcomes=outcomes, request=request, config=config)

    return call


def test_ws_uses_one_model_frame_and_recovers_original_completion(run_ws, tmp_path):
    from websockets.exceptions import ConnectionClosedError

    terminal = event("response.completed")

    async def frames():
        yield event("response.created")
        yield terminal
        pytest.fail("read after the definitive WebSocket completion")

    check = run_ws(frames, close_error=ConnectionClosedError(None, None))
    assert check.actions == ["tcp", "upgrade", "reserved", "single_send", "receipt", "raw", "websocket_close", "socket_close"]
    assert check.frames == [{"type": "response.create", **{k: v for k, v in check.request["payload"].items() if k != "stream"}}]
    assert check.raw == [terminal["response"]] and check.receipts[0]["raw_response_sha256"] == digest(check.raw[0])
    assert check.reply.provider_request_id == "req_model" and check.reply.response_id == "resp_exact"
    assert check.reply.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "reasoning_tokens": 1}
    assert any(row.get("http_status") == 101 for row in check.progress)
    assert check.outcomes[0][0] == "completed"
    reply, identity = provider.recover_completed_response(check.raw[0], check.config, metadata=check.receipts[0],
                                                         model_lock=tmp_path / "recovered.json")
    assert reply == check.reply and identity["reported_reasoning_effort"] == "xhigh"
    with pytest.raises(TransportFailure, match="websocket_ipv4_dispatcher_required"):
        responses_call(check.request, check.config)


@pytest.mark.parametrize(("status", "expected"), [(401, "http_401"), (429, "http_429"), (502, "http_502")])
def test_ws_rejected_upgrade_never_reserves_a_model_submission(run_ws, status, expected):
    check = run_ws(None, status=status)
    assert check.actions == ["tcp", "upgrade", "socket_close"] and not check.frames
    assert check.reply.code == expected and not check.reply.ambiguous
    assert check.reply.retry_after == (4 if status == 429 else None)


def test_ws_limits_require_bound_inactivity_and_capture(run_ws):
    check = run_ws(None, idle=None)
    assert check.reply.code == "websocket_event_inactivity_required" and not check.actions
    check = run_ws(None, with_capture=False)
    assert check.reply.code == "websocket_ipv4_capture_required" and not check.actions


def test_ws_idle_cancels_heartbeat_only_frames_without_resending(run_ws):
    async def frames():
        yield event("response.created")
        yield {"type": "response.output_text.delta", "delta": "draft", "sequence_number": 1}
        while True:
            await asyncio.sleep(.02)
            yield {"type": "relay_ping"}

    check = run_ws(frames, idle=.1)
    assert len(check.frames) == 1 and check.reply.code == "timeout" and check.reply.ambiguous
    assert check.progress[-1]["error_type"] == "websocket_inactivity_timeout"
    assert not check.raw and not check.receipts


def test_ws_write_timeout_is_ambiguous_after_one_reservation(run_ws):
    check = run_ws(None, send_delay=20)
    assert len(check.frames) == 1 and check.reply.code == "timeout" and check.reply.ambiguous
    assert check.progress[-1]["error_type"] == "websocket_write_timeout"
    assert not check.raw and not check.receipts


@pytest.mark.parametrize(("terminal", "code"), [
    (event("response.completed", "resp_other"), "response_id_mismatch"),
    (event("response.completed", model="different"), "served_model_mismatch"),
    (event("response.completed", usage=None), "invalid_usage"),
    (event("response.incomplete"), "response.incomplete"),
    ({"type": "error", "status": 429, "error": {"code": "rate_limit_exceeded"}}, "http_429"),
])
def test_ws_validates_terminal_or_preserves_explicit_shared_limit(run_ws, terminal, code):
    async def frames():
        yield event("response.created")
        yield terminal

    check = run_ws(frames)
    assert len(check.frames) == 1 and check.reply.code == code
    if code != "served_model_mismatch":
        assert not check.raw and not check.receipts
    if code == "http_429":
        assert check.outcomes[0][1]["scope"] == "shared"

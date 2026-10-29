"""Offline transport and handover verification. No provider requests."""
import importlib.util
import json
from pathlib import Path
import signal

import httpx
import pytest


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    file = Path(__file__).resolve().parents[2] / "scripts/studies/run_qgeognn_v2_row_llm_nocw_stream.py"
    spec = importlib.util.spec_from_file_location("nocw_stream_test", file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "STUDY", tmp_path)
    policy = tmp_path / "stream_policy.json"
    policy.write_text('{}')
    monkeypatch.setattr(m, "POLICY", policy)
    monkeypatch.setattr(m.retry, "AMENDMENT", policy)
    monkeypatch.setattr(m.retry, "WORKER_SEED", 157)
    monkeypatch.setattr(m.retry, "retry_status", lambda **kw: None)
    monkeypatch.setattr(m.retry, "safe_error", lambda e: type(e).__name__)
    monkeypatch.setattr(m.s, "require_authorization", lambda: None)
    from src.qgeognn_al.active_learning_v2 import scientist_transport
    monkeypatch.setattr(scientist_transport, "provider_key", lambda _: "mock-private-key")
    return m


@pytest.fixture
def config():
    return {"model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "max_output_tokens": 32768,
            "sampling_parameters": {}, "timeout_seconds": 600, "base_url": "https://example.invalid/v1"}


def completed(model="gpt-6.1-sol", status="completed", output=None):
    return {"id": "resp_mock", "status": status, "model": model,
            "output": output if output is not None else [{"type": "reasoning"}, {"type": "message", "content": [
                {"type": "output_text", "text": "mock full answer"}]}], "usage": {"input_tokens": 123}}


def test_exact_frozen_request_body(adapter, config):
    messages = [{"role": "user", "content": "entire candidate pool"}]
    body = adapter.body_for(messages, config)
    assert body == {"model": "gpt-6.1-sol", "input": messages, "reasoning": {"effort": "xhigh"},
                    "tools": [], "store": False, "max_output_tokens": 32768, "stream": True}
    assert "background" not in body and "temperature" not in body


@pytest.mark.parametrize("response", [completed(model="other-model"), completed(status="incomplete"),
    completed(output=[{"type": "function_call"}]), completed(output=[])])
def test_wrong_identity_status_tools_or_empty_are_rejected(adapter, config, response):
    with pytest.raises(adapter.ProtocolViolation):
        adapter.completed_result(response, config)


class FinishedThenBroken(httpx.SyncByteStream):
    def __init__(self, terminal="response.completed"):
        self.terminal = terminal

    def __iter__(self):
        yield b': heartbeat\n\n'
        yield ('data: ' + json.dumps({"type": "response.created", "sequence_number": 1,
                                     "response": {"id": "resp_mock"}}) + '\n\n').encode()
        yield ('data: ' + json.dumps({"type": self.terminal, "sequence_number": 2,
                                     "response": completed()}) + '\n\n').encode()
        raise AssertionError("must not wait for EOF after completed response")

    def close(self):
        raise httpx.RemoteProtocolError("close failure after complete response")


def test_completed_saved_before_close_and_metadata_contains_no_input(adapter, config, tmp_path, monkeypatch):
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, stream=FinishedThenBroken())
    monkeypatch.setattr(adapter, "client_for", lambda c: httpx.Client(transport=httpx.MockTransport(handler)))
    directory = tmp_path / "round_00"
    directory.mkdir()
    answer, provenance = adapter.live_call([{"role": "user", "content": "SENSITIVE-CATALOG"}], config, directory, 0, 6)
    assert answer == "mock full answer" and provenance["response_id"] == "resp_mock"
    assert len(seen) == 1 and seen[0]["input"][0]["content"] == "SENSITIVE-CATALOG"
    saved = json.loads((directory / "stream_00_06.completed.json").read_text())
    assert saved["validated_completed"] and saved["answer"] == answer
    metadata = (directory / "stream_00_06.json").read_text()
    assert "SENSITIVE-CATALOG" not in metadata and "mock full answer" not in metadata
    assert json.loads(metadata)["last_sequence_number"] == 2


def test_pending_started_without_completion_is_never_resent(adapter, config, tmp_path):
    directory = tmp_path / "round_00"
    directory.mkdir()
    adapter.s.once(directory / "attempt_00_00.started.json", {"request_sha256": "original"})
    with pytest.raises(adapter.retry.AmbiguousRequestError):
        adapter.request_with_retries([], config, directory, 0, lambda *a: pytest.fail("extra paid call"))
    assert (directory / "attempt_00_00.started.json").exists()


def test_recover_durable_local_answer_without_model_call(adapter, config, tmp_path, monkeypatch):
    directory = tmp_path / "round_00"
    directory.mkdir()
    digest = adapter.s.stable_hash({"messages": [], "config": config})
    adapter.s.once(directory / "attempt_00_105.started.json", {"request_sha256": digest})
    adapter.s.once(directory / "stream_00_105.completed.json", {"request_sha256": digest,
        "answer": "saved complete answer", "validated_completed": True,
        "amendment_sha256": adapter.s.sha256_file(adapter.POLICY),
        "provenance": {"served_model": "gpt-6.1-sol", "response_id": "resp_mock"}})
    monkeypatch.setattr(adapter, "live_call", lambda *a: pytest.fail("paid call during local recovery"))
    # Fill older recorded failures to preserve the monotonic attempt history.
    for i in range(105):
        adapter.s.once(directory / f"attempt_00_{i:02d}.json", {"request_sha256": digest, "success": False})
    result = adapter.request_with_retries([], config, directory, 0, None)
    assert result["success"] and result["recovered_local_completion"]
    assert result["answer"] == "saved complete answer"


def test_protocol_error_gets_terminal_receipt_and_is_not_retried(adapter, config, tmp_path, monkeypatch):
    directory = tmp_path / "round_00"
    directory.mkdir()
    def wrong(*args):
        raise adapter.ProtocolViolation("model mismatch")
    monkeypatch.setattr(adapter, "live_call", wrong)
    monkeypatch.setattr(adapter.retry.time, "sleep", lambda _: pytest.fail("fatal protocol error retried"))
    with pytest.raises(adapter.FatalTransportStop):
        adapter.request_with_retries([], config, directory, 0, None)
    saved = json.loads((directory / "attempt_00_00.json").read_text())
    assert saved["fatal_protocol_error"] and not saved["success"]
    assert not adapter.unresolved(157)


def test_handover_rechecks_after_stopping_and_keeps_new_inflight_request(adapter, tmp_path, monkeypatch):
    killed, checks = [], iter([[], ["new.started.json"]])
    monkeypatch.setattr(adapter, "unresolved", lambda seed: next(checks))
    states = iter([f"Ss python {adapter.retry.SCRIPT} worker --seed 157", "T python"])
    monkeypatch.setattr(adapter, "process_info", lambda _: next(states))
    monkeypatch.setattr(adapter.os, "kill", lambda pid, sig: killed.append(sig))
    path = tmp_path / "progress/retry_seed_157.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"pid": 12, "phase": "REQUESTING"}))
    result = adapter.migrate_backoff(157, {"pid": 12}, {"pid": 12, "phase": "WAITING_RETRY"})
    assert result is False and killed == [signal.SIGSTOP, signal.SIGCONT]


def test_handover_terminates_only_confirmed_backoff(adapter, tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(adapter, "unresolved", lambda seed: [])
    states = iter([f"Ss python {adapter.retry.SCRIPT} worker --seed 157", "T python"])
    monkeypatch.setattr(adapter, "process_info", lambda _: next(states))
    monkeypatch.setattr(adapter.os, "kill", lambda pid, sig: killed.append(sig))
    path = tmp_path / "progress/retry_seed_157.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"pid": 12, "phase": "WAITING_RETRY"}))
    assert adapter.migrate_backoff(157, {"pid": 12}, {"pid": 12, "phase": "WAITING_RETRY"})
    assert killed == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]


def test_inflight_request_prevents_handover(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "unresolved", lambda seed: ["still_inflight.started.json"])
    monkeypatch.setattr(adapter.os, "kill", lambda *a: pytest.fail("inflight worker interrupted"))
    assert not adapter.migrate_backoff(157, {"pid": 12}, {"pid": 12, "phase": "WAITING_RETRY"})

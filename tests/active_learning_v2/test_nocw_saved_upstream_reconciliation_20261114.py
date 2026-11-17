"""Replay the original fatal-ledger gate without provider calls or label access."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def recovery():
    f = Path(__file__).resolve().parents[2] / "scripts/studies/resume_nocw_upstream_reconciled_20261114.py"
    spec = importlib.util.spec_from_file_location("saved_upstream_test", f)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_case(m, tmp_path, error=None):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    messages, config = [{"role": "user", "content": "frozen request"}], {"model": "gpt-6.1-sol"}
    request = stream.s.stable_hash({"messages": messages, "config": config})
    receipt = tmp_path / "attempt_00_00.json"
    value = {"request_sha256": request, "success": False, "fatal_protocol_error": True,
             "error_message": error or m.ERROR}
    receipt.write_text(json.dumps(value))
    receipt.with_suffix(".started.json").write_text(json.dumps({"request_sha256": request}))
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_text("{}")
    reader = m.reconciled_reader(stream.s.read, receipt, m.m.sha(receipt), request,
                                  reconciliation, m.m.sha(reconciliation))
    return stream, messages, config, receipt, reader


def test_original_stream_guard_blocks_saved_upstream_before_call(recovery, tmp_path):
    stream, messages, config, receipt, reader = saved_case(recovery, tmp_path)
    with pytest.raises(stream.FatalTransportStop, match="requires reconciliation"):
        stream.request_with_retries(messages, config, tmp_path, 0, None)


def test_reconciled_saved_failure_replays_real_stream_ledger_at_next_attempt(recovery, tmp_path, monkeypatch):
    stream, messages, config, receipt, reader = saved_case(recovery, tmp_path)
    original = receipt.read_bytes()
    calls = []
    def completed(m, c, directory, turn, attempt):
        calls.append((m, c, directory, turn, attempt))
        return "saved offline answer", {"served_model": "gpt-6.1-sol"}
    monkeypatch.setattr(stream.s, "read", reader)
    monkeypatch.setattr(stream.s, "require_authorization", lambda: None)
    monkeypatch.setattr(stream.retry, "retry_status", lambda **kwargs: None)
    monkeypatch.setattr(stream, "live_call", completed)
    result = stream.request_with_retries(messages, config, tmp_path, 0, None)
    assert result["success"] is True and result["answer"] == "saved offline answer"
    assert calls == [(messages, config, tmp_path, 0, 1)]
    assert receipt.read_bytes() == original
    assert json.loads(receipt.read_text())["fatal_protocol_error"] is True
    assert (tmp_path / "attempt_00_01.json").exists()


@pytest.mark.parametrize("error", ["wrong model", "provider HTTP 401", "provider terminal failure code: invalid_api_key"])
def test_other_fatal_errors_cannot_be_reconciled(recovery, tmp_path, error):
    stream, messages, config, receipt, reader = saved_case(recovery, tmp_path, error)
    with pytest.raises(RuntimeError, match="restricted"):
        reader(receipt)


def test_other_receipts_keep_their_fatal_flag(recovery, tmp_path):
    stream, messages, config, receipt, reader = saved_case(recovery, tmp_path)
    other = tmp_path / "attempt_00_01.json"
    other.write_text(json.dumps({"fatal_protocol_error": True, "error_message": "wrong model"}))
    assert reader(other)["fatal_protocol_error"] is True


def test_changed_original_receipt_fails_closed(recovery, tmp_path):
    stream, messages, config, receipt, reader = saved_case(recovery, tmp_path)
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(RuntimeError, match="source drift"):
        reader(receipt)


def test_unresolved_started_marker_still_blocks_retry(recovery, tmp_path, monkeypatch):
    stream, messages, config, receipt, reader = saved_case(recovery, tmp_path)
    (tmp_path / "attempt_00_01.started.json").write_text("{}")
    monkeypatch.setattr(stream.s, "read", reader)
    with pytest.raises(stream.retry.AmbiguousRequestError, match="ambiguous"):
        stream.request_with_retries(messages, config, tmp_path, 0, None)

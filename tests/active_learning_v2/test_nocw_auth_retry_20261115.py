"""An explicit retry of a saved401 must never turn new401 into network retries."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/studies/resume_nocw_auth_retry_20261115.py"
    spec = importlib.util.spec_from_file_location("auth_retry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    messages, config = [{"role": "user", "content": "offline frozen request"}], {"model": "gpt-6.1-sol"}
    request_hash = stream.s.stable_hash({"messages": messages, "config": config})
    receipt = tmp_path / "attempt_00_00.json"
    receipt.write_text(json.dumps({"request_sha256": request_hash, "success": False,
        "fatal_protocol_error": True, "error_message": "provider HTTP 401"}))
    receipt.with_suffix(".started.json").write_text(json.dumps({"request_sha256": request_hash}))
    reconciliation = tmp_path / "approved_retry.json"
    reconciliation.write_text("{}")
    reader = module.reader.reconciled_reader(stream.s.read, receipt, module.m.sha(receipt),
        request_hash, reconciliation, module.m.sha(reconciliation))
    monkeypatch.setattr(stream.s, "require_authorization", lambda: None)
    monkeypatch.setattr(stream.retry, "retry_status", lambda **kwargs: None)
    return module, stream, messages, config, receipt, reader


def test_unreconciled_saved401_still_stops(setup_case, tmp_path):
    module, stream, messages, config, receipt, reader = setup_case
    with pytest.raises(stream.FatalTransportStop, match="requires reconciliation"):
        stream.request_with_retries(messages, config, tmp_path, 0, None)


def test_explicit_saved401_retry_uses_next_attempt_preserving_original(setup_case, tmp_path, monkeypatch):
    module, stream, messages, config, receipt, reader = setup_case
    original = receipt.read_bytes()
    calls = []
    def response(m, c, directory, turn, attempt):
        calls.append(attempt)
        return "offline answer", {"served_model": "gpt-6.1-sol"}
    monkeypatch.setattr(stream.s, "read", reader)
    monkeypatch.setattr(stream, "live_call", response)
    saved = stream.request_with_retries(messages, config, tmp_path, 0, None)
    assert saved["success"] is True and calls == [1]
    assert receipt.read_bytes() == original


def test_new401_is_still_fatal_and_retained(setup_case, tmp_path, monkeypatch):
    module, stream, messages, config, receipt, reader = setup_case
    def rejected(*args):
        raise stream.ProtocolViolation("provider HTTP 401")
    monkeypatch.setattr(stream.s, "read", reader)
    monkeypatch.setattr(stream, "live_call", rejected)
    with pytest.raises(stream.FatalTransportStop, match="provider HTTP 401"):
        stream.request_with_retries(messages, config, tmp_path, 0, None)
    new = json.loads((tmp_path / "attempt_00_01.json").read_text())
    assert new["fatal_protocol_error"] is True and new["success"] is False
    with pytest.raises(stream.FatalTransportStop, match="requires reconciliation"):
        stream.request_with_retries(messages, config, tmp_path, 0, None)

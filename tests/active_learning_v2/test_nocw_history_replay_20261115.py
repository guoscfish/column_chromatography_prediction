"""Approved past failures stay reconciled across rounds; new fatal errors stop."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[2] / "scripts/studies/repair_nocw_history_replay_20261115.py"
    spec = importlib.util.spec_from_file_location("history_replay_repair_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def case(m, directory, error, tag):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    directory.mkdir()
    messages, config = [{"role": "user", "content": tag}], {"model": "gpt-6.1-sol"}
    request_hash = stream.s.stable_hash({"messages": messages, "config": config})
    receipt = directory / "attempt_00_00.json"
    receipt.write_text(json.dumps({"request_sha256": request_hash, "success": False,
        "fatal_protocol_error": True, "error_message": error}))
    receipt.with_suffix(".started.json").write_text(json.dumps({"request_sha256": request_hash}))
    (directory / "attempt_00_01.json").write_text(json.dumps({"request_sha256": request_hash,
        "success": True, "answer": tag + " saved answer"}))
    reconciliation = directory / "approved.json"
    reconciliation.write_text("{}")
    entry = {"seed": 157, "path": str(receipt), "error": error, "failure_sha256": m.m.sha(receipt),
        "request_sha256": request_hash, "reconciliation": str(reconciliation),
        "reconciliation_sha256": m.m.sha(reconciliation)}
    return messages, config, entry


def test_historical_upstream_and_current401_replay_without_network(module, tmp_path, monkeypatch):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    older = case(module, tmp_path / "round_27", "provider terminal failure code: upstream_error", "old")
    current = case(module, tmp_path / "round_28", "provider HTTP 401", "current")
    originals = {Path(c[2]["path"]): Path(c[2]["path"]).read_bytes() for c in [older, current]}
    read = module.reader_for_seed(stream.s.read, [older[2], current[2]], 157)
    monkeypatch.setattr(stream.s, "read", read)
    monkeypatch.setattr(stream, "live_call", lambda *args: pytest.fail("No provider call allowed in historical replay"))
    for messages, config, entry in [older, current]:
        saved = stream.request_with_retries(messages, config, Path(entry["path"]).parent, 0, None)
        assert saved["success"] is True
    assert all(path.read_bytes() == value for path, value in originals.items())


def test_current_only_overlay_reproduces_historical_failure(module, tmp_path, monkeypatch):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    older = case(module, tmp_path / "round_27", "provider terminal failure code: upstream_error", "old")
    current = case(module, tmp_path / "round_28", "provider HTTP 401", "current")
    monkeypatch.setattr(stream.s, "read", module.reader_for_seed(stream.s.read, [older[2], current[2]], 157, include_upstream=False))
    with pytest.raises(stream.FatalTransportStop, match="requires reconciliation"):
        stream.request_with_retries(older[0], older[1], Path(older[2]["path"]).parent, 0, None)


def test_new_unapproved401_keeps_fatal_guard(module, tmp_path, monkeypatch):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    older = case(module, tmp_path / "round_27", "provider terminal failure code: upstream_error", "old")
    current = case(module, tmp_path / "round_28", "provider HTTP 401", "current")
    future = case(module, tmp_path / "round_29", "provider HTTP 401", "new")
    monkeypatch.setattr(stream.s, "read", module.reader_for_seed(stream.s.read, [older[2], current[2]], 157))
    with pytest.raises(stream.FatalTransportStop, match="requires reconciliation"):
        stream.request_with_retries(future[0], future[1], Path(future[2]["path"]).parent, 0, None)


def test_historical_receipt_hash_drift_fails_closed(module, tmp_path):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    old = case(module, tmp_path / "round_27", "provider terminal failure code: upstream_error", "old")
    read = module.history_reader(stream.s.read, [old[2]])
    receipt = Path(old[2]["path"])
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(RuntimeError, match="changed"):
        read(receipt)

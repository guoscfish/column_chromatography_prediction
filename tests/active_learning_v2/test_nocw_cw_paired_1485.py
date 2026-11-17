"""Resume boundaries and interrupted-response accounting for the paired run."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def m(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/studies/run_nocw_cw_paired_to1485_20261112.py"
    spec = importlib.util.spec_from_file_location("paired_1485", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "NOCW", tmp_path / "nocw")
    return module


def interrupted(m):
    p = m.NOCW / "selections/seed_157/llm32_nocw/round_23/attempt_02_01.started.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"request_sha256": "same-request", "retry_amendment_sha256": "same-policy"}))
    stop = {"all_target_processes_stopped": True, "seeds": [{"seed": 157, "lock_released": True,
        "unresolved_started_preserved": [str(p.relative_to(m.NOCW))]}]}
    return p, stop


def test_schedule_reuses_completed_rounds_and_stops_at_1485(m):
    assert m.BUDGETS[0] == 333 and m.BUDGETS[-1] == 1485 and len(m.BUDGETS) == 37
    assert sum(len(m.remaining_rounds(r)) for r in (23,21,25,26,28)) == 57
    assert sum(len(m.remaining_rounds(r)) for r in (21,0,0,21,0)) == 138
    assert list(m.remaining_rounds(35)) == [35]
    assert not list(m.remaining_rounds(36))
    with pytest.raises(ValueError):
        m.remaining_rounds(-1)


def test_interrupted_request_is_local_unknown_and_original_is_preserved(m):
    p, stop = interrupted(m)
    before = p.read_bytes()
    m.reconcile_interruption(p, stop, "comparison")
    terminal = p.with_name("attempt_02_01.json")
    result = json.loads(terminal.read_text())
    assert p.read_bytes() == before
    assert result["request_sha256"] == "same-request" and result["success"] is False
    assert result["remote_completion_status"] == result["billing_status"] == "unknown"
    assert result["local_interruption_reconciliation"] is True
    receipt = terminal.read_bytes()
    m.reconcile_interruption(p, stop, "comparison")
    assert terminal.read_bytes() == receipt


def test_durable_completed_response_cannot_be_discarded(m):
    p, stop = interrupted(m)
    (p.parent / "stream_02_01.completed.json").write_text("{}")
    with pytest.raises(RuntimeError, match="recovered instead"):
        m.reconcile_interruption(p, stop, "comparison")
    assert not (p.parent / "attempt_02_01.json").exists()


@pytest.mark.parametrize("failure", ("active", "locked", "unregistered", "frozen"))
def test_only_verified_unfrozen_stopped_requests_can_resume(m, failure):
    p, stop = interrupted(m)
    if failure == "active":
        stop["all_target_processes_stopped"] = False
    elif failure == "locked":
        stop["seeds"][0]["lock_released"] = False
    elif failure == "unregistered":
        stop["seeds"][0]["unresolved_started_preserved"] = []
    else:
        (p.parent / "batch_freeze.json").write_text("{}")
    with pytest.raises(RuntimeError, match="outside"):
        m.reconcile_interruption(p, stop, "comparison")
    assert not (p.parent / "attempt_02_01.json").exists()


def test_missing_fit_cannot_be_counted_as_complete(m):
    assert m.last_round(m.NOCW, 157, m.ARM) == -1
    for rnd in (0, 2):
        p = m.NOCW / f"runtime/seed_157/{m.ARM}/round_{rnd:02d}/prediction_freeze.json"
        p.parent.mkdir(parents=True)
        p.write_text("{}")
    with pytest.raises(RuntimeError, match="Noncontiguous"):
        m.last_round(m.NOCW, 157, m.ARM)

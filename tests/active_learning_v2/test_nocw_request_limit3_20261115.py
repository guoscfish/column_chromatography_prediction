"""Real cross-process admission tests and safe-handover boundary checks; no API."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/studies/run_nocw_request_limit3_20261115.py"
CHILD = r'''
import fcntl, importlib.util, json, sys, time
from pathlib import Path
spec=importlib.util.spec_from_file_location("limited_request_test",sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
study=Path(sys.argv[2]);seed=int(sys.argv[3])
def event(delta):
    with (study/"events.jsonl").open("a") as out:
        fcntl.flock(out,fcntl.LOCK_EX)
        out.write(json.dumps([time.monotonic_ns(),delta,seed])+"\n");out.flush()
        fcntl.flock(out,fcntl.LOCK_UN)
with m.request_slot(seed,0,0,study):
    event(1);time.sleep(0.2);event(-1)
'''


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("request_cap_boundary_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def status(study, count):
    path = study / "request_limit3_status_20261115.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"jobs": [{"mode": "legacy"} for _ in range(count)]}))
    temp.replace(path)


def spawn(study, seed):
    return subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPT), str(study), str(seed)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.mark.parametrize("reserved,maximum", [(0, 3), (2, 1)])
def test_physical_request_overlap_is_bounded_across_five_processes(tmp_path, reserved, maximum):
    status(tmp_path, reserved)
    children = [spawn(tmp_path, s) for s in range(1, 6)]
    for child in children:
        out, err = child.communicate(timeout=20)
        assert child.returncode == 0, out + err
    active = peak = starts = 0
    for _, delta, seed in sorted(json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()):
        active += delta
        starts += delta == 1
        peak = max(peak, active)
        assert 0 <= active <= maximum
    assert active == 0 and starts == 5 and peak <= maximum


def test_legacy_drain_waits_without_entering_request(tmp_path):
    status(tmp_path, 5)
    child = spawn(tmp_path, 1)
    try:
        gate = tmp_path / "progress/request_gate_seed_1.json"
        deadline = time.monotonic() + 10
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert gate.exists()
        time.sleep(0.25)
        assert not (tmp_path / "events.jsonl").exists()
        status(tmp_path, 0)
        out, err = child.communicate(timeout=10)
        assert child.returncode == 0, out + err
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def boundary_files(tmp_path, phase="SELECTING", unresolved=False):
    (tmp_path / "progress").mkdir()
    (tmp_path / "progress/nocw_157.json").write_text(json.dumps({"pid": 1234, "phase": phase}))
    if unresolved:
        directory = tmp_path / "nocw/selections/seed_157/llm32_nocw/round_28"
        directory.mkdir(parents=True)
        (directory / "attempt_01_21.started.json").write_text("{}")
    return {"seed": 157, "pid": 1234}, {"pid": 1234, "phase": "WAITING_RETRY",
        "retry_at_utc": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()}


def test_only_verified_backoff_is_a_handover_boundary(module, tmp_path):
    job, retry = boundary_files(tmp_path)
    assert module.safe_backoff(job, retry, tmp_path)
    assert not module.safe_backoff(job, {**retry, "phase": "REQUESTING"}, tmp_path)


def test_active_or_ambiguous_request_is_never_handed_over(module, tmp_path):
    job, retry = boundary_files(tmp_path, unresolved=True)
    assert not module.safe_backoff(job, retry, tmp_path)


def test_failed401_or_training_is_not_automatically_restarted(module, tmp_path):
    job, retry = boundary_files(tmp_path, phase="FAILED_CLOSED")
    assert not module.safe_backoff(job, retry, tmp_path)
    (tmp_path / "progress/nocw_157.json").write_text(json.dumps({"pid": 1234, "phase": "TRAINING_OR_WAITING_FOR_SLOT"}))
    assert not module.safe_backoff(job, retry, tmp_path)

"""Offline exact-error recovery; no credentials, requests, labels or fits."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
spec = importlib.util.spec_from_file_location("h2_transport_recovery_test", STUDY / "transport_recovery_20261102/upstream_h2_worker.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
spec2 = importlib.util.spec_from_file_location("existing_persistent_test_fixture", ROOT / "tests/active_learning_v2/test_free_cw_persistent.py")
existing = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(existing)
adapter = existing.adapter
spec3 = importlib.util.spec_from_file_location("h2_controller_overlay_test", STUDY / "transport_recovery_20261102/controller_overlay.py")
overlay = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(overlay)


def test_saved_failure_retained_and_next_attempt_reused(adapter, tmp_path):
    p = adapter
    shim.install(p)
    directory = tmp_path / "round_07"
    receipt = {"request_sha256": p.request_hash([], p.test_config), "success": False,
               "fatal_protocol_error": True, "reason": shim.REASON}
    p.a.once(directory / "attempt_00_00.json", receipt)
    original = (directory / "attempt_00_00.json").read_bytes()
    calls = []
    def call(messages, config):
        calls.append((messages, config))
        return "{}", {"offline": True}
    result = p.request_with_retries([], p.test_config, directory, 0, call)
    assert result["success"] and len(calls) == 1
    assert (directory / "attempt_00_00.json").read_bytes() == original
    assert (directory / "attempt_00_01.json").exists()
    assert list((p.EXT / "reclassified").rglob("attempt_00_00.json"))
    assert p.request_with_retries([], p.test_config, directory, 0, lambda *a: pytest.fail("duplicate request")) == result


@pytest.mark.parametrize("reason", ["provider HTTP 401", "provider HTTP 402", "wrong-model response", "unsupported reasoning effort", "provider terminal code: unknown"])
def test_other_fatal_errors_still_stop(adapter, tmp_path, reason):
    p = adapter
    shim.install(p)
    def call(*args):
        raise p.m.FatalProtocolError(reason)
    with pytest.raises(p.m.FatalProtocolError, match=reason):
        p.request_with_retries([], p.test_config, tmp_path / "round_07", 0, call)
    assert len(list((tmp_path / "round_07").glob("*.started.json"))) == 1


def test_live_wrapper_preserves_arguments_and_only_converts_exact_code():
    import httpx
    class Fatal(RuntimeError):
        pass
    messages, config = object(), object()
    seen = []
    def original(*args):
        seen.append(args)
        raise Fatal(shim.REASON)
    p = SimpleNamespace(legacy_transient=lambda record: False, m=SimpleNamespace(live_stream=original, FatalProtocolError=Fatal))
    shim.install(p)
    with pytest.raises(httpx.RemoteProtocolError, match="upstream_http2_stream_error"):
        p.m.live_stream(messages, config)
    assert seen == [(messages, config)]
    assert not p.legacy_transient({"fatal_protocol_error": True, "reason": shim.REASON + " other"})


def test_controller_only_reclassifies_exact_error_and_dispatches_overlay():
    p = SimpleNamespace(load_base=lambda: None, validate=lambda: {},
        transient_stop=lambda state: state.get("phase") == "QUEUED",
        retire_worker=lambda *a, **kw: False, process_command=lambda pid: "", ENTRY=overlay.BASE_ENTRY)
    overlay.bind(p, shim)
    assert p.ENTRY == overlay.ENTRY
    assert p.transient_stop({"phase": "FAILED_CLOSED", "reason": shim.REASON})
    assert p.transient_stop({"phase": "QUEUED"})
    for reason in ["provider HTTP 401", "provider HTTP 402", "wrong-model response", shim.REASON + " unknown"]:
        assert not p.transient_stop({"phase": "FAILED_CLOSED", "reason": reason})


@pytest.mark.parametrize("prior", [overlay.BASE_ENTRY, overlay.HERE / "upstream_h2_worker.py", overlay.ENTRY])
def test_watchdog_retains_actual_adopted_worker_identity(prior):
    seen = []
    p = SimpleNamespace(load_base=lambda: None, validate=lambda: {}, transient_stop=lambda state: False,
        process_command=lambda pid: f"python {prior} worker --seed 887", ENTRY=overlay.BASE_ENTRY)
    def retire(seed, pid, reason, **kwargs):
        seen.append((p.ENTRY, seed, pid, reason, kwargs))
        return False
    p.retire_worker = retire
    overlay.bind(p, shim)
    assert p.retire_worker(887, 123, "idle", idle_limit=660) is False
    assert seen == [(prior, 887, 123, "idle", {"idle_limit": 660})]
    assert p.ENTRY == overlay.ENTRY

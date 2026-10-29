"""Offline launch boundaries for the five-key NoCW-only revision."""
import importlib.util
from pathlib import Path

import pytest

from src.qgeognn_al.active_learning_v2 import nocw_study as s
from src.qgeognn_al.active_learning_v2 import nocw_inputs as views


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py"
    spec = importlib.util.spec_from_file_location("nocw_five_test_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("STUDY", "SEEDS", "VERSION", "ARMS", "METHODS", "initial_paths", "Context", "code_hashes",
                 "once", "exclusive_lock", "input_view", "prompt", "validate_plan"):
        monkeypatch.setattr(s, name, getattr(s, name))
    monkeypatch.setattr(views, "VERSION", views.VERSION)
    return module


def test_five_distinct_keys_required_without_disclosure(adapter, tmp_path, monkeypatch):
    paths = {}
    for seed in adapter.SEEDS:
        paths[seed] = tmp_path / f"{seed}.key"
        paths[seed].write_text(f"private-test-credential-{seed}")
    monkeypatch.setattr(adapter, "KEY_FILES", paths)
    result = adapter.credential_audit()
    assert result["distinct_keys"] == 5 and not result["key_values_logged"]
    assert "private-test-credential" not in str(result)
    paths[887].write_bytes(paths[157].read_bytes())
    with pytest.raises(ValueError, match="different key"):
        adapter.credential_audit()


def test_seed_locks_allow_parallel_distinct_workers(adapter, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "STUDY", tmp_path)
    with adapter.seed_operation_lock(157):
        with adapter.seed_operation_lock(887):
            pass
        with pytest.raises(RuntimeError):
            with adapter.seed_operation_lock(157):
                pytest.fail("same-seed duplicate lock accepted")


def test_shared_served_model_identity_is_immutable(adapter, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "STUDY", tmp_path)
    p = tmp_path / "served_model_lock.json"
    adapter.once(p, {"served_model": "gpt-6.1-sol"})
    adapter.once(p, {"served_model": "gpt-6.1-sol"})
    with pytest.raises(RuntimeError):
        adapter.once(p, {"served_model": "other"})


def test_worker_never_selects_or_fits_cw_aware(adapter, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "STUDY", tmp_path)
    monkeypatch.setattr(adapter, "credential_audit", lambda: {})
    monkeypatch.setattr(s, "require_authorization", lambda: None)
    monkeypatch.setattr(s, "BUDGETS", (333, 365, 397))
    calls = []
    monkeypatch.setattr(s, "select", lambda seed, arm, r, **kw: calls.append(("select", arm, r)))
    monkeypatch.setattr(s, "advance", lambda seed, arm, r, **kw: calls.append(("advance", arm, r)))
    adapter.worker(157, allow_paid_llm=True, allow_training=True, allow_reveal=True)
    assert calls == [("select", "llm32_nocw", 0), ("advance", "llm32_nocw", 0),
                     ("select", "llm32_nocw", 1), ("advance", "llm32_nocw", 1)]
    with pytest.raises(PermissionError):
        adapter.launch()


@pytest.mark.parametrize("seed", (157, 887, 2357, 6101, 12203))
def test_all_five_l333_catalogs_match_historical_free_order(adapter, seed):
    adapter.activate()
    raw, roles, source = s.initial_state(seed)
    catalog, _, packet = adapter.input_view(raw, seed, "llm32_nocw", 0, [])
    reference = s.ReadOnlyCatalog(**raw)
    assert [r[0] for r in s.encode_catalog(reference)["candidates"]["rows"]] == [r[0] for r in packet["candidates"]["rows"]]
    assert len(roles["l0"]) == 333 and len(catalog.candidates) == 2997
    assert source["checkpoint_sha256"]
    views.audit_no_cw({"prompt": adapter.prompt("no_cw"), "packet": packet})
    assert "Do not cite row IDs in free prose" not in adapter.prompt("no_cw")

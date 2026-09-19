import json

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2 import batch_adaptivity as study
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select


def test_static_order_is_prefix_consistent_and_not_reacquired():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(400, 8))
    indices = np.arange(400)
    selected, _ = study.select_static(features, indices, indices[:20], indices[20:])
    first = lcmd_tp_select(features[20:], features[:20], 32).selected_pool_positions + 20
    assert np.array_equal(selected[:32], first)
    assert len(set(selected)) == 320
    assert set(selected).isdisjoint(indices[:20])
    for stop in range(32, 321, 32):
        assert len(selected[:stop]) == stop


def test_gradient_bank_order_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="order"):
        study.select_static(np.ones((400, 2)), np.arange(400)[::-1], np.arange(20), np.arange(20, 400))


def test_test_reveal_cannot_run_without_global_barrier(monkeypatch):
    def incomplete():
        raise RuntimeError("incomplete trajectories")
    monkeypatch.setattr(study, "finalize", incomplete)
    def forbidden(*args, **kwargs):
        pytest.fail("truth store constructed before barrier")
    monkeypatch.setattr(study, "RestrictedLabelStore", forbidden)
    with pytest.raises(RuntimeError, match="incomplete"):
        study.reveal_and_report()


def test_tampered_or_nonblind_freeze_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    (tmp_path / "seal.json").write_text("{}")
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"seal_sha256": "wrong", "test_truth_access_count": 0}))
    with pytest.raises(RuntimeError, match="invalid"):
        study.verify_record(path)


def test_only_matched_total_budget_is_scheduled():
    assert study.BUDGETS == tuple(range(333, 654, 32))
    assert study.BUDGETS[-1] - study.BUDGETS[0] == 320
    assert len(study.BUDGETS) == 11

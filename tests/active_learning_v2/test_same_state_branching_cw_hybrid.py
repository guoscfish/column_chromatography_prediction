"""Scientific contracts for the exact-selector pilot; no model fits."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.qgeognn_al.active_learning_v2 import same_state_branching_cw_hybrid as study
from src.qgeognn_al.active_learning_v2.same_state_selector_audit import audit_firewall, _save
from src.qgeognn_al.active_learning_v2.gradient_transforms import center_width_transform
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.ivr import conditional_batch_ivr
from src.qgeognn_al.active_learning_v2.maxdet import conditional_gradient_maxdet
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore


def fixture():
    rng = np.random.default_rng(41)
    # Non-contiguous and permuted L/U order to catch incorrect positional indexing.
    outer = list(range(100, 196))
    labeled = outer[::2][:40]
    unlabeled = [i for i in outer[::-1] if i not in labeled]
    state = {"outer_indices": outer, "labeled_indices": labeled, "unlabeled_indices": unlabeled,
             "labeled_ids": [f"id{i}" for i in labeled], "unlabeled_ids": [f"id{i}" for i in unlabeled]}
    return {"ordinary_q50": rng.normal(size=(96, 12)), "center_width": rng.normal(size=(96, 12))}, state


@pytest.mark.parametrize("strategy", study.STRATEGIES)
def test_same_state_selection_matches_direct_historical_algorithm(strategy):
    banks, state = fixture()
    lpos, upos = study._positions(state)
    if strategy == "center_width_lcmd":
        local = lcmd_tp_select(banks["center_width"][upos], banks["center_width"][lpos], 32).selected_pool_positions
    elif strategy == "kernel_ivr":
        local = conditional_batch_ivr(banks["ordinary_q50"], lpos, upos, 32)["selected_candidate_positions"]
    else:
        local = conditional_gradient_maxdet(banks["ordinary_q50"], lpos, upos, 32).selected_candidate_positions
    actual = study.select_batch(strategy, banks, state)
    assert actual["selected_ids"] == np.asarray(state["unlabeled_ids"])[local].tolist()
    assert actual == study.select_batch(strategy, banks, state)
    assert len(set(actual["selected_ids"])) == 32


def test_cw_must_receive_explicit_geometry_and_reject_q50_alias():
    banks, state = fixture()
    with pytest.raises(TypeError, match="method-specific"):
        study.select_batch("center_width_lcmd", banks["ordinary_q50"], state)
    with pytest.raises(KeyError):
        study.select_batch("center_width_lcmd", {"ordinary_q50": banks["ordinary_q50"]}, state)
    first = study.select_batch("center_width_lcmd", banks, state)["selected_ids"]
    banks["ordinary_q50"] *= 500
    assert study.select_batch("center_width_lcmd", banks, state)["selected_ids"] == first
    assert first != study.select_batch("center_width_lcmd", {"center_width": banks["ordinary_q50"]}, state)["selected_ids"]


@pytest.mark.parametrize("name", ["hybrid", "gradient_norm_coreset", "historical_hybrid_exact"])
def test_nonprimary_algorithms_cannot_be_mislabeled_or_executed(name):
    banks, state = fixture()
    with pytest.raises(ValueError, match="unknown strategy"):
        study.select_batch(name, banks, state)


def test_center_width_transform_is_population_scaled_l0_only():
    truth = np.array([[1., 4.], [3., 5.], [8., 13.], [4., 8.]])
    transform, audit = center_width_transform(truth)
    center = truth.mean(axis=1)
    width = truth[:, 1] - truth[:, 0]
    assert np.allclose(truth @ transform.T, np.c_[center / center.std(), width / width.std()])
    assert audit["target_source"] == "L0_labels_only" and audit["scale_ddof"] == 0


def test_postfit_extracts_separate_current_checkpoint_banks_with_fixed_transform(monkeypatch, tmp_path):
    banks, state = fixture()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint-one")
    study.atomic_json(tmp_path / "fit_audit.json", {"checkpoint_state_hash": "hash-one"})
    transform, transform_audit = center_width_transform(np.array([[1., 3.], [2., 7.], [4., 9.]]))
    context = SimpleNamespace(seed=157, outer=np.array(state["outer_indices"]), atom=[], angle=[],
                              preprocessing={"target_scales": {"V1": 2., "V2": 3.}},
                              cw_transform=transform, cw_transform_audit=transform_audit)
    calls = []
    monkeypatch.setattr(study, "load_predictor_checkpoint", lambda p: p.read_bytes())
    def extract(kind):
        def run(model, atom, angle, indices, value, **kwargs):
            calls.append((kind, model, np.array(value), kwargs))
            return SimpleNamespace(features=banks[kind], audit={"elapsed_seconds": 0})
        return run
    monkeypatch.setattr(study, "extract_q50_gradient_sketches", extract("ordinary_q50"))
    monkeypatch.setattr(study, "extract_linear_output_gradient_sketches", extract("center_width"))
    result, _ = study._extract_branch_bank(context, checkpoint, tmp_path, state)
    assert np.array_equal(result["center_width"], banks["center_width"])
    assert [call[0] for call in calls] == ["ordinary_q50", "center_width"]
    assert np.array_equal(calls[1][2], transform)
    assert calls[1][3] == {"dimension": 512, "sketch_seed": study.sketch_seed(157)}
    # Reuse succeeds only for exactly the same checkpoint and state.
    study._extract_branch_bank(context, checkpoint, tmp_path, state)
    assert len(calls) == 2
    checkpoint.write_bytes(b"checkpoint-two")
    with pytest.raises(RuntimeError, match="mismatch"):
        study._extract_branch_bank(context, checkpoint, tmp_path, state)


@pytest.mark.parametrize("seed", study.SEEDS)
def test_actual_branch_context_uses_historical_cw_context_without_training(seed):
    with audit_firewall() as accesses:
        context = study.BranchContext(seed)
    frozen = study.read_json(study.CW_STUDY / f"runtime/seed_{seed}/context.json")
    assert context.cw_transform_audit == frozen["center_width_transform"]
    assert all(row["roles"] in {"l0", "validation"} for row in accesses)
    assert context.outer.shape == (3330,)


def test_diagnostics_keep_transform_specific_names_and_common_counts():
    banks, state = fixture()
    result = study.bank_diagnostics(banks, state)
    assert result["active_label_count"] == 40 and result["candidate_pool_size"] == 56
    assert "raw_gradient_norm_mean" in result and "cw_gradient_norm_mean" in result
    assert "cw_gradient_coverage_nearest_distance_mean" in result
    assert "gradient_norm_mean" not in result
    assert not any("gradient_gradient" in key for key in result)


def test_firewall_rejects_test_truth_even_when_store_is_frozen():
    store = object.__new__(RestrictedLabelStore)
    store.roles = {"test-id": "test"}
    store.acquisitions_frozen = store.predictions_frozen = True
    with audit_firewall():
        with pytest.raises(PermissionError, match="L0/validation"):
            store.reveal(["test-id"], "final_test_evaluation")
        with pytest.raises(RuntimeError, match="forbidden"):
            study.fit_from_same_initialization()


def test_regression_rejects_same_set_with_different_order(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    with pytest.raises(RuntimeError, match="ordered selection"):
        _save("audit.csv", [{"intersection_count": 32, "exact_match": False}])


def test_seal_requires_exact_audit_before_any_anchor_work(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    xml = tmp_path / "tests.xml"
    xml.write_text('<testsuite>' + '<testcase classname="test_same_state_branching_cw_hybrid"/>' * 12 + '</testsuite>')
    with pytest.raises(FileNotFoundError, match="selector_audit"):
        study.prepare(xml)
    assert not (tmp_path / "seal.json").exists()


def test_unrelated_old_passing_tests_cannot_seal_pilot(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    xml = tmp_path / "tests.xml"
    xml.write_text('<testsuite>' + '<testcase classname="test_same_state_branching"/>' * 15 + '</testsuite>')
    with pytest.raises(RuntimeError, match="preflight"):
        study.prepare(xml)


def test_stale_smoke_cannot_authorize_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "validate_seal", lambda: {})
    study.atomic_json(tmp_path / "seal.json", {"new": "seal"})
    study.atomic_json(tmp_path / "engineering_smoke.json", {"status": "PASSED", "test_truth_access_count": 0,
                                                          "seal_sha256": "obsolete"})
    with pytest.raises(RuntimeError, match="smoke"):
        study.execute_seed(157)


def test_test_reveal_cannot_precede_global_freeze(monkeypatch):
    accessed = []
    monkeypatch.setattr(study, "validate_seal", lambda: {})
    def fail():
        raise RuntimeError("incomplete global branch freeze")
    monkeypatch.setattr(study, "_verify_global_branch_freeze", fail)
    monkeypatch.setattr(study, "RestrictedLabelStore", lambda *args: accessed.append(args))
    with pytest.raises(RuntimeError, match="incomplete"):
        study.reveal_test_and_report()
    assert accessed == []


def test_primary_design_and_selector_inputs():
    assert study.STRATEGIES == ("center_width_lcmd", "kernel_ivr", "gradient_maxdet")
    assert len(study.SEEDS)*len(study.SOURCES)*len(study.ANCHOR_BUDGETS)*len(study.STRATEGIES)*2 == 48
    assert not set(inspect.signature(study.select_batch).parameters) & {"labels", "truth", "test", "performance"}


def test_sealed_audit_tables_are_immutable_and_include_hybrid_inventory():
    from src.qgeognn_al.active_learning_v2.same_state_selector_audit import AUDIT_TABLES
    assert "historical_hybrid_artifact_inventory.csv" in AUDIT_TABLES
    assert "anchor_state_diagnostics_audit.csv" in AUDIT_TABLES
    assert not set(AUDIT_TABLES) & {"test_label_access_audit.csv", "state_diagnostics.csv", "branch_results.csv"}

import ast
from dataclasses import fields
import json

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_study as study
from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_runner as runner
from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_reporting as reporting
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore
from src.qgeognn_al.active_learning_v2.strategy_policy import FixedSwitchPolicy, StrategyState


def test_policy_boundary_and_state_exclude_test_or_hidden_labels():
    policy = FixedSwitchPolicy()
    assert policy.select(StrategyState(621, 2709)) == "gradient_lcmd"
    assert policy.select(StrategyState(653, 2677)) == "kernel_ivr"
    assert policy.select(StrategyState(685, 2645)) == "kernel_ivr"
    assert not any("test" in f.name or "hidden" in f.name for f in fields(StrategyState))
    with pytest.raises(TypeError):
        StrategyState(653, 2677, test_nrmse=.1)


@pytest.mark.parametrize("seed", study.CONFIRMATION_SEEDS)
def test_real_frozen_L653_reconstruction(seed):
    state = study.reconstruct(seed)
    assert study.ACTIVE_LABEL_BUDGETS[state["source_round"]] == 653
    assert state["source_round"] == study.switch_round()
    assert len(state["labeled_ids"]) == 653
    assert len(state["unlabeled_ids"]) == 2677
    assert len(set(state["labeled_ids"]) | set(state["unlabeled_ids"])) == 3330


def test_transition_exact_batch_and_constant_universe():
    outer = list(range(3330))
    labeled, unlabeled = outer[:653], outer[653:]
    budgets = [len(labeled)]
    while len(labeled) < 1005:
        labeled, unlabeled = study.advance(labeled, unlabeled, unlabeled[:32], outer)
        budgets.append(len(labeled))
    assert budgets == list(study.ACTIVE_LABEL_BUDGETS[study.switch_round():])
    assert budgets[1] == 685
    with pytest.raises(RuntimeError, match="exactly 32"):
        study.advance(labeled, unlabeled, unlabeled[:31], outer)
    with pytest.raises((ValueError, RuntimeError)):
        study.advance(labeled, unlabeled, [unlabeled[0]] * 32, outer)
    with pytest.raises((ValueError, RuntimeError)):
        study.advance(labeled, unlabeled, labeled[:32], outer)


def test_selector_uses_only_features_and_current_partition():
    bank = np.random.default_rng(76).normal(size=(100, 8))
    # Use the real switch count with a larger reference universe.
    bank = np.tile(bank, (8, 1))
    args = (bank, list(range(800)), list(range(653)), list(range(653, 800)), StrategyState(653, 147))
    first, a = runner.select(*args)
    second, b = runner.select(*args)
    assert first == second and a["trace"] == b["trace"]
    assert len(first) == len(set(first)) == 32 and min(first) >= 653
    assert a["audit"]["labeled_rows"] == 653
    assert a["audit"]["reference_rows"] == 800
    with pytest.raises(RuntimeError, match="historical prefix"):
        runner.select(bank, list(range(800)), list(range(621)), list(range(621, 800)), StrategyState(621, 179))


def test_reviewed_code_compatibility_is_narrow():
    old = 'from typing import Sequence\ndef original(x):\n    return x + 1\n'
    new = old.replace("Sequence", "Mapping, Sequence") + '\ndef extract_linear_output_gradient_sketches(x):\n    return x\n'
    assert study.compatible_gradient_extension(old, new)
    assert not study.compatible_gradient_extension(old, new.replace("x + 1", "x + 2"))
    assert not study.compatible_gradient_extension(old, new + '\noriginal = lambda x: 0\n')
    assert not study.compatible_gradient_extension(old, new + '\ndef original(x):\n    return 0\n')


@pytest.mark.parametrize("checkpoint_available", [False, True])
def test_reuse_failure_plans_only_L653_work(tmp_path, monkeypatch, checkpoint_available):
    def incompatible(*args):
        raise RuntimeError("incompatible cache under test")
    monkeypatch.setattr(study, "source_features" if checkpoint_available else "check_checkpoint", incompatible)
    state = study.inspect_seed(157, tmp_path)
    assert state["checkpoint_reused"] == checkpoint_available
    assert not state["gradient_reused"]
    assert state["planned_new_fits"] == (11 if checkpoint_available else 12)
    assert len(state["labeled_ids"]) == 653 and state["new_fits"] == 0
    assert state["fallback_reasons"] and state["test_truth_access_count"] == 0


def test_actual_gradient_reindexing_matches_canonical_ids():
    state = study.reconstruct(157)
    directory = study.BASELINE / f'runtime/seed_157/lcmd/round_{state["source_round"]:02d}'
    record = study.read_json(directory / "contract.json")
    state.update(source_checkpoint_sha256=record["checkpoint_sha256"], checkpoint_state_hash=record["checkpoint_state_hash"])
    reordered, _, _ = study.source_features(state)
    with np.load(directory / "acquisition_artifacts/current_gradient_features.npz") as source:
        positions = {int(i): j for j, i in enumerate(source["canonical_indices"])}
        expected = source["features"][[positions[i] for i in state["outer_indices"]]]
        assert not np.array_equal(source["canonical_indices"], state["outer_indices"])
    np.testing.assert_array_equal(reordered, expected)


def test_frozen_artifact_mutation_and_seal_change_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "STUDY", tmp_path)
    (tmp_path / "seal.json").write_text("{}")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("[1]")
    record = runner.freeze(tmp_path / "freeze.json", [artifact])
    assert runner.verify_record(tmp_path / "freeze.json") == record
    artifact.write_text("[2]")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        runner.verify_record(tmp_path / "freeze.json")
    (tmp_path / "seal.json").write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="invalid continuation"):
        runner.verify_record(tmp_path / "freeze.json")


def test_report_barrier_precedes_any_label_access(monkeypatch):
    accessed = []
    def denied():
        raise RuntimeError("missing continuation")
    monkeypatch.setattr(runner, "global_freeze", denied)
    monkeypatch.setattr(reporting, "RestrictedLabelStore", lambda *a: accessed.append(a))
    with pytest.raises(RuntimeError, match="missing continuation"):
        reporting.report()
    assert not accessed


def test_synthetic_five_method_reporting_and_crossing_rebounds():
    rows = []
    for seed in study.CONFIRMATION_SEEDS:
        for method in reporting.METHODS:
            for r, n in enumerate(study.ACTIVE_LABEL_BUDGETS):
                e = 1 - r * .02
                rows.append({"outer_seed": seed, "method": method, "active_label_count": n,
                             reporting.METRIC: e, "V1_RMSE": 2*e, "V2_RMSE": 3*e,
                             "V1_MAE": e, "V2_MAE": e, "V1_R2": 1-e, "V2_R2": 1-e})
    curves = pd.DataFrame(rows)
    full = curves.loc[curves.method.eq("random") & curves.active_label_count.eq(1005)].copy()
    full[reporting.METRIC] = .3
    tables = reporting.summarize(curves, full, {s: {"V1": 2., "V2": 3.} for s in study.CONFIRMATION_SEEDS})
    assert len(tables["summary"]) == 30
    assert set(tables["labels_to_target"].target) == {"T_R30", "T80", "T90", "T95"}
    np.testing.assert_allclose(tables["summary"].normalized_AULC, .79)
    np.testing.assert_allclose(tables["summary"].late_normalized_partial_AULC, .69)
    np.testing.assert_allclose(tables["learning_curve_metrics"].V1_NRMSE, curves[reporting.METRIC])
    with pytest.raises(ValueError):
        reporting.summarize(curves.iloc[1:], full, {})
    c = reporting.crossings([653, 685, 717, 749], [1., .7, .9, .6], .8)
    assert c["interpolated_labels"] == pytest.approx(674.333333)
    assert c["first_observed_labels"] == 685 and c["sustained_labels"] == 749


@pytest.fixture
def synthetic_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    destination = tmp_path / "switch"
    destination.mkdir()
    monkeypatch.setattr(study, "STUDY", destination)
    (destination / "seal.json").write_text("{}")
    roles = {"l0": np.arange(333), "u0": np.arange(333, 3330),
             "validation": np.arange(3330, 3746), "test": np.arange(3746, 4163)}
    partition = pd.DataFrame({"sample_id": [f"id{i}" for i in range(4163)], "canonical_index": np.arange(4163),
                              "role": [k for k, v in roles.items() for _ in v]})
    source = tmp_path / "source.csv"
    pd.DataFrame({"sample_id": partition.sample_id,
                  "V1_ml": [i if i < 3746 else "TEST_FORBIDDEN" for i in range(4163)],
                  "V2_ml": [2*i if i < 3746 else "TEST_FORBIDDEN" for i in range(4163)]}).to_csv(source, index=False)
    lineage = {"seed": 157, "source_round": study.switch_round(), "checkpoint_reused": True, "gradient_reused": True,
               "labeled_indices": list(range(653)), "unlabeled_indices": list(range(653, 3330)),
               "labeled_ids": [f"id{i}" for i in range(653)], "outer_indices": list(range(3330))}
    study.atomic_json(destination / "lineage/seed_157.json", lineage)
    stores, fit_calls = [], []
    class Context:
        def __init__(self, seed, runtime):
            self.seed, self.runtime = seed, runtime
            self.roles, self.outer = roles, np.arange(3330)
            self.l0_truth = np.column_stack([roles["l0"], 2*roles["l0"]])
        def ids(self, indices):
            return [f"id{i}" for i in indices]
        def new_method_store(self):
            store = RestrictedLabelStore(source, partition)
            store.reveal(self.ids(roles["l0"]), "initial_fit")
            store.reveal(self.ids(roles["validation"]), "initial_fit")
            stores.append(store)
            return store
    def fake_fit(context, lineage, r, labeled, truth, directory):
        assert r >= study.switch_round()
        np.testing.assert_array_equal(truth, np.column_stack([labeled, 2*np.asarray(labeled)]))
        if r > study.switch_round():
            fit_calls.append(r)
        model = directory / "model"
        model.mkdir(exist_ok=True)
        audit = {"train_rows": len(labeled), "test_labels_used_for_fit_or_checkpoint_selection": 0,
                 "best_validation_combined_normalized_rmse": .5, "initialization_hash": "same"}
        (model / "best.pt").write_text("synthetic")
        (model / "predictions.csv.gz").write_text("synthetic")
        study.atomic_json(model / "fit_audit.json", audit)
        return model, audit
    bank = np.random.default_rng(79).normal(size=(3330, 4))
    monkeypatch.setattr(study, "validate_seal", lambda: {})
    monkeypatch.setattr(study, "_partition", lambda seed: partition)
    monkeypatch.setattr(study, "make_context", Context)
    monkeypatch.setattr(runner, "fit", fake_fit)
    monkeypatch.setattr(runner, "features", lambda *a: (bank, {"elapsed_seconds": 0.}, []))
    return destination, stores, fit_calls, fake_fit


def test_interrupted_resume_is_deterministic_and_never_fits_prefix(synthetic_runtime, monkeypatch):
    destination, stores, fit_calls, fake_fit = synthetic_runtime
    interrupted = [False]
    def interrupt(context, lineage, r, labeled, truth, directory):
        if r == study.switch_round() + 2 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        return fake_fit(context, lineage, r, labeled, truth, directory)
    monkeypatch.setattr(runner, "fit", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.execute_seed(157)
    first = study.read_json(destination / "runtime/seed_157/round_10/selection.json")
    result = runner.execute_seed(157)
    assert result["final_active_labels"] == 1005
    assert fit_calls == list(range(study.switch_round()+1, len(study.ACTIVE_LABEL_BUDGETS)))
    assert runner.execute_seed(157) == result
    assert study.read_json(destination / "runtime/seed_157/round_10/selection.json") == first
    assert all(a["purpose"] != "final_test_evaluation" for s in stores for a in s.audit)
    assert not list((destination / "runtime/seed_157").glob("round_0*"))
    # All continuation truth accesses are the restored prefix or frozen batches.
    assert all(a["requested_rows"] in (333, 416, 320, 32) for s in stores for a in s.audit)
    original_ids = [study.read_json(destination / f"runtime/seed_157/round_{r:02d}/selection.json")["selected_ids"]
                    for r in range(study.switch_round(), 21)]
    fresh = destination.parent / "fresh"
    fresh.mkdir()
    (fresh / "seal.json").write_text("{}")
    study.atomic_json(fresh / "lineage/seed_157.json", study.read_json(destination / "lineage/seed_157.json"))
    monkeypatch.setattr(study, "STUDY", fresh)
    runner.execute_seed(157)
    assert original_ids == [study.read_json(fresh / f"runtime/seed_157/round_{r:02d}/selection.json")["selected_ids"]
                            for r in range(study.switch_round(), 21)]


def test_completed_trajectory_checks_nested_artifacts(synthetic_runtime):
    destination, _, _, _ = synthetic_runtime
    runner.execute_seed(157)
    model = destination / "runtime/seed_157/round_15/model/best.pt"
    model.write_text("tampered")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        runner.execute_seed(157)


def test_unknown_seed_rejected_before_execution():
    with pytest.raises(ValueError, match="outside"):
        runner.execute_seed(42)


def test_execution_modules_have_no_test_reveal_calls():
    for module in (study, runner):
        tree = ast.parse(__import__("inspect").getsource(module))
        assert not any(isinstance(n, ast.Constant) and n.value == "final_test_evaluation" for n in ast.walk(tree))

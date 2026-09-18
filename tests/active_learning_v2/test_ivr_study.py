import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2 import ivr_study as study
from src.qgeognn_al.active_learning_v2.sequential_protocol import ACTIVE_LABEL_BUDGETS, CONFIRMATION_SEEDS


def test_report_cannot_access_test_truth_before_global_barrier(monkeypatch):
    calls = []
    def denied():
        raise RuntimeError("incomplete trajectories")
    monkeypatch.setattr(study, "global_freeze", denied)
    monkeypatch.setattr(study, "RestrictedLabelStore", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="incomplete trajectories"):
        study.report()
    assert calls == []


def test_resume_rejects_mutated_frozen_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "STUDY", tmp_path)
    (tmp_path / "seal.json").write_text("{}")
    artifact = tmp_path / "selected.json"
    artifact.write_text("[1,2]")
    record = {"test_truth_access_count": 0, "seal_sha256": study.sha256_file(tmp_path / "seal.json"),
              "files": study.hashes([artifact])}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(record))
    assert study.verify_record(freeze) == record
    artifact.write_text("[1,3]")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        study.verify_record(freeze)


def test_crossings_distinguish_interpolation_observation_and_rebound():
    result = study.target_crossings(np.array([333, 365, 397, 429]), np.array([1., .7, .9, .6]), .8)
    assert result["interpolated_labels"] == pytest.approx(354.3333333)
    assert result["first_observed_labels"] == 365
    assert result["sustained_through_final_labels"] == 429
    assert not result["censored"]
    assert study.target_crossings(np.array([333, 365]), np.array([1., .9]), .8)["censored"]


def test_complete_synthetic_reporting_and_reject_incomplete_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    rows = []
    for seed in CONFIRMATION_SEEDS:
        for method, final in (("random", .7), ("lcmd", .43), (study.METHOD, .36)):
            for round_index, budget in enumerate(ACTIVE_LABEL_BUDGETS):
                error = 1 + (final - 1) * round_index / 21
                rows.append({"outer_seed": seed, "method": method, "round": round_index,
                             "active_label_count": budget, study.METRIC: error,
                             "V1_RMSE": error * 3, "V2_RMSE": error * 4})
    curves = pd.DataFrame(rows)
    full = pd.DataFrame({"outer_seed": CONFIRMATION_SEEDS, study.METRIC: [.3] * 5})
    result = study.write_report(curves, full, pd.DataFrame(), pd.DataFrame())
    assert result["decision"] == "PROMISING_IVR_EXTENSION"
    assert result["paired_AULC_wins_vs_lcmd"] == 5
    assert (tmp_path / "figures/nrmse_vs_active_labels.png").stat().st_size > 1000
    with pytest.raises(RuntimeError, match="incomplete comparison matrix"):
        study.write_report(curves.iloc[1:], full, pd.DataFrame(), pd.DataFrame())


def test_seed_outside_frozen_cohort_rejected_before_execution():
    with pytest.raises(ValueError, match="outside the frozen cohort"):
        study.execute_seed(42)


def test_interrupted_trajectory_resumes_without_refitting_or_revealing_test(tmp_path, monkeypatch):
    root, baseline, destination = tmp_path, tmp_path / "baseline", tmp_path / "ivr"
    destination.mkdir()
    (destination / "seal.json").write_text("{}")
    old = baseline / "runtime/seed_157"
    initial_model = old / "lcmd/round_00/model"
    initial_model.mkdir(parents=True)
    initial_model.joinpath("fit_audit.json").write_text(json.dumps({"initialization_hash": "same"}))
    context_contract = {"preregistration_commit": "test"}
    old.joinpath("context.json").write_text(json.dumps({"contract": context_contract}))
    source = root / "source.csv"
    pd.DataFrame({"sample_id": [f"id{i}" for i in range(4163)],
                  "V1_ml": np.arange(4163), "V2_ml": np.arange(4163) * 2}).to_csv(source, index=False)
    roles = {"l0": np.arange(333), "u0": np.arange(333, 3330),
             "validation": np.arange(3330, 3746), "test": np.arange(3746, 4163)}
    partition = pd.DataFrame({"sample_id": [f"id{i}" for i in range(4163)],
                              "canonical_index": np.arange(4163),
                              "role": [key for key, values in roles.items() for _ in values]})
    stores, fit_calls = [], []
    interrupted = [False]
    class Context:
        def __init__(self, seed, partition, runtime, commit):
            self.seed, self.runtime = seed, runtime
            self.context_contract = context_contract
            self.roles, self.outer = roles, np.arange(3330)
            self.l0_truth = np.column_stack([roles["l0"], roles["l0"] * 2])
        def ids(self, indices):
            return [f"id{i}" for i in indices]
        def new_method_store(self):
            store = study.RestrictedLabelStore(source, partition)
            store.reveal(self.ids(roles["l0"]), "initial_fit")
            store.reveal(self.ids(roles["validation"]), "initial_fit")
            stores.append(store)
            return store
    def fit(context, round_index, labeled, truth, directory):
        if round_index == 3 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("simulated interruption")
        np.testing.assert_array_equal(truth, np.column_stack([labeled, labeled * 2]))
        fit_calls.append(round_index)
        model = directory / "model"
        model.mkdir(exist_ok=True)
        audit = {"initialization_hash": "same", "train_rows": len(labeled)}
        (model / "fit_audit.json").write_text(json.dumps(audit))
        (model / "best.pt").write_text("mock model")
        (model / "predictions.csv.gz").write_text("mock frozen predictions")
        return model, audit
    features = np.random.default_rng(6).normal(size=(3330, 3))
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "STUDY", destination)
    monkeypatch.setattr(study, "BASELINE", baseline)
    monkeypatch.setattr(study, "validate_seal", lambda: {})
    monkeypatch.setattr(study, "_partition", lambda seed: partition)
    monkeypatch.setattr(study, "SequentialSeedContext", Context)
    monkeypatch.setattr(study, "_fit", fit)
    monkeypatch.setattr(study, "_features", lambda *args: (features, {"elapsed_seconds": 0.}, []))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.execute_seed(157)
    result = study.execute_seed(157)
    assert result["final_active_labels"] == 1005
    assert fit_calls == list(range(22))
    assert study.execute_seed(157) == result
    assert fit_calls == list(range(22))
    assert all(row["purpose"] != "final_test_evaluation" for store in stores for row in store.audit)
    last = study.read_json(destination / "runtime/seed_157/round_21/input.json")
    assert len(set(last["ordered_labeled_indices"])) == 1005

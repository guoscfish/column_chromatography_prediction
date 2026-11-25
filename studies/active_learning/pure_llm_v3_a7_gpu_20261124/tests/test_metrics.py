import json

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import metrics


def test_auc_uses_raw_curve_without_smoothing_or_bridge():
    values = [{"active_labels": n, metrics.SCORE: v} for n, v in zip([333, 365, 397], [1.0, 0.5, 0.7])]
    assert metrics.aulc(values) == pytest.approx(0.675)
    assert metrics.aulc(values[:1]) is None
    with pytest.raises(ValueError, match="contiguous"):
        metrics.aulc(values[::2])


def test_snapshot_preserves_physical_logical_and_label_denominators(tmp_path):
    run = tmp_path
    seed = run / "seed_157"
    initial = seed / "initial"
    initial_metrics = initial / "anchor/validation_metrics.json"
    once(initial_metrics, {metrics.SCORE: 0.5, "training_seconds": 23.0})
    once(initial / "initial.json", {"fit_path": "anchor",
                                    "validation_metrics_sha256": file_digest(initial_metrics)})
    first = seed / "round_00/turn_00"
    once(first / "request.json", {})
    began = {"request_id": "mock-a", "attempt": 0, "attempt_id": "one"}
    once(first / "attempt_00.started.json", began)
    once(first / "attempt_00.terminal.json", {**began, "status": "retryable", "error": "http_429",
                                             "elapsed_seconds": 1.25})
    once(first / "attempt_01.started.json", {**began, "attempt": 1, "attempt_id": "two"})
    pending = metrics.snapshot(run)["seeds"][0]
    assert pending["latest_budget"] == 333
    assert pending["training_seconds_observed"] == 0
    assert pending["total_logical_requests_logged"] == 1
    assert pending["total_physical_attempts_started"] == 2
    assert pending["total_complete_logical_responses"] == 0
    assert pending["rounds"][0]["open_or_unreconciled_attempts"] == 1
    assert pending["transport_failures"] == {"http_429": 1}
    assert pending["first_response_legality"]["fraction"] is None


def test_local_repair_preservation_compares_full_locked_choices():
    locked = {"candidate_id": "C-1", "reason": "Reason with bound evidence",
              "references": [{"reference_id": "O-2"}]}
    changed = {"candidate_id": "C-2", "reason": "Original reason"}
    decision = {"state": "rejected"}
    answer = {"text": json.dumps({"choices": [locked, changed]})}
    summary = metrics._repair_metrics(
        [locked, {**changed, "reason": "Rewritten"}], {"decision": decision, "answer": answer},
        {"locked_choice_slots": [0, 1]},
    )
    assert summary == {"initial_locked_choices": 2, "retained_initial_legal_id_count": 2,
                       "retained_initial_legal_choice_count": 1}


def test_pilot_scope_and_frozen_acquisition_diagnostics(tmp_path):
    once(tmp_path / "formal_contract.json", {
        "profile": {"confirmation_seeds": [157, 887, 2357], "target_labels": 653,
                    "requires_initial_qualification": False},
    })
    report = metrics.snapshot(tmp_path)
    assert report["planned_rounds"] == 10 and report["target_labels"] == 653
    assert [row["seed"] for row in report["seeds"]] == [157, 887, 2357]
    assert not report["scope_complete"]

    round_dir = tmp_path / "seed_157" / "round_00"
    selection = round_dir / "selection.json"
    base = dict.fromkeys(metrics.CONDITION_AXES, "")
    base.update({"CAS": "1", "column_specs": "4g", "purity_raw": "99%",
                 "PE/EA": "5/1", "flow": "2", "density": "1",
                 "sample_volume": "5", "loading_solvent": "DCM", "loading_solvent_volume": "20"})
    once(selection, {"choices": [{"candidate_id": "C-1"}, {"candidate_id": "C-2"},
                                  {"candidate_id": "C-3"}], "hypothesis_updates": []})
    freeze = {"selection_file_sha256": file_digest(selection), "freeze_sha256": "f" * 64}
    observed = {"O-1": {"molecule_id": "M-1", "conditions": base}}
    candidates = {"C-1": {"molecule_id": "M-1", "conditions": {**base, "flow": "3"}},
                  "C-2": {"molecule_id": "M-2", "conditions": base},
                  "C-3": {"molecule_id": "M-2", "conditions": {**base, "flow": "3", "PE/EA": "8/1"}}}
    metrics.record_batch_diagnostics(round_dir, freeze, candidates, observed)
    report = metrics.optional(round_dir / metrics.ACQUISITION_DIAGNOSTICS)
    assert report["freeze_sha256"] == freeze["freeze_sha256"]
    assert report["new_molecules"] == 1 and report["row_fraction_on_new_molecules"] == pytest.approx(2 / 3)
    assert report["condition_coverage"]["complete_profiles"]["added"] == 2
    assert report["same_molecule_pairs"]["within_batch"]["multifactor"] == 1
    assert report["same_molecule_pairs"]["with_existing_observations"]["single_factor"] == 1

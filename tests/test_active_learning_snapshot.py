import json
from pathlib import Path

import pytest

from scripts import report_active_learning_snapshot as reporting


def test_completed_fit_conflicts_are_rejected():
    rows = {333: .8}
    reporting.merge_point(rows, 333, .8)
    with pytest.raises(ValueError, match="Conflicting"):
        reporting.merge_point(rows, 333, .7)
    with pytest.raises(ValueError, match="Invalid"):
        reporting.merge_point(rows, 365, float("nan"))


def test_llm_snapshot_requires_a_matching_prediction_freeze(tmp_path, monkeypatch):
    base = tmp_path / "studies/active_learning"
    scientist = base / "scientist"
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    monkeypatch.setattr(reporting, "BASE", base)
    monkeypatch.setattr(reporting, "SCIENTIST", scientist)
    monkeypatch.setattr(reporting, "SPEC", {"Random32": [("qgeognn_v2_row_sequential_b32", "random")]})

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def audit(labels, value):
        return {"train_rows": labels, "best_validation_combined_normalized_rmse": value,
                "checkpoint_sha256": "sealed", "test_labels_used_for_fit_or_checkpoint_selection": 0}

    for seed in (157, 6101):
        write(base / f"qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/random/round_00/model/fit_audit.json",
              audit(333, .8))
    runtime = scientist / "runtime/seed_157/free_llm32_scientist_v2"
    model = runtime / "round_01/model"
    write(model / "fit_audit.json", audit(365, .7))
    freeze = runtime / "round_01/prediction_freeze.json"
    write(freeze, {"checkpoint_path": str(model / "best.pt"), "checkpoint_sha256": "sealed"})
    write(runtime / "round_02/model/fit_audit.json", audit(397, .6))
    snapshot = reporting.collect("unused")
    assert snapshot["curves"]["157"]["Free LLM32"] == [{"labels": 365, "nrmse": .7}]
    assert not any("round_02" in p for p in snapshot["fit_audits"])
    write(freeze, {"checkpoint_path": str(model / "best.pt"), "checkpoint_sha256": "different"})
    with pytest.raises(ValueError, match="disagrees"):
        reporting.collect("unused")

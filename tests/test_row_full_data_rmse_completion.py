"""Ensure completion cannot evaluate tests before the global prediction freeze."""
import json

import pytest

from scripts.studies import run_row_full_data_rmse_completion as completion


def test_test_truth_is_blocked_before_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "STUDY", tmp_path)
    parsed = []
    monkeypatch.setattr(completion.parent, "_read_authorized_truth", lambda *args: parsed.append(args))
    with pytest.raises(RuntimeError, match="not globally frozen"):
        completion.truth_for_role("8g", completion.SEEDS[0], "test")
    assert parsed == []


def test_prediction_tamper_blocks_test_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, "STUDY", tmp_path)
    (tmp_path / "protocol.json").write_text("{}")
    records = {}
    for i in range(15):
        runtime = tmp_path / f"context_{i}"
        runtime.mkdir()
        prediction = runtime / "predictions_blind.csv.gz"
        prediction.write_bytes(b"frozen-blind-predictions")
        frozen = runtime / "frozen.json"
        frozen.write_text(json.dumps({
            "contract": {"test_labels_used_for_fit_or_selection": 0},
            "files": {prediction.name: completion.sha256_file(prediction)},
        }))
        records[str(frozen.relative_to(tmp_path))] = completion.sha256_file(frozen)
    (tmp_path / "all_predictions_frozen.json").write_text(json.dumps({
        "contexts": 15, "methods": list(completion.METHODS),
        "protocol_sha256": completion.sha256_file(tmp_path / "protocol.json"), "files": records,
    }))
    (tmp_path / "context_0/predictions_blind.csv.gz").write_bytes(b"changed")
    parsed = []
    monkeypatch.setattr(completion.parent, "_read_authorized_truth", lambda *args: parsed.append(args))
    with pytest.raises(RuntimeError, match="Global prediction drift"):
        completion.truth_for_role("8g", completion.SEEDS[0], "test")
    assert parsed == []

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/data_consumption_register.json"
MANIFEST = ROOT / "experiments/data_audit/data_manifest.csv"


def test_data_consumption_register_consistency() -> None:
    register = json.loads(REGISTER.read_text())
    rows = register["datasets"]
    assert [row["dataset_id"] for row in rows] == [
        "4g", "8g", "25g", "40g", "C18", "CN", "NH2", "DCM"
    ]

    required = {
        "dataset_id", "source_file", "source_sha256", "training_use",
        "validation_model_selection", "test_truth_read", "outcomes_inspected",
        "experiments", "developmental_available", "pristine_confirmatory_available", "notes",
    }
    assert all(required <= set(row) for row in rows)
    assert all(row["outcomes_inspected"] for row in rows)
    assert all(not row["pristine_confirmatory_available"] for row in rows)
    assert all(row["developmental_available"] for row in rows)

    with MANIFEST.open(newline="") as handle:
        raw_rows = [row for row in csv.DictReader(handle) if row["stage"] == "raw_csv"]
    manifest = {row["dataset_id"]: row for row in raw_rows}
    assert set(manifest) == {row["dataset_id"] for row in rows}

    for row in rows:
        audit = manifest[row["dataset_id"]]
        assert row["source_file"] == audit["source_file"]
        assert row["source_sha256"] == audit["source_sha256"]
        source = ROOT / row["source_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]

    eight_g = next(row for row in rows if row["dataset_id"] == "8g")
    for stage in ("G0", "E1", "E4", "S1", "T1", "T1b-1"):
        assert stage in eight_g["experiments"]

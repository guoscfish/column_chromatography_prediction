"""Check research accounting, original-source immutability, and offline mock tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def csv_rows(name):
    with (HERE / name).open() as file:
        return list(csv.DictReader(file))


def main():
    initial = read("preflight_initial.json")
    source_drift = [name for name, expected in initial["protected_hashes"].items()
                    if sha(ROOT / name) != expected]
    if source_drift:
        raise AssertionError(f"Protected production source/protocol changed: {source_drift}")
    snap = read("history_snapshot_manifest.json")
    for name, entry in snap["files"].items():
        if sha(ROOT / name) != entry["sha256"]:
            raise AssertionError(f"Historical evidence drift: {name}")
    summary = read("history_audit_summary.json")
    metrics = csv_rows("reliability_metrics.csv")
    failures = csv_rows("failure_events.csv")
    repairs = csv_rows("repair_trajectory_analysis.csv")
    responses = [json.loads(line) for line in (HERE / "response_manifest.jsonl").read_text().splitlines()]
    corrections = csv_rows("correction_request_causes.csv")
    benchmark = csv_rows("offline_benchmark_summary.csv")
    assert len(metrics) == len(snap["batches"]) == summary["batches"]
    assert len(responses) == summary["complete_responses"]
    assert len(repairs) == summary["corrected_rounds"]
    assert sum(int(x["completed_llm_responses"]) for x in metrics) == len(responses)
    assert sum(int(x["llm_correction_requests"]) for x in metrics) == len(corrections)
    assert len(failures) == summary["failure_event_rows"]
    assert len({(r["method"], r["seed"], r["round"]) for r in metrics}) == len(metrics)
    assert all(r["final_frozen_batch_legal"] == "True" for r in metrics)
    assert all(not r["cost"] for r in metrics), "Missing tariffs cannot be made-up costs"
    assert all(r["A_legal"] == r["B_legal"] == r["C_legal"] for r in benchmark)
    assert all(float(r["A_B_agreement"]) == 1 for r in benchmark)
    assert all(float(r["A_original_audit_agreement"]) == 1 for r in benchmark)
    assert sum(int(r["complete_responses"]) for r in benchmark) == len(responses)
    assert sum(int(r["A_legal"]) for r in benchmark) == sum(r["original_validator_legal"] for r in responses)
    assert sum(int(r["safe_format_rescues"]) for r in benchmark) == 0
    for group in summary["groups"]:
        rows = [r for r in metrics if r["method_group"] == group["method_group"]]
        assert len(rows) == group["frozen_batches"]
        assert sum(r["first_output_legal"] == "True" for r in rows) == group["first_output_legal_numerator"]
    geometry = read("repair_geometry_summary.json")
    geometry_rows = csv_rows("repair_geometry_profiles.csv")
    assert geometry["snapshot_captured_at"] == snap["captured_at"]
    assert geometry["correction_rounds"] == len(repairs)
    assert geometry["profile_rows"] == len(geometry_rows)
    assert geometry["available_bank_rounds"] + geometry["missing_bank_rounds"] == len(repairs)
    for name, expected in geometry["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise AssertionError(f"Geometry source drift: {name}")
    assert geometry["new_gradient_extractions"] == geometry["new_training_runs"] == geometry["new_llm_calls"] == 0
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py", "-v"]
    run = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                         text=True, capture_output=True, check=False)
    output = run.stdout + run.stderr
    (HERE / "test_results.txt").write_text(output)
    if run.returncode:
        print(output)
        raise AssertionError("Offline test suite failed")
    match = re.search(r"Ran (\d+) tests? in", output)
    if not match:
        raise AssertionError("Unittest count not available")
    result = {
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS",
        "protected_source_files_unchanged": len(initial["protected_hashes"]),
        "historical_source_files_verified": len(snap["files"]),
        "cohort_cutoff": snap["captured_at"],
        "frozen_batches": len(metrics), "complete_responses": len(responses),
        "invalid_complete_responses": sum(not r["original_validator_legal"] for r in responses),
        "additional_complete_correction_requests": len(corrections),
        "A_B_agreement": 1, "safe_format_rescues": 0,
        "repair_geometry_available_rounds": geometry["available_bank_rounds"],
        "repair_geometry_missing_rounds": geometry["missing_bank_rounds"],
        "repair_geometry_source_files_verified": len(geometry["source_sha256"]),
        "test_command": command, "test_count": int(match.group(1)),
        "test_exit_code": run.returncode,
        "prohibited_actions": {"new_real_llm_calls": 0, "new_training_runs": 0,
                               "new_label_requests": 0, "test_truth_reads": 0,
                               "worker_or_lock_mutations": 0},
        "boundary_evidence": "Pure offline new modules; protected original code/protocol byte hashes unchanged. Action zeros describe this audit execution, not historical runs.",
    }
    (HERE / "preflight_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

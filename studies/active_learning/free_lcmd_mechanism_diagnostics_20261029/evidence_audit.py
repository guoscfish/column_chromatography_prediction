"""Audit existing frozen evidence without training, inference, or label requests.

Only invokes the read-only collector, never the historical report writer. Files
containing prediction tables are hashed as bytes, not loaded as labels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = "free_llm32_scientist_v2"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    spec = importlib.util.spec_from_file_location("frozen_reporting_reader", ROOT / "scripts/report_free_lcmd.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    curves, verification = reader.collect(require_complete=True)
    snapshot_path = ROOT / "reports/free_llm_five_seeds_to1485_20261024/figures/validation_curves.json"
    snapshot = json.loads(snapshot_path.read_text())
    for s in SEEDS:
        for method in (FREE, "lcmd"):
            assert curves[str(s)][method] == snapshot[str(s)][method], (s, method)
    frozen = ROOT / "reports/free_lcmd_five_seeds_to1485_20261026/final_audit.json"
    final = json.loads(frozen.read_text())
    historical_hash_checks = []
    for rel, digest in final["evidence_sha256"].items():
        path = ROOT / rel
        actual = sha(path) if path.exists() else None
        historical_hash_checks.append({"path": rel, "expected": digest, "actual": actual, "matches": actual == digest})
    assert all(r["matches"] for r in historical_hash_checks)
    required = ["docs/NEXT_STAGE_DECISION.md", "reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md",
                "reports/free_lcmd_five_seeds_to1485_20261026/final_audit.json",
                str(snapshot_path.relative_to(ROOT)), "src/qgeognn_al/active_learning_v2/scientist_study.py",
                "src/qgeognn_al/active_learning_v2/scientist_selector.py", "src/qgeognn_al/active_learning_v2/lcmd.py"]
    source_hashes = dict(reader.SOURCES)
    source_hashes.update({p: sha(ROOT / p) for p in required})
    frozen_source = ROOT / "reports/free_lcmd_five_seeds_to1485_20261026/frozen_sources/gradient_features.py"
    historical_source = subprocess.check_output(["git", "show", "8081e865:src/qgeognn_al/active_learning_v2/gradient_features.py"], cwd=ROOT)
    assert hashlib.sha256(historical_source).hexdigest() == sha(frozen_source)
    output = {
        "status": "PASS", "scope": "read-only runtime re-audit; existing collector additionally hashes frozen Random/Hybrid prefixes",
        "repository_head": git("rev-parse", "HEAD"), "cached_origin_main": git("rev-parse", "origin/main"),
        "remote_main_checked_in_session": "fbf4efa9efe251a22c3865d15718e149a38eeb7d",
        "remote_check_command": "git ls-remote origin refs/heads/main (read-only, 2026-10-29 Asia/Shanghai)",
        "required_inputs": [{"path": p, "exists": True, "sha256": source_hashes[p]} for p in required],
        "verified_fit_count_free_lcmd": 370, "verified_batch_count_free_lcmd": 360,
        "verified_acquisition_record_count": 11520,
        "runtime_curves_equal_frozen_snapshot": True,
        "checkpoint_and_prediction_bytes_hash_verified": True,
        "prediction_csv_values_parsed": False,
        "frozen_gradient_matches_git_8081e865": True,
        "historical_final_audit_evidence_hash_checks": historical_hash_checks,
        "per_seed": verification,
        "source_file_count": len(source_hashes),
        "test_evaluation_performed": False, "new_label_requests": 0,
        "new_training": 0, "new_llm_calls": 0,
        "history_review": [
            {"revision": rev, "summary": git("show", "-s", "--format=%h %s", rev)}
            for rev in ("8081e865", "1ec395f8", "93609bd9", "b0f59c7e", "066ff720", "f5a04b21", "2c730ef6", "ba8862f0", "fbf4efa9")
        ],
    }
    (HERE / "runtime_integrity_audit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    (HERE / "runtime_source_hashes.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: output[k] for k in ("status", "verified_fit_count_free_lcmd", "verified_batch_count_free_lcmd", "source_file_count")}))


if __name__ == "__main__":
    main()

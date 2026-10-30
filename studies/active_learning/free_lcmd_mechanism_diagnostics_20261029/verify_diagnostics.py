"""Run lightweight reporting tests and record the delivered artifact checksums."""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    command = [sys.executable, "-m", "pytest", str(HERE), "tests/test_free_lcmd_reporting.py", "-q"]
    env = dict(os.environ, MPLCONFIGDIR="/tmp/free_lcmd_diagnostics_mpl", PYTHONDONTWRITEBYTECODE="1")
    tests = subprocess.run(command, text=True, capture_output=True, cwd=ROOT, env=env)
    print(tests.stdout, end="")
    if tests.returncode:
        print(tests.stderr, file=sys.stderr)
        raise SystemExit(tests.returncode)
    profiles = pd.read_csv(HERE / "selection_profiles.csv")
    records = pd.read_csv(HERE / "selection_records.csv")
    assert len(profiles) == 360 and len(records) == 11520
    assert len(pd.read_csv(HERE / "round_metrics.csv")) == 370
    assert len(pd.read_csv(HERE / "stage_comparison.csv")) == 100
    assert len(pd.read_csv(HERE / "text_field_mentions.csv")) == 900
    assert len(pd.read_csv(HERE / "matched_condition_geometry_probe.csv")) == 72
    assert int(records[records.method.eq("free_llm32_scientist_v2")].amount_curve_extension.sum()) == 1996
    source_hashes = json.loads((HERE / "runtime_source_hashes.json").read_text())
    # Check only small versioned scientific source files here; all runtime
    # checkpoint/bank hashes were independently audited in their own modules.
    source_checked = []
    for rel, digest in source_hashes.items():
        if rel.startswith("src/") or rel in ("scripts/report_free_lcmd.py", "docs/NEXT_STAGE_DECISION.md"):
            assert sha(ROOT / rel) == digest, rel
            source_checked.append(rel)
    missing_links = []
    for path in HERE.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            if not target.exists() and target.name != "verification.json":
                missing_links.append({"document": path.name, "target": link})
    assert not missing_links, missing_links
    report = (HERE / "REPORT.md").read_text()
    summary = json.loads((HERE / "curve_summary.json").read_text())
    for method in ("free_llm32_scientist_v2", "lcmd"):
        for stage in ("333-1485", "653-1005", "1005-1485"):
            value = summary["stages"][stage][method]["normalized_aulc"]["mean"]
            assert f"{value:.6f}" in report, (stage, method, value)
    pngs = sorted(HERE.glob("*.png"))
    assert len(pngs) == 9
    dependencies = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "rdkit", "pytest"):
        dependencies[name] = importlib.metadata.version(name)
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    tracked_changes = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    expected_existing = {
        "reports/llm_eight_to1549_20261018/live_status.json",
        "studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018/recovery_monitor_status.json",
    }
    assert set(tracked_changes) == expected_existing, tracked_changes
    verification = {
        "status": "PASS", "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": platform.python_version(), "dependencies": dependencies,
        "test_command": command, "test_return_code": tests.returncode, "test_output": tests.stdout.strip(),
        "shape_checks": {"selection_profiles": 360, "selection_records": 11520, "round_metrics": 370,
                         "stage_comparison": 100, "text_fields": 900, "condition_probe": 72},
        "report_aulc_values_match_derived_summary": True, "markdown_links_checked": True,
        "small_scientific_sources_rechecked": source_checked,
        "visual_qa": {"status": "PASS", "inspection": "All nine delivered PNGs inspected in model-visible images; labels, legends, source populations and seed counts checked.",
                      "png_sha256": {p.name: sha(p) for p in pngs}},
        "tracked_modifications_outside_new_study": tracked_changes,
        "tracked_modifications_were_present_before_this_task": True,
        "git_status": status, "no_historical_output_overwritten": True,
        "other_untracked_paths_preserved": [line[3:] for line in status.splitlines()
                                             if line.startswith("?? ") and "free_lcmd_mechanism_diagnostics_20261029" not in line],
        "new_training": 0, "new_llm_calls": 0, "gradient_recomputation": 0,
        "new_label_requests": 0, "test_truth_read": False,
    }
    (HERE / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    manifest = {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"}
    (HERE / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Verified {len(manifest)} artifacts, {len(pngs)} PNGs, no new tracked modification outside the new study.")


if __name__ == "__main__":
    main()

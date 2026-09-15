"""Seal verified Phase 1 artifacts without producing formal experiment results."""

import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from ..artifacts import sha256_file
from ..resources import ROOT
from ..training.predictor import atomic_json
from .benchmark_protocol import BASE_COMMIT, OLD_STUDY, STUDY, code_hashes, protocol_record
from .protocol import stable_hash


def seal_phase1(junit_path, smoke_report_path, study=STUDY):
    record = json.loads((study / "protocol.json").read_text())
    if record != protocol_record():
        raise RuntimeError("prepare must be rerun after code/protocol changes")
    suites = ET.parse(junit_path).getroot()
    leaves = [suites] if suites.tag == "testsuite" else list(suites.iter("testsuite"))
    totals = {name: sum(int(s.attrib.get(name, 0)) for s in leaves)
              for name in ("tests", "failures", "errors", "skipped")}
    if totals["tests"] == 0 or totals["failures"] or totals["errors"] or totals["skipped"]:
        raise RuntimeError("complete passing, unskipped test report required")
    smoke = json.loads(Path(smoke_report_path).read_text())
    if (smoke["status"] != "PASS" or smoke["formal_test_truth_access_count"] != 0
            or smoke["formal_benchmark_run"] or smoke["formal_decision_generated"]
            or smoke["code_contract_hash"] != stable_hash(code_hashes())):
        raise RuntimeError("passing smoke of the frozen code required")
    forbidden = ["runtime", "formal_results", "secondary_results", "FINAL_REPORT.md", "formal_decision.json",
                 "primary_results_freeze.json", "NEXT_STAGE_RECOMMENDATION.md"]
    if any((study / name).exists() for name in forbidden):
        raise RuntimeError("Phase 1 cannot contain formal execution or conclusions")
    subprocess.run(["git", "diff", "--exit-code", BASE_COMMIT, "--", str(OLD_STUDY.relative_to(ROOT)),
                    "src/qgeognn_al/active_learning_v2/lcmd.py",
                    "src/qgeognn_al/active_learning_v2/gradient_features.py"], cwd=ROOT, check=True)
    old_tree = subprocess.check_output(["git", "rev-parse", f"{BASE_COMMIT}:{OLD_STUDY.relative_to(ROOT)}"],
                                       cwd=ROOT, text=True).strip()
    provenance = json.loads((study / "results/retrospective_audit_provenance.json").read_text())
    for path, digest in provenance["read_only_sources"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("protected retrospective source changed")
    atomic_json(study / "results/smoke_report.json", smoke)
    atomic_json(study / "results/phase1_validation.json", {
        "status": "PASS", "phase": 1, "tests": totals,
        "pytest_junit_sha256": sha256_file(Path(junit_path)),
        "pytest_command": "python -m pytest -q --junitxml=/tmp/qgeognn_small_batch_phase1_tests.xml",
        "code_contract_hash": stable_hash(code_hashes()), "preserved_old_study_tree": old_tree,
        "protected_old_study_base_commit": BASE_COMMIT, "old_lcmd_and_gradient_code_unchanged": True,
        "formal_test_performance_read": False, "formal_benchmark_run": False,
        "confirmation_training_run": False, "end_to_end_hidden_label_mutation": "PASS",
        "formal_decision": None,
    })
    files = {**code_hashes(), **{str(p.relative_to(ROOT)): sha256_file(p)
                               for p in (ROOT / "tests").rglob("*.py")}}
    for path in study.rglob("*"):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(ROOT))] = sha256_file(path)
    atomic_json(study / "artifact_manifest.json", {
        "phase": "PHASE_1_PREREGISTERED", "base_commit": BASE_COMMIT,
        "formal_test_truth_access_count": 0, "files": dict(sorted(files.items())),
        "self_hash_excluded": True, "commit_binding": "Git Commit A containing this manifest",
    })
    return {"status": "SEALED_PHASE_1", "tests": totals, "manifest_files": len(files)}

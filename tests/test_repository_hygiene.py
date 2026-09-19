from pathlib import Path
import json
import re

from scripts.audit_repository_hygiene import retirement_audit

ROOT = Path(__file__).parents[1]

def test_index_keys_unique_and_d45_registered():
    text = (ROOT / "experiments/INDEX.md").read_text()
    keys = re.findall(r"^\| ([^|]+) \|", text, re.MULTILINE)
    assert len(keys) == len(set(keys))
    assert keys.count("d45_oracle_marginal_utility") == 1

def test_d45_script_registered():
    registry = json.loads((ROOT / "docs/repository/RETIREMENTS.json").read_text())
    records = {record["path"]: record for record in registry["records"]}
    assert records["scripts/run_d45_oracle_marginal_utility.py"]["retained_record"] == "experiments/d45_oracle_marginal_utility"


def test_d46_registered_once_and_runner_mapped():
    index = (ROOT / "experiments/INDEX.md").read_text()
    keys = re.findall(r"^\| ([^|]+) \|", index, re.MULTILINE)
    assert keys.count("d46_oracle_utility_reliability") == 1
    registry = json.loads((ROOT / "docs/repository/RETIREMENTS.json").read_text())
    records = {record["path"]: record for record in registry["records"]}
    assert records["scripts/run_d46_oracle_utility_reliability.py"]["retained_record"] == "experiments/d46_oracle_utility_reliability"


def test_d46_readme_contract():
    readme = (ROOT / "experiments/d46_oracle_utility_reliability/README.md").read_text()
    for section in "ABCDEFGHIJKLM":
        assert f"## {section}." in readme


def test_retired_code_has_recoverable_sources_retained_evidence_and_no_importers():
    audit = retirement_audit(ROOT)
    assert audit["violations"] == []


def test_current_navigation_links_resolve():
    pages = [
        "README.md", "docs/README.md", "docs/NEXT_STAGE_DECISION.md",
        "docs/repository/STRUCTURE.md", "docs/repository/REPOSITORY_HYGIENE_AUDIT.md",
        "docs/research/CROSS_COLUMN_TRANSFER_STATUS.md", "docs/CODEBASE_CONVENTIONS.md",
        "scripts/README.md", "src/qgeognn_al/README.md", "studies/README.md",
        "studies/predictor/README.md", "studies/transfer/README.md",
        "studies/active_learning/README.md", "studies/active_learning/transfer/README.md",
    ]
    broken = []
    for name in pages:
        page = ROOT / name
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", page.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if not (page.parent / path).exists():
                broken.append((name, target))
    assert not broken, broken

"""Small, deterministic guards for repository and experiment boundaries."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
SCHEDULE_SHA256 = "b52b937aa750586ec8f2efb2f10f93dfddd2252bb03afaf304556e16b4fab5ee"
SEEDS = {"769539383", "1425370602", "536279090", "2767143051", "1362771960"}
BUDGETS = {"30", "50", "70", "100"}
COLUMNS = {"8g", "25g", "40g"}
PROTOCOLS = {"row", "compound"}
RUNTIME_NAMES = {
    "best.pt",
    "last.pt",
    "history.csv",
    "fit_result.json",
    "result.json",
    "predictions.csv",
    "predictions.csv.gz",
}
PAPER_STUDY = ROOT / "studies/transfer/paper_transfer_reproduction"
PAPER_RUN_SUMMARY_FIELDS = {
    "column",
    "protocol",
    "seed",
    "method",
    "best_epoch",
    "epochs_run",
    "trainable_parameters",
    "source_checkpoint_sha256",
    "normalized_valid_score",
    "V1_test_rmse",
    "V1_test_mae",
    "V1_test_r2",
    "V2_test_rmse",
    "V2_test_mae",
    "V2_test_r2",
}
PAPER_REMOVED_DERIVED = {
    "canonical_25g_legacy_filtered.csv",
    "canonical_40g_legacy_filtered.csv",
    "predictions.csv.gz",
    "splits",
}
MATCHED_METHODS = {
    "zero_shot",
    "scale_only",
    "affine",
    "local_identity_shrinkage",
    "conditional_EA",
    "target_head_only",
    "standard_shallow_finetune",
    "paper_style_current_v2",
}


def _tracked() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def _workspace_paths() -> list[str]:
    """Index plus non-ignored files, excluding the intended runtime cache."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_src_scientific_package_has_no_script_runner_imports() -> None:
    """Reusable scientific code must not depend on historical entry points."""
    offenders = {
        str(path.relative_to(ROOT)): module
        for path in (ROOT / "src/qgeognn_al").rglob("*.py")
        for module in _imports(path)
        if module == "scripts" or module.startswith("scripts.")
    }
    assert not offenders, offenders


def test_matched_study_runners_have_no_historical_top_level_imports() -> None:
    """A future matched runner may be added without reintroducing script coupling."""
    candidates = sorted(
        path
        for path in (ROOT / "scripts/studies").glob("*.py")
        if "matched" in path.name.lower()
        or "matched_rmse_benchmark" in path.read_text(encoding="utf-8")
    )
    offenders = {
        str(path.relative_to(ROOT)): module
        for path in candidates
        for module in _imports(path)
        if module == "scripts" or module.startswith("scripts.")
    }
    assert not offenders, offenders


def test_protected_artifact_manifest_entries_exist() -> None:
    payload = json.loads((ROOT / "docs/PROTECTED_ARTIFACTS.json").read_text(encoding="utf-8"))
    entries = payload["artifacts"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert entries
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert not missing, missing


def test_frozen_source_and_schedule_hashes_are_unchanged() -> None:
    protocol_path = ROOT / "studies/transfer/cross_column/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = ROOT / protocol["source_checkpoint"]
    schedule = ROOT / "studies/transfer/cross_column/splits/schedule_manifest.csv"
    assert protocol["source_checkpoint_sha256"] == SOURCE_SHA256
    assert _sha256(source) == SOURCE_SHA256
    assert protocol["schedule_sha256"] == SCHEDULE_SHA256
    assert _sha256(schedule) == SCHEDULE_SHA256


def test_primary_protocol_is_no_threshold_and_test_blind() -> None:
    protocol = json.loads(
        (ROOT / "studies/transfer/cross_column/protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["target_threshold"] is None
    assert protocol["test_tuning"] is False
    assert protocol["neural_selection"] == "validation only"
    assert protocol["simple_fit"] == "gradient_train only"
    assert protocol["budget_definition"] == "gradient_train plus validation revealed target rows"


def test_matched_protocol_reuses_the_frozen_no_threshold_schedule() -> None:
    """The required matched study must materialize the complete locked design."""
    study = ROOT / "studies/transfer/matched_rmse_benchmark"
    protocol_path = study / "protocol.json"
    assert study.is_dir(), "the matched benchmark study is required"
    assert protocol_path.is_file(), "run the benchmark --prepare phase before tests"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent_path = ROOT / "studies/transfer/cross_column/protocol.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    assert protocol["study_name"] == "MATCHED_CROSS_COLUMN_ABSOLUTE_ERROR_BENCHMARK"
    assert protocol["classification"] == "STRATEGY_MATCHED_ABSOLUTE_ERROR_BENCHMARK"
    assert protocol["parent_protocol"] == "studies/transfer/cross_column/protocol.json"
    assert protocol["target_threshold"] is None
    assert protocol["source_checkpoint"] == parent["source_checkpoint"]
    assert protocol["source_checkpoint_sha256"] == SOURCE_SHA256
    assert protocol["schedule"] == "studies/transfer/cross_column/splits/schedule_manifest.csv"
    schedule = ROOT / protocol["schedule"]
    assert _sha256(schedule) == SCHEDULE_SHA256
    assert protocol["schedule_sha256"] == SCHEDULE_SHA256
    assert set(protocol["columns"]) == COLUMNS
    assert set(protocol["protocols"]) == PROTOCOLS
    assert {str(seed) for seed in protocol["outer_seeds"]} == SEEDS
    budgets = protocol["budgets"]
    assert {int(value) for value in budgets} == {30, 50, 70, 100}
    assert set(protocol["methods"]) >= MATCHED_METHODS
    assert protocol["budget_definition"] == "gradient_train + validation revealed target rows"
    assert protocol["simple_fit_labels"] == "gradient_train only"
    assert protocol["neural_fit_labels"] == "gradient_train; frozen validation only for checkpoint selection"
    assert protocol["test_tuning"] is False
    assert protocol["test_labels_used_for_fit_or_selection"] is False
    assert protocol["canonical_sha256"] == parent["canonical_sha256"]
    for column in COLUMNS:
        canonical = ROOT / "studies/transfer/cross_column/data_audit" / f"canonical_{column}.csv"
        assert _sha256(canonical) == protocol["canonical_sha256"][column]
    tail = protocol["tail_definition"]
    assert tail["threshold_source"] == "target gradient_train labels within each seed/budget context"
    assert tail["quantiles"] == {"q50": 0.5, "q80": 0.8}
    assert tail["test_truth_used_only_after_freeze"] is True


def test_frozen_schedule_budget_ledgers_are_disjoint_and_label_safe() -> None:
    path = ROOT / "studies/transfer/cross_column/splits/schedule_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert {"V1_ml", "V2_ml"}.isdisjoint(reader.fieldnames or [])
        groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            key = (row["column"], row["protocol"], row["outer_seed"], row["planned_budget"])
            groups[key].append(row)

    expected = len(COLUMNS) * len(PROTOCOLS) * len(SEEDS) * len(BUDGETS)
    assert len(groups) == expected
    for key, rows in groups.items():
        column, protocol, seed, budget = key
        assert column in COLUMNS
        assert protocol in PROTOCOLS
        assert seed in SEEDS
        assert budget in BUDGETS
        ids = [row["sample_id"] for row in rows]
        assert len(ids) == len(set(ids)), key
        roles = {row["role"] for row in rows}
        assert roles == {"gradient_train", "validation", "test", "pool"}, key
        actual = {int(row["actual_budget"]) for row in rows}
        assert len(actual) == 1, key
        purchased = sum(row["role"] in {"gradient_train", "validation"} for row in rows)
        assert actual == {purchased}, (key, actual, purchased)
        # A test row is a held-out role, never a training or validation role.
        test_ids = {row["sample_id"] for row in rows if row["role"] == "test"}
        fit_ids = {
            row["sample_id"]
            for row in rows
            if row["role"] in {"gradient_train", "validation"}
        }
        assert test_ids.isdisjoint(fit_ids), key


def test_paper_reproduction_is_explicitly_historical_and_protocol_distinct() -> None:
    path = ROOT / "studies/transfer/paper_transfer_reproduction/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["classification"] == "PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION"
    assert protocol["primary_protocol"] == "legacy_filtered"
    assert protocol["source_checkpoint_sha256"] != SOURCE_SHA256
    assert protocol["columns"]["25g"]["legacy_thresholds_ml"] == {"V1": 60.0, "V2": 120.0}
    assert protocol["columns"]["40g"]["legacy_thresholds_ml"] == {"V1": 150.0, "V2": 200.0}


def test_study_runtime_directories_are_gitignored_and_untracked() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "studies/**/runtime/" in ignore_text
    assert "studies/**/progress/" in ignore_text
    tracked = _tracked()
    assert not [
        path for path in tracked
        if path.startswith("studies/") and ("/runtime/" in path or "/progress/" in path)
    ]


def test_paper_transfer_run_artifacts_are_not_tracked() -> None:
    """Per-run paper artifacts belong in runtime/ after the cleanup migration."""
    prefix = "studies/transfer/paper_transfer_reproduction/"
    offenders = [
        path
        for path in _workspace_paths()
        if path.startswith(prefix)
        and (
            "/runs/" in path
            or Path(path).name in RUNTIME_NAMES
            or Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}
        )
    ]
    assert not offenders, offenders


def test_paper_reproduction_keeps_compact_run_metadata() -> None:
    summary_path = PAPER_STUDY / "run_summary.csv"
    assert summary_path.is_file()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    assert PAPER_RUN_SUMMARY_FIELDS.issubset(fields)
    assert len(rows) >= 20
    contexts = {(row["column"], row["protocol"], row["seed"], row["method"]) for row in rows}
    assert len(contexts) == len(rows)
    assert {"25g", "40g"}.issubset({row["column"] for row in rows})
    assert "legacy_filtered" in {row["protocol"] for row in rows}
    assert {"direct", "paper_transfer"}.issubset({row["method"] for row in rows})
    assert {row["source_checkpoint_sha256"] for row in rows} == {
        "7b9e3d0d4c8036c738ef220802e7ee46bc6ab8261cc541fb7d194e8c17044323"
    }


def test_paper_reproduction_removes_redundant_root_derivatives() -> None:
    assert not [name for name in PAPER_REMOVED_DERIVED if (PAPER_STUDY / name).exists()]
    assert (PAPER_STUDY / "split_manifest.csv").is_file()


def test_paper_reproduction_manifest_hashes_only_compact_records() -> None:
    manifest = json.loads((PAPER_STUDY / "artifact_manifest.json").read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    paths = {record["path"] for record in records}
    assert "run_summary.csv" in paths
    assert not [path for path in paths if "runtime" in Path(path).parts]
    assert not paths & PAPER_REMOVED_DERIVED
    for record in records:
        path = PAPER_STUDY / record["path"]
        assert path.is_file(), record["path"]
        assert path.stat().st_size == record["bytes"], record["path"]
        assert _sha256(path) == record["sha256"], record["path"]


def test_paper_reproduction_runner_writes_reproducible_files_under_runtime() -> None:
    runner = (ROOT / "scripts/studies/run_paper_transfer_reproduction_25g_40g.py").read_text(encoding="utf-8")
    assert 'runtime_root(output_dir) / "runs"' in runner
    assert 'predictions.to_csv(runtime_root(output_dir) / "predictions.csv.gz"' in runner
    assert 'data.to_csv(output_dir / f"canonical_{column}_{protocol}.csv"' not in runner
    assert 'split.to_csv(output_dir / "splits"' not in runner
    assert "write_scientific_artifact_manifest(output_dir)" in runner


def test_repository_hygiene_audit_reports_the_cleaned_paper_boundary() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_repository_hygiene.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(result.stdout)
    paper = audit["paper_transfer_protocol"]
    assert paper["tracked_runtime_candidates"] == 0
    assert paper["forbidden_derived_root_entries_present"] == []
    assert paper["artifact_manifest_runtime_entries"] == []
    assert paper["compact_run_summary"]["rows"] >= 20
    assert paper["compact_run_summary"]["missing_required_fields"] == []
    assert audit["git"]["archive"]["contains_paper_tip"] is True


def test_matched_study_keeps_per_fit_runtime_out_of_the_study_record() -> None:
    """Once created, the matched study may retain only compact records in Git."""
    study = ROOT / "studies/transfer/matched_rmse_benchmark"
    assert study.is_dir()
    prefix = "studies/transfer/matched_rmse_benchmark/"
    offenders = [
        path
        for path in _workspace_paths()
        if path.startswith(prefix)
        and "runtime" not in Path(path).parts
        and (
            Path(path).name in RUNTIME_NAMES
            or Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}
        )
    ]
    assert not offenders, offenders


def test_archive_tag_preserves_the_paper_transfer_snapshot() -> None:
    expected = "cf219034b02b9252e5eea10f2d741304efbfb51e"
    tag = subprocess.run(
        ["git", "rev-parse", "archive/pre-matched-rmse-cleanup-2026-09-08^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tag == expected


def test_linear_research_chain_is_ancestral_when_refs_are_present() -> None:
    chain = (
        "study/4g-to-8g-transfer",
        "study/cross-column-transfer-validation",
        "codex/study-transfer-residual-diagnostics",
        "codex/study-scaling-failure-audit",
        "codex/study-source-anchored-shared-transfer",
        "codex/study-physics-column-conditioned-transfer",
        "codex/reproduce-paper-transfer-25g-40g-rmse",
    )
    available = []
    for branch in chain:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            return
        available.append(branch)
    assert len(available) == len(chain)
    for ancestor, descendant in zip(chain, chain[1:]):
        check = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            capture_output=True,
        )
        assert check.returncode == 0, (ancestor, descendant)
        count = subprocess.run(
            ["git", "rev-list", "--count", f"{ancestor}..{descendant}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert count == "1", (ancestor, descendant, count)

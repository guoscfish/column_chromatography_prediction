#!/usr/bin/env python3
"""Read-only repository hygiene audit.

The audit reports repository state; it never deletes, moves, or rewrites an
artifact.  It is intentionally dependency-free so it can run before the
scientific Python environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
CHAIN = (
    "study/4g-to-8g-transfer",
    "study/cross-column-transfer-validation",
    "codex/study-transfer-residual-diagnostics",
    "codex/study-scaling-failure-audit",
    "codex/study-source-anchored-shared-transfer",
    "codex/study-physics-column-conditioned-transfer",
    "codex/reproduce-paper-transfer-25g-40g-rmse",
)
ARCHIVE_TAG = "archive/pre-matched-rmse-cleanup-2026-09-08"
PAPER_STUDY = "studies/transfer/paper_transfer_reproduction"
PAPER_FORBIDDEN_DERIVED_ROOTS = (
    "canonical_25g_legacy_filtered.csv",
    "canonical_40g_legacy_filtered.csv",
    "predictions.csv.gz",
    "splits",
)
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
RUNTIME_BASENAMES = {
    "best.pt",
    "last.pt",
    "history.csv",
    "fit_result.json",
    "result.json",
    "predictions.csv",
    "predictions.csv.gz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def tracked_paths(root: Path) -> list[str]:
    """Return index paths only; ignored runtime is intentionally absent."""
    return git_lines(root, "ls-files", "--cached")


def protected_paths(root: Path) -> set[str]:
    payload = json.loads((root / "docs/PROTECTED_ARTIFACTS.json").read_text(encoding="utf-8"))
    return {item["path"] for item in payload["artifacts"]}


def runtime_candidates(paths: Iterable[str], *, protected: set[str] | None = None) -> list[str]:
    protected = protected or set()
    candidates = []
    for path in paths:
        if path in protected:
            continue
        parts = Path(path).parts
        name = Path(path).name
        under_runtime = "runtime" in parts or "progress" in parts
        known_runtime_name = name in RUNTIME_BASENAMES or Path(path).suffix.lower() in {
            ".pt",
            ".pth",
            ".ckpt",
        }
        # Existing historical `experiments/` anchors are reported separately;
        # this list focuses on study runtime that policy requires to be ignored.
        if path.startswith("studies/") and (under_runtime or known_runtime_name):
            candidates.append(path)
    return sorted(candidates)


def paper_runtime_candidates(paths: Iterable[str]) -> list[str]:
    prefix = "studies/transfer/paper_transfer_reproduction/"
    return sorted(
        path
        for path in paths
        if path.startswith(prefix)
        and (
            "/runs/" in path
            or Path(path).name in RUNTIME_BASENAMES
            or Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}
        )
    )


def protected_audit(root: Path) -> list[dict[str, object]]:
    manifest_path = root / "docs/PROTECTED_ARTIFACTS.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item in payload["artifacts"]:
        path = root / item["path"]
        record: dict[str, object] = {
            "path": item["path"],
            "role": item.get("role"),
            "exists": path.exists(),
            "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
        }
        if path.is_file():
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def ref_exists(root: Path, reference: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{reference}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    ).returncode == 0


def branches_containing(root: Path, reference: str) -> dict[str, list[str]]:
    """Report local and remote containment using ``git branch --contains``."""
    return {
        "local": git_lines(root, "branch", "--format=%(refname:short)", "--contains", reference),
        "remote": git_lines(root, "branch", "--remotes", "--format=%(refname:short)", "--contains", reference),
    }


def branch_audit(root: Path) -> dict[str, object]:
    try:
        current = git_lines(root, "symbolic-ref", "--short", "HEAD")[0]
    except subprocess.CalledProcessError:
        current = "HEAD"
    tips: list[dict[str, object]] = []
    for branch in CHAIN:
        try:
            commit = git_lines(root, "rev-parse", branch)[0]
            parent = git_lines(root, "rev-parse", f"{branch}^")[0]
            record: dict[str, object] = {"branch": branch, "commit": commit, "parent": parent}
            for reference, field in (("main", "commits_only_on_branch_vs_main"), (current, "commits_only_on_branch_vs_current")):
                try:
                    record[field] = int(git_lines(root, "rev-list", "--count", f"{reference}..{branch}")[0])
                except subprocess.CalledProcessError:
                    record[field] = None
            record["contained_by"] = branches_containing(root, branch)
            tips.append(record)
        except subprocess.CalledProcessError:
            tips.append({"branch": branch, "available": False})
    relations = []
    for left, right in zip(CHAIN, CHAIN[1:]):
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", left, right],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            relations.append(
                {
                    "ancestor": left,
                    "descendant": right,
                    "is_ancestor": True,
                    "commits_only_on_ancestor": int(git_lines(root, "rev-list", "--count", f"{right}..{left}")[0]),
                    "commits_added_by_descendant": int(git_lines(root, "rev-list", "--count", f"{left}..{right}")[0]),
                }
            )
        except subprocess.CalledProcessError:
            relations.append({"ancestor": left, "descendant": right, "is_ancestor": False})
    archive: dict[str, object] = {"tag": ARCHIVE_TAG, "available": ref_exists(root, ARCHIVE_TAG)}
    if archive["available"]:
        archive["commit"] = git_lines(root, "rev-parse", f"{ARCHIVE_TAG}^{{commit}}")[0]
        archive["contains_paper_tip"] = subprocess.run(
            ["git", "merge-base", "--is-ancestor", CHAIN[-1], ARCHIVE_TAG],
            cwd=root,
            capture_output=True,
            text=True,
        ).returncode == 0
    return {
        "current_branch": current,
        "tips": tips,
        "adjacent_relations": relations,
        "archive": archive,
    }


def protocol_audit(root: Path) -> dict[str, object]:
    protocol_path = root / "studies/transfer/cross_column/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    schedule_path = root / "studies/transfer/cross_column/splits/schedule_manifest.csv"
    source_path = root / protocol["source_checkpoint"]
    schedule_audit = audit_schedule(schedule_path)
    return {
        "protocol": "studies/transfer/cross_column/protocol.json",
        "target_threshold": protocol.get("target_threshold"),
        "schedule_sha256_recorded": protocol.get("schedule_sha256"),
        "schedule_sha256_actual": sha256_file(schedule_path),
        "source_checkpoint": protocol.get("source_checkpoint"),
        "source_sha256_recorded": protocol.get("source_checkpoint_sha256"),
        "source_sha256_actual": sha256_file(source_path) if source_path.is_file() else None,
        "source_sha256_expected": EXPECTED_SOURCE_SHA256,
        "budgets": protocol.get("planned_budgets"),
        "outer_seeds": protocol.get("outer_seeds"),
        "test_tuning": protocol.get("test_tuning"),
        "neural_selection": protocol.get("neural_selection"),
        "simple_fit": protocol.get("simple_fit"),
        "schedule_ledger": schedule_audit,
    }


def audit_schedule(path: Path) -> dict[str, object]:
    """Summarize budget and role invariants without reading target truth."""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    truth_columns: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        truth_columns = [name for name in (reader.fieldnames or ()) if name in {"V1_ml", "V2_ml"}]
        for row in reader:
            key = (row["column"], row["protocol"], row["outer_seed"], row["planned_budget"])
            groups.setdefault(key, []).append(row)

    violations: list[dict[str, object]] = []
    for key, rows in groups.items():
        ids = [row["sample_id"] for row in rows]
        actual = {int(row["actual_budget"]) for row in rows}
        purchased = sum(row["role"] in {"gradient_train", "validation"} for row in rows)
        roles = {row["role"] for row in rows}
        if len(ids) != len(set(ids)) or actual != {purchased} or roles != {
            "gradient_train",
            "validation",
            "test",
            "pool",
        }:
            violations.append(
                {
                    "context": list(key),
                    "duplicate_ids": len(ids) != len(set(ids)),
                    "actual_budget_values": sorted(actual),
                    "purchased_rows": purchased,
                    "roles": sorted(roles),
                }
            )
    return {
        "context_count": len(groups),
        "violations": violations,
        "truth_columns": truth_columns,
    }


def paper_protocol_audit(root: Path) -> dict[str, object]:
    path = root / "studies/transfer/paper_transfer_reproduction/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    study = root / PAPER_STUDY
    summary_path = study / "run_summary.csv"
    summary: dict[str, object] = {"path": str(summary_path.relative_to(root)), "exists": summary_path.is_file()}
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            rows = list(reader)
        keys = [(row.get("column"), row.get("protocol"), row.get("seed"), row.get("method")) for row in rows]
        summary.update(
            {
                "rows": len(rows),
                "missing_required_fields": sorted(PAPER_RUN_SUMMARY_FIELDS - fields),
                "duplicate_contexts": len(keys) - len(set(keys)),
            }
        )
    retained_manifest = study / "artifact_manifest.json"
    manifest_runtime_entries: list[str] = []
    if retained_manifest.is_file():
        payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
        manifest_runtime_entries = [
            item["path"]
            for item in payload.get("artifacts", [])
            if "runtime" in Path(item["path"]).parts
        ]
    derived_present = [
        name for name in PAPER_FORBIDDEN_DERIVED_ROOTS if (study / name).exists()
    ]
    return {
        "classification": protocol.get("classification"),
        "primary_protocol": protocol.get("primary_protocol"),
        "source_checkpoint": protocol.get("source_checkpoint"),
        "source_checkpoint_sha256": protocol.get("source_checkpoint_sha256"),
        "current_source_sha256": EXPECTED_SOURCE_SHA256,
        "legacy_filtering_present": bool(protocol.get("columns")),
        "tracked_runtime_candidates": len(
            paper_runtime_candidates(tracked_paths(root))
        ),
        "compact_run_summary": summary,
        "forbidden_derived_root_entries_present": derived_present,
        "artifact_manifest_runtime_entries": manifest_runtime_entries,
    }


def build_audit(root: Path = ROOT) -> dict[str, object]:
    paths = tracked_paths(root)
    protected = protected_paths(root)
    raw_candidates = runtime_candidates(paths)
    candidates = [path for path in raw_candidates if path not in protected]
    paper_candidates = paper_runtime_candidates(paths)
    sizes = {path: (root / path).stat().st_size for path in paper_candidates if (root / path).is_file()}
    return {
        "schema_version": 2,
        "repository": str(root),
        "git": branch_audit(root),
        "tracked": {
            "path_count": len(paths),
            "study_runtime_candidates": candidates,
            "study_runtime_candidate_count": len(candidates),
            "protected_runtime_exceptions": sorted(set(raw_candidates) & protected),
            "paper_transfer_runtime_candidates": paper_candidates,
            "paper_transfer_runtime_candidate_count": len(paper_candidates),
            "paper_transfer_runtime_candidate_bytes": sum(sizes.values()),
        },
        "protected_artifacts": protected_audit(root),
        "cross_column_protocol": protocol_audit(root),
        "paper_transfer_protocol": paper_protocol_audit(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args()
    print(json.dumps(build_audit(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

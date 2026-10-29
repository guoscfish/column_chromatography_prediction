#!/usr/bin/env python3
"""Read-only repository hygiene audit.

The audit reports repository state; it never deletes, moves, or rewrites an
artifact.  It is intentionally dependency-free so it can run before the
scientific Python environment is installed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
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
    return [name for name in git_lines(root, "ls-files", "--cached") if (root / name).is_file()]


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


def current_branch_inventory(root: Path) -> list[dict[str, object]]:
    """Report live local refs, including worktree ownership and ancestry."""
    records = []
    for line in git_lines(root, "for-each-ref", "--format=%(refname:short)\t%(objectname)\t%(worktreepath)", "refs/heads/"):
        branch, commit, worktree = line.split("\t", 2)
        records.append({
            "branch": branch,
            "commit": commit,
            "worktree": worktree or None,
            "commits_only_on_branch_vs_main": int(git_lines(root, "rev-list", "--count", f"main..{branch}")[0])
            if ref_exists(root, "main") else None,
            "commits_only_on_branch_vs_current": int(git_lines(root, "rev-list", "--count", f"HEAD..{branch}")[0]),
        })
    return records


def retirement_audit(root: Path) -> dict[str, object]:
    """Verify recovery and evidence without importing scientific code."""
    registry = json.loads((root / "docs/repository/RETIREMENTS.json").read_text(encoding="utf-8"))
    commit = registry["source_commit"]
    records = registry["records"]
    paths = [record["path"] for record in records]
    violations = []
    if len(paths) != len(set(paths)):
        violations.append({"reason": "duplicate_retirement_paths"})
    for record in records:
        path = record["path"]
        if (root / path).exists():
            violations.append({"path": path, "reason": "retired_file_still_present"})
        retained = record["retained_record"]
        if record.get("retained_commit"):
            available = subprocess.run(
                ["git", "cat-file", "-e", f"{record['retained_commit']}:{retained}"],
                cwd=root, capture_output=True,
            ).returncode == 0
        else:
            available = (root / retained).exists()
        if not available:
            violations.append({"path": path, "reason": "retained_record_missing"})
        original = subprocess.run(
            ["git", "show", f"{record.get('source_commit', commit)}:{path}"], cwd=root, capture_output=True,
        )
        if original.returncode or hashlib.sha256(original.stdout).hexdigest() != record["sha256"]:
            violations.append({"path": path, "reason": "recovery_hash_mismatch_or_unavailable"})

    retired_modules = {
        ".".join(Path(path).with_suffix("").parts): path for path in paths if path.endswith(".py")
    }
    candidates = git_lines(root, "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py")
    for name in candidates:
        path = root / name
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=name)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            for module in modules:
                if module in retired_modules:
                    violations.append({"path": name, "reason": "imports_retired_module", "retired": retired_modules[module]})
    return {
        "source_commit": commit,
        "retired_script_count": sum(record["kind"] == "completed_runner" for record in records),
        "consolidated_document_count": sum(record["kind"] == "superseded_document" for record in records),
        "removed_lines": sum(record["lines"] for record in records),
        "violations": violations,
    }


def scope_audit(root: Path) -> dict[str, object]:
    """核验冻结归档、移出文件的 Git 身份及保留代码的导入边界。"""
    scope = json.loads((root / "docs/repository/RESEARCH_SCOPE.json").read_text(encoding="utf-8"))
    archive = scope["archive"]
    violations = []
    actual = git_lines(root, "rev-parse", archive["branch"])[0] if ref_exists(root, archive["branch"]) else None
    if actual != archive["archive_commit"]:
        violations.append({"reason": "archive_branch_changed_or_missing", "actual": actual})
    archived = {}
    if ref_exists(root, archive["archive_commit"]):
        for line in git_lines(root, "ls-tree", "-r", archive["archive_commit"]):
            metadata, path = line.split("\t", 1)
            archived[path] = metadata.split()[2]
    removed = {record["path"] for record in scope["removed_files"]}
    for record in scope["removed_files"]:
        path = record["path"]
        if archived.get(path) != record["git_blob_oid"]:
            violations.append({"path": path, "reason": "archived_blob_missing_or_changed"})
        if (root / path).is_file():
            violations.append({"path": path, "reason": "archived_path_reintroduced"})
    retired_modules = {path[:-3].replace("/", ".") for path in removed if path.endswith(".py")}
    retired_modules |= {m.removesuffix(".__init__") for m in retired_modules if m.endswith(".__init__")}
    for name in tracked_paths(root):
        if not name.endswith(".py"):
            continue
        tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = Path(name).with_suffix("").parts[:-1]
                    base = ".".join(package[:len(package) - node.level + 1] + ((base,) if base else ()))
                modules = [base] + [f"{base}.{alias.name}" for alias in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("qgeognn_al."):
                    module = "src." + module
                if module in retired_modules:
                    violations.append({"path": name, "reason": "imports_archived_module", "module": module})
    return {"branch": archive["branch"], "archive_commit": actual,
            "removed_file_count": len(removed), "merged_document_count": len(scope["document_merges"]),
            "markdown_before": scope["markdown_before"], "violations": violations}


def build_audit(root: Path = ROOT) -> dict[str, object]:
    paths = tracked_paths(root)
    protected = protected_paths(root)
    raw_candidates = runtime_candidates(paths)
    candidates = [path for path in raw_candidates if path not in protected]
    return {
        "schema_version": 4,
        "repository": str(root),
        "git": {"local_branches": current_branch_inventory(root)},
        "research_scope": scope_audit(root),
        "retirements": retirement_audit(root),
        "tracked": {"path_count": len(paths), "study_runtime_candidates": candidates,
                    "study_runtime_candidate_count": len(candidates),
                    "protected_runtime_exceptions": sorted(set(raw_candidates) & protected)},
        "protected_artifacts": protected_audit(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args()
    audit = build_audit(args.root.resolve())
    print(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False))
    if (audit["research_scope"]["violations"] or audit["retirements"]["violations"]
            or any(not record["exists"] for record in audit["protected_artifacts"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

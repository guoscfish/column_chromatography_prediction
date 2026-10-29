"""Small, deterministic guards for repository and experiment boundaries."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tracked() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


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


def test_protected_artifact_manifest_entries_exist() -> None:
    payload = json.loads((ROOT / "docs/PROTECTED_ARTIFACTS.json").read_text(encoding="utf-8"))
    entries = payload["artifacts"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert entries
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert not missing, missing


def test_study_runtime_directories_are_gitignored_and_untracked() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "studies/**/runtime/" in ignore_text
    assert "studies/**/progress/" in ignore_text
    tracked = _tracked()
    assert not [
        path for path in tracked
        if path.startswith("studies/") and ("/runtime/" in path or "/progress/" in path)
    ]


def test_transfer_archive_and_main_import_boundary() -> None:
    from scripts.audit_repository_hygiene import scope_audit
    audit = scope_audit(ROOT)
    assert audit["removed_file_count"] > 0
    assert audit["violations"] == []


def test_immutable_markdown_still_matches_original_commit() -> None:
    scope = json.loads((ROOT / "docs/repository/RESEARCH_SCOPE.json").read_text())
    names = scope["immutable_markdown"]
    assert names
    for name in names:
        original = subprocess.check_output(["git", "show", scope["archive"]["base_commit"] + ":" + name], cwd=ROOT)
        assert (ROOT / name).read_bytes() == original, name


def test_free_llm_and_active_learning_files_remain_versioned() -> None:
    scope = json.loads((ROOT / "docs/repository/RESEARCH_SCOPE.json").read_text())
    base = scope["archive"]["base_commit"]
    roots = ["src/qgeognn_al", "studies/active_learning", "reports", "tests/active_learning_v2",
             "tests/active_learning_v3", "tests/active_learning_v3_2"]
    original = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", base, "--", *roots], cwd=ROOT).decode().splitlines())
    moved_docs = {record["path"] for record in scope["document_merges"]}
    archived_transfer = {record["path"] for record in scope["removed_files"] if record["path"].startswith(("studies/active_learning/transfer/", "reports/a1_a2_audit_20261013/"))}
    expected = original - moved_docs - archived_transfer
    current = set(_tracked())
    assert expected <= current, sorted(expected - current)
    assert all((ROOT / name).is_file() for name in expected)

"""Experiment artifact retention and finalization helpers."""

from __future__ import annotations

import hashlib
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

COMPACT_REQUIRED = ("README.md", "config.json", "environment.json", "decision.json")
RUNTIME_NAMES = {"history.csv", "fit_result.json"}
RUNTIME_DIR_NAMES = {"runtime", "progress", "checkpoints"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_experiment(runtime_dir: Path, experiment_dir: Path, keep: Iterable[str]) -> list[Path]:
    """Copy an explicit compact record from gitignored runtime into an experiment.

    The allowlist is mandatory: finalization never sweeps a runtime tree and
    therefore cannot accidentally commit checkpoints or per-fit traces.
    """
    runtime_dir, experiment_dir = Path(runtime_dir), Path(experiment_dir)
    requested = tuple(dict.fromkeys(str(item) for item in keep))
    missing_required = sorted(set(COMPACT_REQUIRED) - set(requested))
    if missing_required:
        raise ValueError(f"compact record omits required files: {missing_required}")
    copied: list[Path] = []
    for relative in requested:
        source = runtime_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.name in RUNTIME_NAMES or source.suffix in {".pt", ".pth", ".ckpt"} or any(part in RUNTIME_DIR_NAMES for part in Path(relative).parts):
            raise ValueError(f"runtime artifact cannot be finalized: {relative}")
        destination = experiment_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def write_inventory(root: Path, destination: Path) -> dict:
    """Write a deterministic tracked-file inventory for retention audits."""
    import subprocess

    root, destination = Path(root).resolve(), Path(destination)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout.split(b"\0")
    paths = [root / value.decode() for value in tracked if value]
    experiment_files = [path for path in paths if path.relative_to(root).parts[0] == "experiments"]
    checkpoints = [path for path in paths if path.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    def total(items: list[Path]) -> int:
        return sum(path.stat().st_size for path in items)
    payload = {
        "tracked_files_total": len(paths),
        "experiments_tracked_files": len(experiment_files),
        "experiments_total_bytes": total(experiment_files),
        "tracked_checkpoints": {"count": len(checkpoints), "bytes": total(checkpoints)},
        "history_csv_count": sum(path.name == "history.csv" for path in paths),
        "fit_result_json_count": sum(path.name == "fit_result.json" for path in paths),
        "partial_csv_count": sum(path.name.endswith(".partial.csv") for path in paths),
        "experiment_directories": sorted({path.relative_to(root).parts[1] for path in experiment_files if len(path.relative_to(root).parts) > 2}),
        "python_files_in_scripts": sum(path.suffix == ".py" and path.relative_to(root).parts[0] == "scripts" for path in paths),
        "python_files_in_src_qgeognn_al": sum(path.suffix == ".py" and path.relative_to(root).parts[:2] == ("src", "qgeognn_al") for path in paths),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_prune_manifest(root: Path, destination: Path, protected_file: Path) -> list[dict[str, str]]:
    """Classify tracked runtime candidates after a conservative path-reference scan."""
    import subprocess

    root, destination, protected_file = Path(root).resolve(), Path(destination), Path(protected_file)
    protected_payload = json.loads(Path(protected_file).read_text(encoding="utf-8"))
    protected = {item["path"] for item in protected_payload["artifacts"]}
    raw = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    relative_paths = [value.decode() for value in raw.split(b"\0") if value]
    python_text = "\n".join((root / path).read_text(encoding="utf-8", errors="ignore") for path in relative_paths if path.endswith(".py"))
    rows: list[dict[str, str]] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or not relative.startswith("experiments/"):
            continue
        is_checkpoint = path.suffix.lower() in {".pt", ".pth", ".ckpt"}
        is_runtime = path.name in RUNTIME_NAMES or path.name.endswith(".partial.csv") or path.name.startswith("state_round_") or path.name in {"state_after_fit.json", "state_after_query.json"} or any(part in {"runtime", "progress"} for part in path.parts)
        if not (is_checkpoint or is_runtime or relative in protected):
            continue
        referenced = relative in python_text
        if relative in protected:
            category, action, reason = "PROTECTED_ANCHOR", "KEEP", "registered source/cache/data dependency"
        elif referenced:
            category, action, reason = "REPRODUCIBLE_RUNTIME", "REVIEW", "literal current-code reference requires manual review"
        else:
            category, action, reason = "REPRODUCIBLE_RUNTIME", "DELETE", "aggregate record retained; recoverable from Git history"
        branch = "Track_A" if relative.startswith("experiments/e2_") else "Track_B" if relative.startswith(("experiments/e4_", "experiments/d4")) else "shared"
        rows.append({"path": relative, "size_bytes": str(path.stat().st_size), "category": category, "referenced_by_current_code": str(referenced).lower(), "replacement_or_summary": "compact experiment aggregates and decision record" if action == "DELETE" else "docs/PROTECTED_ARTIFACTS.json", "reason": reason, "action": action, "scientific_branch": branch})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "category", "referenced_by_current_code", "replacement_or_summary", "reason", "action", "scientific_branch"])
        writer.writeheader(); writer.writerows(rows)
    return rows

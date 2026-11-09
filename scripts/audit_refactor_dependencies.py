#!/usr/bin/env python3
"""Conservative, reproducible source-only dependency and retirement inventory.

Reads committed Python/shell/Markdown and an allowlist of metadata JSON blobs.
Never imports scientific modules or opens tables, checkpoints or request payloads.
Static absence of a reference is never alone sufficient to authorize retirement.
"""
from __future__ import annotations
import argparse
import ast
import collections
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

FIELDS = "path lines purpose importers subprocess_callers dynamic_references tests frozen_hash_dependents live_runtime_dependents authoritative_outputs planned_action replacement evidence decision_reason risk".split()
JSON_META = re.compile(r"(?:^|/)(?:[^/]*(?:protocol|manifest|config|freeze|contract)[^/]*|RESEARCH_SCOPE|PROTECTED_ARTIFACTS)\.json$", re.I)
PAYLOAD = re.compile(r"(?:^|/)(?:requests?|responses?|raw|turns?|attempts?|streams?)(?:/|_)|(?:request|response|catalogue|catalog|feedback|prediction|label|selection)(?:\b|_)", re.I)
OUTPUT_METHODS = {"savefig", "to_csv", "to_json", "write_text", "write_bytes", "save", "dump"}
DYNAMIC_MARKERS = ("importlib", "sys.path", "subprocess", "getattr(", "__import__", "pickle.load", "torch.load", "spec_from_file_location")


def git(repo, *args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo), *args])


def tree_entries(repo, commit):
    records = {}
    for record in git(repo, "ls-tree", "-rz", "--long", commit).split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        if kind == b"blob":
            records[name.decode()] = {"oid": oid.decode(), "bytes": int(size)}
    return records


def safe_source(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".sh", ".bash", ".zsh", ".md"}:
        return True
    return bool(suffix == ".json" and JSON_META.search(path) and not PAYLOAD.search(path))


def read_blobs(repo, entries, names):
    """One batch process; path allowlist is checked before requesting any blob."""
    proc = subprocess.Popen(["git", "--no-optional-locks", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    results = {}
    try:
        for name in names:
            if not safe_source(name):
                raise ValueError(f"not source/metadata: {name}")
            proc.stdin.write((entries[name]["oid"] + "\n").encode()); proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise RuntimeError(f"unavailable blob: {name}")
            payload = proc.stdout.read(int(header[2])); proc.stdout.read(1)
            results[name] = payload.decode("utf-8", errors="replace")
    finally:
        proc.stdin.close(); proc.wait()
    return results


def module_aliases(path):
    dotted = path[:-3].replace("/", ".")
    if dotted.endswith(".__init__"):
        dotted = dotted[:-9]
    aliases = {dotted}
    if dotted.startswith("src."):
        aliases.add(dotted[4:])
    if path.startswith("application/"):
        aliases.add(Path(path).stem)
    return aliases


def resolve_imports(path, tree, module_map):
    result = set()
    dotted = path[:-3].replace("/", ".")
    package = dotted[:-9] if dotted.endswith(".__init__") else dotted.rpartition(".")[0]
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                except (ImportError, ValueError):
                    continue
            else:
                base = node.module or ""
            names = [base] + [base + "." + alias.name for alias in node.names if alias.name != "*"]
        for name in names:
            result.update(module_map.get(name, ()))
    result.discard(path)
    return result


def purpose(path, tree):
    doc = ast.get_docstring(tree) if tree else None
    if doc:
        return " ".join(doc.split())[:240]
    names = [n.name for n in getattr(tree, "body", []) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
    if Path(path).name.startswith("test_"):
        return "Regression tests: " + ", ".join(names[:5])
    if path.endswith("/__init__.py"):
        return "Package namespace / exports: " + ", ".join(names[:5])
    return "Module definitions: " + ", ".join(names[:7])


def runtime_sources(runtime):
    if not runtime:
        return {}, False
    root = Path(runtime)
    mapping = {}
    for name in ("active_source_hashes.json", "RUNTIME_SNAPSHOT.json", "runtime_frozen_index.json"):
        file = root / name
        if file.exists():
            mapping[name] = json.loads(file.read_text())
    return mapping, bool(mapping)


def source_outputs(tree):
    result = set()
    if not tree:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in OUTPUT_METHODS:
            literals = [v.value for v in ast.walk(node) if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            result.update(literals[:4] or [f"L{node.lineno}:computed-output:{node.func.attr}"])
    return sorted(result)


def inspect(repo, commit, workspace=None, runtime=None):
    entries = tree_entries(repo, commit)
    names = sorted(name for name in entries if safe_source(name))
    source = read_blobs(repo, entries, names)
    baseline_source = dict(source)
    baseline_python = sorted(name for name in entries if name.endswith(".py"))
    workspace = Path(workspace) if workspace else None
    if workspace:
        for candidate in workspace.rglob("*.py"):
            rel = candidate.relative_to(workspace)
            if not any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                source[rel.as_posix()] = candidate.read_text()
    python = sorted(name for name in source if name.endswith(".py"))
    module_map = collections.defaultdict(set)
    by_basename = collections.defaultdict(set)
    for path in python:
        for alias in module_aliases(path):
            module_map[alias].add(path)
        by_basename[Path(path).name].add(path)
    parsed, parse_errors, imports = {}, {}, {}
    importers = collections.defaultdict(set)
    dynamic_features = {}
    for path in python:
        try:
            parsed[path] = ast.parse(source[path], filename=path)
        except SyntaxError as exc:
            parsed[path] = None; parse_errors[path] = str(exc)
        imports[path] = resolve_imports(path, parsed[path], module_map) if parsed[path] else set()
        for dependency in imports[path]:
            importers[dependency].add(path)
        dynamic_features[path] = [marker for marker in DYNAMIC_MARKERS if marker in source[path]]
    references = collections.defaultdict(set)
    subprocess_refs = collections.defaultdict(set)
    frozen_refs = collections.defaultdict(set)
    # Exact full paths, dotted module names and quoted/composed Python basenames.
    # Basenames deliberately overapproximate split Path()/importlib constructions.
    for caller, text in source.items():
        basenames = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*\.py", text))
        targets = set()
        for basename in basenames:
            targets.update(by_basename.get(basename, ()))
        module_tokens = set(re.findall(r"\b(?:src\.)?qgeognn_al(?:\.[A-Za-z_][A-Za-z_0-9]*)+|\bscripts(?:\.[A-Za-z_][A-Za-z_0-9]*)+", text))
        for token in module_tokens:
            while token:
                targets.update(module_map.get(token, ()))
                token = token.rpartition(".")[0]
        subprocess_literals = set()
        for node in ast.walk(parsed[caller]) if parsed.get(caller) else ():
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                subprocess_literals.update(n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str))
        for target in targets - {caller}:
            references[target].add(caller)
            direct_command = any(target in literal or Path(target).name in literal for literal in subprocess_literals)
            if direct_command or Path(caller).suffix in {".sh", ".bash", ".zsh"}:
                subprocess_refs[target].add(caller)
            if caller in entries and any(marker in text for marker in ("code_hash", "frozen_source", "frozen_llm_compatibility", "sha256_file", "source_hash", "source_sha256")):
                frozen_refs[target].add(caller)
    scope = json.loads(source["docs/repository/RESEARCH_SCOPE.json"])
    frozen_scope = set(scope.get("frozen_llm_compatibility_paths", []))
    protected = json.loads(source["docs/PROTECTED_ARTIFACTS.json"])["artifacts"]
    blanket = "src/qgeognn_al/active_learning_v2/benchmark_protocol.py:code_hashes(rglob src/qgeognn_al; glob application)"
    for path in python:
        if path in entries and path.startswith(("src/qgeognn_al/", "application/")):
            frozen_refs[path].add(blanket)
        if path in frozen_scope:
            frozen_refs[path].add("docs/repository/RESEARCH_SCOPE.json:frozen_llm_compatibility_paths")
        if "/frozen_source" in path or "/frozen/" in path:
            frozen_refs[path].add("original frozen snapshot path; immutable identity")
    runtime_data, runtime_seen = runtime_sources(runtime)
    runtime_index = runtime_data.get("runtime_frozen_index.json", {})
    live = collections.defaultdict(set)
    for path, consumers in runtime_index.get("live_runtime_dependents", {}).items():
        live[path].update(json.dumps(value, sort_keys=True) for value in consumers)
    for path in runtime_index.get("cw_frozen_code_paths", []):
        frozen_refs[path].add("CW protocol frozen_code_hashes: runtime_frozen_index.json")
    for path in runtime_index.get("original_source_snapshot_paths", []):
        frozen_refs[path].add("CW original_source_snapshot: runtime_frozen_index.json")
    # Active CW source dependencies are conservatively closed over AST imports.
    live_roots = {p for p in python if p.startswith("scripts/studies/") and Path(p).name.startswith(("run_free_cw_", "recover_cw_", "free_cw_parallel_controller_guard"))}
    live_roots.update(p for p in python if p in runtime_index.get("live_runtime_dependents", {}))
    reached = set(live_roots)
    queue = list(live_roots)
    while queue:
        current = queue.pop()
        for dependency in imports.get(current, ()):
            live[dependency].add(current)
            if dependency not in reached:
                reached.add(dependency); queue.append(dependency)
    for p in live_roots:
        live[p].add("CW protected controller / runtime evidence (may be latent recovery dependency)")
    # History metadata records source commit and content identity only, no deleted data reads.
    history = git(repo, "log", "-1", "--format=%H %s", commit).decode().strip()
    rows = []
    workspace = Path(workspace) if workspace else None
    for path in python:
        outputs = source_outputs(parsed[path])
        if live[path]:
            risk, reason = "LIVE_BLOCKER", "CW controller or transitive import/recovery dependency; preserve path and bytes"
        elif frozen_refs[path]:
            risk, reason = "FROZEN_HASH_BLOCKER", "Frozen path/hash or conservative hash-reference dependency; preserve until protocol exit"
        elif importers[path]:
            risk, reason = "IMPORT_BLOCKER", "Existing callers require dependency migration and output regression before retirement"
        else:
            risk, reason = "UNKNOWN", "No direct import does not prove retirement safe; dynamic calls, historical contracts and unique outputs need review"
        evidence = {"source_commit": commit if path in entries else "UNCOMMITTED_CANDIDATE", "git_blob_oid": entries.get(path, {}).get("oid"), "sha256": hashlib.sha256(baseline_source.get(path, source[path]).encode()).hexdigest(), "candidate_sha256": hashlib.sha256(source[path].encode()).hexdigest(), "parse_error": parse_errors.get(path), "dynamic_capabilities": dynamic_features[path]}
        row = {"path": path, "lines": len(source[path].splitlines()), "purpose": purpose(path, parsed[path]),
            "importers": sorted(importers[path]), "subprocess_callers": sorted(subprocess_refs[path]),
            "dynamic_references": sorted(references[path] - importers[path]),
            "tests": sorted(p for p in importers[path] | references[path] if p.startswith("tests/")),
            "frozen_hash_dependents": sorted(frozen_refs[path]), "live_runtime_dependents": sorted(live[path]),
            "authoritative_outputs": outputs, "planned_action": "DEFERRED" if risk != "IMPORT_BLOCKER" else "KEEP",
            "replacement": "", "evidence": evidence,
            "decision_reason": reason, "risk": risk}
        if path not in entries:
            row["planned_action"] = "ADD_CANDIDATE_ONLY"
            row["decision_reason"] = "New isolated API/audit; not part of original frozen protocol"
            if row["tests"] or path.startswith("tests/"):
                row["risk"] = "SAFE"
        if path.startswith("tests/"):
            row["planned_action"] = "KEEP" if path in entries else "ADD_CANDIDATE_ONLY"; row["decision_reason"] = "Retain regression contract; do not run real-data fixtures during isolated cleanup"
        if path == "src/qgeognn_al/active_learning_v2/lcmd.py":
            row["replacement"] = "src/qgeognn_al/acquisitions/lcmd.py"
            row["planned_action"] = "KEEP_FROZEN_COMPATIBILITY"
            row["decision_reason"] = "Future pure implementation migrated; historical copy remains byte-frozen until CW completion, contract migration and strict equivalence approval"
        rows.append(row)
    # Uncommitted/ignored live Python is inventoried as path/hash metadata only.
    # It is never parsed, imported or copied into the candidate workspace.
    live_root = runtime_data.get("RUNTIME_SNAPSHOT.json", {}).get("active_root")
    live_hashes = runtime_data.get("active_source_hashes.json", {}).get("hashes", {})
    extra_runtime = {path: digest for path, digest in live_hashes.items() if path.endswith(".py") and path not in entries}
    if live_root:
        for encoded in git(live_root, "ls-files", "--others", "--exclude-standard", "-z", "--", "*.py").split(b"\0"):
            if not encoded: continue
            name = encoded.decode()
            file = Path(live_root) / name
            if file.is_file():
                extra_runtime[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    for path, digest in sorted(extra_runtime.items()):
        rows.append({"path": path, "lines": "UNKNOWN_METADATA_ONLY", "purpose": "Uncommitted live-source metadata only; body not parsed/copied/imported", "importers": [], "subprocess_callers": [], "dynamic_references": [], "tests": [], "frozen_hash_dependents": [], "live_runtime_dependents": sorted(live[path]), "authoritative_outputs": [], "planned_action": "KEEP_RUNTIME_OUTSIDE_CANDIDATE", "replacement": "", "evidence": {"source_commit": "UNCOMMITTED_RUNTIME", "sha256": digest, "metadata_only": True}, "decision_reason": "Live/uncommitted research source cannot be migrated or retired by this task", "risk": "LIVE_BLOCKER" if path in live or path.startswith("studies/active_learning/free_cw_") else "UNKNOWN"})
    retirement_manifest = workspace / "docs/repository/RETIREMENT_MANIFEST.json" if workspace else None
    retired = json.loads(retirement_manifest.read_text()).get("retired", []) if retirement_manifest and retirement_manifest.exists() else []
    retirement_validations = []
    for record in retired:
        for row in rows:
            if row["path"] != record["path"]:
                continue
            proof = record.get("evidence", {})
            reviewed = proof.get("reviewed_nonruntime_references", {})
            equivalence_file = workspace / proof.get("equivalence_report", "MISSING")
            equivalent = False
            if equivalence_file.is_file():
                comparison = json.loads(equivalence_file.read_text()).get("results", {})
                equivalent = bool(comparison) and all(value.get("old_equals_new") is True for value in comparison.values())
            replacement_path = workspace / record.get("replacement", "MISSING").split(" ")[0]
            gate = retirement_gate(row, replacement_verified=replacement_path.is_file() and equivalent,
                live_reviewed=runtime_seen, output_reproduced=equivalent,
                reviewed_nonruntime_references=reviewed)
            if record.get("source_commit") != commit or record.get("git_blob_oid") != entries[row["path"]]["oid"] or record.get("sha256") != row["evidence"]["sha256"]:
                gate["blockers"].append("retirement_original_identity_mismatch")
            if any(row["path"] == item["path"] or row["path"].startswith(item["path"].rstrip("/")+"/") for item in protected):
                gate["blockers"].append("protected_artifact")
            gate["safe"] = not gate["blockers"]
            retirement_validations.append({"path":row["path"], "gate":gate})
            if gate["safe"]:
                row["planned_action"] = "RETIRED"
                row["replacement"] = record["replacement"]
                row["decision_reason"] = "Isolated candidate retirement only; current dependency graph, replacement regression and source identities revalidated against RETIREMENT_MANIFEST.json"
                row["evidence"]["retirement_manifest"] = "docs/repository/RETIREMENT_MANIFEST.json"
                row["risk"] = "SAFE"
            else:
                row["planned_action"] = "DEFERRED"
                row["decision_reason"] = "Retirement proof no longer passes current dependency graph: " + ",".join(gate["blockers"])
    report = {"schema_version": 1, "source_commit": commit, "source_head_description": history,
        "baseline_python_files": len(baseline_python), "inventory_python_files": len(rows), "new_candidate_python_files": len(set(python) - set(baseline_python)), "uncommitted_runtime_python_files": len(extra_runtime), "scanned_text_blobs": len(source),
        "metadata_allowlist": JSON_META.pattern, "excluded_payload_pattern": PAYLOAD.pattern, "parse_errors": parse_errors,
        "frozen_compatibility_paths": sorted(frozen_scope), "protected_paths": protected,
        "runtime_snapshot_available": runtime_seen, "risk_counts": dict(collections.Counter(r["risk"] for r in rows)),
        "limits": ["Source-only static audit, not proof of arbitrary runtime/reflection/pickle reachability", "Manifest references are an allowlist; raw provider payloads, tables and checkpoint contents never parsed", "Code-hash blanket protects every original src/qgeognn_al and application Python file", "Subprocess/dynamic edges conservatively overapproximate textual mentions", "Python under ignored live directories is not imported or copied; inspect runtime snapshot metadata separately", "Git history availability/content identity is checked; historical execution is not replayed"],
        "retirement_decision_validations": retirement_validations,
        "gate3": "STOP: do not merge, push or delete active main files"}
    return rows, report


def retirement_gate(row, *, replacement_verified=False, live_reviewed=False, output_reproduced=False, reviewed_nonruntime_references=()):
    blockers = []
    for field in ("importers", "subprocess_callers", "frozen_hash_dependents", "live_runtime_dependents"):
        values = set(row.get(field, ()))
        if field == "subprocess_callers":
            values -= set(reviewed_nonruntime_references)
        if values: blockers.append(field)
    unresolved = set(row.get("dynamic_references", ())) - set(reviewed_nonruntime_references)
    if unresolved: blockers.append("dynamic_references")
    if not live_reviewed: blockers.append("live_process_review_missing")
    if not replacement_verified: blockers.append("replacement_equivalence_missing")
    if row.get("authoritative_outputs") and not output_reproduced: blockers.append("unique_output_reproduction_missing")
    evidence = row.get("evidence", {})
    if not evidence.get("git_blob_oid") or not evidence.get("sha256"): blockers.append("recoverable_baseline_identity_missing")
    return {"safe": not blockers, "blockers": blockers}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True) if isinstance(row.get(key), (dict,list,tuple)) else row.get(key, "") for key in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--commit", required=True)
    ap.add_argument("--workspace", "--overlay-root", dest="workspace", type=Path); ap.add_argument("--runtime", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    commit = git(args.repo, "rev-parse", args.commit + "^{commit}").decode().strip()
    rows, report = inspect(args.repo, commit, args.workspace, args.runtime)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "CODE_INVENTORY.csv", FIELDS, sorted(rows,key=lambda r:r["path"]))
    candidates = []
    for row in rows:
        if row["evidence"].get("source_commit") != commit:
            continue
        if row["path"].startswith("scripts/") or row["path"] in {"src/qgeognn_al/active_learning_v2/phase1.py", "src/qgeognn_al/active_learning_v2/benchmark_reporting.py"} or row["path"].startswith(("src/qgeognn_al/transfer/", "src/qgeognn_al/t1")):
            if row["planned_action"] == "RETIRED":
                gate = {"safe":True, "proof":"RETIREMENT_MANIFEST.json:dependency_gate + synthetic byte-equivalence"}
            else:
                gate = retirement_gate(row)
            candidates.append({"path":row["path"], "status":"RETIRED" if row["planned_action"] == "RETIRED" else "DEFERRED", "risk":row["risk"], "replacement":row["replacement"], "gate":gate, "evidence":row["evidence"], "decision_reason":row["decision_reason"]})
    write_csv(args.output / "RETIREMENT_CANDIDATES.csv", ["path","status","risk","replacement","gate","evidence","decision_reason"], candidates)
    (args.output / "DEPENDENCY_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    manifest = args.output / "RETIREMENT_MANIFEST.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"schema_version":1,"source_commit":commit,"scope":"isolated candidate only; never active main", "retired":[], "policy":"Every deletion requires dependency gate, reproducible replacement, recoverable commit/blob/SHA256 and live/frozen review"},indent=2)+"\n")
    print(json.dumps({"baseline_python_files":report["baseline_python_files"], "inventory_python_files":len(rows), "risk_counts":report["risk_counts"], "parse_errors":len(report["parse_errors"])}))


if __name__ == "__main__":
    main()

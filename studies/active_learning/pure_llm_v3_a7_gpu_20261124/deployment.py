#!/usr/bin/env python3
"""Build and verify an isolated A7 GPU release, or probe an existing platform read-only."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile


BASE = "b295939a2f6ec0bd90ed14e5c9a9b92aae0909d7"
STUDIES = "studies/active_learning"
FORMAL = f"{STUDIES}/pure_llm_v3_a7_gpu_20261124"
PUBLISHED = f"{STUDIES}/pure_llm_v3_20261124"
SPLIT = f"{STUDIES}/qgeognn_v2_row_sequential_b32"
SEEDS = (157, 887, 2357, 6101, 12203)
PROFILE = "execution_gpt62_ultra_a7.json"
PROFILE_V3 = "execution_gpt62_a7_v3.json"
PROFILE_V4 = "execution_gpt62_astra_xhigh_a7_v4.json"
PROFILE_V4_PILOT = "execution_gpt62_astra_xhigh_a7_v4_pilot.json"
RELEASE_MANIFEST = "release_manifest.json"
RELEASE_SCHEMA = "purellm-v3-a7-gpu-release-v2"
REVIEWED_TRANSPORT = f"{PUBLISHED}/pure_llm_v3/transport_manager.py"
REVIEWED_TRANSPORT_SHA256 = "578d8a3c68ee73aa4b815fe65a0be5aadca3310627173932f5d8c9dbea21d2f4"
REMOTE_ROOT = Path("/1/column_chromatography_prediction")
MAX_RELEASE_BYTES = 32 * 1024 * 1024

ASSETS = {
    "application/QGeoGNN.py", "application/utils.py", "dataset/dataset_4g.csv",
    "experiments/e0_4g_baseline/canonical_4g.csv",
    "experiments/e0_4g_baseline/data_metadata.json",
    "experiments/e0_4g_baseline/graph_cache_4g.pt",
    "experiments/e0_4g_baseline/scaler.json",
    f"{STUDIES}/qgeognn_v2_row_small_batch_benchmark/protocol.json",
    f"{SPLIT}/protocol.json", f"{SPLIT}/splits/split_manifest.json",
    *(f"{SPLIT}/splits/row_seed_{seed}.csv" for seed in SEEDS),
    *(f"{PUBLISHED}/{name}" for name in (
        "PROMPT_PURE_LLM_V3.md", "PROMPT_LEGACY_REFERENCE.md", "PROMPT_V3_LEGACY_SCHEMA_COMPATIBLE.md",
    )),
}
FORMAL_FILES = {
    f"{FORMAL}/{PROFILE}", f"{FORMAL}/{PROFILE_V3}", f"{FORMAL}/{PROFILE_V4}",
    f"{FORMAL}/{PROFILE_V4_PILOT}", f"{FORMAL}/deployment.py",
    f"{FORMAL}/DEPLOYMENT.md", f"{FORMAL}/README.md",
    f"{FORMAL}/OFFLINE_TEST_REPORT.md", f"{FORMAL}/DISPATCHER.md",
    f"{FORMAL}/ASTRA_XHIGH_BASELINE.md",
    f"{FORMAL}/requirements_provider.txt", f"{FORMAL}/key_pool.example.json",
    f"{FORMAL}/formal_authorization.template.json",
}
REQUIRED_FORMAL = {
    f"{FORMAL}/{PROFILE}", f"{FORMAL}/deployment.py",
    *(f"{FORMAL}/a7_gpu/{name}.py" for name in ("backend", "bridge_endpoint", "bridge_protocol", "credentials", "cuda_runtime", "dispatcher",
          "formal_contract", "key_pool", "metrics", "provider", "scheduler", "seed_runner", "ssh_courier", "supersession")),
}
V4_PROFILES = {f"{FORMAL}/{name}" for name in (PROFILE_V4, PROFILE_V4_PILOT)}
V4_ENCODER = f"{FORMAL}/a7_gpu/compact_payload.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE).stdout


def baseline_name(name):
    path = PurePosixPath(name)
    return (name in ASSETS or
            path.suffix == ".py" and
            name.startswith(("src/qgeognn_al/", f"{PUBLISHED}/pure_llm_v3/")))


def formal_name(name):
    path = PurePosixPath(name)
    return name in FORMAL_FILES or (
        path.parent.as_posix() == f"{FORMAL}/a7_gpu" and path.suffix == ".py"
    )


def source_origin(name):
    if name == REVIEWED_TRANSPORT:
        return "reviewed_transport_overlay"
    return "formal_overlay" if formal_name(name) else "published_git"


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and path.as_posix() == name
                and all(part not in (".", "..") and not part.startswith(".") for part in path.parts)
                and (name == RELEASE_MANIFEST or baseline_name(name) or formal_name(name)))


def committed_inputs(repo):
    files = git(repo, "ls-tree", "-rz", BASE).split(b"\0")
    discovered = set()
    for entry in filter(None, files):
        header, encoded = entry.split(b"\t")
        name = encoded.decode("utf-8")
        if baseline_name(name):
            mode, kind, _ = header.split()
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise ValueError(f"source must be a Git regular file: {name}")
            discovered.add(name)
    if missing := ASSETS - discovered:
        raise ValueError(f"committed release assets missing: {sorted(missing)}")
    return {name: git(repo, "show", f"{BASE}:{name}") for name in sorted(discovered)}


def formal_inputs(repo):
    directory = repo / FORMAL
    paths = [*directory.glob("a7_gpu/*.py"),
             *(repo / name for name in FORMAL_FILES if (repo / name).is_file())]
    files = {}
    for path in sorted(paths):
        name = path.relative_to(repo).as_posix()
        if not safe_name(name) or not formal_name(name) or not path.is_file() or path.resolve() != path:
            raise ValueError(f"invalid formal overlay source: {path}")
        files[name] = path.read_bytes()
    if missing := REQUIRED_FORMAL - files.keys():
        raise ValueError(f"the formal overlay is incomplete: {sorted(missing)}")
    if V4_PROFILES & files.keys() and V4_ENCODER not in files:
        raise ValueError("the Astra/xhigh release is missing its compact encoder")
    return files


def check_reviewed_transport(raw):
    if sha(raw) != REVIEWED_TRANSPORT_SHA256:
        raise ValueError("the reviewed transport overlay differs from its pinned SHA256")


def reviewed_transport(repo):
    path = repo / REVIEWED_TRANSPORT
    if not path.is_file() or path.resolve() != path:
        raise ValueError("the reviewed transport overlay must be a regular worktree file")
    payload = path.read_bytes()
    check_reviewed_transport(payload)
    return payload


def add_member(archive, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(payload))


def build(repo, output):
    repo = Path(repo).resolve()
    output = Path(output)
    files = {**committed_inputs(repo), **formal_inputs(repo), REVIEWED_TRANSPORT: reviewed_transport(repo)}
    manifest = {
        "schema": RELEASE_SCHEMA,
        "published_git_commit": BASE,
        "files": {name: {"bytes": len(content), "sha256": sha(content),
                         "origin": source_origin(name)}
                  for name, content in sorted(files.items())},
    }
    with output.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, content in sorted(files.items()):
                add_member(archive, name, content)
            add_member(archive, RELEASE_MANIFEST, json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return {"archive": str(output), "sha256": sha(output.read_bytes()), "file_count": len(files),
            "published_git_commit": BASE}


def read_archive(path):
    files = {}
    total = 0
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            total += member.size
            if (not member.isfile() or not safe_name(member.name) or member.name in files
                    or min(member.size, total) < 0 or total > MAX_RELEASE_BYTES):
                raise ValueError(f"invalid archive entry: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive file content missing: {member.name}")
            with source:
                files[member.name] = source.read()
    if RELEASE_MANIFEST not in files:
        raise ValueError("release manifest is missing")
    manifest = json.loads(files[RELEASE_MANIFEST])
    records = manifest["files"]
    if (manifest["schema"] != RELEASE_SCHEMA or manifest["published_git_commit"] != BASE
            or set(files) != records.keys() | {RELEASE_MANIFEST}
            or not (ASSETS | REQUIRED_FORMAL | {REVIEWED_TRANSPORT}) <= records.keys()):
        raise ValueError("archive manifest identity or coverage differs")
    if V4_PROFILES & files.keys() and V4_ENCODER not in files:
        raise ValueError("the Astra/xhigh release is missing its compact encoder")
    check_reviewed_transport(files[REVIEWED_TRANSPORT])
    for name, record in records.items():
        payload = files[name]
        if (record["bytes"], record["sha256"], record["origin"]) != (
            len(payload), sha(payload), source_origin(name)
        ):
            raise ValueError(f"release file SHA or provenance mismatch: {name}")
    return files


def install(path, dest, expected):
    path, dest = Path(path), Path(dest).absolute()
    dest = dest.parent.resolve(strict=True) / dest.name
    if sha(path.read_bytes()) != expected:
        raise ValueError("uploaded archive SHA does not match local release receipt")
    if not dest.parent.is_dir() or os.path.lexists(dest):
        raise ValueError("fresh destination under an existing real directory required")
    files = read_archive(path)
    # Claim the destination without replacing an existing release or symlink.
    dest.mkdir(mode=0o700)
    for name, payload in files.items():
        if name == RELEASE_MANIFEST:
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    # A partial install from an I/O interruption does not get a completion manifest.
    (dest / RELEASE_MANIFEST).write_bytes(files[RELEASE_MANIFEST])
    return {"destination": str(dest), "files": len(files) - 1, "sha256": expected}


def command_result(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": type(error).__name__}
    return {"status": result.returncode, "output": result.stdout.splitlines()}


def read_optional(path):
    return json.loads(path.read_bytes()) if path.is_file() else {}


def probe(root, key_pool_path):
    root = Path(root)
    study = root / "formal_cuda/row4g_b32_20261124"
    work = root / "worktree"
    queue = read_optional(study / "queue.json")
    source = read_optional(study / "source_receipt.json")
    source = source or read_optional(work / ".remote-compute-inputs.json")
    tasks = queue.get("tasks", [])
    packages = {}
    for name in ("openai", "httpx", "torch", "torch-geometric", "numpy", "pandas", "rdkit", "scipy", "scikit-learn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    key_path = Path(key_pool_path).expanduser()
    try:
        information = key_path.stat(follow_symlinks=False)
        pool_present = stat.S_ISREG(information.st_mode)
        pool_owner_only = pool_present and information.st_uid == os.getuid() and not (information.st_mode & 0o077)
    except FileNotFoundError:
        pool_present = pool_owner_only = False
    anchor_paths = {}
    for seed in SEEDS:
        run = study / "trajectories" / f"random-{seed}"
        round_zero = run / "rounds/random/round_00"
        anchor_paths[str(seed)] = {"run": str(run), "has_anchor_files": all(path.is_file() for path in (
            run / "run_contract.json", round_zero / "contract.json", round_zero / "model/fit_audit.json",
            round_zero / "model/best.pt", round_zero / "model/predictions.csv.gz",
        ))}
    return {
        "checked": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
        "packages": packages, "cuda_runtime": command_result(sys.executable, "-B", "-c",
            "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"),
        "workspace": str(work), "workspace_present": work.is_dir(),
        "workspace_git": command_result("git", "-C", str(work), "rev-parse", "HEAD"),
        "source_receipt_git": source.get("git_head"), "queue_git": queue.get("git_head"),
        "numeric_cuda_entry_exists": (work / "scripts/studies/run_qgeognn_cuda_three_methods.py").is_file(),
        "gpu": command_result("nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                              "--format=csv,noheader"),
        "gpu_processes": command_result("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                        "--format=csv,noheader"),
        "mount": command_result("findmnt", "-T", str(root)), "free_disk": command_result("df", "-h", str(root)),
        "queue_tasks_by_state": dict(Counter(f"{task['method']}:{task['state']}" for task in tasks
                                             if task.get("qualification") is None)),
        "running_tasks": [task["name"] for task in tasks if task.get("state") == "running"],
        "gpu_lease_root": str(study), "persistent_anchor_files": anchor_paths,
        "key_environment_present": "OPENAI_API_KEY" in os.environ,
        "key_pool_config_present": pool_present, "key_pool_config_owner_only": pool_owner_only,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    make = actions.add_parser("build")
    make.add_argument("--repo", type=Path, default=Path.cwd())
    make.add_argument("--output", type=Path, required=True)
    put = actions.add_parser("install")
    put.add_argument("--archive", type=Path, required=True)
    put.add_argument("--destination", type=Path, required=True)
    put.add_argument("--sha256", required=True)
    show = actions.add_parser("probe")
    show.add_argument("--platform-root", type=Path, default=REMOTE_ROOT)
    show.add_argument("--key-pool-config", type=Path, default=Path("~/.config/purellm/key_pool.json"))
    args = parser.parse_args()
    if args.action == "build":
        value = build(args.repo, args.output)
    elif args.action == "install":
        value = install(args.archive, args.destination, args.sha256)
    else:
        value = probe(args.platform_root, args.key_pool_config)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

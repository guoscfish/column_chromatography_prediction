#!/usr/bin/env python3
"""Activate each authorized A7 v6 dispatcher when its GPU continuation is ready."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import plistlib
import re
import shlex
import stat
import subprocess
from tempfile import NamedTemporaryFile
import time


ROOT = PurePosixPath("/1/column_chromatography_prediction")
A7 = ROOT / "purellm_a7"
RUN_PREFIX = "a7-gpt62-astra-xhigh-ssh-v6-prompt-full"
SEEDS = (157, 887, 2357)
PILOT = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
MODULE = "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_pilot"
SHA = re.compile(r"[a-f0-9]{64}\Z")
HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
FIELDS = {"seed", "run", "release", "contract_sha256", "authorization_sha256", "completed_rounds", "checkpoint_sha256"}


def valid_sha(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def approved_release(value):
    path = PurePosixPath(value)
    if str(path) != value or path.parent != A7 / "releases" or not valid_sha(path.name.removeprefix("full-")):
        raise ValueError("remote release must be an exact, SHA-addressed GPFS release directory")
    return str(path)


def run_path(seed):
    return A7 / "runs" / f"{RUN_PREFIX}-{seed}"


def read(path):
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("expected a JSON object")
    return value


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def valid_binding(binding, seed, release):
    return (type(binding) is dict and set(binding) == FIELDS and type(binding["seed"]) is int
            and binding["seed"] == seed and binding["run"] == str(run_path(seed))
            and binding["release"] == release and type(binding["completed_rounds"]) is int
            and binding["completed_rounds"] == 10 and all(valid_sha(binding[field]) for field in (
                "contract_sha256", "authorization_sha256", "checkpoint_sha256"))
            and binding["contract_sha256"] != PILOT)


def ready_bindings(release):
    """Run inside the SSH remote. Return only verified, unpaused activation bindings."""
    records = []
    for seed in SEEDS:
        run = Path(run_path(seed))
        directory = run / f"seed_{seed}"
        if any(path.exists() for path in (run / "pause.request", directory / "pause.request",
                                         directory / "round_36/transition.json")):
            continue
        try:
            binding = read(run / "operator/dispatch_binding.json")
            if not valid_binding(binding, seed, release):
                continue
            contract, approval = read(run / "formal_contract.json"), read(run / "authorization.json")
            profile = contract["profile"]
            if (contract["run_directory"] != str(run) or contract["api_execution"] != "mac_ssh_dispatch_v1"
                    or profile["confirmation_seeds"] != [seed] or profile["target_labels"] != 1485
                    or profile["resume_boundaries"][str(seed)]["round"] != 10
                    or contract["predecessor"]["checkpoint_sha256"] != binding["checkpoint_sha256"]
                    or digest(contract) != binding["contract_sha256"]
                    or approval["scope"] != "pure_llm_v3_a7_formal"
                    or approval["contract_sha256"] != binding["contract_sha256"]
                    or digest(approval) != binding["authorization_sha256"]
                    or any(approval[field] is not True for field in (
                        "authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training"))):
                continue
        except (OSError, KeyError, TypeError, ValueError):
            continue
        records.append(binding)
    return records


def ssh_poll(host, release, source):
    command = shlex.join([str(ROOT / "environment/venv/bin/python"), "-B", "-",
                          "--remote-poll", "--remote-release", release])
    reply = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                            "-o", "ConnectTimeout=8", "--", host, command], input=source,
                           text=True, capture_output=True, check=True, timeout=20)
    records = json.loads(reply.stdout)
    if type(records) is not list:
        raise ValueError("unexpected GPFS activation inventory")
    seeds, contracts, verified = set(), set(), []
    for binding in records:
        seed = binding.get("seed") if type(binding) is dict else None
        if seed not in SEEDS or not valid_binding(binding, seed, release):
            continue
        contract = binding["contract_sha256"]
        if seed in seeds or contract in contracts:
            raise ValueError("a GPFS activation identity was listed more than once")
        seeds.add(seed)
        contracts.add(contract)
        verified.append(binding)
    return verified


def private_dir(path, *, create=False):
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise PermissionError("dispatcher spool must be private and owned by the Mac operator")
    return path


def put_plist(path, value):
    payload = plistlib.dumps(value)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077
                        or path.read_bytes() != payload):
                    raise PermissionError("saved launch agent differs from its approved dispatcher configuration") from None
        finally:
            temporary.unlink(missing_ok=True)


def launchctl(*args):
    return subprocess.run(["/bin/launchctl", *args], text=True, capture_output=True, timeout=12)


def activate(args, binding, shared):
    seed, contract = binding["seed"], binding["contract_sha256"]
    spool = private_dir(shared.parent / contract, create=True)
    if spool == shared:
        raise PermissionError("the shared pilot pool cannot own another run's request journal")
    local_binding = spool / "run_binding.json"
    if local_binding.exists():
        saved = read(local_binding)
        if (saved.get("contract_sha256") != contract or saved.get("mock") is not False
                or saved.get("authorization_sha256") != binding["authorization_sha256"]):
            raise PermissionError("existing Mac spool differs from the remote authorization")
    label = f"com.purellm.a7-v6-full-{seed}"
    service = f"{args.domain}/{label}"
    plist = spool / "dispatcher.launch-agent.runtime.plist"
    put_plist(plist, {"Label": label, "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False},
                     "ProcessType": "Background", "ThrottleInterval": 30,
                     "WorkingDirectory": str(args.local_release),
                     "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
                     "StandardOutPath": str(spool / "dispatcher.log"),
                     "StandardErrorPath": str(spool / "dispatcher.stderr.log"),
                     "ProgramArguments": ["/usr/bin/caffeinate", "-i", str(args.python), "-B", "-u", "-m", MODULE,
                                          "--ssh-host", args.ssh_host, "--remote-root", args.remote_release,
                                          "--remote-run-dir", binding["run"], "--contract-sha256", contract,
                                          "--authorization-sha256", binding["authorization_sha256"],
                                          "--pilot-spool", str(shared), "--spool-dir", str(spool), "--seed", str(seed)]})
    if launchctl("print", service).returncode == 0:
        return "already_loaded"
    clear = launchctl("bootstrap", args.domain, str(plist)).returncode == 0
    if not clear and launchctl("print", service).returncode != 0:
        raise RuntimeError("launchctl could not load this seed's dispatcher")
    return "activated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-release", type=Path)
    parser.add_argument("--remote-release", required=True, type=approved_release)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--poll", type=float, default=12)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--remote-poll", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_poll:
        print(json.dumps(ready_bindings(args.remote_release)))
        return 0
    if args.local_release is None or args.python is None or not HOST.fullmatch(args.ssh_host) or not 3 <= args.poll <= 300:
        parser.error("local release, Python, SSH alias and a 3–300 second poll interval are required")
    os.umask(0o077)
    args.local_release = args.local_release.expanduser().resolve(strict=True)
    args.python = args.python.expanduser().absolute()
    if not os.access(args.python, os.X_OK) or not (args.local_release / (MODULE.replace(".", "/") + ".py")).is_file():
        parser.error("the local release or specified interpreter is unavailable")
    args.domain = f"gui/{os.getuid()}"
    shared = private_dir(Path.home() / ".config/purellm/spool" / PILOT)
    source = Path(__file__).read_text()
    previous = None
    while True:
        states = {str(seed): "awaiting_ready_binding" for seed in SEEDS}
        try:
            bindings = ssh_poll(args.ssh_host, args.remote_release, source)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            states["ssh"] = type(error).__name__
        else:
            for binding in bindings:
                seed = str(binding["seed"])
                try:
                    states[seed] = activate(args, binding, shared)
                except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
                    states[seed] = type(error).__name__
        if states != previous:
            print(json.dumps(states, sort_keys=True), flush=True)
            previous = states
        if args.once:
            return int("ssh" in states or any(state.endswith("Error") for state in states.values()))
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())

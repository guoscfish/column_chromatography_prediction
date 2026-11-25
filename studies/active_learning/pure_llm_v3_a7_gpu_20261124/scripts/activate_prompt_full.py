"""Independently transfer sealed A7 pilot trajectories to their authorized L1485 run."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler
from . import run_prompt_full as full
from . import run_prompt_pilot as pilot

PACKAGE = "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts"
WATCHER = Path("studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/watch_transport_replay.py")
TRANSPORT = Path("studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py")


def process_alive(path, tag):
    try:
        pid = int(path.read_text())
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
        return state != "Z" and os.fsencode(tag) in argv
    except (OSError, ValueError, IndexError):
        return False


def start(run, name, cwd, command):
    with (run / "operator" / f"{name}.log").open("ab") as report:
        proc = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=report,
                                stderr=subprocess.STDOUT, start_new_session=True)
    scheduler._atomic_status(run / "operator" / f"{name}.record.json", {"pid": proc.pid, "cmd": command[3]})
    (run / "operator" / f"{name}.pid").write_text(f"{proc.pid}\n")
    return proc


def activate(args, seed):
    runs = args.old_run.parent
    parent = runs / f"{pilot.RUN_PREFIX}-{seed}"
    run = runs / f"{full.RUN_PREFIX}-{seed}"
    boundary = parent / f"seed_{seed}" / "round_10/transition.json"
    if not boundary.is_file():
        return "waiting_for_L653"
    if (parent / "pause.request").is_file() or (run / "pause.request").is_file():
        return "paused"
    operator = run / "operator"
    operator.mkdir(parents=True, exist_ok=True, mode=0o700)
    binding_path = operator / "dispatch_binding.json"
    pid_path = operator / "coordinator.pid"
    full_module = f"{PACKAGE}.run_prompt_full"
    replay_module = f"{PACKAGE}.watch_transport_replay"
    if not binding_path.is_file():
        contract, (state, _, rd) = full.construct(args.release, args.old_release, args.old_run,
            args.first_release, args.first_run, args.parent_release, parent, run, seed)
        approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": contract.digest,
            "authorized_by": "pilot operator", "authorization_reference": (
                "2026-11-25 operator extended the current three improved-prompt trajectories to L1485 "
                "and authorized bounded replays with possible duplicate billing."),
            "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True}
        once(operator / "authorization.json", approval)
        binding = {"seed": seed, "run": str(run), "release": str(args.release), "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "completed_rounds": rd, "checkpoint_sha256": state.checkpoint}
        once(binding_path, binding)
    else:
        binding = read(binding_path)
    calls = ["--seed", str(seed), "--old-release", str(args.old_release), "--old-run", str(args.old_run),
             "--first-release", str(args.first_release), "--first-run", str(args.first_run), "--parent-release",
             str(args.parent_release), "--parent-run", str(parent), "--run-dir", str(run),
             "--authorization", str(operator / "authorization.json")]
    launch_spec = {"module": full_module, "options": calls}
    once(operator / "launch_spec.json", launch_spec)
    if not (run / "coordinator_generation.json").is_file():
        if not process_alive(pid_path, full_module):
            if (operator / "coordinator.record.json").exists():
                return "coordinator_start_failed"
            start(run, "coordinator", args.release, [sys.executable, "-B", "-m", full_module, "run", *calls])
        return "starting_full_coordinator"
    if not (run / f"seed_{seed}/round_10/boundary_memory.json").is_file():
        return "verifying_full_boundary" if process_alive(pid_path, full_module) else "coordinator_start_failed"
    if not (operator / "replay-watch.record.json").is_file():
        policy = {"schema": "a7_v5_bounded_transport_replay_v1", "authorization_reference": (
            "2026-11-25 operator approved continuation to L1485 and possible duplicate billing for bounded retries."),
            "runs": {str(run): binding["contract_sha256"]}, "additional_physical_attempts_per_logical": 1,
            "accept_possible_duplicate_billing": True, "provider_lookup": "unavailable_on_audited_store_false_route",
            "lookup_observations": {"157": "original v5 response GET HTTP 404", "2357": "original v5 response GET HTTP 404"},
            "launcher_sha256": file_digest(args.overlay / WATCHER),
            "transport_replacement_sha256": file_digest(args.overlay / TRANSPORT)}
        policy_file = operator / "rolling_policy.json"
        once(policy_file, policy)
        options = ["--run", str(run), "--base-release", str(args.release), "--overlay", str(args.overlay),
            "--cache", str(args.old_run / "public_graphs" / f"seed_{seed}"), "--policy", str(policy_file),
            "--policy-sha", digest(policy), "--seed", str(seed)]
        once(operator / "replay_launch_spec.json", {"module": replay_module, "options": options})
        start(run, "replay-watch", args.overlay, [sys.executable, "-B", "-m", replay_module, *options])
    return "full_phase_active"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-release", "old-run", "first-release", "first-run", "parent-release", "release", "overlay", "state-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name, path in vars(args).items():
        setattr(args, name, path.resolve())
    args.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    with scheduler._lock(args.state_dir / "handoff.lock", busy="L1485 handoff watcher is already active"):
        previous = {}
        while True:
            for seed in pilot.PHASE_BOUNDARIES:
                try:
                    status = activate(args, seed)
                except (OSError, ValueError, KeyError, PermissionError) as error:
                    status = f"{type(error).__name__}: {error}"
                if previous.get(seed) != status:
                    record = {"seed": seed, "status": status, "time": time.time(), "pid": os.getpid()}
                    scheduler._atomic_status(args.state_dir / f"seed_{seed}.json", record)
                    print(dumps(record), flush=True)
                    previous[seed] = status
            if all(item == "full_phase_active" for item in previous.values()):
                return
            time.sleep(10)


if __name__ == "__main__":
    main()

"""Qualify, run, monitor and resume independent A7/M2 trajectories on CUDA."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from functools import partial
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import ProviderReply, TransportStop

from .formal_contract import LEASE_ROOT, PROFILE, ROOT, SEEDS, load_contract, verify_authorization
from .key_pool import KeyPool, load_pool
from .provider import call_responses, sdk_status, verify_reported_model


def _atomic_status(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(dumps(value, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _lock(path, *, busy):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TransportStop(busy) from None
        yield


def _pause_requested(run, seed=None):
    return (run / "pause.request").exists() or (
        seed is not None and (run / f"seed_{seed}" / "pause.request").exists()
    )


def _gpu_lease_path(lease_root, gpu):
    directory = Path(lease_root)
    if not directory.is_dir():
        raise FileNotFoundError("the shared numeric GPU lock directory is missing")
    return directory / f"physical-gpu-{gpu}.lock"


@contextmanager
def gpu_slot(run, seed, gpu, lease_root, *, wait=True):
    """Share the existing numerical coordinator's physical GPU flock."""
    path = _gpu_lease_path(lease_root, gpu)
    status = run / "gpu" / f"seed_{seed}.json"
    _atomic_status(status, {"seed": seed, "gpu": gpu, "stage": "waiting", "pid": os.getpid()})
    try:
        with path.open("a") as lock:
            while True:
                if _pause_requested(run, seed):
                    raise TransportStop("paused before GPU lease")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not wait:
                        raise TransportStop("shared physical GPU currently busy") from None
                    time.sleep(2)
            _atomic_status(status, {"seed": seed, "gpu": gpu, "stage": "cuda", "pid": os.getpid()})
            yield
    finally:
        _atomic_status(status, {"seed": seed, "gpu": gpu, "stage": "idle", "pid": os.getpid()})


def _query_gpu_status():
    commands = {
        "devices": ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"],
        "compute_processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                              "--format=csv,noheader,nounits"],
    }
    result = {}
    for key, command in commands.items():
        try:
            proc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=6)
            result[key] = proc.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            result[key] = None
    return result


def _seed_worker(root, run, contract, approval, pool_config, seed, gpu, stop, generation):
    """Spawn one seed process before importing Torch so CUDA binding is process-local."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .seed_runner import run_seed

    payload = run / "workers" / f"seed_{seed}.json"
    try:
        with _lock(run / "workers" / f"seed_{seed}.lock", busy="seed already has an active worker"):
            if read(run / "coordinator_generation.json")["id"] != generation:
                raise TransportStop("seed worker belongs to a superseded coordinator")
            from .backend import TrainingBackend

            _atomic_status(payload, {"status": "initializing", "seed": seed, "pid": os.getpid(), "gpu": gpu})
            backend = TrainingBackend(root, run / "public_graphs" / f"seed_{seed}", seed)
            pool = KeyPool(run / "api", pool_config)
            anchor = Path(contract.anchor_root) / f"random-{seed}" if contract.anchor_root else None
            result = run_seed(
                root=root, run_directory=run, contract=contract, authorization=approval, seed=seed,
                backend=backend, anchor_run=anchor, max_rounds=stop,
                call=partial(call_responses, model_lock=run / "provider_model_lock.json"),
                fit_slot=partial(gpu_slot, run, seed, gpu, contract.lease_root),
                transport_options={"attempt_slot": partial(pool.attempt, pause=partial(_pause_requested, run, seed))},
            )
            _atomic_status(payload, {**asdict(result), "gpu": gpu, "pid": os.getpid()})
    except Exception as error:
        reason = {"error_type": type(error).__name__}
        if isinstance(error, TransportStop):
            reason["reason"] = str(error)
        _atomic_status(payload, {"status": "paused" if _pause_requested(run, seed) else "failed",
                                 "seed": seed, "gpu": gpu, "pid": os.getpid(), **reason})
        raise SystemExit(1) from None


def _start(ctx, root, run, contract, approval, pool, seed, gpu, stop, generation):
    process = ctx.Process(target=_seed_worker,
        args=(root, run, contract, approval, pool, seed, gpu, stop, generation), name=f"a7-seed-{seed}")
    process.start()
    return process


def _assert_no_old_workers(run):
    for seed in SEEDS:
        with _lock(run / "workers" / f"seed_{seed}.lock", busy="an earlier seed worker is still alive"):
            pass


def _run(root, run, contract, approval, pool, *, resume):
    from .seed_runner import qualification_status

    if not contract.gpu_slots:
        raise ValueError("set the authorized physical CUDA slot indices with --gpus")
    for gpu in contract.gpu_slots:
        _gpu_lease_path(contract.lease_root, gpu)
    verify_authorization(contract, approval)
    if not sdk_status(contract.transport)["matches"]:
        raise RuntimeError("installed OpenAI SDK does not match the authorized profile")
    ctx = multiprocessing.get_context("spawn")
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _lock(run / "coordinator.lock", busy="an A7 coordinator is already active"):
        _assert_no_old_workers(run)
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        if _pause_requested(run) and not resume:
            raise TransportStop("formal experiment is paused; use resume under its existing authorization")
        generation = uuid4().hex
        KeyPool(run / "api", pool).prepare()
        once(run / "events" / f"{generation}.json", {
            "action": "resume" if resume else "run", "authorization_sha256": digest(approval),
            "timestamp_unix": time.time(), "pool": pool.public(),
        })
        _atomic_status(run / "coordinator_generation.json", {"id": generation, "pid": os.getpid()})
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
        if qualification_status(run, contract, approval) is None:
            seed = contract.qualification_seed
            test = _start(ctx, root, run, contract, approval, pool, seed, contract.gpu_slots[0], 1, generation)
            test.join()
            if test.exitcode or _pause_requested(run) or qualification_status(run, contract, approval) is None:
                print(dumps({"status": "qualification_paused_or_failed", "seed": seed,
                             "worker_exit": test.exitcode}))
                return 1
        if not (run / "provider_model_lock.json").is_file():
            raise TransportStop("qualification is missing the provider model identity lock")
        workers = {}
        for index, seed in enumerate(contract.seed_ids):
            if _pause_requested(run):
                break
            gpu = contract.gpu_slots[index % len(contract.gpu_slots)]
            workers[seed] = _start(ctx, root, run, contract, approval, pool, seed, gpu, None, generation)
        for proc in workers.values():
            proc.join()
        ends = {str(seed): proc.exitcode for seed, proc in workers.items()}
        states = {}
        for seed in workers:
            saved = run / "workers" / f"seed_{seed}.json"
            states[str(seed)] = read(saved)["status"] if saved.is_file() else "unknown"
        if _pause_requested(run) or "paused" in states.values():
            outcome = "paused"
        elif len(workers) == len(contract.seed_ids) and all(
                ends[key] == 0 and states[key] == "complete" for key in ends):
            outcome = "complete"
        else:
            outcome = "incomplete"
        print(dumps({"status": outcome, "worker_exit": ends, "worker_status": states}))
        return int(outcome != "complete")


def _preflight(root, run, contract, pool, *, synthetic):
    import numpy as np
    from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import stage_state
    from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import make_request
    if not synthetic and len(contract.gpu_slots) > 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(contract.gpu_slots[0])
    from .backend import TrainingBackend
    from .cuda_runtime import qualified_packages

    status = {"sdk": sdk_status(contract.transport), "provider_qualified": False, "mode": (
        "synthetic_predictions_only" if synthetic else "shared_cuda_L333_inference"
    ), "credential_files": pool.status() if pool else None, "seeds": {}}
    if not synthetic:
        qualified_packages(root)
        if not contract.gpu_slots or not contract.anchor_root:
            raise ValueError("real no-fit preflight needs --gpus and --anchor-root")
    for seed in contract.seed_ids:
        backend = TrainingBackend(root, run / "public_graphs" / f"seed_{seed}", seed,
            **({} if synthetic else {"gpu_slot": partial(gpu_slot, run, seed, contract.gpu_slots[0],
                                                       contract.lease_root, wait=False)}))
        if synthetic:
            values = np.tile([1., 2., 3., 4., 5., 6.], (len(backend.u0_ids), 1))
            state = backend.state_from_predictions(values, "a" * 64)
        else:
            anchor = Path(contract.anchor_root) / f"random-{seed}"
            state = backend.initial_state(anchor, run / "preflight" / f"seed_{seed}", allow_fresh_fit=False)
        stage = stage_state(root, state, arm="A7", memory_mode="M2")
        memory = stage.packet["memory"]
        counts = (len(stage.context.observed), len(stage.context.candidates))
        if (counts != (333, 2997) or state.history or memory.get("latest_feedback")
                or memory.get("hypotheses") or memory.get("archived_hypotheses")
                or "historical_top_abs_errors" in memory):
            raise ValueError("initial A7 state contains foreign decisions or incorrect roles")
        if any(not {"flow", "purity_raw", "CAS", "name"} <= row["conditions"].keys()
               for row in stage.context.candidates.values()):
            raise ValueError("initial metadata is incomplete")
        request = make_request(stage.binding, 0, stage.messages, contract.transport)
        status["seeds"][str(seed)] = {"L": counts[0], "U": counts[1], "validation": 416, "test": 417,
            "checkpoint": state.checkpoint, "request_id": request["request_id"],
            "payload_bytes": len(dumps(request["payload"]).encode()), "empty_scientific_memory": True}
    _atomic_status(run / ("preflight_synthetic.json" if synthetic else "preflight_cuda.json"), status)
    print(dumps(status))


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "key-status", "preflight", "run", "status",
                                           "pause", "resume", "reconcile"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--asset-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--gpu-lease-root", type=Path, default=LEASE_ROOT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--key-pool-config", type=Path)
    parser.add_argument("--single-key", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="offline preflight only")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--round", type=int)
    parser.add_argument("--logical-index", type=int)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--recovered-response", type=Path)
    return parser.parse_args()


def main():
    from .metrics import snapshot

    args = _parse()
    if args.synthetic and args.action != "preflight":
        raise ValueError("synthetic data are permitted only in preflight")
    if args.action == "key-status":
        try:
            result = load_pool(args.key_pool_config, single=args.single_key, require_usable=False)
        except FileNotFoundError:
            print(dumps({"pool_config_present": False, "credentials": {}, "content_checked": False}))
        else:
            print(dumps({"pool_config_present": not args.single_key, "pool": result.public(),
                         "credentials": result.status(), "content_checked": False}))
        return
    if args.run_dir is None:
        raise ValueError("--run-dir is required")
    run = args.run_dir.resolve()
    if args.action == "status":
        report = snapshot(run)
        pool = KeyPool(run / "api").status()
        api = {row["seed"]: row["state"] for row in pool.get("queue", [])}
        gpu_jobs = [read(path) for path in sorted((run / "gpu").glob("seed_*.json"))]
        gpu = {row["seed"]: row["stage"] for row in gpu_jobs if row["stage"] != "idle"}
        for row in report["seeds"]:
            seed, worker = row["seed"], row["worker"] or {}
            if worker.get("status") in ("failed", "paused", "complete"):
                stage = worker["status"]
            elif seed in gpu:
                stage = "gpu_" + gpu[seed]
            elif seed in api:
                stage = "api_" + api[seed]
            elif seed != 157 and not report["launch_qualification_recorded"]:
                stage = "awaiting_first_round_qualification"
            else:
                stage = worker.get("status", "not_started")
            row["stage"] = stage
        print(dumps({**report, "api_pool": pool, "gpu_jobs": gpu_jobs, "gpu": _query_gpu_status()}))
        return
    if args.action == "pause":
        once(run / "pause.request", {"mode": "stop_before_next_operation"})
        return
    root = args.asset_root.resolve()
    if root != ROOT.resolve():
        raise ValueError("launch the module from the exact asset/release checkout being authorized")
    if run.is_relative_to(root):
        raise ValueError("run output must be outside the isolated source checkout")
    gpus = tuple(int(index) for index in args.gpus.split(",")) if args.gpus else ()
    contract = load_contract(root, args.config or root / PROFILE, anchor_root=args.anchor_root,
                             gpu_slots=gpus, lease_root=args.gpu_lease_root, run_directory=run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.action == "preflight":
        try:
            pool = load_pool(args.key_pool_config, single=args.single_key, require_usable=False)
        except FileNotFoundError:
            pool = None
        _preflight(root, run, contract, pool, synthetic=args.synthetic)
        return
    if args.authorization is None:
        raise PermissionError("formal action requires --authorization for the exact contract")
    approval = verify_authorization(contract, read(args.authorization))
    if args.action == "reconcile":
        from .seed_runner import reconcile_attempt

        if any(value is None for value in (args.seed, args.round, args.logical_index, args.attempt, args.reconciliation)):
            raise ValueError("reconcile requires seed, round, logical index, attempt and operator evidence")
        with _lock(run / "coordinator.lock", busy="pause coordinator before reconciliation"):
            _assert_no_old_workers(run)
            declared = None
            reply = None
            if args.recovered_response:
                recovered = read(args.recovered_response)
                declared = {key: recovered.pop(key, None) for key in ("model_version", "reported_reasoning_effort")}
                reply = ProviderReply(**recovered)
                lock = verify_reported_model(contract.transport, reply, declared)
                if (run / "provider_model_lock.json").is_file() and read(run / "provider_model_lock.json") != lock:
                    raise TransportStop("recovered provider identity differs from the formal model lock")
            reconcile_attempt(run_directory=run, contract=contract, authorization=approval, seed=args.seed,
                round_index=args.round, logical_index=args.logical_index, attempt=args.attempt,
                reconciliation=read(args.reconciliation), reply=reply)
            if reply:
                once(run / "provider_model_lock.json", lock)
        return
    if args.single_key:
        raise PermissionError("the formal five-seed launcher requires a configured dynamic key pool")
    pool = load_pool(args.key_pool_config)
    raise SystemExit(_run(root, run, contract, approval, pool, resume=args.action == "resume"))


if __name__ == "__main__":
    main()

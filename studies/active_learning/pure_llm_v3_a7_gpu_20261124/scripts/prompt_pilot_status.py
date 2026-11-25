#!/usr/bin/env python3
"""Show A7 v6 and predecessor progress, GPUs and the shared Mac Key Pool."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time


ROOT = Path("/1/column_chromatography_prediction")
RUNS = ROOT / "purellm_a7/runs"
CONTINUATION = "a7-gpt62-astra-xhigh-ssh-v5-prompt-continuation"
FULL = "a7-gpt62-astra-xhigh-ssh-v6-prompt-full"
DIRECT = "a7-gpt62-astra-xhigh-ssh-v8-direct-one"
DIRECT_887 = "a7-gpt62-astra-xhigh-ssh-v9-direct-one"
PILOT_ROUNDS = 10
TARGET_ROUNDS = 36
TARGET_LABELS = 1485
ORIGINAL = "a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10"
FIRST_887 = "a7-gpt62-astra-xhigh-ssh-v5-prompt-pilot-887"
PILOT = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
SEEDS = (157, 887, 2357)
KEYS = ("key_01", "key_02", "key_03")
SHA = re.compile(r"[a-f0-9]{64}\Z")
IDENTITY = re.compile(r"[a-f0-9]{32}\Z")
FIT = re.compile(r"fit_[0-9]{3}\Z")
ROUND = re.compile(r"round_([0-9]{2})\Z")
CODE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.]{0,79}\Z")
PROVIDER_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
PHASES = {"request_dispatched", "http_phase", "http_headers", "sse_event",
          "transport_completed", "transport_failure"}
TOKEN_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def read_json(path):
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def checksum(path):
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def safe_code(value):
    return value if isinstance(value, str) and CODE.fullmatch(value) else None


def pid(value):
    return value if type(value) is int and value > 1 else None


def live(proc, *, coordinator=None, run=None):
    if proc is None:
        return False
    try:
        state = Path(f"/proc/{proc}/stat").read_text().rsplit(")", 1)[1].split()
        if state[0] == "Z" or coordinator is not None and int(state[1]) != coordinator:
            return False
        if run is None:
            return True
        argv = Path(f"/proc/{proc}/cmdline").read_bytes().split(b"\x00")
        return os.fsencode(run) in argv
    except (OSError, IndexError, ValueError):
        return False


def latest_round(directory):
    for path in sorted(directory.glob("round_[0-9][0-9]"), reverse=True):
        transition = read_json(path / "transition.json")
        fit = transition.get("fit_path")
        checkpoint = transition.get("checkpoint_sha256", "")
        if (not isinstance(fit, str) or not FIT.fullmatch(fit)
                or not isinstance(checkpoint, str) or not SHA.fullmatch(checkpoint)):
            continue
        evidence = {"source_sha256": path / "source.json",
                    "validation_metrics_sha256": path / fit / "validation_metrics.json"}
        if all(transition.get(field) and checksum(file) == transition[field] for field, file in evidence.items()):
            number = int(ROUND.fullmatch(path.name)[1])
            report = read_json(evidence["validation_metrics_sha256"])
            return number, {"checkpoint_sha256": checkpoint,
                            "validation_nrmse": report.get("combined_normalized_rmse")}
    if (directory / "initial/initial.json").is_file():
        return 0, {}
    return None, {}


def pending_attempt(run, directory, index):
    if index is None:
        return {}
    for path in sorted((directory / f"round_{index:02}").glob("turn_[0-9][0-9]/attempt_*.started.json"), reverse=True):
        started = read_json(path)
        identity = started.get("attempt_id", "")
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            continue
        stem = path.name.removesuffix(".started.json")
        record = read_json(path.with_name(f"{stem}.recovery.terminal.json")) or read_json(path.with_name(f"{stem}.terminal.json"))
        returned = (run / "bridge/results" / f"{identity}.json").is_file()
        status = safe_code(record.get("status")) or ("returned_from_mac" if returned else "waiting_for_mac")
        return {"status": status, "failure": safe_code(record.get("error")), "ambiguous": record.get("ambiguous") is True,
                "round": index, "turn": int(path.parent.name[-2:]), "attempt": started.get("attempt"),
                "attempt_id": identity, "returned_from_mac": returned}
    return {}


def fallback_contract(run):
    record = read_json(run / "formal_contract.json")
    if record.get("run_directory") != str(run):
        return None
    actual = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                       allow_nan=False).encode()).hexdigest()
    approval = read_json(run / "authorization.json")
    return actual if approval.get("contract_sha256") == actual else None


def dispatch_contract(run, seed):
    binding = read_json(run / "operator/dispatch_binding.json")
    sha = binding.get("contract_sha256")
    if (binding.get("seed") == seed and binding.get("run") in (str(run), run.name)
            and isinstance(sha, str) and SHA.fullmatch(sha) and fallback_contract(run) == sha):
        return sha
    return None


def remote_seed(runs, seed):
    full = runs / f"{FULL}-{seed}"
    direct = runs / f"{DIRECT_887 if seed == 887 else DIRECT}-{seed}"
    continuation = runs / f"{CONTINUATION}-{seed}"
    fallback = runs / (FIRST_887 if seed == 887 else ORIGINAL)
    full_contract, direct_contract, continuation_contract = (dispatch_contract(run, seed) for run in (full, direct, continuation))
    if full_contract:
        run, contract, segment = full, full_contract, "v6"
        ancestors = (direct, continuation, fallback)
    elif direct_contract:
        run, contract, segment = direct, direct_contract, "direct"
        ancestors = (continuation, fallback)
    elif continuation_contract:
        run, contract, segment = continuation, continuation_contract, "v5"
        ancestors = (fallback,)
    else:
        run, contract, segment = fallback, fallback_contract(fallback), "historical"
        ancestors = ()
    directory = run / f"seed_{seed}"
    latest, metrics = latest_round(directory)
    for ancestor in ancestors:
        if latest is not None:
            break
        latest, metrics = latest_round(ancestor / f"seed_{seed}")
    worker = read_json(run / "workers" / f"seed_{seed}.json")
    coordinator = pid(read_json(run / "coordinator_generation.json").get("pid"))
    coordinator_live = live(coordinator, run=run)
    worker_pid = pid(worker.get("pid"))
    worker_live = coordinator_live and live(worker_pid, coordinator=coordinator)
    paused = (run / "pause.request").is_file() or (directory / "pause.request").is_file()
    gpu = read_json(run / "gpu" / f"seed_{seed}.json")
    api = pending_attempt(run, directory, latest)
    phase = safe_code(worker.get("status")) or "waiting_to_start"
    if latest == TARGET_ROUNDS:
        phase = "complete"
    elif latest == PILOT_ROUNDS and segment != "v6":
        phase = "awaiting_v6_handoff"
    elif paused:
        phase = "paused"
    elif latest == (5 if seed == 887 else 2) and segment == "direct":
        phase = "direct_round_complete"
    elif phase in {"paused", "failed", "bridge_pending"}:
        pass
    elif worker_live and gpu.get("stage") in ("waiting", "cuda"):
        phase = f"gpu_{gpu['stage']}"
    elif api.get("status") in {"waiting_for_mac", "returned_from_mac"}:
        phase = api["status"]
    elif worker_live and api.get("status") == "completed":
        phase = "selection_or_repair"
    elif not worker_live and phase == "initializing":
        phase = "worker_stopped"
    return {"seed": seed, "run": run.name, "contract_sha256": contract, "segment": segment,
            "continuation": segment in ("v6", "v5", "direct"), "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
            "completed_rounds": latest, "labels": 333 + latest * 32 if latest is not None else None,
            "phase": phase, "worker_pid": worker_pid, "worker_active": worker_live,
            "worker_error": safe_code(worker.get("error_type")), "coordinator_pid": coordinator,
            "coordinator_active": coordinator_live, "gpu": gpu.get("gpu"), "api": api, **metrics}


def command_lines(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=8).splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def remote_report(runs=RUNS):
    return {"seeds": [remote_seed(runs, seed) for seed in SEEDS], "gpus": command_lines([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ]), "gpu_processes": command_lines([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits",
    ])}


def usage(data):
    return {key: data[key] for key in TOKEN_FIELDS if isinstance(data, dict) and type(data.get(key)) is int}


def local_progress(spool, seed, active):
    """Report only the Mac attempt selected by the current GPFS ledger."""
    identity = active.get("attempt_id")
    if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
        return {}
    attempt = spool / "attempts" / identity
    path = attempt / "provider_progress.json"
    status = read_json(path)
    marker = read_json(attempt / "started.json")
    if (status.get("seed") != seed or status.get("attempt_id") != identity or marker.get("attempt_id") != identity
            or not marker.get("job_sha256") or status.get("job_sha256") != marker["job_sha256"]
            or status.get("round") != active.get("round") or status.get("logical_index") != active.get("turn")
            or status.get("phase") not in PHASES):
        return {}
    try:
        updated = path.stat().st_mtime
    except OSError:
        return {}
    result = read_json(spool / "results" / f"{identity}.json")
    ack = read_json(spool / "delivered" / f"{identity}.json")
    sealed = result.get("attempt_id") == identity and result.get("job_sha256") == marker["job_sha256"]
    delivered = sealed and ack == {"attempt_id": identity, "result_sha256": result.get("result_sha256")}
    events = status.get("sse_milestones", {})
    if not isinstance(events, dict):
        events = {}
    first_text_event = events.get("response.output_text.delta")
    first_text = first_text_event.get("elapsed_seconds") if isinstance(first_text_event, dict) else None
    ids = {field: value for field in ("response_id", "provider_request_id")
           if isinstance(value := status.get(field), str) and PROVIDER_ID.fullmatch(value)}
    fields = ("round", "logical_index", "http_status", "elapsed_seconds", "sse_event_count")
    progress = {field: status[field] for field in fields if type(status.get(field)) in (int, float)}
    return {**progress, "phase": status["phase"], "sealed": sealed, "delivered": delivered,
            "progress_age_seconds": max(0, round(time.time() - updated)),
            "event_type": safe_code(status.get("event_type")), "usage": usage(status.get("usage")),
            **ids,
            "first_output_seconds": first_text if type(first_text) in (int, float) else None,
            "complete_event_seen": "response.completed" in events or status.get("event_type") == "response.completed",
            "raw_response_saved": (attempt / "raw_provider_response.json").is_file()
                                  and (attempt / "provider_capture_receipt.json").is_file()}


def dispatcher_pid(seed, segment):
    suffix = {"v6": "v6-full", "v5": "v5-continuation", "direct": ("v9-direct" if seed == 887 else "v8-direct")}[segment]
    label = f"gui/{os.getuid()}/com.purellm.a7-{suffix}-{seed}"
    lines = command_lines(["/bin/launchctl", "print", label])
    for line in lines or ():
        if match := re.fullmatch(r"\s*pid = ([0-9]+)\s*", line):
            return pid(int(match[1]))
    return None


def mac_report(private, seeds):
    pool = private / "spool" / PILOT / "key_pool/key_pool.sqlite3"
    keys, queue, limits = [], [], {}
    try:
        with sqlite3.connect(f"{pool.as_uri()}?mode=ro", uri=True, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            limits = dict(conn.execute("SELECT cap,shared_until,shared_disabled FROM settings").fetchone() or {})
            rows = conn.execute("SELECT seed,round,logical_index,state,key_id,dispatched,response_complete,outcome,usage FROM queue").fetchall()
            for record in conn.execute("SELECT id,present,cooldown_until,disabled_error FROM keys WHERE configured=1 ORDER BY id"):
                alias = record["id"]
                if alias not in KEYS:
                    continue
                state = "available"
                if record["disabled_error"] or not record["present"]:
                    state = "disabled"
                elif any(row["key_id"] == alias and row["state"] == "busy" for row in rows):
                    state = "busy"
                elif record["cooldown_until"] > time.time():
                    state = "cooldown"
                dispatched = [row for row in rows if row["key_id"] == alias and row["dispatched"]]
                used = {name: 0 for name in TOKEN_FIELDS}
                for row in dispatched:
                    for name, count in usage(json.loads(row["usage"] or "{}")).items():
                        used[name] += count
                keys.append({"id": alias, "state": state, "requests": len(dispatched),
                             "complete": sum(row["response_complete"] for row in dispatched),
                             "usage": {field: count for field, count in used.items() if count}})
            queue = [{field: row[field] for field in ("seed", "round", "logical_index", "state", "key_id")}
                     for row in rows if row["state"] in ("queued", "busy") and row["key_id"] in (None, *KEYS)]
    except (OSError, sqlite3.Error, ValueError):
        pass
    for row in seeds:
        contract = row.get("contract_sha256")
        if isinstance(contract, str) and SHA.fullmatch(contract):
            spool = private / "spool" / contract
            if read_json(spool / "run_binding.json").get("contract_sha256") == contract:
                row["provider"] = local_progress(spool, row["seed"], row["api"])
        if row.get("continuation"):
            row["dispatcher_pid"] = dispatcher_pid(row["seed"], row["segment"])
    return {"keys": keys, "queue": queue, "limits": limits}


def show(view):
    print(f"A7 v6 status  {view.get('checked_at', '—')}")
    for row in view.get("seeds", ()):
        api = row["api"]
        provider = row.get("provider") or {}
        rounds = row["completed_rounds"] if row["completed_rounds"] is not None else "—"
        print(f"seed {row['seed']}  L{row['labels'] or '—'}/L{row['target_labels']}  "
              f"rounds {rounds}/{row['target_rounds']}  {row['phase']}  worker {row['worker_pid'] or '—'}  "
              f"coordinator {row['coordinator_pid'] or '—'}  Mac {row.get('dispatcher_pid') or '—'}  {row['run']}")
        if api or provider:
            sealed = ("已回传" if provider.get("delivered") else "已封存待回传" if provider.get("sealed")
                      else "待封存" if provider else "未见当前回执")
            evidence = ("已收到完成终态" if provider.get("complete_event_seen")
                        else "已收到正文" if provider.get("first_output_seconds") is not None
                        else "尚无正文" if provider else "—")
            print(f"  API {api.get('status', '—')}  {api.get('failure') or '—'}  "
                  f"Mac {provider.get('phase', '—')} ({sealed})  HTTP {provider.get('http_status', '—')}  "
                  f"{evidence}  SSE {provider.get('sse_event_count', '—')}  "
                  f"latest event {provider.get('progress_age_seconds', '—')} s ago  tokens {provider.get('usage') or '—'}")
    for gpu in view.get("gpus") or ():
        print(f"GPU {gpu} (MiB used, MiB total, utilization %)")
    for process in view.get("gpu_processes") or ():
        print(f"  GPU 计算进程 {process}")
    if view.get("gpus") is None:
        print("GPU 查询暂不可用")
    mac = view["mac"]
    print("共享 Key 池", ", ".join(f"{row['id']} {row['state']} ({row['complete']}/{row['requests']})"
                                   for row in mac["keys"]) or "暂不可用")
    limits = mac["limits"]
    cooldown = int(max(0, limits.get("shared_until", 0) - time.time()))
    if cooldown or limits.get("shared_disabled"):
        print(f"  共享冷却 {cooldown} 秒，已停用 {bool(limits.get('shared_disabled'))}")
    for row in mac["queue"]:
        print(f"  seed {row['seed']}  round {row['round']}  turn {row['logical_index']}  {row['state']}  {row['key_id'] or '—'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="repeat the read-only snapshot")
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote:
        print(json.dumps(remote_report()))
        return 0
    source = Path(__file__).read_text()
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
               "--", args.ssh_host, f"{ROOT}/environment/venv/bin/python -B - --remote"]
    if args.watch is not None and not 3 <= args.watch <= 300:
        parser.error("watch interval must be between 3 and 300 seconds")
    while True:
        try:
            result = subprocess.run(command, input=source, capture_output=True, text=True, timeout=30, check=True)
            view = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            view = {"remote_unavailable": True}
        view["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        view["mac"] = mac_report(Path.home() / ".config/purellm", view.get("seeds", ()))
        if args.json:
            print(json.dumps(view, ensure_ascii=False, indent=None if args.watch else 2), flush=True)
        else:
            show(view)
            print(flush=True)
        if args.watch is None:
            return int(view.get("remote_unavailable", False))
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 130



if __name__ == "__main__":
    raise SystemExit(main())

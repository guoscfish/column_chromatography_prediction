#!/usr/bin/env python3
"""Show A7 continuation progress, GPUs and the shared Mac Key Pool."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
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
REPAIR = "a7-gpt62-astra-xhigh-ssh-v10-targeted-repair"
DIRECT_FULL = "a7-gpt62-astra-xhigh-ssh-v11-direct-full"
BACKGROUND = "a7-gpt62-astra-xhigh-background-v12"
SERIAL_IPV4 = "a7-gpt62-astra-xhigh-ipv4-v13"
IPV4_EXTENSION = f"{SERIAL_IPV4}-2357-extension-01"
IPV4_CONTINUATION = f"{SERIAL_IPV4}-2357-continuation-01"
IPV4_R2_EXTENSION = f"{SERIAL_IPV4}-2357-r2-extension-01"
IPV4_EXTENSION_DISPATCHERS = {
    157: "dispatch_v13_lane_extension",
    887: "dispatch_v13_lane_extension",
    2357: "dispatch_prepared_ipv4_recovery_2357",
    6101: "dispatch_v13_lane_extension",
}
IPV4_CONTINUATION_DISPATCHERS = {
    887: "dispatch_v13_lane_continuation",
    2357: "dispatch_prepared_ipv4_continue_2357",
    6101: "dispatch_v13_lane_continuation",
}
IPV4_EXTENSION_END_ROUNDS = {157: 4, 887: 7, 2357: 2, 6101: 2}
PILOT_ROUNDS = 10
TARGET_ROUNDS = 36
TARGET_LABELS = 1485
ORIGINAL = "a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10"
FIRST_887 = "a7-gpt62-astra-xhigh-ssh-v5-prompt-pilot-887"
PILOT = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
SEEDS = (157, 887, 2357, 6101, 12203)
UNSTARTED_ANCHORS = {
    6101: "6959c9417166245403dad173ca82347336de53a37d643591ca449b0cf2a929c4",
    12203: "e056a9924b408e8885e7fa619f0ce63a44b8b3b47a2126d57c1fdc80e3df428d",
}
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
        decision = read_json(path.parent / "decision.json")
        diagnosis = decision.get("diagnosis") or {}
        errors = Counter(code for entry in diagnosis.get("errors", ()) if isinstance(entry, dict)
                         if (code := safe_code(entry.get("code"))) is not None)
        return {"status": status, "failure": safe_code(record.get("error")), "ambiguous": record.get("ambiguous") is True,
                "round": index, "turn": int(path.parent.name[-2:]), "attempt": started.get("attempt"),
                "reserved_unix": seconds(started.get("started_unix")),
                "attempt_id": identity, "returned_from_mac": returned,
                "selection": safe_code(decision.get("state")), "selection_errors": dict(errors)}
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
    cohort_contract = None
    for name, mode, dispatcher, capability in (
        (SERIAL_IPV4, "ipv4", "dispatch_prepared_ipv4", "gated_ipv4"),
        (BACKGROUND, "background", "dispatch_prepared_background", "verified_route"),
    ):
        cohort = runs / name
        if candidate := fallback_contract(cohort):
            record = read_json(cohort / "formal_contract.json")
            profile = record.get("profile", {})
            if (tuple(profile.get("confirmation_seeds", ())) == SEEDS
                    and profile.get("provider_capability") == capability
                    and (record.get("predecessor") or {}).get("dispatcher_module") == dispatcher):
                cohort_contract = candidate
                break
    direct_full = runs / f"{DIRECT_FULL}-{seed}"
    full = runs / f"{FULL}-{seed}"
    direct = runs / f"{DIRECT_887 if seed == 887 else DIRECT}-{seed}"
    continuation = runs / f"{CONTINUATION}-{seed}"
    fallback = runs / (FIRST_887 if seed == 887 else ORIGINAL)
    repair = runs / f"{REPAIR}-{seed}"
    direct_full_contract, full_contract, repair_contract, direct_contract, continuation_contract = (
        dispatch_contract(run, seed) for run in (direct_full, full, repair, direct, continuation))
    extension = runs / f"{SERIAL_IPV4}-{seed}-extension-01"
    extension_dispatcher = IPV4_EXTENSION_DISPATCHERS.get(seed)
    extension_record = read_json(extension / "formal_contract.json") if extension_dispatcher else {}
    extension_parent = extension_record.get("predecessor") or {}
    extension_contract = (
        fallback_contract(extension) if cohort_contract and mode == "ipv4"
        and extension_parent.get("parent_contract_sha256") == cohort_contract
        and extension_parent.get("dispatcher_module") == extension_dispatcher
        and extension_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    follow_on = runs / f"{SERIAL_IPV4}-{seed}-continuation-01"
    follow_on_record = read_json(follow_on / "formal_contract.json") if extension_contract else {}
    follow_on_parent = follow_on_record.get("predecessor") or {}
    follow_on_contract = (
        fallback_contract(follow_on) if extension_contract
        and follow_on_parent.get("extension_contract_sha256") == extension_contract
        and follow_on_parent.get("parent_contract_sha256") == cohort_contract
        and follow_on_parent.get("dispatcher_module") == IPV4_CONTINUATION_DISPATCHERS.get(seed)
        and follow_on_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    r2_extension = runs / IPV4_R2_EXTENSION
    r2_record = read_json(r2_extension / "formal_contract.json") if seed == 2357 and follow_on_contract else {}
    r2_parent = r2_record.get("predecessor") or {}
    r2_contract = (
        fallback_contract(r2_extension) if follow_on_contract
        and r2_parent.get("parent_contract_sha256") == follow_on_contract
        and r2_parent.get("dispatcher_module") == "dispatch_v13_2357_r2_sse_extension"
        and r2_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    if r2_contract:
        run, contract, segment = r2_extension, r2_contract, "ipv4_r2_extension"
        ancestors = (follow_on, extension, cohort, direct_full, continuation, fallback)
    elif follow_on_contract:
        run, contract, segment = follow_on, follow_on_contract, "ipv4_continuation"
        ancestors = (extension, cohort, direct_full, continuation, fallback)
    elif extension_contract:
        run, contract, segment = extension, extension_contract, "ipv4_extension"
        ancestors = (cohort, direct_full, continuation, fallback)
    elif cohort_contract:
        run, contract, segment = cohort, cohort_contract, mode
        ancestors = tuple(path for path, approved in (
            (direct_full, direct_full_contract), (repair, repair_contract), (direct, direct_contract),
            (continuation, continuation_contract), (fallback, fallback_contract(fallback)),
        ) if approved)
    elif direct_full_contract:
        run, contract, segment = direct_full, direct_full_contract, "v11"
        ancestors = tuple(path for path, approved in (
            (repair, repair_contract), (direct, direct_contract), (continuation, continuation_contract),
            (fallback, fallback_contract(fallback)),
        ) if approved)
    elif full_contract:
        run, contract, segment = full, full_contract, "v6"
        ancestors = (repair, direct, continuation, fallback)
    elif repair_contract:
        run, contract, segment = repair, repair_contract, "repair"
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
    if seed in UNSTARTED_ANCHORS and not cohort_contract and not any(
        (path / f"seed_{seed}").exists() for path in (run, *ancestors)
    ):
        anchor = (runs.parent.parent / "formal_cuda/row4g_b32_20261124/trajectories"
                  / f"random-{seed}/rounds/random/round_00/model/best.pt")
        expected = UNSTARTED_ANCHORS[seed]
        verified = checksum(anchor) == expected
        return {"seed": seed, "run": None, "contract_sha256": None, "segment": "anchor",
                "continuation": False, "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
                "completed_rounds": 0 if verified else None, "labels": 333 if verified else None,
                "phase": "l333_anchor_verified" if verified else "l333_anchor_unverified",
                "worker_pid": None, "worker_active": False, "worker_error": None, "coordinator_pid": None,
                "coordinator_active": False, "gpu": None, "api": {},
                "checkpoint_sha256": expected if verified else None}
    directory = run / f"seed_{seed}"
    latest, metrics = latest_round(directory)
    for ancestor in ancestors:
        if latest is not None:
            break
        latest, metrics = latest_round(ancestor / f"seed_{seed}")
    if cohort_contract and latest is None:
        lane = record["predecessor"].get("lanes", {}).get(str(seed), {})
        start, checkpoint = lane.get("start_round"), lane.get("checkpoint_sha256")
        if type(start) is int and 0 <= start < TARGET_ROUNDS and isinstance(checkpoint, str) and SHA.fullmatch(checkpoint):
            latest, metrics = start, {"checkpoint_sha256": checkpoint}
    worker = read_json(run / "workers" / f"seed_{seed}.json")
    coordinator = pid(read_json(run / "coordinator_generation.json").get("pid"))
    coordinator_live = live(coordinator, run=run)
    worker_pid = pid(worker.get("pid"))
    worker_live = coordinator_live and (worker_pid == coordinator or live(worker_pid, coordinator=coordinator))
    paused = (run / "pause.request").is_file() or (directory / "pause.request").is_file()
    gpu = read_json(run / "gpu" / f"seed_{seed}.json")
    api = pending_attempt(run, directory, latest)
    phase = safe_code(worker.get("status")) or "waiting_to_start"
    if latest == TARGET_ROUNDS:
        phase = "complete"
    elif segment == "ipv4_extension" and latest == IPV4_EXTENSION_END_ROUNDS[seed] and (
        run / "extension_completion.json"
    ).is_file():
        phase = "transport_extension_complete"
    elif segment == "ipv4_r2_extension" and latest == 3 and (run / "extension_completion.json").is_file():
        phase = "transport_extension_complete"
    elif latest == PILOT_ROUNDS and segment not in ("v6", "v11", "background", "ipv4", "ipv4_continuation"):
        phase = "awaiting_v6_handoff"
    elif paused:
        phase = "paused"
    elif latest == 5 and segment == "repair":
        phase = "targeted_repair_round_complete"
    elif latest == (5 if seed == 887 else 2) and segment == "direct":
        phase = "direct_round_complete"
    elif phase == "failed" and api.get("selection") in {"rejected", "malformed_complete_response", "patch_rejected"}:
        phase = "selection_rejected"
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
            "continuation": segment in ("ipv4", "ipv4_extension", "ipv4_continuation", "ipv4_r2_extension",
                                        "background", "v11", "v6", "v5", "direct", "repair"),
            "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
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


def seconds(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def mac_start_time(marker):
    try:
        parsed = datetime.fromisoformat(marker["started_utc"])
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None


def local_progress(spool, seed, active, *, track_queue=False):
    """Report only the Mac attempt selected by the current GPFS ledger."""
    identity = active.get("attempt_id")
    if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
        return {}
    attempt = spool / "attempts" / identity
    path = attempt / "provider_progress.json"
    status = read_json(path)
    marker = read_json(attempt / "started.json")
    now = time.time()
    start = mac_start_time(marker) if marker.get("attempt_id") == identity else None
    timing = {}
    if track_queue and (reserved := seconds(active.get("reserved_unix"))) is not None:
        if start is not None:
            timing["dispatch_wait_seconds"] = seconds(start - reserved)
        elif not marker and active.get("status") == "waiting_for_mac":
            timing = {"phase": "queued_for_dispatch", "dispatch_wait_seconds": seconds(now - reserved)}
    if (status.get("seed") != seed or status.get("attempt_id") != identity or marker.get("attempt_id") != identity
            or not marker.get("job_sha256") or status.get("job_sha256") != marker["job_sha256"]
            or status.get("round") != active.get("round") or status.get("logical_index") != active.get("turn")
            or status.get("phase") not in PHASES):
        return timing
    try:
        updated = path.stat().st_mtime
    except OSError:
        return timing
    result = read_json(spool / "results" / f"{identity}.json")
    ack = read_json(spool / "delivered" / f"{identity}.json")
    sealed = result.get("attempt_id") == identity and result.get("job_sha256") == marker["job_sha256"]
    delivered = sealed and ack == {"attempt_id": identity, "result_sha256": result.get("result_sha256")}
    events = status.get("sse_milestones", {})
    if not isinstance(events, dict):
        events = {}
    first_text_event = events.get("response.output_text.delta")
    first_text = first_text_event.get("elapsed_seconds") if isinstance(first_text_event, dict) else None
    terminal_event = events.get("response.completed")
    terminal_time = terminal_event.get("elapsed_seconds") if isinstance(terminal_event, dict) else None
    ids = {field: value for field in ("response_id", "conflicting_response_id", "provider_request_id")
           if isinstance(value := status.get(field), str) and PROVIDER_ID.fullmatch(value)}
    fields = ("round", "logical_index", "http_status", "elapsed_seconds", "sse_event_count")
    progress = {field: status[field] for field in fields if type(status.get(field)) in (int, float)}
    elapsed = seconds(status.get("elapsed_seconds"))
    if not sealed and status["phase"] not in ("transport_completed", "transport_failure"):
        if start is not None and (running := seconds(now - start)) is not None:
            elapsed = max(elapsed or 0, running)
    return {**timing, **progress, "phase": status["phase"], "sealed": sealed, "delivered": delivered,
            "progress_age_seconds": max(0, round(now - updated)),
            "event_type": safe_code(status.get("event_type")), "usage": usage(status.get("usage")),
            **ids,
            "provider_elapsed_seconds": elapsed,
            "first_output_seconds": seconds(first_text),
            "last_sse_seconds": seconds(status.get("last_sse_seconds")),
            "last_sse_activity_seconds": seconds(status.get("last_sse_activity_seconds")),
            "last_sse_interval_seconds": seconds(status.get("last_sse_interval_seconds")),
            "complete_event_seen": "response.completed" in events or status.get("event_type") in {"response.completed", "background.completed"},
            "complete_event_seconds": seconds(terminal_time),
            "raw_response_saved": (attempt / "raw_provider_response.json").is_file()
                                  and (attempt / "provider_capture_receipt.json").is_file()}


def dispatcher_pid(seed, segment):
    cohort_services = {"background": "a7-background-v12", "ipv4": "a7-ipv4-v13",
                       "ipv4_extension": f"a7-ipv4-v13-{seed}-extension-01",
                       "ipv4_continuation": f"a7-ipv4-v13-{seed}-continuation-01",
                       "ipv4_r2_extension": "a7-ipv4-v13-2357-r2-extension-01"}
    if segment in cohort_services:
        service = cohort_services[segment]
        label = f"gui/{os.getuid()}/com.purellm.{service}"
    else:
        suffix = {"v11": "v11-direct-full", "v6": "v6-full", "v5": "v5-continuation", "direct": ("v9-direct" if seed == 887 else "v8-direct"),
                  "repair": "v10-targeted-repair"}[segment]
        label = f"gui/{os.getuid()}/com.purellm.a7-{suffix}-{seed}"
    lines = command_lines(["/bin/launchctl", "print", label])
    for line in lines or ():
        if match := re.fullmatch(r"\s*pid = ([0-9]+)\s*", line):
            return pid(int(match[1]))
    return None


def mac_report(private, seeds):
    pool = private / "spool" / PILOT / "key_pool/key_pool.sqlite3"
    background = next((row for row in seeds if row.get("segment") == "background"), None)
    if background and isinstance(contract := background.get("contract_sha256"), str) and SHA.fullmatch(contract):
        pool = private / "spool" / contract / "key_pool/key_pool.sqlite3"
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
                row["provider"] = local_progress(
                    spool, row["seed"], row["api"],
                    track_queue=row["segment"] in ("ipv4", "ipv4_extension", "ipv4_continuation", "ipv4_r2_extension"),
                )
        if row.get("continuation"):
            row["dispatcher_pid"] = dispatcher_pid(row["seed"], row["segment"])
    return {"keys": keys, "queue": queue, "limits": limits}


def duration_text(value):
    return f"{value:.1f} s" if seconds(value) is not None else "—"


def show(view):
    print(f"A7 status  {view.get('checked_at', '—')}")
    for row in view.get("seeds", ()):
        api = row["api"]
        provider = row.get("provider") or {}
        rounds = row["completed_rounds"] if row["completed_rounds"] is not None else "—"
        print(f"seed {row['seed']}  L{row['labels'] or '—'}/L{row['target_labels']}  "
              f"rounds {rounds}/{row['target_rounds']}  {row['phase']}  worker {row['worker_pid'] or '—'}  "
              f"coordinator {row['coordinator_pid'] or '—'}  Mac {row.get('dispatcher_pid') or '—'}  {row['run'] or '—'}")
        if api or provider:
            sealed = ("已回传" if provider.get("delivered") else "已封存待回传" if provider.get("sealed")
                      else "待封存" if provider else "未见当前回执")
            if provider.get("phase") == "queued_for_dispatch":
                sealed = "等待派发"
            print(f"  API {api.get('status', '—')}  {api.get('failure') or '—'}  "
                  f"Mac {provider.get('phase', '—')} ({sealed})  HTTP {provider.get('http_status', '—')}  "
                  f"SSE {provider.get('sse_event_count', '—')}  "
                  f"状态更新 {provider.get('progress_age_seconds', '—')} s 前  tokens {provider.get('usage') or '—'}")
            if provider:
                terminal = ("—" if provider.get("phase") == "queued_for_dispatch"
                            else "已收到" if provider.get("complete_event_seen") else "未收到")
                elapsed = seconds(provider.get("provider_elapsed_seconds"))
                last_event = seconds(provider.get("last_sse_seconds"))
                last_activity = seconds(provider.get("last_sse_activity_seconds"))
                since_event = elapsed - last_event if elapsed is not None and last_event is not None else None
                since_activity = elapsed - last_activity if elapsed is not None and last_activity is not None else None
                inactivity_text = (f"  距上条有效事件 {duration_text(since_activity)}"
                                   if since_activity is not None else "")
                print(f"  派发前等待 {duration_text(provider.get('dispatch_wait_seconds'))}  "
                      f"API 已用时（含预检） {duration_text(elapsed)}  "
                      f"首段正文 {duration_text(provider.get('first_output_seconds'))}  "
                      f"最近两条 SSE 间隔 {duration_text(provider.get('last_sse_interval_seconds'))}  "
                      f"距上条 SSE {duration_text(since_event)}{inactivity_text}  "
                      f"完成终态 {terminal} {duration_text(provider.get('complete_event_seconds'))}")
            if api.get("selection"):
                print(f"  A7 校验 {api['selection']}  {api['selection_errors'] or '—'}")
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

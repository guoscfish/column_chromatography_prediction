#!/usr/bin/env python3
"""Read-only experiment audit plus derivative monitoring/report files.

Never launches workers, calls a provider, opens the source label store, or alters
selection inputs. Process inspection requires the usual read-process approval.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ids_hash(values):
    return stable(sorted(map(str, values)))


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def check_batch(seed, rnd, frozen, audit):
    directory = P / f"nocw/selections/seed_{seed}/llm32_nocw/round_{rnd-1:02d}"
    batch, contract = read(directory / "batch_freeze.json"), read(directory / "contract.json")
    labels, feedback = read(directory / "label_access_receipt.json"), read(directory / "feedback.json")
    ids, labeled, unlabeled = batch["batch_ids"], contract["L_t_ids"], contract["U_t_ids"]
    assert len(ids) == len(set(ids)) == 32 and set(ids) <= set(unlabeled) and not set(ids) & set(labeled)
    assert audit["train_ids_hash"] == frozen["L_t_ids_hash"] == ids_hash(labeled + ids)
    assert batch["checkpoint_sha256"] == contract["checkpoint_sha256"]
    assert labels["batch_freeze_sha256"] == feedback["batch_freeze_sha256"] == sha(directory / "batch_freeze.json")
    assert labels["feedback_sha256"] == stable(feedback)  # Semantic JSON hash, not the file-byte hash.
    assert labels["audit"]["requested_rows"] == 32 and labels["audit"]["requested_ids_hash"] == ids_hash(ids)
    assert labels["audit"]["acquisitions_frozen"] is True
    for name, digest in batch["artifact_hashes"].items():
        assert sha(directory / name) == digest, name
    next_contract = directory.parent / f"round_{rnd:02d}/contract.json"
    if next_contract.exists():
        assert read(next_contract)["L_t_ids"] == labeled + ids
    return {"seed": seed, "fit_round": rnd, "new_unique_ids": 32,
            "frozen_artifacts_and_label_lineage_verified": True, "train_ids_verified": True}


def main():
    old = read(P / "monitor_state.json")
    historical = read(P / "status.json")
    paired_before = read(P / "reports/paired_summary.json")
    launch_name = ("request_limit3_status_20261115.json" if (P / "request_limit3_status_20261115.json").exists()
                   else "auth_retry_launch_20261115.json" if (P / "auth_retry_launch_20261115.json").exists()
                   else "upstream_reconciled_launch_20261114.json")
    worker_script = ("resume_nocw_auth_retry_20261115.py" if launch_name.startswith("auth_retry")
                     else "resume_nocw_upstream_reconciled_20261114.py")
    launch = read(P / launch_name)
    jobs = {j["seed"]: j for j in launch["jobs"]}
    output = subprocess.run(["ps", "-axo", "pid=,ppid=,state=,%cpu=,time=,command="],
                            capture_output=True, text=True, check=True).stdout
    processes = {}
    for line in output.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6:
            processes[int(fields[0])] = {"state": fields[2], "cpu_percent": float(fields[3]),
                                         "cpu_time": fields[4], "command": fields[5]}
    slot_paths = [P / f"locks/compute_{slot}.lock" for slot in range(2)]
    slot_query = subprocess.run(["/usr/sbin/lsof", "-n", "-P", "-F", "pn", *map(str, slot_paths)],
                                capture_output=True, text=True)
    compute_holders, current_pid = {}, None
    for line in slot_query.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            current_pid = int(line[1:])
        elif line.startswith("n") and current_pid is not None:
            path = Path(line[1:]).resolve()
            if path in slot_paths:
                compute_holders.setdefault(current_pid, []).append(path.name)
    rows, workers, new, batch_audits, failures = [], [], [], [], []
    framing_controller = None
    framing_launch_path = P / "sse_frame_repair_launch_20261116.json"
    if framing_launch_path.exists():
        framing_launch = read(framing_launch_path)
        framing_status_path = P / "sse_frame_repair_status_20261116.json"
        framing_status = read(framing_status_path) if framing_status_path.exists() else {}
        controller_pid = framing_launch["controller_pid"]
        controller_proc = processes.get(controller_pid)
        controller_alive = bool(controller_proc and not controller_proc["state"].startswith("Z"))
        pending_adoption = framing_status.get("pending_safe_backoff", [])
        framing_controller = {"pid": controller_pid, "alive": controller_alive,
                              "pending_safe_backoff": pending_adoption,
                              "adopted": framing_status.get("adopted", []),
                              "excluded": framing_status.get("excluded", []),
                              "status_updated_utc": framing_status.get("updated_utc")}
        if controller_proc:
            assert "repair_nocw_sse_frames_20261116.py supervise" in controller_proc["command"]
        startup_age = (datetime.now(timezone.utc) - datetime.fromisoformat(framing_launch["created_utc"])).total_seconds()
        if not controller_alive and (pending_adoption or not framing_status and startup_age > 90):
            failures.append({"method": "sse_frame_controller", "pid": controller_pid,
                             "reason": "safe-handover controller exited before pending adoption completed"})
    totals = {"nocw": 0, "cw": 0}
    for previous in old["per_seed"]:
        method, seed = previous["method"], previous["seed"]
        folder, arm, name = (("nocw", "llm32_nocw", "prediction_freeze.json") if method == "nocw"
                             else ("cw_lcmd", "center_width_lcmd", "fit_receipt.json"))
        files = sorted((P / f"{folder}/runtime/seed_{seed}/{arm}").glob(f"round_*/{name}"))
        rounds = [int(q.parent.name.split("_")[-1]) for q in files]
        if method == "nocw":
            assert rounds == list(range(rounds[-1] + 1))
        last = rounds[-1]
        assert last >= previous["completed_rounds"]
        base_freeze = read(P / f"nocw/runtime/seed_{seed}/llm32_nocw/round_00/prediction_freeze.json")
        baseline = read(Path(base_freeze["checkpoint_path"]).with_name("fit_audit.json"))
        for q, rnd in zip(files, rounds):
            if rnd < previous["completed_rounds"]:
                continue
            receipt = read(q)
            checkpoint = Path(receipt["checkpoint_path"])
            audit = read(checkpoint.with_name("fit_audit.json"))
            assert audit["train_rows"] == 333 + 32 * rnd and audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
            assert audit["initialization_hash"] == baseline["initialization_hash"]
            assert audit["validation_ids_hash"] == baseline["validation_ids_hash"]
            assert sha(checkpoint) == audit["checkpoint_sha256"]
            assert sha(checkpoint.with_name("predictions.csv.gz")) == audit["prediction_sha256"]
            if method == "cw":
                assert sha(checkpoint.with_name("fit_audit.json")) == receipt["fit_audit_sha256"]
            elif rnd > previous["completed_rounds"]:
                batch_audits.append(check_batch(seed, rnd, receipt, audit))
        q = files[-1]
        audit = read(Path(read(q)["checkpoint_path"]).with_name("fit_audit.json"))
        rows.append({**previous, "completed_rounds": last, "completed_budget": 333 + 32 * last,
            "latest_fit_receipt": str(q.relative_to(P)), "metric": audit["best_validation_combined_normalized_rmse"],
            "last_valid_progress_utc": datetime.fromtimestamp(q.stat().st_mtime, timezone.utc).isoformat()})
        totals[method] += last
        if last > previous["completed_rounds"]:
            new.append({"method": method, "seed": seed, "from_round": previous["completed_rounds"], "to_round": last})
        if method == "cw":
            continue
        job = jobs[seed]
        progress = read(P / f"progress/nocw_{seed}.json")
        retry = read(P / f"nocw/progress/retry_seed_{seed}.json")
        stream = read(P / f"nocw/progress/stream_seed_{seed}.json")
        proc = processes.get(job["pid"])
        alive = bool(proc and not proc["state"].startswith("Z"))
        if proc:
            assert str(ROOT / "scripts/studies" / job.get("worker_script", worker_script)) in proc["command"]
            assert re.search(rf"--seed {seed}(\s|$)", proc["command"])
        gate_path = P / f"progress/request_gate_seed_{seed}.json"
        gate = read(gate_path) if gate_path.exists() else {}
        gate = gate if gate.get("pid") == job["pid"] else {}
        handover = job.get("mode") == "handover"
        deadline = job.get("not_before_utc", job.get("dispatched_utc", "1970-01-01T00:00:00+00:00"))
        starting = (job.get("mode") == "limited" and progress.get("pid") != job["pid"]
                    and (datetime.now(timezone.utc) - datetime.fromisoformat(deadline)).total_seconds() < 90)
        if not (starting or handover):
            assert progress["pid"] == job["pid"]
        if retry.get("pid") != job["pid"]:
            # Reusing a saved batch can start training before any new request.
            # The prior worker's retry record is historical in that case.
            retry = {}
        if stream.get("pid") != job["pid"]:
            stream = {}  # Old worker's stream is never evidence for the replacement.
        phase = progress["phase"]
        if handover or starting:
            activity = "GRACEFUL_HANDOVER_OR_ORIGINAL_BACKOFF"
        elif phase == "FAILED_CLOSED" or (not alive and phase != "COMPLETE"):
            activity = "FAILED_OR_EXITED"
            failures.append({"method": "nocw", "seed": seed, "reason": progress.get("reason", "worker missing"),
                             "pid": job["pid"], "completed_rounds": last, "completed_budget": 333 + 32 * last})
        elif phase == "COMPLETE":
            verified = next((entry for entry in old.get("completed_trajectories", [])
                             if entry["method"] == "nocw" and entry["seed"] == seed
                             and entry.get("budget") == 1485), None)
            activity = ("VERIFIED_COMPLETE" if last == 36 and verified
                        and verified.get("deep_audit_sha256") == sha(P / verified["deep_audit"])
                        else "COMPLETE_REQUIRES_FULL_TRAJECTORY_AUDIT")
        elif gate.get("phase") == "WAITING_FOR_REQUEST_SLOT" and gate.get("updated_utc", "") >= retry.get("updated_utc", ""):
            activity = "WAITING_FOR_REQUEST_SLOT"
        elif phase == "TRAINING_OR_WAITING_FOR_SLOT":
            activity = ("TRAINING" if job["pid"] in compute_holders
                        else "REPLAYING_HISTORY_OR_PREPARING_TRAINING" if proc["cpu_percent"] > 5
                        else "WAITING_FOR_COMPUTE_SLOT")
        elif retry.get("updated_utc", "") < progress.get("updated_utc", ""):
            activity = "PREPARING_ACQUISITION" if proc["cpu_percent"] > 5 else "WAITING_FOR_ACQUISITION_COMPUTE_OR_LOCK"
        else:
            activity = "NETWORK_BACKOFF" if retry.get("phase") == "WAITING_RETRY" else "MODEL_REQUEST_ACTIVE"
        batch = P / f"nocw/selections/seed_{seed}/llm32_nocw/round_{last:02d}/batch_freeze.json"
        awake = processes.get(job["keep_awake_pid"])
        awake_matches = bool(awake and re.search(rf"/usr/bin/caffeinate -(?:i|is|si) -w {job['pid']}(\s|$)", awake["command"]))
        workers.append({"seed": seed, "pid": job["pid"], "keep_awake_pid": job["keep_awake_pid"],
            "alive": alive, "keep_awake_alive": awake_matches,
            "phase": phase, "progress_updated_utc": progress.get("updated_utc"), "observed_activity": activity,
            "request_gate": gate, "worker_mode": job.get("mode"),
            "compute_slots_held": compute_holders.get(job["pid"], []),
            "cpu_percent": proc["cpu_percent"] if proc else None,
            "actual_completed_rounds": last, "actual_completed_budget": 333 + 32 * last,
            "next_fit_target_budget": 333 + 32 * (last + 1) if last < 36 else None,
            "pending_fit_round": last + 1 if last < 36 and batch.exists() else None,
            "pending_batch_frozen_utc": datetime.fromtimestamp(batch.stat().st_mtime, timezone.utc).isoformat() if batch.exists() else None,
            **{f"retry_{k}": retry.get(k) for k in ("phase", "updated_utc", "turn", "attempt_index", "retry_at_utc", "backoff_seconds", "last_error_type", "last_error_message")},
            **{f"stream_{k}": stream.get(k) for k in ("phase", "updated_utc", "event_count", "http_status", "error_type")}, "billing_status": "unknown"})
    if new:
        snapshots = P / "reports/monitor_snapshots"
        snapshots.mkdir(exist_ok=True)
        for name in ("validation_curves.csv", "paired_summary.json"):
            source = P / "reports" / name
            target = snapshots / (sha(source) + "_" + name)
            if not target.exists():
                with target.open("xb") as handle:
                    handle.write(source.read_bytes())
        spec = importlib.util.spec_from_file_location("saved_fit_only_report", ROOT / "scripts/studies/run_nocw_cw_paired_to1485_20261112.py")
        frozen_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen_runner)
        frozen_runner.report()  # Only saved fit audits; never worker/supervise/prepare.
    now = datetime.now(timezone.utc).isoformat()
    new_failures = [item for item in failures if item not in old.get("failures", [])]
    decision = "NOTIFY" if new_failures else "DONT_NOTIFY"
    unfinished = [w for w in workers if w["actual_completed_rounds"] < 36]
    bottleneck = "; ".join(
        f"seed {w['seed']}: {w['observed_activity']}, fitted r{w['actual_completed_rounds']}/L{w['actual_completed_budget']}, "
        + (f"frozen batch awaiting fit r{w['pending_fit_round']}/L{w['next_fit_target_budget']}"
           if w["pending_fit_round"] is not None else "no next frozen batch")
        for w in unfinished) or "All saved fit points present; full trajectory audits and final report still required."
    state = {**old, "checked_utc": now, "per_seed": rows, "totals": totals, "new_progress": new,
        "new_batch_audits": batch_audits, "live_workers": sum(w["alive"] for w in workers),
        "keep_awake_active": any(w["keep_awake_alive"] for w in workers), "recovery_workers": workers,
        "failures": failures, "new_failures": new_failures, "active_recovery_launch": launch_name,
        "saved_fit_report_refreshed": bool(new), "decision": decision,
        "sse_frame_repair_controller": framing_controller,
        "historical_effective_bottleneck": old.get("historical_effective_bottleneck", old.get("effective_bottleneck")),
        "effective_bottleneck": bottleneck,
        "reason": "Current process identity/activity checked; only frozen fitted points counted. New fits and batch lineage audited; derivative reports refreshed only on fitted progress. No worker or provider actions performed."}
    if batch_audits:
        history = P / "reports/monitor_fit_audits"
        history.mkdir(exist_ok=True)
        atomic(history / (now.replace(":", "-") + ".json"), {"checked_utc": now, "new_progress": new, "batch_audits": batch_audits})
    atomic(P / "monitor_state.json", state)
    atomic(P / "recovery_status.json", {"updated_utc": now, "active_launch": launch_name,
        "historical_supervisor_status_updated_utc": historical["updated_utc"], "workers": workers,
        "cw_status": "ALL_FIVE_VERIFIED_COMPLETE", "actual_completed_rounds": totals,
        "remaining_nocw_rounds": 180 - totals["nocw"], "complete": totals["nocw"] == totals["cw"] == 180,
        "test_truth_reads": 0})
    result = {"checked_utc": now, "totals": totals, "new_progress": new, "failures": failures,
        "report_refreshed": bool(new), "paired_common_budget_before": paired_before["common_budget_all_five"],
        "workers": workers, "new_failures": new_failures, "decision": decision,
        "sse_frame_repair_controller": framing_controller,
        "known_stall_diagnosis": old.get("request_stall_diagnosis"),
        "transport_activity_is_fitted_progress": False,
        "request_concurrency": {k: launch.get(k) for k in ("request_limit", "legacy_workers_reserved", "transition_complete")}}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Five distinct-key Free selectors; serialize expensive compute, adopt live work."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/qgeognn_al/resources.py").exists())
STUDY = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
EXT = STUDY / "concurrency_v2"
ENTRY = EXT / "frozen_runner.py"
POLICY = STUDY / "parallel_execution_amendment.json"
ORIGINAL = STUDY / "frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py"
TEST = ROOT / "tests/active_learning_v2/test_free_cw_parallel.py"


def load_base():
    global m
    spec = importlib.util.spec_from_file_location("frozen_free_cw_original", ORIGINAL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.import_frozen()
    m.ORIGINAL_FIT, m.ORIGINAL_TRAIN = m.fitrunner.fit_from_same_initialization, m.fitrunner.train_source


def validate():
    result = m.s.validate()
    policy = m.read(POLICY)
    if (result["protocol_sha256"] != policy["original_protocol_sha256"] or
        m.sha(STUDY / "config.json") != policy["original_config_sha256"] or
        policy["requested_model"] != "gpt-6.1-sol" or policy["reasoning_effort"] != "xhigh" or
        policy["maximum_seed_workers"] != 5 or policy["maximum_expensive_compute"] != 1):
        raise RuntimeError("parallel amendment/protocol drift")
    for path, digest in policy["code_hashes"].items():
        if m.sha(ROOT / path) != digest:
            raise RuntimeError("parallel adapter source drift")
    return {**result, "parallel_amendment_sha256": m.sha(POLICY)}


def once(path, value):
    path = Path(path)
    if path == STUDY / "served_model_lock.json":
        path.parent.mkdir(parents=True, exist_ok=True)
        with (STUDY / "locks/served_model.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                return m.base.BASE_ONCE(path, value)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return m.base.BASE_ONCE(path, value)


def activate(seed=None):
    m.activate(seed)
    m.WORKER_SEED = seed
    m.once = once
    m.s.once = once
    m.s.request_with_retries = request_with_retries


def transient_http_reason(reason):
    match = re.fullmatch(r"provider HTTP (\d{3})", reason or "")
    return int(match[1]) if match and 500 <= int(match[1]) <= 599 else None


def attempt_cap(directory, turn, config):
    return 8 if Path(directory).name == "round_00" and turn == 0 else config["transport_attempts"]


def request_with_retries(messages, config, directory, turn, call):
    """Keep request hashes/body and old receipts; bounded gateway recovery only."""
    import httpx
    validate()
    directory = Path(directory)
    request_hash = m.s.stable_hash({"messages": messages, "config": config})
    cap = attempt_cap(directory, turn, config)
    backoff = (2, 10, 60, 120, 120, 300, 600)
    for attempt in range(cap):
        path = directory / f"attempt_{turn:02d}_{attempt:02d}.json"
        started = path.with_suffix(".started.json")
        prefix = directory / f"stream_{turn:02d}_{attempt:02d}"
        if path.exists():
            saved = m.read(path)
            if saved["request_sha256"] != request_hash:
                raise RuntimeError("request drift")
            if saved.get("success"):
                return saved
            if saved.get("fatal_protocol_error"):
                status = transient_http_reason(saved.get("reason"))
                if status is None:
                    raise m.FatalProtocolError(saved["reason"])
                once(EXT / f"reclassified/seed_{m.WORKER_SEED}/{directory.name}/{path.name}", {
                    "original_attempt_path": str(path.relative_to(STUDY)), "original_attempt_sha256": m.sha(path),
                    "http_status": status, "classification": "recoverable provider gateway/server transport failure",
                    "parallel_amendment_sha256": m.sha(POLICY)})
            continue
        completed, raw = prefix.with_suffix(".completed.json"), prefix.with_suffix(".response.json")
        if completed.exists() or raw.exists():
            cached = m.read(completed if completed.exists() else raw)
            if not started.exists() or m.read(started)["request_sha256"] != request_hash or cached["request_sha256"] != request_hash:
                raise RuntimeError("local completion binding mismatch")
            if completed.exists():
                answer, provenance = cached["answer"], cached["provenance"]
                model = provenance["served_model"]
                if not (model == config["model"] or model.startswith(config["model"] + "-")):
                    raise m.FatalProtocolError("cached response has wrong model")
                once(STUDY / "served_model_lock.json", {"served_model": model})
            else:
                answer, provenance = m.response_result(cached["response"], config, cached["request_id"])
            saved = {"request_sha256": request_hash, "success": True, "answer": answer, "provenance": provenance,
                "elapsed_seconds": max(0, (completed if completed.exists() else raw).stat().st_mtime-started.stat().st_mtime),
                "recovered_local_completion": True}
            once(path, saved)
            return saved
        if started.exists():
            raise RuntimeError("ambiguous interrupted provider request; never automatically resend")
        once(started, {"request_sha256": request_hash, "started_utc": m.now(),
            "requested_model": config["model"], "reasoning_effort": config["reasoning_effort"]})
        begin = time.monotonic()
        try:
            answer, provenance = (m.live_stream(messages, config, directory, turn, attempt)
                                  if call is m.live_call else call(copy.deepcopy(messages), copy.deepcopy(config)))
            if not isinstance(answer, str) or not answer.strip():
                raise m.FatalProtocolError("empty answer")
        except (m.FatalProtocolError, httpx.TransportError, httpx.HTTPStatusError) as error:
            fatal = isinstance(error, m.FatalProtocolError)
            status = transient_http_reason(str(error)) if fatal else (
                error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None)
            if fatal and status is None:
                once(path, {"request_sha256": request_hash, "success": False, "fatal_protocol_error": True,
                    "error_type": type(error).__name__, "reason": str(error), "ended_utc": m.now(),
                    "elapsed_seconds": time.monotonic()-begin})
                raise
            once(path, {"request_sha256": request_hash, "success": False, "http_status": status,
                "error_type": type(error).__name__, "ended_utc": m.now(), "elapsed_seconds": time.monotonic()-begin,
                "parallel_amendment_sha256": m.sha(POLICY), "recoverable_transport_error": True})
            if attempt+1 < cap:
                m.base.status_record(m.WORKER_SEED, "TRANSPORT_BACKOFF", turn=turn, attempt=attempt,
                    seconds=backoff[attempt], transport_attempt_cap=cap)
                time.sleep(backoff[attempt])
            continue
        saved = {"request_sha256": request_hash, "success": True, "answer": answer, "provenance": provenance,
            "elapsed_seconds": time.monotonic()-begin, "ended_utc": m.now()}
        once(path, saved)
        return saved
    raise RuntimeError("bounded transport attempt budget exhausted; no fallback")


@contextmanager
def compute_slot(seed):
    # Same lease held by the adopted old worker: it has priority to finish.
    path = STUDY / "locks/new_free_resource.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        m.training_priority_wait()
        handle = path.open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            m.base.status_record(seed, "WAITING_COMPUTE_SLOT", completed_rounds=m.completed_rounds(seed),
                                 current_budget=m.BUDGETS[m.completed_rounds(seed)], target_budget=1485)
            time.sleep(10)
            continue
        try:
            # Recheck source priority after acquiring our independent lease.
            m.training_priority_wait()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return


def select(seed, r):
    m.s.require_authorization()
    with m.s.exclusive_lock(STUDY / ".operation.lock"):
        d = m.s.selection_directory(seed, m.ARM, r)
        if r:
            with compute_slot(seed):
                m.s.stage(seed, m.ARM, r, allow_feature_compute=True)
        else:
            m.s.stage(seed, m.ARM, r)
        if (d / "batch_freeze.json").exists():
            return m.s.audit_batch(d)
        packet = m.read(d / "input.json")
        catalog = m.s.ReadOnlyCatalog(**m.read(d / "catalog.json"))
        result = m.s.run_selection(packet, catalog, d, "cw_aware", m.read(STUDY / "config.json")["transport"], m.live_call)
        return m.s.freeze_batch(d, result, catalog, packet)


def worker(seed):
    activate(seed)
    validate()
    m.s.require_authorization()
    m.credential_audit()
    os.environ.pop("SCIENTIST_API_KEY", None)
    os.environ["SCIENTIST_API_KEY_FILE"] = str(m.base.KEY_FILES[seed])
    m.torch.set_num_threads(2)
    with m.base.BASE_EXCLUSIVE_LOCK(STUDY / f"locks/worker_seed_{seed}.lock"):
        r = m.completed_rounds(seed)
        if r == 36:
            return
        try:
            for j in range(r):
                m.s.audit_batch(m.s.selection_directory(seed, m.ARM, j))
                m.verify_fit(seed, j+1)
            m.base.status_record(seed, "SELECTING", round=r, current_budget=m.BUDGETS[r], target_budget=1485,
                credential_alias=f"existing-independent-seed-{seed}", parallel_amendment_sha256=m.sha(POLICY))
            select(seed, r)
            with compute_slot(seed):
                m.base.status_record(seed, "ADVANCING", round=r, next_budget=m.BUDGETS[r+1], target_budget=1485)
                m.s.advance(seed, m.ARM, r, allow_reveal=True, allow_training=True)
                m.verify_fit(seed, r+1)
            d = m.s.selection_directory(seed, m.ARM, r)
            runtime = STUDY / f"runtime/seed_{seed}/{m.ARM}/round_{r+1:02d}"
            files = [d/n for n in ("batch_freeze.json", "feedback.json", "label_access_receipt.json")]
            files += [runtime/"prediction_freeze.json", runtime/"model/fit_audit.json", runtime/"model/history.csv"]
            once(STUDY / f"progress/completed_seed_{seed}_round_{r:02d}.json", {"seed":seed, "round":r,
                "budget":m.BUDGETS[r+1], "completed_utc":m.now(), "files":{str(p.relative_to(STUDY)):m.sha(p) for p in files}})
            m.base.status_record(seed, "COMPLETE" if r==35 else "QUEUED", completed_rounds=r+1,
                current_budget=m.BUDGETS[r+1], final_budget=1485 if r==35 else None, target_budget=1485)
        except BaseException as error:
            recoverable = isinstance(error, (OSError, MemoryError))
            m.base.status_record(seed, "RECOVERABLE_STOP" if recoverable else "FAILED_CLOSED", round=r,
                current_budget=m.BUDGETS[r], error_type=type(error).__name__, recoverable=recoverable,
                reason=str(error)[:700] if isinstance(error, (RuntimeError,ValueError)) else type(error).__name__, target_budget=1485)
            raise


def lock_busy(seed):
    path = STUDY / f"locks/worker_seed_{seed}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def seed_can_recover(seed, progress):
    if progress.get("phase") != "FAILED_CLOSED":
        return True
    # Only exhausted initial-network failures or misclassified HTTP 5xx can
    # use the finite eight-total-attempt first-answer amendment.
    d = STUDY / f"selections/seed_{seed}/{m.ARM}/round_00"
    if (d/"batch_freeze.json").exists() or progress.get("round", 0) != 0:
        return False
    attempts = [p for p in d.glob("attempt_00_*.json") if re.fullmatch(r"attempt_00_\d+\.json",p.name)]
    if not attempts or len(attempts)>=8:
        return False
    for p in attempts:
        record = m.read(p)
        if record.get("fatal_protocol_error") and transient_http_reason(record.get("reason")) is None:
            return False
    return "transport attempt budget exhausted" in progress.get("reason", "") or transient_http_reason(progress.get("reason")) is not None


def prepare():
    activate()
    original = m.s.validate()
    m.s.require_authorization()
    credentials = m.credential_audit()
    EXT.mkdir(parents=True, exist_ok=True)
    for source, target in ((Path(__file__).resolve(),ENTRY),(TEST,EXT/"frozen_tests.py")):
        if target.exists():
            if m.sha(source) != m.sha(target):
                raise RuntimeError("frozen parallel source already differs")
        else:
            shutil.copy2(source,target)
    policy = {"version":"free_cw_five_distinct_keys_parallel_v2", "user_authorization":"五个seed用五个不同的key跑一下",
        "original_protocol_sha256":original["protocol_sha256"], "original_config_sha256":m.sha(STUDY/"config.json"),
        "requested_model":"gpt-6.1-sol","reasoning_effort":"xhigh","maximum_seed_workers":5,"maximum_expensive_compute":1,
        "key_audit":credentials,"NoCW_modified":False,"frozen_original_source_modified":False,
        "migration":"stop only the old Free supervisor; adopt every live seed worker without interruption or repeat request",
        "scientific_changes":None,"request_body_hash_changes":False,"budget":1485,"accepted_batches":180,
        "initial_round_first_answer_total_transport_attempts":8,"all_other_answer_transport_attempts":4,
        "answer_attempts_per_round":8,"transport_backoff_seconds":[2,10,60,120,120,300,600],
        "gateway_classification":"HTTP 5xx is recoverable transport; preserve original failed receipts and write reclassification sidecars",
        "ambiguous_interruption":"stop without auto-resend; local completed response and completed fit are reused",
        "maximum_local_worker_restarts_per_seed":2,
        "code_hashes":{str(p.relative_to(ROOT)):m.sha(p) for p in (Path(__file__).resolve(),ENTRY,TEST,EXT/"frozen_tests.py")}}
    once(POLICY,policy)
    return {**validate(),"distinct_keys":credentials["distinct_keys"],"real_api_calls":0}


def supervisor():
    activate()
    validate()
    children = {}
    with m.base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/supervisor.lock"):
        while True:
            validate()
            changed = False
            for seed, child in list(children.items()):
                if child.poll() is not None:
                    state = m.read(STUDY/f"progress/seed_{seed}.json")
                    if child.returncode and state.get("phase") not in ("FAILED_CLOSED","RECOVERABLE_STOP"):
                        m.base.status_record(seed,"RECOVERABLE_STOP",reason="unrecorded worker exit",recoverable=True)
                    del children[seed]
                    changed=True
            rows=[]
            for seed in m.SEEDS:
                state=m.read(STUDY/f"progress/seed_{seed}.json")
                if lock_busy(seed):
                    rows.append({"seed":seed,"supervision":"OWN_WORKER_RUNNING" if seed in children else "ADOPTED_EXISTING_WORKER",
                        "pid":children[seed].pid if seed in children else state.get("pid"),"progress":state})
                    continue
                if seed in children:
                    rows.append({"seed":seed,"supervision":"STARTING","pid":children[seed].pid,"progress":state})
                    continue
                if state["phase"]=="COMPLETE":
                    rows.append({"seed":seed,"supervision":"COMPLETE","progress":state})
                    continue
                if not seed_can_recover(seed,state):
                    rows.append({"seed":seed,"supervision":"NEEDS_ATTENTION","progress":state})
                    continue
                ledger=EXT/f"local_restarts_seed_{seed}.json"
                count=m.read(ledger)["count"] if ledger.exists() else 0
                if state["phase"]=="RECOVERABLE_STOP":
                    if count>=2:
                        rows.append({"seed":seed,"supervision":"NEEDS_ATTENTION","progress":state})
                        continue
                    m.base.atomic_json(ledger,{"count":count+1,"updated_utc":m.now()})
                env=dict(os.environ,KMP_DUPLICATE_LIB_OK="TRUE",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
                env.pop("SCIENTIST_API_KEY",None)
                with (STUDY/f"progress/seed_{seed}_parallel.log").open("ab") as out:
                    child=subprocess.Popen([sys.executable,str(ENTRY),"worker","--seed",str(seed),
                        "--allow-paid-llm","--allow-training","--allow-label-reveal"],cwd=ROOT,env=env,
                        stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                children[seed]=child
                m.base.atomic_json(EXT/f"dispatch_seed_{seed}.json",{"seed":seed,"pid":child.pid,"time":m.now(),"amendment_sha256":m.sha(POLICY)})
                rows.append({"seed":seed,"supervision":"DISPATCHED","pid":child.pid,"progress":state})
                time.sleep(3)
            phase="ALL_FIVE_COMPLETE" if all(row["supervision"]=="COMPLETE" for row in rows) else "RUNNING"
            if all(row["supervision"] in ("COMPLETE","NEEDS_ATTENTION") for row in rows) and phase!="ALL_FIVE_COMPLETE":
                phase="NEEDS_ATTENTION"
            m.base.atomic_json(STUDY/"progress/parallel_supervisor.json",{"pid":os.getpid(),"updated_utc":m.now(),
                "phase":phase,"maximum_seed_workers":5,"maximum_expensive_compute":1,"target_budget":1485,"seeds":rows})
            if changed or phase!="RUNNING":
                m.update_reports()
            if phase!="RUNNING":
                return
            time.sleep(10)


def launch():
    activate()
    validate()
    m.credential_audit()
    with m.base.BASE_EXCLUSIVE_LOCK(EXT/"launch.lock"):
        if (STUDY/"progress/parallel_launch.json").exists():
            raise RuntimeError("parallel launch already recorded")
        old=m.read(STUDY/"progress/launch.json")["pid"]
        info=subprocess.run(["ps","-p",str(old),"-o","command="],capture_output=True,text=True).stdout.strip()
        if info:
            if str(ORIGINAL) not in info or " supervise " not in info:
                raise RuntimeError("old supervisor identity mismatch")
            once(EXT/"controller_handover.json",{"old_supervisor_pid":old,"verified_command":info,"time":m.now(),
                "workers_interrupted":False,"amendment_sha256":m.sha(POLICY)})
            os.kill(old,signal.SIGTERM)
        # Wait only for the controller lease, never for or against a live API.
        for _ in range(50):
            try:
                with m.base.BASE_EXCLUSIVE_LOCK(STUDY/"locks/supervisor.lock"):
                    break
            except RuntimeError:
                time.sleep(.1)
        else:
            raise RuntimeError("old controller has not released supervisor lease")
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK="TRUE",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
        with (STUDY/"progress/parallel_supervisor.log").open("ab") as out:
            child=subprocess.Popen([sys.executable,str(ENTRY),"supervise","--allow-paid-llm","--allow-training","--allow-label-reveal"],
                cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        record={"pid":child.pid,"started_utc":m.now(),"previous_supervisor_pid":old,"workers_interrupted":False,
            "maximum_seed_workers":5,"maximum_expensive_compute":1,"amendment_sha256":m.sha(POLICY)}
        once(STUDY/"progress/parallel_launch.json",record)
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","validate","launch","supervise","worker","status"))
    parser.add_argument("--seed",type=int,choices=(157,887,2357,6101,12203))
    for flag in ("allow-paid-llm","allow-training","allow-label-reveal"):
        parser.add_argument("--"+flag,action="store_true")
    a=parser.parse_args()
    if a.action in ("launch","supervise","worker") and not (a.allow_paid_llm and a.allow_training and a.allow_label_reveal):
        parser.error("three explicit authorized execution flags required")
    load_base()
    activate(a.seed if a.action=="worker" else None)
    if a.action=="worker":
        if a.seed is None:
            parser.error("worker requires --seed")
        result=worker(a.seed)
    elif a.action=="supervise":result=supervisor()
    elif a.action=="status":result=m.read(STUDY/"progress/parallel_supervisor.json")
    else:result=globals()[a.action]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

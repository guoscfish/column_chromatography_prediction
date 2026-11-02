#!/usr/bin/env python3
"""User-authorized persistent Free32 retries; frozen science, five distinct keys."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"src/qgeognn_al/resources.py").exists())
STUDY=ROOT/"studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
EXT=STUDY/"persistent_20261102"
ENTRY=EXT/"frozen_runner.py"
POLICY=EXT/"authorization.json"
TEST=ROOT/"tests/active_learning_v2/test_free_cw_persistent.py"


def load_base():
    global a,m,OLD_VALIDATE,OLD_RUN_SELECTION
    spec=importlib.util.spec_from_file_location("frozen_parallel_v2_for_persistent",STUDY/"concurrency_v2/frozen_runner.py")
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    a.load_base();m=a.m
    OLD_VALIDATE=a.validate
    OLD_RUN_SELECTION=m.s.run_selection


def validate():
    original=OLD_VALIDATE()
    policy=m.read(POLICY)
    if (policy["original_protocol_sha256"]!=original["protocol_sha256"] or
        policy["parallel_amendment_sha256"]!=original["parallel_amendment_sha256"] or
        policy["transport_attempt_limit"] is not None or policy["answer_attempt_limit"] is not None or
        policy["local_restart_limit"] is not None or policy["target_budget"]!=1485 or
        policy["requested_model"]!="gpt-6.1-sol" or policy["reasoning_effort"]!="xhigh"):
        raise RuntimeError("persistent authorization drift")
    for path,digest in policy["code_hashes"].items():
        if m.sha(ROOT/path)!=digest:
            raise RuntimeError("persistent frozen source drift")
    return {**original,"persistent_amendment_sha256":m.sha(POLICY)}


def activate_worker(seed=None):
    # a.worker calls a.activate again; use the original m adapter directly.
    m.activate(seed)
    m.WORKER_SEED=seed
    m.once=a.once
    m.s.once=a.once
    m.s.run_selection=run_selection
    m.s.request_with_retries=request_with_retries


def request_hash(messages,config):
    return m.s.stable_hash({"messages":messages,"config":config})


def backoff(attempt):
    return (2,10,60,120,300,600)[min(attempt,5)]


def legacy_transient(record):
    return record.get("fatal_protocol_error") and a.transient_http_reason(record.get("reason")) is not None


def recover_completed(directory,turn,attempt,h,config):
    prefix=directory/f"stream_{turn:02d}_{attempt:02d}"
    completed,raw=prefix.with_suffix(".completed.json"),prefix.with_suffix(".response.json")
    if not completed.exists() and not raw.exists():return None
    cached=m.read(completed if completed.exists() else raw)
    started=directory/f"attempt_{turn:02d}_{attempt:02d}.started.json"
    if not started.exists() or m.read(started)["request_sha256"]!=h or cached["request_sha256"]!=h:
        raise RuntimeError("durable response/request binding mismatch")
    if completed.exists():
        answer,provenance=cached["answer"],cached["provenance"]
        model=provenance["served_model"]
        if not (model==config["model"] or model.startswith(config["model"]+"-")):
            raise m.FatalProtocolError("cached response has wrong model")
        a.once(STUDY/"served_model_lock.json",{"served_model":model})
    else:
        answer,provenance=m.response_result(cached["response"],config,cached.get("request_id"))
    return {"request_sha256":h,"success":True,"answer":answer,"provenance":provenance,
        "elapsed_seconds":max(0,(completed if completed.exists() else raw).stat().st_mtime-started.stat().st_mtime),
        "recovered_local_completion":True}


def reconcile_interruption(directory,turn,attempt,h):
    """A held per-seed lease proves no old worker can still own this attempt."""
    started=directory/f"attempt_{turn:02d}_{attempt:02d}.started.json"
    if m.read(started)["request_sha256"]!=h:raise RuntimeError("interrupted request drift")
    stream=directory/f"stream_{turn:02d}_{attempt:02d}.json"
    state=m.read(stream) if stream.exists() else {}
    owner=state.get("pid")
    if owner and process_command(owner):
        raise RuntimeError("previous request owner remains alive; never duplicate it")
    if any((directory/f"stream_{turn:02d}_{attempt:02d}{suffix}").exists() for suffix in (".completed.json",".response.json")):
        raise RuntimeError("completion appeared during interruption reconciliation")
    receipt={"request_sha256":h,"success":False,"error_type":"InterruptedWorkerWithoutLocalCompletion",
        "elapsed_seconds":max(0,time.time()-started.stat().st_mtime),"ended_utc":m.now(),
        "provider_outcome":"unconfirmed; no durable successful response locally",
        "prior_response_id":state.get("response_id"),"started_sha256":m.sha(started),
        "persistent_amendment_sha256":m.sha(POLICY)}
    a.once(directory/f"attempt_{turn:02d}_{attempt:02d}.json",receipt)
    return receipt


def request_with_retries(messages,config,directory,turn,call):
    import httpx
    validate();directory=Path(directory);h=request_hash(messages,config)
    for attempt in itertools.count():
        path=directory/f"attempt_{turn:02d}_{attempt:02d}.json"
        started=path.with_suffix(".started.json")
        if path.exists():
            saved=m.read(path)
            if saved["request_sha256"]!=h:raise RuntimeError("request drift")
            if saved.get("success"):return saved
            if saved.get("fatal_protocol_error") and not legacy_transient(saved):
                raise m.FatalProtocolError(saved.get("reason","recorded fatal protocol failure"))
            if legacy_transient(saved):
                a.once(EXT/f"reclassified/seed_{m.WORKER_SEED}/{directory.name}/{path.name}",{
                    "attempt_sha256":m.sha(path),"original_attempt":str(path.relative_to(STUDY)),
                    "classification":"recoverable HTTP server/gateway transport failure",
                    "persistent_amendment_sha256":m.sha(POLICY)})
            continue
        recovered=recover_completed(directory,turn,attempt,h,config)
        if recovered is not None:
            a.once(path,recovered);return recovered
        if started.exists():
            reconcile_interruption(directory,turn,attempt,h)
            continue
        validate()
        a.once(started,{"request_sha256":h,"started_utc":m.now(),"pid":os.getpid(),
            "requested_model":config["model"],"reasoning_effort":config["reasoning_effort"],
            "persistent_amendment_sha256":m.sha(POLICY),"effective_transport_limit":None})
        m.base.status_record(m.WORKER_SEED,"SELECTING",round=int(directory.name.split("_")[1]),
            current_budget=333+32*int(directory.name.split("_")[1]),turn=turn,attempt=attempt,
            transport_attempt_number=attempt+1,transport_attempt_cap=None,target_budget=1485)
        begin=time.monotonic()
        try:
            answer,provenance=(m.live_stream(messages,config,directory,turn,attempt)
                if call is m.live_call else call(copy.deepcopy(messages),copy.deepcopy(config)))
            if not isinstance(answer,str) or not answer.strip():raise m.FatalProtocolError("empty answer")
        except (m.FatalProtocolError,httpx.TransportError,httpx.HTTPStatusError,json.JSONDecodeError) as error:
            fatal=isinstance(error,m.FatalProtocolError)
            status=a.transient_http_reason(str(error)) if fatal else (
                error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None)
            if (fatal and status is None) or (isinstance(error,httpx.HTTPStatusError) and
                    status not in (408,409,429) and not 500<=status<=599):
                a.once(path,{"request_sha256":h,"success":False,"fatal_protocol_error":True,
                    "error_type":type(error).__name__,"reason":str(error),"ended_utc":m.now(),
                    "elapsed_seconds":time.monotonic()-begin})
                if fatal:raise
                raise m.FatalProtocolError("provider HTTP "+str(status)) from error
            delay=backoff(attempt)
            a.once(path,{"request_sha256":h,"success":False,"error_type":type(error).__name__,
                "http_status":status,"ended_utc":m.now(),"elapsed_seconds":time.monotonic()-begin,
                "persistent_amendment_sha256":m.sha(POLICY),"next_backoff_seconds":delay})
            m.base.status_record(m.WORKER_SEED,"TRANSPORT_BACKOFF",turn=turn,attempt=attempt,
                seconds=delay,retry_at_utc=datetime.fromtimestamp(time.time()+delay,timezone.utc).isoformat(),
                transport_attempt_cap=None,target_budget=1485)
            time.sleep(delay)
            continue
        saved={"request_sha256":h,"success":True,"answer":answer,"provenance":provenance,
            "elapsed_seconds":time.monotonic()-begin,"ended_utc":m.now()}
        a.once(path,saved);return saved


def run_selection(packet,catalog,directory,variant,config,call):
    if any(packet.get(k)!=v for k,v in m.s.encode_catalog(catalog).items()):
        raise ValueError("packet differs from complete catalog")
    if (packet.get("selection_count")!=32 or catalog.pending or
        packet.get("candidate_count")!=len(catalog.candidates) or
        packet.get("observed_count")!=len(catalog.observed) or
        packet.get("active_label_count")!=len(catalog.observed)):
        raise ValueError("full Free32 packet count mismatch")
    if packet["packet_sha256"]!=m.s.stable_hash({k:v for k,v in packet.items() if k!="packet_sha256"}):
        raise ValueError("packet binding mismatch")
    directory=Path(directory)
    messages=[{"role":"system","content":m.s.prompt(variant)},
        {"role":"user","content":json.dumps(packet,ensure_ascii=False,separators=(",",":"),allow_nan=False)}]
    for turn in itertools.count():
        if variant=="no_cw":m.s.audit_no_cw(messages)
        a.once(directory/f"request_{turn:02d}.json",{"messages":messages,"config":config})
        saved=request_with_retries(messages,config,directory,turn,call)
        a.once(directory/f"turn_{turn:02d}.json",saved)
        try:
            result=m.s.validate_plan(json.loads(saved["answer"]),catalog,packet,variant)
        except (ValueError,TypeError,KeyError) as error:
            correction={"error":str(error)[:1000],"instruction":
                "Return corrected final JSON from the SAME complete pool. Exactly 32 distinct candidate IDs. "
                "Measured evidence belongs only in typed observed-ID lists. No queries or tools."}
            a.once(directory/f"rejected_{turn:02d}.json",correction)
            messages += [{"role":"assistant","content":saved["answer"]},
                         {"role":"user","content":json.dumps(correction)}]
            continue
        a.once(directory/"selection.json",result);return result


def process_command(pid):
    if not pid:return ""
    r=subprocess.run(["ps","-p",str(pid),"-o","stat=,command="],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else ""


def unresolved(seed):
    root=STUDY/f"selections/seed_{seed}/{m.ARM}"
    return [p for p in root.glob("round_*/attempt_*.started.json")
        if not p.with_name(p.name.replace(".started.json",".json")).exists()]


def transient_stop(state):
    if state.get("phase") in ("QUEUED","RECOVERABLE_STOP","SELECTING","ADVANCING","WAITING_COMPUTE_SLOT","WAITING_NOCW_TRAINING","TRANSPORT_BACKOFF"):
        return True
    reason=state.get("reason","")
    return ("transport attempt budget exhausted" in reason or "selection remained invalid after bounded corrections" in reason or
            a.transient_http_reason(reason) is not None or state.get("error_type") in ("OSError","MemoryError","TimeoutError","ConnectionError"))


def retire_worker(seed,pid,reason,*,idle_limit=None):
    """Freeze and recheck state/completion before terminating only our worker."""
    info=process_command(pid)
    owners=(str(STUDY/"concurrency_v2/frozen_runner.py"),str(ENTRY))
    if not info or not any(owner in info for owner in owners) or f"worker --seed {seed}" not in info:
        return False
    if idle_limit is not None:
        # Healthy in-flight requests are only observed; never pause them on polling.
        stream_path=STUDY/f"progress/stream_seed_{seed}.json"
        if not stream_path.exists():return False
        stream=m.read(stream_path)
        stamp=stream.get("last_line_utc") or stream.get("started_utc")
        if (stream.get("pid")!=pid or stream.get("phase") not in
                ("CONNECTING","CONNECTED","CREATED","QUEUED","IN_PROGRESS") or not stamp or
                time.time()-datetime.fromisoformat(stamp).timestamp()<idle_limit):return False
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(30):
            if process_command(pid).startswith("T"):break
            time.sleep(.05)
        else:return False
        state=m.read(STUDY/f"progress/seed_{seed}.json")
        if state.get("pid")!=pid:return False
        pending=unresolved(seed)
        receipts=[]
        if idle_limit is None:
            if state.get("phase")!="TRANSPORT_BACKOFF" or pending:return False
        else:
            stream_path=STUDY/f"progress/stream_seed_{seed}.json"
            stream=m.read(stream_path)
            if stream.get("pid")!=pid or stream.get("phase") not in ("CONNECTING","CONNECTED","CREATED","QUEUED","IN_PROGRESS"):
                return False
            stamp=stream.get("last_line_utc") or stream.get("started_utc")
            if not stamp or time.time()-datetime.fromisoformat(stamp).timestamp()<idle_limit:return False
            if len(pending)!=1 or pending[0].name!=f"attempt_{stream['turn']:02d}_{stream['attempt']:02d}.started.json":
                return False
            for started in pending:
                stem=started.name.replace("attempt_", "stream_").replace(".started.json", "")
                if any((started.parent/(stem+suffix)).exists() for suffix in (".completed.json",".response.json")):
                    return False
                begin=m.read(started)
                receipt=started.with_name(started.name.replace(".started.json",".json"))
                value={"request_sha256":begin["request_sha256"],"success":False,
                    "error_type":"WallClockInactivityTimeout","ended_utc":m.now(),
                    "elapsed_seconds":max(0,time.time()-started.stat().st_mtime),
                    "provider_outcome":"unconfirmed; no durable local completed response",
                    "prior_response_id":stream.get("response_id"),"started_sha256":m.sha(started),
                    "persistent_amendment_sha256":m.sha(POLICY)}
                a.once(receipt,value);receipts.append(str(receipt.relative_to(STUDY)))
        a.once(EXT/f"handover/seed_{seed}_pid_{pid}.json",{"seed":seed,"pid":pid,"time":m.now(),
            "reason":reason,"verified_command":info,"pending_failure_receipts":receipts,
            "no_durable_completed_response_discarded":True,"persistent_amendment_sha256":m.sha(POLICY)})
        os.kill(pid,signal.SIGTERM)
        return True
    finally:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass


def worker(seed):
    activate_worker(seed)
    validate()
    # Reuse the unchanged select/freeze/reveal/scratch-fit/round marker lifecycle.
    return a.worker(seed)


def prepare():
    activate_worker()
    prior=OLD_VALIDATE();m.s.require_authorization();keys=m.credential_audit()
    EXT.mkdir(parents=True,exist_ok=True)
    for source,target in ((Path(__file__).resolve(),ENTRY),(TEST,EXT/"frozen_tests.py")):
        if target.exists():
            if m.sha(target)!=m.sha(source):raise RuntimeError("persistent snapshot differs")
        else:shutil.copy2(source,target)
    policy={"version":"free_cw_persistent_recovery_20261102_v1",
        "authorized_utc":m.read(POLICY)["authorized_utc"] if POLICY.exists() else m.now(),"user_authorization":"帮我继续跑完，如果不行就一直重试，不用问我超过上限次数怎么办。一直跑就行",
        "original_protocol_sha256":prior["protocol_sha256"],"parallel_amendment_sha256":prior["parallel_amendment_sha256"],
        "requested_model":"gpt-6.1-sol","reasoning_effort":"xhigh","target_budget":1485,"planned_accepted_batches":180,
        "seeds":list(m.SEEDS),"key_audit":keys,"maximum_seed_workers":5,"maximum_expensive_compute":1,
        "transport_attempt_limit":None,"answer_attempt_limit":None,"local_restart_limit":None,
        "backoff_seconds":[2,10,60,120,300,600],"maximum_backoff_seconds":600,
        "local_restart_backoff_seconds":60,"wall_clock_no_stream_data_watchdog_seconds":660,
        "unchanged_science":"split, L333, input encoding/order/precision, prompts, memory, schema/validators, scratch training, checkpoint rule and endpoint",
        "unchanged_request_binding":"original config and request hashes retained; this authorization supersedes only retry/recovery bounds",
        "completion_reuse":"never resend a durable successful response or redo a durable completed fit",
        "interrupted_request_accounting":"only after prior owner gone or verified stopped; preserve started marker, record unconfirmed provider outcome, retry under explicit persistent authorization",
        "fatal_conditions":"credential/quota/model-support/returned-model changes, source/protocol/artifact drift and unknown non-recoverable failures are reported; no fallback",
        "handover":"retain in-flight legacy requests; retire old workers only at verified backoff or wall-clock inactivity without local completed response",
        "sleep":"caffeinate -i tied to supervisor; clamshell sleep still pauses local work",
        "NoCW_modified":False,"frozen_predecessor_modified":False,
        "code_hashes":{str(p.relative_to(ROOT)):m.sha(p) for p in (Path(__file__).resolve(),ENTRY,TEST,EXT/"frozen_tests.py")}}
    a.once(POLICY,policy)
    return {**validate(),"keys":keys["distinct_keys"],"paid_offline_probes":0}


def supervisor():
    activate_worker();validate();m.s.require_authorization()
    children={};report_at=0
    with m.base.BASE_EXCLUSIVE_LOCK(STUDY/"locks/supervisor.lock"):
        while True:
            validate();changed=False
            for seed,child in list(children.items()):
                if child.poll() is not None:
                    state=m.read(STUDY/f"progress/seed_{seed}.json")
                    if child.returncode and state.get("phase") not in ("FAILED_CLOSED","RECOVERABLE_STOP"):
                        m.base.status_record(seed,"RECOVERABLE_STOP",reason="unexpected worker exit",recoverable=True)
                    del children[seed];changed=True
                    a.once(EXT/f"worker_exits/seed_{seed}_pid_{child.pid}.json",{"seed":seed,"pid":child.pid,"exit_code":child.returncode,"time":m.now()})
                    m.base.atomic_json(EXT/f"restart_after_seed_{seed}.json",{"epoch":time.time()+60,"reason":"persistent local recovery cooldown"})
            rows=[]
            for seed in m.SEEDS:
                state=m.read(STUDY/f"progress/seed_{seed}.json")
                if a.lock_busy(seed):
                    pid=children[seed].pid if seed in children else state.get("pid")
                    info=process_command(pid)
                    legacy=str(STUDY/"concurrency_v2/frozen_runner.py") in info
                    if legacy and state.get("phase")=="TRANSPORT_BACKOFF" and not unresolved(seed):
                        moved=retire_worker(seed,pid,"safe completed-failure backoff migration")
                        rows.append({"seed":seed,"pid":pid,"supervision":"MIGRATING_AT_SAFE_BACKOFF" if moved else "ADOPTED_EXISTING_WORKER","progress":state})
                    else:
                        retired=retire_worker(seed,pid,"wall-clock stream inactivity",idle_limit=660)
                        rows.append({"seed":seed,"pid":pid,"supervision":"RECOVERING_IDLE_CONNECTION" if retired else ("PERSISTENT_WORKER_RUNNING" if seed in children else "ADOPTED_EXISTING_WORKER"),"progress":state})
                    continue
                if seed in children:
                    rows.append({"seed":seed,"pid":children[seed].pid,"supervision":"STARTING","progress":state});continue
                if state.get("phase")=="COMPLETE":
                    rows.append({"seed":seed,"supervision":"COMPLETE","progress":state});continue
                if not transient_stop(state):
                    rows.append({"seed":seed,"supervision":"BLOCKED_CONFIGURATION_OR_PROTOCOL","progress":state});continue
                cooldown=EXT/f"restart_after_seed_{seed}.json"
                if cooldown.exists() and m.read(cooldown)["epoch"]>time.time():
                    rows.append({"seed":seed,"supervision":"LOCAL_RECOVERY_BACKOFF","progress":state});continue
                env=dict(os.environ,KMP_DUPLICATE_LIB_OK="TRUE",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
                env.pop("SCIENTIST_API_KEY",None)
                with (STUDY/f"progress/seed_{seed}_persistent.log").open("ab") as out:
                    child=subprocess.Popen([sys.executable,str(ENTRY),"worker","--seed",str(seed),"--allow-paid-llm","--allow-training","--allow-label-reveal"],
                        cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                children[seed]=child
                m.base.atomic_json(EXT/f"dispatch_seed_{seed}.json",{"seed":seed,"pid":child.pid,"time":m.now(),"authorization_sha256":m.sha(POLICY)})
                rows.append({"seed":seed,"pid":child.pid,"supervision":"DISPATCHED","progress":state})
                time.sleep(2)
            complete=all(row["supervision"]=="COMPLETE" for row in rows)
            blocked=all(row["supervision"] in ("COMPLETE","BLOCKED_CONFIGURATION_OR_PROTOCOL") for row in rows)
            phase="ALL_FIVE_COMPLETE" if complete else "BLOCKED" if blocked else "RUNNING"
            record={"pid":os.getpid(),"updated_utc":m.now(),"phase":phase,"target_budget":1485,
                "maximum_seed_workers":5,"maximum_expensive_compute":1,"transport_attempt_cap":None,"answer_attempt_cap":None,
                "persistent_authorization_sha256":m.sha(POLICY),"seeds":rows}
            m.base.atomic_json(STUDY/"progress/persistent_supervisor.json",record)
            m.base.atomic_json(STUDY/"progress/parallel_supervisor.json",record)
            if changed or phase!="RUNNING" or time.time()-report_at>300:
                m.update_reports();report_at=time.time()
            if complete or blocked:return
            time.sleep(15)


def launch():
    activate_worker();validate();m.s.require_authorization();m.credential_audit()
    with m.base.BASE_EXCLUSIVE_LOCK(EXT/"launch.lock"):
        if (EXT/"launch.json").exists():raise RuntimeError("persistent launch already recorded; inspect status")
        active=m.read(STUDY/"active_runtime.json")
        old=active["supervisor_pid"];info=process_command(old)
        if info:
            allowed=(str(STUDY/"concurrency_v2/controller_guard.py"),str(STUDY/"concurrency_v2/frozen_runner.py"))
            if not any(path in info for path in allowed) or "supervise" not in info:
                raise RuntimeError("old Free controller identity mismatch")
            a.once(EXT/"controller_handover.json",{"old_pid":old,"command":info,"time":m.now(),"NoCW_signaled":False})
            os.kill(old,signal.SIGTERM)
        for _ in range(100):
            try:
                with m.base.BASE_EXCLUSIVE_LOCK(STUDY/"locks/supervisor.lock"):break
            except RuntimeError:time.sleep(.1)
        else:raise RuntimeError("old controller lease not released")
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK="TRUE",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
        with (STUDY/"progress/persistent_supervisor.log").open("ab") as out:
            child=subprocess.Popen([sys.executable,str(ENTRY),"supervise","--allow-paid-llm","--allow-training","--allow-label-reveal"],
                cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        keepawake=subprocess.Popen(["/usr/bin/caffeinate","-i","-w",str(child.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        record={"pid":child.pid,"caffeinate_pid":keepawake.pid,"started_utc":m.now(),"old_supervisor_pid":old,
            "entry":str(ENTRY),"authorization_sha256":m.sha(POLICY),"transport_cap":None,"answer_cap":None,"target_budget":1485}
        a.once(EXT/"launch.json",record)
        m.base.atomic_json(STUDY/"active_runtime.json",{"entry":str(ENTRY),"controller_entry":str(ENTRY),"supervisor_pid":child.pid,
            "status":"progress/persistent_supervisor.json","launch":"persistent_20261102/launch.json","maximum_seed_workers":5,
            "maximum_expensive_compute":1,"target_budget":1485,"distinct_keys":5,"transport_attempt_cap":None,"answer_attempt_cap":None})
        return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","validate","launch","supervise","worker","status"))
    p.add_argument("--seed",type=int,choices=(157,887,2357,6101,12203))
    for flag in ("allow-paid-llm","allow-training","allow-label-reveal"):p.add_argument("--"+flag,action="store_true")
    args=p.parse_args()
    if args.action in ("launch","supervise","worker") and not (args.allow_paid_llm and args.allow_training and args.allow_label_reveal):
        p.error("explicit authorized paid/retrain/reveal flags required")
    load_base();activate_worker(args.seed if args.action=="worker" else None)
    a.activate=activate_worker;a.validate=validate
    if args.action=="worker":
        if args.seed is None:p.error("worker requires --seed")
        result=worker(args.seed)
    elif args.action=="status":result=m.read(STUDY/"progress/persistent_supervisor.json")
    else:result=globals()[args.action]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()

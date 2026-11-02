#!/usr/bin/env python3
"""Bound controller recovery to the failed initial answer, never later answers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"src/qgeognn_al/resources.py").exists())
STUDY=ROOT/"studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
EXT=STUDY/"concurrency_v2"
ENTRY=EXT/"controller_guard.py"
POLICY=EXT/"controller_guard.json"


def allow_initial_recovery(records, eligible):
    return eligible and not any(record.get("success") for record in records)


def load():
    global a
    spec=importlib.util.spec_from_file_location("frozen_parallel_v2",EXT/"frozen_runner.py")
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    a.load_base();a.activate()
    original_validate=a.validate
    original_eligible=a.seed_can_recover
    def validate():
        result=original_validate()
        policy=a.m.read(POLICY)
        if a.m.sha(ENTRY)!=policy["controller_sha256"] or a.m.sha(a.POLICY)!=policy["parallel_amendment_sha256"]:
            raise RuntimeError("controller guard source drift")
        return result
    def eligible(seed,state):
        result=original_eligible(seed,state)
        if state.get("phase")!="FAILED_CLOSED":
            return result
        d=STUDY/f"selections/seed_{seed}/{a.m.ARM}/round_00"
        records=[a.m.read(p) for p in d.glob("attempt_00_*.json") if re.fullmatch(r"attempt_00_\d+\.json",p.name)]
        return allow_initial_recovery(records,result)
    a.validate=validate
    a.seed_can_recover=eligible


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("self-test","launch","supervise"))
    args=p.parse_args()
    if args.action=="self-test":
        assert allow_initial_recovery([{"success":False}],True)
        assert not allow_initial_recovery([{"success":False},{"success":True}],True)
        assert not allow_initial_recovery([{"success":False}],False)
        print("3 controller recovery guard checks passed; zero API calls")
        return
    load();a.validate()
    if args.action=="supervise":
        return a.supervisor()
    with a.m.base.BASE_EXCLUSIVE_LOCK(EXT/"controller_guard_launch.lock"):
        record_path=EXT/"controller_guard_launch.json"
        if record_path.exists():
            raise RuntimeError("guard controller already launched")
        old=a.m.read(STUDY/"progress/parallel_launch.json")["pid"]
        info=subprocess.run(["ps","-p",str(old),"-o","command="],capture_output=True,text=True).stdout.strip()
        if info:
            if str(EXT/"frozen_runner.py") not in info or " supervise " not in info:
                raise RuntimeError("parallel controller identity mismatch")
            os.kill(old,signal.SIGTERM)
        for _ in range(50):
            try:
                with a.m.base.BASE_EXCLUSIVE_LOCK(STUDY/"locks/supervisor.lock"):break
            except RuntimeError:time.sleep(.1)
        else:raise RuntimeError("controller lease not released")
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK="TRUE",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
        with (STUDY/"progress/parallel_supervisor_guard.log").open("ab") as out:
            child=subprocess.Popen([sys.executable,str(ENTRY),"supervise"],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        record={"pid":child.pid,"previous_controller_pid":old,"time":a.m.now(),"workers_interrupted":False,
            "worker_entry_unchanged":str(EXT/"frozen_runner.py"),"maximum_seed_workers":5,"maximum_expensive_compute":1}
        a.once(record_path,record)
        print(json.dumps(record,indent=2))


if __name__=="__main__":main()

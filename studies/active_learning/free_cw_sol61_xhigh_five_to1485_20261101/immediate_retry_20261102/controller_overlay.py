#!/usr/bin/env python3
"""User-authorized zero transport backoff; preserve all scientific execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ENTRY=Path(__file__).resolve()
PRIOR=STUDY/'http403_recovery_20261102/controller_overlay.py'
BASE=STUDY/'persistent_20261102_v2/frozen_runner.py'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def install(adapter):
    adapter.backoff=lambda attempt:0

def bind(adapter):
    original_validate,original_retire=adapter.validate,adapter.retire_worker
    def validate():
        policy=json.loads((HERE/'authorization.json').read_text())
        if (policy['transport_backoff_seconds']!=0 or policy['maximum_seed_workers']!=5 or
            policy['maximum_expensive_compute']!=1 or policy['model']!='gpt-6.1-sol' or
            policy['reasoning_effort']!='xhigh' or policy['target_budget']!=1485):
            raise RuntimeError('immediate retry authorization drift')
        for name,h in policy['code_hashes'].items():
            if hashlib.sha256((STUDY/name).read_bytes()).hexdigest()!=h:
                raise RuntimeError('immediate retry binding mismatch')
        return original_validate()
    def retire(seed,pid,reason,**kwargs):
        entry=adapter.ENTRY
        try:
            if str(PRIOR) in adapter.process_command(pid):adapter.ENTRY=PRIOR
            return original_retire(seed,pid,reason,**kwargs)
        finally:adapter.ENTRY=entry
    adapter.validate,adapter.retire_worker=validate,retire
    adapter.ENTRY=ENTRY
    install(adapter)

def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('worker','supervise','status','validate'):
        raise SystemExit('use worker, supervise, status or validate with existing authorization flags')
    adapter=module('unchanged_persistent_immediate',BASE)
    h2=module('unchanged_h2_immediate',STUDY/'transport_recovery_20261102/controller_overlay.py')
    shim=module('unchanged_h2_shim_immediate',STUDY/'transport_recovery_20261102/upstream_h2_worker.py')
    recovery=module('unchanged_403_immediate',PRIOR)
    h2.bind(adapter,shim);recovery.bind(adapter);bind(adapter);adapter.main()

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Explicit manual resume past five pinned historical 401 receipts only."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ENTRY=Path(__file__).resolve()
PRIOR=STUDY/'three_parallel_lagging_20261107/controller_overlay.py'

def fingerprint(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def historical_401(record,policy):
    return (record.get('fatal_protocol_error') is True and record.get('reason')=='provider HTTP 401'
            and fingerprint(record) in policy['historical_receipt_fingerprints'])

def resume_state(state,policy):
    return (state.get('phase')=='FAILED_CLOSED' and state.get('reason')=='provider HTTP 401'
            and fingerprint(state) in policy['historical_state_fingerprints'])

def build():
    spec=importlib.util.spec_from_file_location('retained_three_manual_resume',PRIOR)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    # The retained scheduler uses its ENTRY global for dispatch and watchdog identity.
    prior.ENTRY=ENTRY
    adapter=prior.build()
    old_load,old_validate,old_transient=adapter.load_base,adapter.validate,adapter.transient_stop
    policy=json.loads((HERE/'authorization.json').read_text())
    def load():
        old_load()
        old_legacy=adapter.legacy_transient
        adapter.legacy_transient=lambda record:historical_401(record,policy) or old_legacy(record)
    def validate():
        result=old_validate()
        if (policy.get('user_authorization')!='帮我重新开始跑' or policy.get('new_http401_retry') is not False
                or policy.get('maximum_seed_workers')!=3 or policy.get('maximum_expensive_compute')!=3):
            raise RuntimeError('manual resume authorization drift')
        for group in ('code_hashes','preserved_fatal_receipts'):
            for name,digest in policy[group].items():
                if hashlib.sha256((STUDY/name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('manual resume binding mismatch: '+name)
        return {**result,'manual_resume_authorization_sha256':adapter.m.sha(HERE/'authorization.json')}
    adapter.load_base=load
    adapter.validate=validate
    adapter.transient_stop=lambda state:resume_state(state,policy) or old_transient(state)
    adapter.ENTRY=ENTRY
    return adapter

if __name__=='__main__':build().main()

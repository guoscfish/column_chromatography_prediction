#!/usr/bin/env python3
"""Exact upstream_error retry authorized by the user on 2026-11-04."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ENTRY = Path(__file__).resolve()
PRIOR = STUDY / 'immediate_retry_20261102/controller_overlay.py'
REASON = 'provider terminal code: upstream_error'


def install(adapter):
    old_legacy, old_stream = adapter.legacy_transient, adapter.m.live_stream
    def legacy(record):
        return (record.get('fatal_protocol_error') is True and record.get('reason') == REASON) or old_legacy(record)
    def stream(*args, **kwargs):
        import httpx
        try:
            return old_stream(*args, **kwargs)
        except adapter.m.FatalProtocolError as error:
            if str(error) != REASON:
                raise
            raise httpx.RemoteProtocolError(REASON) from error
    adapter.legacy_transient, adapter.m.live_stream = legacy, stream


def bind(adapter):
    old_load, old_validate = adapter.load_base, adapter.validate
    old_transient, old_retire = adapter.transient_stop, adapter.retire_worker
    def load():
        old_load()
        install(adapter)
    def validate():
        policy = json.loads((HERE / 'authorization.json').read_text())
        expected = {'exact_retry_reason': REASON, 'user_authorization': '持续重试',
                    'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh',
                    'target_budget': 1485, 'maximum_seed_workers': 5,
                    'maximum_expensive_compute': 1, 'transport_backoff_seconds': 0,
                    'local_restart_cooldown_seconds': 60}
        if any(policy.get(k) != v for k, v in expected.items()):
            raise RuntimeError('upstream_error recovery authorization drift')
        for group in ('code_hashes', 'preserved_fatal_receipts'):
            for name, digest in policy[group].items():
                if hashlib.sha256((STUDY / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('upstream_error recovery binding mismatch: ' + name)
        return old_validate()
    def transient(state):
        return state.get('reason') == REASON or old_transient(state)
    def retire(seed, pid, reason, **kwargs):
        entry = adapter.ENTRY
        try:
            if str(PRIOR) in adapter.process_command(pid):
                adapter.ENTRY = PRIOR
            return old_retire(seed, pid, reason, **kwargs)
        finally:
            adapter.ENTRY = entry
    adapter.load_base, adapter.validate = load, validate
    adapter.transient_stop, adapter.retire_worker = transient, retire
    adapter.ENTRY = ENTRY


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('worker', 'supervise', 'status', 'validate'):
        raise SystemExit('use worker, supervise, status or validate with existing authorization flags')
    import importlib.util
    spec = importlib.util.spec_from_file_location('prior_immediate_retry', PRIOR)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    adapter = prior.module('frozen_persistent_upstream_error', prior.BASE)
    h2 = prior.module('h2_upstream_error', STUDY / 'transport_recovery_20261102/controller_overlay.py')
    shim = prior.module('h2_shim_upstream_error', STUDY / 'transport_recovery_20261102/upstream_h2_worker.py')
    recovery = prior.module('http403_upstream_error', prior.PRIOR)
    h2.bind(adapter, shim)
    recovery.bind(adapter)
    prior.bind(adapter)
    bind(adapter)
    adapter.main()


if __name__ == '__main__':
    main()

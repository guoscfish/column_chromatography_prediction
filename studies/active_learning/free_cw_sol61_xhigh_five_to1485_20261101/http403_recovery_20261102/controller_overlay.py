#!/usr/bin/env python3
"""Explicitly authorized exact HTTP 403 retries over unchanged frozen science."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ENTRY = Path(__file__).resolve()
REASON = "provider HTTP 403"
PREVIOUS = STUDY / "transport_recovery_20261102"
BASE_ENTRY = STUDY / "persistent_20261102_v2/frozen_runner.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def install(adapter):
    original = adapter.a.transient_http_reason

    def classify(reason):
        return 403 if reason == REASON else original(reason)

    adapter.a.transient_http_reason = classify


def bind(adapter):
    old_load, old_validate = adapter.load_base, adapter.validate
    old_retire = adapter.retire_worker

    def load():
        old_load()
        install(adapter)

    def validate():
        policy = json.loads((HERE / "authorization.json").read_text())
        if (policy["exact_retry_reason"] != REASON or
                policy["user_authorization"] != "直接开始重试啊，等什么呢" or
                policy["model"] != "gpt-6.1-sol" or policy["reasoning_effort"] != "xhigh" or
                policy["target_budget"] != 1485 or policy["maximum_seed_workers"] != 5 or
                policy["maximum_expensive_compute"] != 1):
            raise RuntimeError("HTTP403 retry authorization drift")
        for name, expected in policy["code_hashes"].items():
            if hashlib.sha256((STUDY / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("HTTP403 retry source binding mismatch")
        for name, expected in policy["preserved_fatal_receipts"].items():
            if hashlib.sha256((STUDY / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("HTTP403 original receipt binding mismatch")
        return old_validate()

    def retire(seed, pid, reason, **kwargs):
        command, entry = adapter.process_command(pid), adapter.ENTRY
        try:
            for prior in (PREVIOUS / "controller_overlay.py", BASE_ENTRY,
                          PREVIOUS / "upstream_h2_worker.py"):
                if str(prior) in command:
                    adapter.ENTRY = prior
                    break
            return old_retire(seed, pid, reason, **kwargs)
        finally:
            adapter.ENTRY = entry

    adapter.load_base, adapter.validate = load, validate
    adapter.retire_worker, adapter.ENTRY = retire, ENTRY


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("worker", "supervise", "status", "validate"):
        raise SystemExit("use worker, supervise, status or validate with existing authorization flags")
    adapter = module("unchanged_persistent_v2_403", BASE_ENTRY)
    previous = module("unchanged_h2_overlay_403", PREVIOUS / "controller_overlay.py")
    shim = module("unchanged_h2_shim_403", PREVIOUS / "upstream_h2_worker.py")
    previous.bind(adapter, shim)
    bind(adapter)
    adapter.main()


if __name__ == "__main__":
    main()

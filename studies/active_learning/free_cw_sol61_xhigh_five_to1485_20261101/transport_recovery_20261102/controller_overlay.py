#!/usr/bin/env python3
"""Preserve the frozen controller and apply only exact HTTP/2 recovery handling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ENTRY = Path(__file__).resolve()
BASE_ENTRY = STUDY / "persistent_20261102_v2/frozen_runner.py"


def bind(adapter, shim):
    original_load = adapter.load_base
    original_validate = adapter.validate
    original_transient = adapter.transient_stop
    original_retire = adapter.retire_worker

    def validate():
        policy = json.loads((HERE / "controller_authorization.json").read_text())
        for name, expected in policy["code_hashes"].items():
            if hashlib.sha256((STUDY / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("HTTP/2 recovery controller binding mismatch")
        return original_validate()

    def load():
        original_load()
        shim.install(adapter)

    def transient(state):
        return state.get("reason") == shim.REASON or original_transient(state)

    def retire(seed, pid, reason, **kwargs):
        # The unchanged watchdog checks the exact entry of adopted old workers.
        # Healthy requests still return before any signal, as in the frozen code.
        command = adapter.process_command(pid)
        entry = adapter.ENTRY
        try:
            for prior in (BASE_ENTRY, HERE / "upstream_h2_worker.py"):
                if str(prior) in command:
                    adapter.ENTRY = prior
                    break
            return original_retire(seed, pid, reason, **kwargs)
        finally:
            adapter.ENTRY = entry

    adapter.load_base = load
    adapter.validate = validate
    adapter.transient_stop = transient
    adapter.retire_worker = retire
    adapter.ENTRY = ENTRY


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("worker", "supervise", "status", "validate"):
        raise SystemExit("use worker, supervise, status or validate with the existing authorization flags")
    spec = importlib.util.spec_from_file_location("frozen_persistent_controller_h2_overlay", BASE_ENTRY)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    spec = importlib.util.spec_from_file_location("frozen_h2_transport_error_shim", HERE / "upstream_h2_worker.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    bind(adapter, shim)
    adapter.main()


if __name__ == "__main__":
    main()

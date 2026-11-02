#!/usr/bin/env python3
"""One-round recovery of an explicit upstream HTTP/2 transport failure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REASON = "provider terminal code: upstream_http2_stream_error"


def install(adapter):
    """Preserve original retry accounting, request construction and validators."""
    original_legacy = adapter.legacy_transient
    original_stream = adapter.m.live_stream

    def legacy(record):
        return (record.get("fatal_protocol_error") is True and record.get("reason") == REASON) or original_legacy(record)

    def stream(*args, **kwargs):
        import httpx
        try:
            return original_stream(*args, **kwargs)
        except adapter.m.FatalProtocolError as error:
            if str(error) != REASON:
                raise
            raise httpx.RemoteProtocolError(REASON) from error

    adapter.legacy_transient = legacy
    adapter.m.live_stream = stream


def main():
    # This adapter cannot dispatch a supervisor or change the scientific protocol.
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise SystemExit("only the existing one-round worker action is supported")
    policy = json.loads((HERE / "authorization.json").read_text())
    for name, expected in policy["code_hashes"].items():
        if hashlib.sha256((STUDY / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("transport recovery adapter binding mismatch")
    spec = importlib.util.spec_from_file_location("unchanged_persistent_v2_h2_recovery", STUDY / "persistent_20261102_v2/frozen_runner.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    original_load = adapter.load_base

    def load():
        original_load()
        install(adapter)

    adapter.load_base = load
    adapter.main()


if __name__ == "__main__":
    main()

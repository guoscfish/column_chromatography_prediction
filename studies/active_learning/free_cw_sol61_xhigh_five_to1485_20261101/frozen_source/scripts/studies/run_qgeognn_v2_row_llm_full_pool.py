#!/usr/bin/env python3
"""Prepare, validate, resume, and report the full-pool feedback LLM study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.qgeognn_al.active_learning_v2.full_pool_study import (METHODS, execute, prepare,
                                                                 refreeze_preselection, report, validate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "refreeze-preselection", "validate", "execute", "report"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--retry-selector", action="store_true",
                        help="Start a separately logged selector attempt after a failed/incomplete one")
    args = parser.parse_args()
    if args.action == "execute":
        result = execute(args.seed, args.method, retry_selector=args.retry_selector)
    else:
        result = {"prepare": prepare, "refreeze-preselection": refreeze_preselection,
                  "validate": validate, "report": report}[args.action]()
    print(json.dumps(result, indent=2, ensure_ascii=False))

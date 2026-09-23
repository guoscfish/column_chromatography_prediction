#!/usr/bin/env python3
"""Prepare, execute, freeze, and report same-state short-rollout branching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_same_state_matplotlib")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "2" if name == "OMP_NUM_THREADS" else "1")

from src.qgeognn_al.active_learning_v2 import same_state_branching_cw_hybrid as study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path, metavar="JUNIT_XML")
    action.add_argument("--selector-audit", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--selection-smoke", action="store_true")
    action.add_argument("--execute-seed", type=int)
    action.add_argument("--stage-check", type=int)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = study.prepare(args.prepare)
    elif args.selector_audit:
        from src.qgeognn_al.active_learning_v2.same_state_selector_audit import run
        result = run()
    elif args.validate:
        result = study.validate_seal()
    elif args.selection_smoke:
        result = study.selection_smoke()
    elif args.execute_seed is not None:
        result = study.execute_seed(args.execute_seed)
    elif args.stage_check is not None:
        result = study.stage_check(args.stage_check)
    elif args.freeze:
        result = study.global_freeze()
    else:
        result = study.reveal_test_and_report()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

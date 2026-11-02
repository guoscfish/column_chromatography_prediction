#!/usr/bin/env python3
"""Prepare, inspect, or explicitly execute the LCMD-L653 to IVR continuation."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "2")

from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_study as study
from src.qgeognn_al.active_learning_v2 import lcmd_to_ivr_runner as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path, metavar="JUNIT_XML")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--selection-smoke", action="store_true")
    action.add_argument("--execute-seed", type=int)
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = study.prepare(args.prepare)
    elif args.validate:
        result = study.validate_seal()
    elif args.selection_smoke:
        result = runner.selection_smoke()
        result = {"new_fits": 0, "new_test_evaluations": 0,
                  "checks": [{k: c[k] for k in ("seed", "status")} for c in result["checks"]]}
    elif args.execute_seed is not None:
        result = runner.execute_seed(args.execute_seed)
    elif args.run_all:
        result = runner.run_all()
    else:
        from src.qgeognn_al.active_learning_v2.lcmd_to_ivr_reporting import report
        result = report()
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "source_files")}, indent=2))


if __name__ == "__main__":
    main()

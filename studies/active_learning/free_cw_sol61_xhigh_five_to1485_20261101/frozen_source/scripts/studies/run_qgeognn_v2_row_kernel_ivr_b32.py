#!/usr/bin/env python3
"""Seal, execute, resume, or report the single matched Kernel-IVR extension."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_ivr_matplotlib")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "2")

from src.qgeognn_al.active_learning_v2 import ivr_study  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--execute-seed", type=int)
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--test-report", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.test_report is None:
            parser.error("--prepare requires --test-report")
        result = ivr_study.prepare(args.test_report)
    elif args.validate:
        seal = ivr_study.validate_seal()
        result = {"status": seal["status"], "created_at": seal["created_at"]}
    elif args.execute_seed is not None:
        result = ivr_study.execute_seed(args.execute_seed)
        result = {key: result[key] for key in ("seed", "status", "round_points")}
    elif args.run_all:
        result = ivr_study.run_all()
    else:
        result = ivr_study.report()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

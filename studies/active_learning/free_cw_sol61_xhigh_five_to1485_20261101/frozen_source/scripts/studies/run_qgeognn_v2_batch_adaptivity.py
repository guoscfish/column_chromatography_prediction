#!/usr/bin/env python3
"""Seal, run or report the matched 333+320 static/adaptive control."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_adaptivity_matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from src.qgeognn_al.active_learning_v2 import batch_adaptivity as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path, metavar="TEST_REPORT")
    actions.add_argument("--select", action="store_true")
    actions.add_argument("--execute-seed", type=int)
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = study.prepare(args.prepare)
    elif args.select:
        result = study.select_all()
    elif args.execute_seed is not None:
        result = study.execute_seed(args.execute_seed)
    elif args.run:
        result = study.run_all()
    else:
        result = study.reveal_and_report()
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

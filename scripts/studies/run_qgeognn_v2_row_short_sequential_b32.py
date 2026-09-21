#!/usr/bin/env python3
"""Prepare, execute, freeze, or report the 333--429 CW/Direction screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_short_sequential_matplotlib")

from src.qgeognn_al.active_learning_v2.short_sequential_runner import (  # noqa: E402
    execute_seed,
    finalize_pre_test,
    reveal_and_report,
)
from src.qgeognn_al.active_learning_v2.short_sequential_study import prepare, validate_prepared  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--execute-seed", type=int)
    action.add_argument("--finalize-pre-test", action="store_true")
    action.add_argument("--reveal-test-and-report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.validate:
        result = validate_prepared()
    elif args.execute_seed is not None:
        result = execute_seed(args.execute_seed)
    elif args.finalize_pre_test:
        result = finalize_pre_test()
    else:
        result = reveal_and_report()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

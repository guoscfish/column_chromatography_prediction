#!/usr/bin/env python3
"""Prepare, validate, or run the preregistered sequential B=32 study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_sequential_matplotlib")

from src.qgeognn_al.active_learning_v2.sequential_protocol import (  # noqa: E402
    prepare,
    seal_commit_a,
    validate_preflight,
)
from src.qgeognn_al.active_learning_v2.sequential_runner import (  # noqa: E402
    execute_pre_test,
    execute_pre_test_seed,
    finalize_pre_test,
    reveal_test_and_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--validate-preflight", action="store_true")
    action.add_argument("--seal-commit-a", action="store_true")
    action.add_argument("--execute-pre-test", action="store_true")
    action.add_argument("--execute-seed", type=int, metavar="CONFIRMATION_SEED")
    action.add_argument("--finalize-pre-test", action="store_true")
    action.add_argument("--reveal-test-and-report", action="store_true")
    parser.add_argument("--preregistration-commit")
    parser.add_argument("--test-report", type=Path)
    args = parser.parse_args()

    if args.prepare:
        result = prepare()
    elif args.validate_preflight:
        result = validate_preflight()
    elif args.seal_commit_a:
        if args.test_report is None:
            parser.error("--seal-commit-a requires --test-report PATH")
        result = seal_commit_a(args.test_report)
    else:
        if not args.preregistration_commit:
            parser.error("formal actions require --preregistration-commit COMMIT_A_SHA")
        if args.execute_pre_test:
            result = execute_pre_test(args.preregistration_commit)
        elif args.execute_seed is not None:
            result = execute_pre_test_seed(args.preregistration_commit, args.execute_seed)
        elif args.finalize_pre_test:
            result = finalize_pre_test(args.preregistration_commit)
        else:
            result = reveal_test_and_report(args.preregistration_commit)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

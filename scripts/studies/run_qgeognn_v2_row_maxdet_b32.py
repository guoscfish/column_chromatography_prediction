#!/usr/bin/env python3
"""Prepare, execute, freeze, and report the row Gradient-MaxDet study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_maxdet_matplotlib")

from src.qgeognn_al.active_learning_v2.maxdet_reporting import reveal_test_and_report  # noqa: E402
from src.qgeognn_al.active_learning_v2.maxdet_runner import (  # noqa: E402
    build_global_pre_test_freeze,
    execute_seed,
)
from src.qgeognn_al.active_learning_v2.maxdet_study import (  # noqa: E402
    CONFIRMATION_SEEDS,
    prepare,
    validate_pre_test_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--reprepare", action="store_true")
    action.add_argument("--validate-pre-test", action="store_true")
    action.add_argument("--execute-seed", type=int)
    action.add_argument("--execute-all", action="store_true")
    action.add_argument("--finalize-pre-test", action="store_true")
    action.add_argument("--reveal-test-and-report", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        result = prepare()
    elif args.reprepare:
        result = prepare(allow_existing_runtime=True)
    elif args.validate_pre_test:
        result = validate_pre_test_manifest()
    elif args.execute_seed is not None:
        result = execute_seed(args.execute_seed)
    elif args.execute_all:
        result = {str(seed): execute_seed(seed) for seed in CONFIRMATION_SEEDS}
    elif args.finalize_pre_test:
        result = build_global_pre_test_freeze()
    else:
        result = reveal_test_and_report()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

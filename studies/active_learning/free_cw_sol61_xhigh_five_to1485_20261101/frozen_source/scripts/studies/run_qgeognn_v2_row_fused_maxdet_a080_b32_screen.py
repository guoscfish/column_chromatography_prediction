#!/usr/bin/env python3
"""Run the alpha=.8 Fusion-MaxDet truncated developmental screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qgeognn_al.active_learning_v2.fused_maxdet_reporting import reveal_test_and_report  # noqa: E402
from src.qgeognn_al.active_learning_v2.fused_maxdet_runner import build_global_pre_test_freeze, execute_seed  # noqa: E402
from src.qgeognn_al.active_learning_v2.fused_maxdet_study import (  # noqa: E402
    A080_SCREEN_SPEC,
    prepare,
    validate_pre_test_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--validate-pre-test", action="store_true")
    action.add_argument("--execute-seed", type=int)
    action.add_argument("--execute-all", action="store_true")
    action.add_argument("--finalize-pre-test", action="store_true")
    action.add_argument("--reveal-test-and-report", action="store_true")
    args = parser.parse_args()
    spec = A080_SCREEN_SPEC
    if args.prepare:
        result = prepare(spec=spec)
    elif args.validate_pre_test:
        result = validate_pre_test_manifest(spec=spec)
    elif args.execute_seed is not None:
        result = execute_seed(args.execute_seed, spec=spec)
    elif args.execute_all:
        result = {str(seed): execute_seed(seed, spec=spec) for seed in spec.seeds}
    elif args.finalize_pre_test:
        result = build_global_pre_test_freeze(spec=spec)
    else:
        result = reveal_test_and_report(spec=spec)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare, run, freeze, or report CW-LCMD continuation 429->525."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_cw_525_matplotlib")
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_study import prepare, validate_prepared
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import execute_seed, finalize_pre_test, reveal_and_report
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_reporting import regenerate_from_frozen_results
def main():
    parser = argparse.ArgumentParser(description=__doc__); action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true"); action.add_argument("--validate", action="store_true"); action.add_argument("--execute-seed", type=int); action.add_argument("--finalize-pre-test", action="store_true"); action.add_argument("--reveal-test-and-report", action="store_true"); action.add_argument("--regenerate-report-only", action="store_true")
    args = parser.parse_args()
    result = prepare() if args.prepare else validate_prepared() if args.validate else execute_seed(args.execute_seed) if args.execute_seed is not None else finalize_pre_test() if args.finalize_pre_test else reveal_and_report() if args.reveal_test_and_report else regenerate_from_frozen_results()
    print(json.dumps(result, indent=2, default=str))
if __name__ == "__main__": main()

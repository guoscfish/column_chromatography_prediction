#!/usr/bin/env python3
"""Phase 1 preparation/smoke, or explicitly requested post-Commit-A execution."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")

from src.qgeognn_al.active_learning_v2.benchmark import execute_primary, execute_secondary, smoke
from src.qgeognn_al.active_learning_v2.benchmark_protocol import prepare
from src.qgeognn_al.active_learning_v2.diagnostics import audit_existing_study
from src.qgeognn_al.active_learning_v2.phase1 import seal_phase1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--audit-existing", action="store_true")
    action.add_argument("--smoke", type=Path, metavar="ISOLATED_DIRECTORY")
    action.add_argument("--execute-primary", action="store_true")
    action.add_argument("--execute-secondary", action="store_true")
    action.add_argument("--seal-phase1", action="store_true")
    parser.add_argument("--preregistration-commit")
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.audit_existing:
        audit_existing_study()
    elif args.smoke:
        print(json.dumps(smoke(args.smoke), indent=2))
    elif args.seal_phase1:
        if not args.test_report or not args.smoke_report:
            parser.error("--seal-phase1 requires --test-report and --smoke-report")
        print(json.dumps(seal_phase1(args.test_report, args.smoke_report), indent=2))
    elif not args.preregistration_commit:
        parser.error("formal execution requires --preregistration-commit COMMIT_A_SHA")
    elif args.execute_primary:
        execute_primary(args.preregistration_commit)
    else:
        execute_secondary(args.preregistration_commit)


if __name__ == "__main__":
    main()

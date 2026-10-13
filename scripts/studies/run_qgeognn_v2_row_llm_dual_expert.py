#!/usr/bin/env python3
"""Resumable Dual Expert selection, measured feedback and shared QGeoGNN training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
from src.qgeognn_al.active_learning_v2.dual_expert_transport import settings
from src.qgeognn_al.active_learning_v2.scientist_transport import check_transport
from src.qgeognn_al.active_learning_v2.scientist_study import exclusive_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "check-transport", "stage", "select", "advance", "run", "report"))
    parser.add_argument("--seed", type=int, choices=study.SEEDS, default=157)
    parser.add_argument("--method", choices=study.METHODS, default=study.METHOD)
    parser.add_argument("--round", type=int, choices=range(len(study.BUDGETS)-1), default=0)
    parser.add_argument("--rounds", type=int, choices=range(1, len(study.BUDGETS)), default=2,
                        help="number of acquisition/retraining cycles; default 2 (333 -> 397 labels)")
    parser.add_argument("--backend", choices=("codex_cli", "responses"), default="responses")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="xhigh")
    parser.add_argument("--from-revision", help="inherit an already frozen trajectory prefix")
    parser.add_argument("--start-round", type=int, choices=range(len(study.BUDGETS)-1), default=None,
                        help="continue from this completed round (2 means start from 397 labels)")
    parser.add_argument("--base-url", default="https://token4research.cn/v1")
    parser.add_argument("--revision", default=study.DEFAULT_REVISION)
    parser.add_argument("--status-interval", type=float, default=30.0)
    args = parser.parse_args()
    study.configure_revision(args.revision)
    config = settings(args.backend, args.model, args.base_url, args.reasoning_effort)
    if args.action == "run":
        result = study.run(args.seed, args.method, args.rounds, config, args.status_interval,
                           start_round=args.start_round, parent_revision=args.from_revision)
    elif args.action == "validate":
        result = study.validate()
    elif args.action == "check-transport":
        study.validate()
        result = check_transport(study.read(study.STUDY / "protocol.json")["transport"])
    elif args.action == "report":
        result = study.report(args.seed, args.method, args.rounds)
    else:
        with exclusive_lock(study.STUDY / ".operation.lock"):
            result = study.prepare(config) if args.action == "prepare" else getattr(study, args.action)(args.seed, args.method, args.round)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

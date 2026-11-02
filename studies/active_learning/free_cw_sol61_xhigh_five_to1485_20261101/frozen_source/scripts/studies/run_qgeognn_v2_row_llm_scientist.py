#!/usr/bin/env python3
"""V2: resumable one-command runner plus label-safe inspection actions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2.scientist_transport import settings, check_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "check-transport", "dry-run", "stage", "select", "advance", "run", "report", "trajectory-report"))
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--method", choices=study.METHODS)
    parser.add_argument("--round", type=int, choices=range(6), default=0)
    parser.add_argument("--backend", choices=("codex_cli", "responses"), default="responses")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--base-url", default="https://token4research.cn/v1")
    parser.add_argument("--revision", default=study.DEFAULT_REVISION,
                        help="isolated protocol revision directory name")
    parser.add_argument("--status-interval", type=float, default=30.0,
                        help="terminal heartbeat interval in seconds for run (default: 30)")
    args = parser.parse_args()
    study.configure_revision(args.revision)
    config = settings(args.backend, args.model, args.base_url)
    if args.action == "prepare":
        result = study.prepare(config)
    elif args.action == "check-transport":
        study.validate()
        result = check_transport(study.read(study.STUDY / "protocol.json")["transport"])
    elif args.action in ("validate", "report"):
        result = getattr(study, args.action)()
    elif args.action == "trajectory-report":
        if args.seed is None or args.method is None:
            parser.error("--seed and --method are required")
        result = study.trajectory_report(args.seed, args.method)
    elif args.action == "run":
        result = study.run(args.seed, args.method, config, status_interval=args.status_interval)
        if args.seed is not None:
            result = study.trajectory_report(args.seed, args.method)
    else:
        if args.seed is None or args.method is None:
            parser.error("--seed and --method are required")
        action = "stage" if args.action == "dry-run" else args.action
        result = getattr(study, action)(args.seed, args.method, args.round)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

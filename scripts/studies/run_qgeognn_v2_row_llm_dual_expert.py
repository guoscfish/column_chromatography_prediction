#!/usr/bin/env python3
"""Prepare or select round 0 for the independent Dual Expert study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2 import dual_expert_study as study
from src.qgeognn_al.active_learning_v2.dual_expert_transport import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "stage", "select", "advance", "report"))
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--method", choices=study.METHODS, default=study.METHOD)
    parser.add_argument("--round", type=int, choices=(0,), default=0)
    parser.add_argument("--backend", choices=("codex_cli", "responses"), default="codex_cli")
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--base-url", default="https://token4research.cn")
    args = parser.parse_args()
    config = settings(args.backend, args.model, args.base_url)
    if args.action == "prepare":
        result = study.prepare(config)
    elif args.action == "validate":
        result = study.validate()
    elif args.action == "stage":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.stage(args.seed, args.method, args.round)
    elif args.action == "select":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.select(args.seed, args.method, args.round)
    elif args.action == "advance":
        result = study.advance(args.seed, args.method, args.round)
    else:
        result = study.report()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

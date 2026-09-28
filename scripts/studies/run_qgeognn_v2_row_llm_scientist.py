#!/usr/bin/env python3
"""V2: stage is label-safe; select freezes only; advance trains exactly one batch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2.scientist_transport import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "dry-run", "stage", "select", "advance", "report"))
    parser.add_argument("--seed", type=int, choices=study.SEEDS)
    parser.add_argument("--method", choices=study.METHODS)
    parser.add_argument("--round", type=int, choices=range(6), default=0)
    parser.add_argument("--backend", choices=("codex_cli", "responses"), default="codex_cli")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://token4research.cn")
    args = parser.parse_args()
    if args.action == "prepare":
        result = study.prepare(settings(args.backend, args.model, args.base_url))
    elif args.action in ("validate", "report"):
        result = getattr(study, args.action)()
    else:
        if args.seed is None or args.method is None:
            parser.error("--seed and --method are required")
        action = "stage" if args.action == "dry-run" else args.action
        result = getattr(study, action)(args.seed, args.method, args.round)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

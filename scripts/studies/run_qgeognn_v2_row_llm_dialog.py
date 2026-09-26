#!/usr/bin/env python3
"""Prepare, resume, relay and report the audited keyless dialog study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.dialog_study import STUDY, METHODS, SEEDS, prepare, validate, execute, report
from src.qgeognn_al.active_learning_v2.dialog_bridge import bind, process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate", "execute", "bind", "process", "report"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--round", type=int, choices=range(6))
    parser.add_argument("--thread-id")
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()
    if args.action in ("bind", "process"):
        validate()
        directory = STUDY / f"selections/seed_{args.seed}/{METHODS[1]}/round_{args.round:02d}"
        result = bind(directory, args.thread_id, args.first) if args.action == "bind" else process(directory)
    elif args.action == "execute":
        result = execute(args.seed, METHODS[1])
    else:
        result = {"prepare": prepare, "validate": validate, "report": report}[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))

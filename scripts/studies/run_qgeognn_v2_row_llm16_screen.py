#!/usr/bin/env python3
"""Prepare, resume one arm, or report the CW16 + conversation-LLM16 screen."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.llm_screen import prepare, validate, execute, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'execute', 'report'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', choices=['cw16_random16', 'cw16_llm16'])
    args = parser.parse_args()
    result = execute(args.seed, args.method) if args.action == 'execute' else globals()[args.action]()
    print(json.dumps(result, indent=2))

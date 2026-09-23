#!/usr/bin/env python3
"""Frozen balanced portfolio: prepare, smoke, seeds 157/6101, freeze, report."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/qgeognn_portfolio_matplotlib')
from src.qgeognn_al.active_learning_v2.cw_ivr_portfolio_study import prepare,validate_prepared
from src.qgeognn_al.active_learning_v2.cw_ivr_portfolio_runner import execute_seed,finalize_pre_test,selection_smoke
from src.qgeognn_al.active_learning_v2.cw_ivr_portfolio_reporting import report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','validate','selection-smoke','finalize-pre-test','reveal-test-and-report'):
        group.add_argument('--'+name,action='store_true')
    group.add_argument('--execute-seed',type=int)
    args=parser.parse_args()
    if args.prepare: result=prepare()
    elif args.validate: result=validate_prepared()
    elif args.selection_smoke: result=selection_smoke()
    elif args.execute_seed is not None: result=execute_seed(args.execute_seed)
    elif args.finalize_pre_test: result=finalize_pre_test()
    else: result=report()
    print(json.dumps(result,indent=2,default=str),flush=True)


if __name__=='__main__':
    main()

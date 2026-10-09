#!/usr/bin/env python
"""Four-round, 24-fit exploratory extension to653; no LLM or test evaluation."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.cw_factorial_extension import main
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);main(p.parse_args().action)

#!/usr/bin/env python
"""Bounded CW factorial and offline feedback audit."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.cw_factorial_study import main
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'audit', 'run', 'report', 'offline'])
    main(p.parse_args().action)

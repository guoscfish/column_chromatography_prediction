#!/usr/bin/env python
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.cw_factorial_to1005 import main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run','report'])
    main(parser.parse_args().action)

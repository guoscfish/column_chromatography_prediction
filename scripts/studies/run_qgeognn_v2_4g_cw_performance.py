"""Fixed five-seed Center/Width performance study."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2.center_width_performance import execute, preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--artifact-repository", type=Path, required=True)
    parser.add_argument("--innovation-repository", type=Path, required=True)
    args = parser.parse_args()
    (preflight if args.preflight else execute)(args.artifact_repository, args.innovation_repository)

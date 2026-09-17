#!/usr/bin/env python3
"""Run the fixed, selection-only QGeoGNN-V2 row AL innovation screen."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_innovation_matplotlib")

from src.qgeognn_al.active_learning_v2.innovation_screen import execute_innovation_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-repository",
        type=Path,
        required=True,
        help="Read-only repository containing frozen development runtime artifacts.",
    )
    args = parser.parse_args()
    print(json.dumps(execute_innovation_screen(args.artifact_repository), indent=2))


if __name__ == "__main__":
    main()

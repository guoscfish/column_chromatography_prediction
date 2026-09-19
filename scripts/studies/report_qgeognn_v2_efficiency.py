#!/usr/bin/env python3
"""Re-evaluate published sequential results without training or label access."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_efficiency_matplotlib")

from src.qgeognn_al.active_learning_v2.efficiency_reporting import run_phase0

if __name__ == "__main__":
    print(json.dumps(run_phase0(), indent=2))

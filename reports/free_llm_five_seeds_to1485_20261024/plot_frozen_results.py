"""Compatibility entry for the active monitor; implementation lives in scripts/."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/report_free_lcmd.py'), run_name='__main__')

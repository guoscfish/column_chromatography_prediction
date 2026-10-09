#!/usr/bin/env python
"""Supervise the existing653 run, then execute the frozen1005 continuation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_653'
STUDY = ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005'


def state(value):
    path = STUDY/'pipeline_state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def command(name, *args):
    state({'status':'running','step':name,'args':args,'pid':os.getpid()})
    subprocess.run([sys.executable,str(ROOT/'scripts/studies'/name),*args],cwd=ROOT,check=True)


def main():
    with (STUDY/'pipeline.lock').open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state({'status':'running','step':'waiting_for_existing653_run','pid':os.getpid()})
        while json.loads((PRIOR/'RUN_STATE.json').read_text())['status'] != 'completed':
            with (PRIOR/'execution.lock').open('a+') as prior:
                try:
                    fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    # Re-read after acquiring: final state can change between checks.
                    if json.loads((PRIOR/'RUN_STATE.json').read_text())['status'] != 'completed':
                        raise RuntimeError('653 process stopped before completion; preserve reservations and inspect its log')
            time.sleep(30)
        command('run_cw_factorial_extension.py','report')
        command('audit_cw_factorial_extension.py')
        command('run_cw_factorial_to1005.py','run')
        command('run_cw_factorial_to1005.py','report')
        command('audit_cw_factorial_to1005.py')
        command('finalize_cw_factorial_to1005.py')
        state({'status':'completed','budget':1005,'new_fits_from525':90,'paid_llm_calls':0,'new_test_reveals':0})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state({'status':'blocked','error':repr(error),'pid':os.getpid()})
        raise

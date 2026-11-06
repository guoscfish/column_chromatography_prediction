"""Reproduce offline outputs and tests without any online execution capability."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RESULTS=HERE/'results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    environment=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        environment[key]='1'
    environment['PYTHONDONTWRITEBYTECODE']='1'
    protected=[HERE/'input_manifest.json',HERE/'configs/offline_plan.json']
    before_protected={str(p.relative_to(HERE)):digest(p) for p in protected}
    names=('metrics.json','fit_audits.json','signal_comparison.json','batch_information.json',
           'predictions.csv','residual_provenance.csv')
    before={name:digest(RESULTS/name) for name in names}
    logs=[]
    commands=[
        [sys.executable,'-B','-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
        [sys.executable,'-B',str(HERE/'offline.py')],
        [sys.executable,'-B',str(HERE/'mock_controller_demo.py')],
        [sys.executable,'-B',str(HERE/'verify_frozen_inputs.py')],
        [sys.executable,'-B',str(HERE/'write_offline_report.py')],
    ]
    for command in commands:
        completed=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True)
        text=completed.stdout+completed.stderr
        logs.append({'command':command,'returncode':completed.returncode})
        if 'unittest' in command:
            (RESULTS/'test_results.txt').write_text(text)
        if completed.returncode:
            print(text)
            raise RuntimeError('offline verification failed: '+str(command))
    after={name:digest(RESULTS/name) for name in names}
    after_protected={str(p.relative_to(HERE)):digest(p) for p in protected}
    if before!=after or before_protected!=after_protected:
        raise ValueError('frozen numerical output or source-plan drift')
    source_paths=sorted(HERE.glob('*.py'))+sorted((HERE/'configs').glob('*.json'))
    sources={str(p.relative_to(HERE)):digest(p) for p in source_paths}
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'all_commands_succeeded':True,'numerical_outputs_byte_identical':True,
            'source_manifest_and_plan_unchanged':True,'output_sha256':after,
            'source_sha256':sources,'commands':logs,
            'environment':{'python':sys.version,'numpy':version('numpy'),'scipy':version('scipy'),
                           'configured_numeric_threads':1,'python_executable':sys.executable},
            'historical_data_replay_requires_local_files':True,
            'scope':'Phase0 only; no model/API/oracle/test truth calls'}
    (RESULTS/'reproduction_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('all_commands_succeeded','numerical_outputs_byte_identical','source_manifest_and_plan_unchanged')},indent=2))


if __name__=='__main__':main()

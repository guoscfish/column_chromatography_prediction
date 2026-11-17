#!/usr/bin/env python3
"""Explicit offline reproduction entry point; historical experiments are read-only."""
from pathlib import Path
import argparse, json, os, subprocess, sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]

def call(script,*args):
    subprocess.run([sys.executable,'-B',str(script),*map(str,args)],cwd=ROOT,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=OUT/'config.json')
    ap.add_argument('--check',action='store_true');ap.add_argument('--recompute',action='store_true');args=ap.parse_args()
    if args.check:
        call(OUT/'verify.py','--check');return
    if not args.recompute:ap.error('Choose --check or --recompute explicitly')
    call(ROOT/'studies/active_learning/comprehensive_history_diagnostics_20261112/scripts/finalize.py','--check')
    if (OUT/'INPUT_MANIFEST.json').exists():call(OUT/'verify.py','--inputs-only')
    call(OUT/'inventory.py','--output',OUT)
    call(OUT/'hybrid_text.py','--extract-only')
    call(OUT/'collect.py','--config',args.config)
    call(OUT/'analyze.py','--config',args.config)
    call(OUT/'preferences.py','--states',OUT/'states.json.xz','--bootstrap','100','--negative-seeds','7001','7002','7003')
    call(OUT/'hybrid_text.py')
    call(OUT/'performance.py');call(OUT/'concentration.py');call(OUT/'figures.py');call(OUT/'write_report.py')
    test=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(OUT/'tests')],cwd=ROOT,capture_output=True,text=True)
    (OUT/'test_output.txt').write_text(test.stdout+test.stderr)
    (OUT/'test_results.json').write_text(json.dumps(dict(returncode=test.returncode,command='python -B -m pytest -q -p no:cacheprovider tests'),indent=2))
    test.check_returncode();call(OUT/'verify.py');call(OUT/'verify.py','--check')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Reproduce only offline diagnostics; never start experiments or query labels."""
from pathlib import Path
import argparse,json,os,subprocess,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]

def call(script,*args):
    subprocess.run([sys.executable,'-B',str(script),*map(str,args)],cwd=ROOT,check=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--recompute',action='store_true')
    ap.add_argument('--config',type=Path,default=OUT/'config.json');args=ap.parse_args()
    if args.check:call(OUT/'verify.py','--check');return
    if not args.recompute:ap.error('Choose --check or --recompute explicitly')
    cfg=json.loads(args.config.read_text())
    call(ROOT/'studies/active_learning/llm_decision_behavior_audit_20261116/run.py','--check')
    if (OUT/'INPUT_MANIFEST.json').exists():call(OUT/'verify.py','--inputs-only')
    call(OUT/'model_learnability.py','--config',args.config)
    call(OUT/'counterfactual.py')
    call(OUT/'phase_protocol.py','--curves',ROOT/cfg['historical_curves'],'--plots')
    call(OUT/'llm_reproducibility.py','--snapshot',ROOT/cfg['historical_states'])
    call(OUT/'llm_prefix_diagnostic.py','--snapshot',ROOT/cfg['historical_states'])
    call(OUT/'model_report.py');call(OUT/'synthesize.py');call(OUT/'plot_llm_mechanism.py')
    call(OUT/'verify_counterfactual.py');call(OUT/'verify_response_numeric.py')
    test=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(OUT/'tests')],cwd=ROOT,capture_output=True,text=True)
    (OUT/'test_output.txt').write_text(test.stdout+test.stderr)
    (OUT/'test_results.json').write_text(json.dumps(dict(returncode=test.returncode,command='python -B -m pytest -q -p no:cacheprovider tests'),indent=2))
    test.check_returncode();call(OUT/'verify.py');call(OUT/'verify.py','--check')

if __name__=='__main__':main()

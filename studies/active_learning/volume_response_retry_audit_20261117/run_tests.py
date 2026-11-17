#!/usr/bin/env python3
"""Run isolated study checks and record their exit code without fitting."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}
command=[sys.executable,'-B','-m','pytest','-q','--disable-warnings','-p','no:cacheprovider',str(OUT/'tests')]
started=time.time()
with (OUT/'test_stdout.txt').open('w') as f:
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
record=dict(command=command,exit_code=result.returncode,seconds=time.time()-started,
    tests_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'tests').glob('test_*.py'))},
    stdout_sha256=hashlib.sha256((OUT/'test_stdout.txt').read_bytes()).hexdigest(),new_fits=0,new_llm_calls=0)
(OUT/'test_result.json').write_text(json.dumps(record,indent=2)+'\n')
print((OUT/'test_stdout.txt').read_text()[-4000:]);sys.exit(result.returncode)

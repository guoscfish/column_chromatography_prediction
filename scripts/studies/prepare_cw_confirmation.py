#!/usr/bin/env python
"""Read-only next-stage eligibility/unused-seed preflight; no training or LLM."""
from pathlib import Path
import argparse
import json
import re
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate',required=True,choices=['cw_maxdet','cw_ivr'])
p.add_argument('--seeds',required=True,type=int,nargs=5)
a=p.parse_args()
s=ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_factorial'
d=json.loads((s/'decision.json').read_text())
if d['status']!='completed':sys.exit('Blocked: development matrix incomplete; no next-stage freeze.')
gate=next(x for x in d['gates'] if x['selector']==a.candidate.removeprefix('cw_'))
if not gate['eligible']:sys.exit('Blocked: this selector did not meet preregistered validation threshold.')
if len(set(a.seeds))!=5 or min(a.seeds)<0:sys.exit('Require five unique nonnegative seeds.')
seen=set();sources=[]
for parent in (ROOT/'studies',ROOT/'experiments'):
 for path in parent.rglob('*'):
  if path.name=='__pycache__':continue
  matches=re.findall(r'(?:seed[_-]?)(\d+)',path.name)
  if matches:
   seen.update(map(int,matches));sources.append(str(path.relative_to(ROOT)))
conflict=seen&set(a.seeds)
if conflict:sys.exit(f'Blocked: seeds already present in historical artifact names: {sorted(conflict)}')
# Also inspect protocol seed lists without loading metrics, labels or payloads.
def seed_values(x,key=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from seed_values(v,k)
 elif isinstance(x,list):
  for v in x:yield from seed_values(v,key)
 elif 'seed' in key.lower() and isinstance(x,int):yield x
for parent in (ROOT/'studies',ROOT/'experiments'):
 for path in parent.rglob('protocol.json'):
  found=set(seed_values(json.loads(path.read_text())))&set(a.seeds)
  if found:sys.exit(f'Blocked: seeds {sorted(found)} occur in {path.relative_to(ROOT)}')
print(json.dumps({'status':'eligible_preflight_only','candidate':a.candidate,'proposed_seeds':a.seeds,
 'uniform_budgets':list(range(333,526,32)),'strong_controls':['matched_raw_selector','cw_lcmd','whitening_lcmd','random32'],
 'decision_sha256':hashlib.sha256((s/'decision.json').read_bytes()).hexdigest(),
 'no_training_started':True,'no_paid_calls':True,'next_action':'freeze independent confirmation protocol, exact splits and budget before execution; no automatic extension to813/1005'},indent=2))

import sys,time,json,subprocess,os
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));import resume_routed_transport_20261016 as routed
m=routed.m;routed.guard()
assert routed.sha(__file__)==m.read(P/'healthy_adoption_manifest.json')['source_sha256']
def active():
 result=set()
 for l in subprocess.check_output(['ps','-axo','pid,stat,args'],text=True).splitlines():
  parts=l.strip().split(None,2)
  if len(parts)!=3 or parts[1].startswith('Z') or not parts[2].startswith(sys.executable) or str(P) not in parts[2] or '--worker' not in parts[2]:continue
  args=parts[2].split();i=args.index('--worker');result.add((int(args[i+1]),args[i+2]))
 return result
while not (P/'result.json').exists():
 live=active();healthy=set()
 if (P/'routed_transport_events.jsonl').exists():
  for l in (P/'routed_transport_events.jsonl').read_text().splitlines():
   try:r=json.loads(l)
   except ValueError:continue
   if r['action']=='complete':healthy.add(r['seed'])
 for seed in (157,6101):
  if seed not in healthy or (P/f'quota_seed_{seed}.json').exists() or (P/f'transport_circuit_seed_{seed}.json').exists():continue
  for method in (m.HYBRID,m.FREE):
   if (seed,method) in live or (P/f'{seed}_{method}_result.json').exists():continue
   if any((m.directory(seed,method,r)/'rejected_turn_02.json').exists() and not (m.directory(seed,method,r)/'batch_freeze.json').exists() for r in range(30,34)):continue
   block=P/f'adoption_block_{seed}_{method}.json'
   if block.exists():continue
   routed.guard();log=P/f'adopted_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
   with log.open('ab') as out:c=subprocess.Popen([sys.executable,'-u',str(P/'resume_routed_diagnostic_20261017.py'),'--worker',str(seed),method],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
   block.write_text(json.dumps({'pid':c.pid,'time':time.time(),'policy':'one safe adoption after service produced completed replies; no repeated restart from exit'})+'\n')
   print(json.dumps({'action':'adopted_missing_worker','seed':seed,'method':method,'pid':c.pid,'log':str(log)}),flush=True)
 time.sleep(10)

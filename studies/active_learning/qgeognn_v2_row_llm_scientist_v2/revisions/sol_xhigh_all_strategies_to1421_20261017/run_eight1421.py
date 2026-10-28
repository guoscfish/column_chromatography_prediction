import os,sys,json,time,subprocess,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));import resume_routed_transport_20261016 as routed
m=routed.m
routed.guard()
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==m.read(P/'execution_manifest.json')['coordinator_sha256']
with m.s.exclusive_lock(P/'coordinator.lock'):
 children={}
 for seed in (157,6101):
  for method in m.METHODS:
   if (P/f'{seed}_{method}_result.json').exists():continue
   path=P/f'worker_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
   entry='resume_routed_transport_20261016.py' if method in (m.HYBRID,m.FREE) else 'continue_eight_to1421.py'
   with path.open('ab') as log:c=subprocess.Popen([sys.executable,'-u',str(P/entry),'--worker',str(seed),method],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=os.environ.copy())
   children[seed,method]=c
 launch={'time':time.time(),'coordinator_pid':os.getpid(),'worker_pids':{f'{s}:{method}':c.pid for (s,method),c in children.items()}}
 (P/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
 while any(c.poll() is None for c in children.values()):time.sleep(5)
 exits={f'{s}:{method}':c.returncode for (s,method),c in children.items()}
 (P/'worker_exit_codes.json').write_text(json.dumps(exits,indent=2)+'\n')
 if any(exits.values()):print('INCOMPLETE: inspect worker logs, quota/circuit/budget; no automatic duplicate restart',exits,flush=True);sys.exit(79)
 routed.guard()
 for seed in (157,6101):
  for method in m.METHODS:
   result=m.read(P/f'{seed}_{method}_result.json')
   assert result['final_labels']==1421 and len(result['selected_ids'])==len(set(result['selected_ids']))==1088 and len(result['additional_ids'])==len(set(result['additional_ids']))==128
   for round_index in range(31,35):
    freeze=m.read(P/f'runtime/seed_{seed}/{method}/round_{round_index:02d}/prediction_freeze.json')
    for k in ('checkpoint','prediction'):assert m.sha(freeze[k+'_path'])==freeze[k+'_sha256']
 m.once(P/'result.json',{'status':'ALL_EIGHT_COMPLETE1421','test_evaluation_performed':False,'completed_time':time.time()})
 print('ALL_EIGHT_COMPLETE1421',flush=True)

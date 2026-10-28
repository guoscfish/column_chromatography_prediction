"""Read-only local proxy uptime evidence; no credentials or provider requests."""
from pathlib import Path
import subprocess,time,json
P=Path(__file__).resolve().parent
last=None
while not (P/'result.json').exists():
 r=subprocess.run(['lsof','-nP','-t','-iTCP:3066','-sTCP:LISTEN'],capture_output=True,text=True)
 pids=tuple(sorted(int(x) for x in r.stdout.split() if x.isdigit()))
 if pids!=last:
  record={'time':time.time(),'proxy_listener_pids':pids,'available':bool(pids)}
  with (P/'proxy_uptime.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True);last=pids
 time.sleep(10)

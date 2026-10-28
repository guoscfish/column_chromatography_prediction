import sys,time,json,os
from pathlib import Path
import resume_routed_transport_20261016 as routed
P=routed.P
assert routed.sha(__file__)==routed.m.read(P/'diagnostic_transport_manifest.json')['runner_sha256']
original=routed.raw_call
secrets=[Path(f).read_text().strip() for f in routed.m.guard()['credential_files'].values()]
def diagnosed_call(messages,config):
 try:return original(messages,config)
 except Exception as error:
  chain=[];e=error;seen=set()
  while e is not None and id(e) not in seen:
   seen.add(id(e));message=str(e)
   for secret in secrets:message=message.replace(secret,'[REDACTED]')
   chain.append({'type':type(e).__name__,'message':message[:1500]})
   e=e.__cause__ if e.__cause__ is not None else e.__context__
  with (P/'transport_exception_chains.jsonl').open('a') as out:out.write(json.dumps({'time':time.time(),'seed':routed.m.SEED,'pid':os.getpid(),'chain':chain})+'\n')
  raise
routed.raw_call=diagnosed_call
sys.exit(routed.main())

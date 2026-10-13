"""After current request probe freezes749, restore request serialization for157."""
import os,signal,subprocess,time,json,sys
from pathlib import Path
OLD=Path(__file__).resolve().parent
NEW=OLD.parent/'free_sol_xhigh_continuation_20261012'
pid=int(sys.argv[1]);target=NEW/'runtime/seed_157/free_llm32_scientist_v2/round_13/prediction_freeze.json'
while not target.exists():
    state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
    if not state or state.startswith('Z'):raise RuntimeError('Probeworker exited before749; inspect logs')
    time.sleep(.2)
try:os.kill(pid,signal.SIGSTOP)
except ProcessLookupError:pass
import hashlib
f=json.loads(target.read_text())
for kind in ('checkpoint','prediction'):assert hashlib.sha256(Path(f[kind+'_path']).read_bytes()).hexdigest()==f[kind+'_sha256']
try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
except ProcessLookupError:pass
for _ in range(100):
    state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
    if not state or state.startswith('Z'):break
    time.sleep(.1)
else:raise RuntimeError('Oldprobeworker stillactive')
log=NEW/'observed_serial_seed157.log'
with log.open('ab') as out:
    child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',str(OLD/'run_free157_observed_serial.py')],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
with (NEW/'recovery_monitor.jsonl').open('a') as out:out.write(json.dumps({'action':'probe_to_serial_after749','old_pid':pid,'new_pid':child.pid,'log':str(log)})+'\n')
print('seed157749 frozen; serial observed worker PID',child.pid,flush=True)

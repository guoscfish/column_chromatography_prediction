"""Prioritize the oldest frozen Free157 batch at unchanged two-fit slots.
Only lock polling interval changes from1second to0.01second. All scientific
functions, messages, retries, seed locks, initialization and fit threads unchanged.
"""
import sys,inspect,runpy,time,fcntl
from pathlib import Path
from contextlib import contextmanager
import resume_routed_transport_20261016 as routed
P=routed.P;m=routed.m
record=m.read(P/'priority_free157_manifest.json')
assert routed.sha(__file__)==record['runner_sha256']
routed.guard()
i=sys.argv.index('--worker');assert (int(sys.argv[i+1]),sys.argv[i+2])==(157,m.FREE)
original=inspect.getsource(m.training_slot)
import hashlib
assert hashlib.sha256(original.encode()).hexdigest()==record['original_allocator_source_sha256']
assert original.count('time.sleep(1)')==1 and 'range(2)' in original
namespace={'contextmanager':contextmanager,'HERE':P,'blocking_lock':m.blocking_lock,'fcntl':fcntl,'time':time}
exec(compile(original.replace('time.sleep(1)','time.sleep(0.01)'),'<audited lock polling interval only>','exec'),namespace)
m.training_slot=namespace['training_slot']
runpy.run_path(str(P/'resume_routed_diagnostic_20261017.py'),run_name='__main__')

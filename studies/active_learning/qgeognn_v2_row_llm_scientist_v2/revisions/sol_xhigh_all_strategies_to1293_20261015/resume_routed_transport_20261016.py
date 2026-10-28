"""Process-local explicit proxy, stream telemetry, and bounded transport circuit breaker.
No scientific request, model, budget, or source file is changed.
"""
import resume_upstream_http2_compatible as adapter
import os,sys,time,json,threading,hashlib,subprocess
from pathlib import Path
import httpx,openai
m=adapter.runner;P=m.HERE;PROXY='http://127.0.0.1:3066'
OriginalOpenAI=openai.OpenAI
raw_call=m.base.raw_call
active=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_event(action,**fields):
    item={'time':time.time(),'action':action,'seed':m.SEED,'pid':os.getpid(),**fields}
    with (P/'routed_transport_events.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
def guard():
    record=m.read(P/'routed_transport_manifest_20261016.json')
    assert sha(__file__)==record['runner_sha256']
    assert sha(P/'resume_upstream_http2_compatible.py')==record['adapter_sha256']
    adapter.guard()
    assert PROXY==record['proxy'] and record['max_consecutive_transport_failures']==2

class TrackedStream(httpx.SyncByteStream):
    def __init__(self,inner,state):self.inner=inner;self.state=state;self.buffer=b''
    def __iter__(self):
        for chunk in self.inner:
            st=self.state;st['bytes_received']+=len(chunk);st['last_byte_time']=time.time()
            self.buffer+=chunk
            while b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1)
                if not line.startswith(b'data:'):continue
                try:data=json.loads(line[5:].strip())
                except (ValueError,UnicodeDecodeError):continue
                typ=data.get('type','unknown');counts=st['events'];counts[typ]=counts.get(typ,0)+1
                if counts[typ]==1:
                    response=data.get('response') or {}
                    write_event('stream_event',event_type=typ,elapsed=time.time()-st['started'],response_id=response.get('id'),served_model=response.get('model'))
            if len(self.buffer)>8_000_000:raise RuntimeError('Diagnostic stream line exceeded safe buffer; no partial response accepted')
            yield chunk
    def close(self):self.inner.close()

def response_hook(response):
    if active is None:return
    active['http_status']=response.status_code
    write_event('response_headers',status=response.status_code,http_version=response.http_version,cf_ray=response.headers.get('cf-ray'),request_id=response.headers.get('x-request-id'),elapsed=time.time()-active['started'])
    response.stream=TrackedStream(response.stream,active)

class RoutedOpenAI(OriginalOpenAI):
    def __init__(self,*args,**kwargs):
        assert str(kwargs.get('base_url','')).rstrip('/')=='https://token4research.cn/v1'
        assert 'http_client' not in kwargs
        kwargs['http_client']=httpx.Client(proxy=PROXY,trust_env=False,timeout=kwargs.get('timeout',600),event_hooks={'response':[response_hook]})
        super().__init__(*args,**kwargs)

def monitored_call(messages,config):
    global active
    seed=m.SEED;circuit=P/f'transport_circuit_seed_{seed}.json'
    if circuit.exists():write_event('circuit_open_request_suppressed');raise SystemExit(75)
    failures_path=P/f'routed_failure_count_seed_{seed}.json'
    active={'started':time.time(),'bytes_received':0,'events':{},'last_byte_time':None}
    state=active;stop=threading.Event()
    encoded=json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode()
    write_event('request',proxy=PROXY,message_sha256=hashlib.sha256(encoded).hexdigest(),message_bytes=len(encoded),model=config['model'],effort=config['reasoning_effort'])
    def heartbeat():
        while not stop.wait(30):
            write_event('stream_heartbeat',elapsed=time.time()-state['started'],bytes_received=state['bytes_received'],last_byte_time=state['last_byte_time'],events=dict(state['events']),http_status=state.get('http_status'))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        answer,provenance=raw_call(messages,config)
        failures_path.write_text(json.dumps({'consecutive_failures':0,'time':time.time()})+'\n')
        write_event('complete',elapsed=time.time()-state['started'],bytes_received=state['bytes_received'],served_model=provenance.get('served_model'),response_id=provenance.get('response_id'))
        return answer,provenance
    except Exception as error:
        if any(x in str(error) for x in ('DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','insufficient_quota')):raise
        if isinstance(error,(openai.APIConnectionError,openai.APITimeoutError,httpx.TransportError,openai.InternalServerError,openai.RateLimitError)):
            previous=m.read(failures_path) if failures_path.exists() else {};count=previous.get('consecutive_failures',0)+1
            item={'time':time.time(),'seed':seed,'consecutive_failures':count,'error_type':type(error).__name__,'message_sha256':hashlib.sha256(encoded).hexdigest(),'events':dict(state['events']),'bytes_received':state['bytes_received']}
            failures_path.write_text(json.dumps(item,indent=2)+'\n');write_event('transport_failure',**{k:v for k,v in item.items() if k not in ('time','seed')})
            if count>=2:
                circuit.write_text(json.dumps({**item,'status':'STOPPED_AFTER_TWO_FAILURES_REQUIRES_DIAGNOSIS'},indent=2)+'\n');raise SystemExit(75) from error
        raise
    finally:stop.set();thread.join(timeout=1);active=None

def run_worker(seed,method):
    guard();openai.OpenAI=RoutedOpenAI;m.base.raw_call=monitored_call
    m.previous_extra.stream.classify=adapter.classify
    return m.worker(seed,method)

def main():
    guard()
    if '--check' in sys.argv:
        print('ROUTED_TRANSPORT_PREFLIGHT_PASS: all frozen science/source/prefix guards; proxy only, two-failure circuit.',flush=True);return 0
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');return run_worker(int(sys.argv[i+1]),sys.argv[i+2])
    rows=subprocess.check_output(['ps','-axo','pid,args'],text=True).splitlines()
    assert not any(str(P) in x and '--worker' in x and str(sys.executable) in x for x in rows),'Existing experiment worker; refusing duplicate'
    children={}
    for seed in (157,6101):
        assert not (P/f'quota_seed_{seed}.json').exists(),'Quota remains blocked'
        assert not (P/f'transport_circuit_seed_{seed}.json').exists(),'Circuit remains open'
        for method in (m.HYBRID,m.FREE):
            if (P/f'{seed}_{method}_result.json').exists():continue
            path=P/f'routed_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
            with path.open('ab') as out:
                c=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(seed),method],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
            children[(seed,method)]=c;write_event('worker_started',target_seed=seed,method=method,child_pid=c.pid,log=str(path))
    while children:
        for (seed,method),child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            write_event('worker_exited',target_seed=seed,method=method,child_pid=child.pid,exit_code=rc);del children[(seed,method)]
            if rc==78:
                for (s,_),other in children.items():
                    if s==seed and other.poll() is None:other.terminate()
        time.sleep(3)
    return 0
if __name__=='__main__':sys.exit(main())

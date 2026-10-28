"""Preserve scientific requests; record stream terminal errors before SDK parsing."""
import resume1165_compatible as compatible
from continue_eight_to1165 import *
import httpx
from openai import APIConnectionError
from openai.lib.streaming.responses._responses import ResponseStreamState
original=compatible.original

def classify(code,message):
    text=str(code)+' '+str(message)
    if any(x in text for x in ('DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','insufficient_quota')):return 'quota'
    if code in ('server_error','internal_server_error','rate_limit_exceeded','overloaded','timeout','request_timeout','upstream_error','service_unavailable'):return 'transient'
    return 'decision'

old_handle=ResponseStreamState.handle_event

def handle_event(self,event):
    if event.type in ('response.failed','error'):
        response=getattr(event,'response',None);error=getattr(response,'error',None)
        code=getattr(error,'code',None) or getattr(event,'code',None)
        message=str(getattr(error,'message',None) or getattr(event,'message',''))
        secret_file=os.environ.get('SCIENTIST_API_KEY_FILE')
        if secret_file:
            secret=Path(secret_file).read_text().strip()
            if secret:message=message.replace(secret,'[REDACTED]')
        category=classify(code,message)
        record={'time':time.time(),'pid':os.getpid(),'seed':original.SEED,'event_type':event.type,'response_id':getattr(response,'id',None),'code':code,'message':message[:1000],'classification':category}
        with (HERE/'stream_terminal_errors.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if category=='quota':
            once(HERE/f'quota_seed_{original.SEED}.json',{'seed':original.SEED,'status':'QUOTA_EXHAUSTED','time':time.time(),'terminal_error':record});raise SystemExit(78)
        if category=='transient':raise APIConnectionError(message='Stream terminal failure: '+str(code)+': '+message,request=httpx.Request('POST',CONFIG['base_url']+'/responses'))
        raise RuntimeError('Provider terminal failure requires diagnosis: '+str(code)+': '+message)
    return old_handle(self,event)

def main():
    record=read(HERE/'stream_guard_manifest.json')
    assert sha(__file__)==record['adapter_sha256']
    assert sha(HERE/'resume1165_compatible.py')==record['previous_adapter_sha256']
    compatible.guard_adapter()
    if '--check' in sys.argv:
        assert classify('server_error','backend failed')=='transient'
        assert classify('USAGE_LIMIT_EXCEEDED','DAILY_LIMIT_EXCEEDED')=='quota'
        assert classify('context_length_exceeded','too long')=='decision'
        print('STREAM_GUARD_PREFLIGHT_PASS: transient/quota/nonrecoverable classification; frozen code and requests unchanged.',flush=True);return 0
    ResponseStreamState.handle_event=handle_event
    assert '--worker' in sys.argv
    i=sys.argv.index('--worker');seed=int(sys.argv[i+1]);method=sys.argv[i+2]
    original.activate()
    return original.worker(seed,method)

if __name__=='__main__':sys.exit(main())

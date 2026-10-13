"""User-authorized seed157 request probe/continuation; unchanged request, stream metadata audit."""
import sys,time
import resume_free_sol_full_parallel as runner
from openai import OpenAI
from src.qgeognn_al.active_learning_v2.scientist_transport import provider_key

def observed_call(messages,config):
    started=time.monotonic();last=started;events=0;characters=0;completed=False
    runner.s._progress({'action':'observed_request_started','seed':157,'model':config['model'],'effort':config['reasoning_effort']})
    with OpenAI(api_key=provider_key(config),base_url=config['base_url'],max_retries=0,timeout=600) as client:
        with client.responses.stream(model=config['model'],input=messages,reasoning={'effort':config['reasoning_effort']},tools=[],store=False) as stream:
            for event in stream:
                events+=1
                delta=getattr(event,'delta',None)
                if isinstance(delta,str):characters+=len(delta)
                now=time.monotonic()
                if events==1 or now-last>=30 or event.type in ('response.completed','response.failed','response.incomplete'):
                    runner.s._progress({'action':'observed_stream_event','seed':157,'event_type':event.type,'event_count':events,'delta_characters':characters,'elapsed_seconds':round(now-started,1)})
                    last=now
                if event.type=='response.completed':completed=True
            response=stream.get_final_response()
    if response.status!='completed' or not completed:raise RuntimeError("Didn't receive a `response.completed` event.")
    if response.model!=config['model'] and not response.model.startswith(config['model']+'-'):raise RuntimeError('provider returned different model; no fallback')
    if any(item.type not in ('message','reasoning') for item in response.output):raise RuntimeError('native tool output invalidates response')
    return response.output_text,{'response_id':response.id,'served_model':response.model,'usage':response.usage.model_dump(mode='json') if response.usage else None,'native_tool_calls':0}

if __name__=='__main__':
    runner.base.transport.responses_call=observed_call
    sys.exit(runner.worker(157))

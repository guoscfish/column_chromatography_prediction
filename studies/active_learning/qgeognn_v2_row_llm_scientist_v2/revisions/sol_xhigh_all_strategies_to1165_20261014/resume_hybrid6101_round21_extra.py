"""Human-authorized fourth reply only for seed6101 hybrid round21."""
import resume1165_stream_guard as stream
from continue_eight_to1165 import *
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import DIRECT_PROMPT, encode_catalog, validate_direct
from src.qgeognn_al.active_learning_v2.protocol import stable_hash
original=stream.original
D=HERE/'selections/seed_6101'/HYBRID/'round_21'
old_selector=original.selector

def messages_for(packet,catalog):
    assert (packet['seed'],packet['method'],packet['round'],packet['active_label_count'])==(6101,HYBRID,21,1005)
    assert all(packet.get(k)==v for k,v in encode_catalog(catalog).items())
    assert packet['packet_sha256']==stable_hash({k:v for k,v in packet.items() if k!='packet_sha256'})
    messages=[{'role':'system','content':DIRECT_PROMPT},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
    for i in range(3):
        receipt=read(D/f'turn_{i:02d}.json')
        assert receipt['request_sha256']==stable_hash({'messages':messages,'config':CONFIG}),i
        messages.extend([{'role':'assistant','content':receipt['answer']},{'role':'user','content':json.dumps(read(D/f'rejected_turn_{i:02d}.json'),ensure_ascii=False)}])
    assert '43758e84c67aabf44a9f' in catalog.observed
    assert '43758e84c67aabf44a9f' not in catalog.candidates
    detail={'authorization':'User explicitly authorized exactly one additional gpt-6.1-sol/xhigh correction for seed6101 hybrid round21 (1005 to 1037 labels) on 2026-10-15. This is a protocol deviation. Every other round retains three logical attempts.',
      'instruction':'43758e84c67aabf44a9f is ALREADY MEASURED in the observed table, NOT an eligible candidate or a pending CW ID. Remove it from choices and select a replacement yourself from the eligible candidates table. Return the complete original-schema final JSON with exactly 16 distinct eligible candidate choices and the same packet_sha256. Verify every choice against the candidates table and every evidence reference against the observed table. The fixed 16 pending CW IDs remain unchanged. This is the sole additional logical correction; do not invent or locally repair IDs.'}
    messages.append({'role':'user','content':json.dumps(detail,ensure_ascii=False)})
    return messages,detail

def extra_selector(packet,catalog,path,config,call=None):
    if Path(path)!=D:return old_selector(packet,catalog,path,config,call=call)
    messages,detail=messages_for(packet,catalog)
    catalog.viewed.update(catalog.candidates)
    once(D/'authorized_extra_correction.json',{**detail,'original_call_budget':3,'additional_logical_calls':1,'packet_sha256':packet['packet_sha256'],'wrapper_sha256':sha(__file__),'historical_transcript_hashes':{f'turn_{i:02d}.json':sha(D/f'turn_{i:02d}.json') for i in range(3)}})
    request_hash=stable_hash({'messages':messages,'config':CONFIG})
    once(D/'extra_correction_request.json',{'request_sha256':request_hash,'feedback':detail})
    target=D/'turn_03.json'
    failures=0
    while not target.exists():
        try:
            answer,provenance=transport._call_with_retries(messages,CONFIG,call or transport.responses_call,D,3,request_hash)
            once(target,{'request_sha256':request_hash,'answer':answer,'provenance':provenance})
        except Exception as error:
            # Reissue the same logical request only for transport failures; never another correction.
            recoverable={'APIConnectionError','APITimeoutError','ConnectTimeout','ReadTimeout','ConnectError','RemoteProtocolError','InternalServerError','RateLimitError'}
            cause=error.__cause__ or error
            if type(cause).__name__ not in recoverable and 'response.completed' not in str(cause):raise
            failures+=1;delay=min(60,10*failures)
            s._progress({'action':'authorized_extra_transport_retry','seed':6101,'method':HYBRID,'round':21,'seconds':delay,'request_sha256':request_hash})
            time.sleep(delay)
    receipt=read(target);assert receipt['request_sha256']==request_hash
    try:result=validate_direct(json.loads(receipt['answer']),catalog,packet)
    except (ValueError,TypeError,KeyError) as error:
        once(D/'rejected_turn_03.json',{'error':str(error),'additional_budget_exhausted':True});raise
    once(D/'selection.json',result)
    return result

def guard():
    m=read(HERE/'round21_extra_authorization_manifest.json')
    assert sha(__file__)==m['wrapper_sha256']
    assert sha(HERE/'resume1165_stream_guard.py')==m['stream_guard_sha256']
    stream_record=read(HERE/'stream_guard_manifest.json')
    assert sha(HERE/'resume1165_stream_guard.py')==stream_record['adapter_sha256']
    assert sha(HERE/'resume1165_compatible.py')==stream_record['previous_adapter_sha256']
    stream.compatible.guard_adapter()
    messages_for(read(D/'input.json'),s.ReadOnlyCatalog(**read(D/'catalog.json')))
    freeze=read(SOURCE/'runtime/seed_6101'/HYBRID/'round_21/prediction_freeze.json')
    for k in ('checkpoint','prediction'):assert sha(freeze[k+'_path'])==freeze[k+'_sha256']

if __name__=='__main__':
    guard()
    if '--check' in sys.argv:print('ROUND21_EXTRA_PREFLIGHT_PASS: original three request hashes replayed, frozen1005 verified; no API call.',flush=True)
    else:
        stream.ResponseStreamState.handle_event=stream.handle_event
        original.selector=extra_selector
        original.activate()
        sys.exit(original.worker(6101,HYBRID))

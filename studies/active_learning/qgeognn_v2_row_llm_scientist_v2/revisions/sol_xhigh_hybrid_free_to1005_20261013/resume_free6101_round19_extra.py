"""One human-authorized additional correction for seed6101 Free round19 only."""
import json, sys
from pathlib import Path
import run_seed_keys_parallel as runner
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import DIRECT_PROMPT, encode_catalog, validate_direct
from src.qgeognn_al.active_learning_v2.protocol import stable_hash
base=runner.base
HERE=runner.HERE
DIRECTORY=base.directory(6101,base.FREE,19)
ORIGINAL_SELECTOR=base.selector
INVALID=['163f6f0793f0d13950de','e8d25b0f93b5bcf88daf','fa7b8202df1e985c228c','9eaa21d2af3158cfa4f5','bd62986a3ea6d28161ec','7e981ba02c83f74f5e65']

def guard():
    record=base.read(HERE/'round19_extra_authorization_manifest.json')
    assert runner.sha(__file__)==record['wrapper_sha256']
    assert runner.sha(HERE/'run_seed_keys_parallel.py')==record['credential_runner_sha256']
    runner.guard()

def build_messages(packet,catalog,config):
    assert (packet['seed'],packet['method'],packet['round'],packet['active_label_count'])==(6101,base.FREE,19,941)
    assert config==base.NEW_CONFIG
    assert all(packet.get(k)==v for k,v in encode_catalog(catalog).items())
    assert packet['packet_sha256']==stable_hash({k:v for k,v in packet.items() if k!='packet_sha256'})
    messages=[{'role':'system','content':DIRECT_PROMPT},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
    for i in range(3):
        receipt=base.read(DIRECTORY/f'turn_{i:02d}.json')
        assert receipt['request_sha256']==stable_hash({'messages':messages,'config':config}),i
        feedback=base.read(DIRECTORY/f'rejected_turn_{i:02d}.json')
        messages += [{'role':'assistant','content':receipt['answer']},{'role':'user','content':json.dumps(feedback,ensure_ascii=False)}]
    previous=json.loads(base.read(DIRECTORY/'turn_02.json')['answer'])
    h=next(h for h in previous['hypotheses'] if h['id']=='H6')
    assert h['supporting_observed_ids']==INVALID
    invalid=sorted(set(INVALID)-set(catalog.observed))
    assert invalid==['9eaa21d2af3158cfa4f5']
    detail={'authorization':'User explicitly authorized exactly one additional correction for seed6101 Free round19 (941 to 973 labels) on 2026-10-14. This is a recorded protocol deviation; all other rounds retain three logical attempts.',
        'cited_evidence_ids':INVALID,'invalid_observed_evidence_ids':invalid,
        'instruction':'H6 incorrectly cites the listed evidence IDs as supporting observed evidence. The invalid_observed_evidence_ids subset is NOT in the supplied observed table. Correct the hypotheses yourself: cite only IDs actually present in the observed table, or use empty observed-evidence lists when relying on chemical priors or unmeasured model predictions. Check every supporting_observed_ids and contradicting_observed_ids entry in all hypotheses against the observed table. Also verify all 32 choices are distinct eligible candidate IDs. Return the complete final JSON with the same packet_sha256 and original schema. Do not invent IDs, change the catalog, treat unmeasured candidates as measured evidence, or claim a predicted error is measured. This is the sole additional logical correction.'}
    messages.append({'role':'user','content':json.dumps(detail,ensure_ascii=False)})
    return messages,detail

def extra_selector(packet,catalog,path,config,call=None):
    if Path(path)!=DIRECTORY:
        return ORIGINAL_SELECTOR(packet,catalog,path,config,call=call)
    config=base.NEW_CONFIG
    messages,detail=build_messages(packet,catalog,config)
    catalog.viewed.update(catalog.candidates)
    base.once(DIRECTORY/'model_transition.json',{'model':'gpt-6.1-sol','reasoning_effort':'xhigh','extension_start_labels':813,'extension_target_labels':1005})
    base.once(DIRECTORY/'authorized_extra_correction.json',{**detail,'original_call_budget':3,'additional_logical_calls':1,'packet_sha256':packet['packet_sha256'],'script_sha256':runner.sha(__file__),'historical_transcript_hashes':{f'turn_{i:02d}.json':runner.sha(DIRECTORY/f'turn_{i:02d}.json') for i in range(3)}})
    request_hash=stable_hash({'messages':messages,'config':config})
    base.once(DIRECTORY/'extra_correction_request.json',{'request_sha256':request_hash,'feedback':detail})
    path=DIRECTORY/'turn_03.json'
    if path.exists():
        receipt=base.read(path)
        assert receipt['request_sha256']==request_hash
    else:
        answer,provenance=base.transport._call_with_retries(messages,config,call or base.transport.responses_call,DIRECTORY,3,request_hash)
        receipt={'request_sha256':request_hash,'answer':answer,'provenance':provenance}
        base.once(path,receipt)
    try:
        result=validate_direct(json.loads(receipt['answer']),catalog,packet)
    except (ValueError,TypeError,KeyError) as error:
        base.once(DIRECTORY/'rejected_turn_03.json',{'error':str(error),'additional_budget_exhausted':True})
        raise
    base.once(DIRECTORY/'selection.json',result)
    return result

if __name__=='__main__':
    guard()
    packet=base.read(DIRECTORY/'input.json')
    catalog=base.s.ReadOnlyCatalog(**base.read(DIRECTORY/'catalog.json'))
    build_messages(packet,catalog,base.NEW_CONFIG)
    freeze=base.read(HERE/'runtime/seed_6101'/base.FREE/'round_19/prediction_freeze.json')
    for key in ('checkpoint','prediction'):
        assert runner.sha(freeze[key+'_path'])==freeze[key+'_sha256']
    if '--check' in sys.argv:
        print('ROUND19_EXTRA_PREFLIGHT_PASS: three exact request hashes replayed; frozen941 verified; no API call.',flush=True)
    else:
        base.selector=extra_selector
        sys.exit(runner.worker(6101,base.FREE))

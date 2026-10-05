"""One user-authorized extra correction for round 6; no test evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2 import scientist_transport as transport
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import DIRECT_PROMPT, encode_catalog, validate_direct
from src.qgeognn_al.active_learning_v2.dialog_bridge import read, once
from src.qgeognn_al.active_learning_v2.protocol import stable_hash

METHOD = 'free_llm32_scientist_v2'
study.configure_revision('astra_high_full_pool_20261001')
DIRECTORY = study.selection_directory(157, METHOD, 5)

def build_messages(packet, catalog, config):
    assert (packet['seed'], packet['method'], packet['round'], packet['active_label_count']) == (157, METHOD, 5, 493)
    assert all(packet.get(k) == v for k, v in encode_catalog(catalog).items())
    assert packet['packet_sha256'] == stable_hash({k:v for k,v in packet.items() if k != 'packet_sha256'})
    messages = [{'role':'system','content':DIRECT_PROMPT},
                {'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
    for i in range(3):
        receipt = read(DIRECTORY / f'turn_{i:02d}.json')
        assert receipt['request_sha256'] == stable_hash({'messages':messages,'config':config})
        feedback = read(DIRECTORY / f'rejected_turn_{i:02d}.json')
        messages += [{'role':'assistant','content':receipt['answer']},
                     {'role':'user','content':json.dumps(feedback,ensure_ascii=False)}]
    previous = json.loads(read(DIRECTORY / 'turn_02.json')['answer'])
    invalid = sorted({c['id'] for c in previous['choices']} - set(catalog.candidates))
    assert invalid == ['88d94bacb5a3b54ac7d1']
    detail = {'authorization':'User approved one additional correction beyond the original three-call budget for round 6.',
              'invalid_candidate_ids':invalid,
              'instruction':'88d94bacb5a3b54ac7d1 is already measured and appears in the observed table, not the eligible candidates table. Remove it from choices and choose a replacement yourself from the supplied eligible candidates table. Return exactly 32 distinct eligible candidate IDs and the full final JSON, with the same packet_sha256. Verify every choice ID against the candidates table. Do not select observed IDs, invent IDs, or change the catalog. All original output schema and evidence requirements still apply.'}
    messages.append({'role':'user','content':json.dumps(detail,ensure_ascii=False)})
    return messages, detail

def extra_selector(packet, catalog, directory, config, call=None):
    if Path(directory) != DIRECTORY:
        return transport.run_selector(packet, catalog, directory, config, call=call)
    messages, detail = build_messages(packet, catalog, config)
    catalog.viewed.update(catalog.candidates)
    once(DIRECTORY / 'authorized_extra_correction.json', {
        **detail, 'original_call_budget':3, 'additional_logical_calls':1,
        'packet_sha256':packet['packet_sha256'],
        'script_sha256':study.sha256_file(Path(__file__)),
        'historical_transcript_hashes':{p.name:study.sha256_file(p) for p in sorted(DIRECTORY.glob('turn_0[012].json'))}})
    request_hash = stable_hash({'messages':messages,'config':config})
    once(DIRECTORY / 'extra_correction_request.json', {'request_sha256':request_hash,'feedback':detail})
    path = DIRECTORY / 'turn_03.json'
    if path.exists():
        receipt = read(path)
        assert receipt['request_sha256'] == request_hash
    else:
        answer, provenance = transport._call_with_retries(messages,config,call or transport.responses_call,DIRECTORY,3,request_hash)
        receipt = {'request_sha256':request_hash,'answer':answer,'provenance':provenance}
        once(path,receipt)
    try:
        result = validate_direct(json.loads(receipt['answer']),catalog,packet)
    except (ValueError,TypeError,KeyError) as error:
        once(DIRECTORY / 'rejected_turn_03.json',{'error':str(error),'additional_budget_exhausted':True})
        raise
    once(DIRECTORY / 'selection.json',result)
    return result

if __name__ == '__main__':
    study.validate()
    packet, raw = read(DIRECTORY/'input.json'), read(DIRECTORY/'catalog.json')
    config = read(study.STUDY/'protocol.json')['transport']
    build_messages(packet,study.ReadOnlyCatalog(**raw),config)
    if '--check' in sys.argv:
        print('Preflight passed: historical request hashes, catalog and round-6 identity verified; no API call.')
    else:
        study.run_selector = extra_selector
        details = {'seed':157,'method':METHOD,'round':5}
        with study.exclusive_lock(study.STUDY/'execution.lock'):
            try:
                study._progress({'action':'authorized_round6_resume_started',**details})
                with study.status_phase('round 6: one authorized extra correction',30,**details):
                    result = study.select(157,METHOD,5)
                study._progress({'action':'select_completed',**details,'result':result})
                with study.status_phase('round 6: train final 525-label model',30,**details):
                    result = study.advance(157,METHOD,5)
                study._progress({'action':'advance_completed',**details,'result':result})
                trajectory = read(study.STUDY/f'runtime/seed_157/{METHOD}/trajectory_freeze.json')
                assert trajectory['final_active_labels'] == 525 and len(set(trajectory['selected_ids'])) == 192
                study.verify_files(study.STUDY,trajectory['files'])
                study.state(study.Context(157),METHOD,6)
                result = {'status':'TRAJECTORY_COMPLETE_NO_TEST_EVALUATION','final_active_labels':525,'authorized_extra_correction':True}
                once(study.STUDY/'round6_resume_result.json',result)
                study._progress({'action':'trajectory_completed',**details,'result':result})
                print(json.dumps(result),flush=True)
            except Exception as error:
                study._progress({'action':'authorized_round6_resume_failed','error_type':type(error).__name__,'message':str(error)[:1000],**details})
                raise

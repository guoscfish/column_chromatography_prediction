"""One authorized seed6101 Free round-zero correction; resume 813 extension."""
import json
import sys
from pathlib import Path

ROOT = Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
sys.path.insert(0, str(ROOT))
import extend_authorized_to813 as extension
from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2 import scientist_transport as transport
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import DIRECT_PROMPT, encode_catalog, validate_direct
from src.qgeognn_al.active_learning_v2.dialog_bridge import read, once
from src.qgeognn_al.active_learning_v2.protocol import stable_hash

METHOD = 'free_llm32_scientist_v2'
study.configure_revision('astra_high_full_pool_20261001')
DIRECTORY = study.selection_directory(6101, METHOD, 0)

def build_messages(packet, catalog, config):
    assert (packet['seed'], packet['method'], packet['round'], packet['active_label_count']) == (6101, METHOD, 0, 333)
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
    assert invalid == ['f90a5a5a54509c085ebd']
    detail = {'authorization':'User approved one additional correction beyond the original three-call budget for seed6101 Free round 0 on 2026-10-06.',
              'invalid_candidate_ids':invalid,
              'instruction':'f90a5a5a54509c085ebd is already measured and appears in the observed table, not the eligible candidates table. Remove it from choices and choose a replacement yourself from the supplied eligible candidates table. Return exactly 32 distinct eligible candidate IDs and the full final JSON, with the same packet_sha256. Verify every choice ID against the candidates table. Do not select observed IDs, invent IDs, or change the catalog. All original output schema and evidence requirements still apply.'}
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
    import runpy
    study.validate()
    packet, raw = read(DIRECTORY/'input.json'), read(DIRECTORY/'catalog.json')
    config = read(study.STUDY/'protocol.json')['transport']
    build_messages(packet,study.ReadOnlyCatalog(**raw),config)
    study.run_selector = extra_selector
    if '--check' in sys.argv:
        with study.exclusive_lock(study.STUDY/'execution.lock'):
            extension.preflight()
        print('PREFLIGHT_PASS: exact previous transcript verified; only seed6101 Free round0 gets turn03; no API call.',flush=True)
    else:
        # Original runner and its manifest remain byte-for-byte unchanged.
        runpy.run_path(str(Path(__file__).with_name('extend_authorized_to813.py')),run_name='__main__')

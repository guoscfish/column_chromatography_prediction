"""User-authorized, recorded single-character ID repair; no model calls or test evaluation."""
import copy
import json
import sys
from pathlib import Path
ROOT = Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2 import scientist_transport as transport
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import validate_direct, decode_table
from src.qgeognn_al.active_learning_v2.dialog_bridge import read, once

METHOD = 'free_llm32_scientist_v2'
study.configure_revision('astra_high_full_pool_20261001')
DIRECTORY = study.selection_directory(157, METHOD, 5)
BAD, GOOD = '97318de99bab4460cf4e8', '97318de99bab4460cf4e'

def corrected_selector(packet, catalog, directory, config, call=None):
    if Path(directory) != DIRECTORY:
        return transport.run_selector(packet,catalog,directory,config,call=call)
    assert (packet['seed'],packet['method'],packet['round'],packet['active_label_count']) == (157,METHOD,5,493)
    original = json.loads(read(DIRECTORY/'turn_03.json')['answer'])
    corrected = copy.deepcopy(original)
    matches = [c for c in corrected['choices'] if c['id'] == BAD]
    assert len(matches) == 1 and BAD == GOOD + '8'
    assert BAD not in catalog.candidates and GOOD in catalog.candidates
    row = next(r for r in decode_table(packet,'candidates') if r['id'] == GOOD)
    assert abs(row['loading_amount_mg'] - 64.585) < 1e-8
    assert 'm200' in matches[0]['reason'] and '64.585' in matches[0]['reason'] and '1/1' in matches[0]['reason']
    matches[0]['id'] = GOOD
    catalog.viewed.update(catalog.candidates)
    result = validate_direct(corrected,catalog,packet)
    once(DIRECTORY/'authorized_id_transcription_correction.json',{
        'authorization':'User approved continuing with the returned selection after the explained one-character ID correction.',
        'source':'turn_03.json','source_sha256':study.sha256_file(DIRECTORY/'turn_03.json'),
        'old_id':BAD,'new_id':GOOD,'reason':'Extra trailing 8; candidate molecule and conditions agree with the returned rationale.',
        'selection_reason_unchanged':True,'all_other_fields_unchanged':True,
        'new_llm_calls':0,'new_labels_accessed_for_correction':0,
        'protocol_deviation':'Round 6 used a fourth model response and a user-authorized deterministic ID transcription repair.',
        'script_sha256':study.sha256_file(Path(__file__))})
    once(DIRECTORY/'selection.json',result)
    return result

if __name__ == '__main__':
    study.validate()
    study.run_selector = corrected_selector
    details = {'seed':157,'method':METHOD,'round':5}
    with study.exclusive_lock(study.STUDY/'execution.lock'):
        try:
            study._progress({'action':'round6_transcription_resume_started',**details})
            with study.status_phase('round 6: validate authorized transcription correction and freeze batch',30,**details):
                result = study.select(157,METHOD,5)
            study._progress({'action':'select_completed',**details,'result':result})
            with study.status_phase('round 6: train final 525-label model',30,**details):
                result = study.advance(157,METHOD,5)
            study._progress({'action':'advance_completed',**details,'result':result})
            trajectory_path = study.STUDY/f'runtime/seed_157/{METHOD}/trajectory_freeze.json'
            trajectory = read(trajectory_path)
            assert trajectory['final_active_labels'] == 525 and len(trajectory['selected_ids']) == len(set(trajectory['selected_ids'])) == 192
            study.verify_files(study.STUDY,trajectory['files'])
            study.state(study.Context(157),METHOD,6)
            for r in range(7):
                f = read(study.STUDY/f'runtime/seed_157/{METHOD}/round_{r:02d}/prediction_freeze.json')
                for kind in ['checkpoint','prediction']:
                    assert study.sha256_file(Path(f[f'{kind}_path'])) == f[f'{kind}_sha256']
            curves=[]
            for r in range(7):
                if r == 0:
                    f=read(study.STUDY/f'runtime/seed_157/{METHOD}/round_00/prediction_freeze.json')
                    audit=read(Path(f['checkpoint_path']).parent/'fit_audit.json')
                else:
                    audit=read(study.STUDY/f'runtime/seed_157/{METHOD}/round_{r:02d}/model/fit_audit.json')
                curves.append({'round':r,'labels':audit['train_rows'],'validation_nrmse':audit['best_validation_combined_normalized_rmse']})
            result={'status':'TRAJECTORY_COMPLETE_NO_TEST_EVALUATION','final_active_labels':525,
                    'validation_learning_curve':curves,'test_evaluation_performed':False,
                    'protocol_deviation':'Round 6 fourth response plus user-authorized one-character ID repair.',
                    'trajectory_freeze_sha256':study.sha256_file(trajectory_path),
                    'correction_receipt_sha256':study.sha256_file(DIRECTORY/'authorized_id_transcription_correction.json'),
                    'audit_runner':str(Path(__file__))}
            once(study.STUDY/'round6_transcription_result.json',result)
            study._progress({'action':'trajectory_completed',**details,'result':result})
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        except Exception as error:
            study._progress({'action':'round6_transcription_resume_failed','error_type':type(error).__name__,'message':str(error)[:1000],**details})
            raise

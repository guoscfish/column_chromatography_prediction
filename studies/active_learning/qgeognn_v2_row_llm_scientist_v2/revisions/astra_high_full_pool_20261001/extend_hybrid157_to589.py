"""Explicit two-round extension of completed hybrid157; preserve six-round study."""
import json
import sys
from pathlib import Path
ROOT = Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
sys.path.insert(0,str(ROOT))
from src.qgeognn_al.active_learning_v2 import scientist_study as s
from src.qgeognn_al.active_learning_v2.dialog_bridge import read, once
METHOD = 'cw16_llm16_scientist_v2'
s.configure_revision('astra_high_full_pool_20261001')
ORIGINAL_DIRECTORY = s.selection_directory
ORIGINAL_BUDGETS = s.BUDGETS
s.BUDGETS = (*ORIGINAL_BUDGETS,557,589)

def extension_directory(seed,method,r):
    if r in (6,7) and seed == 157 and method == METHOD:
        return s.STUDY/f'selections/seed_{seed}/{method}/round_{r:02d}'
    return ORIGINAL_DIRECTORY(seed,method,r)
s.selection_directory = extension_directory
BASE = s.STUDY/f'runtime/seed_157/{METHOD}/trajectory_freeze.json'
EXT = s.STUDY/'extension_hybrid157_to589'

def preflight():
    s.validate()
    base=read(BASE)
    assert base['final_active_labels']==525 and len(set(base['selected_ids']))==192
    s.verify_files(s.STUDY,base['files'])
    labeled,_,_,_=s.state(s.Context(157),METHOD,6)
    assert len(labeled)==525
    manifest={'authorization':'User requested two additional rounds after reviewing the six-round validation result.',
        'classification':'exploratory post-six-round extension',
        'seed':157,'method':METHOD,'original_budgets':list(ORIGINAL_BUDGETS),
        'additional_round_indices':[6,7],'additional_label_budgets':[557,589],
        'batch_size':32,'CW_count':16,'LLM_count':16,
        'base_trajectory_sha256':s.sha256_file(BASE),
        'base_protocol_sha256':s.sha256_file(s.STUDY/'protocol.json'),
        'runner_sha256':s.sha256_file(Path(__file__)),
        'transport':read(s.STUDY/'protocol.json')['transport'],
        'selection_prompt_changed':False,'selection_call_budget_changed':False,
        'base_code_changed':False,'training_changed':False,'test_evaluation':False}
    return manifest

if __name__=='__main__':
    manifest=preflight()
    if '--check' in sys.argv:
        print(json.dumps({'status':'EXTENSION_PREFLIGHT_PASSED','base_labels':525,'targets':[557,589]}))
        sys.exit(0)
    with s.exclusive_lock(s.STUDY/'execution.lock'):
        EXT.mkdir(exist_ok=True)
        once(EXT/'manifest.json',manifest)
        try:
            for r in (6,7):
                details={'seed':157,'method':METHOD,'round':r,'extension':True}
                s._progress({'action':'extension_select_started',**details})
                with s.status_phase(f'hybrid extension cycle={r+1}/8 labels={s.BUDGETS[r]}->{s.BUDGETS[r+1]} selection',30,**details):
                    result=s.select(157,METHOD,r)
                d=s.selection_directory(157,METHOD,r)
                once(EXT/f'round_{r:02d}_binding.json',{'manifest_sha256':s.sha256_file(EXT/'manifest.json'),
                    'input_sha256':s.sha256_file(d/'input.json'),'batch_freeze_sha256':s.sha256_file(d/'batch_freeze.json')})
                s._progress({'action':'extension_select_completed',**details,'result':result})
                with s.status_phase(f'hybrid extension cycle={r+1}/8 train {s.BUDGETS[r+1]} labels',30,**details):
                    result=s.advance(157,METHOD,r)
                s._progress({'action':'extension_advance_completed',**details,'result':result})
            ctx=s.Context(157);labeled,_,_,_=s.state(ctx,METHOD,8)
            assert len(labeled)==589
            curves=[];hashes={};ids=[]
            for r in range(8):
                d=s.selection_directory(157,METHOD,r);f=s.audit_batch(d)
                assert len(f['batch_ids'])==32 and len(f['pending_ids'])==16
                ids+=f['batch_ids']
                for n in ('batch_freeze.json','feedback.json','label_access_receipt.json'):
                    hashes[str((d/n).relative_to(s.STUDY))]=s.sha256_file(d/n)
            assert len(ids)==len(set(ids))==256
            for r in range(9):
                path=s.STUDY/f'runtime/seed_157/{METHOD}/round_{r:02d}/prediction_freeze.json'
                f=read(path);hashes[str(path.relative_to(s.STUDY))]=s.sha256_file(path)
                for k in ('checkpoint','prediction'):
                    assert s.sha256_file(Path(f[k+'_path']))==f[k+'_sha256']
                a=read(Path(f['checkpoint_path']).parent/'fit_audit.json')
                curves.append({'round':r,'labels':a['train_rows'],'validation_nrmse':a['best_validation_combined_normalized_rmse']})
            assert s.sha256_file(BASE)==manifest['base_trajectory_sha256']
            s.verify_files(s.STUDY,read(BASE)['files'])
            once(EXT/'trajectory_freeze.json',{'seed':157,'method':METHOD,'final_active_labels':589,'selected_ids':ids,'files':hashes,'manifest_sha256':s.sha256_file(EXT/'manifest.json')})
            result={'status':'EXTENSION_COMPLETE_NO_TEST_EVALUATION','final_active_labels':589,'validation_learning_curve':curves,'classification':'exploratory extension chosen after six-round validation inspection','test_evaluation_performed':False}
            once(EXT/'result.json',result);s._progress({'action':'extension_completed','result':result})
            print(json.dumps(result,indent=2),flush=True)
        except Exception as error:
            s._progress({'action':'extension_stopped_on_error','error_type':type(error).__name__,'message':str(error)[:1000]})
            raise

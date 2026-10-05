"""Run registered seed157 hybrid trajectory and summarize validation only."""
import json
import sys
from pathlib import Path
ROOT = Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
sys.path.insert(0,str(ROOT))
from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2.dialog_bridge import read, once
METHOD = 'cw16_llm16_scientist_v2'
study.configure_revision('astra_high_full_pool_20261001')
if __name__ == '__main__':
    study.validate()
    config = read(study.STUDY/'protocol.json')['transport']
    if '--check' in sys.argv:
        print(json.dumps({'validation':study.validate(),'transport':study.check_transport(config)}))
        sys.exit(0)
    result = study.run(157,METHOD,config,status_interval=30)
    trajectory = read(study.STUDY/f'runtime/seed_157/{METHOD}/trajectory_freeze.json')
    assert trajectory['final_active_labels'] == 525 and len(trajectory['selected_ids']) == len(set(trajectory['selected_ids'])) == 192
    study.verify_files(study.STUDY,trajectory['files'])
    curves = []
    for r in range(7):
        freeze = read(study.STUDY/f'runtime/seed_157/{METHOD}/round_{r:02d}/prediction_freeze.json')
        for kind in ['checkpoint','prediction']:
            assert study.sha256_file(Path(freeze[f'{kind}_path'])) == freeze[f'{kind}_sha256']
        audit=read(Path(freeze['checkpoint_path']).parent/'fit_audit.json')
        curves.append({'round':r,'labels':audit['train_rows'],'validation_nrmse':audit['best_validation_combined_normalized_rmse']})
    result.update(test_evaluation_performed=False,final_active_labels=525,validation_learning_curve=curves)
    once(study.STUDY/'hybrid157_result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

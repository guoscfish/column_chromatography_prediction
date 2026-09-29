"""Post-fit report from frozen artifacts only; never sent to the selector."""
from pathlib import Path
import subprocess
import numpy as np
from .artifacts import read, once, file_hash, verify_audit
from .memory import quantiles
from .study import directory, METHODS, state
from .protocol import validate


def summarize(root, round_index=0):
    root = Path(root)
    validate(root)
    current = state(root, 157, METHODS[0])
    if current['round'] < round_index+1:
        raise RuntimeError('report requires the requested acquisition to be trained')
    d = directory(root,157,METHODS[0],round_index)
    protocol = read(root/'protocol.json')
    events = verify_audit(d/'audit.jsonl')
    selection = read(d/'selection.json')
    measurement = read(d/'measurement.json')
    fit = read(Path(read(d/'training_complete.json')['checkpoint']).parent/'fit_audit.json')
    baseline = read(protocol['initial']['157']['checkpoint_audit'])
    previous = baseline if round_index == 0 else read(Path(read(directory(root,157,METHODS[0],round_index-1)/'training_complete.json')['checkpoint']).parent/'fit_audit.json')
    errors = np.array([[r['error_V1_ml'],r['error_V2_ml']] for r in measurement['records']])
    scales = np.array(protocol['initial']['157']['target_scales'])
    combined = np.sqrt(np.mean((errors/scales)**2,axis=1))
    requests = [e for e in events if e['event_type']=='request']
    forbidden = ('validation_combined_normalized_rmse','best_validation_combined_normalized_rmse','validation_ids_hash','test_metrics','true_V1_ml','true_V2_ml')
    import json
    if any(term in json.dumps(requests) for term in forbidden):
        raise RuntimeError('selector context contains forbidden metrics/measurement payload')
    before,after=(x['best_validation_combined_normalized_rmse'] for x in (previous,fit))
    result = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':file_hash(root/'protocol.json'), 'model':protocol['transport']['model'],
        'provider':protocol['transport']['base_url'], 'seed':157,'method':METHODS[0], 'round':round_index,
        'labels_before':333+32*round_index, 'labels_after':365+32*round_index,
        **read(d/'interaction_summary.json'),
        'transport_calls':sum(e['event_type']=='transport_attempt' for e in events),
        'normalized_query_aliases':sum(e['event_type']=='query' and e['payload']['result'].get('normalized_query_alias',False) for e in events),
        'json_recoveries':sum(e['event_type']=='json_recovery' for e in events),
        'final_32_ids':[c['id'] for c in selection['choices']], 'batch_diagnostic':read(d/'batch_diagnostic.json'),
        'selected_point_acquisition_time_errors':{t:{'MAE_ml':float(np.mean(abs(errors[:,i]))),'RMSE_ml':float(np.sqrt(np.mean(errors[:,i]**2)))} for i,t in enumerate(('V1','V2'))},
        'combined_normalized_selected_point_error_distribution':quantiles(combined),
        'previous_fixed_validation_combined_NRMSE':before, 'current_fixed_validation_combined_NRMSE':after,
        'delta_current_minus_previous':after-before,'relative_reduction_percent':100*(before-after)/before,
        'same_initialization':fit['initialization_hash']==baseline['initialization_hash'],
        'fixed_validation':fit['validation_ids_hash']==baseline['validation_ids_hash'],
        'test_truth_access_count':read(d/'training_complete.json')['test_truth_access_count'],
        'selector_received_validation_metrics':False, 'registered_final_budget':525,
        'limitation':'One seed and one acquisition; checkpoint-validation scores are not an independent test or matched strategy comparison.'}
    once(root/f'round{round_index}_summary.json',result)
    for name in ('selection.json','batch_diagnostic.json','measurement.json','training_complete.json','label_access_audit.json','interaction_summary.json'):
        once(root/f'compact/round_{round_index:02d}'/name,read(d/name))
    once(root/'compact/L333_fit_audit.json',baseline)
    once(root/f'compact/L{365+32*round_index}_fit_audit.json',fit)
    lines = [f'# Scientist V3.2 round {round_index}: L{333+32*round_index} → L{365+32*round_index}', '',
             'This deterministic report was generated after training; metrics were not sent to the selector.', '',
             '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '',
             'Scientific interpretation will be included in the final analysis. No test truth was accessed.']
    target = root/f'ROUND{round_index}_REPORT.md'
    if not target.exists():
        target.write_text('\n'.join(lines)+'\n')
    return result

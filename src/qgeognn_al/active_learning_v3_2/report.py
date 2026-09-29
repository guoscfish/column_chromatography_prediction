"""Post-fit report from frozen artifacts only; never sent to the selector."""
from pathlib import Path
import subprocess
import numpy as np
from .artifacts import read, once, file_hash, verify_audit
from .memory import quantiles
from .study import directory, METHODS, state
from .protocol import validate


def summarize(root):
    root = Path(root)
    validate(root)
    current = state(root, 157, METHODS[0])
    if current['active_label_count'] != 365 or current['next_action'] != 'complete_no_test_evaluation':
        raise RuntimeError('report requires exactly one completed L333→L365 round')
    d = directory(root,157,METHODS[0],0)
    protocol = read(root/'protocol.json')
    events = verify_audit(d/'audit.jsonl')
    selection = read(d/'selection.json')
    measurement = read(d/'measurement.json')
    fit = read(Path(read(d/'training_complete.json')['checkpoint']).parent/'fit_audit.json')
    baseline = read(protocol['initial']['157']['checkpoint_audit'])
    errors = np.array([[r['error_V1_ml'],r['error_V2_ml']] for r in measurement['records']])
    scales = np.array(protocol['initial']['157']['target_scales'])
    combined = np.sqrt(np.mean((errors/scales)**2,axis=1))
    requests = [e for e in events if e['event_type']=='request']
    forbidden = ('validation_combined_normalized_rmse','best_validation_combined_normalized_rmse','validation_ids_hash','test_metrics','true_V1_ml','true_V2_ml')
    import json
    if any(term in json.dumps(requests) for term in forbidden):
        raise RuntimeError('selector context contains forbidden metrics/measurement payload')
    before,after=(x['best_validation_combined_normalized_rmse'] for x in (baseline,fit))
    result = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':file_hash(root/'protocol.json'), 'model':protocol['transport']['model'],
        'provider':protocol['transport']['base_url'], 'seed':157,'method':METHODS[0],
        **read(d/'interaction_summary.json'),
        'transport_calls':sum(e['event_type']=='transport_attempt' for e in events),
        'normalized_query_aliases':sum(e['event_type']=='query' and e['payload']['result'].get('normalized_query_alias',False) for e in events),
        'json_recoveries':sum(e['event_type']=='json_recovery' for e in events),
        'final_32_ids':[c['id'] for c in selection['choices']], 'batch_diagnostic':read(d/'batch_diagnostic.json'),
        'selected_point_acquisition_time_errors':{t:{'MAE_ml':float(np.mean(abs(errors[:,i]))),'RMSE_ml':float(np.sqrt(np.mean(errors[:,i]**2)))} for i,t in enumerate(('V1','V2'))},
        'combined_normalized_selected_point_error_distribution':quantiles(combined),
        'L333_fixed_validation_combined_NRMSE':before, 'L365_fixed_validation_combined_NRMSE':after,
        'delta_L365_minus_L333':after-before,'relative_reduction_percent':100*(before-after)/before,
        'same_initialization':fit['initialization_hash']==baseline['initialization_hash'],
        'fixed_validation':fit['validation_ids_hash']==baseline['validation_ids_hash'],
        'test_truth_access_count':read(d/'training_complete.json')['test_truth_access_count'],
        'selector_received_validation_metrics':False, 'stopped_at_L365':True,'next_round_started':False,
        'limitation':'One seed and one acquisition; checkpoint-validation scores are not an independent test or matched strategy comparison.'}
    once(root/'round0_summary.json',result)
    for name in ('selection.json','batch_diagnostic.json','measurement.json','training_complete.json','label_access_audit.json','interaction_summary.json'):
        once(root/'compact'/name,read(d/name))
    once(root/'compact/L333_fit_audit.json',baseline)
    once(root/'compact/L365_fit_audit.json',fit)
    files={str(p.relative_to(root)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','.operation.lock','ROUND0_REPORT.md')}
    once(root/'artifact_manifest.json',{'local_artifact_root':str(root.resolve()),'runtime_retention':'Full requests, responses, hash-chain audit and checkpoints retained locally; compact scientific outputs tracked in Git. No runtime artifacts deleted.','files':files})
    return result

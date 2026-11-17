"""Read saved validation summaries only; never opens source truth or predictions."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HISTORY = ROOT / 'studies/active_learning/comprehensive_history_diagnostics_20261112/tables/round_metrics.csv'
SEEDS = [157, 887, 2357, 6101, 12203]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {str(HISTORY.relative_to(ROOT)): digest(HISTORY)}
    src = pd.read_csv(HISTORY)
    allowed = ['random32', 'gradient_lcmd32', 'numerical_hybrid32', 'historical_free_mixed_model', 'new_free_cw_sol61_xhigh', 'nocw', 'cw_gradient_lcmd32']
    src = src.loc[src.method.isin(allowed) & src.seed.isin(SEEDS) & src.budget.le(1485)].copy()
    rows = []
    for _, rec in src.iterrows():
        audit_path = ROOT / rec.source
        audit = json.loads(audit_path.read_text())
        audit_hash = digest(audit_path)
        if audit_hash != rec.fit_audit_sha256:
            raise RuntimeError(f'changed saved baseline audit: {audit_path}')
        score = float(audit['best_validation_combined_normalized_rmse'])
        if not np.isclose(score, rec.nrmse, atol=1e-12, rtol=0):
            raise RuntimeError('saved validation score drift')
        manifest[str(audit_path.relative_to(ROOT))] = audit_hash
        rows.append({**rec.to_dict(), 'nrmse': score, 'evaluation_role': 'validation_checkpoint_selection', 'checkpoint_present': (audit_path.parent / 'best.pt').exists()})
    ivr = ROOT / 'studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/runtime'
    for seed in SEEDS:
        for r in range(22):
            freeze_path = ivr / f'seed_{seed}/round_{r:02d}/freeze.json'
            frozen = json.loads(freeze_path.read_text())
            model = ROOT / frozen['model_directory']
            audit_path, history_path = model / 'fit_audit.json', model / 'history.csv'
            audit = json.loads(audit_path.read_text())
            hist = pd.read_csv(history_path)
            best = hist.loc[hist.epoch.eq(audit['best_epoch'])]
            if len(best) != 1:
                raise RuntimeError('ambiguous best epoch')
            best = best.iloc[0]
            score = float(audit['best_validation_combined_normalized_rmse'])
            if not np.isclose(best.validation_combined_normalized_rmse, score, atol=1e-12, rtol=0):
                raise RuntimeError('IVR validation score/history mismatch')
            reference = src.loc[src.seed.eq(seed) & src.method.eq('gradient_lcmd32') & src.budget.eq(333)].iloc[0]
            for key in ('initialization_hash', 'validation_ids_hash'):
                if audit[key] != reference[key]:
                    raise RuntimeError(f'IVR {key} mismatch')
            for path in (freeze_path, audit_path, history_path):
                manifest[str(path.relative_to(ROOT))] = digest(path)
            rows.append(dict(method='historical_kernel_ivr32', seed=seed, budget=333+32*r, nrmse=score,
                **{k: best[k] for k in hist.columns if k.startswith('validation_') and k != 'validation_selection_score'},
                checkpoint_sha256=audit['checkpoint_sha256'], history_sha256=digest(history_path),
                source=str(audit_path.relative_to(ROOT)), fit_audit_sha256=digest(audit_path),
                validation_selection_score=score, initialization_hash=audit['initialization_hash'],
                initialization_seed=audit['initialization_seed'], validation_ids_hash=audit['validation_ids_hash'],
                train_ids_hash=audit['train_ids_hash'], training_seconds=audit['training_seconds'],
                epochs_run=audit['epochs_run'], best_epoch=audit['best_epoch'], parameter_count=audit['parameter_count'],
                fit_round=r, evaluation_role='validation_checkpoint_selection', checkpoint_present=(model/'best.pt').exists()))
    curves = pd.DataFrame(rows).sort_values(['method','seed','budget'])
    if curves.duplicated(['method','seed','budget']).any():
        raise RuntimeError('duplicate validation points')
    scales_path = ROOT / 'studies/active_learning/comprehensive_history_diagnostics_20261112/tables/normalization_scales.csv'
    manifest[str(scales_path.relative_to(ROOT))] = digest(scales_path)
    scales = pd.read_csv(scales_path)
    # Audit fixed denominator from latest CW protocol; no label reads.
    protocol_path = ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    manifest[str(protocol_path.relative_to(ROOT))] = digest(protocol_path)
    by_seed = {v['seed']: v['preprocessing']['target_scales'] for v in protocol['initial_states']}
    for rec in curves.itertuples():
        scale = by_seed[rec.seed]
        direct = math.sqrt(((rec.validation_V1_rmse/scale['V1'])**2+(rec.validation_V2_rmse/scale['V2'])**2)/2)
        if not np.isclose(direct, rec.nrmse, atol=2e-7, rtol=0):
            raise RuntimeError('fixed L333 scale mismatch')
    curves.to_csv(OUT / 'baseline_validation_curves.csv', index=False)
    metrics=[]
    for (method, seed), group in curves.groupby(['method','seed']):
        for end in (525,813,1005,1485):
            values = group.loc[group.budget.between(333,end)].sort_values('budget')
            expected = list(range(333,end+1,32))
            if values.budget.tolist()!=expected:
                continue
            metrics.append(dict(method=method,seed=int(seed),start_budget=333,end_budget=end,
                points=len(values),raw_aulc=float(np.trapezoid(values.nrmse,values.budget)/(end-333)),
                terminal_nrmse=float(values.nrmse.iloc[-1]),
                terminal_v1_mae=float(values.validation_V1_mae.iloc[-1]),
                terminal_v2_mae=float(values.validation_V2_mae.iloc[-1]),
                terminal_v1_rmse=float(values.validation_V1_rmse.iloc[-1]),
                terminal_v2_rmse=float(values.validation_V2_rmse.iloc[-1])))
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(OUT / 'baseline_seed_metrics.csv',index=False)
    summary=metrics.groupby(['method','end_budget']).agg(seeds=('seed','count'),aulc_mean=('raw_aulc','mean'),aulc_sample_sd=('raw_aulc','std'),terminal_mean=('terminal_nrmse','mean'),terminal_sample_sd=('terminal_nrmse','std')).reset_index()
    summary.to_csv(OUT / 'baseline_summary.csv',index=False)
    inventory=[]
    for method,g in curves.groupby('method'):
        inventory.append(dict(method=method,seeds=sorted(g.seed.unique().tolist()),points=len(g),max_budget_by_seed={str(int(s)):int(a.budget.max()) for s,a in g.groupby('seed')},checkpoint_files_present=int(g.checkpoint_present.sum()),validation_only=True))
    (OUT/'baseline_inventory.json').write_text(json.dumps(dict(status='HISTORICAL_DEVELOPMENT_VALIDATION_ONLY',new_fits=0,source_label_reads=0,test_prediction_reads=0,methods=inventory,input_sha256=manifest,ivr_report_caveat='Historical FINAL_REPORT values are test metrics; this inventory separately reconstructs validation from fit_audit/history best_epoch.'),ensure_ascii=False,indent=2)+'\n')
    print(summary.loc[summary.end_budget.eq(1005)].to_string(index=False))
    print('saved',len(curves),'validation points;',len(metrics),'seed-window records')

if __name__ == '__main__':
    main()

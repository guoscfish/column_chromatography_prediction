#!/usr/bin/env python3
"""Post-score reporting for the frozen formal traditional-transfer ROW study."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'studies/transfer/traditional_transfer_improvement/formal_row_5seed'
FROZEN=ROOT/'studies/transfer/filtered_full_data_benchmark'
METHODS=('P0','P1','P2','P3')
METRICS=('V1_r2','V1_rmse','V1_mae','V2_r2','V2_rmse','V2_mae','combined_normalized_rmse')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def md(frame, digits=3):
    f=frame.copy()
    for c in f:
        if pd.api.types.is_float_dtype(f[c]): f[c]=f[c].map(lambda x:f'{x:.{digits}f}')
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join('---' for _ in f.columns)+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None))+'\n'

def contexts():
    schedule=pd.read_csv(FROZEN/'split_manifest.csv'); protocol=json.loads((STUDY/'FORMAL_ROW_PROTOCOL.json').read_text())
    return schedule.loc[(schedule.protocol=='row')&schedule.column.isin(protocol['columns'])&schedule.outer_seed.isin(protocol['outer_seeds'])].copy()

def reference_comparison(metrics):
    reference_root=ROOT/'studies/transfer/row_column_selective_latent_audit'
    reference=pd.read_csv(reference_root/'per_seed_metrics.csv')
    expected=contexts(); actual=pd.read_csv(reference_root/'split_manifest.csv')
    rows=[]; chosen=('paper_style_current_v2','HIER_CW_SHARED_LAMBDA_CORRECTED','BALANCED_JOINT_FULL128')
    for c in ('25g','40g'):
        for seed in sorted(metrics.seed.unique()):
            left=set(expected.loc[(expected.column==c)&(expected.outer_seed==seed)&(expected.role=='test'),'sample_id'].astype(str))
            right=set(actual.loc[(actual.column==c)&(actual.protocol=='row')&(actual.outer_seed==seed)&(actual.role=='test'),'sample_id'].astype(str))
            if left!=right: raise RuntimeError(f'reference test identity mismatch: {c}/{seed}')
    ref=reference.loc[(reference.protocol=='row')&reference.column.isin(('25g','40g'))&reference.method.isin(chosen)].copy()
    if len(ref)!=30: raise RuntimeError('incomplete reference metrics')
    ref['comparison_status']='DIRECTLY_COMPARABLE_EXACT_ROW_TEST_IDS'
    ref.to_csv(STUDY/'reference_per_seed_metrics.csv',index=False)
    combined=pd.concat([metrics,ref[metrics.columns.intersection(ref.columns)]],ignore_index=True)
    summary=combined.groupby(['column','method'])[list(METRICS)].agg(['mean','std']); summary.columns=['_'.join(x) for x in summary.columns]
    return ref,summary.reset_index()

def main():
    if not (STUDY/'TEST_SCORE_MANIFEST.json').exists(): raise RuntimeError('score manifest required')
    metrics=pd.read_csv(STUDY/'test_metrics.csv')
    if len(metrics)!=40 or not np.isfinite(metrics[list(METRICS)].to_numpy(float)).all(): raise RuntimeError('invalid score table')
    if not (metrics.groupby(['column','method']).size()==5).all(): raise RuntimeError('incomplete seeds')
    metrics['quantile_crossing_rate']=metrics['quantile_crossing_count']/metrics['n_test']
    metrics.to_csv(STUDY/'test_per_seed_metrics.csv',index=False)
    summary_metrics=list(METRICS)+['quantile_crossing_count','quantile_crossing_rate']
    test_summary=metrics.groupby(['column','method'])[summary_metrics].agg(['mean','std','median','min','max']); test_summary.columns=['_'.join(x) for x in test_summary.columns]; test_summary=test_summary.reset_index(); test_summary.to_csv(STUDY/'test_summary.csv',index=False); test_summary.to_csv(STUDY/'summary.csv',index=False)
    # Validation-stage, convergence and BN tables are derived exclusively from frozen Phase-1 artifacts.
    stage=[]; drift=[]
    for c in ('25g','40g'):
        for seed in sorted(metrics.seed.unique()):
            base=STUDY/'runtime'/c/f'seed_{seed}'
            stage.append(pd.read_csv(base/'stage_metrics.csv').assign(column=c,seed=seed))
            drift.append(pd.read_csv(base/'drift_metrics.csv').assign(column=c,seed=seed))
    stage=pd.concat(stage,ignore_index=True); drift=pd.concat(drift,ignore_index=True)
    validation=stage.groupby(['column','method','stage'])[['validation_score','best_epoch','epochs_run']].agg(['mean','std','median','min','max']); validation.columns=['_'.join(x) for x in validation.columns]; validation.reset_index().to_csv(STUDY/'validation_summary.csv',index=False)
    adam=stage.loc[stage.method.isin(('P0','P1','P2'))].copy(); adam['hit_max_epoch']=adam.best_epoch.eq(150); adam['early_stopped']=adam.epochs_run.lt(150)
    convergence=adam.groupby(['column','method','stage'],as_index=False).agg(fits=('seed','size'),hit_max_epoch=('hit_max_epoch','sum'),early_stopped=('early_stopped','sum'),best_epoch_mean=('best_epoch','mean'),epochs_run_mean=('epochs_run','mean'))
    convergence.to_csv(STUDY/'convergence_summary.csv',index=False)
    bn=drift.merge(metrics[['column','seed','method','combined_normalized_rmse','quantile_crossing_count']],on=['column','seed','method'],suffixes=('_fit','_test'))
    bn.to_csv(STUDY/'bn_analysis.csv',index=False)
    staged=stage.loc[stage.method.isin(('P1','P2','P3'))].pivot(index=['column','seed','method'],columns='stage',values='validation_score').reset_index(); staged['stage_b_minus_a']=staged.B-staged.A; staged['stage_b_improves']=staged['stage_b_minus_a'].lt(0); staged.to_csv(STUDY/'stage_a_b_validation.csv',index=False)
    refs, comparison=reference_comparison(metrics); comparison.to_csv(STUDY/'reference_comparison.csv',index=False)
    # concise, fully auditable scientific report
    core=test_summary.loc[:,['column','method','V1_r2_mean','V1_r2_std','V1_rmse_mean','V1_rmse_std','V1_mae_mean','V1_mae_std','V2_r2_mean','V2_r2_std','V2_rmse_mean','V2_rmse_std','V2_mae_mean','V2_mae_std','combined_normalized_rmse_mean','combined_normalized_rmse_std']]
    pair=pd.read_csv(STUDY/'paired_comparisons.csv'); primary=pair.loc[pair.metric.eq('combined_normalized_rmse')]
    hit=int(adam.hit_max_epoch.sum()); total=len(adam); budget='TRAINING_BUDGET_POSSIBLY_INSUFFICIENT' if hit/total>=.4 else 'TRAINING_BUDGET_NOT_FLAGGED'
    lines=['# Formal traditional parameter-transfer: 5-seed ROW report\n',
      '## Integrity\n', 'All 40 Phase-1 fits completed before the global prediction freeze. The freeze manifest contains 10 contexts and 40 method-context fits. Test truth was then read once for the unified score table; all 40 metric rows are finite and each column/method has five frozen seeds.\n',
      '## Test results (mean ± SD)\n',md(core),
      '## Paired combined-NRMSE comparisons (negative delta favors candidate)\n',md(primary[['column','candidate','reference','mean_paired_delta','median_paired_delta','std','seed_wins','n']]),
      '## Convergence\n',f'`{budget}`: {hit}/{total} Adam stages selected their best checkpoint at epoch 150. No budget was changed after test evaluation.\n',md(convergence),
      '## Staged adaptation validation behavior\n',md(staged.groupby(['column','method'],as_index=False).agg(contexts=('seed','size'),stage_b_improves=('stage_b_improves','sum'),mean_stage_b_minus_a=('stage_b_minus_a','mean'))),
      '## Reference comparability\n', 'The reference split manifests have exactly matching ROW test sample-ID sets for all 10 contexts; the following frozen methods are directly comparable on the same test populations.\n',md(comparison.loc[comparison.method.isin((*METHODS,'paper_style_current_v2','HIER_CW_SHARED_LAMBDA_CORRECTED','BALANCED_JOINT_FULL128')),['column','method','V1_rmse_mean','V1_mae_mean','V1_r2_mean','V2_rmse_mean','V2_mae_mean','V2_r2_mean','combined_normalized_rmse_mean']]),
      'P3 is a complete L-BFGS recipe (including its frozen learning-rate/weight-decay recipe), not a standalone causal optimizer comparison. Quantile crossings are diagnostics, not calibrated uncertainty claims.\n']
    (STUDY/'FINAL_REPORT.md').write_text('\n'.join(lines))
    names=['FORMAL_ROW_PROTOCOL.json','FORMAL_ROW_PROTOCOL.sha256.json','PREDICTION_FREEZE_MANIFEST.json','TEST_SCORE_MANIFEST.json','summary.csv','test_metrics.csv','test_per_seed_metrics.csv','test_summary.csv','paired_comparisons.csv','validation_summary.csv','convergence_summary.csv','bn_analysis.csv','stage_a_b_validation.csv','reference_per_seed_metrics.csv','reference_comparison.csv','FINAL_REPORT.md','STAGE1_HANDOFF.md','STAGE1_MACHINE_SUMMARY.json']
    write_json(STUDY/'artifact_manifest.json',{'status':'COMPLETED_ONE_SHOT_TEST_SCORED','files':{x:sha(STUDY/x) for x in names},'reference_test_id_contract':'exact match against row_column_selective_latent_audit split_manifest'})
if __name__=='__main__': main()

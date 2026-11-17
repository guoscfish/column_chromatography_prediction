#!/usr/bin/env python3
"""Summarize frozen inference tables and static import boundaries without model calls."""
from pathlib import Path
import ast
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    d = pd.read_csv(HERE / 'frozen_response_seed_budget.csv')
    d.groupby(['budget', 'axis', 'target']).agg(
        seeds=('seed', 'nunique'), observed_abs_seed_mean=('observed_abs_mean', 'mean'),
        predicted_abs_seed_mean=('predicted_abs_mean', 'mean'), difference_mae_seed_mean=('difference_mae', 'mean'),
        flat_fraction_meaningful_seed_mean=('flat_fraction_meaningful', 'mean'),
        correct_nonflat_seed_mean=('correct_nonflat_fraction_meaningful', 'mean'),
        wrong_nonflat_seed_mean=('wrong_nonflat_fraction_meaningful', 'mean')).reset_index().to_csv(
            HERE / 'frozen_response_equal_seed.csv', index=False)
    rows = pd.read_csv(HERE / 'frozen_response_pairs.csv')
    examples = []
    for target in ('V1', 'V2'):
        volume = rows[(rows.axis == 'V/ul') & (rows.target == target) & rows.meaningful_0p5ml]
        for seed in (157, 887, 2357, 6101, 12203):
            flat = volume[(volume.seed == seed) & volume.prediction_flat_0p1ml]
            if len(flat):
                item = flat.loc[flat.actual_delta_ml.abs().idxmax()].to_dict()
                item['case_role'] = 'large_observed_flat_prediction'
                examples.append(item)
        correct = volume[volume.correct_nonflat].copy()
        correct['difference_error'] = (correct.actual_delta_ml - correct.predicted_delta_ml).abs()
        if len(correct):
            item = correct.sort_values(['difference_error', 'seed', 'budget', 'low_id']).iloc[0].drop('difference_error').to_dict()
            item['case_role'] = 'correct_nonflat_small_difference_error_counterexample'
            examples.append(item)
    pd.DataFrame(examples).to_csv(HERE / 'frozen_response_examples.csv', index=False)
    audit = []
    files = [ROOT / 'src/qgeognn_al' / name for name in ('data.py', 'models/qgeognn_v2.py', 'schemas/conditions.py')]
    files += [ROOT / 'application/QGeoGNN.py', ROOT / 'application/utils.py']
    for source in files:
        tree = ast.parse(source.read_text())
        top = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and ast.unparse(call.func) in ('pd.read_csv', 'np.load', 'torch.load', 'open'):
                    top.append({'line': call.lineno, 'call': ast.unparse(call.func)})
        audit.append({'path': str(source.relative_to(ROOT)), 'top_level_direct_data_load_calls': top,
                      'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    for p in (Path(__file__), HERE / 'frozen_response_pairs.csv', HERE / 'frozen_response_seed_budget.csv'):
        audit.append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (HERE / 'sensitivity_summary_manifest.json').write_text(json.dumps({'sources': audit,
        'new_model_calls': 0, 'new_fits': 0, 'scope': 'Static AST does not prove third-party library internals; relevant repository modules make no direct top-level target-table loads.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()

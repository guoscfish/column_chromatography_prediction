"""Verify full frozen response-series isolation without fitting or reading targets."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_training_qa', HERE / 'independent_training_qa.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_horizon_and_entire_history_group_isolation():
    report, overlap = module.validate()
    assert report['frozen_training_sources_unchanged']
    assert len(report['per_seed']) == 5
    assert all(row['training_same_series_overlap_n'] == 0 for row in report['per_seed'])
    assert all(row['training_exact_condition_overlap_n'] == 0 for row in report['per_seed'])
    assert all(row['initial_same_series_overlap_n'] == 0 for row in report['per_seed'])
    assert all(row['check_series_n'] == 10 and row['validation_n'] == 416 for row in report['per_seed'])
    assert sum(row['validation_exact_condition_rows'] for row in report['per_seed']) == 2
    assert overlap

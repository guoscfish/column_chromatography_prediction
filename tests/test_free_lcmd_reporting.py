"""Offline publication must preserve frozen endpoints and reject false completion."""
import json

import numpy as np
import pytest

from scripts import report_free_lcmd as reporting


def snapshot(directory, *, complete=False):
    curves = {}
    for seed in reporting.SEEDS:
        curves[str(seed)] = {}
        for method in (*reporting.TRADITIONAL, reporting.FREE):
            n = 22 if method in ('random', 'hybrid') else 37
            if not complete and seed == 12203 and method == reporting.FREE:
                n = 34
            curves[str(seed)][method] = [{'labels': 333+32*r, 'nrmse': .8-r*.01} for r in range(n)]
    summary = reporting.summarize(curves, {}, np)
    for name, value in [('validation_curves.json', curves), ('current_summary.json', summary),
                        ('source_hashes.json', {'frozen/source.json': 'original-provenance'})]:
        (directory/name).write_text(json.dumps(value))
    return curves, summary


def test_offline_snapshot_keeps_partial_endpoint_and_original_provenance(tmp_path):
    original, _ = snapshot(tmp_path)
    curves, summary, hashes = reporting.load_snapshot(tmp_path, False, np)
    assert curves == original
    assert summary['free_lcmd_matched_budget'] == 1389
    assert hashes == {'frozen/source.json': 'original-provenance'}
    reporting.report(summary, tmp_path/'REPORT.md', tmp_path)
    report = (tmp_path/'REPORT.md').read_text()
    assert '已完成9/10条，还剩3次拟合' in report
    assert '| 12203 | 1389 |' in report
    with pytest.raises(ValueError, match='ten trajectories'):
        reporting.load_snapshot(tmp_path, True, np)


def test_offline_snapshot_rejects_missing_budget_and_changed_summary(tmp_path):
    curves, _ = snapshot(tmp_path)
    curves['12203'][reporting.FREE][-1]['labels'] = 1421
    (tmp_path/'validation_curves.json').write_text(json.dumps(curves))
    with pytest.raises(ValueError, match='curve point'):
        reporting.load_snapshot(tmp_path, False, np)
    _, summary = snapshot(tmp_path)
    summary['status'] = 'COMPLETE'
    (tmp_path/'current_summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='disagrees'):
        reporting.load_snapshot(tmp_path, False, np)


def test_complete_gate_does_not_extrapolate_random_or_hybrid(tmp_path):
    snapshot(tmp_path, complete=True)
    curves, summary, _ = reporting.load_snapshot(tmp_path, True, np)
    assert summary['status'] == 'COMPLETE'
    assert all(curves[str(seed)]['random'][-1]['labels'] == 1005 for seed in reporting.SEEDS)

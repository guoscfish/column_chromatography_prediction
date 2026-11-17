"""Regression test for response-independent structured feature provenance."""
import json
from data_boundary import HERE
from repair_metadata import expected_provenance


def test_actual_structured_results_have_correct_provenance():
    files=sorted((HERE/'phase1').glob('*_structured_frozen_*.json'))
    assert files
    for p in files:
        d=json.loads(p.read_text())
        assert d['representation_provenance']==expected_provenance('structured')

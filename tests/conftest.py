"""Test classes are explicit; private resources never fall back to live data."""
import json
import os
from pathlib import Path
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if not any(item.get_closest_marker(name) for name in ('unit','frozen_regression','historical_archive')):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope='session', autouse=True)
def numerical_evidence():
    yield
    destination=os.getenv('QGEOGNN_EQUIVALENCE_OUTPUT')
    if destination:
        from tests.support.fixtures import ALGORITHM_RECORDS
        file=Path(destination);file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps({'source_commit':'16eca74769f664bc378a58ccbf380a806723f587',
            'scope':'Explicit pinned-source and authorized snapshot regressions; never new AL acquisition/labels/training',
            'records':ALGORITHM_RECORDS},indent=2)+'\n')

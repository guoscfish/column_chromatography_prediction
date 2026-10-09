import pytest
from src.qgeognn_al.active_learning_v2.cw_factorial_to1005 import reserve


def test_quota_survives_interruption_and_completed_evidence_loss():
    ledger = reserve({}, '11', False)
    with pytest.raises(RuntimeError, match='interrupted'):
        reserve(ledger, '11', False)
    assert reserve(ledger, '11', True) is ledger
    ledger['11']['status'] = 'completed'
    with pytest.raises(RuntimeError, match='restore evidence'):
        reserve(ledger, '11', False)
    full = {str(r): {'status': 'completed'} for r in range(11, 22)}
    assert reserve(full, '21', True) is full
    with pytest.raises(RuntimeError, match='allowance exhausted'):
        reserve(full, '22', False)

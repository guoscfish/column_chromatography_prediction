import numpy as np
import pytest
from src.qgeognn_al.al.evaluation.curves import normalized_aulc,labels_to_target
import legacy_reference


@pytest.mark.parametrize('kind',['declining','nonmonotone','flat'])
def test_aulc_and_threshold_math_matches_frozen_reporting(kind):
    labels=np.arange(333,1006,32)
    values={'declining':np.linspace(.9,.3,len(labels)), 'nonmonotone':.6+.2*np.sin(np.arange(len(labels))), 'flat':np.full(len(labels),.5)}[kind]
    old=legacy_reference.load('src/qgeognn_al/active_learning_v2/sequential_reporting.py',['normalized_aulc','labels_to_target'],injected={'ACTIVE_LABEL_BUDGETS':labels,'INITIAL_ACTIVE_LABELS':333,'FINAL_ACTIVE_LABELS':1005})
    assert normalized_aulc(labels,values,expected_budgets=labels)==old.normalized_aulc(labels,values)
    for target in [.2,.5,1.]:assert labels_to_target(labels,values,target)==old.labels_to_target(labels,values,target)
    with pytest.raises(ValueError):normalized_aulc(labels[:-1],values[:-1],expected_budgets=labels)

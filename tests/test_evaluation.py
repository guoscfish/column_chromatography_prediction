import numpy as np
import pytest
from src.qgeognn_al.al.evaluation.curves import normalized_aulc,labels_to_target
from tests.support import historical_reference as legacy_reference


@pytest.mark.historical_archive
@pytest.mark.parametrize('kind',['declining','nonmonotone','flat'])
def test_aulc_and_threshold_math_matches_frozen_reporting(kind):
    labels=np.arange(333,1006,32)
    values={'declining':np.linspace(.9,.3,len(labels)), 'nonmonotone':.6+.2*np.sin(np.arange(len(labels))), 'flat':np.full(len(labels),.5)}[kind]
    old=legacy_reference.load('src/qgeognn_al/active_learning_v2/sequential_reporting.py',['normalized_aulc','labels_to_target'],injected={'ACTIVE_LABEL_BUDGETS':labels,'INITIAL_ACTIVE_LABELS':333,'FINAL_ACTIVE_LABELS':1005})
    assert normalized_aulc(labels,values,expected_budgets=labels)==old.normalized_aulc(labels,values)
    for target in [.2,.5,1.]:assert labels_to_target(labels,values,target)==old.labels_to_target(labels,values,target)
    with pytest.raises(ValueError):normalized_aulc(labels[:-1],values[:-1],expected_budgets=labels)



def test_point_metrics_use_q50_and_explicit_per_target_source_scales():
    from src.qgeognn_al.al.evaluation.point import point_metrics
    truth=np.array([[0.,1.],[2.,5.]])
    prediction=np.array([[99.,1.,0.,99.,3.,0.],[99.,3.,0.,99.,3.,0.]])
    metrics=point_metrics(truth,prediction,{'V1':2.,'V2':4.})
    assert metrics['V1_rmse']==1. and metrics['V2_rmse']==2.
    assert metrics['V1_mae']==1. and metrics['V2_mae']==2.
    assert metrics['combined_normalized_rmse']==.5 and metrics['all_outputs_finite']


def test_aulc_interpolation_and_budget_failures_are_explicit():
    labels=np.array([0,10,30]);values=np.array([.9,.6,.3])
    assert normalized_aulc(labels,values)==pytest.approx((16.5,.55))
    assert labels_to_target(labels,values,.45)==pytest.approx(20.)
    assert labels_to_target(labels,values,.2) is None
    assert labels_to_target(labels,values,1.)==0.
    assert labels_to_target(np.array([0,10,30]),np.array([.8,.4,.6]),.5)==pytest.approx(7.5)
    for bad in [np.array([0,10,10]),np.array([0,30,10])]:
        with pytest.raises(ValueError):normalized_aulc(bad,values)
    with pytest.raises(ValueError):normalized_aulc(labels,values,expected_budgets=[0,10,20])


def test_saved_validation_plot_preserves_every_stored_point_and_input_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from matplotlib.axes import Axes
    from tools.plot_saved_validation import load_curves, plot

    curves = {'887': {'LCMD': [{'labels': 333, 'nrmse': .31}, {'labels': 365, 'nrmse': .29}],
                       'Free': [{'labels': 333, 'nrmse': .32}, {'labels': 429, 'nrmse': .28}]}}
    source = tmp_path/'curves.json'
    source.write_text(json.dumps(curves))
    original = source.read_bytes()
    assert load_curves(source) == curves
    plotted = []
    original_plot = Axes.plot
    def capture(self, x, y, *args, **kwargs):
        plotted.append((list(x), list(y), kwargs['label']))
        return original_plot(self, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, 'plot', capture)
    output = tmp_path/'raw.png'
    result = plot(source, output)
    assert plotted == [([333, 365], [.31, .29], 'LCMD'), ([333, 429], [.32, .28], 'Free')]
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert source.read_bytes() == original
    assert result['input_sha256'] == hashlib.sha256(original).hexdigest()
    assert result['seeds'] == ['887'] and result['rows'] == 4


def test_saved_validation_rejects_unordered_or_truth_rows_and_source_output(tmp_path):
    import json
    from pathlib import Path
    from tools.plot_saved_validation import load_curves, independent_output

    source = tmp_path/'curves.json'
    invalid_rows = [
        [{'labels': 365, 'nrmse': .3}, {'labels': 333, 'nrmse': .2}],
        [{'labels': 333, 'nrmse': .3}, {'labels': 333, 'nrmse': .2}],
        [{'labels': 333, 'nrmse': .3, 'test_truth': [1, 2]}],
        [{'labels': 333, 'nrmse': float('nan')}],
    ]
    for rows in invalid_rows:
        source.write_text(json.dumps({'157': {'LCMD': rows}}))
        with pytest.raises(ValueError):
            load_curves(source)
    with pytest.raises(PermissionError, match='outside all source worktrees'):
        independent_output(Path(__file__).resolve().parents[1]/'forbidden-test-plot.png')

import json
import pytest
from scripts.studies.run_qgeognn_v3_loop import run_loop, summarize_validation


class FakeStudy:
    def __init__(self, *, fail=False, phase='stage'):
        self.round, self.labels, self.phase = 1, 365, phase
        self.calls = []
        self.fail = fail

    def state(self, *args):
        return {'round': self.round, 'active_label_count': self.labels, 'next_action': self.phase}

    def stage(self, seed, method, r, **kwargs):
        self.calls.append(('stage', r)); self.phase = 'select'
        return {'status': 'STAGED'}

    def select(self, seed, method, r, **kwargs):
        self.calls.append(('select', r))
        if self.fail:
            raise RuntimeError('STOP: transport failure; see sanitized failure receipt')
        self.phase = 'advance'
        return {'status': 'FROZEN'}

    def advance(self, seed, method, r, **kwargs):
        assert self.phase == 'advance'
        self.calls.append(('advance', r)); self.round += 1; self.labels += 32
        self.phase = 'complete_no_test_evaluation' if self.round == 6 else 'stage'
        return {'status': 'TRAINED'}


def prepare(path):
    (path/'protocol.json').write_text(json.dumps({'budgets': [333, 365, 397, 429, 461, 493, 525]}))


def test_continues_all_remaining_rounds_and_stops_at_registered_budget(tmp_path):
    prepare(tmp_path); api = FakeStudy()
    result = run_loop(tmp_path, 157, 'free', api=api, validate=lambda _: None)
    assert result['status'] == 'COMPLETED' and result['active_label_count'] == 525
    assert api.calls == [(action, r) for r in range(1, 6) for action in ('stage', 'select', 'advance')]
    with pytest.raises(RuntimeError, match='already started'):
        run_loop(tmp_path, 157, 'free', api=api, validate=lambda _: None)
    assert len(api.calls) == 15


def test_failed_selection_never_advances_or_restarts(tmp_path):
    prepare(tmp_path); api = FakeStudy(fail=True)
    result = run_loop(tmp_path, 157, 'free', api=api, validate=lambda _: None)
    assert result['status'] == 'STOPPED' and api.labels == 365
    assert api.calls == [('stage', 1), ('select', 1)]


@pytest.mark.parametrize('phase', ['STOP_failed_or_interrupted_selector', 'advance_resume_training'])
def test_interrupted_operations_are_not_silently_repeated(tmp_path, phase):
    prepare(tmp_path); api = FakeStudy(phase=phase)
    result = run_loop(tmp_path, 157, 'free', api=api, validate=lambda _: None)
    assert result['status'] == 'STOPPED' and not api.calls


def test_no_training_when_protocol_guard_fails_between_selection_and_advance(tmp_path):
    prepare(tmp_path); api = FakeStudy()
    def validate(_):
        if api.phase == 'advance':
            raise RuntimeError('protocol drift')
    result = run_loop(tmp_path, 157, 'free', api=api, validate=validate)
    assert result['status'] == 'STOPPED' and api.labels == 365
    assert api.calls == [('stage', 1), ('select', 1)]


@pytest.mark.parametrize('validation_drift', [False, True])
def test_learning_curve_uses_only_comparable_saved_fit_audits(tmp_path, validation_drift):
    baseline = {'train_rows': 333, 'validation_ids_hash': 'same-validation',
                'initialization_hash': 'same-initialization',
                'test_labels_used_for_fit_or_checkpoint_selection': 0,
                'best_validation_combined_normalized_rmse': 1.0}
    initial = tmp_path/'initial_fit.json'; initial.write_text(json.dumps(baseline))
    (tmp_path/'protocol.json').write_text(json.dumps({'budgets': [333, 365],
        'initial': {'157': {'checkpoint_audit': str(initial)}}}))
    fit = tmp_path/'model'; fit.mkdir()
    (fit/'fit_audit.json').write_text(json.dumps({**baseline, 'train_rows': 365,
        'validation_ids_hash': 'different' if validation_drift else 'same-validation',
        'best_validation_combined_normalized_rmse': 0.8}))
    round_dir = tmp_path/'selections/seed_157/free/round_00'; round_dir.mkdir(parents=True)
    (round_dir/'training_complete.json').write_text(json.dumps({'checkpoint': str(fit/'best.pt')}))
    if validation_drift:
        with pytest.raises(RuntimeError, match='comparability'):
            summarize_validation(tmp_path, 157, 'free')
    else:
        result = summarize_validation(tmp_path, 157, 'free')
        assert result['relative_reduction_from_L333_percent'] == pytest.approx(20.)
        assert [p['training_labels'] for p in result['points']] == [333, 365]
        assert (tmp_path/'execution/continuous_loop/validation_learning_curve.md').exists()

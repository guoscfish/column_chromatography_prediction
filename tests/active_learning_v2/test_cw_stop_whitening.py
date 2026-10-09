from pathlib import Path
import runpy
from src.qgeognn_al.active_learning_v2 import cw_factorial_to1005 as last
from src.qgeognn_al.active_learning_v2.cw_factorial import ARMS


def test_amended_worker_excludes_whitening_without_mutating_frozen_arm_map(monkeypatch):
    scope = runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/studies/continue_cw_without_whitening.py'))
    seen = []
    monkeypatch.setattr(last, 'ARMS', dict(ARMS))
    monkeypatch.setattr(last, 'run_seed', lambda seed: seen.append((seed,tuple(last.ARMS))))
    saved = []
    scope['worker'].__globals__['atomic_json'] = lambda path,value: saved.append(value)
    scope['worker'](157)
    assert seen == [(157,('cw_maxdet','cw_ivr'))]
    assert 'whitening_lcmd' in ARMS
    assert saved[-1]['fits'] == 22 and saved[-1]['whitening'] == 'stopped_at525_by_user'

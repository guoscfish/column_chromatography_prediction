"""Exercise OS-held/stale locks and supervisor adoption without training."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'studies/active_learning/epig_prediction_oriented_v2_20261117'
sys.path.insert(0,str(HERE))
import supervisor


def test_missing_or_stale_lock_is_not_active(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'HERE',tmp_path)
    assert supervisor.worker_lock_held() is False
    lock=tmp_path/'runtime/exploratory.lock';lock.parent.mkdir();lock.write_text('999999 old PID text')
    assert supervisor.worker_lock_held() is False
    assert lock.read_text()=='999999 old PID text'


def test_actual_subprocess_held_lock_is_detected_then_released(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'HERE',tmp_path)
    lock=tmp_path/'runtime/exploratory.lock';lock.parent.mkdir()
    code='import fcntl,sys; h=open(sys.argv[1],"a+"); fcntl.flock(h,fcntl.LOCK_EX); print("READY",flush=True); sys.stdin.readline()'
    process=subprocess.Popen([sys.executable,'-c',code,str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='READY'
        assert supervisor.worker_lock_held() is True
    finally:
        process.communicate('\n',timeout=5)
    assert process.returncode==0
    assert supervisor.worker_lock_held() is False


def test_restart_monitors_held_worker_without_launching_another(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'HERE',tmp_path)
    monkeypatch.setattr(supervisor,'worker_lock_held',lambda:True)
    monkeypatch.setattr(supervisor,'refresh_report',lambda:0)
    def forbidden(*args,**kwargs): raise AssertionError('must not spawn competing worker')
    monkeypatch.setattr(supervisor.subprocess,'Popen',forbidden)
    class StopObservation(Exception): pass
    def observed_sleep(seconds):
        assert seconds==45
        state=supervisor.read(tmp_path/'runtime/supervisor_status.json')
        assert state['status']=='MONITORING_EXISTING_WORKER'
        assert state['launches']==[]
        raise StopObservation
    monkeypatch.setattr(supervisor.time,'sleep',observed_sleep)
    with pytest.raises(StopObservation): supervisor.main()

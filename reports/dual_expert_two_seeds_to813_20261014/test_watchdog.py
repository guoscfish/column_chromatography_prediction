import importlib.util
import os
from pathlib import Path
import sys
import time

spec=importlib.util.spec_from_file_location('dual_watchdog',Path(__file__).with_name('watchdog.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

def test_heartbeat_and_old_receipts_do_not_hide_stall(tmp_path):
    role=tmp_path/'agents/ml_scientist';role.mkdir(parents=True)
    old=role/'turn_00.json';old.write_text('{}');os.utime(old,(50,50))
    (tmp_path/'execution_progress.jsonl').write_text('heartbeat')
    assert w.overdue_selection(tmp_path,100,1001)

def test_complete_reply_resets_deadline(tmp_path):
    role=tmp_path/'agents/ml_scientist';role.mkdir(parents=True)
    reply=role/'turn_01.json';reply.write_text('{}');os.utime(reply,(950,950))
    assert not w.overdue_selection(tmp_path,100,1001)

def test_training_is_not_interrupted_by_selection_deadline(tmp_path):
    (tmp_path/'batch_freeze.json').write_text('{}')
    assert not w.overdue_selection(tmp_path,100,5000)

def test_historical_errors_do_not_block_a_new_credential_run(tmp_path):
    role=tmp_path/'agents/ml_scientist/retries/turn_00';role.mkdir(parents=True)
    old=role/'attempt_01.json';old.write_text('{"message":"DAILY_LIMIT_EXCEEDED"}');os.utime(old,(50,50))
    new=role/'attempt_02.json';new.write_text('{"error_type":"APIConnectionError"}');os.utime(new,(150,150))
    assert w.recent_failures(tmp_path,100)==[{'error_type':'APIConnectionError'}]

def test_stalled_process_is_actually_stopped_and_probe_recorded(tmp_path):
    emitted=[];started=time.monotonic()
    result=w.run_monitored([sys.executable,'-c','import time;time.sleep(30)'],cwd=tmp_path,
        log=tmp_path/'worker.log',directory=tmp_path,probe=lambda:{'http_status':200},
        emit=lambda **data:emitted.append(data),idle_seconds=.1,poll_seconds=.02)
    assert result['watchdog_timeout'] and result['returncode']==124
    assert result['provider_probe']=={'http_status':200}
    assert time.monotonic()-started<5
    assert emitted[0]['event']=='selection_reply_timeout'
    try:os.kill(result['pid'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('Timed-out worker is still running')

"""User explicitly requested one immediate retry of seed6101 on2026-10-12."""
import sys,time
import resume_free_sol_full_parallel as runner
original_sleep=time.sleep
used=False

def skip_initial_backoff(seconds):
    global used
    if not used and seconds>900:
        used=True
        time.sleep=original_sleep
        runner.s._progress({'action':'user_authorized_immediate_retry','seed':6101,'skipped_backoff_seconds':seconds})
        return
    return original_sleep(seconds)

if __name__=='__main__':
    time.sleep=skip_initial_backoff
    sys.exit(runner.worker(6101))

"""Observed stream adapter plus shared one-request-at-a-time gate."""
import sys,fcntl
import run_free157_observed_stream as observed
runner=observed.runner

def serial_observed_call(messages,config):
    with (runner.base.NEW/'llm_request.execution.lock').open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:return observed.observed_call(messages,config)
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

if __name__=='__main__':
    runner.base.transport.responses_call=serial_observed_call
    sys.exit(runner.worker(157))

"""Request gate scoped to the two Free trajectories; independent of unrelated studies."""
import fcntl,os,httpx
from openai import APIConnectionError
import run_free157_observed_stream as observed
runner=observed.runner
original=runner.base.transport.responses_call

def call(messages,config):
    with (runner.base.NEW/'free_only_request.execution.lock').open('a+') as handle:
        runner.s._progress({'action':'free_request_queued','pid':os.getpid()})
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:
            runner.s._progress({'action':'free_request_started','pid':os.getpid()})
            try:
                return observed.observed_call(messages,config) if SEED==157 else original(messages,config)
            except httpx.RemoteProtocolError as exc:
                # Incomplete HTTP message bodies are transport failures, not illegal selections.
                raise APIConnectionError(message='Transport RemoteProtocolError: '+str(exc),request=httpx.Request('POST',config['base_url'].rstrip('/')+'/responses')) from exc
        finally:
            runner.s._progress({'action':'free_request_finished','pid':os.getpid()})
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
SEED=None

def activate(seed):
    global SEED
    SEED=seed
    runner.activate()
    runner.base.transport.responses_call=call

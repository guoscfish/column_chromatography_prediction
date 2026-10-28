"""Classify the provider's upstream HTTP/2 stream error as recoverable transport."""
import continue_eight_to1421 as runner
import sys
from pathlib import Path
stream=runner.previous_extra.stream
previous_classify=stream.classify

def classify(code,message):
    if code=='upstream_http2_stream_error':return 'transient'
    return previous_classify(code,message)

def guard():
    m=runner.read(runner.HERE/'upstream_http2_compatibility.json')
    assert runner.sha(__file__)==m['adapter_sha256']
    assert runner.sha(runner.HERE/'continue_eight_to1421.py')==m['runner_sha256']
    runner.guard()
    assert classify('upstream_http2_stream_error','Upstream HTTP/2 stream failed')=='transient'
    assert classify('USAGE_LIMIT_EXCEEDED','DAILY_LIMIT_EXCEEDED')=='quota'
    assert classify('context_length_exceeded','too long')=='decision'

if __name__=='__main__':
    guard()
    if '--check' in sys.argv:print('HTTP2_COMPATIBILITY_PREFLIGHT_PASS: transport classification only; frozen science/requests unchanged.',flush=True)
    else:
        i=sys.argv.index('--worker');seed=int(sys.argv[i+1]);method=sys.argv[i+2]
        stream.classify=classify
        sys.exit(runner.worker(seed,method))

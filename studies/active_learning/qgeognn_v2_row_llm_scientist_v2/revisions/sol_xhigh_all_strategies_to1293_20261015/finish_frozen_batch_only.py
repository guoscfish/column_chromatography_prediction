"""Fit one already-frozen batch without any provider calls or quota-marker changes."""
import resume_upstream_http2_compatible as adapter
import sys,os
m=adapter.runner

def deny(*args,**kwargs):raise RuntimeError('Provider calls and new selection are forbidden in frozen-batch-only mode')

if __name__=='__main__':
    record=m.read(m.HERE/'frozen_batch_fit_manifest.json')
    assert m.sha(__file__)==record['wrapper_sha256']
    assert m.sha(m.HERE/'resume_upstream_http2_compatible.py')==record['adapter_sha256']
    adapter.guard();m.activate()
    i=sys.argv.index('--worker');seed=int(sys.argv[i+1]);method=sys.argv[i+2];r=int(sys.argv[i+3])
    assert method==m.HYBRID and r==27 and seed in (157,6101)
    d=m.directory(seed,method,r);m.s.audit_batch(d)
    assert m.sha(d/'batch_freeze.json')==record['batches'][str(seed)]
    m.transport.responses_call=deny
    m.base.raw_call=deny
    m.s._progress({'action':'worker_started','seed':seed,'method':method,'pid':os.getpid(),'provider_calls_allowed':False})
    try:
        with m.s.exclusive_lock(m.HERE/f'{seed}_{method}.lock'):
            m.s._progress({'action':'training_queued','seed':seed,'method':method,'round':r})
            with m.training_slot():
                m.s._progress({'action':'training_started','seed':seed,'method':method,'round':r,'pid':os.getpid()})
                with m.s.status_phase(f'seed{seed} {method} frozen-batch fit1229',30,seed=seed,method=method,round=r):m.s.advance(seed,method,r)
                freeze=m.read(m.HERE/f'runtime/seed_{seed}/{method}/round_28/prediction_freeze.json')
                for k in ('checkpoint','prediction'):assert m.sha(freeze[k+'_path'])==freeze[k+'_sha256']
                m.s._progress({'action':'training_completed','seed':seed,'method':method,'round':r})
    finally:m.s._progress({'action':'worker_finished','seed':seed,'method':method,'pid':os.getpid()})

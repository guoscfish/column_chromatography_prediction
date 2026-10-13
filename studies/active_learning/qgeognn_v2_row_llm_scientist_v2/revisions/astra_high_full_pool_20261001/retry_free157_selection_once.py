"""User-authorized one acquisition retry; freeze selection then stop, no training."""
import json
import run_free157_observed_serial as transport_adapter
runner=transport_adapter.runner
s=runner.s
if __name__=='__main__':
    runner.activate()
    runner.base.transport.responses_call=transport_adapter.serial_observed_call
    with s.exclusive_lock(runner.base.NEW/'seed_157.execution.lock'):
        assert len(s.state(s.Context(157),runner.base.METHOD,12)[0])==717
        s._progress({'action':'user_retry_seed157_selection_only_started','seed':157,'round':12})
        try:
            with s.status_phase('Seed157 selection-only retry717->749',30,seed=157,round=12):
                result=s.select(157,runner.base.METHOD,12)
            d=runner.base.directory(157,runner.base.METHOD,12)
            frozen=s.audit_batch(d);assert len(frozen['batch_ids'])==32 and not frozen['pending_ids']
            outcome={'status':'SELECTION_FROZEN_TRAINING_PAUSED','seed':157,'round':12,'current_labels':717,
                'selected_ids':frozen['batch_ids'],'batch_freeze_sha256':s.sha256_file(d/'batch_freeze.json'),'new_training_started':False}
            runner.base.original.once(runner.base.NEW/'seed157_selection_only_result.json',outcome)
            s._progress({'action':'seed157_selection_only_completed','seed':157,'round':12})
            print(json.dumps(outcome),flush=True)
        except Exception as exc:
            s._progress({'action':'seed157_selection_only_stopped','seed':157,'round':12,'error_type':type(exc).__name__,'message':str(exc)[:1000]})
            raise

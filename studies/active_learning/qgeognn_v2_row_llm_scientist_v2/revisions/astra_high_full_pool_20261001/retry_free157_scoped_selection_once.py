import json
import free_scoped_request_gate as gate
s=gate.runner.s
if __name__=='__main__':
    gate.activate(157)
    base=gate.runner.base
    with s.exclusive_lock(base.NEW/'seed_157.execution.lock'):
        with s.status_phase('Seed157 scoped selection-only717->749',30,seed=157,round=12):s.select(157,base.METHOD,12)
        d=base.directory(157,base.METHOD,12);b=s.audit_batch(d)
        assert len(b['batch_ids'])==32 and not b['pending_ids']
        base.original.once(base.NEW/'seed157_selection_only_result.json',{'status':'SELECTION_FROZEN_TRAINING_PAUSED','seed':157,'round':12,'current_labels':717,'selected_ids':b['batch_ids'],'batch_freeze_sha256':s.sha256_file(d/'batch_freeze.json'),'new_training_started':False})
        s._progress({'action':'seed157_selection_only_completed','seed':157,'round':12})

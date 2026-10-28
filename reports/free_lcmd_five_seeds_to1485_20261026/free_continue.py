"""Explicit post-primary logical-budget amendment; original science stays frozen."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import control as c

ENTRY=Path(__file__).resolve()
ROOT=c.ROOT
OUT=ENTRY.parent/'free'
OLD=ROOT/'reports/free_llm_five_seeds_to1485_20261024'
spec=importlib.util.spec_from_file_location('frozen_free_continuation',OLD/'continue_to1485.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=m.s
OLD_STUDY=m.STUDY
STUDY=OLD_STUDY.parent/'sol61_xhigh_free_to1485_extra3_20261026'
CUTS={887:33,12203:24}
SEEDS=tuple(CUTS)
MANIFEST=OUT/'continuation_amendment.json'
old_validate=m.validate
old_activate=m.activate
old_selector=s.run_selector

def runtime(seed,r):
    if r<=21:return m.PREFIX/f'runtime/seed_{seed}/{m.FREE}/round_{r:02d}'
    owner=OLD_STUDY if r<=CUTS[seed] else STUDY
    return owner/f'runtime/seed_{seed}/{m.FREE}/round_{r:02d}'

def directory(seed,method,r):
    if seed not in SEEDS or method!=m.FREE or r not in range(36):raise ValueError('Unregistered amendment trajectory')
    owner=m.PREFIX if r<21 else OLD_STUDY if r<CUTS[seed] else STUDY
    return owner/f'selections/seed_{seed}/{method}/round_{r:02d}'

def prediction(context,method,r,labeled):
    if r>CUTS[context.seed]:return m.ORIGINAL_PREDICTION(context,method,r,labeled)
    frozen=s.read(runtime(context.seed,r)/'prediction_freeze.json')
    for kind in ('checkpoint','prediction'):
        if c.sha(frozen[kind+'_path'])!=frozen[kind+'_sha256']:raise RuntimeError('Frozen predictor drift')
    if frozen['L_t_ids_hash']!=s.ids_hash(context.ids(labeled)):raise RuntimeError('Frozen labeled identity drift')
    scratch=context.runtime/method/f'round_{r:02d}';scratch.mkdir(parents=True,exist_ok=True)
    return Path(frozen['checkpoint_path']),scratch

def validate():
    old_validate()
    record=c.read(MANIFEST)
    for path,digest in record['protected_files'].items():
        if c.sha(ROOT/path)!=digest:raise RuntimeError('Frozen amendment source/prefix drift: '+path)
    if record['cuts']!={str(k):v for k,v in CUTS.items()}:raise RuntimeError('Frozen amendment cut drift')
    return {'status':'VALID','amendment_sha256':c.sha(MANIFEST)}

def selector(packet,catalog,path,config,call=None):
    # Exact replay of the original three failed turns precedes new attempts 3..5.
    seed,r=packet['seed'],packet['round']
    budget=6 if seed in CUTS and r==CUTS[seed] and Path(path).is_relative_to(STUDY) else 3
    previous=m.base.transport.FULL_SELECTION_ATTEMPTS
    m.base.transport.FULL_SELECTION_ATTEMPTS=budget
    try:return old_selector(packet,catalog,path,config,call=call)
    finally:m.base.transport.FULL_SELECTION_ATTEMPTS=previous

def activate(seed=None):
    m.OUT=OUT;m.STUDY=STUDY;m.SEEDS=SEEDS
    m.runtime=runtime;m.directory=directory;m.prediction=prediction
    m.validate=validate;m.training_slot=c.training_slot
    s.run_selector=selector
    old_activate(seed)

def complete(seed):return STUDY/f'{seed}_{m.FREE}_result.json'

def prepare():
    old_validate();OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():activate();validate();print('ALREADY_PREPARED');return
    protected={str(ENTRY.relative_to(ROOT)):c.sha(ENTRY),
                str(Path(c.__file__).relative_to(ROOT)):c.sha(c.__file__)}
    records={}
    for seed,cut in CUTS.items():
        result=m.audit_seed(seed)
        assert result['final_labels']==333+32*cut
        records[str(seed)]={'prefix_audit':result,'previous_supervisor':c.read(OLD/f'supervisor_seed_{seed}.json')}
        source=OLD_STUDY/f'selections/seed_{seed}/{m.FREE}/round_{cut:02d}'
        assert not (source/'batch_freeze.json').exists()
        assert len(list(source.glob('rejected_turn_*.json')))==3
        dest=STUDY/f'selections/seed_{seed}/{m.FREE}/round_{cut:02d}'
        dest.mkdir(parents=True,exist_ok=True)
        for path in source.rglob('*'):
            if not path.is_file():continue
            target=dest/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert c.sha(target)==c.sha(path)
            else:shutil.copy2(path,target)
            for item in (path,target):protected[str(item.relative_to(ROOT))]=c.sha(item)
        for path,digest in result['files'].items():protected[str(Path(path).relative_to(ROOT))]=digest
        # Retain the same session transport counters, including historical failures.
        c.once(OUT/f'supervisor_seed_{seed}.json',records[str(seed)]['previous_supervisor'])
        c.once(OUT/f'status_seed_{seed}.json',c.read(OLD/f'status_seed_{seed}.json'))
    STUDY.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','selector_prompt.txt'):
        dest=STUDY/name
        if dest.exists():assert c.sha(dest)==c.sha(OLD_STUDY/name)
        else:shutil.copy2(OLD_STUDY/name,dest)
        protected[str(dest.relative_to(ROOT))]=c.sha(dest)
    shutil.copy2(OLD/'launch.json',OUT/'launch.json')
    record={'time':c.now(),'authorization_date':'2026-10-26',
        'user_request':'继续跑完，另外lcmd32也跑到一样的',
        'interpretation':'Bounded post-primary continuation after recorded logical stops; retain failed turns and original scientific protocol.',
        'cuts':{str(k):v for k,v in CUTS.items()},'extra_logical_attempts_at_cut_only':3,
        'total_logical_attempts_at_cut':6,'all_other_round_attempts':3,
        'target_labels':1485,'target_round':36,'model':'gpt-6.1-sol','reasoning_effort':'xhigh',
        'original_scientific_protocol':str(OLD_STUDY/'protocol.json'),
        'scientific_protocol_sha256':c.sha(OLD_STUDY/'protocol.json'),
        'prompt_changed':False,'training_changed':False,'split_changed':False,
        'test_truth_access_count':0,'transport_budgets_changed':False,
        'training_pool':str(c.POOL),'max_training_concurrency':3,
        'protected_files':protected,'prefix_records':records}
    c.once(MANIFEST,record);c.once(STUDY/'continuation_amendment.json',record)
    activate();validate()
    print(json.dumps({'status':'PREPARED','seeds':SEEDS,'extra_attempts_at_cut_only':3,'target_labels':1485}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--audit',action='store_true');p.add_argument('--worker',type=int,choices=SEEDS);p.add_argument('--supervise',type=int,choices=SEEDS)
    a=p.parse_args()
    if a.prepare:prepare()
    else:
        activate(a.worker if a.worker is not None else a.supervise)
        if a.launch:c.launch(ENTRY,OUT,SEEDS,complete,previous_out=OLD)
        elif a.audit:
            rows=[m.audit_seed(seed) for seed in SEEDS]
            record={'time':c.now(),'rows':rows,'test_evaluation_performed':False}
            c.atomic(OUT/'audit_latest.json',record)
            print(json.dumps({'time':record['time'],'rows':[{'seed':x['seed'],'labels':x['final_labels'],'fits':len(x['validation_learning_curve']),'batches':len(x['selected_ids'])//32} for x in rows]}),flush=True)
        elif a.worker is not None:m.work(a.worker)
        elif a.supervise is not None:c.supervise(ENTRY,OUT,a.supervise,complete(a.supervise))

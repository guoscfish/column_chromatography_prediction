#!/usr/bin/env python3
"""One entry for validation, offline replay and separately authorized Row AL.

No real action starts during validate/dry_run/offline replay. Real training,
provider and selected-label receipts require an independent scoped grant.
The original schema2 synthetic demonstration remains available via run().
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from src.qgeognn_al.al.workflow import RunSpec, RoundWorkflow, LabelBroker
from src.qgeognn_al.al.workflow.backends import SyntheticBackend
from src.qgeognn_al.infra.contracts import content_hash

ROOT = Path(__file__).resolve().parents[1]


def protected_main_worktrees():
    result = subprocess.run(['git','--no-optional-locks','-C',str(ROOT),'worktree','list','--porcelain'],capture_output=True,text=True)
    paths, current = [], None
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith('worktree '):current=line[9:]
            if line=='branch refs/heads/main' and current:paths.append(Path(current))
    return paths


def run(config, output, *, code_commit=None):
    if config.get('schema_version') != 2 or config.get('execution_mode') != 'offline_synthetic':
        raise PermissionError('this CLI only permits explicit synthetic lifecycle fixtures')
    if code_commit is None:
        code_commit = subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    from src.qgeognn_al.infra.execution import runtime_path
    output=runtime_path(output)
    dimension=config['feature_dimension']
    vectors={key: list(value)+[0.]*(dimension-len(value)) for key,value in config['feature_vectors'].items()}
    if any(len(value)!=dimension for value in vectors.values()):raise ValueError('invalid synthetic dimension')
    options=dict(config['run'])
    options.update(initial_ids=tuple(config['initial_ids']),candidate_ids=tuple(config['candidate_ids']),
                   validation_ids=tuple(config['validation_ids']),dataset_hash=content_hash(vectors),
                   split_hash=content_hash({k:config[k] for k in ['initial_ids','candidate_ids','validation_ids']}),
                   initial_state_hash=content_hash({'fixture':'synthetic_initial_state','dimension':dimension}),
                   protocol_hash=content_hash(config),execution_mode='offline_synthetic')
    spec=RunSpec(**options)
    backend=SyntheticBackend(features=vectors,validation_predictions=config['validation_predictions'],
                             initial_state_hash=spec.initial_state_hash,dataset_hash=spec.dataset_hash,
                             split_hash=spec.split_hash,code_commit=code_commit,protocol_hash=spec.protocol_hash)
    def reveal(ids,freeze_hash):return {identity:config['synthetic_reveal_values'][identity] for identity in ids}
    broker=LabelBroker(spec=spec,initial_labels=config['initial_labels'],validation_labels=config['validation_labels'],reveal=reveal)
    workflow=RoundWorkflow(output,spec=spec,backend=backend,labels=broker,forbidden_output_roots=protected_main_worktrees())
    final=workflow.run()
    return {'status':final['phase'],'source_kind':'synthetic','outer_seed':spec.seed,
            'acquisition_seed':spec.acquisition_seed,'acquisition':spec.acquisition,'budget':len(final['labeled_ids']),
            'rounds':final['round'],'code_commit':code_commit,'protocol_hash':spec.protocol_hash,
            'science_training_performed':False,'paid_requests':0,'real_labels_revealed':0}


def main():
    parser=argparse.ArgumentParser(description='Unified AL validation, replay and explicitly authorized Row execution')
    parser.add_argument('command',nargs='?',choices=('validate','dry_run','offline_replay','authorized_real_run','resume','report'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--profile')
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    from src.qgeognn_al.al.workflow.entry import (configuration_plan,build_row_workflow,
                                                run_recorded_workflow,report_run)
    if args.command=='report':
        if args.output is None:parser.error('report requires --output')
        result=report_run(args.output)
    elif args.command=='offline_replay' and args.fixture:
        from src.qgeognn_al.al.agents.recorded import replay_fixture
        result=replay_fixture(args.fixture,profile=args.profile)
    else:
        if args.config is None:parser.error('--config is required')
        config=json.loads(args.config.read_text())
        if args.command in ('validate','dry_run'):
            result=configuration_plan(config)
            result['mode']=args.command
            if args.command=='dry_run':result['stages']=['prepare','fit','predict/represent','select','freeze','reveal_authorized','feedback','next_fit','evaluate']
        elif args.command=='offline_replay':
            if args.output is None:parser.error('workflow replay requires --output')
            result=run_recorded_workflow(config,args.output)
        elif args.command in ('authorized_real_run','resume') and config.get('execution_mode')=='qgeognn_row':
            if args.output is None:parser.error('Row run requires --output')
            if args.command=='resume' and not (args.output/'state.json').exists():
                raise ValueError('resume requires an existing state')
            workflow=build_row_workflow(config,args.output,args.authorization)
            from src.qgeognn_al.al.workflow.state import AwaitingLabels
            waiting=False
            try:workflow.run()
            except AwaitingLabels:waiting=True
            result={**report_run(args.output),'source_kind':'qgeognn_row','qualification':'ONLINE_SCIENCE_NOT_MEASURED'}
            if waiting:result['status']='AWAITING_AUTHORIZED_LABELS'
        else:
            if args.output is None:parser.error('run requires --output')
            if args.command not in (None,'resume','authorized_real_run'):
                raise ValueError('unsupported execution mode')
            if args.command=='authorized_real_run':
                raise ValueError('authorized_real_run requires an explicit qgeognn_row config')
            if args.command=='resume' and not (args.output/'state.json').exists():
                raise ValueError('resume requires an existing state')
            result=run(config,args.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    if str(result.get('status','')).startswith('BLOCKED'):
        raise SystemExit(2)


if __name__=='__main__':main()

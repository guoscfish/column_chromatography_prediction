#!/usr/bin/env python3
"""One configuration-driven synthetic AL lifecycle demonstration.

No real training, provider, dataset reader or true-label loader is exposed by
this CLI. The workflow also has a separately gated QGeoGNN backend for future
explicitly authorized scientific runs; it is never selected by this command.
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()),args.output),sort_keys=True))


if __name__=='__main__':main()

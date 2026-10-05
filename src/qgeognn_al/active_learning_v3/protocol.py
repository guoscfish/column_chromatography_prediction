"""Fail-closed protocol lock: no revision API exists, including before resume."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
from ..resources import ROOT
from ..active_learning_v2.benchmark_protocol import TRAINING_CONFIG
from . import schema
from .artifacts import digest, file_hash, once, read
from .catalog import QUERY_SCHEMA
from .selector import SYSTEM_PROMPT, SELECTION_SCHEMA


def code_hashes():
    paths = sorted((ROOT/'src/qgeognn_al').rglob('*.py'))
    paths += [ROOT/'scripts/studies/run_qgeognn_v3_row_llm_scientist.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def fingerprint(config):
    packages = {}
    for name in ('numpy', 'pandas', 'rdkit', 'torch', 'torch-geometric', 'openai'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return json.loads(schema.dumps({'study_version': schema.VERSION, 'source_code_hashes': code_hashes(),
        'system_prompt': SYSTEM_PROMPT, 'query_schema': QUERY_SCHEMA,
        'memory_policy': schema.MEMORY_POLICY, 'working_state_schema': schema.WORKING_SCHEMA,
        'working_state_chars': schema.WORKING_STATE_CHARS, 'selection_schema': SELECTION_SCHEMA,
        'transport': config, 'training': TRAINING_CONFIG, 'packages': packages,
        'context_budget': config['context_hard_chars'], 'query_budget': schema.QUERY_BUDGET,
        'view_budget': schema.VIEW_BUDGET, 'model_call_budget': schema.MODEL_CALL_BUDGET,
        'queries_per_turn': schema.QUERIES_PER_TURN, 'batch_quotas': None}))


def validate(root):
    root = Path(root)
    record = read(root/'protocol.json')
    if record['fingerprint'] != fingerprint(record['transport']):
        raise RuntimeError('STOP STUDY: source/prompt/schema/transport drift; start a new study version')
    if (root/'selector_prompt.txt').read_text() != SYSTEM_PROMPT:
        raise RuntimeError('STOP STUDY: prompt artifact drift')
    for path, expected in record['input_hashes'].items():
        if file_hash(path) != expected:
            raise RuntimeError('STOP STUDY: frozen input artifact drift')
    if 'cli' in record['transport']:
        cli = record['transport']['cli']
        if file_hash(cli['executable']) != cli['sha256']:
            raise RuntimeError('STOP STUDY: CLI binary drift')
    lock = root/'protocol.lock.json'
    freezes = list((root/'selections').glob('seed_*/*/round_*/batch_freeze.json'))
    if freezes and not lock.exists():
        raise RuntimeError('STOP STUDY: scientific freeze exists without protocol lock')
    if lock.exists():
        frozen = read(lock)
        if frozen != {'protocol_sha256': file_hash(root/'protocol.json'),
                      'fingerprint_sha256': digest(record['fingerprint']), 'policy': 'STOP on drift; new V3.1/V4 required'}:
            raise RuntimeError('STOP STUDY: protocol lock mismatch')
    return {'status': 'VALID', 'locked': lock.exists(), 'protocol_sha256': file_hash(root/'protocol.json')}


def lock_protocol(root):
    # Called under the study operation flock, before the first scientific batch write.
    validate(root)
    record = read(root/'protocol.json')
    once(root/'protocol.lock.json', {'protocol_sha256': file_hash(root/'protocol.json'),
        'fingerprint_sha256': digest(record['fingerprint']), 'policy': 'STOP on drift; new V3.1/V4 required'})

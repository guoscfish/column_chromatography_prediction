"""Independent V3 artifacts, derived state and explicit scientific transitions."""
from __future__ import annotations
import copy
from pathlib import Path
import pandas as pd
from ..resources import ROOT
from ..active_learning_v2.protocol import ids_hash, validate_row_protocol
from . import protocol
from .artifacts import Audit, digest, file_hash, once, read, operation_lock, verify_audit
from .catalog import ReadOnlyCatalog
from .memory import new_ledger
from .schema import VERSION
from .selector import SYSTEM_PROMPT, make_packet, validate_selection
from .transport import build_request, empty_working_state, run_selector, settings

STUDY = ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v3_2'
SEEDS = (157,)
METHODS = ('free_llm32_scientist_v3_2',)
BUDGETS = (333, 365)


def directory(root, seed, method, r):
    if seed not in SEEDS or method not in METHODS or r not in range(1):
        raise ValueError('unregistered seed/method/round')
    return Path(root)/f'selections/seed_{seed}/{method}/round_{r:02d}'


def initial_contract(source, v2_root, seed, method):
    """Locate a seal for the exact clean initial catalog, never repair V2 artifacts."""
    direct = source/'contract.json'
    if direct.exists():
        return direct
    catalog_hash = file_hash(source/'catalog.json')
    candidates = sorted((v2_root/'revisions').glob(
        f'*/selections/seed_{seed}/{method}/round_00/contract.json'))
    for path in candidates:
        seal = read(path)
        if (seal.get('catalog_sha256') == catalog_hash and
                (seal.get('seed'), seal.get('method'), seal.get('round')) == (seed, method, 0)):
            return path
    raise RuntimeError('no immutable provenance for the exact initial catalog')


def prepare(root=STUDY, *, source_root=ROOT, config=None):
    root, source_root = Path(root), Path(source_root).resolve()
    with operation_lock(root):
        if (root/'protocol.json').exists():
            if config is not None and read(root/'protocol.json')['transport'] != config:
                raise RuntimeError('transport change forbidden')
            return protocol.validate(root)
        if (root/'protocol.lock.json').exists() or list(root.glob('selections/**/batch_freeze.json')):
            raise RuntimeError('cannot prepare over scientific freeze')
        config = config or settings()
        inputs, initial = {}, {}
        def pin(path):
            path = Path(path).resolve()
            inputs[str(path)] = file_hash(path)
            return str(path)
        for seed in SEEDS:
            baseline = source_root/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}'
            # The canonical CW split study has the same stable path as V2.
            from ..active_learning_v2.scientist_study import CW_SHORT
            split_path = source_root/CW_SHORT.relative_to(ROOT)/f'splits/row_seed_{seed}.csv'
            partition = pd.read_csv(split_path)
            validate_row_protocol(partition)
            l0 = partition.loc[partition.role.eq('l0'), 'sample_id'].astype(str).tolist()
            u0 = partition.loc[partition.role.eq('u0'), 'sample_id'].astype(str).tolist()
            metadata = read(baseline/'context.json')['contract']
            initial[str(seed)] = {'split': pin(split_path), 'context': pin(baseline/'context.json'),
                'scrubbed_graphs': pin(baseline/'scrubbed_graphs.pt'),
                'checkpoint': pin(baseline/'lcmd/round_00/model/best.pt'),
                'checkpoint_audit': pin(baseline/'lcmd/round_00/model/fit_audit.json'),
                'source_features': pin(source_root/'experiments/e0_4g_baseline/canonical_4g.csv'),
                'target_scales': [metadata['preprocessing']['target_scales'][t] for t in ('V1', 'V2')],
                'L333_ids_hash': ids_hash(l0)}
            for method in METHODS:
                old_method = method.replace('_v3_2', '_v2')
                source = source_root/f'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/selections/seed_{seed}/{old_method}/round_00'
                raw = read(source/'catalog.json')
                contract_path = initial_contract(source, source_root/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2', seed, old_method)
                contract = read(contract_path)
                if file_hash(source/'catalog.json') != contract['catalog_sha256']:
                    raise RuntimeError('initial catalog source binding mismatch')
                if (contract['seed'], contract['method'], contract['round']) != (seed, old_method, 0):
                    raise RuntimeError('foreign initial catalog')
                if contract['checkpoint_sha256'] != inputs[initial[str(seed)]['checkpoint']]:
                    raise RuntimeError('initial catalog checkpoint mismatch')
                if [r['id'] for r in raw['observed']] != l0 or set(r['id'] for r in raw['candidates']+raw['pending']) != set(u0):
                    raise RuntimeError('initial catalog is not clean L333/U0')
                if any(r.get('record_kind') != 'initial_L333' or any('premeasurement' in k for k in r) for r in raw['observed']):
                    raise RuntimeError('V2 trajectory outcomes cannot seed a clean V3 run')
                raw['salt'] = f'{VERSION}:{seed}:{method}:0'
                raw['target_scales'] = initial[str(seed)]['target_scales']
                ReadOnlyCatalog(**raw)
                path = root/f'initial/seed_{seed}/{method}.json'
                once(path, raw)
                pin(path)
                pin(source/'catalog.json'); pin(contract_path)
        once(root/'protocol.json', {'study_version': VERSION, 'transport': config,
            'fingerprint': protocol.fingerprint(config), 'input_hashes': inputs, 'initial': initial,
            'seeds': list(SEEDS), 'methods': list(METHODS), 'budgets': list(BUDGETS),
            'initial_reuse': 'clean L333 observed rows and round-zero predictions only; no V2 memory/choices',
            'test_evaluation': 'not exposed by this runner'})
        prompt = root/'selector_prompt.txt'
        if prompt.exists() and prompt.read_text() != SYSTEM_PROMPT:
            raise RuntimeError('prompt artifact drift')
        if not prompt.exists():
            with prompt.open('x') as stream:
                stream.write(SYSTEM_PROMPT)
        return {**protocol.validate(root), 'prepared': True, 'new_labels_revealed': 0, 'llm_calls': 0, 'fits': 0}


def verify_files(d, hashes):
    for name, expected in hashes.items():
        if file_hash(d/name) != expected:
            raise RuntimeError(f'immutable round artifact drift: {name}')


def audit_stage(root, seed, method, r):
    d = directory(root, seed, method, r)
    contract = read(d/'contract.json')
    if contract['protocol_sha256'] != file_hash(Path(root)/'protocol.json'):
        raise RuntimeError('stage protocol binding mismatch')
    verify_files(d, contract['artifact_hashes'])
    if file_hash(contract['checkpoint']) != contract['checkpoint_sha256']:
        raise RuntimeError('stage checkpoint drift')
    if (contract['seed'], contract['method'], contract['round']) != (seed, method, r):
        raise RuntimeError('foreign stage contract')
    return contract


def audit_batch(root, seed, method, r):
    d = directory(root, seed, method, r)
    contract = audit_stage(root, seed, method, r)
    frozen = read(d/'batch_freeze.json')
    receipt = read(d/'freeze_receipt.json')
    if receipt['batch_freeze_sha256'] != file_hash(d/'batch_freeze.json') or receipt['audit_sha256'] != file_hash(d/'freeze_audit/audit.jsonl'):
        raise RuntimeError('freeze receipt drift')
    verify_audit(d/'freeze_audit/audit.jsonl')
    if frozen['protocol_lock_sha256'] != file_hash(Path(root)/'protocol.lock.json'):
        raise RuntimeError('batch protocol lock mismatch')
    verify_files(d, frozen['artifact_hashes'])
    raw, packet, ledger = read(d/'catalog.json'), read(d/'input.json'), read(d/'ledger_before.json')
    catalog = ReadOnlyCatalog(**raw, ledger=ledger)
    catalog.initial_cards()
    events = verify_audit(d/'audit.jsonl')
    for event in events:
        if (event['seed'], event['method'], event['round']) != (seed, method, r):
            raise RuntimeError('foreign audit event')
        if event['event_type'] == 'query':
            query, expected = event['payload']['query'], event['payload']['result']
            try:
                actual = catalog.query(query['operation'], query['args']) if isinstance(query, dict) and set(query) == {'operation', 'args'} else catalog.query('__invalid_query__', {})
            except ValueError as error:
                actual = {'error': 'query_validation', 'message': str(error), 'remaining_queries': catalog.query_budget-len(catalog.queries)}
            if actual != expected:
                raise RuntimeError('query replay drift')
    result, merged = validate_selection(read(d/'selection.json'), catalog, packet, ledger)
    batch = list(catalog.pending)+[c['id'] for c in result['choices']]
    if frozen['batch_ids'] != batch or len(set(batch)) != 32 or not set(batch) <= set(contract['U_t_ids']):
        raise RuntimeError('frozen batch mismatch')
    lookup = {**catalog.pending, **catalog.candidates}
    predictions = [{'candidate_id': i, 'pred_V1_ml': lookup[i]['pred_V1_ml'], 'pred_V2_ml': lookup[i]['pred_V2_ml']} for i in batch]
    if predictions != frozen['premeasurement_predictions'] or merged != read(d/'ledger_after_selection.json'):
        raise RuntimeError('frozen predictions/ledger mismatch')
    return frozen


def state(root, seed, method):
    """Canonical state derived only from immutable, hash-linked phase artifacts."""
    root = Path(root)
    raw = read(root/f'initial/seed_{seed}/{method}.json')
    initial_count = len(raw['observed'])
    ledger, history, completed = new_ledger(seed, method), [], 0
    all_batch_ids = []
    phase = 'stage'
    for r in range(1):
        d = directory(root, seed, method, r)
        if not (d/'contract.json').exists():
            phase = 'stage'
            break
        audit_stage(root, seed, method, r)
        if read(d/'ledger_before.json') != ledger:
            raise RuntimeError('cross-round hypothesis lineage drift')
        if not (d/'batch_freeze.json').exists():
            phase = 'STOP_failed_or_interrupted_selector' if (d/'selector.started.json').exists() else 'select'
            break
        frozen = audit_batch(root, seed, method, r)
        if set(all_batch_ids) & set(frozen['batch_ids']):
            raise RuntimeError('duplicate batch across rounds')
        if not (d/'measurement.json').exists():
            phase = 'advance'
            break
        measurement = read(d/'measurement.json')
        if measurement['batch_freeze_sha256'] != file_hash(d/'batch_freeze.json'):
            raise RuntimeError('measurement freeze binding mismatch')
        if (measurement['seed'], measurement['method'], measurement['round']) != (seed, method, r):
            raise RuntimeError('foreign measurement')
        if [rec['candidate_id'] for rec in measurement['records']] != frozen['batch_ids']:
            raise RuntimeError('measurement IDs mismatch')
        access = measurement['label_access_receipt']
        if access['purpose'] != 'after_acquisition_fit' or not access['acquisitions_frozen'] or access['requested_ids_hash'] != ids_hash(frozen['batch_ids']):
            raise RuntimeError('invalid measurement access receipt')
        for rec, pred in zip(measurement['records'], frozen['premeasurement_predictions']):
            for t in ('V1', 'V2'):
                if rec[f'pred_{t}_ml'] != pred[f'pred_{t}_ml'] or rec[f'error_{t}_ml'] != pred[f'pred_{t}_ml']-rec[f'true_{t}_ml']:
                    raise RuntimeError('residual/prediction mismatch')
        history.append(measurement)
        all_batch_ids += frozen['batch_ids']
        ledger = read(d/'ledger_after_selection.json')
        if not (d/'training_complete.json').exists():
            phase = 'advance_resume_training'
            break
        trained = read(d/'training_complete.json')
        if trained['measurement_sha256'] != file_hash(d/'measurement.json'):
            raise RuntimeError('training measurement lineage drift')
        for path, expected in trained['artifact_hashes'].items():
            if file_hash(path) != expected:
                raise RuntimeError('trained predictor drift')
        completed = r+1
    else:
        phase = 'complete_no_test_evaluation'
    # Detect holes instead of trusting a stale status file or skipping an earlier failure.
    if phase != 'complete_no_test_evaluation':
        for future in range(r+1, 1):
            if (directory(root, seed, method, future)/'contract.json').exists():
                raise RuntimeError('noncontiguous round artifacts')
    return {'round': completed, 'active_label_count': initial_count+len(all_batch_ids),
            'next_action': phase, 'ledger': ledger, 'history': history, 'acquired_ids': all_batch_ids}


def _stage(root, seed, method, r):
    protocol.validate(root)
    d = directory(root, seed, method, r)
    current = state(root, seed, method)
    if current['round'] != r or current['next_action'] not in ('stage', 'select'):
        raise RuntimeError('requested stage does not match canonical state')
    if (d/'contract.json').exists():
        audit_stage(root, seed, method, r)
        packet = read(d/'input.json')
    else:
        if r == 0:
            raw = read(Path(root)/f'initial/seed_{seed}/{method}.json')
            checkpoint = read(Path(root)/'protocol.json')['initial'][str(seed)]['checkpoint']
        else:
            from .predictor import build_catalog
            raw, checkpoint = build_catalog(root, seed, method, r, current)
        catalog = ReadOnlyCatalog(**raw, ledger=current['ledger'])
        latest = current['history'][-1] if current['history'] else {}
        prior_selection = read(directory(root, seed, method, r-1)/'selection.json') if r else {}
        packet = make_packet(catalog, current['ledger'], seed=seed, method=method, round_index=r,
            selection_count=32,
            latest_ids=[rec['candidate_id'] for rec in latest.get('records', [])],
            prior_interpretation=prior_selection.get('feedback_interpretation', ''),
            open_questions=prior_selection.get('open_scientific_questions', []),
            previous_choices=prior_selection.get('choices', []))
        build_request(packet, empty_working_state(), [])  # reject oversized initial packet before sealing
        once(d/'catalog.json', raw)
        once(d/'input.json', packet)
        once(d/'ledger_before.json', current['ledger'])
        once(d/'contract.json', {'seed': seed, 'method': method, 'round': r,
            'protocol_sha256': file_hash(Path(root)/'protocol.json'), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': file_hash(checkpoint), 'L_t_ids': list(catalog.observed),
            'U_t_ids': list(catalog.candidates)+list(catalog.pending),
            'artifact_hashes': {name: file_hash(d/name) for name in ('catalog.json', 'input.json', 'ledger_before.json')}})
    _, size = build_request(packet, empty_working_state(), [])
    return {'status': 'STAGED_LABEL_SAFE', 'packet': str(d/'input.json'), 'context_budget': size,
            'new_labels_revealed': 0, 'llm_calls': 0, 'fits': 0}


def stage(seed, method, r=0, root=STUDY):
    with operation_lock(root):
        return _stage(Path(root), seed, method, r)


def freeze_batch(root, seed, method, r, result, merged, catalog):
    """Only called by explicit select; tests use synthetic temporary studies."""
    protocol.validate(root)
    d = directory(root, seed, method, r)
    packet = read(d/'input.json')
    validated, expected = validate_selection(result, catalog, packet, read(d/'ledger_before.json'))
    if validated != read(d/'selection.json') or expected != merged or merged != read(d/'ledger_after_selection.json'):
        raise RuntimeError('selection/ledger artifact mismatch')
    batch = list(catalog.pending)+[c['id'] for c in result['choices']]
    if len(batch) != 32 or len(set(batch)) != 32 or not set(batch) <= set(read(d/'contract.json')['U_t_ids']):
        raise RuntimeError('invalid scientific batch')
    protocol.lock_protocol(Path(root))
    lookup = {**catalog.pending, **catalog.candidates}
    payload = {'batch_ids': batch, 'pending_ids': list(catalog.pending),
        'premeasurement_predictions': [{'candidate_id': i, 'pred_V1_ml': lookup[i]['pred_V1_ml'],
                                       'pred_V2_ml': lookup[i]['pred_V2_ml']} for i in batch],
        'protocol_lock_sha256': file_hash(Path(root)/'protocol.lock.json'), 'new_batch_labels_revealed': False}
    audit = Audit(d, packet)
    audit.append('batch_freeze_intent', payload, turn=None, request_sha256=None)
    names = ('input.json', 'catalog.json', 'ledger_before.json', 'ledger_after_selection.json',
             'contract.json', 'selection.json', 'audit.jsonl', 'selector.started.json')
    names = list(names)+[p.name for p in sorted(d.glob('turn_*.json'))]+[p.name for p in sorted(d.glob('context_audit_*.json'))]
    once(d/'batch_freeze.json', {**payload, 'artifact_hashes': {name: file_hash(d/name) for name in names}})
    # Seal selection stream; a separate freeze event can name the resulting batch hash without a hash cycle.
    receipt = {'batch_freeze_sha256': file_hash(d/'batch_freeze.json'),
               'protocol_lock_sha256': file_hash(Path(root)/'protocol.lock.json')}
    Audit(d/'freeze_audit', packet).append('batch_frozen', receipt, turn=None, request_sha256=None)
    once(d/'freeze_receipt.json', {**receipt, 'audit_sha256': file_hash(d/'freeze_audit/audit.jsonl')})
    audit_batch(root, seed, method, r)
    return {'status': 'BATCH_FROZEN_BEFORE_REVEAL', 'batch_count': 32, 'new_labels_revealed': 0}


def select(seed, method, r=0, root=STUDY, *, call=None):
    with operation_lock(root):
        protocol.validate(root)
        d = directory(root, seed, method, r)
        if (d/'batch_freeze.json').exists():
            audit_batch(root, seed, method, r)
            return {'status': 'ALREADY_FROZEN'}
        _stage(Path(root), seed, method, r)
        packet, raw, ledger = read(d/'input.json'), read(d/'catalog.json'), read(d/'ledger_before.json')
        catalog = ReadOnlyCatalog(**raw, ledger=ledger)
        catalog.initial_cards()
        result, merged = run_selector(packet, catalog, ledger, d, read(Path(root)/'protocol.json')['transport'],
                                      call=call, guard=lambda: protocol.validate(root))
        return freeze_batch(root, seed, method, r, result, merged, catalog)


def advance(seed, method, r=0, root=STUDY):
    with operation_lock(root):
        protocol.validate(root)
        current = state(root, seed, method)
        if current['round'] != r or current['next_action'] not in ('advance', 'advance_resume_training'):
            raise RuntimeError('advance does not match canonical state')
        from .predictor import measure_and_fit
        return measure_and_fit(Path(root), seed, method, r, current)


def status(root=STUDY):
    protocol.validate(root)
    return [{'seed': seed, 'method': method, **{k: v for k, v in state(root, seed, method).items()
            if k in ('round', 'active_label_count', 'next_action')}} for seed in SEEDS for method in METHODS]

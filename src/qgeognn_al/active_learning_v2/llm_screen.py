"""Auditable CW16 + conversation-LLM16 development screen.

Selection functions take features, never labels. Historical training helpers
are reused without changing the predictor, optimizer, or stopping rule.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import TRAINING_CONFIG, load_features, sketch_seed
from .benchmark_reporting import metric_row
from .gradient_features import extract_q50_gradient_sketches, extract_linear_output_gradient_sketches
from .gradient_transforms import center_width_transform
from .ivr import conditional_batch_ivr
from .lcmd import lcmd_tp_select, _squared_distances
from .maxdet import conditional_gradient_maxdet
from .maxdet_study import historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import predict_outputs
from .short_sequential_runner import ShortSequentialContext, _round0_evaluation, _write_json_once
from .short_sequential_study import STUDY as CW_STUDY, SEEDS, ACTIVE_LABEL_BUDGETS

STUDY = ROOT / 'studies/active_learning/qgeognn_v2_row_llm16_screen'
METHODS = ('cw16_random16', 'cw16_llm16')
PROMPT = '''Select exactly 16 distinct candidate IDs to supplement 16 already-pending CW selections.
The objective is to improve overall V1/V2 prediction accuracy of the unchanged QGeoGNN,
not to optimize the experimental outcome itself. Use chemical structure, solvent/loading
conditions, predicted V1/V2, coverage distance and labeled-set structure coverage.
Favor complementary information and batch diversity; avoid spending many picks on nearly
identical structures/conditions. Predictions are estimates, not observed outcomes.
Distances are geometric coverage signals, not calibrated uncertainty. The pending CW16
have no observed labels. All 32 labels will be revealed together after selection freezes.
Only IDs in candidates are eligible. Use only this packet for this decision; do not read
other trajectories, result tables, hidden labels, or external sources. Prior selection
reasons and earlier packets must not be used as additional evidence.
No individual observed outcomes are supplied in this conversation-mode variant to avoid
cross-seed outcome leakage. No ensemble is trained. This tests an LLM selector package,
and cannot isolate the contribution of chemical reasoning from numeric signal use.
Return JSON: {"packet_sha256": "<provided hash>", "choices":
[{"id": "candidate ID", "reason": "one specific sentence"}, ...]}.
'''


def read(path):
    return json.loads(Path(path).read_text())


def rng(seed, round_index, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(round_index), int(stream), 1616]))


def shortlist(cw, raw, labeled_count, seed, round_index):
    """Condition feature-only proposals on pending CW16; retain all reference rows."""
    cw, raw = np.asarray(cw), np.asarray(raw)
    n = int(labeled_count)
    if cw.shape != raw.shape or not 0 < n <= len(cw) - 144:
        raise ValueError('aligned L+U banks with at least 144 U rows required')
    if not np.isfinite(cw).all() or not np.isfinite(raw).all():
        raise ValueError('nonfinite feature bank')
    pending = lcmd_tp_select(cw[n:], cw[:n], 16).selected_pool_positions + n
    centers = np.r_[np.arange(n), pending]
    remaining = np.array([i for i in range(n, len(cw)) if i not in set(pending)])
    proposals = {
        'cw': remaining[lcmd_tp_select(cw[remaining], cw[centers], 40).selected_pool_positions].tolist(),
        'ivr': remaining[conditional_batch_ivr(raw, centers, remaining, 40)['selected_candidate_positions']].tolist(),
        'maxdet': conditional_gradient_maxdet(raw, centers, remaining, 40).selected_positions.tolist(),
        'random': rng(seed, round_index, 1).choice(remaining, 8, replace=False).tolist(),
    }
    union = list(dict.fromkeys(p for values in proposals.values() for p in values))
    rest = [p for p in remaining if p not in set(union)]
    fill = rng(seed, round_index, 2).choice(rest, 128 - len(union), replace=False).tolist()
    union += fill
    # Independent reproducible presentation, control and failure-fill streams.
    presentation = rng(seed, round_index, 3).permutation(union).tolist()
    return {'pending_positions': pending.tolist(), 'candidate_positions': presentation,
            'proposals': proposals, 'random_backfill': fill,
            'random16_positions': rng(seed, round_index, 4).choice(presentation, 16, replace=False).tolist(),
            'fallback_positions': rng(seed, round_index, 5).permutation(presentation).tolist()}


def validate_response(response, candidate_ids, fallback_ids, packet_hash):
    """One attempt, retain valid choices, deterministically fill invalid/missing IDs."""
    if len(candidate_ids) != 128 or len(set(candidate_ids)) != 128 or set(fallback_ids) != set(candidate_ids):
        raise ValueError('invalid candidate or fallback universe')
    errors, accepted = [], []
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = None
            errors.append('invalid_json')
    if not isinstance(response, dict):
        errors.append('response_not_object')
        response = {'packet_sha256': packet_hash}
    if response.get('packet_sha256') != packet_hash:
        raise ValueError('response is not bound to this frozen packet')
    choices = response.get('choices', [])
    if not isinstance(choices, list):
        choices = []
        errors.append('choices_not_list')
    if len(choices) != 16:
        errors.append('wrong_choice_count')
    for item in choices[:16]:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str)
                or item['id'] not in candidate_ids or item['id'] in [v['id'] for v in accepted]
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
            errors.append('invalid_duplicate_or_missing_reason')
            continue
        accepted.append({'id': item['id'], 'reason': item['reason'], 'origin': 'llm'})
    valid = len(accepted)
    for value in fallback_ids:
        if len(accepted) == 16:
            break
        if value not in [v['id'] for v in accepted]:
            accepted.append({'id': value, 'reason': 'Preregistered random fallback', 'origin': 'fallback'})
    return {'choices': accepted, 'errors': errors, 'fallback_count': 16-valid,
            'response_failed': bool(errors), 'packet_sha256': packet_hash}


def code_hashes():
    paths = [Path(__file__), ROOT / 'scripts/studies/run_qgeognn_v2_row_llm16_screen.py']
    paths += sorted((ROOT / 'src/qgeognn_al/models').glob('*.py'))
    paths += [ROOT / f'src/qgeognn_al/{p}' for p in (
        'training/predictor.py', 'evaluation/point.py', 'active_learning_v2/runner.py',
        'active_learning_v2/short_sequential_runner.py', 'active_learning_v2/benchmark_protocol.py',
        'active_learning_v2/gradient_features.py', 'active_learning_v2/gradient_transforms.py',
        'active_learning_v2/ivr.py', 'active_learning_v2/maxdet.py', 'active_learning_v2/lcmd.py',
        'active_learning_v2/protocol.py')]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def prepare():
    if (STUDY / 'protocol.json').exists():
        return validate()
    historic = read(CW_STUDY / 'protocol.json')
    assert historic['training'] == TRAINING_CONFIG
    assert historic['seeds'] == list(SEEDS) and historic['active_label_budgets'] == list(ACTIVE_LABEL_BUDGETS)
    assert historic['source_sha256'] == sha256_file(SOURCE_DATA)
    assert historic['graph_cache_sha256'] == sha256_file(SOURCE_GRAPH_CACHE)
    ignored_reporting_drift = []
    for relative, digest in historic['code_hashes'].items():
        if sha256_file(ROOT / relative) != digest:
            if relative == 'src/qgeognn_al/active_learning_v2/short_sequential_reporting.py':
                ignored_reporting_drift.append(relative)
                continue  # Metrics are recomputed from frozen predictions below.
            raise RuntimeError(f'CW historical code drift: {relative}')
    inputs = {str(p.relative_to(ROOT)): sha256_file(p) for p in [SOURCE_DATA, SOURCE_GRAPH_CACHE, CW_STUDY / 'protocol.json']}
    for seed in SEEDS:
        partition = pd.read_csv(CW_STUDY / 'splits' / f'row_seed_{seed}.csv')
        validate_row_protocol(partition)
        assert partition.sample_id.tolist() == load_features().sample_id.tolist()
        paths = historical_round0_paths(seed)
        ctx = read(paths['context'])['contract']
        assert ctx['source_sha256'] == historic['source_sha256']
        assert ctx['source_graph_cache_sha256'] == historic['graph_cache_sha256']
        required = [paths[k] for k in ('context', 'scrubbed_graphs', 'member0_model', 'member0_predictions',
                                      'member0_audit', 'gradient', 'gradient_contract')]
        required += [CW_STUDY / 'splits' / f'row_seed_{seed}.csv']
        base = CW_STUDY / 'runtime' / f'seed_{seed}' / 'center_width_lcmd'
        trajectory = read(base / 'trajectory_freeze.json')
        assert trajectory['final_active_labels'] == 429
        for r in range(4):
            freeze_path = base / f'round_{r:02d}' / 'round_freeze.json'
            frozen = read(freeze_path)
            assert sha256_file(freeze_path) == trajectory['round_freeze_sha256'][r]
            assert frozen['input']['active_label_count'] == ACTIVE_LABEL_BUDGETS[r]
            for kind in ('checkpoint', 'prediction'):
                p = Path(frozen[f'{kind}_path'])
                assert sha256_file(p) == frozen[f'{kind}_sha256']
                required.append(p)
            required += [freeze_path, base / f'round_{r:02d}' / 'state.csv']
        bank = base / 'round_00/acquisition_artifacts/current_transformed_gradient_features.npz'
        required += [bank, bank.with_suffix('.npz.contract.json')]
        for p in required:
            inputs[str(p.relative_to(ROOT))] = sha256_file(p)
    protocol = {
        'study': STUDY.name, 'status': 'FROZEN_BEFORE_SELECTION', 'seeds': list(SEEDS),
        'evidence': 'conversation_model_development_screen_not_independent_confirmation',
        'methods': ['center_width_lcmd', *METHODS], 'budgets': list(ACTIVE_LABEL_BUDGETS),
        'training': TRAINING_CONFIG, 'new_training_fits': 12, 'llm_decisions': 6,
        'cw_batch': 16, 'supplement_batch': 16, 'candidate_count': 128,
        'candidate_rule': '40 CW + 40 IVR + 40 MaxDet + 8 random; deduplicate; random fill to 128',
        'pending_conditioning': 'L_t plus pending CW16 feature rows; no label reveal; IVR reference L_t+U_t',
        'maxdet_scale': 'RMS of L_t plus pending CW16 features',
        'raw_features': '512D current-model q50 gradients; original L0 target scales',
        'cw_features': '512D current-model center/width gradients; original L0 C/W scales',
        'rng': 'SeedSequence([seed, source_round, stream, 1616]); streams 1..5 in code',
        'prompt': PROMPT, 'llm_provider': 'current Codex conversation; user explicitly selected',
        'model_snapshot': None, 'model_snapshot_status': 'not exposed; exact replay not guaranteed',
        'historical_reporting_only_drift': ignored_reporting_drift,
        'observed_truth_in_packets': False, 'ensemble': False,
        'context_limitation': 'persistent conversation; no outcome/metric exposure until all selections frozen',
        'fallback': 'one attempt; valid first-16 entries retained; fixed independent random permutation fills',
        'primary_metric': 'AULC=(E333+2*E365+2*E397+E429)/6; E=combined normalized RMSE',
        'normalization': 'sqrt(0.5*((RMSE_V1/sV1)^2+(RMSE_V2/sV2)^2)); scales fixed from L0',
        'go_gate': 'LLM lower mean AULC than both controls and lower AULC on each of both seeds against each control',
        'attribution': 'LLM selector incremental value; chemical reasoning not isolated',
        'test_barrier': 'all 4 new trajectories and all 16 new-arm prediction points frozen; CW historically exposed',
        'code_hashes': code_hashes(), 'input_hashes': inputs,
    }
    STUDY.mkdir(parents=True, exist_ok=True)
    atomic_json(STUDY / 'protocol.json', protocol)
    (STUDY / 'prompt.txt').write_text(PROMPT)
    return {'status': 'PREPARED', 'historical_CW_reuse': 'verified', 'new_fits': 12, 'llm_calls': 6}


def validate():
    protocol = read(STUDY / 'protocol.json')
    if protocol['code_hashes'] != code_hashes() or (STUDY / 'prompt.txt').read_text() != PROMPT:
        raise RuntimeError('frozen code/prompt drift')
    for relative, digest in protocol['input_hashes'].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f'frozen input drift: {relative}')
    return {'status': 'VALID', 'protocol_hash': stable_hash(protocol)}


class Context(ShortSequentialContext):
    """Use the original fit method with a separate study and frozen input contract."""
    def __init__(self, seed):
        self.seed, self.study = int(seed), STUDY
        self.runtime = STUDY / 'runtime' / f'seed_{seed}'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(CW_STUDY / 'splits' / f'row_seed_{seed}.csv')
        validate_row_protocol(self.partition)
        self.data = load_features()
        self.roles = {r: self.partition.loc[self.partition.role.eq(r), 'canonical_index'].to_numpy(int)
                      for r in ('l0', 'u0', 'validation', 'test')}
        old = read(historical_round0_paths(seed)['context'])['contract']
        self.preprocessing = old['preprocessing']
        self.normalization = ConditionNormalization(**old['normalization'])
        self.atom, self.angle = torch.load(historical_round0_paths(seed)['scrubbed_graphs'], weights_only=False)
        assert all(not torch.count_nonzero(a.y) for a in self.atom)
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles['l0']), 'initial_fit')
        self.validation_truth = store.reveal(self.ids(self.roles['validation']), 'initial_fit')
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = stable_hash(read(STUDY / 'protocol.json'))
        _write_json_once(self.runtime / 'context.json', {
            'protocol_hash': self.protocol_hash, 'preprocessing': self.preprocessing,
            'normalization': asdict(self.normalization), 'split_hash': stable_hash(self.partition.to_dict('list'))})


def banks(context, current, model_path, source_round, directory):
    path = directory / 'banks.npz'
    contract = {'indices': current.tolist(), 'checkpoint': sha256_file(model_path), 'protocol': context.protocol_hash}
    if path.exists():
        receipt = read(path.with_suffix('.json'))
        assert receipt['contract'] == contract and receipt['sha256'] == sha256_file(path)
        with np.load(path) as saved:
            return saved['cw'], saved['raw']
    if source_round == 0:
        cw_path = CW_STUDY / 'runtime' / f'seed_{context.seed}' / 'center_width_lcmd/round_00/acquisition_artifacts/current_transformed_gradient_features.npz'
        raw_path = historical_round0_paths(context.seed)['gradient']
        values = []
        for p in (cw_path, raw_path):
            saved_contract = read(p.with_suffix('.npz.contract.json'))
            assert saved_contract['sha256'] == sha256_file(p)
            with np.load(p) as saved:
                assert np.array_equal(saved['canonical_indices'], current)
                values.append(saved['features'])
        cw, raw = values
    else:
        model = load_predictor_checkpoint(model_path)
        common = dict(dimension=512, sketch_seed=sketch_seed(context.seed))
        cw = extract_linear_output_gradient_sketches(model, context.atom, context.angle, current, context.cw_transform, **common).features
        raw = extract_q50_gradient_sketches(model, context.atom, context.angle, current,
                tuple(context.preprocessing['target_scales'][k] for k in ('V1', 'V2')), **common).features
    np.savez_compressed(path, cw=cw, raw=raw)
    atomic_json(path.with_suffix('.json'), {'contract': contract, 'sha256': sha256_file(path)})
    return cw, raw


def make_packet(context, current, n, model_path, cw, proposal):
    positions = proposal['pending_positions'] + proposal['candidate_positions']
    indices = current[positions]
    model = load_predictor_checkpoint(model_path)
    preds, order = predict_outputs(model, context.atom, context.angle, indices)
    assert np.array_equal(order, indices)
    distance = np.sqrt(_squared_distances(cw[n:], cw[:n]).min(axis=1))
    dist_rank = pd.Series(distance).rank(method='average', pct=True).to_numpy()
    smiles = context.data.iloc[current].canonical_smiles.tolist()
    mols = {s: Chem.MolFromSmiles(s) for s in dict.fromkeys(smiles)}
    if any(m is None for m in mols.values()):
        raise ValueError('invalid SMILES')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: generator.GetFingerprint(m) for s, m in mols.items()}
    labeled_smiles = smiles[:n]
    labeled_fps = [fps[s] for s in dict.fromkeys(labeled_smiles)]
    cards = []
    for p, idx, pred in zip(positions, indices, preds):
        row = context.data.iloc[idx]
        s, mol = row.canonical_smiles, mols[row.canonical_smiles]
        card = {'id': str(row.sample_id), 'smiles': s,
                'conditions': {k: row[k].item() if hasattr(row[k], 'item') else row[k]
                               for k in ('PE/EA', 'Density g/ml', 'V/ul', 'loading solvent', 'Volume of loading solvent/ul')},
                'descriptors': {'MW': round(Descriptors.MolWt(mol), 2), 'LogP': round(Descriptors.MolLogP(mol), 2),
                                'TPSA': round(Descriptors.TPSA(mol), 2), 'HBD': Descriptors.NumHDonors(mol),
                                'HBA': Descriptors.NumHAcceptors(mol)},
                'pred_V1_V2': [round(float(pred[1]), 3), round(float(pred[4]), 3)],
                'cw_distance_percentile': round(float(dist_rank[p-n]), 4),
                'max_Tanimoto_to_Lt': round(max(DataStructs.BulkTanimotoSimilarity(fps[s], labeled_fps)), 4),
                'same_molecule_rows_in_Lt': labeled_smiles.count(s)}
        cards.append(card)
    return {'prompt': PROMPT, 'labeled_count': n, 'pending_CW16': cards[:16], 'candidates': cards[16:],
            'units': {'pred_V1_V2': 'mL', 'MW': 'g/mol', 'TPSA': 'angstrom^2'},
            'observed_outcome_values_in_packet': 0, 'validation_or_test_records_in_packet': 0}


def acquisition(context, method, r, labeled, unlabeled, model_path, directory):
    # All methods share packet schema/rule, but never a foreign trajectory's checkpoint.
    audit_dir = STUDY / 'selections' / f'seed_{context.seed}' / method / f'round_{r:02d}'
    audit_dir.mkdir(parents=True, exist_ok=True)
    frozen = audit_dir / 'batch_freeze.json'
    contract = {'protocol': context.protocol_hash, 'checkpoint': sha256_file(model_path),
                'labeled': context.ids(labeled), 'unlabeled': context.ids(unlabeled)}
    if frozen.exists():
        result = read(frozen)
        assert result['input_hash'] == stable_hash(contract)
        for name, digest in result['artifact_hashes'].items():
            assert sha256_file(audit_dir / name) == digest
        return result['batch_ids']
    current, n = np.r_[labeled, unlabeled], len(labeled)
    if not (audit_dir / 'packet.json').exists():
        cw, raw = banks(context, current, model_path, r, directory)
        proposal = shortlist(cw, raw, n, context.seed, r)
        packet = make_packet(context, current, n, model_path, cw, proposal)
        _write_json_once(audit_dir / 'proposal.json', {'input_hash': stable_hash(contract), **proposal})
        _write_json_once(audit_dir / 'packet.json', packet)
    proposal, packet = read(audit_dir / 'proposal.json'), read(audit_dir / 'packet.json')
    assert proposal['input_hash'] == stable_hash(contract)
    packet_hash = sha256_file(audit_dir / 'packet.json')
    candidate_ids = [c['id'] for c in packet['candidates']]
    assert candidate_ids == context.ids(current[proposal['candidate_positions']])
    if method == 'cw16_llm16':
        response_path = audit_dir / 'response.json'
        if not response_path.exists():
            return {'status': 'AWAITING_LLM', 'packet': str(audit_dir / 'packet.json'), 'packet_sha256': packet_hash}
        response = response_path.read_text()
        supplement = validate_response(response, candidate_ids, context.ids(current[proposal['fallback_positions']]), packet_hash)
    else:
        supplement = {'choices': [{'id': v, 'reason': 'Uniform random from same shortlist', 'origin': 'random'}
                                  for v in context.ids(current[proposal['random16_positions']])],
                      'fallback_count': 0, 'response_failed': False}
    batch = context.ids(current[proposal['pending_positions']]) + [v['id'] for v in supplement['choices']]
    assert len(batch) == len(set(batch)) == 32 and set(batch) <= set(context.ids(unlabeled))
    _write_json_once(audit_dir / 'validated_response.json', supplement)
    names = ['packet.json', 'proposal.json', 'validated_response.json'] + (['response.json'] if method == 'cw16_llm16' else [])
    _write_json_once(frozen, {'input_hash': stable_hash(contract), 'batch_ids': batch,
                            'artifact_hashes': {k: sha256_file(audit_dir / k) for k in names},
                            'new_batch_labels_revealed': False, 'test_truth_access_count': 0})
    return batch


def execute(seed, method):
    if seed not in SEEDS or method not in METHODS:
        raise ValueError('unregistered seed/method')
    validate()
    if (STUDY / 'global_pre_test_freeze.json').exists():
        raise RuntimeError('execution closed after test barrier')
    torch.set_num_threads(2)
    context = Context(seed)
    store = context.new_store()
    labeled, unlabeled, truth = context.roles['l0'].copy(), context.roles['u0'].copy(), context.l0_truth.copy()
    all_selected, round_files = [], {}
    for r, budget in enumerate(ACTIVE_LABEL_BUDGETS):
        assert len(labeled) == budget
        directory = context.runtime / method / f'round_{r:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        _write_json_once(directory / 'state.json', {'labeled': context.ids(labeled), 'unlabeled': context.ids(unlabeled)})
        if r == 0:
            model, prediction, fit = _round0_evaluation(context)
        else:
            print(json.dumps({'event': 'fit_start', 'seed': seed, 'method': method, 'budget': budget}), flush=True)
            fit = context.fit(method, r, labeled, truth, directory / 'model')
            model, prediction = directory / 'model/best.pt', directory / 'model/predictions.csv.gz'
        _write_json_once(directory / 'prediction_freeze.json', {
            'checkpoint_path': str(model), 'checkpoint_sha256': sha256_file(model),
            'prediction_path': str(prediction), 'prediction_sha256': sha256_file(prediction),
            'L_t_hash': ids_hash(context.ids(labeled)), 'initialization_hash': fit['initialization_hash'],
            'test_truth_access_count': 0})
        round_files[str(directory / 'prediction_freeze.json')] = sha256_file(directory / 'prediction_freeze.json')
        if r == 3:
            break
        outgoing = acquisition(context, method, r, labeled, unlabeled, model, directory)
        if isinstance(outgoing, dict):
            return outgoing
        lookup = dict(zip(context.ids(unlabeled), unlabeled))
        selected_indices = np.array([lookup[i] for i in outgoing])
        all_selected += outgoing
        assert len(all_selected) == len(set(all_selected))
        # Crucially both halves have frozen before ANY of this batch is revealed.
        store.freeze_acquisitions(all_selected)
        truth = np.vstack([truth, store.reveal(outgoing, 'after_acquisition_fit')])
        labeled = np.r_[labeled, selected_indices]
        unlabeled = np.array([i for i in unlabeled if i not in set(selected_indices)])
        atomic_json(context.runtime / method / 'label_access_audit.json', store.audit)
    _write_json_once(context.runtime / method / 'trajectory_freeze.json', {
        'status': 'FROZEN_BEFORE_TEST_TRUTH', 'round_files': round_files,
        'selected_ids': all_selected, 'final_active_labels': len(labeled), 'test_truth_access_count': 0})
    return {'status': 'TRAJECTORY_COMPLETE', 'seed': seed, 'method': method, 'final_labels': len(labeled)}


def report():
    validate()
    entries = {}
    for seed in SEEDS:
        for method in METHODS:
            base = STUDY / 'runtime' / f'seed_{seed}' / method
            trajectory = read(base / 'trajectory_freeze.json')
            assert trajectory['final_active_labels'] == 429 and len(trajectory['selected_ids']) == 96
            for p, digest in trajectory['round_files'].items():
                assert sha256_file(Path(p)) == digest
                record = read(p)
                for kind in ('checkpoint', 'prediction'):
                    assert sha256_file(Path(record[f'{kind}_path'])) == record[f'{kind}_sha256']
                entries[p] = digest
            for r in range(3):
                selection = STUDY / 'selections' / f'seed_{seed}' / method / f'round_{r:02d}'
                batch = read(selection / 'batch_freeze.json')
                assert batch['batch_ids'] == trajectory['selected_ids'][r*32:(r+1)*32]
                for name, digest in batch['artifact_hashes'].items():
                    assert sha256_file(selection / name) == digest
    assert len(entries) == 16
    _write_json_once(STUDY / 'global_pre_test_freeze.json', {'status': 'FROZEN_BEFORE_TEST_TRUTH', 'entries': entries})
    rows, access = [], []
    for seed in SEEDS:
        context = Context(seed)
        store = context.new_store()
        store.freeze_acquisitions([])
        store.freeze_predictions()
        truth = store.reveal(context.ids(context.roles['test']), 'final_test_evaluation')
        access += [{'seed': seed, **v} for v in store.audit]
        for method in ('center_width_lcmd', *METHODS):
            for r, budget in enumerate(ACTIVE_LABEL_BUDGETS):
                if method == 'center_width_lcmd':
                    frozen = read(CW_STUDY / 'runtime' / f'seed_{seed}' / method / f'round_{r:02d}/round_freeze.json')
                else:
                    frozen = read(context.runtime / method / f'round_{r:02d}/prediction_freeze.json')
                prediction = pd.read_csv(frozen['prediction_path'])
                assert prediction.sample_id.astype(str).tolist() == context.ids(context.roles['test'])
                rows.append({'seed': seed, 'method': method, 'budget': budget,
                             **metric_row(truth, prediction.drop(columns='sample_id').to_numpy(float), context.preprocessing['target_scales'])})
    results = STUDY / 'results'
    results.mkdir(exist_ok=True)
    curves = pd.DataFrame(rows)
    curves.to_csv(results / 'learning_curves.csv', index=False)
    pd.DataFrame(access).to_csv(results / 'test_label_access_audit.csv', index=False)
    areas = []
    for (seed, method), group in curves.groupby(['seed', 'method']):
        values = group.sort_values('budget').combined_normalized_RMSE.to_numpy()
        areas.append({'seed': seed, 'method': method, 'AULC_333_429': float(values @ np.array([1,2,2,1])/6),
                      'NRMSE_429': float(values[-1])})
    areas = pd.DataFrame(areas)
    areas.to_csv(results / 'aulc.csv', index=False)
    pivot = areas.pivot(index='seed', columns='method', values='AULC_333_429')
    comparisons = []
    for control in ('center_width_lcmd', 'cw16_random16'):
        delta = pivot.cw16_llm16 - pivot[control]
        comparisons.append({'control': control, 'mean_llm_minus_control': float(delta.mean()),
                            'wins_of_2': int((delta < 0).sum()), 'per_seed': delta.to_dict()})
    failures = [read(p) for p in (STUDY / 'selections').glob('seed_*/cw16_llm16/round_*/validated_response.json')]
    go = all(c['mean_llm_minus_control'] < 0 and c['wins_of_2'] == 2 for c in comparisons)
    decision = {'status': 'COMPLETED', 'decision': 'EXTEND_WITH_INDEPENDENT_SEEDS' if go else 'NO_EXTENSION_SIGNAL',
                'comparisons': comparisons, 'LLM_failed_rounds': sum(v['response_failed'] for v in failures),
                'fallback_picks': sum(v['fallback_count'] for v in failures),
                'evidence': 'two-seed three-round development only; chemical reasoning not isolated',
                'model_snapshot_reproducible': False}
    atomic_json(STUDY / 'decision.json', decision)
    means = areas.groupby('method')[['AULC_333_429', 'NRMSE_429']].mean()
    text = ['# CW16 + LLM16 开发筛查', '', '两开发 seed（157、6101），333→365→397→429；越低越好。', '',
            means.to_markdown(), '', '## 每 seed 结果', '', areas.to_markdown(index=False), '',
            '## 最终 V1/V2 指标', '', curves.loc[curves.budget.eq(429)].to_markdown(index=False), '',
            '## 决策', '', json.dumps(decision, ensure_ascii=False, indent=2), '',
            '本结果只评价当前对话 LLM 选择策略的增量收益；不证明化学推理有效，不是独立确认。',
            '沿用单模型训练，不新增 ensemble；为避免跨 seed 标签记忆，选点卡片不含真实观测结果。',
            '当前对话模型快照无法固定，历史测试集已在既有研究中暴露；不做显著性声明。']
    (STUDY / 'FINAL_REPORT.md').write_text('\n'.join(text))
    return decision

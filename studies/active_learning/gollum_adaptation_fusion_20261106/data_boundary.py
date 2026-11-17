"""Read-only, identity-checked historical state access without source responses."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = HERE.parent / 'representation_gp_benchmark_20261104'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(LEGACY))
from representations import audit_lcmd_state, file_hash
from diagnostics import fusion_features


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def group_partition(ids, smiles, seed):
    """Predeclare response-blind molecular groups, shared across budget states."""
    if len(ids) != len(smiles) or len(set(ids)) != len(ids):
        raise ValueError('unique aligned IDs required')
    roles = []
    for smi in smiles:
        u = int(hashlib.sha256(f'gollum_gp_split_v1:{seed}:{smi}'.encode()).hexdigest()[:16], 16) / 2**64
        roles.append('fit' if u < .6 else 'calibration' if u < .8 else 'check')
    return np.asarray(roles)


def assert_disjoint(*groups):
    sets = [set(map(str, g)) for g in groups]
    if any(len(s) != len(g) for s, g in zip(sets, groups)):
        raise PermissionError('duplicate IDs in an access role')
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise PermissionError('training/calibration/check identities overlap')


def load_state(seed, budget, root=ROOT):
    """Reuse only sealed features and already authorized L_t/development labels."""
    root = Path(root).resolve()
    old = root / LEGACY.relative_to(ROOT)
    key = f'seed_{seed}_L{budget}'
    state = audit_lcmd_state(root, seed, budget)
    meta = json.loads((old / 'cache' / (key + '.json')).read_text())
    cache = old / 'cache' / (key + '.npz')
    if file_hash(cache) != meta['cache_sha256'] or state.audit != meta['state']:
        raise ValueError('historical state/feature identity drift')
    source = root / 'experiments/e0_4g_baseline/canonical_4g.csv'
    if file_hash(source) != meta['source_sha256']:
        raise ValueError('source hash drift')
    with np.load(cache, allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    id_to_index = dict(zip(state.partition.sample_id.astype(str), state.partition.canonical_index.astype(int)))
    candidate_ids = meta['candidate_ids']
    if not set(candidate_ids) <= set(state.unlabeled_ids):
        raise PermissionError('candidates outside frozen U_t')
    expected = np.r_[state.labeled_indices, [id_to_index[x] for x in candidate_ids], state.validation_indices]
    if not np.array_equal(data['indices'], expected):
        raise ValueError('ordered feature identities changed')
    if data['train_y'].shape != (budget, 2) or data['validation_y'].shape != (416, 2):
        raise ValueError('label matrix identity mismatch')
    lmeta = json.loads((old / 'cache/llm.json').read_text())
    lp = old / 'cache/llm.npz'
    if file_hash(lp) != lmeta['cache_sha256']:
        raise ValueError('LLM feature cache drift')
    with np.load(lp, allow_pickle=False) as z:
        lookup = {int(k): i for i, k in enumerate(z['indices'])}
        positions = [lookup[int(k)] for k in expected]
        for name in ('llm_conditions', 'llm_template_b', 'llm_smiles'):
            data[name] = z[name][positions]
    indexed = state.partition.set_index('sample_id')
    smiles = indexed.loc[state.labeled_ids, 'canonical_smiles'].astype(str).to_numpy()
    roles = group_partition(state.labeled_ids, smiles, seed)
    subsets = {r: np.flatnonzero(roles == r) for r in ('fit', 'calibration', 'check')}
    assert_disjoint(*(np.asarray(state.labeled_ids)[subsets[r]].tolist() for r in subsets))
    assert_disjoint(*(smiles[subsets[r]].tolist() if False else sorted(set(smiles[subsets[r]])) for r in subsets))
    if any(len(i) < 20 for i in subsets.values()):
        raise ValueError('insufficient response-blind molecular split')
    return state, data, meta, subsets


def representations(data, fit_positions):
    """Fit fusion preprocessing on precisely the declared GP fitting rows."""
    remainder = np.array([i for i in range(len(data['indices'])) if i not in set(fit_positions)], dtype=int)
    order = np.r_[fit_positions, remainder]
    fused, audit = fusion_features(data['gnn'][order], data['llm_conditions'][order], len(fit_positions))
    inverse = np.argsort(order)
    return {'qwen': data['llm_conditions'], 'fusion': fused[inverse]}, audit

"""No raw label tables, no fit/score actions; synthetic losses only."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies import run_column_conditioned_multitask as r
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.transfer.column_conditioned_multitask import cache_frozen_prefix, TASKS
from src.qgeognn_al.transfer.pcgrad import pcgrad_backward, project_conflicting_gradients

@pytest.fixture(scope='module')
def feature_data():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = r.source_path()
    preprocessing = torch.load(source_path, map_location='cpu', weights_only=False)['preprocessing']
    source_cache = torch.load(r.SOURCE_GRAPH_CACHE, map_location='cpu', weights_only=False)
    data = {}
    for col in ('25g', '40g'):
        frame = pd.read_csv(r.PARENT / f'filtered_features_{col}.csv', usecols=r.FEATURES)
        assert 'V1_ml' not in frame and 'V2_ml' not in frame
        frame['V1_ml'], frame['V2_ml'] = 0., 0.
        cache = dict(source_cache)
        cache.update(torch.load(ROOT / f'studies/transfer/cross_column/data_audit/graph_cache_{col}_only.pt', map_location='cpu', weights_only=False))
        atoms, angles = build_model_data(frame, cache, None, preprocessing['scaler'])
        assert all(torch.count_nonzero(a.y) == 0 for a in atoms)
        cache_frozen_prefix(load_predictor_checkpoint(source_path), atoms, angles)
        data[col] = dict(frame=frame, atoms=atoms, angles=angles)
    return data

@pytest.fixture(scope='module')
def mixed(feature_data):
    atom = Batch.from_data_list([feature_data[c]['atoms'][i] for c in ('25g', '40g') for i in range(12)])
    angle = Batch.from_data_list([feature_data[c]['angles'][i] for c in ('25g', '40g') for i in range(12)])
    return atom, angle, torch.arange(24) % 3

def model(arm, seed=17):
    torch.manual_seed(seed)
    return r.build_multitask_model(r.source_path(), r.ARMS[arm])

def synthetic_losses(m, atom, angle, tasks):
    prediction = m(atom, angle, tasks)
    # Fixed arithmetic targets: no observed experimental outcomes.
    return {task: (prediction[tasks.eq(j)] - (j + 1) * 3.).square().mean() for j, task in enumerate(TASKS)}

def test_zero_film_source_equivalence_and_independent_heads(mixed):
    atom, angle, tasks = mixed
    source = load_predictor_checkpoint(r.source_path()).eval()
    a1, a2 = model('A1').eval(), model('A2').eval()
    assert r.head_copies_are_exact(a1) and r.head_copies_are_exact(a2) and r.film_is_zero_initialized(a2)
    assert len({a1.heads[t][0].weight.data_ptr() for t in TASKS}) == 3
    with torch.no_grad():
        expected = source(atom, angle)
        for task in range(3):
            ids = torch.full_like(tasks, task)
            torch.testing.assert_close(a1(atom, angle, ids), expected, rtol=0, atol=1e-6)
            torch.testing.assert_close(a2(atom, angle, ids), expected, rtol=0, atol=1e-6)

def test_routing_and_quantile_head_order(mixed):
    atom, angle, tasks = mixed
    m = model('A2').eval()
    with torch.no_grad():
        for j, task in enumerate(TASKS):
            m.heads[task][0].weight.zero_()
            m.heads[task][0].bias.copy_(torch.arange(6) + 1 + 10 * j)
        actual = m(atom, angle, tasks)
    torch.testing.assert_close(actual, torch.arange(6)[None, :] + 1. + tasks[:, None] * 10.)
    for ids in (torch.full_like(tasks, -1), torch.full_like(tasks, 3), tasks[:-1]):
        with pytest.raises(ValueError):
            m(atom, angle, ids)
    truth = np.zeros((24, 2))
    metrics = r.point_metrics(truth, actual.numpy(), {'V1': 1., 'V2': 1.})
    assert metrics['V1_rmse'] == pytest.approx(np.sqrt(np.square(actual.numpy()[:, 1]).mean()))
    assert metrics['V2_rmse'] == pytest.approx(np.sqrt(np.square(actual.numpy()[:, 4]).mean()))

def test_nonzero_film_uses_each_graph_task_identity(mixed):
    atom, angle, tasks = mixed
    m = model('A2')
    with torch.no_grad():
        m.column_embedding.weight.zero_()
        m.column_embedding.weight[:, 0] = torch.arange(3)
        for gen in m.film_generators.values():
            gen.weight.zero_()
            gen.bias.zero_()
            gen.weight[:128, 0] = .1
            gen.weight[128:, 0] = .5
        nodes = torch.ones((atom.num_nodes, 128))
        for layer in (3, 4):
            actual = m._apply_film(nodes, atom.batch, tasks, layer)
            expected = (1 + .6 * tasks[atom.batch].float())[:, None].expand_as(nodes)
            torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize('arm', ['A1', 'A2'])
def test_mixed_task_nonzero_film_cache_gradient_update_and_bn(mixed, arm):
    atom, angle, tasks = mixed
    uncached = atom.clone()
    del uncached.frozen_nodes
    del uncached.frozen_edges
    cached = model(arm)
    if arm == 'A2':
        with torch.no_grad():
            for gen in cached.film_generators.values():
                gen.weight.normal_(0, .003)
                gen.bias.normal_(0, .003)
    full = model(arm)
    full.load_state_dict(cached.state_dict())
    before = deepcopy(cached.state_dict())
    a, b = cached.optimizer(), full.optimizer()
    for m in (cached, full):
        r.set_training_mode(m)
        for name, module in m.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                expected = any(p.requires_grad for p in module.parameters(recurse=False))
                assert module.training == expected, name
    first, second = cached(atom, angle, tasks), full(uncached, angle, tasks)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
    first.square().mean().backward()
    second.square().mean().backward()
    for (name, p), (name2, q) in zip(cached.named_parameters(), full.named_parameters()):
        assert name == name2
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-4)
        else:
            assert p.grad is None and q.grad is None
    a.step()
    b.step()
    for key, value in cached.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[key], rtol=1e-5, atol=1e-5)
        if key.startswith(('backbone.convs.0.', 'backbone.convs.1.', 'backbone.convs.2.')):
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

def test_pcgrad_nonshared_mean_gradient_and_adam_update(mixed):
    atom, angle, tasks = mixed
    ordinary, candidate = model('A2'), model('A2')
    for m in (ordinary, candidate):
        r.set_training_mode(m)
    ao, ac = ordinary.optimizer(), candidate.optimizer()
    torch.stack(list(synthetic_losses(ordinary, atom, angle, tasks).values())).mean().backward()
    raw, projected, diagnostics = pcgrad_backward(synthetic_losses(candidate, atom, angle, tasks), candidate.shared_gradient_parameters(), permutation_seed=13)
    scope = {id(p) for p in candidate.shared_gradient_parameters()}
    assert len(scope) == 44
    for (_, p), (_, q) in zip(ordinary.named_parameters(), candidate.named_parameters()):
        if q.requires_grad and id(q) not in scope:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-5)
    ao.step()
    ac.step()
    for (_, p), (_, q) in zip(ordinary.named_parameters(), candidate.named_parameters()):
        if q.requires_grad and id(q) not in scope:
            torch.testing.assert_close(p, q, rtol=0, atol=1e-6)
    assert diagnostics.directed_projection_attempts == 6

def test_pcgrad_three_task_sequential_original_gradient_reference():
    original = {'a': (torch.tensor([1., -2.]),), 'b': (torch.tensor([-2., 1.]),), 'c': (torch.tensor([1., 1.]),)}
    expected = {k: v[0].clone() for k, v in original.items()}
    rng = np.random.default_rng(123)
    names = tuple(original)
    for i in rng.permutation(3):
        name = names[i]
        other = [k for k in names if k != name]
        for j in rng.permutation(2):
            ref = original[other[j]][0]
            dot = torch.dot(expected[name], ref)
            if dot < 0:
                expected[name] -= dot / (torch.dot(ref, ref) + 1e-12) * ref
    projected, _ = project_conflicting_gradients(original, permutation_seed=123)
    for name in names:
        torch.testing.assert_close(projected[name][0], expected[name], rtol=0, atol=0)

def test_frozen_final_checkpoints_replay_blind_predictions(feature_data):
    maxima = []
    for protocol in ('compound', 'row'):
        for seed in r.SEEDS:
            for arm in ('A1', 'A2'):
                run = r.STUDY / f'runtime/final/{protocol}/seed_{seed}/{arm}'
                checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
                m = model(arm)
                m.load_state_dict(checkpoint['model_state_dict'])
                saved = pd.read_csv(run / 'predictions_blind.csv')
                for col in ('25g', '40g'):
                    target = saved.loc[saved.column.eq(col)]
                    lookup = {str(sid): i for i, sid in enumerate(feature_data[col]['frame'].sample_id)}
                    positions = [lookup[str(sid)] for sid in target.sample_id]
                    actual = r.predict(m, feature_data, col, positions)
                    expected = target[r.QUANTILES].to_numpy(float)
                    delta = float(np.max(np.abs(actual - expected)))
                    assert delta < 1e-4, (protocol, seed, arm, col, delta)
                    maxima.append(delta)
    print('checkpoint blind replay: 20 checkpoints, 40 contexts, maximum absolute difference', max(maxima))

"""Check physical graph reconstruction and raw-condition identity without training."""
import copy
import importlib.util
import json
import lzma
from pathlib import Path
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('volume_sensitivity', HERE / 'sensitivity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope='module')
def inputs():
    data = json.loads(lzma.decompress((ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz').read_bytes()))
    cache = torch.load(ROOT / 'experiments/e0_4g_baseline/graph_cache_4g.pt', weights_only=False)
    context = json.loads((ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_157/scrubbed_graphs.pt.contract.json').read_text())
    sid = sorted(data['metadata'])[0]
    return sid, data['metadata'], cache, context['contract']['preprocessing']['scaler']


def test_complete_differentiable_builder_matches_original(inputs):
    sid, metadata, cache, scaler = inputs
    original, angle = module.ordinary_graphs([sid], metadata, cache, scaler)
    rebuilt, rebuilt_angle = module.differentiable_graph(sid, metadata, cache, scaler)
    torch.testing.assert_close(rebuilt.edge_attr, original[0].edge_attr, rtol=0, atol=2e-5)
    torch.testing.assert_close(rebuilt_angle.edge_attr, angle[0].edge_attr, rtol=0, atol=0)
    assert rebuilt.sample_id == sid
    assert not torch.count_nonzero(rebuilt.y)


def test_density_times_microliter_is_milligram_and_differentiable(inputs):
    sid, metadata, cache, scaler = inputs
    volume = torch.tensor(100., requires_grad=True)
    changed = copy.deepcopy(metadata[sid])
    changed['conditions']['Density g/ml'] = 1.2
    one = {sid: changed}
    graph, _ = module.differentiable_graph(sid, one, cache, scaler, raw_volume=volume)
    amount_col = len(module.qg.bond_id_names) + 8
    assert graph.edge_attr[0, amount_col].item() == pytest.approx(120.)
    derivative = torch.autograd.grad(graph.edge_attr[0, amount_col], volume)[0]
    assert derivative.item() == pytest.approx(1.2)
    assert graph.sample_volume_ul.item() == 100.


def test_product_alias_does_not_manufacture_independent_volume(inputs):
    sid, metadata, cache, scaler = inputs
    a = copy.deepcopy(metadata[sid]); b = copy.deepcopy(metadata[sid])
    a['conditions'].update({'Density g/ml': 1., 'V/ul': 100.})
    b['conditions'].update({'Density g/ml': 2., 'V/ul': 50.})
    ga, _ = module.ordinary_graphs([sid], {sid: a}, cache, scaler)
    gb, _ = module.ordinary_graphs([sid], {sid: b}, cache, scaler)
    torch.testing.assert_close(ga[0].edge_attr, gb[0].edge_attr, rtol=0, atol=0)
    assert ga[0].sample_volume_ul.item() != gb[0].sample_volume_ul.item()


def test_metadata_frame_labels_are_zero_sentinels(inputs):
    sid, metadata, _, _ = inputs
    frame = module.metadata_frame([sid], metadata)
    np.testing.assert_array_equal(frame[['V1_ml', 'V2_ml']].to_numpy(), np.zeros((1, 2)))
    assert frame.sample_id.tolist() == [sid]

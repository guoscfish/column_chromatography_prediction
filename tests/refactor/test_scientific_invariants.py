"""Synthetic numerical gates. No real labels/checkpoints and no training calls."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.qgeognn_al.models import build_predictor, load_predictor_checkpoint, predictor_checkpoint
from src.qgeognn_al.schemas.conditions import ConditionNormalization, fit_condition_normalization
from src.qgeognn_al.training.predictor import target_loss
from src.qgeognn_al.data import minmax_transform, fit_standardizer, apply_standardizer

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name('fixtures') / 'scientific_baseline.json'


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope='module')
def baseline():
    return json.loads(FROZEN.read_text())


@pytest.fixture(scope='module')
def model(baseline):
    torch.set_num_threads(1)
    torch.manual_seed(baseline['initialization_seed'])
    result = build_predictor(ConditionNormalization(0., 1., 0., 1., '4g', 'source_train', 2, '0'*64, '0'*64))
    result.eval()
    return result


def test_six_output_hidden_and_512_countsketch_match_committed_synthetic_baseline(baseline):
    observed = module('make_synthetic_baseline').generate()
    assert observed['state_dict_sha256'] == baseline['state_dict_sha256']
    assert observed['mapping_sha256'] == baseline['mapping_sha256']
    for key, shape in [('outputs', (3, 6)), ('hidden', (3, 128)), ('gradient_sketch', (3, 512))]:
        assert np.shape(observed[key]) == shape
        np.testing.assert_allclose(observed[key], baseline[key], **baseline['tolerance'])


def test_parameter_geometry_condition_and_output_contract(model):
    assert sum(p.numel() for p in model.parameters()) == 458952
    assert len(model.backbone.convs) == 5
    assert len(model.backbone.convs_bond_angle) == 4
    assert model.head[0].in_features == 128 and model.head[0].out_features == 6
    assert model.condition_branch.output.out_features == 128
    assert model.condition_branch.normalization.fit_role == 'source_train'
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.LayerNorm)) for m in model.modules())


def test_strict_synthetic_checkpoint_roundtrip_and_output_order(model, tmp_path):
    path = tmp_path / 'synthetic.pt'
    payload = predictor_checkpoint(model, preprocessing={'fit_role': 'source_train'}, training_config={},
                                   provenance={'fixture': 'synthetic_only', 'no_fitting': True})
    assert payload['output_order'] == ['V1_q10', 'V1_q50', 'V1_q90', 'V2_q10', 'V2_q50', 'V2_q90']
    torch.save(payload, path)
    restored = load_predictor_checkpoint(path)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    for corruption in ['missing', 'unexpected', 'order']:
        broken = dict(payload)
        broken['model_state_dict'] = dict(payload['model_state_dict'])
        if corruption == 'missing':
            broken['model_state_dict'].pop(next(iter(broken['model_state_dict'])))
        elif corruption == 'unexpected':
            broken['model_state_dict']['unexpected.weight'] = torch.ones(1)
        else:
            broken['output_order'] = list(reversed(payload['output_order']))
        torch.save(broken, path)
        with pytest.raises((RuntimeError, ValueError)):
            load_predictor_checkpoint(path)


def test_current_loss_keeps_q50_mse_soft_crossing_and_equal_endpoint_weights():
    truth = torch.tensor([1., 3., 2.])
    prediction = torch.tensor([[2., 0., 1.], [1., 2., 3.], [4., 1., 0.]])
    low = truth - prediction[:, 0]
    high = truth - prediction[:, 2]
    expected = (torch.maximum(.1 * low, -.9 * low).mean()
                + ((truth - prediction[:, 1]) ** 2).mean()
                + torch.maximum(.9 * high, -.1 * high).mean()
                + torch.relu(prediction[:, 0] - prediction[:, 1]).mean()
                + torch.relu(prediction[:, 1] - prediction[:, 2]).mean())
    torch.testing.assert_close(target_loss(truth, prediction), expected, rtol=0, atol=1e-6)
    tree = ast.parse((ROOT / 'src/qgeognn_al/training/predictor.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'loss' for t in n.targets))
    value = assignment.value
    assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
    assert all(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'target_loss'
               for call in [value.left, value.right])


def test_floating_preprocessing_and_condition_inputs_are_fixed(model):
    values = np.array([[0., 2., 7.], [2., 4., 7.]], dtype=np.float32)
    scaled = minmax_transform(values, {'min': [0., 2., 7.], 'max': [2., 4., 7.]})
    np.testing.assert_array_equal(scaled, [[0., 0., 0.], [1., 1., 0.]])
    mean, std = fit_standardizer(values)
    np.testing.assert_array_equal(apply_standardizer(values, mean, std), [[-1., -1., 0.], [1., 1., 0.]])
    from src.qgeognn_al.training.predictor import loader_pair
    atom, angle = module('synthetic_graphs').graph_pair(1)
    a, _ = next(zip(*loader_pair([atom], [angle], [0])))
    typed = model.condition_branch.typed_inputs(a)
    torch.testing.assert_close(typed[0, :4], torch.tensor([.5, .6, .7, .8]), rtol=0, atol=1e-6)


def test_normalization_uses_synthetic_training_rows_only(tmp_path):
    data = pd.DataFrame({'sample_id': ['train0', 'train1', 'sealed'], 'Density g/ml': [1., 2., 999.],
                         'V/ul': [2., 3., 999.], 'Volume of loading solvent/ul': [4., 8., 999.]})
    split = pd.DataFrame({'sample_id': ['train0', 'train1', 'sealed'], 'split': ['train', 'train', 'test']})
    scaler = tmp_path / 'scaler.json'
    scaler.write_text('{}')
    first = fit_condition_normalization(data, split, scaler)
    data.loc[2, ['Density g/ml', 'V/ul', 'Volume of loading solvent/ul']] = -99999
    assert fit_condition_normalization(data, split, scaler) == first
    assert (first.loading_amount_min, first.loading_amount_max, first.loading_volume_min, first.loading_volume_max) == (2., 6., 4., 8.)
    assert first.fit_row_count == 2 and first.fit_role == 'source_train'

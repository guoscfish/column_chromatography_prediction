"""Explicit regeneration utility; uses synthetic inputs and performs no fitting."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import torch
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.active_learning_v2.gradient_features import extract_q50_gradient_sketches, state_dict_hash
from src.qgeognn_al.training.predictor import loader_pair


def generate():
    spec = importlib.util.spec_from_file_location('synthetic_graphs', Path(__file__).with_name('synthetic_graphs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    model = build_predictor(ConditionNormalization(0., 1., 0., 1., '4g', 'source_train', 2, '0'*64, '0'*64))
    model.eval()
    atom, angle = zip(*(module.graph_pair(i) for i in range(3)))
    a, b = next(zip(*loader_pair(atom, angle, [0, 1, 2])))
    with torch.no_grad():
        output = model(a, b).numpy()
        hidden = model.extract_representation(a, b).numpy()
    sketch = extract_q50_gradient_sketches(model, atom, angle, [0, 1, 2], (2., 3.), dimension=512, sketch_seed=4000194)
    return {'schema_version': 1, 'source_commit': '16eca74769f664bc378a58ccbf380a806723f587',
            'fixture': 'synthetic_only', 'initialization_seed': 20261109, 'sketch_seed': 4000194,
            'torch': torch.__version__, 'numpy': np.__version__, 'tolerance': {'rtol': 0, 'atol': 1e-6},
            'state_dict_sha256': state_dict_hash(model), 'outputs': output.tolist(), 'hidden': hidden.tolist(),
            'gradient_sketch': sketch.features.tolist(), 'mapping_sha256': sketch.audit['sketch_mapping_sha256']}


if __name__ == '__main__':
    output = Path(__file__).with_name('fixtures') / 'scientific_baseline.json'
    if output.exists():
        raise SystemExit('Baseline already exists; review changes before regenerating explicitly.')
    payload = generate()
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k not in {'outputs', 'hidden', 'gradient_sketch'}}))

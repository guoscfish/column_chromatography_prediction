"""Read pinned historical source from Git objects, without importing live source.

Only named pure classes/functions are compiled; module-level IO, controller
entrypoints, provider imports and arbitrary scientific runners never execute.
This utility belongs to regression tests, not the stable runtime API.
"""
from __future__ import annotations
import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pytest

FROZEN_COMMIT = '16eca74769f664bc378a58ccbf380a806723f587'
PHASE1_COMMIT = '4ee480378b0293f1985e3ee4a09df2c78f77901e'


def source(path, commit=FROZEN_COMMIT):
    if commit not in (FROZEN_COMMIT, PHASE1_COMMIT):
        raise ValueError('unregistered immutable historical source')
    repository = Path(os.environ.get('QGEOGNN_HISTORY_REPOSITORY', Path(__file__).resolve().parents[2]))
    requested = Path(path)
    if requested.is_absolute() or '..' in requested.parts or requested.suffix != '.py':
        raise ValueError('historical regression reads named Python source objects only')
    result = subprocess.run(['git', '--no-optional-locks', '-C', str(repository),
                             'show', commit + ':' + path], capture_output=True,
                            env={**os.environ, 'GIT_NO_LAZY_FETCH':'1',
                                 'GIT_NO_REPLACE_OBJECTS':'1', 'GIT_TERMINAL_PROMPT':'0'})
    if result.returncode:
        pytest.skip('BLOCKED: pinned historical Git source missing: ' + commit + ':' + path +
                    '; explicitly fetch this commit or set QGEOGNN_HISTORY_REPOSITORY to its Git object store; no automatic fetch/live-source fallback')
    return result.stdout.decode()


def constant(path, name, commit=FROZEN_COMMIT):
    for node in ast.parse(source(path, commit)).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('missing pinned literal constant: ' + name)


def load(path, names, *, injected=None, commit=FROZEN_COMMIT):
    content = source(path, commit)
    tree = ast.parse(content, filename=path)
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in set(names)]
    found = {node.name for node in body}
    if found != set(names):
        raise ValueError('missing named frozen definitions: ' + repr(set(names)-found))
    name = '_qgeognn_regression_' + hashlib.sha256((commit + path + repr(sorted(names))).encode()).hexdigest()[:16]
    module = types.ModuleType(name)
    module.__dict__.update(np=np, math=math, time=time, dataclass=dataclass,
                           Sequence=Sequence, Mapping=Mapping, Callable=Callable)
    if injected:
        module.__dict__.update(injected)
    sys.modules[name] = module
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *body], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, 'git:' + commit + ':' + path, 'exec'), module.__dict__)
    module.__source_sha256__ = hashlib.sha256(content.encode()).hexdigest()
    return module


def lcmd():
    return load('src/qgeognn_al/active_learning_v2/lcmd.py', ['LCMDResult', '_squared_distances', 'lcmd_tp_select'])


def hybrid():
    distance = lcmd()._squared_distances
    uncertainty = load('src/qgeognn_al/active_learning_v2/uncertainty.py', ['ensemble_uncertainty_select'])
    coverage = load('src/qgeognn_al/active_learning_v2/coverage.py', ['coreset_tp_select'], injected={'_squared_distances': distance})
    return load('scripts/studies/run_qgeognn_v2_row_hybrid_extension.py', ['v2_hybrid_select'],
                injected={'BATCH_SIZE': 32, 'SHORTLIST_FRACTION': .25,
                          'ensemble_uncertainty_select': uncertainty.ensemble_uncertainty_select,
                          'coreset_tp_select': coverage.coreset_tp_select})


def maxdet():
    from scipy.linalg import solve_triangular
    return load('src/qgeognn_al/active_learning_v2/maxdet.py',
                ['MaxDetResult', '_positions', 'conditional_gradient_maxdet'],
                injected={'solve_triangular': solve_triangular})


def ivr():
    from scipy.linalg import cho_factor, cho_solve
    return load('src/qgeognn_al/active_learning_v2/ivr.py', ['conditional_batch_ivr'],
                injected={'cho_factor': cho_factor, 'cho_solve': cho_solve})


def gradient():
    import torch
    from torch_geometric.loader import DataLoader
    import hashlib
    import json
    loader = load('src/qgeognn_al/training/predictor.py', ['loader_pair'], injected={'DataLoader': DataLoader})
    return load('src/qgeognn_al/active_learning_v2/gradient_features.py',
                ['ParameterSlice', 'GradientSketchResult', 'parameter_slices', 'state_dict_hash',
                 'countsketch_mapping', 'sketch_flat_gradient', 'extract_q50_gradient_sketches',
                 'extract_linear_output_gradient_sketches'],
                injected={'torch': torch, 'hashlib': hashlib, 'json': json, 'loader_pair': loader.loader_pair})

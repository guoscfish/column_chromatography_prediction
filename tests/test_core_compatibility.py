from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import scripts.al_acquisition as historical_acquisition
import scripts.al_engine as historical_engine
from src.qgeognn_al import acquisition, engine
from src.qgeognn_al.data import build_model_data as src_build_model_data
from src.qgeognn_al.metrics import regression_metric_row
from src.qgeognn_al.model import build_model as src_build_model
from scripts.run_e0_8g_transfer import build_model as historical_build_model
from scripts.run_e0_8g_transfer import build_model_data as historical_build_model_data


ROOT = Path(__file__).resolve().parents[1]


def test_historical_and_src_acquisition_order_are_identical() -> None:
    scores = np.array([0.2, 0.9, 0.9, -0.1])
    ids = ["d", "b", "a", "c"]
    expected = acquisition.deterministic_descending_order(scores, ids)
    assert np.array_equal(expected, historical_acquisition.deterministic_descending_order(scores, ids))


def test_historical_engine_symbols_are_src_symbols() -> None:
    assert historical_engine.TrainConfig is engine.TrainConfig
    assert historical_engine.QGeoGNNActiveLearningEngine is engine.QGeoGNNActiveLearningEngine
    assert historical_engine.random_query is engine.random_query


def test_metric_and_fixture_prediction_bytes_are_identical() -> None:
    truth = np.array([[1.0, 2.0], [2.0, 4.0]])
    prediction = np.array([[1.5, 1.5], [1.5, 4.5]])
    first = regression_metric_row(truth, prediction, {"V1": 1.0, "V2": 2.0})
    second = regression_metric_row(truth.copy(), prediction.copy(), {"V1": 1.0, "V2": 2.0})
    assert first == second
    assert hashlib.sha256(prediction.tobytes()).hexdigest() == hashlib.sha256(prediction.copy().tobytes()).hexdigest()


def test_historical_and_src_model_parameter_hash_and_prediction_identical() -> None:
    frame = pd.read_csv(ROOT / "experiments/e0_4g_baseline/canonical_4g.csv").head(2)
    cache = torch.load(ROOT / "experiments/e0_4g_baseline/graph_cache_4g.pt", weights_only=False)
    scaler = json.loads((ROOT / "experiments/e0_4g_baseline/scaler.json").read_text(encoding="utf-8"))
    split = pd.DataFrame({"split": ["test", "test"]})
    old_atom, old_angle = historical_build_model_data(frame, cache, split, scaler)
    new_atom, new_angle = src_build_model_data(frame, cache, split, scaler)
    torch.manual_seed(2026)
    old_model = historical_build_model(torch.device("cpu")).eval()
    torch.manual_seed(2026)
    new_model = src_build_model(torch.device("cpu")).eval()
    def parameter_hash(model: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode()); digest.update(value.detach().contiguous().numpy().tobytes())
        return digest.hexdigest()
    assert parameter_hash(old_model) == parameter_hash(new_model)
    with torch.no_grad():
        old_prediction = old_model(next(iter(DataLoader(old_atom, batch_size=2))), next(iter(DataLoader(old_angle, batch_size=2))))[0]
        new_prediction = new_model(next(iter(DataLoader(new_atom, batch_size=2))), next(iter(DataLoader(new_angle, batch_size=2))))[0]
    assert torch.equal(old_prediction, new_prediction)

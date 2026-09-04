from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.qgeognn_al.data import build_model_data, qg
from src.qgeognn_al.engine import QGeoGNNActiveLearningEngine
from src.qgeognn_al.input_schema import (
    CATEGORICAL_BOND_FEATURES, CONTINUOUS_EDGE_FEATURES,
    LEGACY_CONSUMED_CONTINUOUS_POSITIONS, input_schema_hash,
    legacy_input_schema,
)
from src.qgeognn_al.resources import (
    SOURCE_CHECKPOINTS, SOURCE_DATA, SOURCE_GRAPH_CACHE, SOURCE_SCALER,
)


ROOT = Path(__file__).resolve().parents[1]
I0 = ROOT / "studies/i0_predictor_semantic_audit"


def test_input_feature_schema_matches_constructed_edge_attr() -> None:
    frame = pd.read_csv(SOURCE_DATA).head(1)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = json.loads(SOURCE_SCALER.read_text())
    atom, _ = build_model_data(frame, cache, pd.DataFrame(), scaler)
    assert atom[0].edge_attr.shape[1] == len(CATEGORICAL_BOND_FEATURES) + len(CONTINUOUS_EDGE_FEATURES)
    assert list(qg.bond_id_names) == list(CATEGORICAL_BOND_FEATURES)


def test_legacy_consumed_continuous_positions_are_explicit() -> None:
    schema = legacy_input_schema()
    consumed = [
        row["continuous_position"] for row in schema["continuous_feature_mapping"]
        if row["consumed_by_legacy_bond_float_rbf"]
    ]
    assert consumed == list(LEGACY_CONSUMED_CONTINUOUS_POSITIONS) == [0, 1, 2, 3, 4]
    assert schema["constructed_continuous_feature_count"] == 10
    assert schema["consumed_continuous_feature_count"] == 5


def test_loading_condition_perturbation_reachability() -> None:
    audit = pd.read_csv(I0 / "feature_perturbation_audit.csv")
    rows = audit[audit.feature.str.startswith("loading_")]
    assert len(rows) == 6
    assert not rows.feature_reaches_encoder.any()
    assert (rows.delta_prediction_max_abs == 0).all()


def test_eluent_feature_reachability() -> None:
    audit = pd.read_csv(I0 / "feature_perturbation_audit.csv").set_index("feature")
    for name in ("eluent_exact_mol_wt", "eluent_tpsa", "eluent_rotatable_bonds", "eluent_h_donors"):
        assert bool(audit.loc[name, "feature_reaches_encoder"])
    for name in ("eluent_h_acceptors", "eluent_logp"):
        assert not bool(audit.loc[name, "feature_reaches_encoder"])


def test_gradient_reachability_audit() -> None:
    audit = json.loads((I0 / "effective_parameter_audit.json").read_text())
    assert audit["nominal_parameters"] == 775476
    assert audit["gradient_bearing_parameters"] < audit["nominal_parameters"]
    assert audit["nn_descriptor_forward_unreachable"]
    assert audit["outer_geometry_batch_norms_forward_unreachable"]


def test_legacy_prediction_unchanged_by_i0_refactor() -> None:
    frame = pd.read_csv(SOURCE_DATA).head(2)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(frame, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    first = engine.predict(frame.sample_id.tolist(), SOURCE_CHECKPOINTS[42], return_embedding=False).table
    second = engine.predict(frame.sample_id.tolist(), SOURCE_CHECKPOINTS[42], return_embedding=False).table
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_input_schema_hash_is_deterministic() -> None:
    assert input_schema_hash() == input_schema_hash(legacy_input_schema())
    assert len(input_schema_hash()) == 64


def test_historical_checkpoint_still_loads() -> None:
    frame = pd.read_csv(SOURCE_DATA).head(1)
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(frame, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    model = engine._load_model(SOURCE_CHECKPOINTS[42])
    assert sum(parameter.numel() for parameter in model.parameters()) == 775476

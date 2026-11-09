"""Real latent extraction on synthetic graph tensors, no actual labels or fits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.qgeognn_al.active_learning_v2.coverage import extract_representations as frozen_extract
from src.qgeognn_al.infra.contracts import BaselineProvenance
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.representations import Authorization
from src.qgeognn_al.representations.latent import GraphRow, extract_latent, predictor_state_hash
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.workflows import SelectionWorkflow


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    predictor = build_predictor(ConditionNormalization(0., 1., 0., 1., "4g", "source_train", 2, "0"*64, "0"*64))
    predictor.eval()
    return predictor


@pytest.fixture
def rows():
    path = Path(__file__).with_name("synthetic_graphs.py")
    spec = importlib.util.spec_from_file_location("synthetic_graphs_for_latent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = []
    for index in range(6):
        atom, angle = module.graph_pair(index)
        atom = atom.clone()
        del atom.y  # strict API never receives even a synthetic label field.
        atom.x[0, 0] = index + 1  # distinct synthetic atoms avoid zero-distance LCMD degeneracy.
        result.append(GraphRow(f"synthetic-{index}", atom, angle))
    return result


def authorized(rows):
    return Authorization(tuple(row.sample_id for row in rows[:2]), tuple(row.sample_id for row in rows[2:]))


def provenance():
    return BaselineProvenance("1"*40, "2"*64, "3"*64, "4"*64, "5"*64)


def extract(model, rows, **changes):
    kwargs = dict(expected_state_hash=predictor_state_hash(model), authorization=changes.get("authorization") or authorized(rows),
                  rows=rows, provenance=provenance(), batch_size=2)
    kwargs.update(changes)
    return extract_latent(model, **kwargs)


def test_extraction_matches_frozen_latent_exactly_and_freezes_lcmd_result(model, rows, tmp_path):
    before = predictor_state_hash(model)
    old = frozen_extract(model, [row.atom for row in rows], [row.angle for row in rows], list(range(6)), batch_size=2)
    snapshot = extract(model, rows)
    np.testing.assert_array_equal(np.vstack([snapshot.train, snapshot.candidates]), old)
    assert snapshot.train.shape == (2, 128)
    assert snapshot.candidates.shape == (4, 128)
    assert predictor_state_hash(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training
    audit = snapshot.provenance()
    assert audit["feature_source"] == "predictor_extraction"
    assert audit["extraction"]["predictor_state_hash"] == before
    assert audit["extraction"]["predictor_state_after_hash"] == before
    assert len(audit["extraction"]["graph_snapshot_hash"]) == 64
    assert audit["extraction"]["graph_snapshot_hash"] == audit["extraction"]["graph_snapshot_after_hash"]
    assert audit["predictor"]["model_checkpoint_hash"] == "4"*64
    workflow = SelectionWorkflow(snapshot)
    result = workflow.plan(acquisition="lcmd", budget=2, seed=17)
    envelope = workflow.freeze(tmp_path / "synthetic_latent_lcmd.json")
    assert set(result.selected_ids) <= set(authorized(rows).candidate_ids)
    assert len(result.selected_ids) == 2
    assert envelope["payload"]["labels_revealed"] is False
    assert envelope["payload"]["representation_provenance"]["extraction"]["predictor_state_hash"] == before


@pytest.mark.parametrize("field", ["y", "true_label", "future_target", "V1_ml", "test_truth"])
def test_graph_label_or_unapproved_field_rejected_before_model_call(model, rows, field):
    rows[0].atom[field] = torch.tensor([123.])
    with pytest.raises(ValueError, match="label, unapproved field"):
        extract(model, rows)


def test_duplicate_extra_or_reordered_rows_rejected(model, rows):
    auth = authorized(rows)
    for altered in [rows + [rows[0]], rows[::-1], rows[:-1]]:
        with pytest.raises(ValueError, match="duplicate|authorized L then U"):
            extract(model, altered, authorization=auth)


def test_stale_predictor_identity_and_training_mode_rejected(model, rows):
    with pytest.raises(ValueError, match="fingerprint"):
        extract(model, rows, expected_state_hash="0"*64)
    training = build_predictor(ConditionNormalization(0., 1., 0., 1., "4g", "source_train", 2, "0"*64, "0"*64))
    training.train()
    with pytest.raises(ValueError, match="eval mode"):
        extract(training, rows)
    assert training.training  # API does not silently change model mode.


def test_mutation_or_wrong_latent_width_is_detected(rows):
    class SyntheticPredictor(torch.nn.Module):
        def __init__(self, width=128, mutate=False):
            super().__init__()
            self.register_buffer("counter", torch.zeros(1))
            self.width, self.mutate = width, mutate

        def extract_representation(self, atom, angle):
            if self.mutate:
                self.counter.add_(1)
            return torch.zeros((atom.num_graphs, self.width))

    for predictor, reason in [(SyntheticPredictor(width=127), "128-dimensional"),
                              (SyntheticPredictor(mutate=True), "predictor changed")]:
        predictor.eval()
        with pytest.raises(ValueError, match=reason):
            extract(predictor, rows)


def test_graph_position_duplicates_and_nonfinite_features_rejected(model, rows):
    rows[1].atom.canonical_position = rows[0].atom.canonical_position.clone()
    with pytest.raises(ValueError, match="positions must be unique"):
        extract(model, rows)
    rows[1].atom.canonical_position = torch.tensor(1)
    rows[2].atom.edge_attr[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite CPU tensors"):
        extract(model, rows)

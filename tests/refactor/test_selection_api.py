"""Offline public-boundary and exact frozen-LCMD regression checks.

All vectors and callback values here are synthetic. No project data, labels,
checkpoint, provider, graph cache, training routine or live controller is used.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.qgeognn_al.acquisitions import select
from src.qgeognn_al.acquisitions.lcmd import lcmd_tp_select as migrated_lcmd
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select as frozen_lcmd
from src.qgeognn_al.agents import filtered_catalogue, validate_plan
from src.qgeognn_al.infra.contracts import BaselineProvenance, content_hash
from src.qgeognn_al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.workflows import SelectionWorkflow
from scripts.run_offline_selection import run


def synthetic_snapshot(**changes):
    values = dict(authorization=Authorization(("L0", "L1"), ("U0", "U1", "U2", "U3")),
                  train_ids=("L0", "L1"), candidate_ids=("U0", "U1", "U2", "U3"),
                  train_features=[[0., 0.], [10., 0.]],
                  candidate_features=[[1., 0.], [3., 0.], [9., 0.], [5., 1.]],
                  config=RepresentationConfig("latent", 2),
                  predictor=BaselineProvenance("1"*40, "2"*64, "3"*64, "4"*64, "5"*64))
    values.update(changes)
    return FeatureSnapshot(**values)


@pytest.mark.parametrize("seed,dimension,budget", [(12, 1, 13), (17, 128, 32), (157, 512, 5)])
def test_migrated_lcmd_is_elementwise_identical_to_frozen_kernel(seed, dimension, budget):
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(7, dimension))
    pool = rng.normal(size=(40, dimension))
    before = frozen_lcmd(pool, train, budget)
    after = migrated_lcmd(pool, train, budget)
    np.testing.assert_array_equal(after.selected_pool_positions, before.selected_pool_positions)
    np.testing.assert_array_equal(after.initial_nearest_center, before.initial_nearest_center)
    np.testing.assert_array_equal(after.initial_nearest_sq_distance, before.initial_nearest_sq_distance)
    assert after.trace == before.trace


def test_stable_tie_order_and_authorized_ids():
    train = np.array([[0.], [10.]])
    pool = np.array([[1.], [-1.], [9.], [11.]])
    before, after = frozen_lcmd(pool, train, 2), migrated_lcmd(pool, train, 2)
    assert after.selected_pool_positions.tolist() == before.selected_pool_positions.tolist() == [0, 2]
    assert after.trace == before.trace
    snapshot = synthetic_snapshot()
    result = select(snapshot, acquisition="lcmd", budget=2, seed=17)
    assert len(set(result.selected_ids)) == 2
    assert set(result.selected_ids) <= set(snapshot.candidate_ids)
    trace = result.trace
    trace[0]["sample_id"] = "tampered"
    assert result.trace[0]["sample_id"] != "tampered"


def test_degenerate_empty_cluster_preserves_historical_failure():
    # Fixing this changes historical semantics and requires independent review.
    for implementation in (frozen_lcmd, migrated_lcmd):
        with pytest.raises(RuntimeError, match="empty largest cluster"):
            implementation(np.array([[10.], [10.]]), np.array([[0.], [10.]]), 1)


@pytest.mark.parametrize("budget", [0, -1, 5, 1.5, True, "2"])
def test_budget_requires_legal_integer(budget):
    with pytest.raises(ValueError, match="budget"):
        select(synthetic_snapshot(), acquisition="lcmd", budget=budget, seed=17)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_seed_requires_nonnegative_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        select(synthetic_snapshot(), acquisition="random", budget=2, seed=seed)


def test_local_random_generator_is_deterministic_and_does_not_change_global_rng():
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    snapshot = synthetic_snapshot()
    first = select(snapshot, acquisition="random", budget=3, seed=42)
    second = select(snapshot, acquisition="random", budget=3, seed=42)
    assert first.to_dict() == second.to_dict()
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize("ids", [("U0", "U0", "U2", "U3"), ("U1", "U0", "U2", "U3"),
                                  ("U0", "U1", "U2", "test-row"), ("U0", "U1", "U2", ""),
                                  ("U0", "U1", "U2", 3)])
def test_duplicate_unauthorized_missing_or_reordered_feature_ids_rejected(ids):
    with pytest.raises(ValueError):
        synthetic_snapshot(candidate_ids=ids)


@pytest.mark.parametrize("values", [[[1., 2.]], [[1., 2.]]*5,
                                     [[np.nan, 0.]]*4, [[np.inf, 0.]]*4,
                                     [[1j, 0.]]*4, [["1", "2"]]*4])
def test_nonfinite_wrong_shape_or_nonnumeric_features_rejected(values):
    with pytest.raises(ValueError):
        synthetic_snapshot(candidate_features=values)


def test_authorization_disjointness_and_unique_ids():
    with pytest.raises(ValueError, match="disjoint"):
        Authorization(("L0",), ("L0",))
    with pytest.raises(ValueError, match="duplicate"):
        Authorization(("L0", "L0"), ("U0",))


def test_snapshot_owns_vectors_and_checks_frozen_hash():
    source = np.ones((4, 2))
    snapshot = synthetic_snapshot(candidate_features=source)
    source[:] = 9
    assert np.all(snapshot.candidates == 1)
    with pytest.raises(ValueError):
        snapshot.candidates[0, 0] = 4
    snapshot.candidates.setflags(write=True)
    snapshot.candidates[0, 0] = 5
    with pytest.raises(ValueError, match="hash changed"):
        select(snapshot, acquisition="lcmd", budget=2, seed=17)


def test_label_callback_cannot_run_before_freeze_and_freeze_is_exclusive(tmp_path):
    workflow = SelectionWorkflow(synthetic_snapshot())
    calls = []
    callback = lambda ids: calls.append(ids) or {identity: 0 for identity in ids}
    with pytest.raises(ValueError):
        workflow.reveal(callback)
    workflow.plan(acquisition="lcmd", budget=2, seed=17)
    with pytest.raises(ValueError):
        workflow.reveal(callback)
    assert calls == []
    path = tmp_path / "freeze.json"
    workflow.freeze(path)
    sibling = SelectionWorkflow(synthetic_snapshot())
    sibling.plan(acquisition="lcmd", budget=2, seed=17)
    with pytest.raises(FileExistsError):
        sibling.freeze(path)
    restored = SelectionWorkflow.resume_frozen(synthetic_snapshot(), path)
    assert restored.state == "frozen"
    revealed = restored.reveal(callback)
    assert tuple(revealed) == restored.selection.selected_ids
    assert len(calls) == 1
    # Resume sees durable reveal-start marker and must never repeat callbacks.
    restored_again = SelectionWorkflow.resume_frozen(synthetic_snapshot(), path)
    with pytest.raises(ValueError):
        restored_again.reveal(callback)
    assert len(calls) == 1


def test_callback_wrong_ids_or_ambiguous_failure_cannot_be_retried(tmp_path):
    workflow = SelectionWorkflow(synthetic_snapshot())
    workflow.plan(acquisition="lcmd", budget=1, seed=17)
    workflow.freeze(tmp_path / "freeze.json")
    with pytest.raises(ValueError, match="exactly"):
        workflow.reveal(lambda ids: {"unauthorized": 9})
    with pytest.raises(ValueError):
        workflow.reveal(lambda ids: {ids[0]: 9})


def test_tampered_freeze_rejected_before_reveal_and_on_resume(tmp_path):
    workflow = SelectionWorkflow(synthetic_snapshot())
    workflow.plan(acquisition="lcmd", budget=1, seed=17)
    path = tmp_path / "freeze.json"
    workflow.freeze(path)
    envelope = json.loads(path.read_text())
    envelope["payload"]["selected_ids"] = ["test-row"]
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="changed"):
        workflow.reveal(lambda ids: pytest.fail("must not call reveal"))
    with pytest.raises(ValueError, match="hash mismatch"):
        SelectionWorkflow.resume_frozen(synthetic_snapshot(), path)


def test_resume_requires_identical_predictor_and_representation(tmp_path):
    workflow = SelectionWorkflow(synthetic_snapshot())
    workflow.plan(acquisition="lcmd", budget=1, seed=17)
    path = tmp_path / "freeze.json"
    workflow.freeze(path)
    changed = synthetic_snapshot(predictor=BaselineProvenance("1"*40, "2"*64, "3"*64, "6"*64, "5"*64))
    with pytest.raises(ValueError, match="differs"):
        SelectionWorkflow.resume_frozen(changed, path)


def test_agent_catalogue_filters_unknown_fields_and_rejects_hidden_candidates():
    rows = [{"sample_id": "U0", "rank": 1, "V1_truth": 999, "prompt": "secret"},
            {"sample_id": "U1", "nearest_center_sq_distance": 2., "label": 9}]
    safe = filtered_catalogue(rows, candidate_ids=("U1", "U0"))
    assert safe == [{"sample_id": "U1", "nearest_center_sq_distance": 2.}, {"sample_id": "U0", "rank": 1}]
    with pytest.raises(ValueError):
        filtered_catalogue(rows + [{"sample_id": "test-row"}], candidate_ids=("U1", "U0"))
    with pytest.raises(ValueError):
        filtered_catalogue([{"sample_id": "U0", "rank": {"truth": 3}}], candidate_ids=("U0",))


@pytest.mark.parametrize("plan", [{"selected_ids": ["U0", "U0"]}, {"selected_ids": ["U0", "test-row"]},
                                    {"selected_ids": ["U0"]}, {"selected_ids": ["U0", "U1"], "reveal": True}])
def test_agent_plan_cannot_change_candidate_set_budget_or_workflow(plan):
    with pytest.raises(ValueError):
        validate_plan(plan, candidate_ids=("U0", "U1", "U2"), budget=2)


def test_config_to_selection_trace_and_full_result_provenance(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/offline_selection.synthetic.json").read_text())
    output = tmp_path / "selection.json"
    completed = subprocess.run([sys.executable, "-m", "scripts.run_offline_selection", "--config",
                                str(root / "configs/offline_selection.synthetic.json"), "--output", str(output)],
                               cwd=root, check=True, capture_output=True, text=True)
    envelope = json.loads(output.read_text())
    assert json.loads(completed.stdout)["sha256"] == envelope["sha256"] == content_hash(envelope["payload"])
    payload = envelope["payload"]
    assert {"code_commit", "dataset_hash", "split_hash", "model_checkpoint_hash", "protocol_hash",
            "representation", "acquisition", "seed", "budget", "selected_ids", "trace"} <= payload.keys()
    assert len(set(payload["selected_ids"])) == config["budget"]
    assert payload["source_kind"] == "synthetic"
    assert payload["test_accessed"] is payload["labels_revealed"] is payload["training_performed"] is False
    assert payload["representation_provenance"]["representation"]["dimension"] == 128
    forbidden = copy.deepcopy(config)
    forbidden["provenance"]["source_kind"] = "authorized_snapshot"
    with pytest.raises(ValueError, match="synthetic inputs only"):
        run(forbidden, tmp_path / "forbidden.json")

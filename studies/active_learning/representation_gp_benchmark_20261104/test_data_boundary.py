"""Exercise label permissions and cache tamper checks with synthetic data only."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore
from representations import StateArtifacts, file_hash, load_model, load_raw_cache


@pytest.fixture
def synthetic_store(tmp_path):
    path = tmp_path / "synthetic.csv"
    # Nonnumeric strings make any accidental conversion of forbidden rows fail.
    rows = [("initial", "l0", "1", "2"), ("acquired", "u0", "3", "4"),
            ("future", "u0", "FORBIDDEN", "FORBIDDEN"),
            ("validation", "validation", "5", "6"),
            ("heldout", "test", "FORBIDDEN", "FORBIDDEN")]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "V1_ml", "V2_ml"])
        writer.writerows((sid, a, b) for sid, _, a, b in rows)
    partition = pd.DataFrame({"sample_id": [r[0] for r in rows], "role": [r[1] for r in rows]})
    return path, partition, RestrictedLabelStore(path, partition)


def test_only_current_l_and_development_validation_are_converted(synthetic_store):
    _, _, store = synthetic_store
    store.freeze_acquisitions(["acquired"])
    train = store.reveal(["initial", "acquired"], "after_acquisition_fit")
    val = store.reveal(["validation"], "initial_fit")
    np.testing.assert_array_equal(train, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(val, [[5, 6]])
    assert sum(call["requested_rows"] for call in store.audit) == 3


@pytest.mark.parametrize("ids,purpose", [
    (["heldout"], "initial_fit"), (["heldout"], "after_acquisition_fit"),
    (["future"], "initial_fit"), (["future"], "after_acquisition_fit"),
    (["heldout"], "final_test_evaluation"),
])
def test_test_and_future_u_requests_fail_before_source_open(synthetic_store, ids, purpose):
    _, _, store = synthetic_store
    store.freeze_acquisitions(["acquired"])
    with patch.object(Path, "open", side_effect=AssertionError("forbidden request reached source")):
        with pytest.raises(PermissionError):
            store.reveal(ids, purpose)


def test_duplicate_and_unknown_label_ids_fail_before_source_open(synthetic_store):
    _, _, store = synthetic_store
    with patch.object(Path, "open", side_effect=AssertionError("invalid request reached source")):
        with pytest.raises(ValueError):
            store.reveal(["initial", "initial"], "initial_fit")
        with pytest.raises(ValueError):
            store.reveal(["unknown"], "initial_fit")


def synthetic_state(tmp_path):
    return StateArtifacts(tmp_path, 157, 1, 0, tmp_path,
        tmp_path / "checkpoint.pt", tmp_path / "raw.npz", ["initial"], ["future"], ["validation"],
        np.array([0]), np.array([1]), np.array([2]), np.array([0]), pd.DataFrame(), {})


def test_checkpoint_byte_change_is_rejected_before_deserialization(tmp_path):
    state = synthetic_state(tmp_path)
    state.checkpoint.write_bytes(b"original-synthetic-checkpoint")
    state.audit["checkpoint_sha256"] = file_hash(state.checkpoint)
    state.checkpoint.write_bytes(b"changed-synthetic-checkpoint")
    with patch("qgeognn_al.models.load_predictor_checkpoint", side_effect=AssertionError("tampered checkpoint was loaded")):
        with pytest.raises(ValueError, match="checkpoint changed"):
            load_model(state)


def test_raw_byte_change_is_rejected_before_loading_arrays(tmp_path):
    state = synthetic_state(tmp_path)
    state.raw_cache.write_bytes(b"original-synthetic-cache")
    state.audit["raw_cache_sha256"] = file_hash(state.raw_cache)
    state.raw_cache.write_bytes(b"changed-synthetic-cache")
    with patch("representations.np.load", side_effect=AssertionError("tampered cache was loaded")):
        with pytest.raises(ValueError, match="raw cache changed"):
            load_raw_cache(state)


def test_runner_rejects_training_validation_overlap_before_reveal(synthetic_store):
    from run_benchmark import authorized_labels
    source, partition, _ = synthetic_store
    state = SimpleNamespace(partition=partition, labeled_ids=["initial"], validation_ids=["initial"],
                            labeled_indices=np.array([0]), l0_indices=np.array([0]))
    with patch.object(RestrictedLabelStore, "reveal", side_effect=AssertionError("overlap was revealed")):
        with pytest.raises(ValueError, match="overlap"):
            authorized_labels(state, source)


def test_prepared_cache_state_change_is_rejected(tmp_path):
    import json
    import run_benchmark as runner
    cache = tmp_path / "cache" / "seed_157_L333.npz"
    cache.parent.mkdir()
    np.savez_compressed(cache, indices=np.array([0]))
    cache.with_suffix(".json").write_text(json.dumps({"cache_sha256": file_hash(cache), "state": {"seed": 887}}))
    state = SimpleNamespace(audit={"seed": 157}, unlabeled_ids=["future"],
                            unlabeled_indices=np.array([1]), labeled_indices=np.array([0]),
                            validation_indices=np.array([2]))
    with patch.object(runner, "HERE", tmp_path), patch.object(runner, "audit_lcmd_state", return_value=state):
        with pytest.raises(ValueError, match="cache drift"):
            runner.prepare_state(tmp_path, 157, 333, pd.DataFrame(), tmp_path / "unused.csv")


def test_prepared_cache_array_mapping_change_is_rejected(tmp_path):
    import json
    import run_benchmark as runner
    source = tmp_path / "synthetic.csv"
    source.write_text("synthetic source identity only\n")
    cache = tmp_path / "cache" / "seed_157_L333.npz"
    cache.parent.mkdir()
    # The receipt is internally valid, but the row mapping is from another state.
    np.savez_compressed(cache, indices=np.array([99]), train_y=np.zeros((333, 2)),
                        validation_y=np.zeros((416, 2)))
    audit = {"seed": 157}
    cache.with_suffix(".json").write_text(json.dumps({"cache_sha256": file_hash(cache), "state": audit,
                                                     "source_sha256": file_hash(source)}))
    state = SimpleNamespace(audit=audit, unlabeled_ids=["future"], unlabeled_indices=np.array([1]),
                            labeled_indices=np.array([0]), validation_indices=np.array([2]))
    with patch.object(runner, "HERE", tmp_path), patch.object(runner, "audit_lcmd_state", return_value=state):
        with pytest.raises(ValueError, match="mapping drift"):
            runner.prepare_state(tmp_path, 157, 333, pd.DataFrame(), source)


def test_runner_rejects_changed_source_before_any_label_reveal(tmp_path):
    import json
    from run_benchmark import authorized_labels
    directory = tmp_path / "studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_157"
    directory.mkdir(parents=True)
    (directory / "context.json").write_text(json.dumps({"contract": {"source_sha256": "expected-frozen-hash"}}))
    source = tmp_path / "synthetic.csv"
    source.write_text("changed source identity only\n")
    state = SimpleNamespace(root=tmp_path, seed=157, budget=1, labeled_ids=["initial"],
                            validation_ids=[f"development_{i}" for i in range(416)])
    with patch.object(RestrictedLabelStore, "reveal", side_effect=AssertionError("changed source was revealed")):
        with pytest.raises(ValueError, match="source differs"):
            authorized_labels(state, source)

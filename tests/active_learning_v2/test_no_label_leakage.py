from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, make_row_protocol


def test_hidden_u0_and_test_mutation_cannot_change_acquisition() -> None:
    identities = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(100)], "canonical_smiles": [f"m{i % 9}" for i in range(100)]}
    )
    partition = make_row_protocol(identities, 73)
    rng = np.random.default_rng(8)
    features = rng.normal(size=(100, 12))
    l0 = partition.loc[partition.role.eq("l0"), "canonical_index"].to_numpy(int)
    u0 = partition.loc[partition.role.eq("u0"), "canonical_index"].to_numpy(int)
    selected_before = lcmd_tp_select(features[u0], features[l0], 5).selected_pool_positions
    labels = rng.normal(size=(100, 2))
    labels[partition.role.isin(["u0", "test"])] = 1e12
    selected_after = lcmd_tp_select(features[u0], features[l0], 5).selected_pool_positions
    assert np.array_equal(selected_before, selected_after)


def test_test_truth_access_is_blocked_until_predictions_freeze(tmp_path: Path) -> None:
    identities = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(30)], "canonical_smiles": [f"m{i % 5}" for i in range(30)]}
    )
    partition = make_row_protocol(identities, 73)
    source = tmp_path / "labels.csv"
    identities.assign(V1_ml=np.arange(30), V2_ml=np.arange(30) + 1).to_csv(source, index=False)
    store = RestrictedLabelStore(source, partition)
    test_ids = partition.loc[partition.role.eq("test"), "sample_id"].tolist()
    with pytest.raises(PermissionError):
        store.reveal(test_ids, "final_test_evaluation")
    acquired = partition.loc[partition.role.eq("u0"), "sample_id"].head(2).tolist()
    store.freeze_acquisitions(acquired)
    with pytest.raises(PermissionError):
        store.reveal(test_ids, "final_test_evaluation")
    store.freeze_predictions()
    truth = store.reveal(test_ids, "final_test_evaluation")
    assert truth.shape == (len(test_ids), 2)
    assert store.audit[-1]["predictions_frozen"] is True


def test_acquisition_freeze_accepts_only_u0_ids(tmp_path: Path) -> None:
    identities = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(40)], "canonical_smiles": [f"m{i % 7}" for i in range(40)]}
    )
    partition = make_row_protocol(identities, 311)
    source = tmp_path / "labels.csv"
    identities.assign(V1_ml=np.arange(40), V2_ml=np.arange(40) + 1).to_csv(source, index=False)
    store = RestrictedLabelStore(source, partition)
    for forbidden_role in ("l0", "validation", "test"):
        forbidden_id = partition.loc[partition.role.eq(forbidden_role), "sample_id"].iloc[0]
        with pytest.raises(ValueError):
            store.freeze_acquisitions([forbidden_id])
    u0_ids = partition.loc[partition.role.eq("u0"), "sample_id"].head(3).tolist()
    store.freeze_acquisitions(u0_ids)
    assert store.acquisition_ids == set(u0_ids)

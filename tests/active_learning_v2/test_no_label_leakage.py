from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, make_row_protocol


def test_hidden_u0_and_test_mutation_cannot_change_acquisition(tmp_path: Path) -> None:
    from src.qgeognn_al.active_learning_v2.acquisition import acquire_batches
    from src.qgeognn_al.active_learning_v2.benchmark import create_smoke_fixture
    from src.qgeognn_al.active_learning_v2.benchmark_seed import SeedContext
    from src.qgeognn_al.active_learning_v2.cache import array_hash
    from src.qgeognn_al.active_learning_v2.protocol import stable_hash
    from src.qgeognn_al.artifacts import sha256_file

    source, partition = create_smoke_fixture(tmp_path / "canonical_fixture", rows=160)
    original = pd.read_csv(source)
    mutated = original.copy()
    hidden = partition.role.isin(["u0", "test"]).to_numpy()
    mutated.loc[hidden, ["V1_ml", "V2_ml"]] = np.random.default_rng(90210).uniform(1e8, 1e12, (hidden.sum(), 2))
    mutated_source = tmp_path / "mutated.csv"
    mutated.to_csv(mutated_source, index=False)
    pd.testing.assert_frame_equal(original.drop(columns=["V1_ml", "V2_ml"]), mutated.drop(columns=["V1_ml", "V2_ml"]))
    pd.testing.assert_frame_equal(original.loc[~hidden], mutated.loc[~hidden])
    assert sha256_file(source) != sha256_file(mutated_source)

    records = []
    for name, target_store in (("original", source), ("mutated", mutated_source)):
        context = SeedContext(target_store, tmp_path / name, 29, partition, smoke=True)
        baseline, fits, gradients, latent, predictions = context.initial_features()
        assert all(not graph.y.count_nonzero() for graph in context.atom)
        arms = acquire_batches(gradients=gradients, representations=latent, ensemble_predictions=predictions,
            l0_count=len(context.roles["l0"]), scales=tuple(context.preprocessing["target_scales"].values()),
            outer_seed=29, batch_size=32)
        selected_ids = {arm: context.ids(context.roles["u0"][positions]) for arm, positions in arms.items()}
        records.append({"checkpoint_hash": baseline["checkpoint_state_hash"], "gradient_hash": array_hash(gradients),
                        "latent_hash": array_hash(latent), "ensemble_hash": array_hash(predictions),
                        "selected_ids": selected_ids, "selected_ids_hash": stable_hash(selected_ids),
                        "fit_contract_hash": baseline["fit_contract_hash"], "source_hash": sha256_file(target_store)})
        assert not any(row["purpose"] != "initial_fit" for row in context.store.audit)
        assert len({row["initialization_hash"] for row in fits}) == 3
    for key in ("checkpoint_hash", "gradient_hash", "latent_hash", "ensemble_hash", "selected_ids", "selected_ids_hash"):
        assert records[0][key] == records[1][key], key
    assert records[0]["fit_contract_hash"] != records[1]["fit_contract_hash"]
    assert records[0]["source_hash"] != records[1]["source_hash"]
    with pytest.raises(RuntimeError, match="cache contract/content mismatch"):
        SeedContext(mutated_source, tmp_path / "original", 29, partition, smoke=True)
    with pytest.raises(PermissionError):
        context.store.reveal(context.ids(context.roles["u0"][:1]), "initial_fit")
    changed_l0 = context.l0_truth.copy()
    changed_l0[0, 0] += 10
    with pytest.raises(RuntimeError, match="refusing incompatible completed fit"):
        context.fit("baseline_l0", context.roles["l0"], changed_l0)


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

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from scripts.studies.run_qgeognn_v2_4g_row_lcmd_pilot import (
    QUALIFICATION_PROTOCOL,
    load_feature_frame,
)
from src.qgeognn_al.active_learning_v2.protocol import (
    OUTER_SEEDS,
    batch_size,
    make_row_protocol,
    random_control_positions,
)


def _identities(rows: int = 4163) -> pd.DataFrame:
    return pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(rows)], "canonical_smiles": [f"m{i % 217}" for i in range(rows)]}
    )


def test_frozen_outer_seeds_and_integer_role_counts() -> None:
    assert OUTER_SEEDS == (73, 311, 1297, 4093, 8191)
    for seed in OUTER_SEEDS:
        split = make_row_protocol(_identities(), seed)
        assert split.role.value_counts().to_dict() == {
            "u0": 2997,
            "test": 417,
            "validation": 416,
            "l0": 333,
        }
        assert split.sample_id.nunique() == len(split)
        assert batch_size(split) == 333


def test_canonical_counts_are_read_and_checked_against_qualification() -> None:
    qualified = json.loads(QUALIFICATION_PROTOCOL.read_text())
    data = load_feature_frame()
    assert len(data) == qualified["rows"]
    assert data.canonical_smiles.nunique() == qualified["compounds"]


def test_row_partition_uses_no_target_columns_and_is_deterministic() -> None:
    identities = _identities(100)
    with_targets = identities.assign(V1_ml=np.arange(100), V2_ml=-np.arange(100))
    mutated = with_targets.assign(V1_ml=1e12, V2_ml=-1e12)
    first = make_row_protocol(with_targets, 73)
    second = make_row_protocol(mutated, 73)
    pd.testing.assert_frame_equal(first, second)


def test_random_controls_are_independent_deterministic_exact_batches() -> None:
    draws = [random_control_positions(2997, 333, 73, index) for index in range(5)]
    repeats = [random_control_positions(2997, 333, 73, index) for index in range(5)]
    assert all(np.array_equal(left, right) for left, right in zip(draws, repeats))
    assert all(len(np.unique(draw)) == 333 for draw in draws)
    assert len({tuple(draw) for draw in draws}) == 5

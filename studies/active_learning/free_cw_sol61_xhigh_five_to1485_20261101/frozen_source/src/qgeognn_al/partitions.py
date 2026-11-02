"""Shared partition-role validation."""

from __future__ import annotations

from collections.abc import Iterable
import math

import pandas as pd
import numpy as np


def validate_partition_roles(
    partition: pd.DataFrame,
    allowed_roles: Iterable[str],
    identity_column: str = "sample_id",
) -> None:
    required = {identity_column, "role"}
    if missing := required - set(partition.columns):
        raise ValueError(f"partition missing columns: {sorted(missing)}")
    if partition[identity_column].isna().any() or partition[identity_column].duplicated().any():
        raise ValueError(f"{identity_column} must be non-null and unique")
    unknown = sorted(set(partition["role"].astype(str)) - set(allowed_roles))
    if unknown:
        raise ValueError(f"unknown partition roles: {unknown}")


def make_row_partition(
    identities: pd.DataFrame,
    seed: int,
    test_fraction: float,
    l0_fraction: float,
    validation_fraction: float,
    stage: str,
    protocol: str,
) -> pd.DataFrame:
    """Frozen D28 row-partition rule using identity/order only, never truth."""
    required = ["sample_id", "canonical_index", "canonical_smiles"]
    data = identities.loc[:, required].copy().reset_index(drop=True)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(data))
    test_count = max(1, int(math.ceil(test_fraction * len(data))))
    test_mask = np.zeros(len(data), dtype=bool); test_mask[order[-test_count:]] = True
    available = np.flatnonzero(~test_mask)
    l0_size = int(math.ceil(l0_fraction * len(available)))
    selection_rng = np.random.RandomState(seed + 100_003)
    l0 = selection_rng.choice(available, size=l0_size, replace=False)
    validation_size = int(math.ceil(validation_fraction * len(l0)))
    validation = set(selection_rng.choice(l0, size=validation_size, replace=False).tolist())
    l0_set = set(int(value) for value in l0)
    roles = ["test" if test_mask[i] else "l0_validation" if i in validation else "l0_train" if i in l0_set else "u0" for i in range(len(data))]
    data["role"], data["stage"], data["protocol"], data["seed"] = roles, stage, protocol, int(seed)
    validate_partition_roles(data, {"test", "l0_validation", "l0_train", "u0"})
    return data

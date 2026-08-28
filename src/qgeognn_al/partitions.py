"""Shared partition-role validation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


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

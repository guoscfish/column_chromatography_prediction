"""Frozen row-split and label-access protocol for the V2 LCMD study."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


OUTER_SEEDS = (73, 311, 1297, 4093, 8191)
RANDOM_CONTROLS = 5
OUTER_TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
L0_OUTER_TRAIN_FRACTION = 0.10
BATCH_OUTER_TRAIN_FRACTION = 0.10
ROLE_ORDER = ("l0", "u0", "validation", "test")


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ids_hash(values: Iterable[str]) -> str:
    return stable_hash(sorted(str(value) for value in values))


def make_row_protocol(identities: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create the frozen 80/10/10 row partition and nested L0/U0 split.

    Only identity columns and canonical row order are consumed.  For 4,163
    rows the integer rule is 3,330 outer-train, 416 validation, 417 test;
    L0 and the one-step query batch are each 333 rows.
    """

    required = ("sample_id", "canonical_smiles")
    if missing := set(required) - set(identities):
        raise ValueError(f"identities missing columns: {sorted(missing)}")
    data = identities.loc[:, required].copy().reset_index(drop=True)
    if data.sample_id.isna().any() or data.sample_id.astype(str).duplicated().any():
        raise ValueError("sample_id must be complete and unique")
    data["sample_id"] = data.sample_id.astype(str)
    data["canonical_index"] = np.arange(len(data), dtype=int)

    test_count = int(math.ceil(TEST_FRACTION * len(data)))
    validation_count = int(math.floor(VALIDATION_FRACTION * len(data)))
    outer_train_count = len(data) - validation_count - test_count
    order = np.random.default_rng(int(seed)).permutation(len(data))
    outer_train = order[:outer_train_count]
    validation = order[outer_train_count : outer_train_count + validation_count]
    test = order[outer_train_count + validation_count :]
    l0_count = int(round(L0_OUTER_TRAIN_FRACTION * outer_train_count))
    l0_rng = np.random.default_rng(int(seed) + 104_729)
    l0 = l0_rng.choice(outer_train, size=l0_count, replace=False)

    role = np.full(len(data), "u0", dtype=object)
    role[l0] = "l0"
    role[validation] = "validation"
    role[test] = "test"
    data["role"] = role
    data["outer_seed"] = int(seed)
    data["study"] = "qgeognn_v2_row_lcmd"
    validate_row_protocol(data)
    return data


def validate_row_protocol(partition: pd.DataFrame) -> None:
    required = {"sample_id", "canonical_index", "canonical_smiles", "role", "outer_seed"}
    if missing := required - set(partition):
        raise ValueError(f"partition missing columns: {sorted(missing)}")
    if partition.sample_id.isna().any() or partition.sample_id.astype(str).duplicated().any():
        raise ValueError("every sample_id must occur in exactly one role")
    if set(partition.role) != set(ROLE_ORDER):
        raise ValueError("partition must contain exactly L0, U0, validation, and test roles")
    n = len(partition)
    expected = {
        "test": int(math.ceil(TEST_FRACTION * n)),
        "validation": int(math.floor(VALIDATION_FRACTION * n)),
    }
    expected["l0"] = int(round(L0_OUTER_TRAIN_FRACTION * (n - expected["test"] - expected["validation"])))
    expected["u0"] = n - sum(expected.values())
    if partition.role.value_counts().to_dict() != expected:
        raise ValueError(f"unexpected protocol role counts: {partition.role.value_counts().to_dict()}")


def batch_size(partition: pd.DataFrame) -> int:
    outer_train_count = int(partition.role.isin(["l0", "u0"]).sum())
    return int(round(BATCH_OUTER_TRAIN_FRACTION * outer_train_count))


def random_control_positions(pool_size: int, batch: int, outer_seed: int, control: int) -> np.ndarray:
    if not 0 <= int(control) < RANDOM_CONTROLS:
        raise ValueError("unknown random-control index")
    if not 0 < int(batch) <= int(pool_size):
        raise ValueError("invalid random-control batch size")
    seed = int(outer_seed) * 1_000_003 + 50_000 + int(control)
    return np.random.default_rng(seed).choice(int(pool_size), size=int(batch), replace=False)


class RestrictedLabelStore:
    """Expose only labels authorized by the current protocol phase.

    The canonical CSV is streamed and target strings are converted to floats
    only for requested, authorized IDs.  Test targets cannot be requested
    before every acquisition, checkpoint, and prediction has been frozen.
    """

    def __init__(self, source: Path, partition: pd.DataFrame):
        self.source = Path(source)
        self.roles = dict(zip(partition.sample_id.astype(str), partition.role.astype(str)))
        self.acquisition_ids: set[str] = set()
        self.acquisitions_frozen = False
        self.predictions_frozen = False
        self.audit: list[dict[str, object]] = []

    def freeze_acquisitions(self, selected_ids: Iterable[str]) -> None:
        values = [str(value) for value in selected_ids]
        if any(self.roles.get(value) != "u0" for value in values):
            raise ValueError("acquisition freeze contains a non-U0 ID")
        self.acquisition_ids = set(values)
        self.acquisitions_frozen = True

    def freeze_predictions(self) -> None:
        if not self.acquisitions_frozen:
            raise RuntimeError("predictions cannot freeze before acquisitions")
        self.predictions_frozen = True

    def reveal(self, ids: Iterable[str], purpose: str) -> np.ndarray:
        requested = [str(value) for value in ids]
        if len(set(requested)) != len(requested):
            raise ValueError("label request contains duplicate IDs")
        roles = {self.roles.get(value) for value in requested}
        if None in roles:
            raise ValueError("label request contains an unknown ID")
        if purpose == "initial_fit":
            allowed = roles <= {"l0", "validation"}
        elif purpose == "after_acquisition_fit":
            allowed = self.acquisitions_frozen and all(
                self.roles[value] in {"l0", "validation"} or value in self.acquisition_ids
                for value in requested
            )
        elif purpose == "final_test_evaluation":
            allowed = self.predictions_frozen and roles == {"test"}
        else:
            raise ValueError(f"unknown label-access purpose: {purpose}")
        if not allowed:
            raise PermissionError(f"label access is not authorized for purpose={purpose}, roles={sorted(roles)}")

        wanted = set(requested)
        found: dict[str, tuple[float, float]] = {}
        with self.source.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row["sample_id"])
                if sample_id in wanted:
                    found[sample_id] = (float(row["V1_ml"]), float(row["V2_ml"]))
        if set(found) != wanted:
            raise ValueError("canonical target store did not contain all requested IDs")
        values = np.asarray([found[value] for value in requested], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("revealed labels contain non-finite values")
        self.audit.append(
            {
                "purpose": purpose,
                "requested_rows": len(requested),
                "requested_ids_hash": ids_hash(requested),
                "roles": "+".join(sorted(roles)),
                "acquisitions_frozen": self.acquisitions_frozen,
                "predictions_frozen": self.predictions_frozen,
            }
        )
        return values


def audit_payload(normalization: object, preprocessing: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-compatible preprocessing record."""

    return {"condition_normalization": asdict(normalization), "preprocessing": dict(preprocessing)}

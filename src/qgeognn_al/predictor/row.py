"""Canonical Row AL inputs, frozen partitioning and label-free graph preparation.

The partition algorithm is migrated from main b1a6ed7a active_learning_v2/protocol.py.
Only seven public X columns are parsed from a canonical CSV; labels enter prepare
as an explicit L0 array. Graph construction and scaling reuse the qualified
predictor implementation without changing its mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..infra.contracts import content_hash
from ..infra.resources import SnapshotFile
from ..al.representations.latent import GraphRow
from .data import build_model_data
from .preprocessing import fit_al_preprocessing
from .conditions import ConditionNormalization

FEATURE_COLUMNS = ("sample_id", "canonical_smiles", "Density g/ml", "V/ul",
                   "loading solvent", "Volume of loading solvent/ul", "PE/EA")
PARTITION_COLUMNS = ("sample_id", "canonical_index", "canonical_smiles", "role", "outer_seed")
CONFIRMATION_SEEDS = (157, 887, 2357, 6101, 12203)
OUTER_TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
L0_OUTER_TRAIN_FRACTION = 0.10
ROLE_ORDER = ("l0", "u0", "validation", "test")
SOURCE_COMMIT = "b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28"


def safe_output(path, forbidden_output_roots=()):
    """Reject protected roots and symlink traversal before any file creation."""
    result = Path(path).absolute()
    if any(p.is_symlink() for p in (result, *result.parents)):
        raise ValueError("row output may not traverse a symlink")
    resolved = result.resolve()
    if any(resolved == Path(p).resolve() or Path(p).resolve() in resolved.parents
           for p in forbidden_output_roots):
        raise ValueError("row output is inside a protected worktree")
    return result


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be a SHA256 identity")
    return value


@dataclass(frozen=True)
class PreparedRowData:
    normalization: ConditionNormalization
    preprocessing: dict
    rows: Mapping[str, GraphRow]
    dataset_hash: str
    split_hash: str
    graph_cache_hash: str
    outer_seed: int
    l0_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


class RowDataset:
    """Validated row identities/X/cache; never stores a source target table."""

    def __init__(self, feature_frame, partition, graph_cache, *, source_sha256,
                 split_sha256, graph_cache_sha256):
        if set(feature_frame.columns) != set(FEATURE_COLUMNS):
            raise ValueError("RowDataset accepts only the seven label-free feature columns")
        self.feature_frame = feature_frame.loc[:, list(FEATURE_COLUMNS)].copy().reset_index(drop=True)
        if set(partition.columns) - set(PARTITION_COLUMNS) - {"study"}:
            raise ValueError("partition may contain only identity and role metadata")
        self.partition = partition.copy().reset_index(drop=True)
        validate_row_protocol(self.partition)
        if self.feature_frame.isna().any().any() or self.partition.loc[:, list(PARTITION_COLUMNS)].isna().any().any():
            raise ValueError("row features and partition identities must be complete")
        if self.feature_frame.sample_id.astype(str).duplicated().any():
            raise ValueError("row sample IDs must be unique")
        if len(self.partition) != len(self.feature_frame):
            raise ValueError("partition must cover the canonical feature rows")
        if self.partition.sample_id.astype(str).tolist() != self.feature_frame.sample_id.astype(str).tolist():
            raise ValueError("partition identity/order differs from canonical features")
        if self.partition.canonical_smiles.astype(str).tolist() != self.feature_frame.canonical_smiles.astype(str).tolist():
            raise ValueError("partition compound identities differ from canonical features")
        if not np.array_equal(self.partition.canonical_index.to_numpy(), np.arange(len(self.feature_frame))):
            raise ValueError("partition canonical indices must match canonical order")
        seeds = self.partition.outer_seed.unique()
        if len(seeds) != 1 or not float(seeds[0]).is_integer() or int(seeds[0]) < 0:
            raise ValueError("partition requires one nonnegative outer seed")
        self.seed = int(seeds[0])
        self.source_sha256 = _digest(source_sha256, "source")
        self.split_sha256 = _digest(split_sha256, "split")
        self.partition_hash = content_hash(self.partition.to_dict("list"))
        self.graph_cache_sha256 = _digest(graph_cache_sha256, "graph cache")
        self.graph_cache = dict(graph_cache)
        if missing := set(self.feature_frame.canonical_smiles.astype(str)) - set(self.graph_cache):
            raise ValueError(f"graph cache is missing {len(missing)} canonical compounds")
        self.l0_ids = self.ids_for_role("l0")
        self.u0_ids = self.ids_for_role("u0")
        self.validation_ids = self.ids_for_role("validation")
        self.outer_train_indices = tuple(np.r_[np.flatnonzero(self.partition.role.eq("l0")), np.flatnonzero(self.partition.role.eq("u0"))].tolist())

    @classmethod
    def from_files(cls, *, source: SnapshotFile, partition: SnapshotFile, graph_cache: SnapshotFile):
        import torch
        source_path, partition_path, cache_path = source.verify(), partition.verify(), graph_cache.verify()
        features = pd.read_csv(source_path, usecols=list(FEATURE_COLUMNS)).reset_index(drop=True)
        roles = pd.read_csv(partition_path, usecols=lambda c: c in (*PARTITION_COLUMNS, "study"))
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cls(features, roles, cache, source_sha256=source.sha256,
                   split_sha256=partition.sha256, graph_cache_sha256=graph_cache.sha256)

    def ids_for_role(self, role):
        if role not in ROLE_ORDER:
            raise ValueError("unknown row role")
        return tuple(self.partition.loc[self.partition.role.eq(role), "sample_id"].astype(str))

    def prepare(self, l0_truth, scaler_path, *, forbidden_output_roots=()):
        scaler_path = safe_output(scaler_path, forbidden_output_roots)
        values = np.asarray(l0_truth, dtype=np.float32)
        if values.shape != (len(self.l0_ids), 2) or not np.isfinite(values).all():
            raise ValueError("only an aligned finite L0 target array can fit AL preprocessing")
        normalization, preprocessing = fit_al_preprocessing(
            self.feature_frame, self.graph_cache, self.outer_train_indices, self.l0_ids, values, scaler_path)
        scrubbed = self.feature_frame.copy()
        scrubbed["V1_ml"], scrubbed["V2_ml"] = 0.0, 0.0
        atom, angle = build_model_data(scrubbed, self.graph_cache, pd.DataFrame(), preprocessing["scaler"])
        rows = {}
        for identity, a, b in zip(self.feature_frame.sample_id.astype(str), atom, angle):
            a = a.clone()
            del a.y
            if "y" in b:
                raise ValueError("bond-angle graph unexpectedly contains targets")
            rows[identity] = GraphRow(identity, a, b.clone())
        if len(rows) != len(self.feature_frame):
            raise ValueError("graph builder changed canonical row count")
        return PreparedRowData(normalization, preprocessing, rows, self.source_sha256,
                               self.partition_hash, self.graph_cache_sha256, self.seed,
                               self.l0_ids, self.validation_ids)


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

"""Typed, unit-bearing input contract for Clean-QGeoGNN."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


CLEAN_MODEL_VARIANT = "qgeognn_clean_fusion_v1"
SOLVENT_VOCABULARY = ("PE", "EA", "DCM")
CONTINUOUS_CONDITION_NAMES = (
    "ea_fraction",
    "loading_mass_mg",
    "loading_solvent_volume_ul",
)
REQUIRED_FRAME_COLUMNS = (
    "PE/EA",
    "loading solvent",
    "Density g/ml",
    "V/ul",
    "Volume of loading solvent/ul",
)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_ea_fraction(value: str) -> float:
    """Parse a binary PE/EA ratio into its single independent fraction."""
    if not isinstance(value, str):
        raise TypeError("PE/EA must be a string formatted as left/right")
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError("PE/EA must contain exactly one '/' separator")
    try:
        left, right = (float(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("PE/EA components must be numeric") from exc
    if not np.isfinite([left, right]).all() or left < 0 or right < 0:
        raise ValueError("PE/EA components must be finite and non-negative")
    total = left + right
    if total <= 0:
        raise ValueError("PE/EA components must have a positive sum")
    return right / total


def loading_mass_mg(density_g_ml: float, volume_ul: float) -> float:
    """Return mass in mg; numerically g/ml multiplied by ul equals mg."""
    density, volume = float(density_g_ml), float(volume_ul)
    if not np.isfinite([density, volume]).all():
        raise ValueError("loading density and volume must be finite")
    if density < 0 or volume < 0:
        raise ValueError("loading density and volume must be non-negative")
    return density * volume


def clean_input_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_variant": CLEAN_MODEL_VARIANT,
        "molecular_topology": {
            "atom_features": "legacy categorical atom vocabulary",
            "bond_categorical_features": ["bond_dir", "bond_type", "is_in_ring"],
        },
        "molecular_geometry_and_descriptors": {
            "bond_continuous_features": ["bond_length"],
            "bond_angle_features": ["bond_angle", "molecular_descriptors"],
            "dtype": "float32",
        },
        "experimental_conditions": {
            "continuous": [
                {"name": "ea_fraction", "dtype": "float32", "unit": "fraction", "normalization": "source_train_zscore"},
                {"name": "loading_mass_mg", "dtype": "float32", "unit": "mg", "normalization": "source_train_zscore"},
                {"name": "loading_solvent_volume_ul", "dtype": "float32", "unit": "ul", "normalization": "source_train_zscore"},
            ],
            "categorical": [{"name": "loading_solvent", "dtype": "int64", "vocabulary": list(SOLVENT_VOCABULARY)}],
            "scope": "sample_or_graph_level",
        },
        "column_context": {
            "included_in_clean_4g": False,
            "reason": "constant within the 4g source domain",
            "future_fields": [
                "packing_mass", "column_length", "column_diameter", "column_volume",
                "stationary_phase", "flow_or_linear_velocity", "loading_ratio",
            ],
        },
    }


def clean_input_schema_hash(schema: dict[str, Any] | None = None) -> str:
    return _stable_hash(clean_input_schema() if schema is None else schema)


def clean_condition_schema_hash() -> str:
    return _stable_hash(clean_input_schema()["experimental_conditions"])


@dataclass(frozen=True)
class CleanConditionNormalization:
    continuous_names: tuple[str, str, str]
    mean: tuple[float, float, float]
    scale: tuple[float, float, float]
    fit_dataset: str
    fit_role: str
    fit_row_count: int
    fit_ids_hash: str
    validation_rows_used: int = 0
    test_rows_used: int = 0
    target_8g_rows_used: int = 0

    def validate(self) -> None:
        if self.continuous_names != CONTINUOUS_CONDITION_NAMES:
            raise ValueError("clean condition normalization feature order mismatch")
        if self.fit_dataset != "4g" or self.fit_role != "source_train":
            raise ValueError("clean normalization must be fit on 4g source_train only")
        if self.fit_row_count < 1 or len(self.fit_ids_hash) != 64:
            raise ValueError("clean normalization requires auditable source-train IDs")
        if any(value != 0 for value in (self.validation_rows_used, self.test_rows_used, self.target_8g_rows_used)):
            raise ValueError("clean normalization cannot consume validation, test, or 8g rows")
        values = np.asarray((*self.mean, *self.scale), dtype=np.float64)
        if not np.isfinite(values).all() or any(value <= 0 for value in self.scale):
            raise ValueError("clean normalization statistics must be finite with positive scales")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleanConditionBatch:
    continuous: torch.Tensor
    loading_solvent: torch.Tensor
    sample_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.continuous.dtype != torch.float32:
            raise TypeError("continuous clean conditions must use float32")
        if self.loading_solvent.dtype != torch.int64:
            raise TypeError("loading solvent must use int64 categorical indices")
        if self.continuous.ndim != 2 or self.continuous.shape[1] != 3:
            raise ValueError("continuous clean conditions must have shape [batch, 3]")
        if self.loading_solvent.ndim != 1 or self.loading_solvent.shape[0] != self.continuous.shape[0]:
            raise ValueError("loading solvent must have shape [batch]")
        if len(self.sample_ids) != self.continuous.shape[0]:
            raise ValueError("sample ID count must match condition batch size")
        if not torch.isfinite(self.continuous).all():
            raise ValueError("continuous clean conditions must be finite")
        if torch.any(self.loading_solvent < 0) or torch.any(self.loading_solvent >= len(SOLVENT_VOCABULARY)):
            raise ValueError("loading solvent index is outside the clean vocabulary")

    def to(self, device: torch.device | str) -> "CleanConditionBatch":
        return CleanConditionBatch(
            continuous=self.continuous.to(device),
            loading_solvent=self.loading_solvent.to(device),
            sample_ids=self.sample_ids,
        )


def _raw_clean_conditions(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(REQUIRED_FRAME_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"clean condition columns missing: {missing}")
    ea_fraction = np.asarray([parse_ea_fraction(value) for value in frame["PE/EA"]], dtype=np.float32)
    masses = np.asarray([
        loading_mass_mg(density, volume)
        for density, volume in zip(frame["Density g/ml"], frame["V/ul"])
    ], dtype=np.float32)
    loading_volume = frame["Volume of loading solvent/ul"].to_numpy(dtype=np.float32)
    values = np.column_stack([ea_fraction, masses, loading_volume]).astype(np.float32)
    if not np.isfinite(values).all() or np.any(values[:, 2] < 0):
        raise ValueError("clean continuous conditions must be finite and non-negative")
    return values


def fit_clean_condition_normalization(
    canonical_data: pd.DataFrame,
    source_split: pd.DataFrame,
    *,
    dataset: str = "4g",
) -> CleanConditionNormalization:
    if dataset != "4g":
        raise ValueError("clean preflight normalization accepts only the 4g source dataset")
    required_split = {"sample_id", "split"}
    if not required_split.issubset(source_split.columns) or "sample_id" not in canonical_data.columns:
        raise ValueError("canonical data and source split require sample_id and split fields")
    roles = source_split[["sample_id", "split"]]
    joined = canonical_data.merge(roles, on="sample_id", how="left", validate="one_to_one")
    if joined["split"].isna().any():
        raise ValueError("source split does not cover every canonical 4g row")
    train = joined.loc[joined["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("source split has no train rows")
    values = _raw_clean_conditions(train)
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    ids = sorted(train["sample_id"].astype(str))
    normalization = CleanConditionNormalization(
        continuous_names=CONTINUOUS_CONDITION_NAMES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        fit_dataset="4g",
        fit_role="source_train",
        fit_row_count=len(train),
        fit_ids_hash=_stable_hash(ids),
    )
    normalization.validate()
    return normalization


def parse_clean_conditions(
    frame: pd.DataFrame,
    normalization: CleanConditionNormalization,
    *,
    sample_ids: Sequence[str] | None = None,
) -> CleanConditionBatch:
    normalization.validate()
    values = _raw_clean_conditions(frame)
    mean = np.asarray(normalization.mean, dtype=np.float32)
    scale = np.asarray(normalization.scale, dtype=np.float32)
    continuous = torch.tensor((values - mean) / scale, dtype=torch.float32)
    solvent_values = frame["loading solvent"].astype(str)
    unknown = sorted(set(solvent_values) - set(SOLVENT_VOCABULARY))
    if unknown:
        raise ValueError(f"unknown loading solvent categories: {unknown}")
    vocabulary = {value: index for index, value in enumerate(SOLVENT_VOCABULARY)}
    solvent = torch.tensor([vocabulary[value] for value in solvent_values], dtype=torch.int64)
    if sample_ids is None:
        ids = tuple(frame["sample_id"].astype(str)) if "sample_id" in frame else tuple(str(index) for index in frame.index)
    else:
        ids = tuple(str(value) for value in sample_ids)
    result = CleanConditionBatch(continuous=continuous, loading_solvent=solvent, sample_ids=ids)
    result.validate()
    return result

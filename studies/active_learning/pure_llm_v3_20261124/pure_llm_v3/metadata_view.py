"""Label-free metadata projection and exact condition identity for the frozen 4g Row benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.qgeognn_al.active_learning_v2.protocol import ids_hash
from src.qgeognn_al.artifacts import sha256_file


RAW_PATH = Path("dataset/dataset_4g.csv")
E0_PATH = Path("experiments/e0_4g_baseline")
BENCHMARK_PATH = Path("studies/active_learning/qgeognn_v2_row_small_batch_benchmark")
CONFIRMATION_SEEDS = (157, 887, 2357, 6101, 12203)
RAW_COLUMNS = ("CAS", "name", "Purity", "Density g/ml", "V/ul", "loading solvent",
               "Volume of loading solvent/ul", "PE/EA", "Flow mL/min", "Ending time",
               "column_specs", "smiles")
CANONICAL_COLUMNS = (*RAW_COLUMNS, "sample_id", "source_row_1based", "canonical_smiles")
CONDITION_AXES = ("CAS", "column_specs", "purity_raw", "density", "sample_volume",
                  "loading_solvent", "loading_solvent_volume", "PE/EA", "flow")
PUBLIC_CONDITIONS = ("column_specs", "CAS", "name", "purity_raw", "purity_fraction",
                     "purity_kind", "density", "sample_volume", "loading_solvent",
                     "loading_solvent_volume", "PE/EA", "flow", "EA_fraction")
LEGACY_AXES = ("density", "sample_volume", "loading_solvent", "loading_solvent_volume", "PE/EA")
NUMERIC_AXES = ("density", "sample_volume", "loading_solvent_volume", "flow")
RAW_NUMERIC = ("Density g/ml", "V/ul", "Volume of loading solvent/ul", "Flow mL/min")
ROW_COUNTS = {"l0": 333, "u0": 2997, "validation": 416, "test": 417}
_PURITY_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", flags=re.ASCII)


def projected_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read only the explicit projection with the original CSV text and no NA coercion."""
    return pd.read_csv(path, usecols=list(columns), dtype=str, keep_default_na=False,
                       encoding="utf-8-sig").loc[:, list(columns)].to_dict(orient="records")


def purity_value(raw: str) -> tuple[str, float | None]:
    """Preserve grades and ambiguous entries and derive only explicit percentages or fractions."""
    text = raw.strip()
    if text.casefold() in {"", "na", "n/a", "nan", "null", "none"}:
        return "missing", None
    percent = text.endswith("%")
    number = text[:-1].strip() if percent else text
    if _PURITY_NUMBER.fullmatch(number):
        value = Decimal(number)
        if 0 <= value <= (100 if percent else 1):
            return ("numeric_percent", float(value / 100)) if percent else ("numeric_fraction", float(value))
        return "ambiguous_numeric", None
    return "grade", None


def _ea_ratio(record: str) -> Fraction:
    try:
        left, right = (Fraction(Decimal(part.strip())) for part in record.split("/"))
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise ValueError(f"invalid PE/EA record {record!r}") from error
    if min(left, right) < 0 or left + right == 0:
        raise ValueError(f"invalid PE/EA record {record!r}")
    return right / (left + right)


def ea_fraction(record: str) -> float:
    return float(_ea_ratio(record))


def _condition_value(axis: str, value: str):
    if axis in NUMERIC_AXES and value:
        return Decimal(value)
    if axis == "PE/EA" and value:
        return _ea_ratio(value)
    return value


def condition_changes(left: Mapping, right: Mapping, axes=CONDITION_AXES) -> tuple[str, ...]:
    """Compare recorded axes without tolerances or converting raw purity grades."""
    return tuple(axis for axis in axes if _condition_value(axis, left[axis])
                 != _condition_value(axis, right[axis]))


@dataclass(frozen=True)
class Metadata:
    sample_id: str
    source_row_1based: int
    canonical_smiles: str
    source_smiles: str
    ending_time: str
    column_specs: str
    cas: str
    name: str
    purity_raw: str
    density: str
    sample_volume: str
    loading_solvent: str
    loading_solvent_volume: str
    pe_ea: str
    flow: str

    def public_conditions(self) -> dict[str, str | float | None]:
        purity_kind, purity = purity_value(self.purity_raw)
        return {
            "column_specs": self.column_specs, "CAS": self.cas, "name": self.name,
            "purity_raw": self.purity_raw, "purity_fraction": purity, "purity_kind": purity_kind,
            "density": self.density, "sample_volume": self.sample_volume,
            "loading_solvent": self.loading_solvent, "loading_solvent_volume": self.loading_solvent_volume,
            "PE/EA": self.pe_ea, "flow": self.flow, "EA_fraction": ea_fraction(self.pe_ea),
        }

    def identity_key(self, axes=CONDITION_AXES) -> tuple:
        conditions = self.public_conditions()
        return (self.canonical_smiles, *(_condition_value(axis, conditions[axis]) for axis in axes))


def row_id(source_hash: str, source_row_1based: int) -> str:
    return hashlib.sha256(f"4g:{source_hash}:{source_row_1based}".encode()).hexdigest()[:20]


def load_4g_metadata(root: Path) -> dict[str, Metadata]:
    """Verify E0 and Row benchmark credentials, then align each canonical row by source identity."""
    root = Path(root)
    raw_path = root / RAW_PATH
    canonical_path = root / E0_PATH / "canonical_4g.csv"
    e0 = json.loads((root / E0_PATH / "data_metadata.json").read_text())
    benchmark = json.loads((root / BENCHMARK_PATH / "protocol.json").read_text())
    raw_sha = sha256_file(raw_path)
    if e0["source_file"] != RAW_PATH.as_posix() or raw_sha != e0["source_sha256"]:
        raise ValueError("raw 4g source identity differs from E0")
    if sha256_file(canonical_path) != benchmark["source_sha256"]:
        raise ValueError("canonical 4g identity differs from the Row benchmark")
    raw_rows = projected_rows(raw_path, RAW_COLUMNS)
    canonical_rows = projected_rows(canonical_path, CANONICAL_COLUMNS)
    if len(raw_rows) != e0["source_rows"] or len(canonical_rows) != e0["canonical_rows"]:
        raise ValueError("raw or canonical 4g row count drift")

    records = {}
    for canonical in canonical_rows:
        source_row = int(canonical["source_row_1based"])
        sample_id = canonical["sample_id"]
        if not (2 <= source_row <= len(raw_rows) + 1) or sample_id != row_id(raw_sha, source_row):
            raise ValueError(f"invalid E0 source row identity for {sample_id!r}")
        if sample_id in records or not canonical["canonical_smiles"]:
            raise ValueError(f"incomplete or duplicate canonical identity {sample_id!r}")
        raw = raw_rows[source_row - 2]
        for column in RAW_COLUMNS:
            left, right = raw[column], canonical[column]
            if left == right:
                continue
            if not (column in RAW_NUMERIC and left and right and Decimal(left) == Decimal(right)):
                raise ValueError(f"raw/canonical projection mismatch for {sample_id}, field {column}")
        records[sample_id] = Metadata(
            sample_id, source_row, canonical["canonical_smiles"], raw["smiles"], raw["Ending time"],
            raw["column_specs"], raw["CAS"], raw["name"], raw["Purity"], raw["Density g/ml"], raw["V/ul"],
            raw["loading solvent"], raw["Volume of loading solvent/ul"], raw["PE/EA"], raw["Flow mL/min"],
        )
    if len({row.canonical_smiles for row in records.values()}) != e0["unique_canonical_smiles"]:
        raise ValueError("canonical structure count differs from E0")
    return records


def load_frozen_split(root: Path, seed: int, records: Mapping[str, Metadata]) -> dict[str, tuple[str, ...]]:
    """Use the saved Row split and its file, role, canonical index and structure identities."""
    directory = Path(root) / BENCHMARK_PATH / "splits"
    path = directory / f"row_seed_{seed}.csv"
    manifest = json.loads((directory / "split_manifest.json").read_text())
    expected = next((item for item in manifest["splits"] if item["outer_seed"] == seed), None)
    if expected is None or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"unverified Row split for seed {seed}")
    rows = projected_rows(path, ("sample_id", "canonical_smiles", "canonical_index", "role", "outer_seed"))
    found = {role: [] for role in ROW_COUNTS}
    if len(rows) != len(records):
        raise ValueError("split row count differs from canonical metadata")
    for index, (row, record) in enumerate(zip(rows, records.values())):
        if (row["sample_id"], row["canonical_smiles"], int(row["canonical_index"]), int(row["outer_seed"])) != (
                record.sample_id, record.canonical_smiles, index, seed):
            raise ValueError(f"Row split identity mismatch at canonical index {index}")
        found[row["role"]].append(record.sample_id)
    for role, values in found.items():
        if len(values) != ROW_COUNTS[role] or ids_hash(values) != expected[f"{role}_ids_hash"]:
            raise ValueError(f"Row split role mismatch for seed {seed} and role {role}")
    return {role: tuple(values) for role, values in found.items()}

#!/usr/bin/env python3
"""Generate the source-only 4g threshold/data-contract audit; never train a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataset/dataset_4g.csv"
OUT = ROOT / "studies/predictor/4g_source_benchmark/data_audit"
V1_LIMIT = 60.0
V2_LIMIT = 120.0


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(value: object) -> str | None:
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True)


def distribution(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {key: None for key in ("count", "min", "q1", "median", "q3", "max", "mean", "std")}
    return {
        "count": int(len(clean)),
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
    }


def categorical_distribution(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    total = len(frame)
    counts = frame[column].fillna("<NA>").astype(str).value_counts(dropna=False)
    return [
        {"value": value, "rows": int(count), "fraction": float(count / total) if total else None}
        for value, count in counts.items()
    ]


def concentration_table(valid: pd.DataFrame, group_columns: list[str]) -> list[dict[str, object]]:
    grouped = valid.groupby(group_columns, dropna=False)["threshold_affected"].agg(["sum", "size"]).reset_index()
    grouped["affected_fraction"] = grouped["sum"] / grouped["size"]
    grouped = grouped.sort_values(["sum", "affected_fraction", "size"], ascending=False)
    rows = []
    for record in grouped.to_dict(orient="records"):
        rows.append({
            **{name: str(record[name]) for name in group_columns},
            "affected_rows": int(record["sum"]),
            "valid_rows": int(record["size"]),
            "affected_fraction": float(record["affected_fraction"]),
        })
    return rows


def main() -> None:
    frame = pd.read_csv(SOURCE).copy()
    frame["source_row_1based"] = np.arange(len(frame), dtype=int) + 2
    source_digest = file_hash(SOURCE)
    frame["sample_id"] = [
        hashlib.sha256(f"{source_digest}:{row}".encode()).hexdigest()[:20]
        for row in frame["source_row_1based"]
    ]
    frame["canonical_smiles"] = frame["smiles"].map(canonicalize)

    numeric_names = ["t1", "t2", "Flow mL/min", "Density g/ml", "V/ul", "Volume of loading solvent/ul"]
    numeric = {name: pd.to_numeric(frame[name], errors="coerce") for name in numeric_names}
    frame["V1_ml_audit"] = numeric["t1"] * numeric["Flow mL/min"] / 1200.0
    frame["V2_ml_audit"] = numeric["t2"] * numeric["Flow mL/min"] / 1200.0
    frame["label_nan"] = frame[["V1_ml_audit", "V2_ml_audit"]].isna().any(axis=1)
    frame["label_sentinel"] = numeric["t1"].eq(-1) | numeric["t2"].eq(-1)
    frame["label_finite"] = np.isfinite(frame["V1_ml_audit"]) & np.isfinite(frame["V2_ml_audit"])
    frame["valid_label"] = frame["label_finite"] & ~frame["label_sentinel"]
    frame["over_v1_threshold"] = frame["valid_label"] & frame["V1_ml_audit"].gt(V1_LIMIT)
    frame["over_v2_threshold"] = frame["valid_label"] & frame["V2_ml_audit"].gt(V2_LIMIT)
    frame["threshold_affected"] = frame["over_v1_threshold"] | frame["over_v2_threshold"]

    valid = frame.loc[frame["valid_label"]].copy()
    retained = valid.loc[~valid["threshold_affected"]].copy()
    affected = valid.loc[valid["threshold_affected"]].copy()
    compound_groups = valid.groupby("canonical_smiles", dropna=False)["threshold_affected"].agg(["all", "sum", "size"])
    complete = compound_groups.loc[compound_groups["all"]]

    continuous_conditions = ["Flow mL/min", "Density g/ml", "V/ul", "Volume of loading solvent/ul"]
    condition_distributions = {}
    for population_name, population in (("valid_before_filtering", valid), ("retained_after_filtering", retained), ("threshold_affected", affected)):
        condition_distributions[population_name] = {
            "continuous": {name: distribution(population[name]) for name in continuous_conditions},
            "loading_solvent": categorical_distribution(population, "loading solvent"),
            "eluent_ratio": categorical_distribution(population, "PE/EA"),
        }

    audit = {
        "audit_type": "source_only_data_audit_no_model_training",
        "source_file": "dataset/dataset_4g.csv",
        "source_sha256": source_digest,
        "volume_formula": "V_ml = t_raw * Flow_mL_min / 1200",
        "thresholds_ml": {"V1": V1_LIMIT, "V2": V2_LIMIT},
        "counts": {
            "raw_rows": int(len(frame)),
            "valid_label_rows": int(frame["valid_label"].sum()),
            "unique_compounds_among_valid_labels": int(valid["canonical_smiles"].nunique(dropna=True)),
            "v1_over_60_rows": int(frame["over_v1_threshold"].sum()),
            "v2_over_120_rows": int(frame["over_v2_threshold"].sum()),
            "union_removed_rows": int(frame["threshold_affected"].sum()),
            "affected_unique_compounds": int(affected["canonical_smiles"].nunique(dropna=True)),
            "compounds_completely_removed": int(len(complete)),
            "retained_rows": int(len(retained)),
        },
        "label_distributions": {
            "before_filtering": {"V1_ml": distribution(valid["V1_ml_audit"]), "V2_ml": distribution(valid["V2_ml_audit"])},
            "after_filtering": {"V1_ml": distribution(retained["V1_ml_audit"]), "V2_ml": distribution(retained["V2_ml_audit"])},
            "affected_only": {"V1_ml": distribution(affected["V1_ml_audit"]), "V2_ml": distribution(affected["V2_ml_audit"])},
        },
        "removed_label_validity": {
            "affected_rows_with_nan": int(affected["label_nan"].sum()),
            "affected_rows_with_invalid_sentinel": int(affected["label_sentinel"].sum()),
            "affected_rows_nonfinite": int((~affected["label_finite"]).sum()),
            "affected_rows_otherwise_finite_numeric": int((affected["label_finite"] & ~affected["label_sentinel"]).sum()),
            "interpretation": "All threshold-affected labels are finite numeric observations under the repository conversion; this alone does not establish measurement validity or absence of censoring.",
        },
        "experimental_condition_distributions": condition_distributions,
        "concentration": {
            "by_loading_solvent": concentration_table(valid, ["loading solvent"]),
            "by_eluent_ratio": concentration_table(valid, ["PE/EA"]),
            "by_loading_solvent_and_eluent_ratio": concentration_table(valid, ["loading solvent", "PE/EA"]),
            "by_compound_top_20": concentration_table(valid, ["canonical_smiles"])[:20],
        },
        "rationale_trace": {
            "official_released_code": "application/QGeoGNN.py::Construct_dataset hard-codes V1<=60 and V2<=120",
            "paper_methods": "No paper-level physical, instrument-range, censoring, or measurement-validity rationale located",
            "official_repository_readme_docs": "No rationale for 60/120 located",
            "current_repository_docs": "Previously recorded as a legacy implementation threshold pending audit; no scientific rationale supplied",
            "conclusion": "NO_CONFIRMED_PAPER_LEVEL_RATIONALE_FOUND",
        },
        "policy_boundary": "The audit neither concludes thresholds_are_wrong nor thresholds_are_correct. Policy must be resolved from measurement validity, censoring, instrument range, or explicit experimental definition—not model performance.",
    }

    data_contract = {
        "dataset": "4g_source",
        "source_file": "dataset/dataset_4g.csv",
        "source_sha256": source_digest,
        "raw_row_identity": "source_row_1based plus source-derived sample_id",
        "compound_identity": "RDKit canonical_smiles",
        "valid_label_definition": "finite numeric t1, t2, and flow-derived V1/V2, excluding explicit -1 t1/t2 sentinel",
        "volume_formula": "t_seconds * Flow_mL_min / 1200",
        "legacy_code_filter": {"V1_ml_max_inclusive": V1_LIMIT, "V2_ml_max_inclusive": V2_LIMIT},
        "threshold_scientific_status": "UNRESOLVED_CODE_LEVEL_FILTER",
        "rationale_status": "NO_CONFIRMED_PAPER_LEVEL_RATIONALE_FOUND",
        "formal_benchmark_threshold_policy": "UNRESOLVED; no performance-based choice permitted",
        "models_trained_by_audit": 0,
        "target_8g_rows_used": 0,
    }

    affected_columns = [
        "sample_id", "source_row_1based", "CAS", "name", "canonical_smiles", "Density g/ml", "V/ul",
        "loading solvent", "Volume of loading solvent/ul", "PE/EA", "Flow mL/min", "t1", "t2",
        "V1_ml_audit", "V2_ml_audit", "over_v1_threshold", "over_v2_threshold", "label_nan",
        "label_sentinel", "label_finite",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "threshold_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    (OUT / "data_contract.json").write_text(json.dumps(data_contract, indent=2, ensure_ascii=False) + "\n")
    affected[affected_columns].to_csv(OUT / "threshold_affected_rows.csv", index=False)


if __name__ == "__main__":
    main()

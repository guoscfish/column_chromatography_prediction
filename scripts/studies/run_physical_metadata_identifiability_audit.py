#!/usr/bin/env python3
"""Audit physical-column metadata provenance and identifiability without outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "studies/transfer/physical_metadata_identifiability_audit"
PHYSICS = ROOT / "studies/transfer/physics_column_conditioned_transfer"

COLUMNS = {
    "4g": {"mass": 4.0, "flows": [4.0, 5.0, 6.0, 8.0, 10.0], "flow_mode": 10.0,
           "legacy": [1.5, 6.6, 0.4458]},
    "8g": {"mass": 8.0, "flows": [10.0], "flow_mode": 10.0,
           "legacy": [1.5, 13.2, 0.4458]},
    "25g": {"mass": 25.0, "flows": [15.0], "flow_mode": 15.0,
            "legacy": [2.15, 15.6, 0.5248]},
    "40g": {"mass": 40.0, "flows": [30.0], "flow_mode": 30.0,
            "legacy": [2.15, 15.6, 0.5248]},
}
FEATURES = ["packing_mass_g", "flow_ml_min", "legacy_column_dia",
            "legacy_column_len", "legacy_column_den"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_contract() -> pd.DataFrame:
    rows = []
    for name, source, unit in [
        ("eluent_exact_mol_wt", "PE/EA -> RDKit weighted descriptor", "g/mol"),
        ("eluent_tpsa", "PE/EA -> RDKit weighted descriptor", "angstrom^2"),
        ("eluent_rotatable_bonds", "PE/EA -> RDKit weighted descriptor", "count"),
        ("eluent_h_donors", "PE/EA -> RDKit weighted descriptor", "count"),
        ("eluent_h_acceptors", "PE/EA -> RDKit weighted descriptor", "count"),
        ("eluent_logp", "PE/EA -> RDKit weighted descriptor", "dimensionless"),
        ("loading_solvent_code", "loading solvent mapping PE=0, EA=1, DCM=2", "categorical code"),
        ("loading_amount_density_x_volume", "Density g/ml * V/ul", "mg"),
        ("loading_solvent_volume_ul", "Volume of loading solvent/ul", "uL"),
    ]:
        rows.append({"feature_name": name, "source": source, "unit_if_known": unit,
                     "sample_level_or_column_level": "sample_level",
                     "included_in_original_qgeognn": True,
                     "included_in_previous_physics_transfer": "base_model_only"})
    for name, source, unit in [
        ("flow_ml_min", "Flow mL/min", "mL/min"),
        ("packing_mass_g", "parsed from column_specs", "g"),
        ("column_geometry", "not in current model", "unknown"),
        ("column_identity", "not in current model", "not_applicable"),
    ]:
        rows.append({"feature_name": name, "source": source, "unit_if_known": unit,
                     "sample_level_or_column_level": "sample_level" if name == "flow_ml_min" else "column_level",
                     "included_in_original_qgeognn": False,
                     "included_in_previous_physics_transfer": name in {"flow_ml_min", "packing_mass_g"}})
    return pd.DataFrame(rows)


def _provenance() -> pd.DataFrame:
    common = {
        "unit_explicitly_stated": False,
        "physical_meaning_explicitly_stated": "variable_names_only",
        "provenance_class": "SEMANTICS_SUGGESTED",
        "previous_physics_context_input": False,
    }
    rows = [
        {"file": "application/QGeoGNN.py", "line_function": "945-946 feature schema; 1027-1029 RBF registry",
         "column": "all optional legacy paths", "numeric_values": "RBF ranges only",
         "local_variable_names": "column_dia|column_len|column_den",
         "how_values_enter_computation": "optional bond-edge continuous features when Use_column_info=True",
         "comments": "Use_column_info is globally False at line 921", "same_values_reused_elsewhere": True, **common},
        {"file": "application/QGeoGNN.py", "line_function": "1653-1658 Construct_dataset_8g",
         "column": "8g", "numeric_values": "1.5|13.2|0.4458",
         "local_variable_names": "diameter|column_length|density",
         "how_values_enter_computation": "repeated per bond edge and concatenated if Use_column_info=True",
         "comments": "separate 8g descriptor files", "same_values_reused_elsewhere": True, **common},
        {"file": "application/QGeoGNN.py", "line_function": "1720-1725 Construct_dataset_25g",
         "column": "25g", "numeric_values": "2.15|15.6|0.5248",
         "local_variable_names": "diameter|column_length|density",
         "how_values_enter_computation": "repeated per bond edge and concatenated if Use_column_info=True",
         "comments": "separate 25g descriptor files", "same_values_reused_elsewhere": True, **common},
        {"file": "application/QGeoGNN.py", "line_function": "1787-1792 Construct_dataset_40g",
         "column": "40g", "numeric_values": "2.15|15.6|0.5248",
         "local_variable_names": "diameter|column_length|density",
         "how_values_enter_computation": "repeated per bond edge and concatenated if Use_column_info=True",
         "comments": "separate 40g descriptor files; tuple copied exactly from 25g path",
         "same_values_reused_elsewhere": True, **common},
        {"file": "application/QGeoGNN.py", "line_function": "1851-1856 Construct_dataset_DCM",
         "column": "DCM legacy dataset", "numeric_values": "2.15|15.6|0.5248",
         "local_variable_names": "diameter|column_length|density",
         "how_values_enter_computation": "optional bond-edge features",
         "comments": "does not establish a 25g/40g measurement source", "same_values_reused_elsewhere": True, **common},
        {"file": "application/QGeoGNN.py", "line_function": "1917-1922 Construct_dataset_C18",
         "column": "C18 legacy dataset", "numeric_values": "2.15|15.6|0.5248",
         "local_variable_names": "diameter|column_length|density",
         "how_values_enter_computation": "optional bond-edge features",
         "comments": "reuse across a different stationary-phase path weakens provenance",
         "same_values_reused_elsewhere": True, **common},
        {"file": "application/QGeoGNN.py", "line_function": "3852-3857 predict_separate",
         "column": "4g/single-sample path", "numeric_values": "1.5|6.6|0.4458",
         "local_variable_names": "diameter|column_length|density",
         "how_values_enter_computation": "optional bond-edge inference features",
         "comments": "no unit or measurement record", "same_values_reused_elsewhere": True, **common},
        {"file": "scripts/run_g0_4_paper_style_transfer.py", "line_function": "56-73 constants/add_column_spec",
         "column": "4g|8g", "numeric_values": "1.5|6.6|0.4458;1.5|13.2|0.4458",
         "local_variable_names": "COLUMN_SPEC_4G|COLUMN_SPEC_8G|column_dia|column_len|column_den",
         "how_values_enter_computation": "explicitly appended to every graph edge for paper_style arm",
         "comments": "declared as repository-original values, not verified metadata",
         "same_values_reused_elsewhere": True, **common},
        {"file": "scripts/studies/run_paper_transfer_reproduction_25g_40g.py",
         "line_function": "71-87 COLUMN_CONFIG", "column": "25g|40g",
         "numeric_values": "2.15|15.6|0.5248 for both",
         "local_variable_names": "COLUMN_CONFIG.spec|SHARED_SPEC_FLAG",
         "how_values_enter_computation": "add_column_spec appends tuple to every graph edge",
         "comments": "explicitly flags 25g/40g shared legacy values",
         "same_values_reused_elsewhere": True, **common},
    ]
    return pd.DataFrame(rows)


def _column_identifiability() -> pd.DataFrame:
    rows = []
    for column, item in COLUMNS.items():
        values = [item["mass"], item["flow_mode"], *item["legacy"]]
        for feature, value in zip(FEATURES, values):
            observed = item["flows"] if feature == "flow_ml_min" else [value]
            rows.append({
                "column": column, "feature": feature, "representative_value": value,
                "all_observed_values": "|".join(f"{v:g}" for v in observed),
                "n_unique_within_column": len(set(observed)),
                "within_column_variance": float(np.var(observed)),
                "target_column_constant": column in {"8g", "25g", "40g"},
                "provenance_class": "VERIFIED_IN_CODE" if feature in {"packing_mass_g", "flow_ml_min"}
                else "SEMANTICS_SUGGESTED",
                "used_in_previous_physics_transfer": feature in {"packing_mass_g", "flow_ml_min"},
            })
    frame = pd.DataFrame(rows)
    between = frame.groupby("feature").representative_value.var(ddof=0)
    frame["between_column_variance"] = frame.feature.map(between)
    return frame


def _matrix(columns: list[str], feature_names: list[str]) -> np.ndarray:
    rows = []
    for column in columns:
        item = COLUMNS[column]
        mapping = dict(zip(FEATURES, [item["mass"], item["flow_mode"], *item["legacy"]]))
        rows.append([mapping[name] for name in feature_names])
    return np.asarray(rows, dtype=float)


def _rank_record(columns: list[str], feature_names: list[str]) -> dict:
    raw = _matrix(columns, feature_names)
    scale = raw.std(axis=0)
    standardized = (raw - raw.mean(axis=0)) / np.where(scale < 1e-12, 1.0, scale)
    design = np.column_stack([np.ones(len(raw)), standardized])
    correlation = pd.DataFrame(raw, columns=feature_names).corr().fillna(0.0)
    exact_affine_dependencies = []
    for left_index, left_name in enumerate(feature_names):
        for right_index in range(left_index + 1, len(feature_names)):
            right_name = feature_names[right_index]
            left = raw[:, left_index]
            right = raw[:, right_index]
            if np.ptp(left) < 1e-12 or np.ptp(right) < 1e-12:
                continue
            coefficient, intercept = np.polyfit(left, right, 1)
            if np.allclose(right, coefficient * left + intercept, rtol=0.0, atol=1e-12):
                exact_affine_dependencies.append({
                    "dependent_feature": right_name,
                    "reference_feature": left_name,
                    "coefficient": float(coefficient),
                    "intercept": float(intercept),
                })
    return {
        "columns": columns, "features": feature_names, "includes_intercept": True,
        "observations": len(columns), "rank": int(np.linalg.matrix_rank(design)),
        "maximum_possible_rank": min(design.shape),
        "condition_number_standardized_design": float(np.linalg.cond(design)),
        "unique_values": {name: int(len(np.unique(raw[:, index])))
                          for index, name in enumerate(feature_names)},
        "pairwise_correlation": correlation.to_dict(),
        "exact_affine_dependencies": exact_affine_dependencies,
    }


def _rank_audit() -> dict:
    target = ["8g", "25g", "40g"]
    all_columns = ["4g", *target]
    existing = ["packing_mass_g", "flow_ml_min"]
    legacy = ["legacy_column_dia", "legacy_column_len", "legacy_column_den"]
    records = {}
    for scope, columns in (("targets_only", target), ("source_and_targets", all_columns)):
        records[scope] = {
            "existing_mass_flow": _rank_record(columns, existing),
            "existing_plus_legacy": _rank_record(columns, existing + legacy),
        }
    return {
        "representative_flow_policy": "modal flow; full distributions retained in column_context_identifiability.csv",
        "records": records,
        "target_rank_increment_from_legacy": (
            records["targets_only"]["existing_plus_legacy"]["rank"]
            - records["targets_only"]["existing_mass_flow"]["rank"]
        ),
        "source_plus_target_rank_increment_from_legacy": (
            records["source_and_targets"]["existing_plus_legacy"]["rank"]
            - records["source_and_targets"]["existing_mass_flow"]["rank"]
        ),
        "packing_mass_and_flow_independently_identifiable": False,
        "reason": "one target flow per column and only three target-column contexts; mass, flow, legacy tuple, and column identity are confounded",
        "legacy_descriptors_deterministic_by_column_identity": True,
        "causal_interpretation_supported": False,
    }


def _eligibility(rank: dict) -> dict:
    criteria = {
        "new_descriptor_unused_by_previous_physics_study": True,
        "provenance_verified_or_very_strong_semantics": False,
        "not_equivalent_to_mass_or_flow": True,
        "adds_target_context_rank_or_new_independent_contrast": rank["target_rank_increment_from_legacy"] > 0,
        "not_bare_column_id_encoding": True,
    }
    return {
        "status": "ELIGIBLE_FOR_PHYSICS_CENTER_WIDTH_TRANSFER" if all(criteria.values())
        else "NO_NEW_IDENTIFIABLE_PHYSICAL_CONTEXT",
        "criteria": criteria,
        "eligibility_pass": all(criteria.values()),
        "model_training_authorized": False,
        "outer_truth_read": False,
        "blocking_findings": [
            "legacy names suggest diameter/length/density but units and measurement provenance are absent",
            "legacy tuple adds zero rank across the eligible 8g/25g/40g target contexts",
            "25g and 40g share the exact tuple, so it supplies no contrast between them",
            "all column-level descriptors are deterministic functions of four column identities in this repository",
        ],
        "next_actions": ["verify metadata with experimental staff", "proceed to a separately preregistered filtered random learning curve"],
    }


def _report(rank: dict, eligibility: dict) -> str:
    target_existing = rank["records"]["targets_only"]["existing_mass_flow"]
    target_all = rank["records"]["targets_only"]["existing_plus_legacy"]
    source_existing = rank["records"]["source_and_targets"]["existing_mass_flow"]
    source_all = rank["records"]["source_and_targets"]["existing_plus_legacy"]
    return f"""# Physical column metadata provenance and identifiability audit

## Decision

**{eligibility['status']}**. No physics-conditioned center/width model was trained and no outer truth was read.

## Previous physics study

The prior study completed 120 contexts, 240 neural fits, and 120 residual fits with zero failed or non-finite fits. Predictions were frozen before test evaluation; no test labels entered fitting. It tested packing-mass physical scaling, a physics-scale residual model, and raw/mass-normalized column-conditioned neural arms. Its explicit physical context was nominal packing mass [g], flow [mL/min], loading mass per packing mass [mg/g], and loading-solvent volume per packing mass [uL/g]. The final decision was `CURRENT_DATA_DO_NOT_IDENTIFY_COLUMN_PHYSICS_SUFFICIENTLY`.

Flow and packing mass are therefore not new candidates. Target flow is constant within each target column: 8g=10, 25g=15, and 40g=30 mL/min. Adding flow to separately fitted 25g or 40g models provides zero within-column information.

## Current QGeoGNN condition contract

The current model has nine sample-level condition inputs: six PE/EA-derived eluent descriptors, loading-solvent code, density times loading volume, and loading-solvent volume. It does not include flow, packing mass, geometry, or column identity. See `current_model_condition_contract.csv` for sources and units.

## Legacy constants and provenance

Released code labels optional fields `column_dia`, `column_len`, and `column_den`, and local variables `diameter`, `column_length`, and `density`. This supports `SEMANTICS_SUGGESTED`, not `VERIFIED_IN_CODE`: no unit, measurement record, product/lot, bed-versus-housing definition, or experimental provenance is present. `Use_column_info=False` in the original application path.

Classification is therefore: nominal packing mass and recorded flow are `VERIFIED_IN_CODE`; every legacy tuple is `SEMANTICS_SUGGESTED`; no tuple reaches `VERIFIED_IN_CODE`. None is placed in `UNVERIFIED_LEGACY_CONSTANT` because the variable names do suggest semantics, but that naming evidence is not measurement verification.

Tuples are 4g `(1.5, 6.6, 0.4458)`, 8g `(1.5, 13.2, 0.4458)`, and both 25g and 40g `(2.15, 15.6, 0.5248)`. The 25g and 40g constructors use separate dataset files but repeat the same constants. The repository supplies no reason proving common cartridge geometry or a 40g-specific alternative. Status: `UNRESOLVED_25G_40G_LEGACY_METADATA`.

The old physics study did not use these tuples, so they are new only in the narrow implementation-history sense. They are not newly verified physical measurements.

## Rank and confounding

With an intercept, target-only 8g/25g/40g mass+flow design rank is {target_existing['rank']} of {target_existing['maximum_possible_rank']}; adding all legacy fields remains rank {target_all['rank']}, increment 0. Across 4g/8g/25g/40g, mass+flow rank is {source_existing['rank']} and legacy fields raise it to {source_all['rank']}. That extra fourth-context contrast is saturated by four named columns and does not create an estimable geometry effect for the three target-column shared model.

Legacy dia and den take only two values; 25g/40g are identical. All target-column variables are fixed within column and therefore deterministic functions of column identity. Packing mass and flow cannot be causally or independently identified from this observational design. The repository does not support causal mass, flow, diameter, length, density, area, velocity, or bed-volume claims. No derived geometry proxies were constructed.

## Eligibility

The legacy descriptors were absent from the previous physics context and are not numerically identical to mass/flow, but fail the provenance and independent-target-contrast requirements. Treating them as standardized features would effectively encode uncertain column identity. The eligibility gate therefore fails and the study stops before model construction.

## Required external verification

Ask experimental staff for inner diameter and units, actual packed-bed length and units, meaning and units of `column_den`, cartridge manufacturer/model/lot, whether values refer to bed or housing, measured packed-bed and void/dead volumes with method, particle size, silica bulk density, 4g+4g connection/tubing volume, and the reason 25g/40g share a tuple. Crossed mass-by-flow experiments and independent column batches are needed for identifiability.

## Next stage

After recording this negative eligibility result, proceed only through a separately preregistered filtered random learning curve at B=30/50/100/150/200/FULL. This audit does not start that curve or Active Learning.
"""


def run() -> dict:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    contract = _condition_contract()
    provenance = _provenance()
    identifiability = _column_identifiability()
    rank = _rank_audit()
    eligibility = _eligibility(rank)
    contract.to_csv(OUT / "current_model_condition_contract.csv", index=False)
    provenance.to_csv(OUT / "legacy_column_constant_provenance.csv", index=False)
    identifiability.to_csv(OUT / "column_context_identifiability.csv", index=False)
    (OUT / "column_context_design_rank.json").write_text(json.dumps(rank, indent=2) + "\n")
    (OUT / "PHYSICS_MODEL_ELIGIBILITY.json").write_text(json.dumps(eligibility, indent=2) + "\n")
    (OUT / "README.md").write_text("# Physical metadata identifiability audit\n\nOutcome-blind provenance and rank audit governing eligibility for any new physics-conditioned center/width transfer model.\n")
    (OUT / "FINAL_REPORT.md").write_text(_report(rank, eligibility))
    metadata = {
        "elapsed_seconds": time.perf_counter() - started,
        "physics_execution_audit_sha256": _sha(PHYSICS / "execution_audit.json"),
        "physics_decision_sha256": _sha(PHYSICS / "decision.json"),
        "outer_truth_read": False,
        "old_physics_study_rerun": False,
        "new_model_trained": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    names = ["README.md", "FINAL_REPORT.md", "current_model_condition_contract.csv",
             "legacy_column_constant_provenance.csv", "column_context_identifiability.csv",
             "column_context_design_rank.json", "PHYSICS_MODEL_ELIGIBILITY.json", "run_metadata.json"]
    (OUT / "artifact_manifest.json").write_text(json.dumps({
        "study": "PHYSICAL_COLUMN_METADATA_PROVENANCE_AND_IDENTIFIABILITY_AUDIT",
        "files": {name: _sha(OUT / name) for name in names},
    }, indent=2) + "\n")
    return eligibility


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

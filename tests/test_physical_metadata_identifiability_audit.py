from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/transfer/physical_metadata_identifiability_audit"


def test_current_qgeognn_has_exact_nine_original_conditions_without_column_context():
    contract = pd.read_csv(STUDY / "current_model_condition_contract.csv")
    original = contract.loc[contract.included_in_original_qgeognn.astype(bool)]
    assert len(original) == 9
    assert set(original.feature_name) == {
        "eluent_exact_mol_wt", "eluent_tpsa", "eluent_rotatable_bonds",
        "eluent_h_donors", "eluent_h_acceptors", "eluent_logp",
        "loading_solvent_code", "loading_amount_density_x_volume",
        "loading_solvent_volume_ul",
    }
    excluded = contract.set_index("feature_name").included_in_original_qgeognn
    assert not bool(excluded["flow_ml_min"])
    assert not bool(excluded["packing_mass_g"])
    assert not bool(excluded["column_geometry"])
    assert not bool(excluded["column_identity"])


def test_legacy_constants_are_suggested_not_verified_and_unused_by_physics_study():
    provenance = pd.read_csv(STUDY / "legacy_column_constant_provenance.csv")
    assert set(provenance.provenance_class) == {"SEMANTICS_SUGGESTED"}
    assert not provenance.unit_explicitly_stated.astype(bool).any()
    assert not provenance.previous_physics_context_input.astype(bool).any()
    joined = " ".join(provenance.numeric_values.astype(str))
    for value in ("1.5", "6.6", "13.2", "2.15", "15.6", "0.4458", "0.5248"):
        assert value in joined


def test_flow_support_and_shared_25g_40g_tuple_are_recorded():
    context = pd.read_csv(STUDY / "column_context_identifiability.csv")
    flow = context.loc[context.feature.eq("flow_ml_min")].set_index("column")
    assert flow.loc["4g", "all_observed_values"] == "4|5|6|8|10"
    assert flow.loc["8g", "all_observed_values"] == "10"
    assert flow.loc["25g", "all_observed_values"] == "15"
    assert flow.loc["40g", "all_observed_values"] == "30"
    legacy = context.loc[context.feature.str.startswith("legacy_")]
    pivot = legacy.pivot(index="column", columns="feature", values="representative_value")
    pd.testing.assert_series_equal(pivot.loc["25g"], pivot.loc["40g"], check_names=False)


def test_target_rank_does_not_increase_and_causal_identifiability_fails():
    audit = json.loads((STUDY / "column_context_design_rank.json").read_text())
    target = audit["records"]["targets_only"]
    assert target["existing_mass_flow"]["rank"] == 3
    assert target["existing_plus_legacy"]["rank"] == 3
    exact = target["existing_plus_legacy"]["exact_affine_dependencies"]
    exact_pairs = {(item["reference_feature"], item["dependent_feature"]) for item in exact}
    assert ("legacy_column_dia", "legacy_column_len") in exact_pairs
    assert ("legacy_column_dia", "legacy_column_den") in exact_pairs
    assert audit["target_rank_increment_from_legacy"] == 0
    assert audit["source_plus_target_rank_increment_from_legacy"] == 1
    assert audit["packing_mass_and_flow_independently_identifiable"] is False
    assert audit["legacy_descriptors_deterministic_by_column_identity"] is True
    assert audit["causal_interpretation_supported"] is False


def test_eligibility_fails_before_any_model_or_outer_stage():
    decision = json.loads((STUDY / "PHYSICS_MODEL_ELIGIBILITY.json").read_text())
    metadata = json.loads((STUDY / "run_metadata.json").read_text())
    assert decision["status"] == "NO_NEW_IDENTIFIABLE_PHYSICAL_CONTEXT"
    assert decision["eligibility_pass"] is False
    assert decision["model_training_authorized"] is False
    assert decision["outer_truth_read"] is False
    assert metadata["old_physics_study_rerun"] is False
    assert metadata["new_model_trained"] is False
    assert metadata["outer_truth_read"] is False
    assert not (ROOT / "studies/transfer/physics_center_width_transfer").exists()


def test_previous_physics_execution_counts_remain_complete():
    audit = json.loads((ROOT / "studies/transfer/physics_column_conditioned_transfer/execution_audit.json").read_text())
    assert audit["contexts"] == 120
    assert audit["neural_fits"] == 240
    assert audit["residual_fits"] == 120
    assert audit["failed_fits"] == 0
    assert audit["test_labels_used_for_fit"] == 0

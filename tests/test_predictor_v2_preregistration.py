from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/track_b_transfer/predictor_v2_preregistration"


def read(name: str) -> dict:
    return json.loads((STUDY / name).read_text())


def test_predictor_v2_is_preregistration_only_and_unauthorized() -> None:
    config = read("config.json")
    decision = read("decision.json")
    plan = read("formal_plan_audit.json")
    for artifact in (config, decision, plan):
        assert artifact["status" if "status" in artifact else "decision"] == "PREREGISTRATION_ONLY"
        assert artifact["implementation_started"] is False
        assert artifact["formal_authorized"] is False
    assert config["active_transfer"] == "deferred"
    assert plan["expected_formal_fits"] == 0
    assert plan["gate_pass"] is False


def test_predictor_v2_design_is_typed_and_condition_complete() -> None:
    design = read("design_spec.json")
    inputs = design["intended_inputs"]
    conditions = (
        inputs["eluent_continuous_features"]
        + inputs["loading_categorical_features"]
        + inputs["loading_continuous_features"]
    )
    assert len(conditions) == 9
    assert len(set(conditions)) == 9
    assert design["typing"]["loading_solvent"].startswith("categorical")
    assert design["normalization"]["continuous_loading_features"] == "source_train_only_fitted_scaler"
    assert design["normalization"]["forbid_old_unscaled_rbf_centers_for_mass_or_volume"] is True
    assert design["checkpoint_contract"]["input_schema_hash_required"] is True


def test_residual_recommendation_requires_identity_but_is_not_locked() -> None:
    design = read("design_spec.json")
    recommendation = design["recommendation"]
    residual = design["alternatives"]["B_legacy_compatible_residual_condition_completion"]
    assert recommendation["candidate"].startswith("B_")
    assert recommendation["final_architecture_locked"] is False
    assert residual["source_function_identity"] is True
    assert residual["parameter_count"] == "TBD_by_implementation_preflight"

"""Synthetic validation, repair and memory tests. No labels or provider access."""
from copy import deepcopy
from functools import partial

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.deterministic_validator import validate_plan
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import ea_fraction
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.repair_protocol import (
    ReplanRequested, apply_repair, build_repair_request,
)
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError, parse_response

CHECKPOINT = "a" * 64


def card(key, *, same=True, **overrides):
    conditions = {
        "CAS": "example", "column_specs": "4g", "purity_raw": "AR",
        "density": "1.00009", "sample_volume": "5", "loading_solvent": "EA",
        "loading_solvent_volume": "100", "PE/EA": "20/1", "flow": "10",
    }
    return {
        "id": key, "molecule_id": "M-0001" if same else "M-0002",
        "smiles": "CCO" if same else "CC=O", "conditions": {**conditions, **overrides},
    }


def choice(key, *, references=None, reason="Clarify the predictor's local response."):
    return {"candidate_id": key, "molecule_id": "M-0001", "condition_anchor": {"PE/EA": "20/1"},
            "reason": reason, "hypothesis_id": None, "references": references or []}


def reference(key, relationship="observed_evidence", kind="observed", **fields):
    return {"reference_id": key, "reference_kind": kind, "relationship": relationship, **fields}


def update(*, operation="create", version=None, claim="The response varies with loading volume.",
           support=None, contradiction=None, status="proposed"):
    return {"hypothesis_id": "H-loading", "operation": operation, "expected_version": version,
            "claim": claim, "revision_reason": "Update the evidence assessment.", "basis": "mixed",
            "supporting_observed_ids": support if support is not None else ["O-0041"],
            "contradicting_observed_ids": contradiction if contradiction is not None else ["O-0042"],
            "status": status, "confidence": "low", "unresolved_question": "Does it transfer?",
            "relevance_to_model_learning": "It may explain recurring local error."}


@pytest.fixture
def case():
    candidates = {f"C-{i:04}": card(f"C-{i:04}") for i in range(1, 37)}
    candidates["C-0002"]["conditions"]["flow"] = "5"
    candidates["C-0003"]["conditions"]["purity_raw"] = "99%"
    candidates["C-0004"]["conditions"]["sample_volume"] = "7"
    observed = {"O-0041": card("O-0041"), "O-0042": card("O-0042", same=False)}
    for row in observed.values():
        row.update(V1_ml=1.2, V2_ml=4.0, record_kind="initial_L333")
    all_rows = {f"{role}-{i:04}": f"sample-{i:04}"
                for i in [*range(1, 37), 40, 41, 42] for role in "CO"}
    plan = {"batch_strategy": "Improve transferable learning.", "batch_review": "Checked the joint plan.",
            "feedback_interpretation": "", "hypothesis_updates": [],
            "choices": [choice(key) for key in list(candidates)[:32]]}
    validator = partial(validate_plan, candidates=candidates, observed=observed, all_row_ids=all_rows)
    return plan, validator, candidates, observed, all_rows


def codes(result):
    return {issue["code"] for issue in result.errors}


def patch_template(request):
    return {
        "mode": "patch", "repair_id": request["repair_id"], "reason": "Preserve the stated scientific aim.",
        "replace_choices": [], "replace_hypothesis_updates": [], "append_choices": [], "replace_fields": {},
    }


def test_complete_json_boundary():
    assert parse_response(' ```json\n{"x": 1}\n``` ') == {"x": 1}
    for response in ('{}{}', '{"x": 2, "x": 1}', '{"x": 1e9999}', '{"x": NaN}', '{"x":',
                     'prefix {"x": 1}', 'prefix\n```json\n{}\n```'):
        with pytest.raises(ResponseFormatError):
            parse_response(response)


def test_all_errors_and_role_diagnostics(case):
    plan, validate, *_ = case
    assert validate(plan).valid
    plan["choices"][0]["candidate_id"] = "O-0041"
    plan["choices"][1]["candidate_id"] = "C-0041"
    plan["choices"][2]["candidate_id"] = "C-0040"
    plan["choices"][3]["candidate_id"] = "C-3908"
    plan["choices"][4]["candidate_id"] = plan["choices"][5]["candidate_id"]
    plan["choices"][6]["molecule_id"] = "M-0002"
    plan["choices"][7]["reason"] = "The identity C-0003 has this behavior."
    plan["choices"].pop()
    result = validate(plan)
    assert codes(result) == {
        "already_observed", "not_in_current_pool", "unknown_candidate_id", "duplicate_candidate_id",
        "molecule_id_mismatch", "unbound_reason_identity", "selection_count",
    }
    assert sum(x["code"] == "already_observed" for x in result.errors) == 2
    assert "/choices/3/candidate_id" in {x["path"] for x in result.errors}
    assert result.valid_choice_count == 24


def test_objective_anchors_full_conditions_and_predictions(case):
    plan, validate, candidates, *_ = case
    # Higher PE/EA gives less EA; raw anchors must retain input precision.
    assert ea_fraction("20/1") < ea_fraction("5/1")
    plan["choices"][0]["condition_anchor"]["density"] = "1.000"
    plan["choices"][0]["references"] = [reference("O-0041", "replicate")]
    plan["choices"][1]["references"] = [reference("O-0041", "single_factor_contrast",
                                                   changed_axis="sample_volume")]
    plan["choices"][2]["references"] = [reference("O-0041", "replicate")]
    plan["choices"][3]["references"] = [reference("O-0041", "single_factor_contrast",
                                                   changed_axis="loading_solvent_volume")]
    plan["choices"][4]["references"] = [reference("O-0042", "single_factor_contrast", changed_axis="flow")]
    plan["choices"][5]["references"] = [reference("C-0007", kind="candidate")]
    plan["choices"][6]["references"] = [reference("C-0008", "prediction_comparison", kind="candidate")]
    errors = validate(plan).errors
    assert {"condition_anchor_mismatch", "single_factor_mismatch", "different_molecules",
            "replicate_condition_mismatch", "evidence_requires_observation"} <= {x["code"] for x in errors}
    assert ["flow"] in [x.get("actual_changed_axes") for x in errors]
    assert ["purity_raw"] in [x.get("actual_changed_axes") for x in errors]
    assert ["sample_volume"] in [x.get("actual_changed_axes") for x in errors]
    assert not any(x["path"].startswith("/choices/6/references") for x in errors)
    plan["choices"][1]["references"][0]["changed_axis"] = "flow"
    plan["choices"][3]["references"][0]["changed_axis"] = "sample_volume"
    assert not any(x["path"].startswith(("/choices/1/references", "/choices/3/references"))
                   for x in validate(plan).errors)
    candidates["C-0002"]["conditions"]["purity_raw"] = ""
    assert "condition_unrecorded" in codes(validate(plan))


def test_packet_and_long_id_compatibility(case):
    plan, validate, *_ = case
    plan["packet_sha256"] = CHECKPOINT
    assert validate(plan, legacy_packet_sha256=CHECKPOINT).valid
    assert "legacy_packet_sha256_mismatch" in codes(validate(plan, legacy_packet_sha256="b" * 64))
    assert "unexpected_field" in codes(validate(plan))


@pytest.mark.parametrize("bad_slots", [[7], [7, 19]])
def test_local_repair_locks_all_other_choices_and_uses_authored_slots(case, bad_slots):
    plan, validate, *_ = case
    plan["choices"][7]["candidate_id"] = "O-0041"
    if len(bad_slots) == 2:
        plan["choices"][19]["candidate_id"] = "C-3908"
    plan["choices"][7]["external_cw"] = 99999
    binding = {"packet_sha256": CHECKPOINT, "seed": 157, "round": 5}
    request = build_repair_request(plan, validation=validate(plan), binding=binding)
    assert request["replace_choice_slots"] == bad_slots
    assert len(request["locked_choice_slots"]) == 32 - len(bad_slots)
    assert "external_cw" not in dumps(request)
    assert "99999" not in dumps(request)
    patch = patch_template(request)
    patch["replace_choices"] = [{"slot": i, "choice": choice(f"C-{33+j:04}", reason="New authored reason.")}
                                for j, i in enumerate(bad_slots)]
    with pytest.raises(ValueError, match="original proposal or state"):
        apply_repair(plan, request, patch, binding={**binding, "seed": 887}, validator=validate)
    tampered = deepcopy(request)
    tampered["replace_choice_slots"].append(0)
    with pytest.raises(ValueError, match="original proposal or state"):
        apply_repair(plan, tampered, patch, binding=binding, validator=validate)
    unauthorized = deepcopy(patch)
    unauthorized["replace_choices"].append({"slot": 0, "choice": choice("C-0035")})
    with pytest.raises(ValueError, match="unapproved"):
        apply_repair(plan, request, unauthorized, binding=binding, validator=validate)
    bad_prediction = deepcopy(patch)
    bad_prediction["replace_choices"][0]["choice"]["candidate_id"] = "C-3908"
    with pytest.raises(ValueError, match="failed full validation"):
        apply_repair(plan, request, bad_prediction, binding=binding, validator=validate)
    corrected = apply_repair(plan, request, patch, binding=binding, validator=validate)
    assert validate(corrected).valid
    assert all(original == fixed for i, (original, fixed) in enumerate(zip(plan["choices"], corrected["choices"]))
               if i not in bad_slots)


def test_underfill_overfill_and_explicit_replanning(case):
    plan, validate, *_ = case
    plan["choices"].pop()
    binding = {"checkpoint": CHECKPOINT}
    request = build_repair_request(plan, validation=validate(plan), binding=binding)
    assert request["append_choice_count"] == 1
    with pytest.raises(ReplanRequested, match="coupled controls"):
        apply_repair(plan, request, {"mode": "replan", "repair_id": request["repair_id"],
                                    "reason": "The coupled controls require replanning."},
                     binding=binding, validator=validate)
    patch = patch_template(request)
    patch["append_choices"] = [choice("C-0032")]
    fixed = apply_repair(plan, request, patch, binding=binding, validator=validate)
    assert len(fixed["choices"]) == 32
    fixed["choices"].append(choice("C-0033"))
    assert build_repair_request(fixed, validation=validate(fixed), binding=binding)["mode"] == "full_replan"


def test_repair_preserves_choice_and_hypothesis_evidence(case):
    plan, validate, *_ = case
    plan["choices"][0]["references"] = [reference("O-0041", "replicate"), reference("C-0033")]
    plan["hypothesis_updates"] = [update(contradiction=["O-0042", "C-0033"])]
    plan["choices"][1]["hypothesis_id"] = "H-loading"
    binding = {"checkpoint": CHECKPOINT}
    request = build_repair_request(plan, validation=validate(plan), binding=binding)
    assert request["replace_choice_slots"] == [0]
    assert request["replace_hypothesis_update_slots"] == [0]
    patch = patch_template(request)
    patch["replace_choices"] = [{"slot": 0, "choice": choice("C-0001")}]
    patch["replace_hypothesis_updates"] = [{"slot": 0, "update": update()}]
    with pytest.raises(ValueError, match="dropped valid structured choice evidence"):
        apply_repair(plan, request, patch, binding=binding, validator=validate)
    patch["replace_choices"][0]["choice"]["references"] = [reference("O-0041", "replicate")]
    patch["replace_hypothesis_updates"][0]["update"]["contradicting_observed_ids"] = []
    with pytest.raises(ValueError, match="discarded valid hypothesis evidence"):
        apply_repair(plan, request, patch, binding=binding, validator=validate)
    patch["replace_hypothesis_updates"][0]["update"]["contradicting_observed_ids"] = ["O-0042"]
    fixed = apply_repair(plan, request, patch, binding=binding, validator=validate)
    assert fixed["choices"][1] == plan["choices"][1]


def test_memory_versioning_proposition_lock_and_contradiction_retention(case):
    plan, validate, candidates, observed, aliases = case
    plan["hypothesis_updates"] = [update()]
    memory = MemoryManager(157)
    memory.register_observed(observed=observed, all_row_ids=aliases)

    def record(round_number, current_plan):
        memory.record_round(round_number, current_plan, observed=observed, all_row_ids=aliases,
                            checkpoint_sha256=CHECKPOINT, candidates=candidates)

    def hypotheses():
        return memory.hypothesis_view(observed=observed, all_row_ids=aliases)

    assert validate(plan, hypotheses=hypotheses()).valid
    record(0, plan)
    assert memory.hypotheses["H-loading"]["supporting_observed_ids"] == ["sample-0041"]
    assert hypotheses()["H-loading"]["contradicting_observed_ids"] == ["O-0042"]
    record(0, plan)
    incoming = deepcopy(plan)
    incoming["hypothesis_updates"] = [update(operation="revise", version=1, claim="The response is constant.")]
    assert "changed_hypothesis_proposition" in codes(validate(incoming, hypotheses=hypotheses()))
    incoming["hypothesis_updates"] = [update(operation="revise", version=1, contradiction=[], status="rejected")]
    assert "missing_retained_evidence" in codes(validate(incoming, hypotheses=hypotheses()))
    incoming["hypothesis_updates"] = [update(operation="archive", version=1, status="rejected")]
    incoming["choices"] = [choice("C-0033")]
    record(1, incoming)
    view = memory.prompt_view("M2", observed=observed, all_row_ids=aliases)
    assert view["hypotheses"] == []
    assert view["archived_hypotheses"][0]["version"] == 2
    incoming["hypothesis_updates"] = [update(operation="reactivate", version=2, status="unresolved")]
    incoming["choices"] = [choice("C-0034")]
    record(2, incoming)
    assert hypotheses()["H-loading"]["version"] == 3
    original = memory.snapshot()
    assert MemoryManager.from_snapshot(original, seed=157).snapshot() == original
    with pytest.raises(ValueError, match="snapshot binding mismatch"):
        MemoryManager.from_snapshot(original, seed=887)
    original["hypotheses"]["H-loading"][0]["confidence"] = "high"
    with pytest.raises(ValueError, match="snapshot binding mismatch"):
        MemoryManager.from_snapshot(original, seed=157)
    original["hypotheses"]["H-loading"][0]["current_claim"] = {"external_cw": 1}
    original["snapshot_sha256"] = digest({k: v for k, v in original.items() if k != "snapshot_sha256"})
    with pytest.raises(ValueError, match="invalid memory snapshot hypothesis"):
        MemoryManager.from_snapshot(original, seed=157)


def test_immutable_fact_layer_and_memory_ablation(case):
    _, _, _, observed, aliases = case
    extra = card("O-0040")
    extra.update(V1_ml=2.0, V2_ml=4.0, acquisition_round=0,
                 record_kind="acquired_with_premeasurement_prediction",
                 premeasurement_prediction_V1_ml=1.5, premeasurement_prediction_V2_ml=6.0,
                 premeasurement_error_V1_ml=-0.5, premeasurement_error_V2_ml=2.0)
    observed["O-0040"] = extra
    facts = {"sample-0040": {"choice_reason": "Recorded prior rationale.", "hypothesis_id": None,
                             "acquisition_checkpoint_sha256": CHECKPOINT, "acquisition_round": 0}}
    memory = MemoryManager(157)
    memory.register_observed(observed=observed, all_row_ids=aliases, facts=facts)
    views = {mode: memory.prompt_view(mode, observed=observed, all_row_ids=aliases)
             for mode in ("M0", "M1", "M2")}
    assert all(view["latest_feedback"][0]["premeasurement_error_V1_ml"] == -0.5 for view in views.values())
    assert "historical_top_abs_errors" not in views["M2"]
    assert views["M1"]["historical_top_abs_errors"] == views["M0"]["historical_top_abs_errors"]
    assert "Recorded prior rationale." not in dumps(views)
    assert memory.snapshot()["facts"]["sample-0040"]["acquisition_checkpoint_sha256"] == CHECKPOINT
    with pytest.raises(ValueError, match="unexpected observation fact field"):
        memory.register_observed(observed=observed, all_row_ids=aliases,
                                 facts={"sample-0040": {"cw_percentile": 1}})
    observed["O-0040"]["V1_ml"] = 12.0
    with pytest.raises(ValueError, match="immutable fact differs"):
        memory.register_observed(observed=observed, all_row_ids=aliases, facts=facts)


def test_scientific_terms_in_model_prose_are_not_external_data(case):
    plan, validate, *_ = case
    plan["choices"][0]["reason"] = (
        "A BALD or LCMD intuition is a conceptual analogy. Check the C-2 substitution."
    )
    assert validate(plan).valid


def test_hidden_memory_ids_cannot_be_reassigned_in_m0(case):
    plan, validate, _, observed, aliases = case
    plan["hypothesis_updates"] = [update()]
    memory = MemoryManager(157)
    memory.record_round(0, plan, observed=observed, all_row_ids=aliases, checkpoint_sha256=CHECKPOINT)
    hidden = memory.hypothesis_view(observed=observed, all_row_ids=aliases)
    result = validate(plan, hypotheses=hidden, visible_hypothesis_ids=set())
    assert {"hypothesis_id_already_allocated", "hypothesis_not_in_memory_view"} <= codes(result)
    plan["hypothesis_updates"] = []
    plan["choices"][0]["hypothesis_id"] = "H-loading"
    assert "unknown_hypothesis_reference" in codes(validate(plan, hypotheses=hidden, visible_hypothesis_ids=set()))

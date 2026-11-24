"""Aggregate objective diagnostics against the exact pre-selection L/U state."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .metadata_view import CONDITION_AXES, condition_changes
from .response_schema import (
    BASIS, CHOICE_FIELDS, CONFIDENCE, EVIDENCE_FIELDS, ID_PATTERN, KINDS,
    OPERATIONS, PLAN_FIELDS, REFERENCE_FIELDS, RELATIONSHIPS, STATUS, UPDATE_FIELDS,
)

_ROLE_REFERENCE = re.compile(r"(?<![\w-])(?:[CO]-[0-9]{4}|M-[0-9]{3,4}|[a-f0-9]{20})(?![\w-])")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _coordinate_id(value):
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else None


def _text(value, *, empty=False):
    return isinstance(value, str) and (empty or bool(value.strip()))


def _same_condition(value, actual):
    # Raw strings are exact; derived numeric fields may serialize 1 or 1.0.
    if type(value) in (int, float) and type(actual) in (int, float):
        return (type(value) is int or math.isfinite(value)) and value == actual
    return type(value) is type(actual) and value == actual


@dataclass
class ValidationResult:
    errors: list[dict] = field(default_factory=list)
    legal_choice_indices: list[int] = field(default_factory=list)
    legal_candidate_indices: list[int] = field(default_factory=list)
    valid_reference_indices: dict[int, list[int]] = field(default_factory=dict)
    retained_hypothesis_evidence: dict[int, list[str]] = field(default_factory=dict)
    expected_count: int = 32

    @property
    def valid(self):
        return not self.errors

    @property
    def valid_choice_count(self):
        return len(self.legal_choice_indices)

    def as_dict(self):
        return {
            "valid": self.valid, "valid_choice_count": self.valid_choice_count,
            "expected_count": self.expected_count, "errors": [dict(issue) for issue in self.errors],
        }


def _check_fields(value, allowed, path, issue, *, required=None):
    if not isinstance(value, dict):
        issue("expected_object", path)
        return False
    required = allowed if required is None else required
    for name in sorted(required - value.keys()):
        issue("missing_field", f"{path}/{name}")
    unknown = set(value) - allowed
    if unknown:
        # Unknown keys may be stray output or provenance-sensitive numbers/text.
        issue("unexpected_field", path, unexpected_field_count=len(unknown))
    return True


def _check_update(update, path, issue, observed, prior, round_ids):
    if not _check_fields(update, UPDATE_FIELDS, path, issue):
        return
    identity = update.get("hypothesis_id")
    if not isinstance(identity, str) or not re.fullmatch(ID_PATTERN, identity):
        issue("invalid_hypothesis_id", f"{path}/hypothesis_id")
        identity = None
    previous = prior.get(identity)
    if identity in round_ids:
        issue("duplicate_hypothesis_update", f"{path}/hypothesis_id", hypothesis_id=identity)
    elif identity is not None:
        round_ids.add(identity)
    for name, valid in (("operation", OPERATIONS), ("basis", BASIS),
                        ("confidence", CONFIDENCE), ("status", STATUS)):
        if update.get(name) not in valid:
            issue("invalid_enum", f"{path}/{name}", allowed=list(valid))
    for name in ("claim", "revision_reason", "unresolved_question", "relevance_to_model_learning"):
        if not _text(update.get(name), empty=name == "unresolved_question"):
            issue("invalid_text", f"{path}/{name}")
    operation, version = update.get("operation"), update.get("expected_version")
    if operation == "create":
        if previous is not None:
            issue("hypothesis_id_already_allocated", f"{path}/hypothesis_id", hypothesis_id=identity)
        if version is not None:
            issue("expected_null_version", f"{path}/expected_version")
    elif operation in OPERATIONS:
        if previous is None:
            issue("unknown_hypothesis", f"{path}/hypothesis_id", hypothesis_id=identity)
        else:
            if type(version) is not int or version != previous["version"]:
                issue("stale_hypothesis_version", f"{path}/expected_version",
                      expected_version=previous["version"])
            if bool(previous["archived"]) != (operation == "reactivate"):
                issue("invalid_archive_transition", f"{path}/operation")
            if update.get("claim") != previous["current_claim"]:
                # A new or reversed proposition must get a new stable identity.
                issue("changed_hypothesis_proposition", f"{path}/claim")
    seen = {}
    for name in EVIDENCE_FIELDS:
        refs = update.get(name)
        if not isinstance(refs, list):
            issue("expected_array", f"{path}/{name}")
            continue
        for index, identity in enumerate(refs):
            pointer = f"{path}/{name}/{index}"
            if not isinstance(identity, str) or identity not in observed:
                issue("unobserved_evidence", pointer, reference_id=_coordinate_id(identity))
            elif identity in seen:
                code = "duplicate_evidence" if seen[identity] == name else "conflicting_evidence_roles"
                issue(code, pointer, reference_id=identity)
            else:
                seen[identity] = name
    if previous is not None:
        previous_evidence = {item for name in EVIDENCE_FIELDS for item in previous.get(name, [])}
        for identity in sorted((previous_evidence & observed.keys()) - seen.keys()):
            issue("missing_retained_evidence", path, reference_id=identity)
    return sorted(seen)


def _check_reference(reference, path, issue, candidate, candidates, observed, seen):
    allowed = REFERENCE_FIELDS | {"changed_axis"}
    if not _check_fields(reference, allowed, path, issue, required=REFERENCE_FIELDS):
        return
    identity, kind = reference.get("reference_id"), reference.get("reference_kind")
    relation, axis = reference.get("relationship"), reference.get("changed_axis")
    if kind not in KINDS:
        issue("invalid_reference_kind", f"{path}/reference_kind")
    if relation not in RELATIONSHIPS:
        issue("invalid_relationship", f"{path}/relationship")
    if relation == "single_factor_contrast":
        if axis not in CONDITION_AXES:
            issue("invalid_changed_axis", f"{path}/changed_axis", allowed=list(CONDITION_AXES))
    elif "changed_axis" in reference:
        issue("unexpected_changed_axis", f"{path}/changed_axis")
    target = None
    if isinstance(identity, str):
        target = (observed if kind == "observed" else candidates).get(identity)
    if kind == "observed" and target is None:
        issue("unobserved_evidence", f"{path}/reference_id", reference_id=_coordinate_id(identity))
    elif kind == "candidate" and target is None:
        issue("ineligible_prediction_reference", f"{path}/reference_id", reference_id=_coordinate_id(identity))
    if relation == "observed_evidence" and kind != "observed":
        issue("evidence_requires_observation", f"{path}/reference_kind")
    if candidate is None or target is None:
        return
    if target["id"] == candidate["id"]:
        issue("self_reference", f"{path}/reference_id", reference_id=identity)
    if isinstance(relation, str):
        key = (identity, relation)
        if key in seen:
            issue("duplicate_reference", f"{path}/reference_id", reference_id=identity)
        seen.add(key)

    same_structure = candidate["molecule_id"] == target["molecule_id"]
    same_structure_relations = ("single_factor_contrast", "condition_response",
                                "replicate", "multifactor_contrast")
    if relation in same_structure_relations and not same_structure:
        issue("different_molecules", f"{path}/relationship", reference_id=identity)
    if relation == "cross_molecule_analogy" and same_structure:
        issue("same_molecule", f"{path}/relationship", reference_id=identity)
    if relation not in same_structure_relations:
        return
    left, right = candidate["conditions"], target["conditions"]
    missing_axes = sorted(name for name in CONDITION_AXES
                          if left.get(name) in (None, "") or right.get(name) in (None, ""))
    comparable = tuple(name for name in CONDITION_AXES if name not in missing_axes)
    changed = list(condition_changes(left, right, axes=comparable))
    if relation in ("single_factor_contrast", "replicate") and missing_axes:
        issue("condition_unrecorded", f"{path}/relationship", missing_axes=missing_axes)
    if relation == "single_factor_contrast" and changed != [axis]:
        issue("single_factor_mismatch", f"{path}/changed_axis", actual_changed_axes=changed)
    if relation == "replicate" and changed:
        issue("replicate_condition_mismatch", f"{path}/relationship", actual_changed_axes=changed)
    if relation == "condition_response" and not changed:
        issue("condition_response_requires_change", f"{path}/relationship")
    if relation == "multifactor_contrast" and len(changed) < 2:
        issue("multifactor_mismatch", f"{path}/relationship", actual_changed_axes=changed)


def _check_choice(choice, path, issue, result, index, *, candidates, observed, all_rows, hypothesis_ids):
    if not _check_fields(choice, CHOICE_FIELDS, path, issue):
        return None
    identity = choice.get("candidate_id")
    candidate = candidates.get(identity) if isinstance(identity, str) else None
    if not isinstance(identity, str):
        issue("invalid_candidate_id", f"{path}/candidate_id", position=index)
    elif candidate is None:
        real = all_rows.get(identity)
        observed_real = {all_rows.get(item, item) for item in observed}
        if identity in observed or (real is not None and real in observed_real):
            code = "already_observed"
        elif identity in all_rows:
            code = "not_in_current_pool"
        else:
            code = "unknown_candidate_id"
        issue(code, f"{path}/candidate_id", position=index, candidate_id=_coordinate_id(identity))

    molecule = choice.get("molecule_id")
    if not _text(molecule):
        issue("invalid_molecule_id", f"{path}/molecule_id")
    elif candidate is not None and molecule != candidate["molecule_id"]:
        issue("molecule_id_mismatch", f"{path}/molecule_id", expected_molecule_id=candidate["molecule_id"])
    anchor = choice.get("condition_anchor")
    if not isinstance(anchor, dict) or not anchor:
        issue("invalid_condition_anchor", f"{path}/condition_anchor")
    else:
        for name, value in anchor.items():
            anchor_path = f"{path}/condition_anchor"
            if type(value) not in (str, float, int, type(None)):
                issue("invalid_anchor_value", anchor_path)
            if candidate is None:
                continue
            conditions = candidate["conditions"]
            if name not in conditions:
                issue("unknown_condition_axis", anchor_path)
            elif not _same_condition(value, conditions[name]):
                issue("condition_anchor_mismatch", f"{anchor_path}/{name}", expected_value=conditions[name])

    hypothesis_id = choice.get("hypothesis_id")
    if hypothesis_id is not None and (not isinstance(hypothesis_id, str) or hypothesis_id not in hypothesis_ids):
        issue("unknown_hypothesis_reference", f"{path}/hypothesis_id", hypothesis_id=_coordinate_id(hypothesis_id))
    refs = choice.get("references")
    verified = []
    if not isinstance(refs, list):
        issue("expected_array", f"{path}/references")
    else:
        seen = set()
        for slot, reference in enumerate(refs):
            before = len(result.errors)
            _check_reference(reference, f"{path}/references/{slot}", issue, candidate, candidates, observed, seen)
            if candidate is not None and len(result.errors) == before:
                verified.append(slot)
    result.valid_reference_indices[index] = verified

    reason = choice.get("reason")
    if not _text(reason):
        issue("invalid_text", f"{path}/reason")
    else:
        permitted = {item for item in (identity, molecule) if isinstance(item, str)}
        for slot in verified:
            target_id = refs[slot]["reference_id"]
            permitted.update((target_id, (observed.get(target_id) or candidates[target_id])["molecule_id"]))
        for mentioned in sorted(set(_ROLE_REFERENCE.findall(reason)) - permitted):
            issue("unbound_reason_identity", f"{path}/reason", reference_id=mentioned)
    return identity if candidate is not None else None


def validate_plan(plan, *, candidates, observed, all_row_ids=None, hypotheses=None,
                  visible_hypothesis_ids=None, selection_count=32, legacy_packet_sha256=None):
    """Validate exact schema, references and recorded conditions, with no scoring."""
    result = ValidationResult(expected_count=selection_count)

    def issue(code, path, **details):
        result.errors.append({"code": code, "path": path, **details})

    allowed = PLAN_FIELDS
    if legacy_packet_sha256 is not None:
        allowed = allowed | {"packet_sha256"}
    if not _check_fields(plan, allowed, "", issue):
        return result
    if legacy_packet_sha256 is not None and plan.get("packet_sha256") != legacy_packet_sha256:
        issue("legacy_packet_sha256_mismatch", "/packet_sha256")
    for name in ("batch_strategy", "batch_review", "feedback_interpretation"):
        if not _text(plan.get(name), empty=name == "feedback_interpretation"):
            issue("invalid_text", f"/{name}")

    prior = hypotheses or {}
    visible = set(prior) if visible_hypothesis_ids is None else set(visible_hypothesis_ids)
    knowledge = {identity for identity, hypothesis in prior.items()
                 if identity in visible and not hypothesis["archived"]}
    updates = plan.get("hypothesis_updates")
    if not isinstance(updates, list):
        issue("expected_array", "/hypothesis_updates")
    else:
        seen = set()
        for index, update in enumerate(updates):
            registered = set(seen)
            result.retained_hypothesis_evidence[index] = (
                _check_update(update, f"/hypothesis_updates/{index}", issue, observed, prior, seen) or []
            )
            if isinstance(update, dict) and update.get("operation") in ("create", "reactivate"):
                # A declaration can bind its dependent choices even while its
                # evidence or version needs local repair in this proposal.
                knowledge.update(identity for identity in seen - registered
                                 if identity not in prior or identity in visible)
            for identity in sorted((seen - registered) & prior.keys() - visible):
                issue("hypothesis_not_in_memory_view", f"/hypothesis_updates/{index}/hypothesis_id",
                      hypothesis_id=identity)
            # A plan can select experiments about a hypothesis it archives in the
            # same response. The reference binds to that version in the audit log.
    choices = plan.get("choices")
    if not isinstance(choices, list):
        issue("expected_array", "/choices")
        return result
    if len(choices) != selection_count:
        issue("selection_count", "/choices", expected=selection_count, actual=len(choices))

    all_rows = all_row_ids if all_row_ids is not None else {item: item for item in (*candidates, *observed)}
    candidates_by_identity = {}
    for index, choice in enumerate(choices):
        before = len(result.errors)
        identity = _check_choice(choice, f"/choices/{index}", issue, result, index,
                                 candidates=candidates, observed=observed,
                                 all_rows=all_rows, hypothesis_ids=knowledge)
        if identity is not None:
            candidates_by_identity.setdefault(identity, []).append((index, len(result.errors) == before))
    for identity, positions in candidates_by_identity.items():
        lock_index, fully_valid = next((item for item in positions if item[1]), positions[0])
        result.legal_candidate_indices.append(lock_index)
        if fully_valid:
            result.legal_choice_indices.append(lock_index)
        for index, _ in positions:
            if index != lock_index:
                issue("duplicate_candidate_id", f"/choices/{index}/candidate_id",
                      position=index, candidate_id=identity, retained_position=lock_index)
    result.legal_candidate_indices.sort()
    result.legal_choice_indices.sort()
    return result

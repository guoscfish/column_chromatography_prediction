"""Host-bound local proposal repair with explicit model-authored replacements."""
from __future__ import annotations

import copy
import math

from .metadata_view import PUBLIC_CONDITIONS
from .records import digest
from .response_schema import CHOICE_FIELDS, EVIDENCE_FIELDS, REFERENCE_FIELDS, UPDATE_FIELDS

_REPLACEABLE_FIELDS = {"batch_strategy", "batch_review", "feedback_interpretation"}
_CONDITION_FIELDS = {*PUBLIC_CONDITIONS, "Density g/ml", "V/ul", "loading solvent",
                     "Volume of loading solvent/ul"}
_IDENTITY_ERRORS = {"invalid_hypothesis_id", "duplicate_hypothesis_update",
                    "hypothesis_id_already_allocated", "unknown_hypothesis", "hypothesis_not_in_memory_view"}
_COMMON_PATCH = {"mode", "repair_id", "reason"}
_PATCH_FIELDS = _COMMON_PATCH | {"replace_choices", "append_choices",
                                "replace_hypothesis_updates", "replace_fields"}


class ReplanRequested(ValueError):
    """The model explicitly requests a separately budgeted full replan."""


def _scalar(value):
    return value is None or type(value) in (str, int) or (type(value) is float and math.isfinite(value))


def _project_fields(record, fields):
    if not isinstance(record, dict):
        return {}
    return {field: record[field] for field in sorted(fields & record.keys()) if _scalar(record[field])}


def _project_proposal(plan):
    """Keep only the typed response vocabulary in the outward repair message."""
    proposal = _project_fields(plan, _REPLACEABLE_FIELDS)
    proposal["choices"] = []
    for item in plan["choices"]:
        choice = _project_fields(item, CHOICE_FIELDS)
        if isinstance(item, dict):
            choice["condition_anchor"] = _project_fields(item.get("condition_anchor"), _CONDITION_FIELDS)
            if isinstance(item.get("references"), list):
                choice["references"] = [
                    _project_fields(reference, REFERENCE_FIELDS | {"changed_axis"})
                    for reference in item["references"]
                ]
        proposal["choices"].append(choice)
    proposal["hypothesis_updates"] = []
    for item in plan["hypothesis_updates"]:
        update = _project_fields(item, UPDATE_FIELDS)
        for field in EVIDENCE_FIELDS:
            if isinstance(item, dict) and isinstance(item.get(field), list):
                update[field] = [key for key in item[field] if isinstance(key, str)]
        proposal["hypothesis_updates"].append(update)
    return proposal


def build_repair_request(plan, *, validation, binding):
    """Preserve prototypes' locked slots and expand faulty slots to whole choices."""
    choices, updates, fields = set(), set(), set()
    unlocalized = False
    for issue in validation.errors:
        tokens = issue["path"].split("/")[1:]
        if not tokens:
            unlocalized = True
        elif len(tokens) >= 2 and tokens[0] == "choices":
            choices.add(int(tokens[1]))
        elif len(tokens) >= 2 and tokens[0] == "hypothesis_updates":
            updates.add(int(tokens[1]))
        elif len(tokens) == 1 and tokens[0] in _REPLACEABLE_FIELDS:
            fields.add(tokens[0])
        elif issue["code"] != "selection_count":
            unlocalized = True
    actual = len(plan["choices"]) if isinstance(plan, dict) and isinstance(plan.get("choices"), list) else 0
    structurally_patchable = (isinstance(plan, dict) and isinstance(plan.get("choices"), list)
                              and isinstance(plan.get("hypothesis_updates"), list))
    mode = "patch"
    if validation.valid:
        mode = "verified"
    elif (unlocalized or not structurally_patchable or actual > validation.expected_count
          or validation.valid_choice_count == 0):
        mode = "full_replan"
    contract = {
        "mode": mode,
        "protocol": "pure-llm-v3-local-repair-1",
        "binding_sha256": digest(binding),
        "proposal_sha256": digest(plan),
        "diagnosis": validation.as_dict(),
        "replace_choice_slots": sorted(choices) if mode == "patch" else [],
        "replace_hypothesis_update_slots": sorted(updates) if mode == "patch" else [],
        "replace_fields": sorted(fields) if mode == "patch" else [],
        "append_choice_count": max(0, validation.expected_count - actual) if mode == "patch" else 0,
        "locked_choice_slots": list(validation.legal_choice_indices) if mode == "patch" else [],
        "locked_candidate_id_slots": list(validation.legal_candidate_indices) if mode == "patch" else [],
    }
    if mode == "patch":
        contract["proposal"] = _project_proposal(plan)
    return {**contract, "repair_id": digest(contract)[:20]}


def _replacements(operations, allowed, name):
    if not isinstance(operations, list):
        raise ValueError(f"expected {name} array")
    result = {}
    for operation in operations:
        if not isinstance(operation, dict) or operation.keys() != {"slot", name}:
            raise ValueError("malformed replacement operation")
        slot = operation["slot"]
        if type(slot) is not int or slot not in allowed or slot in result:
            raise ValueError("unapproved or repeated repair slot")
        result[slot] = copy.deepcopy(operation[name])
    if result.keys() != set(allowed):
        raise ValueError("every authorized faulty slot requires an explicit replacement")
    return result


def apply_repair(plan, request, patch, *, binding, validator):
    """Check exact scope, preserve valid work, and fully revalidate the result."""
    validation = validator(plan)
    if request != build_repair_request(plan, validation=validation, binding=binding):
        raise ValueError("repair request differs from the host-bound original proposal or state")
    if request["mode"] != "patch":
        raise ValueError("no local replacement is authorized")
    if not isinstance(patch, dict) or patch.get("repair_id") != request["repair_id"]:
        raise ValueError("repair response belongs to a different proposal")
    reason = patch.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("repair requires an explicit scientific reason")
    if patch.get("mode") == "replan" and patch.keys() == _COMMON_PATCH:
        raise ReplanRequested(reason)
    if patch.get("mode") != "patch" or patch.keys() != _PATCH_FIELDS:
        raise ValueError("invalid repair response fields")

    result = copy.deepcopy(plan)
    choices = _replacements(patch["replace_choices"], request["replace_choice_slots"], "choice")
    for slot, replacement in choices.items():
        if not isinstance(replacement, dict):
            raise ValueError("replacement choice must be an object")
        original = plan["choices"][slot]
        if slot in request["locked_candidate_id_slots"]:
            if replacement.get("candidate_id") != original["candidate_id"]:
                raise ValueError("legal candidate identity must be preserved")
            retained = [original["references"][i] for i in validation.valid_reference_indices.get(slot, [])]
            proposed = replacement.get("references")
            if not isinstance(proposed, list) or any(item not in proposed for item in retained):
                raise ValueError("local repair dropped valid structured choice evidence")
        result["choices"][slot] = replacement
    appended = patch["append_choices"]
    if not isinstance(appended, list) or len(appended) != request["append_choice_count"]:
        raise ValueError("patch must supply exactly the missing experiments")
    result["choices"].extend(copy.deepcopy(appended))

    updates = _replacements(patch["replace_hypothesis_updates"], request["replace_hypothesis_update_slots"], "update")
    for slot, replacement in updates.items():
        if not isinstance(replacement, dict):
            raise ValueError("replacement hypothesis update must be an object")
        original = plan["hypothesis_updates"][slot]
        unstable = any(issue["code"] in _IDENTITY_ERRORS
                       and issue["path"] == f"/hypothesis_updates/{slot}/hypothesis_id"
                       for issue in validation.errors)
        if isinstance(original, dict) and not unstable and replacement.get("hypothesis_id") != original.get("hypothesis_id"):
            raise ValueError("local repair changed the bound hypothesis ID")
        retained = set(validation.retained_hypothesis_evidence.get(slot, []))
        proposed = {item for field in EVIDENCE_FIELDS
                    if isinstance(replacement.get(field), list)
                    for item in replacement[field] if isinstance(item, str)}
        if not retained <= proposed:
            raise ValueError("local repair discarded valid hypothesis evidence")
        result["hypothesis_updates"][slot] = replacement

    fields = patch["replace_fields"]
    if not isinstance(fields, dict) or fields.keys() != set(request["replace_fields"]):
        raise ValueError("unexpected or missing authorized plan field repair")
    result.update(copy.deepcopy(fields))
    final = validator(result)
    if not final.valid:
        raise ValueError("patched proposal failed full validation",
                         final.as_dict())
    return result

"""Versioned response vocabulary and an unambiguous local JSON boundary."""
from __future__ import annotations

import json
import math
import re

SCHEMA_VERSION = "pure-llm-v3-plan-1"
PLAN_FIELDS = frozenset({"batch_strategy", "choices", "hypothesis_updates",
                         "feedback_interpretation", "batch_review"})
CHOICE_FIELDS = frozenset({"candidate_id", "molecule_id", "condition_anchor",
                           "reason", "hypothesis_id", "references"})
REFERENCE_FIELDS = frozenset({"reference_id", "reference_kind", "relationship"})
UPDATE_FIELDS = frozenset({"hypothesis_id", "operation", "expected_version", "claim",
                           "revision_reason", "basis", "supporting_observed_ids",
                           "contradicting_observed_ids", "status", "confidence",
                           "unresolved_question", "relevance_to_model_learning"})
EVIDENCE_FIELDS = ("supporting_observed_ids", "contradicting_observed_ids")
BASIS = ("chemical_prior", "observed_data", "model_behavior", "mixed")
STATUS = ("proposed", "supported", "weakened", "rejected", "unresolved")
CONFIDENCE = ("low", "medium", "high")
OPERATIONS = ("create", "revise", "archive", "reactivate")
KINDS = ("observed", "candidate")
RELATIONSHIPS = ("single_factor_contrast", "cross_molecule_analogy", "condition_response",
                 "replicate", "multifactor_contrast", "prediction_comparison", "observed_evidence")
ID_PATTERN = r"H-[A-Za-z0-9][A-Za-z0-9_-]{0,47}"


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": sorted(required),
            "additionalProperties": False}


_string = {"type": "string"}
_nullable_string = {"type": ["null", "string"]}
_reference = _object({
    "reference_id": _string, "reference_kind": {"enum": list(KINDS)},
    "relationship": {"enum": list(RELATIONSHIPS)}, "changed_axis": _string,
}, REFERENCE_FIELDS)
_choice = _object({
    "candidate_id": _string,
    "molecule_id": _string,
    "condition_anchor": {"type": "object", "minProperties": 1,
                         "additionalProperties": {"type": ["string", "number", "null"]}},
    "reason": _string,
    "hypothesis_id": _nullable_string,
    "references": {"type": "array", "items": _reference},
}, CHOICE_FIELDS)
_update = _object({
    "hypothesis_id": {"type": "string", "pattern": "^" + ID_PATTERN + "$"},
    "operation": {"enum": list(OPERATIONS)},
    "expected_version": {"type": ["integer", "null"], "minimum": 1},
    "claim": _string,
    "revision_reason": _string,
    "basis": {"enum": list(BASIS)},
    **{name: {"type": "array", "items": _string, "uniqueItems": True} for name in EVIDENCE_FIELDS},
    "status": {"enum": list(STATUS)},
    "confidence": {"enum": list(CONFIDENCE)},
    "unresolved_question": _string,
    "relevance_to_model_learning": _string,
}, UPDATE_FIELDS)
PLAN_SCHEMA = _object({
    "batch_strategy": _string, "batch_review": _string, "feedback_interpretation": _string,
    "choices": {"type": "array", "items": _choice, "minItems": 32, "maxItems": 32},
    "hypothesis_updates": {"type": "array", "items": _update},
}, PLAN_FIELDS)


class ResponseFormatError(ValueError):
    """Text has no complete, unique JSON interpretation."""


def _unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ResponseFormatError("duplicate JSON object key")
        result[name] = value
    return result


def _invalid_constant(_value):
    raise ResponseFormatError("non-finite JSON number")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _invalid_constant(value)
    return result


def parse_response(text):
    """Adapt the 20261030 planner's single-value, whole-fence-only boundary."""
    if not isinstance(text, str):
        raise ResponseFormatError("expected textual response")
    fence = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*", text)
    content = fence.group(1) if fence else text
    try:
        return json.loads(content, object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant, parse_float=_finite_float)
    except (ValueError, RecursionError) as error:
        if isinstance(error, ResponseFormatError):
            raise
        raise ResponseFormatError("expected a single complete JSON value") from error

"""Project one explicitly authorized A7 reference correction onto the archived plan."""
from copy import deepcopy
import re

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError, parse_response

CHOICE_INDEX = 15
REFERENCE_INDEX = 0
CANDIDATE = "C-0249"
MOLECULE = "M-0073"
OBSERVATION = "O-2702"
EXPECTED_ALIAS_COUNT = 69
_REQUIRED_REFERENCE = {"reference_id": OBSERVATION, "reference_kind": "observed",
                       "relationship": "single_factor_contrast", "changed_axis": "flow"}
_REASON_FACT = "only flow changes from 5 to 10 mL/min"
_OTHER_AXES = re.compile(r"\b(?:solvent|volume|density|purity|reagent|ratio|temperature|concentration|sample)\b", re.I)


def _target(plan):
    choices = plan.get("choices") if type(plan) is dict else None
    if (type(choices) is not list or len(choices) != 32
            or any(type(choice) is not dict or not isinstance(choice.get("candidate_id"), str) for choice in choices)
            or len({choice["candidate_id"] for choice in choices}) != 32):
        raise ValueError("targeted repair requires the 32 distinct archived candidates")
    choice = choices[CHOICE_INDEX]
    if (choice.get("candidate_id"), choice.get("molecule_id")) != (CANDIDATE, MOLECULE):
        raise ValueError("the archived choice does not match the authorized repair")
    return choice


def normalize_original(text):
    original = parse_response(text)
    _target(original)
    normalized = deepcopy(original)
    removed = 0
    for choice in normalized["choices"]:
        references = choice.get("references")
        if type(references) is not list or any(type(reference) is not dict for reference in references):
            raise ValueError("the archived choice has no typed reference array")
        for reference in references:
            for alias, kind in (("observed_id", "observed"), ("candidate_id", "candidate")):
                if alias not in reference:
                    continue
                if (reference.get("reference_kind") != kind or not isinstance(reference.get("reference_id"), str)
                        or reference[alias] != reference["reference_id"]):
                    raise ValueError("a reference alias conflicts with the archived identity or role")
                del reference[alias]
                removed += 1
    choice = _target(normalized)
    expected = {"reference_id": OBSERVATION, "reference_kind": "observed", "relationship": "multifactor_contrast"}
    if removed != EXPECTED_ALIAS_COUNT or not choice["references"] or choice["references"][REFERENCE_INDEX] != expected:
        raise ValueError("the normalized historical plan differs from the audited repair scope")
    return normalized, removed


def repair_guide():
    return (
        "Scoped correction for the already chosen batch. This request asks for one patch only. "
        "Do not propose, change, or reorder any candidate. Use the supplied experiment table to verify candidate "
        f"{CANDIDATE} versus observed reference {OBSERVATION} of {MOLECULE}. "
        "Their recorded contrast differs only in flow, from 5 mL/min in the observed row to 10 mL/min in the candidate. "
        "The reason should state this comparison and its learning value in one concise sentence. Include the exact phrase "
        f'"{_REASON_FACT}" and cite {OBSERVATION}. Mention no other experimental axis or unobserved outcome. '
        "Return one JSON object with exactly reference and reason. The reference object must have exactly "
        f'reference_id="{OBSERVATION}", reference_kind="observed", relationship="single_factor_contrast", '
        'and changed_axis="flow". The reason is your sentence. This patch format replaces the full-plan output '
        "format for this request only. Do not return a plan, explanations outside JSON, or raw row data."
    )


def project_reply(patch, *, plan):
    if type(patch) is not dict or set(patch) != {"reference", "reason"} or patch["reference"] != _REQUIRED_REFERENCE:
        raise ResponseFormatError("the reply exceeds the authorized reference patch scope")
    reason = patch["reason"]
    if (not isinstance(reason, str) or not 15 <= len(reason) <= 300 or "\n" in reason
            or _REASON_FACT.casefold() not in reason.casefold() or OBSERVATION not in reason
            or _OTHER_AXES.search(reason)):
        raise ResponseFormatError("the reason must describe the approved flow-only comparison")
    full = deepcopy(plan)
    choice = _target(full)
    choice["references"][REFERENCE_INDEX] = deepcopy(patch["reference"])
    choice["reason"] = reason
    return full

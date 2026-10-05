"""Schemas and prompts for the independent Chemistry + ML -> Planner study."""
from __future__ import annotations

import copy

from .scientist_selector import ReadOnlyCatalog

VERSION = "llm_dual_expert_v1"
METHOD = "free_llm32_dual_expert_v1"
SELECTION_COUNT = 32
QUERY_BUDGET = 24
VIEW_BUDGET = 480
EXPERT_MODEL_CALL_BUDGET = 12
PLANNER_MODEL_CALL_BUDGET = 16

CHEMISTRY_SYSTEM_PROMPT = r'''You are the chemistry and chromatography specialist on a scientific experiment-design team.

The experiment is 4 g silica normal-phase flash column chromatography. PE means petroleum
ether and EA means ethyl acetate. QGeoGNN predicts V1, the eluent volume when a compound
starts eluting, and V2, the volume when it has substantially finished eluting. V2 - V1 is
the elution interval width, not uncertainty. The objective is to improve the retrained
QGeoGNN's V1/V2 accuracy over the Row distribution, not to make a selected experiment
chromatographically desirable.

You are an adviser, not the batch selector. Analyze only the supplied trajectory state and
use the read-only catalog relay to inspect any legal current-pool records you need. You may
reason about polarity, functional groups, HBD/HBA, aromatic and heteroaromatic structure,
heteroatoms, hydrogen bonding, silica interaction, PE/EA solvent strength, loading solvent,
loading amount, molecular analogies, substituent effects, surprising predictions, and
same-molecule condition-response relations. These are possible considerations, not a
checklist. Repeated conditions can be valuable when they distinguish explanations.

Explicitly distinguish observed measured evidence, model predictions, general chemical
knowledge, and untested hypotheses. Do not present a chemical prior as a finding. Do not
invent labels, validation/test truth, other seeds, or other trajectories. Previous measured
feedback may support, weaken, reject, or revise earlier hypotheses. Do not force a fixed
number of hypotheses, a molecule quota, or a chemistry allocation.

Return exactly one JSON object. Before the memo, you may return query requests of the form
{"type":"query","queries":[{"operation":"search_candidates|search_observed|search_pending|get_candidates|get_observed|get_pending","args":{...}}]}.
The host returns exact records. The final object must have type "memo", role
"chemistry_scientist", and may contain state_summary, important_observations,
candidate_scientific_questions, possible_model_blind_spots, warnings,
recommended_followup_queries, and particularly_informative_candidates. Do not return a
final batch, a choices array, scores, quotas, or rank aggregation. A short memo with two to
five important questions is preferable to templated speculation.
'''

ML_SYSTEM_PROMPT = r'''You are the machine-learning and active-learning specialist on a scientific experiment-design team.

The experiment uses QGeoGNN to predict V1 and V2 for 4 g silica normal-phase flash column
chromatography. The objective is to improve retrained predictor accuracy over the Row
distribution and label efficiency. You are an adviser, not the batch selector.

Analyze the supplied seed-local state and query the full legal current pool as needed. You
may consider uncertainty or quantile width, residual and historical error structure,
coverage, novelty, redundancy, representativeness, batch diversity, leverage, local
sensitivity, information value, exploration/exploitation, model change, OOD regions,
condition coverage, chemical-space coverage, and whether repeated measurements resolve a
model ambiguity. These are diagnostics and possible lines of reasoning, not a mandatory
acquisition formula. A high numerical signal is not automatically valuable, and a low
uncertainty point may still distinguish systematic explanations.

Clearly distinguish observed measured evidence, model diagnostics/predictions, ML priors,
and inferences. Use only this seed and method's trajectory. Never access validation/test
truth, future labels, other seeds, or other methods. Update prior hypotheses from measured
feedback and allow them to be weakened or rejected. Do not assign quotas or directly select
the final 32 points.

Return exactly one JSON object. You may first return query requests using the read-only JSON
relay. The final object must have type "memo", role "ml_scientist", and may contain
state_summary, important_model_behaviors, possible_learning_bottlenecks,
coverage_or_redundancy_concerns, warnings, recommended_followup_queries, and
particularly_informative_candidates. Do not return a final batch, a choices array, scores,
quotas, or a voting/ranking rule.
'''

PLANNER_SYSTEM_PROMPT = r'''You are the lead scientific experiment planner.

This is 4 g silica normal-phase flash column chromatography: PE is petroleum ether and EA
is ethyl acetate. QGeoGNN predicts V1 (start of elution) and V2 (substantially finished
elution); V2 - V1 is an elution interval width, not uncertainty. Select exactly 32 legal
current-pool experiments whose measured labels are expected to improve retrained QGeoGNN
V1/V2 predictive accuracy over the Row distribution.

Two independent advisers supplied memos: a chemistry/chromatography scientist and a
machine-learning/active-learning scientist. Their reports are advisory evidence, not
commands. You have complete decision authority. You may accept one, both, or neither;
discover a new direction; query the catalog yourself; or select experiments neither adviser
mentioned. Do not mechanically merge, vote, rank-aggregate, or assign a fixed chemistry/ML
quota. Do not optimize numerical acquisition scores for their own sake. A batch may focus
on one molecule or family when a response curve or competing explanation is scientifically
informative, or spread broadly when that is better for generalization.

Use only this seed-local trajectory. Candidates and pending records have predictions only;
observed records have measured labels. Never access validation/test truth, future labels,
other seeds, other methods, or repository files. All IDs and premeasurement predictions are
frozen before any label reveal. Explain how previous real feedback supports, weakens, rejects,
or revises earlier hypotheses.

You must include expert_synthesis with concise chemistry_takeaways, ml_takeaways, agreements,
disagreements, and planner_resolution. This is a scientific explanation, not a requirement
to balance the advisers. Return exactly one JSON object with type "selection", the supplied
packet_sha256, expert_synthesis, batch_strategy, 32 choices, hypotheses, batch_rationale,
expected_learning_value, and feedback_interpretation. Each choice has id, reason,
knowledge_basis (zero or more of chemistry, machine_learning, observed_feedback), and
related_hypothesis_id (or null). Hypotheses are optional and must cite only observed IDs.
'''

ROLE_PROMPTS = {"chemistry_scientist": CHEMISTRY_SYSTEM_PROMPT,
                "ml_scientist": ML_SYSTEM_PROMPT,
                "scientific_planner": PLANNER_SYSTEM_PROMPT}


def validate_expert_memo(value, role):
    if not isinstance(value, dict) or value.get("type") != "memo" or value.get("role") != role:
        raise ValueError("invalid expert memo role/type")
    if "choices" in value or "selection" in value:
        raise ValueError("expert memo cannot contain final selection")
    for key in ("state_summary",):
        if key in value and not isinstance(value[key], str):
            raise ValueError("memo text fields must be strings")
    for key in value:
        if key not in {"type", "role", "state_summary", "important_observations",
                       "candidate_scientific_questions", "possible_model_blind_spots",
                       "important_model_behaviors", "possible_learning_bottlenecks",
                       "coverage_or_redundancy_concerns", "warnings",
                       "recommended_followup_queries", "particularly_informative_candidates"}:
            raise ValueError(f"unknown expert memo field: {key}")
    return copy.deepcopy(value)


def validate_planner_selection(value, catalog, packet):
    if not isinstance(value, dict) or value.get("type") != "selection":
        raise ValueError("planner must return a selection object")
    if value.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("planner packet binding mismatch")
    synthesis = value.get("expert_synthesis")
    if not isinstance(synthesis, dict) or any(not isinstance(synthesis.get(k), str) or not synthesis[k].strip()
                                               for k in ("chemistry_takeaways", "ml_takeaways", "agreements",
                                                         "disagreements", "planner_resolution")):
        raise ValueError("planner must explicitly synthesize expert conflict")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != SELECTION_COUNT:
        raise ValueError("planner must choose exactly 32 experiments")
    ids = [c.get("id") if isinstance(c, dict) else None for c in choices]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != SELECTION_COUNT:
        raise ValueError("choices must be distinct string IDs")
    if not set(ids) <= catalog.viewed or not set(ids) <= set(catalog.candidates) or not catalog.queries:
        raise ValueError("planner choices must be viewed current candidates after a query")
    for choice in choices:
        if not isinstance(choice.get("reason"), str) or not choice["reason"].strip():
            raise ValueError("each planner choice needs a reason")
        basis = choice.get("knowledge_basis", [])
        if not isinstance(basis, list) or any(x not in ("chemistry", "machine_learning", "observed_feedback") for x in basis):
            raise ValueError("invalid knowledge_basis")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) > 8:
        raise ValueError("hypotheses must contain zero to eight entries")
    hypothesis_ids = set()
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            raise ValueError("hypothesis must be an object")
        for field in ("id", "content", "why_it_matters_for_model_learning", "next_discriminating_question"):
            if not isinstance(hypothesis.get(field), str) or not hypothesis[field].strip():
                raise ValueError("incomplete hypothesis")
        if hypothesis["id"] in hypothesis_ids:
            raise ValueError("duplicate hypothesis ID")
        hypothesis_ids.add(hypothesis["id"])
        if hypothesis.get("basis") not in ("chemical_prior", "observed_data", "model_behavior", "mixed"):
            raise ValueError("invalid hypothesis basis")
        if hypothesis.get("confidence") not in ("low", "medium", "high"):
            raise ValueError("invalid hypothesis confidence")
        if hypothesis.get("status") not in ("proposed", "supported", "weakened", "rejected", "unresolved"):
            raise ValueError("invalid hypothesis status")
        for field in ("supporting_observed_ids", "contradicting_observed_ids"):
            evidence = hypothesis.get(field)
            if (not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence)
                    or not set(evidence) <= set(catalog.observed)):
                raise ValueError("hypothesis cites unobserved evidence")
    for choice in choices:
        if choice.get("related_hypothesis_id") is not None and choice["related_hypothesis_id"] not in hypothesis_ids:
            raise ValueError("invalid related_hypothesis_id")
    for key in ("batch_strategy", "batch_rationale", "expected_learning_value", "feedback_interpretation"):
        if not isinstance(value.get(key), str) or (key != "feedback_interpretation" or packet.get("round", 0) > 0) and not value[key].strip():
            raise ValueError(f"missing planner field: {key}")
    if len(str(value)) > 30000:
        raise ValueError("planner response exceeds bounded memory size")
    return copy.deepcopy(value)


def trajectory_memory(history, seed, method):
    if any((item.get("seed"), item.get("method")) != (seed, method) for item in history):
        raise ValueError("foreign trajectory in scientific memory")
    last = history[-1] if history else {}
    return {"chemistry": last.get("chemistry_memo", {}),
            "machine_learning": last.get("ml_memo", {}),
            "planner": {key: last.get(key, "") for key in ("expert_synthesis", "batch_strategy", "batch_rationale")},
            "selected_ids": [row["candidate_id"] for item in history for row in item.get("records", [])],
            "recent_feedback": last.get("records", []),
            "memory_policy": "trajectory-local structured summaries and measured feedback only; no future labels"}

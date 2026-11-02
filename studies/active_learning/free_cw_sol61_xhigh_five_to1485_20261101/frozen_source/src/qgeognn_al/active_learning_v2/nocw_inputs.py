"""Paired full-pool Scientist views. No model, label-store or network access here."""
from __future__ import annotations

import copy
import json
import math
import re

from .protocol import stable_hash
from .scientist_selector import (CONDITIONS, PREDICTIONS, OBSERVED_FIELDS,
                                 DESCRIPTORS, ReadOnlyCatalog, scientific_memory)
from .scientist_full_delivery import DIRECT_PROMPT, encode_catalog, validate_direct

VERSION = "llm32_nocw_paired_v1"
ARMS = {"llm32_cwaware_control": "cw_aware", "llm32_nocw": "no_cw"}
CW_FIELD = "cw_distance_percentile"
CW_DESCRIPTION = ("CW distance is a percentile\nof distance to labeled rows in the current "
                  "center/width gradient feature space.\n")
# Both arms get the same free-arm-only instructions. Only the CW definition differs.
COMMON_PROMPT = DIRECT_PROMPT.replace(
    "The required LLM count is packet.selection_count: 16 supplements to the pending 16 in\n"
    "the hybrid arm, or all 32 with no pending experiments in the free arm.",
    "Choose all 32 candidates freely; there are zero pending experiments.").replace(
    "Pending-CW outcomes are identified separately.\n", "")
COMMON_PROMPT += """
Only supporting_observed_ids and contradicting_observed_ids may cite measured evidence.
Use those typed lists for ALL measured-evidence citations, including evidence used in
choice reasons. Do not cite row IDs in free prose; reasons may refer to hypothesis_id.
No validation scores, learning-curve rewards or other trajectories are supplied.
"""


def prompt(variant):
    if variant not in ARMS.values():
        raise ValueError("unknown input variant")
    if CW_DESCRIPTION not in COMMON_PROMPT:
        raise RuntimeError("legacy CW prompt definition drift")
    return COMMON_PROMPT if variant == "cw_aware" else COMMON_PROMPT.replace(CW_DESCRIPTION, "")


# A second boundary in addition to the positive schema. Scan values AND keys and
# the actual serialized messages, including correction transcripts and memory.
CW_PATTERN = re.compile(
    r"\bcw\b|cw[_ -]?(?:distance|rank|score|coverage|percentile)|\blcmd\b|"
    r"(?:center[ /_-]*(?:and[ /_-]*)?width|gradient|geometric)[\s_-]*"
    r"(?:gradient[\s_-]*)?(?:distance|coverage|rank|percentile|score)|"
    r"梯度.{0,12}(?:距离|覆盖|排名|分位|评分)|中心.{0,6}宽度|"
    r"(?:distance|coverage|rank|percentile|score)[\s_-]*(?:in[\s_-]*)?(?:gradient|cw)", re.I)


def audit_no_cw(value):
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    match = CW_PATTERN.search(serialized)
    if match:
        raise ValueError(f"NoCW visible content rejected: {match.group(0)}")
    return {"passed": True, "serialized_sha256": stable_hash(value),
            "serialized_characters": len(serialized)}


BASE = {"id", "smiles", "conditions", "descriptors"}
PRED = {"pred_V1_ml", "pred_V2_ml", *PREDICTIONS}
FEEDBACK_FIELDS = {"candidate_id", "source", "pred_V1_ml", "pred_V2_ml", "true_V1_ml",
                   "true_V2_ml", "error_V1_ml", "error_V2_ml", "reason", "hypothesis_id"}
HYPOTHESIS_FIELDS = {"id", "content", "basis", "supporting_observed_ids", "contradicting_observed_ids",
                     "confidence", "status", "why_it_matters_for_model_learning", "next_discriminating_question"}
PLAN_FIELDS = {"type", "packet_sha256", "batch_strategy", "choices", "hypotheses",
               "feedback_interpretation", "batch_rationale"}
HISTORY_FIELDS = {"seed", "method", "round", "batch_freeze_sha256", "records", "hypotheses",
                  "feedback_interpretation", "batch_rationale"}


def _only(value, fields, kind):
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"unauthorized {kind} fields")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric field required")


def validate_card(row, observed):
    if not isinstance(row["id"], str) or not isinstance(row["smiles"], str):
        raise ValueError("string identity/SMILES required")
    conditions = row["conditions"]
    ratio = conditions["PE/EA"]
    if not isinstance(ratio, str) or not re.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", ratio):
        raise ValueError("invalid PE/EA ratio")
    if conditions["loading solvent"] not in ("PE", "EA", "DCM"):
        raise ValueError("unregistered loading solvent")
    for k in ("Density g/ml", "V/ul", "Volume of loading solvent/ul"):
        _finite(conditions[k])
    if "descriptors" in row:
        _only(row["descriptors"], set(DESCRIPTORS), "descriptors")
        for value in row["descriptors"].values():
            _finite(value)
    for k in (set(OBSERVED_FIELDS)-{"record_kind"} if observed else PRED | {CW_FIELD}):
        if k in row:
            _finite(row[k])
    if observed and (not {"V1_ml", "V2_ml", "record_kind"} <= set(row) or
            row["record_kind"] not in ("initial_L333", "acquired_with_premeasurement_prediction")):
        raise ValueError("measured record schema mismatch")


def validate_history(history, seed, arm, r):
    if len(history) != r:
        raise ValueError("incomplete trajectory history")
    for j, h in enumerate(history):
        _only(h, HISTORY_FIELDS, "history")
        if (h["seed"], h["method"], h["round"]) != (seed, arm, j):
            raise ValueError("foreign trajectory history")
        if len(h["records"]) != 32:
            raise ValueError("incomplete feedback batch")
        for record in h["records"]:
            _only(record, FEEDBACK_FIELDS, "feedback")
            if record["source"] != "LLM":
                raise ValueError("foreign selection source")
        for hypothesis in h["hypotheses"]:
            _only(hypothesis, HYPOTHESIS_FIELDS, "hypothesis")
    if ARMS[arm] == "no_cw":
        # Arm metadata is private, so do not scan its literal identifier.
        audit_no_cw([{k: v for k, v in h.items() if k != "method"} for h in history])


def input_view(raw, seed, arm, r, history):
    """Only removal is CW_FIELD; salt and every other value are shared exactly.

    IDs are the existing opaque sample IDs (no new method-dependent remapping).
    Unknown fields fail closed rather than allowing a renamed CW channel.
    """
    variant = ARMS[arm]
    validate_history(history, seed, arm, r)
    _only(raw, {"candidates", "observed", "pending", "salt"}, "catalog")
    if raw["pending"]:
        raise ValueError("free32 requires zero pending experiments")
    result = copy.deepcopy(raw)
    result["salt"] = f"{VERSION}:{seed}:paired:{r}"
    for name in ("candidates", "observed"):
        for row in result[name]:
            allowed = BASE | (set(OBSERVED_FIELDS) if name == "observed" else PRED | {CW_FIELD})
            _only(row, allowed, name)
            if name == "candidates" and not PRED | {CW_FIELD} <= set(row):
                raise ValueError("candidate prediction/CW schema incomplete")
            if set(row["conditions"]) != set(CONDITIONS):
                raise ValueError("condition schema mismatch")
            validate_card(row, name == "observed")
            if variant == "no_cw":
                row.pop(CW_FIELD, None)
    catalog = ReadOnlyCatalog(**result)
    memory = scientific_memory(history, seed, arm)
    memory["memory_policy"] = memory["memory_policy"].replace("older records queryable", "older records included in observed table")
    packet = {"study_version": VERSION, "seed": seed, "round": r,
              "active_label_count": len(catalog.observed), "selection_count": 32,
              "objective": "overall Row V1/V2 predictive accuracy after retraining",
              "memory": memory, "validation_test_records": 0,
              "delivery": "all_candidates_one_request", "candidate_count": len(catalog.candidates),
              "observed_count": len(catalog.observed), **encode_catalog(catalog)}
    # The arm and variant remain in the host contract, not the LLM packet.
    packet["packet_sha256"] = stable_hash(packet)
    if variant == "no_cw":
        audit_no_cw({"prompt": prompt(variant), "catalog": result, "packet": packet})
    return catalog, result, packet


def validate_plan(result, catalog, packet, variant):
    _only(result, PLAN_FIELDS, "plan")
    validate_direct(result, catalog, packet)
    for h in result["hypotheses"]:
        _only(h, HYPOTHESIS_FIELDS, "hypothesis")
    prose = [result[k] for k in ("batch_strategy", "batch_rationale", "feedback_interpretation")]
    for choice in result["choices"]:
        _only(choice, {"id", "reason", "hypothesis_id"}, "choice")
        prose.append(choice["reason"])
    for h in result["hypotheses"]:
        prose.extend(h[k] for k in ("content", "why_it_matters_for_model_learning", "next_discriminating_question"))
    # Prevent untyped evidence citations: typed lists are already checked against L_t.
    for row_id in catalog.references:
        if any(re.search(r"(?<![\w])" + re.escape(row_id) + r"(?![\w])", s) for s in prose):
            raise ValueError("row IDs in prose are forbidden; use typed observed evidence lists")
    if variant == "no_cw":
        audit_no_cw(result)
    return result


def pair_diff(aware, nocw):
    """Exact machine-readable structural diff, including ordering and precision."""
    expected = copy.deepcopy(aware)
    removed = []
    for name in ("candidates", "observed", "pending"):
        columns = expected[name]["columns"]
        if CW_FIELD in columns:
            i = columns.index(CW_FIELD)
            columns.pop(i)
            for row in expected[name]["rows"]:
                removed.append({"table": name, "id": row[0], "field": CW_FIELD, "value": row.pop(i)})
    expected.pop("packet_sha256")
    actual = {k: v for k, v in nocw.items() if k != "packet_sha256"}
    if expected != actual:
        raise ValueError("unregistered paired input difference")
    if prompt("cw_aware").replace(CW_DESCRIPTION, "") != prompt("no_cw"):
        raise ValueError("unregistered prompt difference")
    return {"passed": True, "candidate_ids_and_order_identical": True,
            "other_values_and_precision_identical": True, "removed": removed,
            "derived_difference": "packet_sha256", "prompt_removed": CW_DESCRIPTION,
            "aware_sha256": aware["packet_sha256"], "nocw_sha256": nocw["packet_sha256"]}

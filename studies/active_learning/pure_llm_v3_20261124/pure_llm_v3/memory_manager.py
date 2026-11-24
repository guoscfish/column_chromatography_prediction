"""Seed-local, versioned memory with immutable records and separate prompt views."""
from __future__ import annotations

import copy
import math
import re

from .records import digest
from .response_schema import BASIS, CONFIDENCE, EVIDENCE_FIELDS, ID_PATTERN, OPERATIONS, STATUS, UPDATE_FIELDS

MEMORY_VERSION = "pure-llm-v3-memory-1"
TARGETS = ("V1", "V2")
PREMEASUREMENT_FIELDS = tuple(
    f"premeasurement_{metric}_{target}_ml"
    for metric in ("prediction", "error") for target in TARGETS
)
MEASUREMENT_FIELDS = ("V1_ml", "V2_ml", "record_kind", *PREMEASUREMENT_FIELDS, "acquisition_round")
FROZEN_FACT_FIELDS = {"choice_reason", "hypothesis_id", "acquisition_checkpoint_sha256", "acquisition_round"}
PREDICTION_FIELDS = tuple(f"pred_{target}_{q}_ml" for target in TARGETS for q in ("q10", "q50", "q90"))
HYPOTHESIS_FIELDS = {
    "persistent_hypothesis_id", "version", "original_claim", "current_claim", "created_round",
    "last_updated_round", "basis", *EVIDENCE_FIELDS, "status", "confidence", "unresolved_question",
    "relevance_to_model_learning", "archived", "revision_reason",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HYPOTHESIS_ID = re.compile(ID_PATTERN)
_ROUND_FIELDS = {
    "round", "checkpoint_sha256", "plan_sha256", "selected", "hypothesis_updates",
    "batch_strategy", "feedback_interpretation", "batch_review",
}


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _visible_observations(observed, all_row_ids):
    aliases = {all_row_ids[key]: key for key in observed}
    if len(aliases) != len(observed):
        raise ValueError("ambiguous visible observation aliases")
    return aliases


def _project_evidence(record, aliases, *, compact=False):
    fields = HYPOTHESIS_FIELDS if compact else record.keys()
    result = {field: copy.deepcopy(record[field]) for field in fields}
    for field in EVIDENCE_FIELDS:
        result[field] = [aliases[item] for item in result[field]]
    return result


def _put_immutable(record, incoming):
    for field, value in incoming.items():
        if record.setdefault(field, copy.deepcopy(value)) != value:
            raise ValueError(f"immutable fact differs at {field}")


def _validate_metadata_fact(extra):
    if not isinstance(extra, dict) or set(extra) - FROZEN_FACT_FIELDS:
        raise ValueError("unexpected observation fact field")
    if "choice_reason" in extra and not isinstance(extra["choice_reason"], str):
        raise ValueError("invalid recorded choice reason")
    if "hypothesis_id" in extra and extra["hypothesis_id"] is not None and not isinstance(extra["hypothesis_id"], str):
        raise ValueError("invalid recorded hypothesis ID")
    if "acquisition_checkpoint_sha256" in extra:
        checkpoint = extra["acquisition_checkpoint_sha256"]
        if not isinstance(checkpoint, str) or not _SHA256.fullmatch(checkpoint):
            raise ValueError("invalid acquisition checkpoint digest")


def _evidence_is_valid(item, facts):
    if any(not isinstance(item.get(field), list) for field in EVIDENCE_FIELDS):
        return False
    rows = [key for field in EVIDENCE_FIELDS for key in item[field]]
    return all(isinstance(key, str) and key in facts for key in rows) and len(set(rows)) == len(rows)


def _scientific_fields_are_valid(item):
    return (item.get("basis") in BASIS and item.get("status") in STATUS and item.get("confidence") in CONFIDENCE
            and all(isinstance(item.get(field), str) for field in
                    ("revision_reason", "unresolved_question", "relevance_to_model_learning")))


def _check_snapshot_structure(state):
    facts, hypotheses, rounds = (state[field] for field in ("facts", "hypotheses", "rounds"))
    for sample_id, fact in facts.items():
        if (not isinstance(fact, dict)
                or fact.keys() - ({"sample_id", "canonical_smiles", "conditions"}
                                  | set(MEASUREMENT_FIELDS) | FROZEN_FACT_FIELDS)
                or fact.get("sample_id") != sample_id
                or not isinstance(fact.get("canonical_smiles"), str)
                or not isinstance(fact.get("conditions"), dict)
                or not all(_number(fact.get(field)) for field in ("V1_ml", "V2_ml"))
                or any(field in fact and not _number(fact[field]) for field in PREMEASUREMENT_FIELDS)
                or ("record_kind" in fact and not isinstance(fact["record_kind"], str))
                or (fact.get("acquisition_round") is not None
                    and (type(fact["acquisition_round"]) is not int or fact["acquisition_round"] < 0))):
            raise ValueError("invalid memory snapshot fact")
        _validate_metadata_fact({name: fact[name] for name in FROZEN_FACT_FIELDS if name in fact})
    for identity, history in hypotheses.items():
        if not isinstance(identity, str) or not _HYPOTHESIS_ID.fullmatch(identity) or not isinstance(history, list) or not history:
            raise ValueError("invalid memory snapshot hypothesis identity")
        first = None
        last_round = -1
        for version, hypothesis in enumerate(history, 1):
            if (not isinstance(hypothesis, dict) or hypothesis.keys() != HYPOTHESIS_FIELDS
                    or hypothesis["persistent_hypothesis_id"] != identity
                    or type(hypothesis["version"]) is not int or hypothesis["version"] != version
                    or type(hypothesis["archived"]) is not bool
                    or type(hypothesis["created_round"]) is not int
                    or type(hypothesis["last_updated_round"]) is not int
                    or hypothesis["last_updated_round"] <= last_round
                    or not isinstance(hypothesis["current_claim"], str)
                    or hypothesis["original_claim"] != hypothesis["current_claim"]
                    or not _scientific_fields_are_valid(hypothesis) or not _evidence_is_valid(hypothesis, facts)):
                raise ValueError("invalid memory snapshot hypothesis")
            if first is None:
                first = hypothesis
            if (hypothesis["created_round"] != first["last_updated_round"]
                    or hypothesis["current_claim"] != first["current_claim"]):
                raise ValueError("memory snapshot proposition identity drift")
            last_round = hypothesis["last_updated_round"]
    for key, record in rounds.items():
        if (not isinstance(record, dict) or record.keys() != _ROUND_FIELDS
                or type(record["round"]) is not int or record["round"] < 0 or str(record["round"]) != key
                or any(not isinstance(record[f], str) or not _SHA256.fullmatch(record[f])
                       for f in ("checkpoint_sha256", "plan_sha256"))
                or any(not isinstance(record[f], str)
                       for f in ("batch_strategy", "feedback_interpretation", "batch_review"))
                or not isinstance(record["selected"], list) or not isinstance(record["hypothesis_updates"], list)
                or any(not isinstance(row, dict) or not isinstance(row.get("sample_id"), str)
                       for row in record["selected"])):
            raise ValueError("invalid memory snapshot round")
        for update in record["hypothesis_updates"]:
            if (not isinstance(update, dict) or update.keys() != UPDATE_FIELDS
                    or not isinstance(update["hypothesis_id"], str)
                    or not _HYPOTHESIS_ID.fullmatch(update["hypothesis_id"])
                    or update["operation"] not in OPERATIONS or not isinstance(update["claim"], str)
                    or (update["expected_version"] is not None and type(update["expected_version"]) is not int)
                    or not _scientific_fields_are_valid(update) or not _evidence_is_valid(update, facts)):
                raise ValueError("invalid memory snapshot update")


class MemoryManager:
    """The caller supplies authenticated observed cards and validated plans."""

    def __init__(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self._state = {
            "memory_version": MEMORY_VERSION, "seed": seed,
            "facts": {}, "hypotheses": {}, "rounds": {},
        }

    @property
    def hypotheses(self):
        return copy.deepcopy({identity: history[-1] for identity, history in self._state["hypotheses"].items()})

    def hypothesis_view(self, *, observed, all_row_ids):
        aliases = _visible_observations(observed, all_row_ids)
        return {identity: _project_evidence(hypothesis, aliases, compact=True)
                for identity, hypothesis in self.hypotheses.items()}

    def snapshot(self):
        result = copy.deepcopy(self._state)
        return {**result, "snapshot_sha256": digest(result)}

    @classmethod
    def from_snapshot(cls, snapshot, *, seed):
        manager = cls(seed)
        if not isinstance(snapshot, dict) or snapshot.keys() != {*manager._state, "snapshot_sha256"}:
            raise ValueError("unexpected memory snapshot")
        result = {field: copy.deepcopy(snapshot[field]) for field in manager._state}
        if (type(result["seed"]) is not int or result["seed"] != seed or result["memory_version"] != MEMORY_VERSION
                or snapshot["snapshot_sha256"] != digest(result)):
            raise ValueError("memory snapshot binding mismatch")
        if any(not isinstance(result[field], dict) for field in ("facts", "hypotheses", "rounds")):
            raise ValueError("invalid memory snapshot records")
        _check_snapshot_structure(result)
        manager._state = result
        return manager

    def _collect_facts(self, observed, all_row_ids, facts):
        aliases = _visible_observations(observed, all_row_ids)
        new_facts = copy.deepcopy(self._state["facts"])
        if not set(new_facts) <= aliases.keys() or (facts is not None and not set(facts) <= aliases.keys()):
            raise ValueError("facts must belong to the same monotone observed pool")
        for sample_id, key in aliases.items():
            card = observed[key]
            if card["id"] != key:
                raise ValueError("observed card identity mismatch")
            measured = {field: card[field] for field in MEASUREMENT_FIELDS if field in card}
            if not all(_number(measured.get(field)) for field in ("V1_ml", "V2_ml")):
                raise ValueError("observed fact lacks finite measurements")
            extra = (facts or {}).get(sample_id, {})
            _validate_metadata_fact(extra)
            record = new_facts.setdefault(sample_id, {})
            _put_immutable(record, {"sample_id": sample_id, "canonical_smiles": card["smiles"]})
            _put_immutable(record.setdefault("conditions", {}), card["conditions"])
            _put_immutable(record, measured)
            _put_immutable(record, extra)
            acquired = "acquisition_round" in record and record["acquisition_round"] is not None
            if acquired and (type(record["acquisition_round"]) is not int or record["acquisition_round"] < 0
                             or not all(_number(record.get(field)) for field in PREMEASUREMENT_FIELDS)):
                raise ValueError("acquired observation lacks exact feedback lineage")
            if not acquired and any(field in record for field in PREMEASUREMENT_FIELDS):
                raise ValueError("premeasurement data lack acquisition round")
        return new_facts

    def register_observed(self, *, observed, all_row_ids, facts=None):
        """Ingest already-authorized observations, with no selector or label access."""
        self._state["facts"] = self._collect_facts(observed, all_row_ids, facts)

    def record_round(self, round_number, plan, *, observed, all_row_ids, checkpoint_sha256,
                     facts=None, candidates=None):
        """Persist the accepted scientific plan and version history for this round."""
        if type(round_number) is not int or round_number < 0:
            raise ValueError("invalid memory round")
        if not isinstance(checkpoint_sha256, str) or not _SHA256.fullmatch(checkpoint_sha256):
            raise ValueError("invalid memory checkpoint")
        new_facts = self._collect_facts(observed, all_row_ids, facts)
        selected = []
        for choice in plan["choices"]:
            key = choice["candidate_id"]
            sample_id = all_row_ids[key]
            if sample_id in new_facts:
                raise ValueError("accepted memory choice is already observed")
            if candidates is not None and candidates[key]["id"] != key:
                raise ValueError("accepted choice differs from candidate facts")
            record = {
                "sample_id": sample_id, "choice": copy.deepcopy(choice),
                "reference_sample_ids": [all_row_ids[item["reference_id"]] for item in choice["references"]],
            }
            if candidates is not None:
                card = candidates[key]
                record["premeasurement_predictions"] = {
                    name: card[name] for name in PREDICTION_FIELDS if name in card
                }
            selected.append(record)
        if len({item["sample_id"] for item in selected}) != len(selected):
            raise ValueError("repeated accepted memory choice")
        updates = copy.deepcopy(plan["hypothesis_updates"])
        for item in updates:
            for field in EVIDENCE_FIELDS:
                if not set(item[field]) <= observed.keys():
                    raise ValueError("unobserved memory evidence")
                item[field] = [all_row_ids[key] for key in item[field]]
        entry = {
            "round": round_number, "checkpoint_sha256": checkpoint_sha256, "plan_sha256": digest(plan),
            "selected": selected, "hypothesis_updates": updates, "batch_strategy": plan["batch_strategy"],
            "feedback_interpretation": plan["feedback_interpretation"], "batch_review": plan["batch_review"],
        }
        rounds = self._state["rounds"]
        original = rounds.get(str(round_number))
        if original is not None:
            if original != entry or new_facts != self._state["facts"]:
                raise ValueError("frozen scientific memory round differs on replay")
            return
        if rounds and round_number != max(map(int, rounds)) + 1:
            raise ValueError("scientific memory rounds must be consecutive after initialization")
        previously_selected = {item["sample_id"] for previous in rounds.values() for item in previous["selected"]}
        if {item["sample_id"] for item in selected} & previously_selected:
            raise ValueError("selection reuses a previous experiment")

        new_hypotheses = copy.deepcopy(self._state["hypotheses"])
        updated = set()
        for item in updates:
            identity = item["hypothesis_id"]
            if identity in updated:
                raise ValueError("duplicate hypothesis memory update")
            updated.add(identity)
            history = new_hypotheses.setdefault(identity, [])
            last = history[-1] if history else None
            operation = item["operation"]
            if last is None:
                if operation != "create" or item["expected_version"] is not None:
                    raise ValueError("invalid new hypothesis identity")
            elif (operation not in ("revise", "archive", "reactivate")
                  or type(item["expected_version"]) is not int or item["expected_version"] != last["version"]
                  or last["archived"] != (operation == "reactivate")
                  or last["current_claim"] != item["claim"]):
                raise ValueError("invalid hypothesis memory revision")
            previous_evidence = {value for field in EVIDENCE_FIELDS for value in (last or {}).get(field, [])}
            new_evidence = [value for field in EVIDENCE_FIELDS for value in item[field]]
            if len(set(new_evidence)) != len(new_evidence) or not previous_evidence <= set(new_evidence):
                raise ValueError("invalid hypothesis evidence history")
            history.append({
                "persistent_hypothesis_id": identity, "version": len(history) + 1,
                "original_claim": last["original_claim"] if last else item["claim"],
                "current_claim": item["claim"],
                "created_round": last["created_round"] if last else round_number,
                "last_updated_round": round_number, "archived": operation == "archive",
                **{field: item[field] for field in (*EVIDENCE_FIELDS, "basis", "status", "confidence",
                                                   "unresolved_question", "relevance_to_model_learning",
                                                   "revision_reason")},
            })
        self._state = {**self._state, "facts": new_facts, "hypotheses": new_hypotheses,
                       "rounds": {**rounds, str(round_number): entry}}

    def prompt_view(self, mode, *, observed, all_row_ids):
        """M0 uses the last update list; M1 persists claims; M2 removes Top-8 salience."""
        if mode not in ("M0", "M1", "M2"):
            raise ValueError("unknown scientific memory mode")
        aliases = _visible_observations(observed, all_row_ids)
        facts = self._state["facts"]
        if set(facts) != set(aliases):
            raise ValueError("register the exact current observations before projecting memory")
        acquired = [fact for fact in facts.values() if fact.get("acquisition_round") is not None]
        latest_round = max((fact["acquisition_round"] for fact in acquired), default=None)
        rounds = self._state["rounds"]
        last = rounds[str(max(map(int, rounds)))] if rounds else {}

        def feedback(fact):
            return {"observed_id": aliases[fact["sample_id"]],
                    **{field: fact[field] for field in MEASUREMENT_FIELDS if field in fact}}

        # Keep the supplied observation order for the latest measured records.
        result = {
            "mode": mode,
            "latest_feedback": [feedback(facts[key]) for key in aliases
                                if facts[key].get("acquisition_round") == latest_round and latest_round is not None],
            "previous_feedback_interpretation": last.get("feedback_interpretation", ""),
        }
        if mode == "M0":
            result["hypothesis_updates"] = [_project_evidence(update, aliases)
                                            for update in last.get("hypothesis_updates", [])]
        else:
            hypotheses = self.hypothesis_view(observed=observed, all_row_ids=all_row_ids)
            result["hypotheses"] = [state for state in hypotheses.values() if not state["archived"]]
            compact = ("persistent_hypothesis_id", "version", "current_claim", "status",
                       "last_updated_round", *EVIDENCE_FIELDS)
            result["archived_hypotheses"] = [
                {key: state[key] for key in compact} for state in hypotheses.values() if state["archived"]
            ]
        if mode in ("M0", "M1"):
            def rank(fact):
                return (-max(abs(fact[f"premeasurement_error_{target}_ml"]) for target in TARGETS),
                        fact["sample_id"])

            result["historical_top_abs_errors"] = [feedback(fact) for fact in sorted(acquired, key=rank)[:8]]
        return result

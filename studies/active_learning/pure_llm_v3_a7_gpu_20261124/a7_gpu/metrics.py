"""Summarize immutable A7 validation, acquisition and transport receipts."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import (
    CONDITION_AXES, _condition_value, condition_changes,
)
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import terminal_record

from .formal_contract import SEEDS


INITIAL, BATCH, ROUNDS = 333, 32, 36
MILESTONES = (525, 813, 1005, 1485)
SCORE = "combined_normalized_rmse"
ENDPOINT_METRICS = (SCORE, *(f"{target}_{metric}" for target in ("V1", "V2")
                            for metric in ("rmse", "mae", "r2")))
ATTEMPT_PATH = re.compile(r"attempt_(\d{2})\.started\.json")
ACQUISITION_DIAGNOSTICS = "acquisition_diagnostics.json"


def optional(path):
    return read(path) if path.is_file() else None


def aulc(curve):
    """Compute normalized raw-curve area only on a contiguous label prefix."""
    labels = [point["active_labels"] for point in curve]
    if labels != list(range(INITIAL, INITIAL + BATCH * len(labels), BATCH)):
        raise ValueError("validation curve must begin at L333 and contain a contiguous B32 prefix")
    if len(curve) < 2:
        return None
    if any(not math.isfinite(point[SCORE]) for point in curve):
        raise ValueError("validation AULC score is non-finite")
    area = sum((point[SCORE] + after[SCORE]) * BATCH / 2 for point, after in zip(curve, curve[1:]))
    return area / (labels[-1] - labels[0])


def _sealed_metrics(directory, receipt, index):
    fit_path = receipt.get("fit_path")
    allowed = re.fullmatch(r"fit_\d{3}", fit_path) if isinstance(fit_path, str) else None
    if not allowed and not (index == 0 and fit_path == "anchor"):
        raise ValueError("metric receipt has an invalid fit path")
    metric_path = directory / fit_path / "validation_metrics.json"
    if file_digest(metric_path) != receipt["validation_metrics_sha256"]:
        raise ValueError("validation metrics differ from the immutable state receipt")
    metrics = read(metric_path)
    values = {}
    for name in ENDPOINT_METRICS:
        value = metrics.get(name)
        if value is None and name != SCORE:
            values[name] = None
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"invalid validation metric {name}")
        else:
            values[name] = value
    for name in ("training_seconds", "epochs_run", "best_epoch"):
        values[name] = metrics.get(name)
    values["training_reused"] = index == 0 and fit_path == "anchor"
    return {"round": index, "active_labels": INITIAL + BATCH * index, **values}


def _unique_turns(directory):
    turns, transport = {}, []
    for turn in sorted(directory.glob("turn_[0-9][0-9]")):
        index = int(turn.name.split("_")[1])
        if not (turn / "request.json").is_file():
            continue
        loaded = {"decision": optional(turn / "decision.json"), "answer": None, "repair": None}
        if index:
            messages = read(turn / "request.json")["payload"]["input"]
            if len(messages) > 2:
                _, _, diagnostic_text = messages[-1]["content"].rpartition("\n")
                diagnostic = json.loads(diagnostic_text)
                loaded["repair"] = {key: diagnostic[key] for key in ("mode", "locked_choice_slots",
                                                                       "locked_candidate_id_slots")
                                    if key in diagnostic}
        for started_file in sorted(turn.glob("attempt_*.started.json")):
            match = ATTEMPT_PATH.fullmatch(started_file.name)
            if not match:
                raise ValueError("bad physical-attempt receipt name")
            attempt = int(match[1])
            standard = turn / f"attempt_{attempt:02d}.terminal.json"
            if not standard.is_file():
                transport.append({"status": "unknown_or_in_flight"})
                continue
            started, terminal = read(started_file), terminal_record(turn, attempt)
            if any(terminal.get(key) != value for key, value in started.items()):
                raise ValueError("terminal attempt does not match its started receipt")
            transport.append({**terminal, "observed_elapsed": read(standard).get("elapsed_seconds")})
            if terminal["status"] == "completed":
                if loaded["answer"] is not None:
                    raise ValueError("multiple complete replies share one logical request")
                loaded["answer"] = terminal
        turns[index] = loaded
    return turns, transport


def _repair_metrics(choices, first, first_repair):
    if not first or first.get("decision") is None or first["answer"] is None:
        return {}
    decision = first["decision"]
    if decision["state"] == "legal":
        return {}
    legal_slots = (first_repair or {}).get("locked_choice_slots")
    if not isinstance(legal_slots, list):
        return {}
    try:
        initial = json.loads(first["answer"]["text"])["choices"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(initial, list):
        return {}
    selected = {choice["candidate_id"]: choice for choice in choices}
    original = [initial[slot] for slot in legal_slots if type(slot) is int and 0 <= slot < len(initial)
                and isinstance(initial[slot], dict)]
    return {
        "initial_locked_choices": len(original),
        "retained_initial_legal_id_count": sum(choice.get("candidate_id") in selected for choice in original),
        "retained_initial_legal_choice_count": sum(
            selected.get(choice.get("candidate_id")) == choice for choice in original),
    }


def _recorded_conditions(rows):
    return [
        tuple(_condition_value(axis, row["conditions"][axis])
              if row["conditions"].get(axis) not in (None, "") else None for axis in CONDITION_AXES)
        for row in rows
    ]


def _coverage(before, added):
    observed, selected = _recorded_conditions(before), _recorded_conditions(added)

    def increment(prior, batch):
        return {"before": len(prior), "added": len(batch - prior), "after": len(prior | batch)}

    axes = {}
    for index, axis in enumerate(CONDITION_AXES):
        prior_values = {values[index] for values in observed if values[index] is not None}
        batch_values = {values[index] for values in selected if values[index] is not None}
        axes[axis] = increment(prior_values, batch_values)
    return {"levels": axes, "complete_profiles": increment(
        {values for values in observed if None not in values},
        {values for values in selected if None not in values},
    )}


def _contrast_counts(pairs):
    counts = dict(replicate=0, single_factor=0, multifactor=0, unclassified_missing_axes=0)
    for left, right in pairs:
        conditions = left["conditions"], right["conditions"]
        axes = tuple(axis for axis in CONDITION_AXES if all(row.get(axis) not in (None, "") for row in conditions))
        changed = len(condition_changes(*conditions, axes=axes))
        if changed > 1:
            counts["multifactor"] += 1
        elif len(axes) != len(CONDITION_AXES):
            counts["unclassified_missing_axes"] += 1
        else:
            counts["single_factor" if changed else "replicate"] += 1
    return counts


def record_batch_diagnostics(directory, freeze, candidates, observed):
    """Persist scope-bound summaries of the accepted batch before reading its new labels."""
    choice_path = directory / "selection.json"
    if file_digest(choice_path) != freeze["selection_file_sha256"]:
        raise ValueError("acquisition diagnostics require the frozen choice file")
    plan = read(choice_path)
    selected = [candidates[choice["candidate_id"]] for choice in plan["choices"]]
    selected_by_molecule, observed_by_molecule = defaultdict(list), defaultdict(list)
    for row in selected:
        selected_by_molecule[row["molecule_id"]].append(row)
    for row in observed.values():
        observed_by_molecule[row["molecule_id"]].append(row)
    new_molecules = selected_by_molecule.keys() - observed_by_molecule.keys()
    new_molecule_rows = sum(len(selected_by_molecule[molecule]) for molecule in new_molecules)
    updates = plan["hypothesis_updates"]
    once(directory / ACQUISITION_DIAGNOSTICS, {
        "freeze_sha256": freeze["freeze_sha256"],
        "selection_file_sha256": freeze["selection_file_sha256"],
        "selection_molecules": len(selected_by_molecule),
        "new_molecules": len(new_molecules),
        "new_molecule_fraction": len(new_molecules) / len(selected_by_molecule),
        "rows_on_new_molecules": new_molecule_rows,
        "row_fraction_on_new_molecules": new_molecule_rows / len(selected),
        "condition_coverage": _coverage(observed.values(), selected),
        "same_molecule_pairs": {
            "within_batch": _contrast_counts(
                pair for rows in selected_by_molecule.values() for pair in combinations(rows, 2)
            ),
            "with_existing_observations": _contrast_counts(
                (row, earlier) for row in selected for earlier in observed_by_molecule.get(row["molecule_id"], ())
            ),
        },
        "hypothesis_update_operations": dict(Counter(update["operation"] for update in updates)),
        "hypothesis_update_support_references": sum(len(update["supporting_observed_ids"]) for update in updates),
        "hypothesis_update_counterexample_references": sum(
            len(update["contradicting_observed_ids"]) for update in updates
        ),
    })


def _batch_summary(directory, freeze, turns):
    selection = directory / "selection.json"
    if file_digest(selection) != freeze["selection_file_sha256"]:
        raise ValueError("batch summary selection differs from frozen selection")
    plan = read(selection)
    choices = plan["choices"]
    if len(choices) != BATCH or len(set(freeze["batch_ids"])) != BATCH:
        raise ValueError("batch summary does not describe a frozen B32 selection")
    accepted = freeze["accepted_logical_index"]
    if (accepted not in turns or not turns[accepted]["decision"]
            or turns[accepted]["decision"]["state"] != "legal"):
        raise ValueError("batch does not have a recorded accepted logical decision")
    molecule_counts = Counter(choice["molecule_id"] for choice in choices)
    relationships = Counter(ref["relationship"] for choice in choices for ref in choice.get("references", []))
    diagnostics = optional(directory / ACQUISITION_DIAGNOSTICS)
    if diagnostics and any(diagnostics[key] != freeze[key] for key in ("freeze_sha256", "selection_file_sha256")):
        raise ValueError("acquisition diagnostics differ from the frozen selection")
    return {
        "batch_sealed": True, "accepted_logical_index": accepted,
        "accepted_after_correction": accepted > 0,
        "full_replan_requested_by_model": any(t["decision"] and t["decision"]["state"] == "model_requested_replan"
                                              for t in turns.values()),
        "molecule_count": len(molecule_counts), "max_rows_one_molecule": max(molecule_counts.values()),
        "relationship_counts": dict(relationships), "hypothesis_updates": len(plan.get("hypothesis_updates", [])),
        **({"acquisition_diagnostics": diagnostics} if diagnostics else {}),
        **_repair_metrics(choices, turns.get(0), (turns.get(1) or {}).get("repair")),
    }


def _round_metrics(directory, index):
    freeze = optional(directory / "batch_freeze.json")
    turns, attempts = _unique_turns(directory)
    first = turns.get(0, {})
    first_answer = first.get("answer")
    first_state = (first.get("decision") or {}).get("state")
    provider_complete = [row for row in attempts if row.get("provider_status") == "completed"]
    usage = Counter()
    duration = []
    for attempt in attempts:
        for name in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            value = attempt.get("usage", {}).get(name)
            if type(value) is int:
                usage[name] += value
        seconds = attempt.get("observed_elapsed")
        if type(seconds) in (int, float) and math.isfinite(seconds):
            duration.append(seconds)
    return {
        "round": index, "L_before": INITIAL + BATCH * index,
        "logical_requests_logged": len(turns),
        "complete_logical_responses": sum(turn["answer"] is not None for turn in turns.values()),
        "provider_completed_receipts": len(provider_complete),
        "local_repair_requests_logged": sum(bool(t["repair"] and t["repair"]["mode"] == "patch")
                                             for t in turns.values()),
        "physical_attempts_started": len(attempts),
        "open_or_unreconciled_attempts": sum(row["status"] == "unknown_or_in_flight" or row.get("ambiguous", False)
                                             for row in attempts),
        "first_response_evaluated": bool(first_answer and first_state),
        "first_response_legal": first_state == "legal" if first_answer and first_state else None,
        "provider_usage_recorded_attempts": sum(bool(row.get("usage")) for row in attempts),
        "provider_token_usage": dict(usage),
        "physical_elapsed_seconds_observed": sum(duration),
        "physical_elapsed_records": len(duration),
        "transport_failures": dict(Counter(row["error"] for row in attempts if "error" in row)),
        **(_batch_summary(directory, freeze, turns) if freeze else {"batch_sealed": False}),
    }


def _seed_snapshot(root, seed, planned_rounds=ROUNDS):
    directory = root / f"seed_{seed}"
    initial_path = directory / "initial"
    contract = optional(directory / "seed_contract.json")
    curve, rounds = [], []
    initial = optional(initial_path / "initial.json")
    if initial:
        curve.append(_sealed_metrics(initial_path, initial, 0))
    for number in range(planned_rounds):
        current = directory / f"round_{number:02d}"
        if current.is_dir():
            rounds.append(_round_metrics(current, number))
        next_path = directory / f"round_{number + 1:02d}"
        if transition := optional(next_path / "transition.json"):
            if len(curve) != number + 1:
                raise ValueError("completed validation rounds do not form a prefix")
            if file_digest(current / "batch_freeze.json") != transition["selection_freeze_sha256"]:
                raise ValueError("metric transition does not share the accepted selection freeze")
            curve.append(_sealed_metrics(next_path, transition, number + 1))
    events = []
    for path in sorted((directory / "events").glob("*.json")):
        row = read(path)
        if row.get("status") in {"failed", "paused", "complete", "import_completed", "confirmed_not_processed"}:
            events.append({name: row[name] for name in ("phase", "status", "round", "recorded_at_utc", "error_type")
                           if name in row})
    events.sort(key=lambda record: record.get("recorded_at_utc", ""))
    denominator = sum(row["first_response_evaluated"] for row in rounds)
    numerator = sum(row["first_response_legal"] is True for row in rounds)
    tokens, failures = Counter(), Counter()
    for row in rounds:
        tokens.update(row["provider_token_usage"])
        failures.update(row["transport_failures"])
    return {
        "seed": seed, "mock": (contract or {}).get("formal_run", {}).get("mock"),
        "latest_budget": curve[-1]["active_labels"] if curve else None,
        "acquisitions_with_committed_feedback_and_fit": max(0, len(curve) - 1),
        "validation_curve": curve, "label_aulc_through_latest": aulc(curve),
        "milestones": {str(point["active_labels"]): point for point in curve
                       if point["active_labels"] in (*MILESTONES, INITIAL + BATCH * planned_rounds)},
        "first_response_legality": {"legal": numerator, "evaluated": denominator,
                                    "fraction": numerator / denominator if denominator else None},
        "total_physical_attempts_started": sum(row["physical_attempts_started"] for row in rounds),
        "total_logical_requests_logged": sum(row["logical_requests_logged"] for row in rounds),
        "total_complete_logical_responses": sum(row["complete_logical_responses"] for row in rounds),
        "total_token_usage": dict(tokens), "transport_failures": dict(failures),
        "api_attempt_seconds_observed": sum(row["physical_elapsed_seconds_observed"] for row in rounds),
        "training_seconds_observed": sum(point["training_seconds"] or 0 for point in curve
                                          if not point["training_reused"]),
        "rounds": rounds, "failure_and_pause_events": events,
        "worker": optional(root / "workers" / f"seed_{seed}.json"),
    }


def _statistics(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "sample_sd": stdev(values) if len(values) > 1 else None}


def snapshot(root):
    """Read only scope-bound artifacts. Never prepare features, reveal labels, or fit."""
    root = Path(root)
    profile = (optional(root / "formal_contract.json") or {}).get("profile", {})
    pilot = profile.get("requires_initial_qualification") is False
    planned_seeds = SEEDS[:3] if pilot else SEEDS
    planned_rounds = 10 if pilot else ROUNDS
    target = INITIAL + BATCH * planned_rounds
    if pilot and (profile.get("confirmation_seeds"), profile.get("target_labels")) != (list(planned_seeds), target):
        raise ValueError("invalid pilot status scope")
    seeds = [_seed_snapshot(root, seed, planned_rounds) for seed in planned_seeds]
    formal = [row for row in seeds if row["mock"] is False]
    endpoints = {}
    for budget in MILESTONES if not pilot else (525, target):
        measured = [row["milestones"][str(budget)] for row in formal if str(budget) in row["milestones"]]
        endpoints[str(budget)] = {metric: _statistics([point[metric] for point in measured if point[metric] is not None])
                                  for metric in ENDPOINT_METRICS}
    terminal = [row["label_aulc_through_latest"] for row in formal if row["latest_budget"] == target]
    return {"scope": "A7 M2 validation and execution receipts",
            "initialized": (root / "formal_contract.json").is_file(),
            "paused": (root / "pause.request").is_file(),
            "launch_qualification_recorded": (root / "qualification.json").is_file(),
            "planned_seeds": list(planned_seeds), "planned_rounds": planned_rounds, "target_labels": target,
            "scope_complete": len(terminal) == len(planned_seeds),
            "five_seed_complete": not pilot and len(terminal) == len(SEEDS),
            "formal_seed_milestones": endpoints,
            f"formal_label_aulc_L333_L{target}": _statistics(terminal),
            "seeds": seeds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

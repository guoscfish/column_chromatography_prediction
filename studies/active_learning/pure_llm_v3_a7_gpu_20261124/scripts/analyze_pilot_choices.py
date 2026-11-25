"""Analyze a read-only export of A7 pilot receipts and accepted choices.

Input is the JSON from a7_gpu.metrics.snapshot, optionally enriched per round
with selection.json as `selection`, the accepted decision or its diagnosis as
`validation`, frozen hashes as `freeze`, and a memory history as `memory`.
Only the export is read. The report has no selection, label or training hooks.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.metrics import BATCH, INITIAL, SCORE, aulc


COMPARISON_METRICS = (SCORE, "V1_r2", "V2_r2")
ROUND_COUNTS = ("open_or_unreconciled_attempts", "provider_completed_receipts", "local_repair_requests_logged",
                "physical_attempts_started", "logical_requests_logged")


def _fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def _summary(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "sample_sd": stdev(values) if len(values) > 1 else None}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count_fields(values):
    return {key: val for key, val in values.items() if type(val) is int and val >= 0}


def _index(rows, key, allowed):
    entries = {}
    for row in rows:
        identity = row[key]
        if type(identity) is not int or identity not in allowed or identity in entries:
            raise ValueError(f"invalid or duplicate {key} in the export")
        entries[identity] = row
    return entries


def _record_sha256(value):
    # records.once publishes canonical sorted JSON plus one newline.
    return hashlib.sha256((dumps(value, sort_keys=True) + "\n").encode()).hexdigest()


def _memory_summary(memory, identity):
    if memory is None:
        return None
    if memory.get("seed", identity) != identity:
        raise ValueError("exported memory belongs to a different seed")
    histories = memory.get("hypotheses")
    if not isinstance(histories, dict):
        return None
    versions = [history for history in histories.values() if isinstance(history, list) and history]
    if len(versions) != len(histories) or any(
        not isinstance(history[-1], dict) or type(history[-1].get("archived")) is not bool for history in versions
    ):
        raise ValueError("invalid exported hypothesis history")
    archived = sum(history[-1]["archived"] for history in versions)
    return {"active_hypotheses": len(histories) - archived, "archived_hypotheses": archived,
            "versions": sum(map(len, versions))}


def _choice_evidence(choice):
    # Deliberately use the short IDs, whose scope is this seed and round.
    return {key: choice[key] for key in ("candidate_id", "molecule_id", "condition_anchor", "hypothesis_id",
                                        "reason", "references")}


def _accepted_selection(row, *, include_choices):
    plan = row.get("selection")
    if row.get("batch_sealed") is not True or plan is None:
        return None
    choices = [_choice_evidence(choice) for choice in plan["choices"]]
    if len(choices) != BATCH or len({choice["candidate_id"] for choice in choices}) != BATCH:
        raise ValueError("exported frozen selection is not an ordered B32 of distinct candidate IDs")
    diagnostics, freeze = row.get("acquisition_diagnostics") or {}, row.get("freeze") or {}
    selection_sha = _record_sha256(plan)
    for evidence in (diagnostics, freeze):
        if "selection_file_sha256" in evidence and selection_sha != evidence["selection_file_sha256"]:
            raise ValueError("exported selection does not match its immutable file SHA")
    if "freeze_sha256" in diagnostics and "freeze_sha256" in freeze:
        if diagnostics["freeze_sha256"] != freeze["freeze_sha256"]:
            raise ValueError("exported acquisition diagnostics mismatch the freeze SHA")
    decision = row.get("validation") or {}
    diagnosis = decision.get("diagnosis", {}) if "state" in decision else decision
    if decision.get("state", "legal") != "legal" or (diagnosis and (
        diagnosis.get("valid") is not True or diagnosis.get("valid_choice_count") != BATCH
        or diagnosis.get("expected_count", BATCH) != BATCH or diagnosis.get("errors", [])
    )):
        raise ValueError("exported accepted decision disagrees with B32 validation")

    molecules = Counter(choice["molecule_id"] for choice in choices)
    refs = Counter((ref["reference_kind"], ref["relationship"]) for choice in choices for ref in choice["references"])
    if "selection_molecules" in diagnostics and diagnostics["selection_molecules"] != len(molecules):
        raise ValueError("frozen molecule counts disagree with the exported choices")
    updates = plan["hypothesis_updates"]
    first_evidence = next((choice for choice in choices if choice["references"]), choices[0])
    return {
        "accepted_logical_index": row.get("accepted_logical_index"),
        "accepted_after_correction": row.get("accepted_after_correction"),
        "accepted_validation_included": bool(diagnosis), "selected_rows": len(choices),
        "selection_file_sha256": selection_sha, "distinct_molecule_groups": len(molecules),
        "max_rows_one_molecule": max(molecules.values()), "repeated_molecule_groups": sum(n > 1 for n in molecules.values()),
        "rows_with_references": sum(bool(choice["references"]) for choice in choices),
        "rows_with_hypothesis_reference": sum(choice["hypothesis_id"] is not None for choice in choices),
        "references_by_kind_and_relationship": {f"{kind}/{relation}": n for (kind, relation), n in refs.items()},
        "hypothesis_update_operations": dict(Counter(update["operation"] for update in updates)),
        "hypothesis_update_support_references": sum(len(update["supporting_observed_ids"]) for update in updates),
        "hypothesis_update_counterexample_references": sum(len(update["contradicting_observed_ids"]) for update in updates),
        "batch_strategy": plan["batch_strategy"], "batch_review": plan["batch_review"],
        "feedback_interpretation": plan["feedback_interpretation"],
        "reason_reference_examples": choices if include_choices else [first_evidence],
        **({"hypothesis_updates": updates} if include_choices else {}),
    }


def _performance(seed, planned_rounds):
    points = _index(seed.get("validation_curve", []), "round", range(planned_rounds + 1))
    metrics = []
    for index, point in sorted(points.items()):
        budget = INITIAL + BATCH * index
        if point["active_labels"] != budget or not _number(point.get(SCORE)):
            raise ValueError("validation point is outside its B32 budget or has invalid NRMSE")
        projected = {"round": index, "active_labels": budget}
        for name in COMPARISON_METRICS:
            value = point.get(name)
            if value is not None and not _number(value):
                raise ValueError(f"non-finite validation metric {name}")
            projected[name] = value
        metrics.append(projected)
    contiguous = []
    for point in metrics:
        if point["round"] != len(contiguous):
            break
        contiguous.append(point)
    end = contiguous[-1]["active_labels"] if contiguous else None
    return {"validation_curve": metrics, "n_validation_points": len(metrics),
            "n_committed_rounds_with_metrics": sum(point["round"] > 0 for point in metrics),
            "latest_budget_with_metrics": metrics[-1]["active_labels"] if metrics else None,
            "aulc": {"L_from": INITIAL if contiguous else None, "L_through": end, "n_points": len(contiguous),
                     "value": aulc(contiguous), "n_points_after_first_gap_excluded": len(metrics) - len(contiguous)},
            "target_complete": len(contiguous) == planned_rounds + 1}


def _round_evidence(seed, planned_rounds, *, include_choices):
    identity = seed["seed"]
    rounds = _index(seed.get("rounds", []), "round", range(planned_rounds))
    for index, row in sorted(rounds.items()):
        budget = INITIAL + BATCH * index
        if row.get("L_before", budget) != budget:
            raise ValueError("round is outside its B32 budget")
        legality = row.get("first_response_legal")
        evaluated = row.get("first_response_evaluated", legality is not None)
        if type(evaluated) is not bool or evaluated != (type(legality) is bool):
            raise ValueError("inconsistent first-response evaluation and legality flags")
        metrics = _count_fields({name: row.get(name) for name in ROUND_COUNTS})
        yield {
            "round": index, "L_before": budget, "batch_sealed": row.get("batch_sealed") is True,
            "first_response_legal": legality, **metrics,
            "transport_failures": _count_fields(row.get("transport_failures") or {}),
            "accepted_selection": _accepted_selection(row, include_choices=include_choices),
            "acquisition_diagnostics": row.get("acquisition_diagnostics") if row.get("batch_sealed") is True else None,
            "memory": _memory_summary(row.get("memory"), identity) if row.get("batch_sealed") is True else None,
        }


def _acquisition_aggregate(diagnostics):
    groups, new_groups, choices, new_choices = 0, 0, 0, 0
    pairs = defaultdict(Counter)
    axes = defaultdict(lambda: {"n": 0, "added_total": 0})
    profiles = {"n": 0, "added_total": 0}
    for row in diagnostics:
        groups += row["selection_molecules"]
        new_groups += row["new_molecules"]
        choices += BATCH
        new_choices += row["rows_on_new_molecules"]
        for kind, counts in row["same_molecule_pairs"].items():
            pairs[kind].update(_count_fields(counts))
        coverage = row["condition_coverage"]
        for axis, levels in coverage["levels"].items():
            axes[axis]["n"] += 1
            axes[axis]["added_total"] += levels["added"]
        profiles["n"] += 1
        profiles["added_total"] += coverage["complete_profiles"]["added"]
    return {"diagnostic_batches": len(diagnostics), "new_molecule_group_occurrences": _fraction(new_groups, groups),
            "rows_on_new_molecules": _fraction(new_choices, choices),
            "same_molecule_pairs": {kind: {"n": sum(counts.values()), "by_changed_axes": dict(counts)}
                                    for kind, counts in pairs.items()},
            "condition_levels_added_sum_by_batch": dict(axes), "complete_condition_profiles_added_sum_by_batch": profiles}


def _rollup(rows, planned):
    batches = [row for row in rows if row["batch_sealed"]]
    choices = [row["accepted_selection"] for row in batches if row["accepted_selection"] is not None]
    diagnostics = [row["acquisition_diagnostics"] for row in batches if row["acquisition_diagnostics"] is not None]
    legality = [row["first_response_legal"] for row in rows if row["first_response_legal"] is not None]
    reference_counts, hypothesis_ops, failures = Counter(), Counter(), Counter()
    for row in choices:
        reference_counts.update(row["references_by_kind_and_relationship"])
        hypothesis_ops.update(row["hypothesis_update_operations"])
    for row in rows:
        failures.update(row["transport_failures"])
    selected_rows = sum(row["selected_rows"] for row in choices)
    attempt_counts = {name: [row[name] for row in rows if name in row] for name in ROUND_COUNTS}
    return {
        "planned_rounds": planned, "formal_rounds_exported": len(rows), "sealed_batches": len(batches),
        "sealed_batches_with_selection": len(choices), "sealed_batches_with_diagnostics": len(diagnostics),
        "sealed_batches_with_accepted_validation": sum(row["accepted_validation_included"] for row in choices),
        "sealed_batches_with_memory": sum(row["memory"] is not None for row in batches),
        "first_response_legality": _fraction(sum(legality), len(legality)),
        "accepted_after_correction": _fraction(sum(row["accepted_after_correction"] is True for row in choices),
                                               sum(type(row["accepted_after_correction"]) is bool for row in choices)),
        "execution_counts": {name: {"n_rounds": len(values), "total": sum(values)}
                             for name, values in attempt_counts.items()},
        "transport_failures": dict(failures), "selected_rows_with_full_export": selected_rows,
        "distinct_molecule_groups_per_batch": _summary([row["distinct_molecule_groups"] for row in choices]),
        "max_rows_one_molecule_per_batch": _summary([row["max_rows_one_molecule"] for row in choices]),
        "rows_with_references": _fraction(sum(row["rows_with_references"] for row in choices), selected_rows),
        "rows_with_hypothesis_reference": _fraction(sum(row["rows_with_hypothesis_reference"] for row in choices), selected_rows),
        "references_by_kind_and_relationship": dict(reference_counts), "hypothesis_update_operations": dict(hypothesis_ops),
        "hypothesis_update_support_references": sum(row["hypothesis_update_support_references"] for row in choices),
        "hypothesis_update_counterexample_references": sum(row["hypothesis_update_counterexample_references"] for row in choices),
        "acquisition": _acquisition_aggregate(diagnostics),
    }


def analyze_export(export, *, include_choices=False):
    """Return seed-local receipt statistics and metrics at like-for-like label budgets."""
    expected = export["planned_seeds"]
    if not expected or any(type(seed) is not int or seed < 0 for seed in expected) or len(expected) != len(set(expected)):
        raise ValueError("planned seeds must be distinct nonnegative integers")
    planned = export["planned_rounds"]
    if type(planned) is not int or planned < 1 or export["target_labels"] != INITIAL + BATCH * planned:
        raise ValueError("planned rounds and target do not agree with L333/B32")
    seeds = _index(export.get("seeds", []), "seed", expected)
    result, all_rounds, points_by_budget, aulc_by_budget = [], [], defaultdict(list), defaultdict(list)
    for identity in expected:
        exported = seeds.get(identity, {})
        formal = exported.get("mock") is False
        rounds = list(_round_evidence(exported, planned, include_choices=include_choices)) if formal else []
        performance = _performance(exported if formal else {}, planned)
        local_metrics = {point["round"]: point for point in performance["validation_curve"]}
        for row in rounds:
            row["validation_metrics_after_batch"] = local_metrics.get(row["round"] + 1)
        for point in performance["validation_curve"]:
            points_by_budget[point["active_labels"]].append(point)
        for count in range(2, performance["aulc"]["n_points"] + 1):
            prefix = performance["validation_curve"][:count]
            aulc_by_budget[prefix[-1]["active_labels"]].append(aulc(prefix))
        all_rounds.extend(rounds)
        result.append({"seed": identity, "worker": (exported.get("worker") or {}).get("stage"),
                       "real_receipts": formal, "exported_seed_record": identity in seeds, **performance,
                       "summary": _rollup(rounds, planned), "rounds": rounds})
    return {
        "planned_seeds": expected, "target_labels": export["target_labels"],
        "complete_seeds": [seed["seed"] for seed in result if seed["target_complete"]],
        "summary": _rollup(all_rounds, planned * len(expected)), "seeds": result,
        "metrics_by_equal_label_budget": {str(budget): {
            name: {**_summary([point[name] for point in points if point[name] is not None]), "planned_seeds": len(expected)}
            for name in COMPARISON_METRICS} for budget, points in sorted(points_by_budget.items())},
        "normalized_nrmse_aulc_by_equal_span": {f"L{INITIAL}_L{budget}": {
            **_summary(values), "planned_seeds": len(expected)} for budget, values in sorted(aulc_by_budget.items())},
        "limits": [
            "Only seed records marked mock=false contribute. An absent or incomplete receipt is not a zero or a failed validation.",
            "AULC uses the contiguous L333/B32 NRMSE prefix. Aggregates compare only equal budgets or equal spans.",
            "Molecule counts and new condition levels are summed as seed-round occurrences, not distinct cross-round IDs.",
            "Reasons and references are accepted model evidence. This export alone cannot independently verify every claim or infer causality.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="enriched pilot metrics snapshot JSON, or - for stdin")
    parser.add_argument("--include-choices", action="store_true", help="include every accepted reason, reference and hypothesis update")
    args = parser.parse_args()
    contents = sys.stdin.read() if args.snapshot == "-" else Path(args.snapshot).read_text()
    try:
        report = analyze_export(json.loads(contents), include_choices=args.include_choices)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"invalid pilot export ({type(error).__name__}): {error}")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

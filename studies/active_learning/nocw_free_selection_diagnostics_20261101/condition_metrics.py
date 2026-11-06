#!/usr/bin/env python3
"""Exact, response-free same-molecule condition comparisons for one batch.

Public API
----------
compute_condition_metrics(selected_ids, labeled_ids, meta, return_examples=3)
    Return ``(metrics, examples)``. ``metrics`` is a flat dictionary whose
    condition-pair columns begin with ``within_`` (unordered selected/selected)
    or ``cross_`` (selected/previously labeled). ``examples`` contains at most
    ``return_examples`` reviewable pairs per scope and elementary category.

Each metadata record must have a real molecule key in ``smiles`` and a mapping
in ``conditions``. The supported condition keys are documented in FIELD_KEYS.
No row ID is ever used as a molecule identifier. Numeric values are compared
as exact Decimal(str(value)); PE/EA is additionally reduced as a Fraction.
Solvent strings are compared exactly, including their whitespace and case.
No responses, models, fingerprints or feature caches are read by this module.

For every category, ``*_pairs`` is the raw pair count, while
``*_selected_experiments`` and ``*_molecules`` are union counts. The denominator
of ``*_selected_slots_fraction`` is the full accepted batch size, including
slots with unavailable metadata. Cross comparisons also expose the union count
of ``*_previous_experiments``. ``any_single`` unions all five only-one-factor
categories, so one selected experiment is counted once even if it participates
in many matches. Multi-factor means two or more of the five observed factors
differ, not evidence of redundant or uninformative selection.

Missing/invalid metadata never matches another missing value. Such comparisons
are skipped and counted explicitly. This means exact comparison is conservative
with respect to unrecorded conditions; conclusions concern these five fields.

Run the small deterministic, in-memory checks with::

    python3 condition_metrics.py --self-test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import combinations
from typing import Any, Mapping, Sequence


FIELD_KEYS = {
    "pe": ("PE/EA", "pe"),
    "density": ("Density g/ml", "density"),
    "sample_volume": ("V/ul", "sample_volume"),
    "loading_solvent": ("loading solvent", "loading_solvent"),
    "loading_volume": (
        "Volume of loading solvent/ul",
        "loading_volume",
    ),
}
FACTORS = tuple(FIELD_KEYS)
SINGLE_CATEGORIES = tuple("only_" + factor for factor in FACTORS)
CATEGORIES = ("exact",) + SINGLE_CATEGORIES + ("multi_factor", "any_single")


def _numeric(value: Any) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError("missing or boolean numeric condition")
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid numeric condition") from exc
    if not result.is_finite():
        raise ValueError("nonfinite numeric condition")
    return result


def _pe_ratio(value: Any) -> Fraction:
    if value is None or isinstance(value, bool):
        raise ValueError("missing or boolean PE/EA condition")
    text = str(value).strip()
    separator = "/" if "/" in text else ":" if ":" in text else None
    if separator is None:
        return Fraction(_numeric(value))
    parts = text.split(separator)
    if len(parts) != 2:
        raise ValueError("invalid PE/EA ratio")
    denominator = _numeric(parts[1])
    if not denominator:
        raise ValueError("zero PE/EA denominator")
    return Fraction(_numeric(parts[0])) / Fraction(denominator)


def _canonical_conditions(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    conditions = record.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    result = []
    try:
        for factor, aliases in FIELD_KEYS.items():
            present = [key for key in aliases if key in conditions]
            if not present:
                return None
            # When both aliases exist, require semantic equality; do not choose
            # whichever field happens to be first and hide a disagreement.
            normalized = []
            for key in present:
                value = conditions[key]
                if factor == "pe":
                    value = _pe_ratio(value)
                elif factor == "loading_solvent":
                    if not isinstance(value, str) or value == "":
                        raise ValueError("missing or nonstring loading solvent")
                else:
                    value = _numeric(value)
                normalized.append(value)
            if any(value != normalized[0] for value in normalized[1:]):
                return None
            result.append(normalized[0])
    except (ValueError, InvalidOperation, TypeError):
        return None
    return tuple(result)


def _unique_ids(ids: Sequence[str], description: str) -> list[str]:
    result = [str(row_id) for row_id in ids]
    if len(result) != len(set(result)):
        raise ValueError(f"{description} contains duplicate row IDs")
    return result


def compute_condition_metrics(
    selected_ids: Sequence[str],
    labeled_ids: Sequence[str],
    meta: Mapping[str, Mapping[str, Any]],
    return_examples: int = 3,
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Compute one batch's exact same-molecule condition-pair measurements.

    ``labeled_ids`` must describe the label set immediately BEFORE selection,
    and must be disjoint from ``selected_ids``. Duplicate IDs or overlap raise
    ValueError instead of silently corrupting slot denominators. Either empty
    collection is supported. Set ``return_examples=0`` to disable examples.
    The function stores sets of participating IDs, never a complete pair list.
    """
    selected = _unique_ids(selected_ids, "selected_ids")
    previous = _unique_ids(labeled_ids, "labeled_ids")
    if set(selected) & set(previous):
        raise ValueError("selected_ids overlap the pre-selection labeled_ids")
    if not isinstance(return_examples, int) or return_examples < 0:
        raise ValueError("return_examples must be a nonnegative integer")

    groups: dict[str, dict[str, list[str]]] = {
        "selected": defaultdict(list),
        "previous": defaultdict(list),
    }
    canonical: dict[str, tuple[Any, ...] | None] = {}
    metrics: dict[str, Any] = {
        "condition_selected_slots": len(selected),
        "condition_previous_slots": len(previous),
    }
    for population, ids in (("selected", selected), ("previous", previous)):
        missing_metadata = missing_molecule = incomplete_conditions = 0
        for row_id in ids:
            record = meta.get(row_id)
            if not isinstance(record, Mapping):
                missing_metadata += 1
                continue
            smiles = record.get("smiles")
            if not isinstance(smiles, str) or not smiles:
                missing_molecule += 1
                continue
            groups[population][smiles].append(row_id)
            canonical[row_id] = _canonical_conditions(record)
            if canonical[row_id] is None:
                incomplete_conditions += 1
        metrics[f"condition_{population}_missing_metadata"] = missing_metadata
        metrics[f"condition_{population}_missing_molecule"] = missing_molecule
        metrics[f"condition_{population}_incomplete_conditions"] = incomplete_conditions

    examples = {
        scope: {category: [] for category in CATEGORIES if category != "any_single"}
        for scope in ("within", "cross")
    }
    for scope in ("within", "cross"):
        counters = {
            category: {
                "pairs": 0,
                "selected": set(),
                "previous": set(),
                "molecules": set(),
            }
            for category in CATEGORIES
        }
        candidate_pairs = skipped_pairs = 0
        for smiles, selected_group in groups["selected"].items():
            if scope == "within":
                pairs = combinations(selected_group, 2)
            else:
                pairs = (
                    (selected_id, previous_id)
                    for selected_id in selected_group
                    for previous_id in groups["previous"].get(smiles, ())
                )
            for selected_id, other_id in pairs:
                candidate_pairs += 1
                first, second = canonical[selected_id], canonical[other_id]
                if first is None or second is None:
                    skipped_pairs += 1
                    continue
                changed = [
                    factor
                    for factor, before, after in zip(FACTORS, first, second)
                    if before != after
                ]
                category = (
                    "exact" if not changed else
                    "only_" + changed[0] if len(changed) == 1 else
                    "multi_factor"
                )
                update_categories = [category]
                if len(changed) == 1:
                    update_categories.append("any_single")
                for update_category in update_categories:
                    counter = counters[update_category]
                    counter["pairs"] += 1
                    counter["selected"].add(selected_id)
                    counter["selected" if scope == "within" else "previous"].add(other_id)
                    counter["molecules"].add(smiles)
                if len(examples[scope][category]) < return_examples:
                    examples[scope][category].append({
                        "selected_id": selected_id,
                        "other_id": other_id,
                        "other_population": "selected" if scope == "within" else "previous",
                        "smiles": smiles,
                        "changed_factors": changed,
                        "selected_conditions": {
                            factor: str(value) for factor, value in zip(FACTORS, first)
                        },
                        "other_conditions": {
                            factor: str(value) for factor, value in zip(FACTORS, second)
                        },
                    })
        metrics[f"{scope}_same_molecule_candidate_pairs"] = candidate_pairs
        metrics[f"{scope}_condition_comparable_pairs"] = candidate_pairs - skipped_pairs
        metrics[f"{scope}_condition_incomplete_pairs"] = skipped_pairs
        for category, counter in counters.items():
            prefix = f"{scope}_{category}"
            metrics[prefix + "_pairs"] = counter["pairs"]
            metrics[prefix + "_selected_experiments"] = len(counter["selected"])
            metrics[prefix + "_molecules"] = len(counter["molecules"])
            metrics[prefix + "_selected_slots_fraction"] = (
                len(counter["selected"]) / len(selected) if selected else 0.0
            )
            if scope == "cross":
                metrics[prefix + "_previous_experiments"] = len(counter["previous"])
        assert sum(counters[c]["pairs"] for c in CATEGORIES if c != "any_single") == (
            candidate_pairs - skipped_pairs
        )
        assert counters["any_single"]["pairs"] == sum(
            counters[c]["pairs"] for c in SINGLE_CATEGORIES
        )
    return metrics, examples


def run_self_tests() -> None:
    """Exercise exactness, pair counts, slot unions, missing data and IDs."""
    def record(smiles="A", **updates):
        conditions = dict(pe="1:1", density="0.1", sample_volume=10,
                          loading_solvent="PE", loading_volume=20)
        conditions.update(updates)
        return {"smiles": smiles, "conditions": conditions}

    meta = {
        "s1": record(), "s2": record(pe=2), "s3": record(density="0.2"),
        "s4": record(pe="2/2"), "s5": record("B"),
        "p1": record(), "p2": record(sample_volume=11), "p3": record("B"),
    }
    result, examples = compute_condition_metrics(
        ["s1", "s2", "s3", "s4", "s5"], ["p1", "p2", "p3"], meta, 1
    )
    assert result["within_condition_comparable_pairs"] == 6
    assert result["within_exact_pairs"] == 1
    assert result["within_only_pe_pairs"] == 2
    assert result["within_only_density_pairs"] == 2
    assert result["within_multi_factor_pairs"] == 1
    assert result["within_any_single_pairs"] == 4
    assert result["within_any_single_selected_experiments"] == 4
    assert result["within_any_single_selected_slots_fraction"] == 4 / 5
    assert result["cross_condition_comparable_pairs"] == 9
    assert result["cross_exact_pairs"] == 3
    assert result["cross_any_single_pairs"] == 4
    assert result["cross_any_single_selected_experiments"] == 4
    assert result["cross_any_single_previous_experiments"] == 2
    assert result["cross_any_single_molecules"] == 1
    assert all(len(items) <= 1 for categories in examples.values() for items in categories.values())

    # A single selected slot may form five single-factor pairs; its slot share
    # is still 100%, and its molecule union is still one.
    factor_meta = {"s": record(), "p0": record()}
    changes = dict(pe=2, density="0.10000000001", sample_volume=11,
                   loading_solvent="PE ", loading_volume=21)
    for factor, value in changes.items():
        factor_meta["p_" + factor] = record(**{factor: value})
    result, _ = compute_condition_metrics(
        ["s"], [key for key in factor_meta if key != "s"], factor_meta, 0
    )
    assert result["cross_exact_pairs"] == 1
    assert all(result["cross_" + category + "_pairs"] == 1 for category in SINGLE_CATEGORIES)
    assert result["cross_any_single_pairs"] == 5
    assert result["cross_any_single_selected_experiments"] == 1
    assert result["cross_any_single_previous_experiments"] == 5
    assert result["cross_any_single_selected_slots_fraction"] == 1.0

    raw = {"smiles": "A", "conditions": {
        "PE/EA": "10/10", "Density g/ml": "0.100", "V/ul": "10.0",
        "loading solvent": "PE", "Volume of loading solvent/ul": "20.00",
    }}
    result, _ = compute_condition_metrics(["s"], ["p"], {"s": record(), "p": raw})
    assert result["cross_exact_pairs"] == 1
    broken = record(density=float("nan"))
    result, _ = compute_condition_metrics(
        ["s", "missing", "no_molecule"], ["p"],
        {"s": broken, "p": record(), "no_molecule": record("")}
    )
    assert result["condition_selected_missing_metadata"] == 1
    assert result["condition_selected_missing_molecule"] == 1
    assert result["condition_selected_incomplete_conditions"] == 1
    assert result["cross_condition_incomplete_pairs"] == 1
    assert result["cross_exact_pairs"] == 0
    assert compute_condition_metrics([], [], {})[0]["within_any_single_selected_slots_fraction"] == 0
    for selected, previous in ((["s", "s"], []), (["s"], ["s"]), ([], ["p", "p"])):
        try:
            compute_condition_metrics(selected, previous, {})
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate/overlapping IDs were not rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic in-memory checks")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("this standalone module only accepts --self-test")
    run_self_tests()
    print("condition_metrics self-tests passed")

"""The Agent boundary never exposes labels or mutates experiment state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from ..infra.contracts import budget_value, sample_ids

# Only this explicit, small metadata schema can cross the boundary.
CATALOGUE_FIELDS = frozenset({"sample_id", "rank", "nearest_center_sq_distance"})


def filtered_catalogue(rows: Sequence[Mapping[str, object]], *, candidate_ids: Sequence[str]) -> list[dict[str, object]]:
    authorized = sample_ids(candidate_ids, name="catalogue authorization")
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        identity = row.get("sample_id")
        if not isinstance(identity, str) or identity not in authorized or identity in by_id:
            raise ValueError("catalogue has duplicate or unauthorized sample ID")
        filtered = {key: value for key, value in row.items() if key in CATALOGUE_FIELDS}
        for key, value in filtered.items():
            if key != "sample_id" and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("catalogue metadata must contain finite numeric scalars")
        by_id[identity] = filtered
    if set(by_id) != set(authorized):
        raise ValueError("catalogue must cover authorized candidates exactly")
    return [by_id[identity] for identity in authorized]


def validate_plan(plan: Mapping[str, object], *, candidate_ids: Sequence[str], budget: int) -> tuple[str, ...]:
    authorized = sample_ids(candidate_ids, name="plan authorization")
    budget_value(budget, len(authorized))
    if set(plan) != {"selected_ids"} or not isinstance(plan["selected_ids"], (list, tuple)):
        raise ValueError("plan must contain only a selected_ids sequence")
    chosen = sample_ids(plan["selected_ids"], name="agent plan")
    if len(chosen) != budget or not set(chosen) <= set(authorized):
        raise ValueError("agent plan violates authorized candidate set or budget")
    return chosen

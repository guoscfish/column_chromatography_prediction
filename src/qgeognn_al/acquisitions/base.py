"""Pure acquisition adapter with identity and budget validation."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from ..infra.contracts import budget_value, canonical_bytes, content_hash, sample_ids
from ..representations.base import FeatureSnapshot
from .lcmd import lcmd_tp_select


@dataclass(frozen=True)
class Selection:
    selected_ids: tuple[str, ...]
    acquisition: str
    seed: int
    budget: int
    feature_hash: str
    trace_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_ids", sample_ids(self.selected_ids, name="selection"))
        if len(self.selected_ids) != self.budget:
            raise ValueError("selection length differs from budget")

    @property
    def trace(self) -> list[dict[str, object]]:
        # Return an owned copy, never mutable internal audit state.
        return json.loads(self.trace_json)

    def to_dict(self) -> dict[str, object]:
        return {"selected_ids": list(self.selected_ids), "acquisition": self.acquisition,
                "seed": self.seed, "budget": self.budget,
                "feature_hash": self.feature_hash, "trace": self.trace}


def select(snapshot: FeatureSnapshot, *, acquisition: str, budget: int, seed: int) -> Selection:
    snapshot.verify()
    budget_value(budget, len(snapshot.candidate_ids))
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if acquisition == "lcmd":
        result = lcmd_tp_select(snapshot.candidates, snapshot.train, budget)
        positions = result.selected_pool_positions.tolist()
        trace = [dict(row, sample_id=snapshot.candidate_ids[int(row["pool_position"])])
                 for row in result.trace]
    elif acquisition == "random":
        positions = np.random.default_rng(seed).choice(len(snapshot.candidate_ids), budget, replace=False).tolist()
        trace = [{"selection_order": index, "pool_position": position,
                  "sample_id": snapshot.candidate_ids[position]} for index, position in enumerate(positions)]
    else:
        raise ValueError(f"unsupported acquisition: {acquisition}")
    ids = tuple(snapshot.candidate_ids[position] for position in positions)
    return Selection(ids, acquisition, seed, budget, snapshot.sha256, canonical_bytes(trace).decode())

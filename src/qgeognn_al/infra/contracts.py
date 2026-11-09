"""Canonical audit serialization without scientific or provider dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Sequence


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sample_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an ordered sequence of sample IDs")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def budget_value(value: int, candidate_count: int) -> int:
    if type(value) is not int or not 1 <= value <= candidate_count:
        raise ValueError("budget must be an integer in [1, candidate_count]")
    return value


@dataclass(frozen=True)
class BaselineProvenance:
    """Identity of an already frozen predictor and authorized input snapshot.

    These are supplied by the workflow's snapshot owner. No raw model,
    label-table path, provider handle or mutable training object is accepted.
    Hashes identify bytes; they do not prove independent authorization.
    """

    code_commit: str
    dataset_hash: str
    split_hash: str
    model_checkpoint_hash: str
    protocol_hash: str
    source_kind: str = "synthetic"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.code_commit):
            raise ValueError("code_commit must be a full Git commit SHA")
        for name in ("dataset_hash", "split_hash", "model_checkpoint_hash", "protocol_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        if self.source_kind not in ("synthetic", "authorized_snapshot"):
            raise ValueError("source_kind must explicitly identify synthetic or authorized_snapshot")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

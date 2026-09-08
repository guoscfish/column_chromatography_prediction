"""Small, side-effect-free guards for matched transfer protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..artifacts import sha256_file


def verify_source_checkpoint(path: Path, expected_sha256: str) -> str:
    """Verify and return the SHA256 of a frozen source checkpoint."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = sha256_file(checkpoint)
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    if observed != expected:
        raise RuntimeError(f"source checkpoint hash mismatch: expected {expected}, observed {observed}")
    return observed


def validate_label_roles(
    gradient_train: Iterable[str],
    validation: Iterable[str],
    test: Iterable[str],
    *,
    planned_budget: int | None = None,
) -> dict[str, object]:
    """Validate disjoint target roles and return a compact accounting record."""

    roles = {
        "gradient_train": tuple(str(value) for value in gradient_train),
        "validation": tuple(str(value) for value in validation),
        "test": tuple(str(value) for value in test),
    }
    if any(len(set(values)) != len(values) for values in roles.values()):
        raise ValueError("duplicate sample IDs within a label role")
    names = tuple(roles)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = set(roles[first]) & set(roles[second])
            if overlap:
                raise ValueError(f"label roles overlap: {first}/{second}: {sorted(overlap)[:3]}")
    actual = len(roles["gradient_train"]) + len(roles["validation"])
    if planned_budget is not None and int(planned_budget) != actual:
        raise ValueError(f"planned budget {planned_budget} != actual revealed rows {actual}")
    return {
        **{name: list(values) for name, values in roles.items()},
        "planned_budget": int(planned_budget) if planned_budget is not None else None,
        "actual_budget": actual,
        "test_rows_used_for_fit": 0,
        "test_rows_used_for_checkpoint_selection": 0,
        "target_rows_used_for_preprocessing_fit": 0,
    }

def assert_fit_ids_authorized(
    ids: Sequence[str],
    roles: Mapping[str, Sequence[str]],
    *,
    allow_validation: bool = True,
) -> None:
    """Reject unknown or test IDs before a fit/selection operation."""

    test = {str(value) for value in roles.get("test", ())}
    allowed = {str(value) for value in roles.get("gradient_train", ())}
    if allow_validation:
        allowed.update(str(value) for value in roles.get("validation", ()))
    selected = {str(value) for value in ids}
    leaked = selected & test
    if leaked:
        raise ValueError(f"test IDs are not authorized for fitting: {sorted(leaked)[:3]}")
    unknown = selected - allowed
    if unknown:
        raise ValueError(f"fit IDs are outside authorized roles: {sorted(unknown)[:3]}")


def budget_accounting(
    gradient_train: Sequence[str],
    validation: Sequence[str],
    test: Sequence[str],
    planned_budget: int,
) -> dict[str, int | bool]:
    """Return a machine-readable budget audit after role validation."""

    record = validate_label_roles(gradient_train, validation, test, planned_budget=planned_budget)
    return {
        "planned_budget": int(planned_budget),
        "actual_budget": int(record["actual_budget"]),
        "gradient_train_rows": len(gradient_train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "budget_matches": True,
        "test_rows_used_for_fit": 0,
        "test_rows_used_for_checkpoint_selection": 0,
    }

"""Contracts for fixed held-out populations and nested filtered budgets."""
import numpy as np
import pandas as pd
import pytest

from scripts.studies import run_filtered_row_learning_curve as curve


def example():
    ids = [f"r{i}" for i in range(14)]
    kept = ids[:12]
    block = pd.DataFrame({"column": ["8g"] * 12, "protocol": ["row"] * 12,
                          "outer_seed": [7] * 12, "sample_id": kept,
                          "role": ["test", "test", "validation"] + ["gradient_train"] * 9})
    return block, {"8g": ids}


def test_filtered_prefix_preserves_held_out_sets_counts_and_nesting():
    block, originals = example()
    schedule, accounting = curve.nested_schedule(block, originals, budgets=(3, 6), columns=("8g",), seeds=(7,))
    train_sets = []
    for budget in (3, 6):
        part = schedule.loc[schedule.budget.eq(budget)]
        assert set(part.loc[part.role.eq("test"), "sample_id"]) == {"r0", "r1"}
        assert set(part.loc[part.role.eq("validation"), "sample_id"]) == {"r2"}
        selected = set(part.loc[part.role.eq("gradient_train"), "sample_id"])
        assert len(selected) == budget - 1
        assert selected <= set(block.loc[block.role.eq("gradient_train"), "sample_id"])
        train_sets.append(selected)
    assert train_sets[0] < train_sets[1]
    assert accounting.actual_budget.tolist() == [3, 6]
    assert accounting.test_ids_hash.nunique() == accounting.validation_ids_hash.nunique() == 1
    permutation = np.random.default_rng(7).permutation(14)
    expected = [originals["8g"][i] for i in permutation if 3 <= i < 12]
    assert train_sets[0] == set(expected[:2]) and train_sets[1] == set(expected[:5])


def test_prefix_is_independent_of_full_manifest_row_order():
    block, originals = example()
    a, _ = curve.nested_schedule(block, originals, budgets=(3, 6), columns=("8g",), seeds=(7,))
    b, _ = curve.nested_schedule(block.iloc[::-1], originals, budgets=(3, 6), columns=("8g",), seeds=(7,))
    keys = ["sample_id", "budget", "role", "eligible_train_rank"]
    pd.testing.assert_frame_equal(a[keys].sort_values(keys).reset_index(drop=True),
                                  b[keys].sort_values(keys).reset_index(drop=True))


def test_budget_at_or_above_full_is_rejected():
    block, originals = example()
    with pytest.raises(RuntimeError, match="below FULL"):
        curve.nested_schedule(block, originals, budgets=(3, 10), columns=("8g",), seeds=(7,))


def test_test_labels_are_not_parsed_without_global_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "STUDY", tmp_path)
    parsed = []
    monkeypatch.setattr(curve.full.parent, "_read_authorized_truth", lambda *args: parsed.append(args))
    with pytest.raises(RuntimeError, match="not globally frozen"):
        curve.truth_for_role("8g", curve.SEEDS[0], 100, "test")
    assert parsed == []

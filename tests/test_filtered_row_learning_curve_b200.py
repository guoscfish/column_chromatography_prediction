"""B200 uses the same held-out identities and an intermediate nested prefix."""
import pandas as pd
import pytest

from scripts.studies import run_filtered_row_learning_curve_b200 as supplement


def test_b200_is_strictly_nested_between_existing_budgets():
    original = [f"r{i}" for i in range(350)]
    block = pd.DataFrame({"column": ["25g"] * 330, "protocol": ["row"] * 330,
                          "outer_seed": [7] * 330, "sample_id": original[:330],
                          "role": ["test"] * 2 + ["validation"] * 6 + ["gradient_train"] * 322})
    schedule, accounting = supplement.nested_schedule(
        block, {"25g": original}, budgets=(100, 200, 300), columns=("25g",), seeds=(7,))
    train_sets = []
    for budget in (100, 200, 300):
        ctx = schedule.loc[schedule.budget.eq(budget)]
        assert set(ctx.loc[ctx.role.eq("test"), "sample_id"]) == set(original[:2])
        assert set(ctx.loc[ctx.role.eq("validation"), "sample_id"]) == set(original[2:8])
        train = set(ctx.loc[ctx.role.eq("gradient_train"), "sample_id"])
        assert len(train) == budget - 6
        train_sets.append(train)
    assert train_sets[0] < train_sets[1] < train_sets[2]
    assert accounting.actual_budget.tolist() == [100, 200, 300]


def test_b200_test_labels_stay_blocked_before_global_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(supplement, "STUDY", tmp_path)
    reads = []
    monkeypatch.setattr(supplement.full.parent, "_read_authorized_truth", lambda *args: reads.append(args))
    with pytest.raises(RuntimeError, match="not globally frozen"):
        supplement.truth_for_role("25g", supplement.SEEDS[0], 200, "test")
    assert reads == []

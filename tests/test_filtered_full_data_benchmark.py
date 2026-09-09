"""Contracts specific to the filtered full-data transfer pilot."""

from __future__ import annotations

import pandas as pd
import pytest

from src.qgeognn_al.transfer.full_data import build_full_data_schedule, filter_operational_domain


def _data() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e", "f"],
        "canonical_smiles": ["A", "A", "B", "C", "D", "E"],
        "V1_ml": [10.0, 61.0, 20.0, 30.0, 40.0, 50.0],
        "V2_ml": [20.0, 30.0, 21.0, 50.0, 60.0, 70.0],
    })


def _parent() -> pd.DataFrame:
    return pd.DataFrame({
        "column": ["25g"] * 6,
        "protocol": ["compound"] * 6,
        "outer_seed": [7] * 6,
        "planned_budget": [100] * 6,
        "sample_id": ["a", "b", "c", "d", "e", "f"],
        "canonical_smiles": ["A", "A", "B", "C", "D", "E"],
        "role": ["gradient_train", "pool", "validation", "test", "test", "pool"],
    })


def test_operational_filter_is_declared_and_keeps_an_audit() -> None:
    filtered, audit = filter_operational_domain(_data(), column="25g", v1_limit_ml=60, v2_limit_ml=120)
    assert filtered.sample_id.tolist() == ["a", "c", "d", "e", "f"]
    assert audit.loc[audit.sample_id.eq("b"), "filter_reason"].item() == "V1_above_limit"
    assert audit.loc[audit.sample_id.eq("c"), "filter_reason"].item() == "within_legacy_operational_domain"


def test_full_data_schedule_exposes_pool_only_to_train_and_keeps_compounds_isolated() -> None:
    filtered, _ = filter_operational_domain(_data(), column="25g", v1_limit_ml=60, v2_limit_ml=120)
    schedule, accounting = build_full_data_schedule(
        _parent(), {"25g": filtered}, columns=("25g",), protocols=("compound",), seeds=(7,)
    )
    assert set(schedule.loc[schedule.role.eq("gradient_train"), "sample_id"]) == {"a", "f"}
    assert set(schedule.loc[schedule.role.eq("test"), "sample_id"]) == {"d", "e"}
    assert accounting.loc[0, "training_label_count"] == 2
    assert accounting.loc[0, "compound_leakage_count"] == 0


def test_compound_schedule_rejects_parent_leakage() -> None:
    data = _data()
    data.loc[data.sample_id.eq("b"), "V1_ml"] = 11.0
    filtered, _ = filter_operational_domain(data, column="25g", v1_limit_ml=60, v2_limit_ml=120)
    parent = _parent()
    parent.loc[parent.sample_id.eq("a"), "role"] = "test"
    parent.loc[parent.sample_id.eq("b"), "role"] = "pool"
    with pytest.raises(ValueError, match="compound leakage"):
        build_full_data_schedule(parent, {"25g": filtered}, columns=("25g",), protocols=("compound",), seeds=(7,))

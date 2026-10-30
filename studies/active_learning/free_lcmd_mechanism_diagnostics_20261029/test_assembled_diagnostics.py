"""Small integration tests for evidence alignment and explicit missingness."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("assemble_diagnostics", HERE / "assemble_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_join_rejects_duplicate_missing_and_conflicting_evidence():
    left = pd.DataFrame({"seed": [157], "method": ["lcmd"], "round": [0], "labels_before": [333]})
    with pytest.raises(ValueError, match="Duplicate"):
        module.checked_merge(pd.concat([left, left]), left)
    with pytest.raises(ValueError, match="Unmatched"):
        module.checked_merge(left, left.assign(round=1))
    with pytest.raises(ValueError, match="Conflicting"):
        module.checked_merge(left, left.assign(labels_before=365))


def test_full_selection_alignment_and_lagged_validation():
    p = pd.read_csv(HERE / "selection_profiles.csv")
    assert len(p) == 360
    assert p.groupby(["seed", "method"]).size().eq(36).all()
    assert not p.duplicated(module.KEY).any()
    assert np.all(p.labels_before == 333 + 32 * p["round"])
    assert np.all(p.labels_after == p.labels_before + 32)
    assert np.allclose(p.validation_nrmse_before - p.validation_nrmse_after, p.next_fit_validation_gain)
    assert p.batch_size.sum() == 11520


def test_structural_missingness_is_not_zero_imputation():
    p = pd.read_csv(HERE / "selection_profiles.csv")
    lcmd = p[p.method == "lcmd"]
    free = p[p.method == module.FREE]
    assert lcmd.cw_selected_distance_percentile_mean.isna().all()
    assert lcmd.high8_same_molecule_enrichment.isna().all()
    assert free.cw_selected_distance_percentile_mean.notna().all()
    assert free[free["round"] == 0].high8_same_molecule_enrichment.isna().all()
    assert free[free["round"] > 0].high8_same_molecule_enrichment.notna().all()


def test_selection_record_ids_are_unique_within_each_trajectory():
    p = pd.read_csv(HERE / "selection_records.csv")
    assert len(p) == 11520
    id_column = "sample_id" if "sample_id" in p else "candidate_id"
    assert not p.duplicated(["seed", "method", id_column]).any()

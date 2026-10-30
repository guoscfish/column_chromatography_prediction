"""Cross-module scientific-contract tests; no model or acquisition is run."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hybrid_assembly", HERE / "assemble_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frames():
    return pd.read_csv(HERE / "behavior_profiles.csv"), pd.read_csv(HERE / "geometry_profiles.csv")


def test_complete_source_counts_and_coverage_identity():
    b, g = frames()
    module.validate_profiles(module.merge_profiles(b, g))


@pytest.mark.parametrize("field,value,error", [("round", 17, "exactly 0"),
    ("labels_after", 999, "after budget"), ("stage", "653-813", "stage does not")])
def test_wrong_round_after_budget_or_stage_rejected(field, value, error):
    b, g = frames()
    merged = module.merge_profiles(b, g)
    merged.loc[0, field] = value
    with pytest.raises(ValueError, match=error):
        module.validate_profiles(merged)


def test_duplicate_missing_and_conflicting_metadata_rejected():
    b, g = frames()
    with pytest.raises(ValueError, match="duplicate"):
        module.merge_profiles(pd.concat([b, b.iloc[:1]]), g)
    with pytest.raises(ValueError, match="key sets"):
        module.merge_profiles(b.iloc[1:], g)
    g.loc[0, "labels_before"] += 32
    with pytest.raises(ValueError, match="conflicting"):
        module.merge_profiles(b, g)


def test_cap_candidate_counts_independent_of_validation():
    b, g = frames()
    frame = module.merge_profiles(b, g)
    first = module.cap_sensitivity(frame)
    frame = frame.copy()
    frame["validation_nrmse"] = -999999.0
    pd.testing.assert_frame_equal(first, module.cap_sensitivity(frame))
    assert set(first.candidate_K) == {8, 12}
    assert set(first.status) == {"DESIGN_CANDIDATE_ONLY_NOT_PREREGISTERED"}


def test_performance_attaches_to_whole_hybrid_not_each_source():
    b, g = frames()
    result = module.add_performance(module.merge_profiles(b, g), pd.read_csv(HERE / "validation_rounds.csv"))
    h = result[result.source.str.startswith("hybrid_")]
    assert h.groupby(["seed", "round"]).validation_nrmse_after.nunique().eq(1).all()
    assert h[h.source.isin(["hybrid_cw16", "hybrid_llm16"])].performance_attribution.str.contains("no source-specific causal").all()


def test_no_selected_candidate_truth_columns_in_analysis_exports():
    for name in ("behavior_profiles.csv", "geometry_profiles.csv", "selection_point_flags.csv", "geometry_points.csv"):
        columns = pd.read_csv(HERE / name, nrows=0).columns
        assert not any(c.startswith(("true_", "test_", "validation_truth")) for c in columns)

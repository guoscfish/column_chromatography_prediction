import pandas as pd
import pytest

from src.qgeognn_al.diagnostics.oracle_utility import (
    binary_auroc,
    build_diagnostic_candidate_subset,
    compute_enrichment,
    compute_marginal_utility,
    source_reset_audit_rows,
    validate_candidate_fit_contract,
)


def pool_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(20)],
        "ensemble_score": range(20),
        "quantile_width": range(20, 0, -1),
        "coverage_score": [i % 7 for i in range(20)],
        "V1_ml": [999.0] * 20,
        "V2_ml": [-999.0] * 20,
    })


def test_utility_sign():
    assert compute_marginal_utility(2, 1) > 0
    assert compute_marginal_utility(1, 2) < 0


def test_candidate_subset_is_deterministic():
    first = build_diagnostic_candidate_subset(pool_frame(), random_n=12)
    second = build_diagnostic_candidate_subset(pool_frame(), random_n=12)
    assert first.equals(second)


def test_candidate_subset_does_not_read_or_return_truth_columns():
    original = pool_frame()
    changed = original.copy()
    changed[["V1_ml", "V2_ml"]] *= -12345
    first = build_diagnostic_candidate_subset(original, random_n=12)
    second = build_diagnostic_candidate_subset(changed, random_n=12)
    assert first.equals(second)
    assert not {"V1_ml", "V2_ml"} & set(first.columns)


def test_no_duplicate_candidate_training():
    subset = build_diagnostic_candidate_subset(pool_frame(), random_n=12)
    assert subset.sample_id.is_unique
    duplicate_pool = pd.concat([pool_frame(), pool_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_diagnostic_candidate_subset(duplicate_pool)


def test_fixed_validation_preserved_and_candidate_only_addition():
    checks = validate_candidate_fit_contract(
        ["train1", "train2", "valid1", "valid2"], ["valid1", "valid2"],
        ["train1", "train2", "valid1", "valid2", "candidate"],
        "candidate", ["candidate", "other"], ["test"],
    )
    assert all(checks.values())
    with pytest.raises(ValueError, match="contract"):
        validate_candidate_fit_contract(
            ["train1", "valid1", "valid2"], ["valid1", "valid2"],
            ["train1", "valid1", "candidate"], "candidate", ["candidate"], ["test"]
        )


def test_source_reset_audit_schema():
    rows = source_reset_audit_rows(
        [{"member_seed": 42, "init_source_sha256": "abc", "checkpoint_sha256": "fit"}],
        {42: "abc"}, "candidate", "s1",
    )
    assert set(rows[0]) == {
        "phase", "candidate_id", "member_seed", "init_source_sha256",
        "expected_source_sha256", "source_reset_pass", "checkpoint_sha256",
    }
    assert rows[0]["source_reset_pass"] is True


def test_representative_and_challenge_subset_tagging():
    subset = build_diagnostic_candidate_subset(
        pool_frame(), random_n=5, top_n=2,
        challenge_ids={"ensemble_top8": ["s19", "s18"],
                       "qwidth_top8": ["s0", "s1"],
                       "coverage_top8": ["s6", "s13"]},
    )
    assert {"random_sample", "ensemble_top8", "qwidth_top8", "coverage_top8"} <= set(subset)
    assert subset["random_sample"].sum() == 5
    assert subset["ensemble_top8"].sum() == 2


def test_auroc_and_enrichment_known_case():
    scores, high = [4, 3, 2, 1], [True, True, False, False]
    assert binary_auroc(scores, high) == 1.0
    assert compute_enrichment(scores, high, 0.5)["enrichment"] == 2.0

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.diagnostics.oracle_reliability import (
    compare_d45_d46,
    paired_test_bootstrap,
    paired_utility,
    pairwise_ranking_stability,
    select_reliability_candidates,
    summarize_candidate_reliability,
    variance_decomposition,
)


def d45_frame() -> pd.DataFrame:
    utility = np.linspace(-0.2, 0.2, 30)
    return pd.DataFrame({
        "sample_id": [f"s{i:02d}" for i in range(30)],
        "random_sample": True,
        "oracle_utility_V1": utility,
        "oracle_utility_V2": utility / 2,
        "oracle_utility_combined": utility,
        "ensemble_score": np.arange(30)[::-1],
        "quantile_width": np.arange(30),
        "coverage_score": 1.0,
        "ensemble_top8": False, "qwidth_top8": False, "coverage_top8": False,
    })


def utility_panel() -> pd.DataFrame:
    rows = []
    for sample, stratum, base in (("a", "high_positive", 1.0), ("b", "near_zero", 0.0), ("c", "strongly_negative", -1.0)):
        for seed, offset in zip((4601, 4602, 4603), (-0.1, 0.0, 0.1)):
            rows.append({"sample_id": sample, "stratum": stratum, "repetition_seed": seed,
                         "oracle_utility_combined": base + offset, "random_sample": True})
    return pd.DataFrame(rows)


def test_candidate_stratified_selection_deterministic():
    assert select_reliability_candidates(d45_frame()).equals(
        select_reliability_candidates(d45_frame())
    )


def test_selection_has_no_duplicates_and_exact_counts():
    selected = select_reliability_candidates(d45_frame())
    assert selected.sample_id.is_unique


def test_selection_has_exact_strata_counts():
    selected = select_reliability_candidates(d45_frame())
    assert selected.stratum.value_counts().to_dict() == {
        "high_positive": 6, "near_zero": 6, "strongly_negative": 6
    }


def test_paired_baseline_candidate_utility():
    assert paired_utility({"NRMSE": 0.8}, {"NRMSE": 0.6}) == pytest.approx(0.2)


def test_repetition_seeds_differ_and_summary_uses_all():
    seeds = [4601, 4602, 4603]
    assert len(seeds) == len(set(seeds))
    summary = summarize_candidate_reliability(utility_panel(), seeds)
    assert len(summary) == 3
    assert summary.loc[summary.sample_id == "a", "positive_repetitions"].item() == 3


def test_source_checkpoint_hashes_are_fixed():
    manifest = pd.read_csv("experiments/e4_a2a_low_budget_formal/source_checkpoint_manifest.csv")
    assert manifest.member_seed.tolist() == [42, 525, 1101]
    assert manifest.checkpoint_sha256.str.len().eq(64).all()


def test_fixed_validation_contract_helper_remains_available():
    from src.qgeognn_al.diagnostics.oracle_utility import validate_candidate_fit_contract
    checks = validate_candidate_fit_contract(
        ["t", "v1", "v2"], ["v1", "v2"], ["t", "v1", "v2", "c"],
        "c", ["c"], ["z"],
    )
    assert checks["fixed_validation_preserved"]


def test_paired_bootstrap_uses_same_sampled_indices():
    truth = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    baseline = truth + 2.0
    candidate = truth + 1.0
    indices = np.array([[0, 0, 1], [2, 1, 2]])
    result = paired_test_bootstrap(
        truth, baseline, candidate, {"V1": 1.0, "V2": 1.0},
        sampled_indices=indices, return_draws=True,
    )
    assert np.array_equal(result["sampled_indices"], indices)
    assert np.all(np.asarray(result["draws"]) > 0)


def test_variance_decomposition_known_example():
    panel = utility_panel()
    result = variance_decomposition(panel)
    assert result["between_candidate_variance"] > result["within_candidate_variance"]
    assert 0.9 < result["reliability_ratio"] <= 1.0
    assert result["icc"] == result["reliability_ratio"]


def test_ranking_stability():
    ranking = pairwise_ranking_stability(utility_panel(), [4601, 4602, 4603])
    assert len(ranking) == 9
    assert np.allclose(ranking.spearman, 1.0)


def test_d45_d46_consistency_calculation():
    reliability = pd.DataFrame({"sample_id": ["a", "b", "c"], "mean_utility": [1.0, 0.0, -1.0]})
    subset = pd.DataFrame({"sample_id": ["a", "b", "c"], "stratum": ["high", "near", "negative"],
                           "D45_oracle_utility_combined": [0.9, 0.0, -0.8]})
    comparison = compare_d45_d46(reliability, subset)
    assert comparison.D45_D46_spearman.iloc[0] == 1.0
    assert comparison.D45_D46_sign_agreement.iloc[0] == 1.0


def test_selection_truth_use_is_confined_to_declared_d45_utility():
    original = d45_frame()
    changed_scores = original.copy()
    changed_scores[["ensemble_score", "quantile_width", "coverage_score"]] *= -999
    assert select_reliability_candidates(original).sample_id.tolist() == \
        select_reliability_candidates(changed_scores).sample_id.tolist()

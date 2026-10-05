# Scientist V3.2 Round 0 analysis: L333 to L365

Completed seed157 / Free-LLM32 round 0. The later user instruction explicitly extends the formal trajectory to six acquisitions ending at L525; this report covers the first acquisition only.

Source commit: `d277aa7aed8f47403db66dd19db33b61bdadc54c`. Protocol SHA-256: `ecd8f964d1bdb5cec26961ebdd8d10776a4599e81db492c0d727eba44ed99673`. Model/provider: gpt-6-sol / https://token4research.cn, strict Responses JSON Schema with streaming.

The successful trajectory used 17 logical calls / 17 HTTP attempts, 24 accepted scientific queries, zero invalid queries, three response-validation repairs (two unknown shortlist IDs and one response containing multiple JSON objects), zero normalized aliases and zero JSON recoveries. One FINALIZE_ONLY call succeeded after the full query budget, using 200 distinct viewed candidates.

Prior failure costs are separate and must not be hidden: the non-streaming attempt used 15 logical calls / 18 HTTP attempts / 24 accepted queries; the first streaming attempt used 4 logical calls / 4 HTTP attempts / 6 accepted queries. There was also one failed diagnostic replay and three data-free capability probes. Neither failed attempt froze a batch or revealed new labels. Their complete records and hash manifests are retained.

## Scientific results

| Metric | Value |
|---|---:|
| L333 fixed-validation combined NRMSE | 0.853886566 |
| L365 fixed-validation combined NRMSE | 0.812055654 |
| L365 minus L333 | -0.041830912 |
| Relative reduction | 4.899% |
| Selected-point V1 acquisition-time MAE / RMSE, mL | 4.968324 / 9.342509 |
| Selected-point V2 acquisition-time MAE / RMSE, mL | 9.771231 / 17.628167 |

These selected-point errors are historical measurement-time residuals, not evidence of errors persisting after retraining. The complete normalized-error distribution and final 32 IDs appear below. Fixed-validation scores are checkpoint-selection scores, not an independent test or a matched comparison of acquisition strategies.

## Interpretation

1. Interaction waste decreased but was not eliminated. V3.1 round 0 had six invalid queries and five response errors, leaving 18 successful queries out of 24 charged attempts. This round had zero invalid queries and three response errors, with all 24 scientific queries accepted. The exhausted-budget transition actually reached a successful selection. Structured output still returned multiple objects once; the host rejected that response instead of choosing an object. Network failures in earlier attempts remain visible in total cost accounting.
2. Concentration is much lower in this batch: 32 points cover 28 molecules (25 singletons, two doublets, one triplet). The largest molecule occupies 9.375%, compared with eight points from one molecule in V3.1 round 0. Hypothesis-linked counts are 4, 6 and 3, plus 19 unlinked points. No diagnostic imposed a quota or rejected a batch.
3. The batch combines targeted hypothesis experiments and broader exploration: role metadata labels 12 targeted experiments and 20 exploration/coverage points. Hypotheses concern basic nitrogen, H-bond donation and loading response of a multifunctional ester. Twenty-one selected points are molecules novel relative to L333. Because L333 has no previous acquisition residuals, these are primarily hypothesis tests, not demonstrated repairs of historical model failures. Role metadata alone does not establish learning utility.
4. Continuing the preregistered six-round observation is reasonable: the batch is diverse and the first validation change is favorable. A single seed and checkpoint-validation curve cannot establish superiority; paired strategies, more seeds and independent evaluation would be needed. Continued acquisition follows explicit user authorization, not adaptive use of validation metrics by the selector.
5. The updated endpoint is exactly L525 after six acquisitions. Hybrid 16+16, other seeds and acquisitions beyond L525 remain unregistered.

Same initialization and fixed validation checks passed. Test truth access count is zero. The LLM received no validation metrics.

## Full deterministic result

```json
{
  "source_commit": "d277aa7aed8f47403db66dd19db33b61bdadc54c",
  "protocol_sha256": "ecd8f964d1bdb5cec26961ebdd8d10776a4599e81db492c0d727eba44ed99673",
  "model": "gpt-6-sol",
  "provider": "https://token4research.cn",
  "seed": 157,
  "method": "free_llm32_scientist_v3_2",
  "round": 0,
  "labels_before": 333,
  "labels_after": 365,
  "accepted_scientific_queries": 24,
  "distinct_viewed_candidates": 200,
  "finalization_mode_calls": 1,
  "invalid_query_attempts": 0,
  "total_llm_calls": 17,
  "validation_repairs_used": 3,
  "transport_calls": 17,
  "normalized_query_aliases": 0,
  "json_recoveries": 0,
  "final_32_ids": [
    "33ffc69a253a89f15995",
    "e7bf6bd74f8498f1cb11",
    "a7bc3fb68bdecbbe086f",
    "3c0e2ce9feb019d1dd30",
    "ca467c94f14b2219503e",
    "4908bb1bf155952cd305",
    "8b51e50fec9dd89974f1",
    "5edc83d62a273a5ae9b2",
    "f1dafc187f7b1ec15297",
    "4db6b065867770d9dbd9",
    "e586f782dfd7bb09f66f",
    "aa614da3137ee8938fc9",
    "4d0055bdff0f507c778b",
    "a775926fce57b3bf9ec1",
    "30ae74055d1ffabed86a",
    "304fb0fdb486ae4b7a07",
    "570497bf792f134f2223",
    "4ca9a1703e2a0e780967",
    "aa43fe84c0d8c825fe58",
    "7150b292e8a20dd06f1c",
    "92f7fe9d8581717a9949",
    "e1888b5ca18659a8d8eb",
    "13cfc8ad0582fcbd26bc",
    "72ab072149ea3aa635af",
    "67e9e23958a468891c07",
    "a00b9b271e1fe3ae2351",
    "02c053f1544310b38b17",
    "08155ffdcb0d27576050",
    "2029b30e714e5ff498bf",
    "4659b6752f23c0b85f08",
    "74ce463c8423fbd22dc4",
    "2dbfdf9c2d5defd09f7a"
  ],
  "batch_diagnostic": {
    "broader_exploration_count": 20,
    "enforced_quotas": null,
    "hypothesis_linked_counts": {
      "H0001": 4,
      "H0002": 6,
      "H0003": 3,
      "unlinked": 19
    },
    "largest_same_molecule_count": 3,
    "largest_same_molecule_fraction": 0.09375,
    "max_similarity_to_observed": 1.0,
    "mean_pairwise_structural_similarity": 0.1496946180118501,
    "mean_similarity_to_observed": 0.6671374761242157,
    "molecule_count_histogram": {
      "1": 25,
      "2": 2,
      "3": 1
    },
    "molecule_counts": {
      "Brc1ccc(Br)c2ccccc12": 1,
      "Brc1ccccc1-c1ccccc1": 1,
      "C=C(C)C(=O)OCCOC(=O)CC(C)=O": 3,
      "C=CC(=O)OCCN(CC)CC": 1,
      "CC(=O)CC(=O)c1ccccc1": 1,
      "CC(=O)c1c(C)cc(C)cc1C": 1,
      "CC(=O)c1ccc(C)cc1C": 1,
      "CC(=O)c1ccc([N+](=O)[O-])cc1": 1,
      "CC(=O)c1ccc2ccccc2c1": 1,
      "CC(C)(C)C(=O)CC#N": 1,
      "CC(C)(C)c1cc(C=O)cc(C(C)(C)C)c1": 1,
      "CC(O)CCc1ccccc1": 1,
      "CCC(=O)c1cccnc1": 1,
      "CCOC(=O)/C=C/c1ccccc1": 1,
      "CCOC(=O)CC(=O)OCC": 1,
      "CCOC(=O)c1ccc(F)cc1": 1,
      "COC(=O)c1ccc(I)cc1": 1,
      "COc1cc(F)cc(Br)c1": 1,
      "COc1ccc(C=O)c(O)c1": 2,
      "COc1ccc(CCO)cc1": 1,
      "CSc1ccc(C(C)=O)cc1": 1,
      "Cc1ccc(CO)cc1": 1,
      "Clc1ccccc1I": 1,
      "FC(F)(F)Oc1ccc(I)c(Br)c1": 1,
      "O=C(c1ccccc1)c1ccc([N+](=O)[O-])cc1": 1,
      "O=Cc1ccc(-c2ccccc2)cc1": 1,
      "O=Cc1ccc(Br)cc1O": 1,
      "c1ccc2ncccc2c1": 2
    },
    "molecule_novelty_count": 21,
    "other_count": 0,
    "role_counts_are_metadata_only": true,
    "scientific_role_counts": {
      "condition_curve": 2,
      "coverage": 7,
      "hypothesis_test": 8,
      "matched_control": 2,
      "structural_exploration": 13
    },
    "selected_CW_percentiles": {
      "count": 32,
      "max": 0.99883,
      "median": 0.76343,
      "min": 0.08976,
      "q10": 0.23333399999999999,
      "q25": 0.3601075,
      "q75": 0.885965,
      "q90": 0.973637
    },
    "selected_count": 32,
    "targeted_follow_up_count": 12,
    "unique_molecules": 28
  },
  "selected_point_acquisition_time_errors": {
    "V1": {
      "MAE_ml": 4.968324221670628,
      "RMSE_ml": 9.342508748717925
    },
    "V2": {
      "MAE_ml": 9.77123111486435,
      "RMSE_ml": 17.628166510509548
    }
  },
  "combined_normalized_selected_point_error_distribution": {
    "count": 32,
    "min": 0.014432052104080752,
    "q10": 0.050325518708627164,
    "q25": 0.13335562528289868,
    "median": 0.21000423256472756,
    "q75": 0.9195186330875448,
    "q90": 2.638338173480887,
    "max": 4.993503019919149
  },
  "previous_fixed_validation_combined_NRMSE": 0.8538865655827187,
  "current_fixed_validation_combined_NRMSE": 0.8120556538617575,
  "delta_current_minus_previous": -0.041830911720961206,
  "relative_reduction_percent": 4.898883927564136,
  "same_initialization": true,
  "fixed_validation": true,
  "test_truth_access_count": 0,
  "selector_received_validation_metrics": false,
  "registered_final_budget": 525,
  "limitation": "One seed and one acquisition; checkpoint-validation scores are not an independent test or matched strategy comparison."
}
```

Scientific interpretation will be included in the final analysis. No test truth was accessed.

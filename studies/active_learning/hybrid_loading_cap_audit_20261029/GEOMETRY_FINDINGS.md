# Frozen geometry findings

Scope: Scientist V2, seeds 157 / 6101, acquisition rounds 0–14 (333→813 labels). All banks are method-local frozen inputs. No training, inference, selection, gradient reconstruction, or unseen/test target-table read occurred.

## Main observations

- CW16 consistently selects high-distance points (mean CW percentiles 0.9408 / 0.9573 for seeds 157 / 6101) and supplies most Hybrid geometric coverage. The LLM16 component has lower mean percentiles (0.6804 / 0.6087), with a positive extra coverage gain after CW16.
- LLM points sharing any molecule with the current CW16 are only 9.58% / 7.92%. This does not support a claim of widespread direct molecular overlap between the two halves.
- Whole-Hybrid relative coverage exceeds Free in both seeds over 333–813, yet the late 653–813 interval reverses in seed 6101 (CW-space common remaining-U drop 0.0351 for Hybrid vs 0.0396 for Free). A uniform superiority claim is unsupported.
- The late Hybrid LLM16 becomes local in gradient space, especially seed 6101: CW percentile 0.3574 and marginal after-CW relative coverage gain 0.0023 at 653–813. Geometry therefore supports a changing division of labor, not a claim that Hybrid LLM consistently explores more than Free.
- Seed 157: CW16 receives 75.1% of Hybrid CW coverage gain under the symmetric mean of its solo and after-LLM marginal contributions. LLM16 retains 91.8% of its solo gain after CW16. This is only a within-bank distance decomposition.
- Seed 6101: CW16 receives 74.4% of Hybrid CW coverage gain under the symmetric mean of its solo and after-LLM marginal contributions. LLM16 retains 92.3% of its solo gain after CW16. This is only a within-bank distance decomposition.

## Per-seed mean of round metrics

| seed | source | cw_selected_distance_percentile_mean | raw_selected_distance_percentile_mean | cw_common_remaining_U_coverage_distance_relative_drop | raw_common_remaining_U_coverage_distance_relative_drop |
| --- | --- | --- | --- | --- | --- |
| 157 | cw_lcmd32 | 0.9272 | NA | 0.0567 | NA |
| 157 | free32 | 0.6312 | 0.6295 | 0.0321 | 0.0334 |
| 157 | hybrid_all32 | 0.8106 | 0.8065 | 0.0449 | 0.0462 |
| 157 | hybrid_cw16 | 0.9408 | 0.9321 | 0.0342 | 0.0352 |
| 157 | hybrid_llm16 | 0.6804 | 0.6808 | 0.0116 | 0.0120 |
| 157 | raw_lcmd32 | NA | 0.9319 | NA | 0.0579 |
| 6101 | cw_lcmd32 | 0.9460 | NA | 0.0618 | NA |
| 6101 | free32 | 0.6646 | 0.6628 | 0.0364 | 0.0369 |
| 6101 | hybrid_all32 | 0.7830 | 0.7814 | 0.0456 | 0.0475 |
| 6101 | hybrid_cw16 | 0.9573 | 0.9512 | 0.0344 | 0.0361 |
| 6101 | hybrid_llm16 | 0.6087 | 0.6117 | 0.0121 | 0.0126 |
| 6101 | raw_lcmd32 | NA | 0.9467 | NA | 0.0653 |

Values compare ranks or relative within-state coverage. They do not put different models in a common geometry; cross-method comparisons after round 0 remain matched-budget observations. Each seed has 15 dependent rounds, not 15 independent replicates.

## Same-state Hybrid attribution

| seed | source | cw_common_remaining_U_coverage_distance_relative_drop | cw_common_remaining_U_marginal_after_other_relative_to_original_coverage | raw_common_remaining_U_coverage_distance_relative_drop | raw_common_remaining_U_marginal_after_other_relative_to_original_coverage |
| --- | --- | --- | --- | --- | --- |
| 157 | hybrid_cw16 | 0.0342 | 0.0333 | 0.0352 | 0.0342 |
| 157 | hybrid_llm16 | 0.0116 | 0.0107 | 0.0120 | 0.0110 |
| 6101 | hybrid_cw16 | 0.0344 | 0.0335 | 0.0361 | 0.0350 |
| 6101 | hybrid_llm16 | 0.0121 | 0.0112 | 0.0126 | 0.0114 |

“Alone” adds a source to L_t. “After other” adds it after the other source on the same frozen bank. Both use U_t minus all32 as the same denominator and the original L_t coverage as the same normalizer. This is a geometric decomposition, not a model-performance causal effect. Solo gains overlap and must not be added; one solo plus the other marginal equals all32.

## Stage detail

| seed | stage | source | cw_common_remaining_U_coverage_distance_relative_drop | cw_common_remaining_U_marginal_after_other_relative_to_original_coverage | raw_common_remaining_U_coverage_distance_relative_drop | raw_common_remaining_U_marginal_after_other_relative_to_original_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 157 | 333-525 | hybrid_cw16 | 0.0379 | 0.0360 | 0.0394 | 0.0372 |
| 157 | 333-525 | hybrid_llm16 | 0.0145 | 0.0125 | 0.0148 | 0.0126 |
| 157 | 525-653 | hybrid_cw16 | 0.0324 | 0.0323 | 0.0330 | 0.0330 |
| 157 | 525-653 | hybrid_llm16 | 0.0121 | 0.0120 | 0.0131 | 0.0131 |
| 157 | 653-813 | hybrid_cw16 | 0.0313 | 0.0308 | 0.0320 | 0.0315 |
| 157 | 653-813 | hybrid_llm16 | 0.0079 | 0.0074 | 0.0078 | 0.0072 |
| 6101 | 333-525 | hybrid_cw16 | 0.0348 | 0.0337 | 0.0371 | 0.0360 |
| 6101 | 333-525 | hybrid_llm16 | 0.0193 | 0.0183 | 0.0198 | 0.0186 |
| 6101 | 525-653 | hybrid_cw16 | 0.0359 | 0.0344 | 0.0379 | 0.0359 |
| 6101 | 525-653 | hybrid_llm16 | 0.0130 | 0.0115 | 0.0137 | 0.0117 |
| 6101 | 653-813 | hybrid_cw16 | 0.0328 | 0.0324 | 0.0335 | 0.0331 |
| 6101 | 653-813 | hybrid_llm16 | 0.0027 | 0.0023 | 0.0029 | 0.0025 |

## Molecular overlap

| seed | cross_source_molecule_intersection_count | llm_points_same_molecule_as_cw_fraction | cw_points_same_molecule_as_llm_fraction | cross_source_molecule_jaccard |
| --- | --- | --- | --- | --- |
| 157 | 1.4667 | 0.0958 | 0.0917 | 0.0575 |
| 6101 | 1.1333 | 0.0792 | 0.0708 | 0.0406 |

Overlap refers to molecules, not necessarily duplicate conditions or redundant experimental value. Geometry can show a close gradient region, but cannot establish that any selected experiment was unnecessary.

## Coverage denominator sensitivity

| seed | source | cw_coverage_distance_relative_drop | cw_remaining_U_coverage_distance_relative_drop | cw_common_remaining_U_coverage_distance_relative_drop | cw_selected_zeroing_share_of_total_gain | raw_coverage_distance_relative_drop | raw_remaining_U_coverage_distance_relative_drop | raw_common_remaining_U_coverage_distance_relative_drop | raw_selected_zeroing_share_of_total_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | cw_lcmd32 | 0.0905 | 0.0567 | 0.0567 | 0.3989 | NA | NA | NA | NA |
| 157 | free32 | 0.0537 | 0.0321 | 0.0321 | 0.4249 | 0.0562 | 0.0334 | 0.0334 | 0.4280 |
| 157 | hybrid_all32 | 0.0738 | 0.0449 | 0.0449 | 0.4110 | 0.0760 | 0.0462 | 0.0462 | 0.4116 |
| 157 | hybrid_cw16 | 0.0529 | 0.0344 | 0.0342 | 0.3648 | 0.0541 | 0.0353 | 0.0352 | 0.3625 |
| 157 | hybrid_llm16 | 0.0230 | 0.0121 | 0.0116 | 0.4816 | 0.0241 | 0.0124 | 0.0120 | 0.4843 |
| 157 | raw_lcmd32 | NA | NA | NA | NA | 0.0941 | 0.0579 | 0.0579 | 0.4132 |
| 6101 | cw_lcmd32 | 0.0973 | 0.0618 | 0.0618 | 0.3893 | NA | NA | NA | NA |
| 6101 | free32 | 0.0594 | 0.0364 | 0.0364 | 0.4055 | 0.0605 | 0.0369 | 0.0369 | 0.4060 |
| 6101 | hybrid_all32 | 0.0731 | 0.0456 | 0.0456 | 0.3969 | 0.0760 | 0.0475 | 0.0475 | 0.3955 |
| 6101 | hybrid_cw16 | 0.0537 | 0.0345 | 0.0344 | 0.3709 | 0.0561 | 0.0362 | 0.0361 | 0.3670 |
| 6101 | hybrid_llm16 | 0.0212 | 0.0124 | 0.0121 | 0.4777 | 0.0220 | 0.0129 | 0.0126 | 0.4735 |
| 6101 | raw_lcmd32 | NA | NA | NA | NA | 0.1036 | 0.0653 | 0.0653 | 0.3988 |

Full-U coverage includes exact zero distance at chosen centers; `mechanical_selected_zeroing_drop_component` records that contribution explicitly. `remaining_U` excludes the source selection, matching the five-seed report. `common_remaining_U` excludes the entire32 for the two16 sources, ensuring a common denominator for attribution. Differences in small batch sizes and denominator alone do not show sample usefulness.

## Availability and interpretation boundaries

- Verified frozen states: 120. Raw banks: 90; CW banks: 90.
- Hybrid and Free each have raw and CW frozen banks. Independent CW-LCMD32 has CW only. Independent raw LCMD32 has raw only; unavailable spaces are NA and are never borrowed or reconstructed.
- CW catalog ranks are recalculated on the full preselection U including pending16, then compared to catalog.candidates + catalog.pending; five-decimal rounding tolerance is 5.1e-6.
- Effective ranks use centered singular-value entropy (and separately covariance participation rank), with no inference from 16-vs32 rank magnitudes alone.
- No claim about current-model true high-error regions is made from feature distances. Such evidence requires already measured premeasurement residuals and is addressed separately.
- Geometry does not identify loading as the largest difference or show that a loading cap will improve validation AULC. It provides an observational check of coverage allocation; loading behavior and paired prospective arms are needed for the scientific question.

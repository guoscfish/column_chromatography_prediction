# Frozen Scientist V2 selection behavior audit

Two seeds, acquisitions r0–14, 333→813. Fractions are per selected row; CW16 and LLM16 denominator16, Hybrid/Free/CW32/raw-LCMD denominator32. All source rows use the full32 peer context. These are matched-budget observational comparisons; rounds/points are not independent replicates, and no population significance or causal claim is made.

## Whole-window results by seed

| seed | source | strict_loading_prior_fraction | strict_loading_union_fraction | legacy_v_prior_extension_fraction | pe_extension_union_fraction | broad_loading_union_fraction | new_molecule_fraction | unique_molecules |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.6229 | 0.7562 | 0.1083 | 29.8000 |
| 157 | free32 | 0.2521 | 0.2854 | 0.2521 | 0.7208 | 0.7042 | 0.1313 | 23.3333 |
| 157 | hybrid_all32 | 0.1250 | 0.1437 | 0.1250 | 0.6729 | 0.7438 | 0.1083 | 27.2000 |
| 157 | hybrid_cw16 | 0.0000 | 0.0083 | 0.0000 | 0.6250 | 0.7875 | 0.0750 | 15.2000 |
| 157 | hybrid_llm16 | 0.2500 | 0.2792 | 0.2500 | 0.7208 | 0.7000 | 0.1417 | 13.4667 |
| 157 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.5979 | 0.7312 | 0.1083 | 29.3333 |
| 6101 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.6146 | 0.7438 | 0.1167 | 28.9333 |
| 6101 | free32 | 0.2625 | 0.2917 | 0.2562 | 0.8542 | 0.7292 | 0.1667 | 19.2667 |
| 6101 | hybrid_all32 | 0.1625 | 0.1771 | 0.1521 | 0.5979 | 0.7500 | 0.1062 | 28.5333 |
| 6101 | hybrid_cw16 | 0.0000 | 0.0125 | 0.0000 | 0.5708 | 0.8208 | 0.0458 | 15.1333 |
| 6101 | hybrid_llm16 | 0.3250 | 0.3417 | 0.3042 | 0.6250 | 0.6792 | 0.1667 | 14.5333 |
| 6101 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.5896 | 0.7271 | 0.1083 | 30.2667 |

`strict_loading_prior` uses own prebatch L_t only. `strict_loading_union` adds full-batch peers and counts each point once. The legacy V-only metric exactly reproduces the previous five-seed audit in the same seed/window; the published34.65% is the other five-seed 333→1485 window. Density×V is a derived mass and receives no extra count.

## Stage-specific strict loading

| seed | stage | source | strict_loading_prior_fraction | strict_loading_batch_only_fraction | strict_loading_both_fraction | strict_loading_union_fraction | new_molecule_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | 333-525 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1615 |
| 157 | 525-653 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0781 |
| 157 | 653-813 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0688 |
| 157 | 333-525 | free32 | 0.1510 | 0.0104 | 0.0000 | 0.1615 | 0.2344 |
| 157 | 525-653 | free32 | 0.2812 | 0.0469 | 0.0469 | 0.3281 | 0.0938 |
| 157 | 653-813 | free32 | 0.3500 | 0.0500 | 0.0250 | 0.4000 | 0.0375 |
| 157 | 333-525 | hybrid_all32 | 0.0885 | 0.0469 | 0.0052 | 0.1354 | 0.1927 |
| 157 | 525-653 | hybrid_all32 | 0.0312 | 0.0000 | 0.0000 | 0.0312 | 0.1172 |
| 157 | 653-813 | hybrid_all32 | 0.2437 | 0.0000 | 0.0250 | 0.2437 | 0.0000 |
| 157 | 333-525 | hybrid_cw16 | 0.0000 | 0.0208 | 0.0000 | 0.0208 | 0.1458 |
| 157 | 525-653 | hybrid_cw16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0625 |
| 157 | 653-813 | hybrid_cw16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 157 | 333-525 | hybrid_llm16 | 0.1771 | 0.0729 | 0.0104 | 0.2500 | 0.2396 |
| 157 | 525-653 | hybrid_llm16 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.1719 |
| 157 | 653-813 | hybrid_llm16 | 0.4875 | 0.0000 | 0.0500 | 0.4875 | 0.0000 |
| 157 | 333-525 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1875 |
| 157 | 525-653 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0781 |
| 157 | 653-813 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0375 |
| 6101 | 333-525 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1615 |
| 6101 | 525-653 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0703 |
| 6101 | 653-813 | cw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1000 |
| 6101 | 333-525 | free32 | 0.2656 | 0.0521 | 0.0208 | 0.3177 | 0.2240 |
| 6101 | 525-653 | free32 | 0.1719 | 0.0156 | 0.0156 | 0.1875 | 0.2891 |
| 6101 | 653-813 | free32 | 0.3312 | 0.0125 | 0.0250 | 0.3438 | 0.0000 |
| 6101 | 333-525 | hybrid_all32 | 0.0156 | 0.0104 | 0.0000 | 0.0260 | 0.2188 |
| 6101 | 525-653 | hybrid_all32 | 0.1484 | 0.0156 | 0.0000 | 0.1641 | 0.0547 |
| 6101 | 653-813 | hybrid_all32 | 0.3500 | 0.0187 | 0.0437 | 0.3688 | 0.0125 |
| 6101 | 333-525 | hybrid_cw16 | 0.0000 | 0.0104 | 0.0000 | 0.0104 | 0.1042 |
| 6101 | 525-653 | hybrid_cw16 | 0.0000 | 0.0156 | 0.0000 | 0.0156 | 0.0156 |
| 6101 | 653-813 | hybrid_cw16 | 0.0000 | 0.0125 | 0.0000 | 0.0125 | 0.0000 |
| 6101 | 333-525 | hybrid_llm16 | 0.0312 | 0.0104 | 0.0000 | 0.0417 | 0.3333 |
| 6101 | 525-653 | hybrid_llm16 | 0.2969 | 0.0156 | 0.0000 | 0.3125 | 0.0938 |
| 6101 | 653-813 | hybrid_llm16 | 0.7000 | 0.0250 | 0.0875 | 0.7250 | 0.0250 |
| 6101 | 333-525 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1406 |
| 6101 | 525-653 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1250 |
| 6101 | 653-813 | raw_lcmd32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0563 |

## Interpretation and Phase B gate

Free has more strict loading than Hybrid all32 on average in each seed, but not more than its internal LLM16 consistently. In seed157 their full-window union fractions are nearly identical; in seed6101 Hybrid LLM16 is higher. CW16 has zero prior-L_t strict loading at every round, so much of the full32 contrast is mechanically induced by the16-CW quota. Batch-only cross-source contrasts add a small nonzero CW contribution.

The stage pattern is heterogeneous: seed157 early Free loading is lower than Hybrid LLM16; seed6101 middle and late Free loading is lower than Hybrid LLM16. In seed6101 at653–813 even Hybrid all32 has higher strict loading than Free. Original H3 asks whether pending CW exposure changes LLM loading preferences and does not specify a direction. Heterogeneous differences are observed, but they cannot be attributed to pending exposure because selection size, trajectory state and other inputs also differ. The stronger subhypothesis that pending CW uniformly lowers loading is not supported. H1/H4 cannot be inferred from historical performance curves. H2 remains compatible with complementary geometric and local designs, subject to the separate geometry audit.

Loading is not uniformly the largest observed distinction. In seed157 Free-minus-Hybrid all32 strict-union loading differs by14.17 percentage points and PE/EA by4.79 points. In seed6101 PE/EA differs by25.63 points, larger than loading's11.46 points; Free also selects19.27 distinct molecules per batch versus Hybrid28.53. Broad loading-related design reverses the strict comparison: Free70.42%/72.92% versus Hybrid74.38%/75.00%. The method distinction includes response-curve design and molecule concentration, not only loading.

Pause Phase B implementation under the user's stated gate. There is no defensible common main K from these two trajectories. If a future same-version paired acquisition study is separately authorized, K=8 or12 could be transparent25%/37.5% constraints, but neither is preregistered or selected by validation results here. Their heterogeneous binding rates are reported below; these are descriptive candidates, not a recommended main cap.

## Source division, overlap and duplicates

| seed | source | unique_molecules | molecule_repeated_rows_beyond_first | cw_llm_shared_molecule_count | cw_unique_new_molecule_count | llm_unique_new_molecule_count | pending_same_molecule_fraction | pending_strict_loading_peer_fraction | within_source_exact_tuple_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | cw_lcmd32 | 29.8000 | 2.2000 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 |
| 157 | free32 | 23.3333 | 8.6667 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 |
| 157 | hybrid_all32 | 27.2000 | 4.8000 | 1.4667 | 1.2000 | 2.0000 | 0.0979 | 0.0042 | 0.0000 |
| 157 | hybrid_cw16 | 15.2000 | 0.8000 | 1.4667 | 1.2000 | 2.0000 | 0.1000 | 0.0000 | 0.0000 |
| 157 | hybrid_llm16 | 13.4667 | 2.5333 | 1.4667 | 1.2000 | 2.0000 | 0.0958 | 0.0083 | 0.0000 |
| 157 | raw_lcmd32 | 29.3333 | 2.6667 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 |
| 6101 | cw_lcmd32 | 28.9333 | 3.0667 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 |
| 6101 | free32 | 19.2667 | 12.7333 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 |
| 6101 | hybrid_all32 | 28.5333 | 3.4667 | 1.1333 | 0.7333 | 2.6000 | 0.0938 | 0.0063 | 0.0000 |
| 6101 | hybrid_cw16 | 15.1333 | 0.8667 | 1.1333 | 0.7333 | 2.6000 | 0.1083 | 0.0000 | 0.0000 |
| 6101 | hybrid_llm16 | 14.5333 | 1.4667 | 1.1333 | 0.7333 | 2.6000 | 0.0792 | 0.0125 | 0.0000 |
| 6101 | raw_lcmd32 | 30.2667 | 1.7333 | NA | NA | NA | 0.0000 | 0.0000 | 0.0000 |

A new molecule means absent from that method's own L_t. CW's coverage need not mean new chemistry: it operates in learned response-gradient geometry and often chooses known molecules in new condition regions. Molecule identity overlap with pending CW points is not identical point selection; frozen CW/LLM IDs are disjoint by construction. Counts of unique new molecules exclude molecules selected by the other source in the same batch.

## Reasons versus actual IDs: lexical proxies

| seed | source | choices | loading_word_choices | loading_word_and_actual_strict | loading_word_without_strict | strict_without_loading_word | reason_unknown_id_count | reason_other_molecule_token_count | reason_unknown_molecule_token_count | reason_exact_strict_loading_reference_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | free32 | 480 | 305 | 124 | 181 | 13 | 1 | 38 | 0 | 120 |
| 157 | hybrid_llm16 | 240 | 224 | 66 | 158 | 1 | 1 | 64 | 0 | 43 |
| 6101 | free32 | 480 | 266 | 123 | 143 | 17 | 0 | 94 | 0 | 121 |
| 6101 | hybrid_llm16 | 240 | 198 | 72 | 126 | 10 | 2 | 23 | 0 | 75 |

Loading vocabulary is only a text proxy. `reason_loading_mention` is a conservative literal regex, not a judgment that a claim is true or causal. Actual strict classification uses selected IDs and five frozen conditions. Exact20-hex reference IDs, SMILES, and packet-local molecule tokens are audited independently. Other molecule tokens can be legitimate analog comparisons, so they are not labeled erroneous. Other unselected candidate references were legally visible but are not observed/pending/full-batch contrast support. A zero lexical match can still be a real matched contrast; a nonzero match can describe broad/multiaxis design rather than strict completion. Profile reason fractions use that source's16/32 denominator, while `_among_llm_fraction` uses explicit `reason_choice_denominator`.

## Historical residual boundary

Historical high8-neighborhood features use only each LLM trajectory's own prebatch observed acquisition-time errors. They describe whether choices revisit previously revealed errors; they do not establish current-model high-error neighborhoods. Initial333 rows lack acquisition-time errors, and CW/raw-LCMD receipts do not carry own frozen residual payloads, so unavailable control values remain NA. No candidate truth or held-out/test truth was loaded.


## Single-axis extension totals

| seed | source | density_extension_union_count | v_extension_union_count | loading_type_extension_union_count | loading_volume_extension_union_count | strict_loading_union_count |
| --- | --- | --- | --- | --- | --- | --- |
| 157 | cw_lcmd32 | 0 | 0 | 0 | 0 | 0 |
| 157 | free32 | 0 | 135 | 0 | 9 | 137 |
| 157 | hybrid_all32 | 0 | 68 | 2 | 2 | 69 |
| 157 | hybrid_cw16 | 0 | 2 | 0 | 0 | 2 |
| 157 | hybrid_llm16 | 0 | 66 | 2 | 2 | 67 |
| 157 | raw_lcmd32 | 0 | 0 | 0 | 0 | 0 |
| 6101 | cw_lcmd32 | 0 | 0 | 0 | 0 | 0 |
| 6101 | free32 | 0 | 135 | 3 | 11 | 140 |
| 6101 | hybrid_all32 | 0 | 81 | 1 | 9 | 85 |
| 6101 | hybrid_cw16 | 0 | 3 | 0 | 0 | 3 |
| 6101 | hybrid_llm16 | 0 | 78 | 1 | 9 | 82 |
| 6101 | raw_lcmd32 | 0 | 0 | 0 | 0 | 0 |

One point can have different single-axis contrasts against different references. Axis counts therefore may overlap; the primary strict-loading union counts each selected point once. Density-only extensions are zero in this frozen window. CW32 and raw-gradient LCMD32 have zero strict loading in333→813; the historical6.65% raw-LCMD figure belongs to the longer five-seed window.

## Own historical error neighborhoods

| seed | source | high8_same_molecule_fraction | high8_same_molecule_enrichment |
| --- | --- | --- | --- |
| 157 | cw_lcmd32 | NA | NA |
| 157 | free32 | 0.0893 | 0.0656 |
| 157 | hybrid_all32 | 0.1116 | 0.0818 |
| 157 | hybrid_cw16 | 0.0446 | 0.0148 |
| 157 | hybrid_llm16 | 0.1786 | 0.1488 |
| 157 | raw_lcmd32 | NA | NA |
| 6101 | cw_lcmd32 | NA | NA |
| 6101 | free32 | 0.1473 | 0.1293 |
| 6101 | hybrid_all32 | 0.1094 | 0.0810 |
| 6101 | hybrid_cw16 | 0.0357 | 0.0074 |
| 6101 | hybrid_llm16 | 0.1830 | 0.1547 |
| 6101 | raw_lcmd32 | NA | NA |

Across rounds with prior acquisitions, Hybrid LLM16 selects a molecule in its own historical high8-error set at17.86%/18.30%, compared with CW16 at4.46%/3.57%. Free is8.93%/14.73%. The pools and residual histories differ, so this supports a descriptive division of attention rather than a controlled effect of pending CW exposure.

## Hypothetical cap binding rates (not preregistration)

| seed | stage | candidate_K | rounds | rounds_above_K | mean_count | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | 333-525 | 8 | 6 | 1 | 5.1667 | 2 | 13 |
| 157 | 333-525 | 12 | 6 | 1 | 5.1667 | 2 | 13 |
| 157 | 525-653 | 8 | 4 | 3 | 10.5000 | 8 | 12 |
| 157 | 525-653 | 12 | 4 | 0 | 10.5000 | 8 | 12 |
| 157 | 653-813 | 8 | 5 | 4 | 12.8000 | 8 | 24 |
| 157 | 653-813 | 12 | 5 | 1 | 12.8000 | 8 | 24 |
| 6101 | 333-525 | 8 | 6 | 3 | 10.1667 | 5 | 16 |
| 6101 | 333-525 | 12 | 6 | 3 | 10.1667 | 5 | 16 |
| 6101 | 525-653 | 8 | 4 | 0 | 6.0000 | 2 | 8 |
| 6101 | 525-653 | 12 | 4 | 0 | 6.0000 | 2 | 8 |
| 6101 | 653-813 | 8 | 5 | 3 | 11.0000 | 6 | 19 |
| 6101 | 653-813 | 12 | 5 | 2 | 11.0000 | 6 | 19 |

## Auditable examples

- Seed157 free32, r0, ID`8143c4fff6f999b7ad0e`: strict loading union=True; reason: “Ortho-methoxyacetophenone at 20/1, 54.5 mg, supplies the missing low-load counterpart to ebdeaba52880c7eade16. It also matches the load of measured 50/1 row 52b9af4df1f1e7cd4d9b, separating loading from the apparent solvent-response reversal.”
- Seed157 free32, r0, ID`9c10e0827e86bb3c7681`: loading vocabulary but strict=False (may be broad design); reason: “Same malonate and loading as the selected 10/1 row, now 50/1. Predicted V1/V2 are 41.08/69.20 mL; the matched pair identifies solvent sensitivity rather than merely adding a high-retention label.”
- Seed157 hybrid_llm16, r0, ID`ae8805bc02bce03072e2`: strict loading union=True; reason: “Add the 150-mg, 5/1 methylenedioxyacetophenone row to the pending 50-mg row at identical solvent conditions. Predictions are almost unchanged, 22.98/42.58 versus 22.99/42.57 mL. Observed m166 rows ea9ef1bc8766196ca96d and 20acd2ae49790ab82053 show a substantial load-associated change at fixed 10/1. This pair tests whether the near-zero loading response is justified for another oxygenated aromatic compound.”
- Seed157 hybrid_llm16, r0, ID`714004602e6452764d9a`: loading vocabulary but strict=False (may be broad design); reason: “The same sulfoxide at 5/1 holds sample amount and loading solvent fixed. Predictions change to 5.24/15.29 mL: earlier onset but later completion in weaker eluent. Measuring both rows can distinguish a real change in interval shape from an unsupported response learned from other oxygen-containing compounds.”
- Seed6101 free32, r0, ID`aef5cafc63ca445e92fa`: strict loading union=True; reason: “M177 at 10/1 with 150 uL loading and 1000 uL loading solvent is predicted to have anomalously early V1 and broad uncertainty. It tests the response against measured m177 anchor 5e276c1730fc0a9deba7.”
- Seed6101 free32, r0, ID`0164f18711777547e68d`: loading vocabulary but strict=False (may be broad design); reason: “M177 at low solvent strength and lower loading contrasts the measured 20/1 m177 result and tests whether the predicted long V2 tail is solvent- or loading-driven.”
- Seed6101 hybrid_llm16, r3, ID`1b1a9ec85268aa66d327`: strict loading union=True; reason: “Measure amino ester m72 at 20/1 with a 50-uL sample in 200-uL DCM, predicted 25.58/98.35 mL. The measured 20/1 row uses 150 uL and gave 29.26/109.68; the measured 10/1 row uses 50 uL and gave 12.85/32.50. This controlled loading contrast separates a solvent-driven completion jump from a sample-amount contribution.”
- Seed6101 hybrid_llm16, r0, ID`0b247863fd6e8aaf44d8`: loading vocabulary but strict=False (may be broad design); reason: “Cinnamic acid at 2/1 uses the same recorded density, sample volume and 2000-uL DCM loading volume as observed f90a5a5a54509c085ebd and 1e775e7d8dcbf48fd6f8. Those endpoints are approximately 6.51/108.89 at 1/1 and 12.92/60.92 at 5/1. The missing intermediate condition discriminates a reproducible endpoint-specific response from an isolated anomalous record.”

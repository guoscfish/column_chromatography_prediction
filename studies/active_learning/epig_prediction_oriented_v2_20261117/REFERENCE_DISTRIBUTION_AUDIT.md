# Full-population target and reference sampling audit

The original P1/P2 comparison contains substantial reference-sampling confounding. Removing that confounding leaves a real EPIG–IVR objective difference on the fixed Gaussian posterior. Neither result is a QGeoGNN efficacy claim.

This **post-stage1/stage2 exploratory development study** was frozen before the new acquisitions in `reference_protocol.json` (SHA256 `851ac28a195209620c73e7f1b19c4e64e362a8b703ecff6623e551f5540fba2b`). All five seeds use the original hash-verified F0/PCA16, single-start, 60-iteration Matérn-3/2 full-L333 posterior cache. This is a fixed modeling rule, not post-hoc selection of the best family. Original qualification failures remain unchanged. No model was refit and no real labels were read by this audit.

## Actual computation

For each of 157, 887, 2357, 6101 and 12203, the candidate population is the legal 2997-row U. P1 integrates over every 3330 outer-training input; P2 over every current U input. We completed 70 states: five seeds × two populations × [full, coupled M256/1024/2048, original salted M256/1024/2048]. Each state has EPIG and IVR conditional greedy B32, pointwise Top32 measured by direct batch MI, and same-posterior conditional BALD. BALD has no target reference distribution and therefore is computed once per seed and evaluated on each declared reference. Every chosen batch is also scored under a common full-P1 posterior target.

Original `study_support.py:128-133` includes `distribution` in the SHA256 key. Even identical populations consequently get independent target orders. V2 `acquisition_v2.py:19-29` uses `sha256(epig_v2_crn:seed:row_id)`, so common IDs retain a common ordering. P1 and P2 can still contain different rows at finite M; the coupling separates much of the Monte Carlo variability from the actual change in target population. Full integration removes this source of approximation entirely.

The measured acquisition audit loop took **447.418 seconds**; interpreter imports and initial feature reconstruction precede this timer and were not separately timed. Each read-only disk cross cache occupies **159,680,160 bytes** and took 0.759–1.258 seconds to build. Cache-construction covariance calls are at most 64×256 rows; selected-winner updates request C×1. Batch MI forms a 64×64 observation matrix (B32 with two outputs), then streams target blocks. No dense U×U covariance ndarray is requested or allocated. Actual maximum process RSS was **1,124,319,232 bytes**, including Python/torch, model caches, temporary arrays and resident mmap pages. Disk backing does not guarantee that mapped pages remain absent from RAM; the code's broad docstring should be read with this explicit memory qualification. Temporary cross caches were cleaned after each seed. There were **0 surrogate fits, 0 GNN fits, 0 future-U label reads and 0 test-label reads**.

## EPIG versus IVR after full integration

All numbers below refer to P1 full, the same fitted posterior and the same L/U state within a seed. MI is the mean I(Y_t;Y_B) in nats per target row, not MI with the entire target population jointly stacked.

| Seed | Singleton Spearman | Pointwise Top32 overlap | Conditional B32 overlap | EPIG batch MI | IVR batch MI |
|---:|---:|---:|---:|---:|---:|
| 157 | .964184 | .46875 | .68750 | .035420 | .034922 |
| 887 | .983925 | .71875 | .75000 | .025602 | .025343 |
| 2357 | .993901 | .87500 | .78125 | .018253 | .018114 |
| 6101 | .991882 | .93750 | .68750 | .023324 | .023167 |
| 12203 | .965682 | .81250 | .65625 | .048795 | .047299 |

The objectives remain distinct after reference confounding is removed. EPIG has the greater MI for the five measured greedy batches on its own declared objective, while IVR optimizes an absolute variance objective. This observation does not imply better Row-level generalization, a global batch optimum, or statistically independent evidence from the 32 selected rows.

## Target population effect is much smaller than the original sampling effect

| Reference rule | Mean P1/P2 EPIG singleton Spearman | Mean Top32 overlap | Mean conditional B32 overlap |
|---|---:|---:|---:|
| Legacy independent M256 | .335779 | .11875 | .07500 |
| Coupled M256 | .944353 | .86250 | .84375 |
| Legacy independent M1024 | .693672 | .27500 | .13750 |
| Coupled M1024 | .962738 | .83750 | .72500 |
| Legacy independent M2048 | .885759 | .61875 | .24375 |
| Coupled M2048 | .978721 | .91250 | .66250 |
| Full population | .999312 | .98750 | .96875 |

At full integration, P1/P2 EPIG conditional batch overlaps are 32/32, 32/32, 31/32, 28/32 and 32/32 for the ordered seeds. Their singleton ranking correlations are .998879–.999648. IVR full P1/P2 B32 overlaps are 32/32, 32/32, 31/32, 32/32 and 31/32. Thus the earlier large reference-sample P1/P2 differences cannot be attributed solely to a genuine target-distribution effect. This result is limited to the measured L333 posterior states; later AL states need their own assessment.

![Reference coupling](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/epig_prediction_oriented_v2_20261117/reference/report/P1_P2_coupling.png)

## M256/1024/2048 convergence against exact P1

The following averages use the shared ordering; each approximation is compared with its own full population. High singleton rank correlation does not establish conditional batch convergence.

| Objective | M | Score relative L2 error | Singleton Spearman | Top32 overlap | Conditional B32 overlap |
|---|---:|---:|---:|---:|---:|
| EPIG | 256 | .793716 | .475385 | .24375 | .08750 |
| EPIG | 1024 | .333870 | .797449 | .63125 | .14375 |
| EPIG | 2048 | .168472 | .927602 | .80625 | .37500 |
| IVR | 256 | .812151 | .521989 | .34375 | .10625 |
| IVR | 1024 | .339213 | .848058 | .63750 | .19375 |
| IVR | 2048 | .166442 | .953160 | .76875 | .45000 |

M2048 remains a substantial approximation for conditional greedy batches; M1024 is especially unstable. This supports the frozen v2 primary choice of full P1 integration. It is not evidence that EPIG is ineffective: it is evidence that this finite reference estimator can change the experimental policy substantially. Convergence is not required to be monotone for a deterministic nested finite sample or a discrete greedy policy.

For a concrete seed157 example, the original P1/M1024 EPIG batch has sampled-target MI .0444022, yet its exact-full-P1 MI is .0288788. The full-P1 EPIG batch has exact-full-P1 MI .0354200. An optimistic score on a finite reference draw should not be interpreted as a better full-population batch. Full-P1 EPIG greedy batches have greater proxy MI than the corresponding pointwise EPIG Top32 batches for all five seeds; that is a batch-redundancy mechanism result, not a demonstrated training benefit.

![Reference-size convergence](/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/epig_prediction_oriented_v2_20261117/reference/report/reference_convergence.png)

## Independent QA and reproducibility

`reference_qa.py` verifies every completed artifact hash and independently compares the sum of 32 conditional EPIG gains with direct dense-B Gaussian target MI. **70 comparisons pass; maximum absolute discrepancy is 5.55e−17 nats.** For all 60 available original legacy EPIG/IVR states, singleton score error is at most **6.94e−18** and the entire ordered conditional B32 is identical. This is strong evidence that the new full-reference and coupling results arise from changed references, rather than a different acquisition formula.

Detailed evidence is in `reference/report/reference_summary.csv`, `reference_comparisons.csv`, `candidate_scores_seed_*.csv.gz`, `batch_chain_rule_qa.csv`, `legacy_reproduction_qa.csv` and `independent_qa.json`. Each `reference/seed_*/P*.json` retains the references, full score vectors, selected positions, all conditional gains and MI evaluated under sampled, full-own and common-full-P1 targets. `reference/seed_*/complete.json` hashes the run artifacts; `reference/report/manifest.json` hashes the QA exports and figures. The CSV `seconds=0` entries for BALD are an unrecorded per-policy timing sentinel from the runner, not zero physical computation; the total 447.418-second measurement includes BALD.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/reference_audit.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/reference_qa.py
```

All comparisons are descriptive within five overlapping outer seed splits. The 70 reference states, folds, candidates and batch steps are not extra independent experimental replicates. No true one-step QGeoGNN learning utility was computed by this audit.

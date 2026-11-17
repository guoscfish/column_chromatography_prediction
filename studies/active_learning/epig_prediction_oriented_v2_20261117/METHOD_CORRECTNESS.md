# Gaussian acquisition correctness

This is a **post-stage1/stage2 exploratory development study**. Original code, freezes, qualification thresholds and receipts remain unchanged. Correct Gaussian algebra validates an acquisition on its declared posterior; it does not establish posterior calibration or QGeoGNN learning efficacy.

## Declared random variables

Given legally observed training data D and chosen batch observations Y_B, a candidate is Y_x = f(x) + ε_x and a target is an independent future replicate Y_t = f(t) + ε_t. The noises ε_x, ε_t and noise in B are mutually independent across experimental repetitions. Their within-row multi-output noise blocks may be correlated. Even t=x denotes a new replicate. Consequently the candidate and target marginals contain observation noise, while their cross covariance contains latent covariance only.

Original `epig.py:3-8,157-163,179-209` implements this correctly. `GaussianPosterior.cov(...,noisy=True)` at `epig_models.py:112-117` instead denotes the same observation at intersecting positions; using that noisy cross covariance for a replicate would be wrong. Acquisition always calls `noisy=False` and adds noise to marginal blocks. P2 therefore predicts future replicates at current U inputs, rather than sharing the already indexed U observation's noise. A different finite-observation estimand would need a distinct explicit implementation.

## Three objectives on one posterior

Write S_x = Cov(Y_x|D,Y_B), S_t = Cov(Y_t|D,Y_B), C_tx = Cov(f_t,f_x|D,Y_B). The Schur reduction is R_t = C_tx S_x⁻¹ C_xt.

* **EPIG** averages ½[log det S_t − log det(S_t−R_t)] over the declared target input distribution.
* **IVR** averages tr(R_t), the absolute predictive variance reduction in the declared output units.
* **BALD** is I(Y_x;f_x|D,Y_B) = ½[log det S_x − log det N_x], where N_x is observation noise. It has no reference-input distribution. The same conditional candidate latent covariance is used after every selected batch observation.

These differ in objective, with identical fitted posterior and candidate universe in every frozen-state comparison. Along divergent AL trajectories only the posterior **construction rule** remains identical; the data and fitted posterior differ.

The original two objectives use Cholesky and Schur algebra at `epig.py:30-87,192-210`. BALD is added only in v2 `acquisition_v2.py:32-53`. Singular noise may admit finite EPIG but makes ordinary differential-entropy BALD infinite or undefined; v2 BALD explicitly requires positive definite observation noise.

## Batch conditioning and units

Selecting a candidate conditions the full latent posterior on that candidate's noisy observation without reading its value. Gaussian conditional covariance is label independent. Original rank-q updates at `epig.py:212-234` correctly update candidate diagonal blocks, target diagonal blocks and target-candidate cross covariance. Conditional gains sum to the integrated I(Y_t;Y_B) by the chain rule. This integrated quantity is the average MI with each target row; it is **not** MI with a jointly stacked target population.

For an invertible output transform A, transform means by A and every latent/noise/cross block by AΣAᵀ. Gaussian MI and EPIG/BALD ordering remain invariant. IVR uses a Euclidean trace and is generally unit dependent. Here both arms use original fixed L333 endpoint scales to fix that weighting. A consistent global DΣD temperature changes neither EPIG nor BALD; it cannot repair their acquisition rankings.

## Independent numerical evidence

`math_oracle.py` computes dense conditional Gaussian covariance directly from the original synthetic joint matrix; it does not call acquisition scoring. Its receipt is `math_oracle_receipt.json`.

| Verification | Actual result | Evidence level |
|---|---|---|
| q=1/2/3, observation scales .001/.2/20, EPIG/IVR/BALD, five sequential conditioning steps | 135 score-vector comparisons pass; maximum absolute error 6.9824146464725345e−12 | Independent deterministic numerical oracle |
| Disk-backed independent-output implementation, q=1/2/3, all three objectives | Nine complete five-point greedy batches agree with dense oracle | Independent deterministic numerical oracle |
| Batch target MI, dense B covariance against streamed target integration | Nine comparisons pass; EPIG conditional sum equals direct batch MI | Independent deterministic numerical oracle |
| Shared-reference ordering reproduces and removes distribution-string salt | Regression test passes, including identical populations and reversed input order | Failure reproduction and regression test |
| Candidate permutation and cache block sizes | IDs and scores agree with dense oracle | Engineering regression test |
| Dependent outputs in independent-output streaming path | Explicit rejection, no diagonal approximation | Engineering regression test |

The 15 new pytest cases pass (latest receipt: 0.49 seconds). Together with the original Gaussian acquisition cases, 56 tests pass in 0.70 seconds (`math_regression_tests.xml`). They extend rather than replace the original mathematical tests, which already exercise rectangular q=1/2/3 blocks, weak correlations, near-singular valid pairs, identical input independent noise, invertible mixed-output transformations, weighted/repeated target inputs, invalid Gaussian rejection and batch chain rule (`tests/active_learning_v2/test_epig.py`). No synthetic verification consumes real experiment labels.

An additional 80-decimal mpmath oracle (`near_singular_stress.py/json`) deliberately pushes valid rotated multi-output Schur complements towards singularity. With a designed singular-value gap 1e−10, the original q=2 fast determinant branch differs by 5.25e−7 nats; at gap 1e−12, by 1.30e−4 nats. The q=3 float64 Cholesky branch also loses accuracy near this extreme boundary (7.39e−6 nats in the latter construction). These are documented finite precision limits, not silently repaired Gaussians. The actual fixed GP has substantial fitted positive noise; the independent-output streaming formula uses scalar-channel `log1p` rather than the q=2 determinant combination. No failed real-state acquisition or changed real-state ordering has been established from this artificial boundary stress.

## Full-integral implementation boundary

`IndependentStreamingCache` (`acquisition_v2.py:56-100`) exploits the **verified** independent-output GP structure. It checks every covariance block's off-diagonal channels and rejects dependence. Full target-by-candidate scalar channel covariances are written incrementally to disk; cache-construction posterior covariance calls contain at most 64 target rows by 256 candidate rows. A candidate-to-winner call contains C×1 rows. A batch-MI check creates only Bq×Bq covariance (B=32) and streamed target-to-batch blocks. There is no dense U×U covariance ndarray call or allocation. The operating system can retain mapped file pages in RAM, so this does **not** imply process RSS is O(chunk) or that disk-mapped pages never become resident. Actual peak RSS is reported rather than inferred from array working buffers. This restriction is an engineering optimization for the fixed main GP, not a claim that output dependence is irrelevant.

Reproduction from repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/fish/miniforge3/envs/fish/bin/python -m pytest -q tests/active_learning_v2/test_epig_v2_acquisition.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/math_oracle.py
```

Numerical correctness is strong evidence for the implemented Gaussian objectives and conditioning. It is not empirical evidence for five-seed Row-level AULC improvement.

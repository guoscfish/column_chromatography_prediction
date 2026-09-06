# Scaling failure audit: frozen analysis plan

Base: `6e10498`; no predictor training or modification. Project research status was updated before this audit. This is a developmental analysis of repeatedly used data, not a pristine confirmatory experiment.

## Roles and order

1. Use identities/features to audit exact/relaxed pairing and source-train eligibility over all canonical rows. No target truth is needed.
2. Model-direction evidence uses only each frozen context's **budget-100 gradient_train** labels, independently for every column/protocol/seed. Never merge labels across seeds to fit a model. Fit scale/affine within compound GroupKFold and diagnose out-of-fold residuals; ratios require no target fitting. All-budget scale coefficient stability uses gradient_train only. Small fold counts and support counts are reported.
3. Select at most two A/B/C directions from this training evidence, freeze explicit model/feature/regularization choices in `model_protocol.json`, and freeze all predictions before reading corresponding test truth. Focal validation may select only the preregistered penalty/fallback. Test plots and tables are descriptive and do not trigger another method.

## Fixed statistical definitions

`ratio = target/source_q50` only when source_q50 >= 0.5 mL; otherwise ratio is undefined and the count is retained. Errors retain every row. This excludes denominator instability without altering prediction/evaluation data. Ratio associations are not causal: source error and division by source_q50 can create apparent magnitude effects.

Train-derived source-q50 bins: below q33 / q33–q67 / q67–q90 / above q90 (extreme tail). Frozen train thresholds also define test bins. Report sample/compound counts, median/IQR ratio, signed scale/affine residual, RMSE/MAE/R²/source NRMSE, and tail SSE share. Sparse strata are descriptive, not evidence of absent effects.

Conditions: EA fraction, loading solvent (DCM vs PE), loading volume `V/ul`, actual amount `density*V/ul`, and loading-solvent volume. Report original categories plus source-bin interactions. A partial-dependence-style summary standardizes high/low condition-group mean ratios over **common observed source bins**, equally weighted (>=3 rows and >=2 compounds per group/bin); at least two supported bins are required. It is a support-restricted association, not a counterfactual PD or physical effect. Partial rank correlations also control source magnitude; within-compound rank residualization checks whether conditions are distinguishable from molecular composition. Constant/collinear features are marked non-estimable.

Molecular analysis uses seven auditable RDKit descriptors (MolWt, MolLogP, TPSA, HBD, HBA, rotatable bonds, ring count), not a new representation network. Each compound gets equal weight in descriptor associations and 3-nearest-neighbor analysis. Neighbor graphs use train-compound descriptor standardization, exclude self, and compare residual consistency to 199 fixed compound-label permutations. Report per-compound mean/median ratio, variance, condition count and sign consistency. Compare low/high-EA halves within repeated compounds. Descriptor neighborhoods are not called QGeoGNN latent space; molecular and condition distributions may remain confounded.

Exact cross-column matching: canonical molecule, exact rational eluent composition, loading solvent, density and loading sample volume, loading-solvent volume, and flow; column specification is necessarily different. Relaxed matching removes **only flow** (and specification); no numeric tolerance, nearest-neighbor matching or descriptor matching. Keeping density and volume separately is stricter than matching only their product. Distinguish exact, same-compound different-condition, and source-absent; report all-source and source-train matches separately. Any observed-source label feature uses source-train rows only; duplicate source matches are averaged and their count/spread recorded. Relaxed matches are not assumed to isolate flow causally.

Coefficient stability: by column/output/protocol/budget, report seed mean/std/median/range and CV. Describe high-budget CV <=10% as relatively stable under these five frozen subsets; >10% as appreciable sampling variation. Also report leave-one-compound-out scale changes, source-range coverage and upper-10%-q50 leverage. This is not a physical-law test.

## Direction gates (training-only, before any new target-test evaluation)

- A CONDITIONAL_SCALING: same condition shows |source-adjusted rank correlation| >=0.3 with the same sign in >=4/5 seeds **and** same-sign common-support ratio contrast >=15% in >=4/5 seeds. Require two columns or one column in both row and compound. Source magnitude alone is documented but does not trigger redoing the prior 1D spline.
- B MOLECULE_DEPENDENT_SCALING: descriptor-neighbor residual correlation >=0.3 and permutation p<=0.10 in >=4/5 seeds, plus low/high-condition compound consistency correlation >=0.3 in >=4/5. Same cross-context replication rule. This is an exploratory screening gate, not corrected hypothesis significance.
- C PAIRED_DELTA_LEARNING: begin with 8g only; >=50% source-train exact-match coverage and source-error anchor (`mean matched source truth - source_q50`) has positive partial rank correlation >=0.3 with OOF scale residual in >=4/5 seeds in **both** row and compound. This distinguishes availability of pairs from evidence that they can explain error. No 25g/40g expansion without a later independent 8g signal; their relaxed matches are backlog evidence only this round.

If multiple directions pass, retain at most two, ranked by number of supporting contexts then predefined A/B/C order; no new search. If none pass: `NO_STRUCTURED_SCALING_FAILURE_IDENTIFIED`, and global scaling remains the strongest supported structure under this audit, without claiming all residual structure absent.

Any selected model must compare scale-only, affine, local identity shrinkage, head-only, and a matching strong linear control. Report full row/compound R², RMSE, MAE, arithmetic/RMS NRMSE, planned/actual-budget AULC, five-seed mean/std/median/range and paired wins. Material: >=5% mean relative AULC gain, negative median delta, >=4/5 paired wins against the strong controls in two columns or one column's row+compound. Separately label high-budget accuracy (>=5% budget-100 NRMSE gain, neither output RMSE worsens) versus label-efficiency-only gains. No test-based iteration, extra architecture, full/adapter sweep, Clean or AL.

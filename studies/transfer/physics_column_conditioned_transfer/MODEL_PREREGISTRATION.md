# Column-conditioned transfer preregistration

The physical audit precedes model training. Read audit_frozen.json and its hashes. Geometry constants in legacy code have unverified units/provenance; no true-CV/linear-velocity arm is permitted. Partial training-only mass-normalization support permits a single mass-normalized arm. Historical test exposure and the first-seed purchased audit labels make this developmental research, not independent confirmation. Audit labels do not enter another context's fit or normalization.

Scientific question: V = shared f(molecule, existing conditions, explicit column context). H1 separates empirical x²-weighted scale from packing_mass_ratio; H2 tests explicit context as a first-class input; H3 tests mL/g targets, not CV or dimensionless targets. No geometry is invented.

## Fixed arms

1. packing_mass_physical_scale: frozen source q50 × target nominal packing mass / 4. No fitting. Numerically the historical descriptive mass-ratio control; reuse is explicitly acknowledged, not renamed as a new finding.
2. physics_scale_residual: same physical prediction plus Ridge residual, standardized frozen 128D source representation + EA fraction + four physical context features. Ridge alpha=100, fit_intercept=True, one fixed value, no sweep. Fit on focal gradient-train only, in mL. The earlier additive condition Ridge used affine and a different input family; this arm tests a fixed physical reference plus frozen molecular representation, not a new empirical scaling function.
3. raw_column_conditioned: qualified V2 with additional 4→16→128 ReLU branch added to existing 128D pooled+condition representation. All existing parameters train jointly. One shared existing six-output head; no independent source head, no anchoring, no adapter/readout/head/backbone redesign.
4. mass_normalized_column_conditioned: same model and training, targets V1/mass and V2/mass; predictions multiplied by row mass. Initialize existing head weights/bias divided by 4 to preserve the qualified source function in mL/g. Raw arm retains the original source head. New branch final layer initialized zero in both arms.

Four physical branch inputs and units: nominal packing_mass_g [g], flow_ml_min [mL/min], density[g/mL]×sample_volume[uL]/mass[g] [mg/g], loading_solvent_volume[uL]/mass[g] [uL/g]. No bare column ID, extra feature or categorical control. Flow per g is descriptive only; not added as a fifth redundant feature.

New context z-score uses exactly source-train + focal gradient-train rows, unweighted row statistics, zero-variance scale=1, retained IDs. Existing qualified descriptor/eluent/condition preprocessing remains frozen source-train-only. Residual feature standardization uses focal gradient-train only. No donor target, validation, test or pool normalization observations.

## Fixed optimization

CPU deterministic single-thread torch, KMP_DUPLICATE_LIB_OK=TRUE (existing environment workaround). Adam lr=1e-4, weight_decay=1e-5, maximum 500 epochs, patience=100; inherited loss family and quantile ordering unchanged. Each epoch is one optimizer step: all focal gradient-train rows in one target batch plus an equally sized uniformly sampled source-train batch without replacement. L=0.5 L_source+0.5 L_target, equal task weighting; report both losses and source coverage. Same shared model/head handles both batches. Freeze BN running statistics for both tasks, retain trainable affine parameters. This avoids source/target batch-order confounding; no source-function penalty. Validation uses only purchased target validation and combined source-SD-normalized RMSE after conversion to mL. No source validation selection. Raw loss retains absolute mL scale as the named raw comparator; normalized arm tests removal of this magnitude imbalance with the same 0.5/0.5 weighting. No test-dependent loss rescaling or weight adjustment.

Use seed*10000+epoch target order and SeedSequence([seed,epoch,41004]) source sampling. Models initialize from the same qualified source and the same seed. Record best epoch, loss history, elapsed seconds, finite checks, training IDs and checkpoint hash. Resume only with identical code/data/protocol hashes.

## Frozen evaluation and label costs

Inherit all 120 contexts, exact purchased-label schedule, actual compound counts and tests from cross_column/splits/schedule_manifest.csv: 8g/25g/40g × row/compound × seeds 769539383,1425370602,536279090,2767143051,1362771960 × planned budgets 30,50,70,100. Each fit uses source-train and its single focal target ledger only. Audit-stage labels may never be pooled into fitting. Hash all predictions in all 120 contexts before this stage's evaluation reads target test labels. Full-data descriptive outcome census is also deferred until then.

Frozen references: scale_only, local_identity_shrinkage, conditional_EA, conditional_policy, target_head_only and current N1 standard_shallow_finetune. Reuse frozen predictions, no tuning/retraining. Verify their hashes and identity order. Metric arithmetic follows the source-anchored evaluation: AULC is trapezoidal mean of arithmetic mean of V1/V2 source-SD-normalized RMSE over planned budgets (divide by 70); separately combined NRMSE is RMS of those two normalized RMSEs. Report both, plus actual-budget AULC, R²/RMSE/MAE and budget100 absolute errors.

Strata: source-q50 q33/q67/q90 from each focal gradient-train, low/mid/high and overlapping top10; fixed EA <=0.1, (0.1,0.5], >0.5; observed flow groups only if >=10 test rows and >=3 compounds, otherwise NA with counts. Compound macro MAE and RMSE average per-compound metrics. Center=(V1+V2)/2 and width=V2−V1 errors in mL. No peak-apex or peak-variance interpretation.

## Frozen success and stopping gates

Pair every new arm against every frozen reference, not only affine or a weak chosen reference. Label-efficiency: >=5% mean AULC gain, favorable median paired difference, >=4/5 wins (ties 1e-7), against all references within a context; replicate in >=2 compound columns or the same column's row+compound. High-budget: same paired criteria for >=5% combined NRMSE gain, neither V1 nor V2 mean RMSE worsens; report >=10% separately and replication. Label efficiency and absolute accuracy remain distinct. Context support is declared only when a column-conditioned arm meets the replicated material gate; full promotion to AL candidate requires both endpoints and independent confirmation. Any isolated signal is explicitly local. Never claim TRANSFER_SOLVED on a small NRMSE gain with large 40g errors.

If no arm passes, retain strong empirical calibration and report CURRENT_DATA_DO_NOT_IDENTIFY_COLUMN_PHYSICS_SUFFICIENTLY as an identifiability assessment, not proof of no physical effects. No post-test features, LR/alpha/architecture sweeps, source anchoring, ID embeddings, donor labels or AL execution. Predictive value does not establish mass/flow causality; geometry and flow remain confounded.

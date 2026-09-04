# 4g predictor benchmark preregistration

Version: 1. Status: `PREREGISTERED / FORMAL_UNAUTHORIZED`.

## Scientific status and data definition

This benchmark is `CONTROLLED_COMPARATIVE / DEVELOPMENTAL`, not `PRISTINE_CONFIRMATORY`. The source is `dataset/dataset_4g.csv`, with row identity derived from the source hash and compound groups defined by RDKit `canonical_smiles`.

The source audit found 4,243 raw rows, 4,229 valid-label rows, and 66 finite numeric rows affected by the code-level 60/120 mL filter. No paper-level physical/instrument rationale was found. The formal threshold policy is therefore unresolved and is an execution blocker: authorization must cite measurement-validity, censoring, instrument-range, or explicit experimental-domain evidence. Model performance must not decide the policy. Once resolved, one source-data manifest and hash must be frozen for every candidate and both estimands.

No 8g row, outcome, statistic, feature scaler, or checkpoint may be used.

## Estimands and splits

**Row interpolation:** five outer row folds. The same compound may occur in other training rows; the estimand is prediction for seen/similar molecules under different chromatography conditions. It is not unseen-compound generalization.

**Compound generalization:** five outer `GroupKFold` folds with `group = canonical_smiles`. The estimand is prediction for compounds absent from training.

Within each outer-training partition, a deterministic 10% validation partition is created. Row-interpolation validation is row-level. Compound-generalization validation is group-aware, with no compound shared between inner train and validation. All candidates receive identical outer folds, inner partitions, and neural seeds `42`, `525`, and `1101`.

## Preprocessing boundary

Every scaler, normalization transform, descriptor statistic, imputation statistic, and normalized-metric scale is fit only on the actual inner-training subset for that outer fold. Validation, outer test, historical test, and 8g are excluded. Fold artifacts must record row IDs, group IDs, source hashes, fit-ID hashes, and zero overlap assertions.

The combined normalized RMSE uses the inner-training target scales only. A zero scale must be floored at `1e-8` and recorded.

## Model roles

1. **Legacy project baseline.** Current frozen cleaned Legacy QGeoGNN under this shared benchmark protocol. It answers whether redesigns improve on the historical scientific baseline; it is not exact paper reproduction.
2. **Condition-completion V2.** `qgeognn_condition_complete_v2`, changing only reachability of the five missing condition dimensions. It estimates the contribution of condition completion itself.
3. **Clean-QGeoGNN.** `qgeognn_clean_fusion_v1` revision 2, using `official_code_first_embedded_for_controlled_comparison`, `official_code_molecular_descriptor_16`, and `clean_typed_sample_level_v1`; `paper_method_equivalent=false`.
4. **Clean-contract MLP.** Same typed experimental conditions and the same declared molecular descriptor/geometry-independent molecular vector contract as Clean, but no graph message passing. Its exact input vectorizer and parameter budget must be implemented and fixture-qualified without held-out performance before authorization; this is a remaining engineering blocker.
5. **Paper-reference ANN.** 167D MACCS + 16D molecular descriptors + 9D experimental parameters; three hidden layers of 50 neurons, LeakyReLU, Adam 0.001, early stopping, maximum 10,000 epochs. Paper text says TPSA while released QGeoGNN code uses nRotB in the named 16D schema. Unless primary-source clarification resolves this, any execution must report two explicitly labeled schema sensitivities (`paper_text...TPSA` and `official_code...nRotB`) and neither may be called an exact paper ANN reproduction.

For graph candidates and the Clean-contract MLP, freeze Adam 0.001, batch size 2048, maximum 1,000 epochs, patience 100, validation-only checkpoint selection, no test-during-training, no scheduler retrofit, no architecture tuning, and the model-specific frozen 1:1 target loss contract. Paper-reference ANN retains its separately reported paper settings.

## Outcomes and diagnostics

All models report q50/point V1 and V2 RMSE, MAE, and R² plus combined normalized RMSE. Quantile-capable models additionally report V1/V2 q10–q90 coverage, mean interval width, mean pinball loss, within-target crossing rate, and cross-target ordering diagnostics. Point-only paper ANN outputs are not converted into invented quantiles.

On held-out outer evaluation, Clean additionally reports molecular and condition latent norms; molecular-projection and condition-encoder gradient norms; V1/V2 gradient contributions; condition-permutation degradation; molecule-only; and condition-only/condition-disabled diagnostics. These are explanatory and may not select or alter architecture.

## Paired questions and statistics

- Q1: Does Condition Completion V2 improve over Legacy?
- Q2: Does Clean-QGeoGNN improve over Legacy?
- Q3: Does Clean-QGeoGNN improve over Condition Completion V2?
- Q4: Does Clean-QGeoGNN improve over the Clean-contract MLP?

Seeds are repeated optimization realizations, not independent scientific replicates. For each model/estimand/outer fold, first aggregate the three seeds. Then compute paired model differences across the same five outer folds: paired mean difference, paired median difference, each fold's direction, and a fold-cluster bootstrap interval (10,000 resamples; seed `20260904`) where defensible. With five folds, wording remains descriptive and uncertainty-aware; no automatic multiple-testing battery or strong significance claim is authorized.

## Decision and stopping boundary

This preregistration does not contain superiority cutoffs that would turn a small five-fold study into a binary proof. The planned comparisons and uncertainty summaries answer the registered questions; scientific interpretation must retain the developmental data status.

Formal execution requires: (1) a threshold/data-definition decision supported independently of model scores, (2) fixture qualification and a frozen vectorizer/parameter contract for the Clean-contract MLP, (3) resolution or dual-labeled handling of the paper ANN descriptor ambiguity, (4) generated/frozen fold manifests, and (5) explicit user authorization. Until then `formal_authorized=false`; 4g→8g transfer and Active Learning remain paused/deferred.

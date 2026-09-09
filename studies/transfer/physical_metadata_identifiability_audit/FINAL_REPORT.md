# Physical column metadata provenance and identifiability audit

## Decision

**NO_NEW_IDENTIFIABLE_PHYSICAL_CONTEXT**. No physics-conditioned center/width model was trained and no outer truth was read.

## Previous physics study

The prior study completed 120 contexts, 240 neural fits, and 120 residual fits with zero failed or non-finite fits. Predictions were frozen before test evaluation; no test labels entered fitting. It tested packing-mass physical scaling, a physics-scale residual model, and raw/mass-normalized column-conditioned neural arms. Its explicit physical context was nominal packing mass [g], flow [mL/min], loading mass per packing mass [mg/g], and loading-solvent volume per packing mass [uL/g]. The final decision was `CURRENT_DATA_DO_NOT_IDENTIFY_COLUMN_PHYSICS_SUFFICIENTLY`.

Flow and packing mass are therefore not new candidates. Target flow is constant within each target column: 8g=10, 25g=15, and 40g=30 mL/min. Adding flow to separately fitted 25g or 40g models provides zero within-column information.

## Current QGeoGNN condition contract

The current model has nine sample-level condition inputs: six PE/EA-derived eluent descriptors, loading-solvent code, density times loading volume, and loading-solvent volume. It does not include flow, packing mass, geometry, or column identity. See `current_model_condition_contract.csv` for sources and units.

## Legacy constants and provenance

Released code labels optional fields `column_dia`, `column_len`, and `column_den`, and local variables `diameter`, `column_length`, and `density`. This supports `SEMANTICS_SUGGESTED`, not `VERIFIED_IN_CODE`: no unit, measurement record, product/lot, bed-versus-housing definition, or experimental provenance is present. `Use_column_info=False` in the original application path.

Classification is therefore: nominal packing mass and recorded flow are `VERIFIED_IN_CODE`; every legacy tuple is `SEMANTICS_SUGGESTED`; no tuple reaches `VERIFIED_IN_CODE`. None is placed in `UNVERIFIED_LEGACY_CONSTANT` because the variable names do suggest semantics, but that naming evidence is not measurement verification.

Tuples are 4g `(1.5, 6.6, 0.4458)`, 8g `(1.5, 13.2, 0.4458)`, and both 25g and 40g `(2.15, 15.6, 0.5248)`. The 25g and 40g constructors use separate dataset files but repeat the same constants. The repository supplies no reason proving common cartridge geometry or a 40g-specific alternative. Status: `UNRESOLVED_25G_40G_LEGACY_METADATA`.

The old physics study did not use these tuples, so they are new only in the narrow implementation-history sense. They are not newly verified physical measurements.

## Rank and confounding

With an intercept, target-only 8g/25g/40g mass+flow design rank is 3 of 3; adding all legacy fields remains rank 3, increment 0. Across 4g/8g/25g/40g, mass+flow rank is 3 and legacy fields raise it to 4. That extra fourth-context contrast is saturated by four named columns and does not create an estimable geometry effect for the three target-column shared model.

Legacy dia and den take only two values; 25g/40g are identical. All target-column variables are fixed within column and therefore deterministic functions of column identity. Packing mass and flow cannot be causally or independently identified from this observational design. The repository does not support causal mass, flow, diameter, length, density, area, velocity, or bed-volume claims. No derived geometry proxies were constructed.

## Eligibility

The legacy descriptors were absent from the previous physics context and are not numerically identical to mass/flow, but fail the provenance and independent-target-contrast requirements. Treating them as standardized features would effectively encode uncertain column identity. The eligibility gate therefore fails and the study stops before model construction.

## Required external verification

Ask experimental staff for inner diameter and units, actual packed-bed length and units, meaning and units of `column_den`, cartridge manufacturer/model/lot, whether values refer to bed or housing, measured packed-bed and void/dead volumes with method, particle size, silica bulk density, 4g+4g connection/tubing volume, and the reason 25g/40g share a tuple. Crossed mass-by-flow experiments and independent column batches are needed for identifiability.

## Next stage

After recording this negative eligibility result, proceed only through a separately preregistered filtered random learning curve at B=30/50/100/150/200/FULL. This audit does not start that curve or Active Learning.

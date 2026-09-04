# Legacy QGeoGNN audit

Status: `FROZEN_HISTORICAL_BASELINE`.

This document consolidates confirmed I0 implementation facts. The machine-readable evidence remains in [`../../studies/i0_predictor_semantic_audit/`](../../studies/i0_predictor_semantic_audit/README.md).

## Input contract as executed

The intended legacy design combines molecular topology, geometry/descriptors, eluent properties, loading conditions, and (in some application paths) column context. The clean legacy data builder actually constructs three categorical bond fields followed by ten continuous edge fields:

1. bond length
2. eluent ExactMolWt
3. eluent TPSA
4. eluent rotatable-bond count
5. eluent H-bond donors
6. eluent H-bond acceptors
7. eluent LogP
8. loading-solvent numeric code
9. `Density g/ml * V/ul`
10. loading-solvent volume

`BondFloatRBF` has only five input positions. The executed forward therefore consumes bond length, ExactMolWt, TPSA, rotatable-bond count, and H-bond donors. It ignores H-bond acceptors, LogP, loading solvent, loading mass, and loading-solvent volume. The legacy encoder names after `bond_length` do not reliably express the dataframe semantics. This issue affects both current 4g and 8g predictor paths.

## Effective capacity

For the audited configuration:

| Count | Parameters |
|---|---:|
| Nominal | 775,476 |
| `requires_grad=True` before execution | 775,476 |
| Gradient-bearing in the audited real forward/backward | 456,620 |
| Forward-unreachable | 318,856 |

Forward-unreachable trainable modules include the complete `NN_descriptor`, the outer geometry batch-normalization modules, and redundant layer modules whose outputs never reach the prediction. Nominal parameter count must not be presented as effective model capacity.

## Output and column contracts

- The clean historical quantile head enforces `q10 <= q50 <= q90` separately for V1 and V2.
- There is no hard `V1 <= V2` architecture constraint. Real 4g and 8g observations contain a small number of `V1 > V2` cases, so no hard cross-target constraint should be introduced before a label-quality audit.
- Historical 4g construction uses `V1 <= 60` and `V2 <= 120`; the authoritative 8g protocol is no-threshold. This inconsistency is a pending source-only data-quality question, not a settled scientific policy.
- Column context is mainly implicit in dataset-specific code paths. A numeric label such as `column=4/8` is not an adequate physical transfer contract.
- V1 and V2 have configured loss weights of 1:1, but loss is computed in raw mL. Equal configured weights do not imply equal gradient contribution.

## Evidence boundary

Historical results remain real measured outputs. Their correct interpretation is:

> Legacy QGeoGNN evidence under the audited legacy implementation contract.

The audit does not mean that all historical experiments are invalid, nor does it show that ignored conditions are scientifically irrelevant. It does mean that claims must not attribute historical behavior to inputs or parameter capacity that were not forward-reachable.

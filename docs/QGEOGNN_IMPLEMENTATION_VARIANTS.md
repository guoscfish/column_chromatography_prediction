# QGeoGNN implementation variants

This register separates method descriptions, executable behavior, and future proposals. It does not retroactively change any checkpoint or result.

| Contract item | Paper-method description | Original repository behavior (`application/`) | Clean reproduction derived from legacy implementation (`src/`, `scripts/`) | Proposed Predictor V2 |
|---|---|---|---|---|
| Scientific status | QGeoGNN-style geometry-enhanced graph predictor | Legacy implementation evidence | Current historical evidence; frozen for existing comparisons | Preregistration only; not implemented or authorized |
| Conformer policy | Molecular 3D geometry | First available embedded conformer in the retained workflow | `first_embedded` | Must be versioned and frozen before training |
| Molecular descriptor dtype | Molecular descriptors supplement graph features | Min-max descriptors are converted to `torch.int64` | Min-max descriptors remain `float32` | Must state dtype explicitly |
| Threshold policy | Not an implementation contract | Dataset-specific hard filtering in construction functions | Historical 4g uses the legacy threshold; authoritative 8g transfer uses the validation-selected 574-row no-threshold dataset | Must be preregistered per dataset |
| Quantile parameterization | Multi-quantile prediction | Independent outputs in the original path | Monotonic softplus within each target: q10 <= q50 <= q90 | Must state both within-target and cross-target contracts |
| V1/V2 loss weighting | Two retention-volume targets | Legacy objective | Equal target weights in the frozen clean protocols | Must be preregistered without test-based tuning |
| Column information | Column-specific datasets/functions | Encoded primarily by choosing a dataset-specific construction/training path | No newly learned explicit column-spec input in the frozen predictor | Must define whether column identity/geometry is an input |
| Constructed continuous edge block | Conditions are attached to graph edges | Ten columns are constructed after three categorical bond columns | Same ten-column construction | Full schema must be machine-readable |
| Actually consumed condition dimensions | Not specified at code-contract precision | Legacy `BondFloatRBF` reads continuous positions 0-4 | Bond length plus eluent ExactMolWt, TPSA, NRotB, and HBD; HBA, LogP, and all loading fields are ignored | Every intended feature must be empirically forward-reachable |
| Loading-feature encoding | Not specified at code-contract precision | Loading solvent code, `Density * V`, and loading-solvent volume are constructed but unreachable | Same legacy behavior | Categorical solvent encoding plus explicitly normalized continuous loading variables |
| Dead/unused modules | Not a paper-level claim | Modules exist that are absent from the executed forward path | 318,856 of 775,476 nominal requires-grad parameters are forward-unreachable in the audited configuration, including all `NN_descriptor` parameters and outer batch norms | No unused trainable modules permitted by the implementation preflight |
| Source/target scaler policy | Not a single universal policy | Script/function dependent | Source-train scaler for 4g-to-8g transfer; fixed L0-train scaler for source-free 4g AL | Must be explicit per protocol; loading continuous scalers fit on source train only for the recommended residual design |
| Legacy checkpoint compatibility | Not applicable | Defines historical checkpoint shapes | Preserved and tested | Residual option must load the legacy anchor and preserve its source function at initialization |

## Audited legacy input contract

The executable clean legacy path constructs 3 categorical and 10 continuous edge columns. The ten continuous columns are, in order: bond length; eluent ExactMolWt, TPSA, NRotB, HBD, HBA, and LogP; loading solvent code; `Density * V`; loading-solvent volume. `BondFloatRBF` has five legacy names (`bond_length`, `prop`, `e`, `m`, `V_e`) and reads only positions 0-4. The names therefore do not describe the current dataframe semantics after `bond_length`: they map to ExactMolWt, TPSA, NRotB, and HBD.

The code trace and perturbation evidence are in `studies/i0_predictor_semantic_audit/`. These are confirmed implementation facts. They do not show that the ignored conditions are scientifically irrelevant, nor do they invalidate historical performance measurements. Historical outputs should be described as **Legacy QGeoGNN evidence** or **clean reproduction derived from the legacy implementation**, depending on the runner used.

The legacy head enforces quantile ordering within V1 and V2 but has no explicit V1 <= V2 constraint. The I0 checkpoint happened to have zero q50 cross-target violations; that observation is not an architectural guarantee. No ordering constraint is added in this audit.

## Capacity terminology

For the audited clean legacy model, nominal parameters and nominal requires-grad parameters are both 775,476, but only 456,620 parameters receive gradients in an actual forward/backward pass. Parameter counts retained in T1/T1b describe configured trainable scopes and remain correct for those comparisons. A full-model count must be called **nominal requires-grad parameters**, not effective trainable capacity.

Predictor V2 is specified separately under `studies/track_b_transfer/predictor_v2_preregistration/`. It is not a silent correction to this legacy contract.

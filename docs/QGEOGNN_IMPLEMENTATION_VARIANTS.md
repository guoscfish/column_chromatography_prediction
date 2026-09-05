> Historical record: its original stage decisions are superseded by the final standalone predictor qualification and final-source transfer study. T1/T1b/G0/S1 are HISTORICAL_LEGACY_PREDICTOR_EVIDENCE. Current authority: `docs/NEXT_STAGE_DECISION.md`; active API: `src/qgeognn_al/models/qgeognn_v2.py`.

# QGeoGNN implementation variants

This register separates method descriptions, executable behavior, and future proposals. It does not retroactively change any checkpoint or result.

| Contract item | Paper-method contract | Official-code Legacy contract (`application/`) | Current Clean research contract | Condition-completion V2 |
|---|---|---|---|---|
| Scientific status | Published method description | Released implementation evidence, not a complete paper reproduction | `qgeognn_clean_fusion_v1`: redesign candidate, not paper-faithful | Controlled ablation; not scientifically qualified |
| Conformer policy | ETKDGv3 ensemble -> MMFF94 optimization -> lowest-energy conformer | Multiple conformers optimized; default/first conformer saved, no energy argmin | `first_embedded` for controlled Legacy comparison | Retains Legacy geometry |
| Molecular descriptor schema | `paper_text_molecular_descriptor_16`: 11 Mordred + MolWt + TPSA + HBD + HBA + LogP | `official_code_molecular_descriptor_16`: 11 Mordred + MolWt + nRotB + HBD + HBA + LogP | Official-code schema, `float32` | Retains Legacy schema |
| Training epochs | 1500 for 4g | Main loops use 1000 | To be frozen by formal preregistration | No performance run authorized |
| Test visibility | Validation monitoring; test for final evaluation | Validation and test both evaluated every 50 epochs; test R² logged | Validation-only selection; outer test excluded | Same current-project boundary |
| Experimental conditions | Full 9D parameter block integrated into graph G | Constructs 9D but only eluent ExactMolWt/TPSA/nRotB/HBD reach forward, alongside bond length | Typed sample-level EA fraction, solvent, mass, and loading volume all reachable | Adds only the five Legacy-missing dimensions |
| Threshold policy | 60/120 not established as a scientific definition in Methods | Hard-coded 4g filter | Pending source-only validity audit; no performance-based choice | Same benchmark data contract when authorized |
| Quantile parameterization | Multi-quantile prediction | Independent `Linear -> ReLU`; loss includes crossing penalties; V1 + 0.5 V2 | softplus median + softplus offsets; eval clamp; 1:1 current-project weights | Current cleaned Legacy head and eval clamp |
| Column information | Transfer narrative integrates new-column specification | Optional branch exists; released default is `Use_column_info=False` | Excluded for constant 4g; 4g→8g is future/not authorized | No new column input |
| Normalization | Mentioned; fit scope insufficiently specified | Full-data min/max before split | Actual-training-subset only | Source-training-only additions |
| Architecture constants | 5 GIN layers, sum pooling, hidden 128, batch 2048, Adam 0.001 | Same defaults | 5 layers, sum pooling, 128 molecular hidden, Adam 0.001 frozen for comparison | Retains Legacy backbone |

## Audited legacy input contract

The executable official and current Legacy paths construct 3 categorical and 10 continuous edge columns. The ten continuous columns are, in order: bond length; eluent ExactMolWt, TPSA, NRotB, HBD, HBA, and LogP; loading solvent code; `Density * V`; loading-solvent volume. `BondFloatRBF` has five legacy names (`bond_length`, `prop`, `e`, `m`, `V_e`) and reads only positions 0-4. The names therefore do not describe the dataframe semantics after `bond_length`: they map to ExactMolWt, TPSA, NRotB, and HBD. This is `PAPER_CODE_CONFLICT_CONFIRMED` in the official released code, not a reproduction bug introduced here.

The code trace and perturbation evidence are in `studies/i0_predictor_semantic_audit/`. These are confirmed implementation facts. They do not show that the ignored conditions are scientifically irrelevant, nor do they invalidate historical performance measurements. Historical outputs should be described as **Legacy QGeoGNN evidence** or **clean reproduction derived from the legacy implementation**, depending on the runner used.

The legacy head enforces quantile ordering within V1 and V2 but has no explicit V1 <= V2 constraint. The I0 checkpoint happened to have zero q50 cross-target violations; that observation is not an architectural guarantee. No ordering constraint is added in this audit.

## Capacity terminology

For the audited clean legacy model, nominal parameters and nominal requires-grad parameters are both 775,476, but only 456,620 parameters receive gradients in an actual forward/backward pass. Parameter counts retained in T1/T1b describe configured trainable scopes and remain correct for those comparisons. A full-model count must be called **nominal requires-grad parameters**, not effective trainable capacity.

## Implemented V2 engineering candidate

`qgeognn_condition_complete_v2` preserves the legacy GNN, sum pooling, and prediction head. It reads only the missing HBA, LogP, loading solvent, loading amount, and loading volume fields. HBA/LogP reuse the frozen 4g source-train eluent scaler; loading amount/volume use a new min-max scaler fit on the same 3,330 source-train rows. All continuous values are `float32`; solvent is categorical, not an ordinal continuous input.

The branch is `4 continuous + 4D solvent embedding -> Linear(8,16) -> ReLU -> Linear(16,128)`. Its zero-initialized output is added after fixed sum pooling and before the existing head. It adds exactly 2,332 parameters: V2 has 777,808 nominal parameters. Under multi-fixture diagnostic activation, 458,952 parameters were gradient-bearing, 318,856 structurally unreachable through the retained legacy path, and none were fixture-dependent inactive in that fixture set.

The implementation and audits are under `studies/track_b_transfer/predictor_v2_preflight/`. Passing this engineering gate does not scientifically qualify V2 or authorize training. It is not a silent correction to the legacy contract.

The full paper/code/current-project comparison, including confirmed matches, protocol conflicts, scheduler non-execution, descriptor-schema names, conformer evidence, and paper-unspecified items, is in [`model/PAPER_CODE_CONTRACT_AUDIT.md`](model/PAPER_CODE_CONTRACT_AUDIT.md).

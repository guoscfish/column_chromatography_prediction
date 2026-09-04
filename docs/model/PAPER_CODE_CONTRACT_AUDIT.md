# Paper–code QGeoGNN contract audit

Status: `PAPER_CODE_CONTRACT_AUDIT_COMPLETE`.

This audit deliberately separates (1) the method described by Wu et al. in *Chem* (2025), (2) the executable default path in the [official released repository](https://github.com/danielwu0415/column_chromatography_prediction) at commit `2b9c611fedc4f80104555821fb8122b8314bd8f5`, (3) this project's frozen Legacy contract, and (4) the current Clean research contract. “Paper unspecified” is not treated as a paper error, and a released-code mismatch is not described as a bug introduced by this project.

Primary paper reference: W. Wu et al., “Intelligent column chromatography prediction model based on automation and machine learning,” *Chem* 11 (2025), 102598, [doi:10.1016/j.chempr.2025.102598](https://doi.org/10.1016/j.chempr.2025.102598).

| Contract item | Paper text | Official released code | Current project legacy | Clean-QGeoGNN |
|---|---|---|---|---|
| Scientific identity | QGeoGNN paper method | Released implementation evidence, not a complete paper-method implementation | Frozen historical cleaned Legacy baseline | `qgeognn_clean_fusion_v1`, a scientific redesign candidate; `paper_method_equivalent: false` |
| Experimental-condition input | Graph G includes 9D parameters: 6D weighted eluent descriptors, loading-solvent type, sample/loading mass, and loading-solvent amount/volume | Constructs bond length + the 9D block, but `BondFloatRBF(["bond_length", "prop", "e", "m", "V_e"])` reads only the first five continuous positions. Effective semantics are bond length + eluent ExactMolWt, TPSA, nRotB, and HBD. Eluent HBA/LogP and all loading fields are unreachable. `PAPER_CODE_CONFLICT_CONFIRMED` | Preserves that effective Legacy path for historical comparability; I0 measured the same five missing dimensions | Typed sample-level condition branch consumes EA fraction, categorical loading solvent, loading mass, and loading-solvent volume; condition/molecule inputs are decoupled |
| Conformer | ETKDGv3 stochastic ensemble, MMFF94 optimization, lowest-energy conformer | Embeds and optimizes multiple conformers, then writes/reads the default first conformer without an energy argmin. `PAPER_CODE_CONFLICT_CONFIRMED` | `first_embedded` retained as the historical contract | `official_code_first_embedded_for_controlled_comparison`; lowest-energy is not the primary geometry |
| 16D molecular descriptor schema | Bond angle + 11 Mordred + MolWt + TPSA + HBD + HBA + LogP | `bond_angle_float_names` and `save_dataset` use 11 Mordred + MolWt + nRotB + HBD + HBA + LogP. `PAPER_CODE_CONFLICT_CONFIRMED` | `official_code_molecular_descriptor_16` | `official_code_molecular_descriptor_16` for controlled Legacy comparison |
| Descriptor dtype | Not specified at executable precision | Min-max descriptor array is cast to `torch.int64` | Corrected to `float32`; a reproducibility/implementation correction | Explicit `float32` |
| GIN layers | 5 | 5 | 5 | 5; `CONFIRMED_MATCH` |
| Graph pooling | Sum | Sum default | Sum | Sum; `CONFIRMED_MATCH` |
| Hidden width | 128 | 128 default | 128 | 128 molecular node/edge width; `CONFIRMED_MATCH` |
| Batch size | 2048 | 2048 default | Protocol-specific historical use | Preregistered as a training-protocol field; architecture preflight does not train |
| Source optimizer/LR | Adam, 0.001 | Adam, 0.001 | Adam, 0.001 in frozen source protocols | Must remain Adam, 0.001 for the formal comparison |
| 4g basic split | Approximately 80/10/10 | 0.8/0.1/0.1 | Historical split retained only for Legacy comparability | Formal benchmark preregisters separate five-fold estimands |
| 4g epochs | 1500 | Main QGeoGNN loops use `range(1000)`. `PAPER_CODE_PROTOCOL_CONFLICT` | Validation-selected cleaned protocols; not silently changed to either released loop | Budget must be frozen before benchmark execution |
| Validation/test visibility | Validation monitors training; test is final evaluation | Every 50 epochs evaluates validation and test and logs test R². `PAPER_CODE_PROTOCOL_CONFLICT` | Validation-only checkpoint/model selection; outer test excluded from decisions | Same validation-only selection boundary |
| LR scheduler | StepLR described | `StepLR(...)` is instantiated but no `scheduler.step()` occurs in the released QGeoGNN path. `SCHEDULER_DECLARED_BUT_NOT_EXECUTED_IN_RELEASED_PATH` | No scheduler is retrofitted merely for paper/code fidelity | Optimizer/schedule must be explicitly frozen in preregistration |
| Source quantile loss | Quantile-capable two-target objective | Per target: q10 pinball + q50 MSE + q90 pinball + crossing penalties; total `loss_V1 + 0.5 * loss_V2` | Later G0 decision installs monotonic within-target parameterization and many cleaned protocols use 1:1 weights. `PROJECT_POST_PAPER_METHOD_DECISION` | Frozen monotonic head aligned to current cleaned Legacy, with configured 1:1 target weights |
| Prediction head/output | Quantile predictions; exact positivity parameterization not fully specified | `Linear -> ReLU`; eval additionally clamps to `[0, 1e8]` | softplus median plus offsets; eval non-negative clamp | Same within-target parameterization and eval clamp as current cleaned Legacy |
| Column specification | New-column specification information is integrated into model input in the transfer narrative | Optional diameter/length/density branch exists, but `Use_column_info=False` in the released default path. `PAPER_RELEASED_CODE_COLUMN_CONTEXT_MISMATCH` | No newly learned column-size input in the frozen 4g predictor | No 4g column input because column size is constant; 4g→8g context is `FUTURE / NOT AUTHORIZED` |
| 4g thresholds | Methods do not establish `V1 <= 60`, `V2 <= 120` as a scientific experiment definition | Hard-coded in `Construct_dataset`. `CODE_LEVEL_FILTER_WITHOUT_CONFIRMED_PAPER_LEVEL_RATIONALE` | Historical 4g uses the code-level thresholds | No threshold change until source-only measurement-validity audit resolves policy |
| Normalization fit scope | Normalization is mentioned; fit scope is insufficiently specified | Eluent and molecular min/max are computed on the full dataset before splitting | Current controlled protocols use training-only statistics | Source/fold-training-only normalization; no validation, outer-test, or 8g use |

## Named descriptor schemas

`official_code_molecular_descriptor_16` = 11 selected Mordred descriptors + MolWt + nRotB + HBD + HBA + LogP.

`paper_text_molecular_descriptor_16` = 11 selected Mordred descriptors + MolWt + TPSA + HBD + HBA + LogP.

Clean-QGeoGNN continues to use the official-code schema. The paper-text schema is registered only as a future sensitivity candidate; it must not be selected using benchmark performance.

## Conformer evidence boundary

The existing [D04 conformer study](../../experiments/d04_conformer_selection/README.md) independently compared `first_embedded` with `lowest_energy`. Lowest-energy did not produce a stable improvement across splits. D04 is sensitivity evidence for the paper-method geometry choice, not authority to rewrite the historical Legacy geometry contract. It was not rerun in this round.

## Official code versus current project classification

- **Bug/implementation fixes:** preserve molecular descriptors as `float32`; make declared Clean inputs forward-reachable; remove known dead modules from the Clean candidate.
- **Reproducibility fixes:** explicit conformer policy, typed schemas, deterministic fixture audits, schema/config hashes, checkpoint validation, and recorded split identities.
- **Leakage prevention:** fit normalization/statistics only on the actual training subset; do not evaluate or select on an outer test during training.
- **Methodological improvements:** monotonic within-target quantiles, validation-only checkpoint selection, explicit group-aware compound evaluation, and auditable non-negative inference policy.
- **New scientific model design:** typed sample-level condition encoder, 64D/64D molecular-condition fusion, and Clean modality diagnostics. These are not paper-faithful reproduction claims.

## Audit boundary

Confirmed matches above must not be mislabeled conflicts. Confirmed conflicts describe a paper/released-code contract mismatch, not proof that all released or historical measurements are invalid. Paper-unspecified details—including exact normalization fit scope and some executable output details—remain unspecified rather than “wrong.”

# Literature Audit

This is a decision-oriented review for the current 4g→25g/40g transfer bottleneck.

| Work | Problem | Corresponding bottleneck / existing equivalent | Minimal experiment | Priority | Not applicable |
|---|---|---|---|---|---|
| Buterez et al., Nat Commun 2024, 10.1038/s41467-024-45566-8 | Fixed molecular readouts limit multi-fidelity transfer; adaptive readouts and selective fine-tuning help | Current model uses fixed pooling and has already tested head/shallow/full transfer; adaptive readout is not equivalent | Add one low-capacity gated readout while freezing the qualified backbone, evaluated only after the optimization ladder | Medium | Their multi-fidelity setup and readout assumptions are not identical to column-condition transfer |
| AdapterGNN, AAAI 2024, 10.1609/aaai.v38i12.29264 | Full fine-tuning overfits small molecular target sets | Current adapter-capacity audit exists, but no single preregistered adapter in this filtered protocol | One bottleneck adapter at the final message-passing block, parameter-count matched to last1 | High | Do not import their full benchmark or assume chemistry/task split equivalence |
| PACIA, IJCAI 2024, 10.24963/ijcai.2024/576 | Parameter-efficient few-shot molecular adaptation | PEFT is not yet a completed 25g/40g experiment | Compare final-block adapter vs last1 under identical seeds and validation | Medium | Meta-learning and few-shot episodic machinery are unnecessary here |
| Pin-Tuning, NeurIPS 2024 | Prevents catastrophic forgetting with adapters and Bayesian consolidation | L2-SP/source replay are present; Bayesian weights are not | One source-distance penalty sweep with drift reporting; no in-context machinery | Medium | In-context task construction does not match this supervised transfer protocol |
| Deng et al., J Hazard Mater 2026, 10.1016/j.jhazmat.2026.141313 | Retention-time GNN transfer and fine-tuning; reports strong L-BFGS behavior | Direct domain analogue, but optimizer claim is untested here | Validation-only Adam vs L-BFGS on the already-qualified shallow scope, one seed pilot before five seeds | High | Different labels, data regime, and architecture prevent direct metric transfer |
| Kumar et al., ICLR 2022 | Fine-tuning distorts pretrained features; LP→FT can help OOD | Current code has staged adaptation primitive but no result | T1 staged LP→last1 with drift and frozen-control comparison | High | Their pretraining/task distribution differs |
| Dey & Ning, J Cheminformatics 2024, 10.1186/s13321-024-00880-7 | Auxiliary tasks can conflict; gradient surgery mitigates interference | Source replay exists, but gradient conflict is not diagnosed | Measure source/target gradient cosine before considering PCGrad | Medium | Do not add auxiliary tasks without evidence of conflict |
| Transferability map, Commun Chem 2024 | Diagnoses relatedness and negative transfer | Current residual/column audits are related but not gradient-based | Compute source-target gradient cosine on train rows and use it only as a diagnostic | Medium | A transfer score is not itself a model improvement |

## Mathematical audit

`quantile_target_loss` mixes pinball terms (target units), q50 MSE (target-units squared), and crossing penalties (target units). The current `scaled_quantile_target_loss` divides the complete sum by `scale**2`. That is not dimensionally coherent: it over-divides q10/q90 pinball and crossing terms. A coherent normalized objective should divide squared-error terms by `s**2`, pinball/crossing terms by `s`, or explicitly define all terms as squared-normalized quantities. Any T3 result must use one declared convention and test-fold-independent scales.

The literature therefore supports a narrow next experiment: staged LP→FT plus source-preserving regularization, with an optimizer pilot motivated by Deng et al.; adaptive readout and PEFT follow only if this ladder fails.

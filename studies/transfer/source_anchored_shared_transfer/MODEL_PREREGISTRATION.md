# Source-anchored shared representation: frozen experiment

Written after PRE_EXPERIMENT_AUDIT.md, before implementation, fitting, or new
test evaluation. Evidence status: DEVELOPMENTAL. No subsequent architecture,
LR, lambda, loss, feature, readout or acquisition experiment may be appended.

## Hypotheses and fixed arms

H1: directly adapting the 128D molecule/condition representation can outperform
source-q50 scalar calibration under low target-label budgets. H2: replaying
source-train with a frozen source head stabilizes that representation and
improves target generalization beyond matched target-only fine-tuning.

| arm | method | trainable modules | count | source loss |
| --- | --- | --- | --- | --- |
| N1 | standard_shallow_finetune | backbone.convs.4, condition_branch, target_head | 36387 | 0 |
| N2 | standard_full_finetune | backbone, condition_branch, target_head | 458952 | 0 |
| M1 | source_anchored_shallow | identical to N1 | 36387 | 1 |
| M2 | source_anchored_full | identical to N2 | 458952 | 1 |

All share the qualified source backbone initialization and two independent
copies of its unchanged Linear(128,6)+ReLU head. Source head stays frozen;
target head starts at exact source weights. Zero-step six-output equivalence
must pass. Frozen baseline predictions are reused for scale_only,
local_identity_shrinkage, conditional_EA, conditional_policy, target_head_only
(N0). No reference is retuned. Strongest eligible calibration means each of
the four fixed calibration references must be beaten; no per-seed test oracle
is fitted. N0 is reported separately. N1 vs M1 and N2 vs M2 are primary
mechanism comparisons; N1/N2 vs calibration address H1.

## Data and optimizer

All 120 original cross-column contexts and exact target role IDs are reused.
Costs are original actual_budget=train+validation; planned budgets are
30/50/70/100. Report actual-budget AULC as sensitivity. No donor labels.
Source replay is restricted to the checkpoint's 3330 source-train rows.
The 416 source-validation rows form a fixed diagnostic probe; not a selector.

Adam lr=1e-4, weight_decay=1e-5, maximum_epochs=500, patience=100; strict
improvement in target validation combined normalized RMSE saves best epoch.
No LR choice/sweep. Same seeds, target ordering, batch_size=2048 and early
stopping for all new arms and historical N0. Target sets fit in one batch.
Each anchored step independently samples without replacement a source batch
with the same row count as its target batch; samples are independent across
steps and identical across matched anchored capacities for the same context.
Sampling RNG is independent of target ordering. No balanced-molecule replay.

Use the exact existing raw-mL `target_loss` for each output and sum outputs.
L_total=L_target+L_source (lambda=1); both task losses are batch means.
No additional normalization or output transformation. Magnitude and gradient
imbalance is a limitation, to be reported through initial/final task losses;
it cannot trigger post-test tuning. Source replay uses the same current
representation and frozen source head, with BN running statistics fixed
during the source pass and differentiable affine parameters. Target BN
updates follow the existing trainable-module convention. This prevents an
extra source BN-stat update from masquerading as an anchoring-loss effect.

## Freeze and diagnostics

Fit code reads target features and only purchased train/validation label
cells. It never constructs an unmasked target-test label array. Train-only
and validation-only objects enter fitting. Predictions (including all reused
references) and checkpoint/fit hashes freeze for all 120 contexts before a
separate evaluation command can reveal any new target-test truth.
Source-probe predictions and before/after parameter drift are saved for all
four arms. Source-probe truth is used only in evaluation after fitting.
Parameter L2 drift reports backbone, condition branch, and target head
separately; source-head drift must be exactly zero. Report source V1/V2
RMSE/R2 before/after and their deltas, plus source combined NRMSE drift.
All fits record best epoch, epochs run, per-task train loss, validation loss
and metric, replay draws/unique IDs, target IDs/counts, duration, counts, and
finite/success status. Failed or missing contexts prohibit final comparison.
Smoke: deterministic synthetic/real-graph unit tests and a fixed first
8g-row seed-769539383 budget-30 context, without reading test truth. The same
contract fit can be reused in the full experiment; no pilot performance gate.

## Metrics and preregistered sensitivity

V1/V2 R2, RMSE, MAE; source-SD normalized arithmetic mean RMSE; combined RMS
NRMSE; normalized AULC=trapezoid(arithmetic NRMSE, planned budget)/70.
Aggregate seeds with mean/std(ddof=1)/median/min/max. At budget100 report both
output RMSE/MAE and combined RMS NRMSE (do not conflate with arithmetic AULC).

For each output source-q50 low/mid/high boundaries are fixed from that
context's gradient-train q33/q67, plus >=train q90 as the overlapping top10
stratum. Report realized test counts, since training quantiles do not force
exact test proportions. Target-magnitude q33/q67/q90 use test truth only as
post-freeze characterization. EA bins use physical fractions <=0.1,
(0.1,0.5], >0.5. Each stratum reports RMSE/MAE/median absolute error; empty
strata get n=0 and NA, never fabricated zero error. Macro-compound MAE/RMSE
first average errors within compound, then give each compound equal weight.
Median relative absolute error uses |prediction-truth|/|truth| only where
|truth|>=1 mL; excluded row counts are reported. The 1 mL physical-unit floor
is fixed here, not chosen from test. All methods use the same definitions.

## Material evidence and decision rules

Label efficiency: >=5% gain in mean AULC, negative median paired delta,
>=4/5 wins beyond 1e-7, against every eligible calibration baseline and (for
M1/M2) its matched N1/N2. Replicate for a single fixed method in two different
columns' compound protocols, or both protocols of one column. No mixing M1
and M2 post hoc to pass replication. Qualifying compound columns must not
worsen row AULC by >5% against a required reference.

High-budget: >=5% gain in mean combined RMS NRMSE, negative median paired
delta, >=4/5 wins; neither output's mean RMSE may worsen against a required
reference. Also report >=10% stronger gain. Apply the same replication rule;
Outcome A additionally requires material budget100 accuracy in 25g or 40g
compound. Low-budget gains alone never imply the transfer problem is solved.

A: replicated anchored label-efficiency and accuracy gains over calibration
and matched FT, with large-column compound accuracy, yield
SOURCE_ANCHORED_SHARED_REPRESENTATION_SUPPORTED.
B: standard FT meets replicated calibration gains, but anchoring adds no
replicated material benefit, yield
LATENT_FINETUNING_SUPPORTED_BUT_SOURCE_ANCHORING_NOT_NEEDED.
C: anchoring reduces source-probe combined NRMSE drift in >=4/5 seeds and
reduces across-seed target AULC std, with nonworse mean target AULC, replicated
under the above rule, but calibration remains stronger, yield
SOURCE_ANCHORING_STABILIZES_NEURAL_TRANSFER_BUT_CALIBRATION_REMAINS_STRONGER.
D: all new neural arms are >=5% worse in mean AULC with >=4/5 losses to the
strongest fixed calibration reference across all large-column compound
contexts, without A/B support, yield
CURRENT_QGEOGNN_REPRESENTATION_NOT_SUFFICIENT_FOR_LOW_LABEL_CROSS_COLUMN_TRANSFER.
C has precedence over D when both conditions hold; D's interpretation is
restricted to the tested training recipe and low-label large-column task.
Mixed outcomes not meeting these conditions are INCONCLUSIVE_MIXED_EVIDENCE,
not forced into a positive or blanket negative conclusion. No new model is
launched. Keep calibration as the defensible baseline absent replicated
neural support; independent validation and UQ qualification precede AL.

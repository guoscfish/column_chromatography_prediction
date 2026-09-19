# Phase 1: matched-total-budget static versus adaptive LCMD

Frozen before acquisition and new training on 2026-09-18. Exploratory matched
development evidence on a previously evaluated cohort, not independent
confirmation. The historical baseline is commit 0e7b2ee on the row-AL branch.

## Design

Use seeds 157, 887, 2357, 6101, 12203; the identical filtered 4,163-row dataset,
3330/416/417 outer split, 333 initial labels, L0 endpoint scales, 512D gradient
extractor and CountSketch seed, CPU two-thread training, Adam .001, batch 2048,
max 1000 epochs, patience 100, scratch initialization member zero, and validation
combined-NRMSE checkpoint rule from the historical sequential protocol.

All arms end at 653 active labels (333 initial + 320 acquired); validation labels
are a shared additional 416. Test labels never participate in acquisition or
checkpoint selection. Existing test results motivated this design, so this cannot
be described as fresh blind confirmation even though new predictions stay blind.

1. OneShot-B320: LCMD-TP selects 320 ordered rows from the initial frozen gradient
   bank in one call, then scratch fits all 653 rows. Only its endpoint is compared.
2. Static-B320-prefix: the same fixed selection order, with scratch fits at +32,
   +64, ..., +320. No new gradients or feedback enter the selection. Its 653
   endpoint is exactly the OneShot model, counted once in incurred computation.
3. Adaptive-B32x10: reuse the first 11 frozen historical LCMD points. Each batch
   used the current scratch model and newly extracted gradients.
4. Random nested: reuse the same first 11 points of the historical deterministic
   permutation trajectory. Its 320-prefix also defines the matched Random
   OneShot endpoint.

The Static first 32 selected IDs must match historical Adaptive round one in
exact order. After matching preprocessing, ordered training IDs, initialization,
configuration and prediction provenance, reuse Static round zero and one. Only
9 new static fits per seed are needed, 45 total. No B64/B160 arms are scheduled:
the primary static-versus-adaptive contrast answers the feedback question first.
This protocol cannot rank intermediate batch sizes or establish a monotone
batch-size response.

## Freeze and execution

Seal this protocol, configuration, code, source data hashes, dependency versions,
passing preflight tests, and reused baseline contracts before selecting Static
trajectories. Freeze all five B320 ordered sequences before revealing any newly
selected labels. Each prefix fit receives only that prefix's labels. Freeze all
55 Static predictions/reuse references before any new test evaluation. Verify
hashes and complete seed/budget matrices on resume and reveal. Preserve incomplete
attempts separately and restart them from the same initialization.

## Analysis fixed in advance

Report endpoint combined NRMSE, V1/V2 RMSE, MAE, R2, plus budget-normalized AULC
for each metric over 333..653. Report uncapped full-data gap closure, N80/N90/N95,
and labels-to-Random@653 with first interpolated, first observed, and
sustained-through-final crossings; censor targets not reached. Do not extrapolate
OneShot learning curves from its two observed endpoints. Label savings compare
matched targets and crossing rules with the empirical Random trajectory.

For each endpoint and AULC report all seeds, mean/median/sample std, paired
differences, directional wins/ties, and descriptive paired bootstrap intervals.
The primary scientific contrast is Adaptive minus Static at 653. A useful
feedback signal requires lower mean NRMSE, at least 4/5 paired NRMSE wins, neither
mean endpoint RMSE worse by more than 2%, and neither mean endpoint R2 lower.
Report these conditions separately, without a weighted score. A curve-efficiency
signal additionally requires lower NRMSE AULC and nonlower V1/V2 R2 AULCs.
Mixed evidence remains mixed; failure is not proof of no feedback value.

Report selection overlap along matched budgets and final membership. Compute
cost distinguishes logical method effort, shared/reused work, and newly incurred
work. OneShot needs two fits including L0, Static and Adaptive need 11, Random
needs 11 for the curve. Static's intermediate models do not influence OneShot.
Training includes validation/prediction time; unavailable separate inference or
selector timing is explicitly missing, never invented. Calendar wall time and
summed task time are distinct. At most two seed workers run concurrently.

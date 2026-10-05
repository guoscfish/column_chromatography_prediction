# Center/Width B32 performance preregistration

Frozen before selected-label reveal, after-acquisition training, and test evaluation.

## Question and fixed matrix

Does Center/Width Gradient-LCMD improve one-step label efficiency over the
authoritative Raw Gradient-LCMD? Only development seeds 73, 311, 1297, 4093,
8191; exactly five new scratch fits. No confirmation or sequential artifacts
are inputs. No additional methods, tuning, architecture or training changes.

Use the small-batch benchmark's canonical splits: outer train 3330, L0 333,
U0 2997, validation 416, test 417. B=32 gives 365 active labels and 781 labels
including shared validation. Freeze all five ordered batches before revealing
selected labels. Freeze all five fitted checkpoints and test predictions before
any test truth access. Test targets are only available through RestrictedLabelStore.

## Acquisition and training

C=(V1+V2)/2; W=V2-V1. Compute population std (ddof=0) s_C and s_W from
L0 truth alone, using the existing float64 transform implementation:
[[0.5/s_C,0.5/s_C],[-1/s_W,1/s_W]]. Full-network q50 gradients, existing
512D CountSketch and seed schedule, authoritative LCMD-TP, all L0 as centers.
Reuse verified innovation CW gradient caches and reproduce their ordered IDs.
Recompute Raw selections and require exact historical B32 ID equality.

Reuse SeedContext.fit / fit_from_same_initialization and benchmark seed_config:
same member-0 initialization, Adam lr=0.001, weight_decay=0, training batch=2048,
maximum_epochs=1000, patience=100, unchanged quantile/point loss, deterministic
epoch shuffling, validation combined NRMSE checkpoint selection. No warm start.
Preprocessing and V1/V2 evaluation normalization remain the original benchmark
values. C/W scales affect acquisition only. Historical Raw and L0 predictions,
checkpoints, selections, fit audits and per-seed pre-test freezes are reused
read-only after provenance validation. No historical baseline is retrained.

## Frozen decision

E is the existing metric_row combined_normalized_RMSE, normalized by L0 V1/V2
population scales. Delta_s=E_CW-E_Raw (negative favors CW). Report all five paired
deltas, wins, means, medians and sample std (ddof=1); endpoint RMSE/MAE/R2.

CENTER_WIDTH_PROMISING requires lower mean E, at least 4/5 strict wins, and
neither endpoint mean RMSE more than 2% worse than Raw. The 2% guard is inherited
from the existing benchmark's endpoint guard, now explicitly relative to Raw.
If a comparable full-data reference exists, mean gap_closed must also beat Raw.
If mean E improves but any other promising gate fails, classify
CENTER_WIDTH_UNSTABLE_SIGNAL (including endpoint trade-off or gap gate failure).
If mean E does not improve, classify CENTER_WIDTH_NO_GAIN.
Any incompatible baseline protocol/provenance blocks execution with
BLOCKED_PROTOCOL_INCOMPATIBILITY. No post-result threshold or method changes.

## Full-data reference

Existing final_4g_qualification uses seeds 42,525,1101, source-train target scales
and different split identities. It is NOT_COMPARABLE to these five L0-normalized
development splits. No matching full-data reference was identified in the
authoritative small-batch or legacy LCMD study. Do not compute gap_closed or
invent a reference. If later supplied, it requires a separate provenance audit.

## Audit and minimal implementation

Authoritative source: qgeognn_v2_row_small_batch_benchmark/runtime/seed_s,
baseline_l0 and b32/lcmd. CW source: innovation_screen/runtime/seed_s.
Validate source and graph hashes, split identity/hash, cache content/contract,
checkpoint/preprocessing/config, initialization, and historical freeze hashes.
The innovation branch only extends gradient_features.py; all historical
training/evaluation modules must match preregistered hashes. New code consists
of a thin study runner/report, scoped tests, and these study artifacts.

Run preflight and tests before --execute. Persist source hashes and the protocol,
runner, selection and test evidence hashes. Revalidate them at execution and
before evaluation. Runtime is gitignored. Report absent measurements honestly.

## Next stage boundary

Only a real positive performance signal justifies considering CW plus target-aware
acquisition. This study never implements Target-IVR, BAIT, MaxDet or CW variants.
With no gain, consider target-aware acquisition using Raw gradients instead.

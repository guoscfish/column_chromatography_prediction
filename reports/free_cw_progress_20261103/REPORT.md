# CW-visible Free LLM32: fixed progress snapshot

Snapshot: 2026-11-03T10:39:22.523864+08:00 (Asia/Shanghai).

Accepted batches: 84/180; completed fits: 84/180; pending accepted batches: 0.

| Seed | Completed L | Completed batches | Accepted batches |
|---|---:|---:|---:|
| 157 | 813 | 15 | 15 |
| 887 | 941 | 19 | 19 |
| 2357 | 909 | 18 | 18 |
| 6101 | 845 | 16 | 16 |
| 12203 | 845 | 16 | 16 |

Five-seed comparisons are restricted to each comparator's common completed budget (current new-Free minimum: L813). Lower validation RMSE and AULC are better. AULC uses every raw nonmonotone point, normalized by budget width; paired mean and sample standard deviation use ddof=1.

Historical Free retains its mixed-model label: seeds 157 and 6101 include Astra/high prefixes; the other seeds use Sol/xhigh. These descriptive comparisons do not isolate a causal CW effect.

The experiment remains running; this is a fixed snapshot, not a final result. Accepted batches without a completion receipt are excluded from fitted curves. No test truth, selector input or scientific memory is read or modified by publication.

Local runtime models, full requests/catalogs, raw SSE mirrors, console logs and live process state are excluded from Git. Compact attempts, frozen selections, protocol/amendments, code and this validation snapshot are retained. Full local lineage is retained for the active run.

Reproduce with the existing local frozen run data: `python scripts/studies/export_free_cw_snapshot_20261102.py --output reports/<new-snapshot-directory>`. A fresh clone needs the original local dataset/initial checkpoints and separately supplied credentials to execute or re-export fits; credentials are never published.

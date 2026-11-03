# CW-visible Free LLM32 five-seed rerun

This is the isolated Sol/xhigh rerun for seeds 157, 887, 2357, 6101 and 12203, starting at L333 and targeting L1485 (36 accepted batches of 32 per seed). The scientific protocol, full input, validation barriers, initialization and scratch fitting remain frozen.

The current runtime entry is `immediate_retry_20261102/controller_overlay.py`. It composes the original frozen persistent controller, the exact upstream HTTP/2 recovery adapter, the explicitly authorized HTTP403 retry adapter and the latest zero-transport-backoff amendment. The maximum remains five seed workers and one expensive computation, with NoCW priority. A seed has only one request in flight. Successes and completed fits are reused; cumulative attempts are retained. Local worker restart cooldown remains 60 seconds; the stream inactivity watchdog remains 660 seconds. Model/parameter/validation changes are not retry fallbacks.

`PROTOCOL.md`, `SCHEMA_DIFF.md` and early launch reports are immutable historical documents. Their original finite retry and serial scheduling descriptions were superseded by `parallel_execution_amendment.json`, `persistent_20261102_v2/authorization.json`, `transport_recovery_20261102/*authorization.json`, `http403_recovery_20261102/authorization.json`, and finally `immediate_retry_20261102/authorization.json`. Read these in that order. Do not edit a frozen file or its binding to resume a run.

The latest fixed publication snapshot is [`reports/free_cw_progress_20261103`](../../../reports/free_cw_progress_20261103/REPORT.md): 84 accepted batches and 84 completed fits, with seeds 157/887/2357/6101/12203 at L813/L941/L909/L845/L845. It distinguishes accepted batches from completed fits and contains raw validation curves, fair five-seed comparisons, fit audits, completed-round receipts, accepted batch IDs and physical request hashes. It is a progress snapshot, not an L1485 final result. The earlier [`2026-11-02 snapshot`](../../../reports/free_cw_progress_20261102_1633/REPORT.md) remains unchanged.

The separate [five-method CW-gradient comparison](../../../reports/five_methods_cw_gradient_20261103/README.md) uses the two shared historical seeds for NoCW, original Free32, Free16 + CW-LCMD16, CW-gradient LCMD32 and numerical Hybrid32. Its LCMD geometry and sample count are explicit; it does not substitute for the five-seed running study.

The repository keeps compact decisions, receipt metadata, source snapshots, policies and validation outputs. Full requests/catalogs, raw SSE mirrors, model/runtime payloads, logs and live process state remain local under the repository's existing runtime policy. A fresh clone therefore does not include the original local data, initial checkpoints or credentials required to execute the study. API key values are never committed.

Obsolete preparation copies in `setup_attempts/` and mock `dry_run/` payloads were removed before publication. `publication_20261102/cleanup_manifest.json` preserves their file names, sizes and hashes. The authoritative frozen source, original offline checks and all real experimental lineage remain intact. Publication never reads test truth or feeds reports back into selection.

To audit the existing local run without sending a request:

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/immediate_retry_20261102/controller_overlay.py validate
```

To export a new immutable progress snapshot from the local saved run:

```sh
python scripts/studies/export_free_cw_snapshot_20261102.py --output reports/<new-snapshot-directory>
```

# CW-visible Free32: completed five-seed results

All five seeds (157, 887, 2357, 6101, 12203) completed 36 rounds / L1485.
180 batches were accepted and trained. All 173511 physical transport attempts
and their started/terminal receipts are retained in the lossless archives.

Open [the final report](FINAL_REPORT_20261112.html) for the Chinese summary,
milestones, per-seed comparisons, paired means/sample standard deviations,
and complete learning curves. [PDF curves](final_curves_20261112/overall_curves.pdf).

Scientific archives retain the original selected batches, contracts, selections,
feedback for selected training rows, label-access receipts, 180 completion markers,
five full trajectory freezes, 185 fit audits/prediction freezes and training histories.
The reused L333 baseline records are included; their existing local paths identify provenance.
Each gzip JSONL row contains path, SHA256 and original UTF-8 bytes in raw_utf8.
Re-encoding raw_utf8 as UTF-8 recovers the exact original file bytes. No source files
were removed or altered. costs.json.gz decompresses byte-for-byte to the automatic costs report.

Per the repository's existing artifact policy, model weights, prediction/feature
arrays, full requests/input packets/catalogs, raw SSE body mirrors, machine logs,
credentials and live locks remain local. Their saved scientific hashes are retained.
The two selection3_compute5 controller directories, tests, authorizations and switch
evidence previously designated local-only are not published. No other experiment's
unrelated working-tree changes are included.

Validation RMSE and raw normalized trapezoidal AULC are lower-is-better.
Every raw nonmonotone point is used; no smoothing or best-so-far curve is substituted.
Five-seed fair common budgets: historical mixed-model Free and Gradient-LCMD L1485;
NoCW and numerical Hybrid L1005. Endpoint improvements do not imply better whole-curve
AULC or isolate a causal CW effect. Historical Free retains its mixed-model tag.

The report's final_verification JSON preserves the original local audit snapshot.
Its report hashes refer to original uncompressed local files, with publication
transformations recorded in publication_manifest.json. Live runtime authorization
evidence remains local. The local terminal audit predates this Git publication.

Publication performs no model API call, fitting or test-truth read and never feeds
scores into acquisition inputs or scientific memory. NoCW was not modified or signaled.
No monetary price is invented; usage and receipts are preserved.

Re-export from the existing local frozen run with:
`python scripts/studies/export_free_cw_final_20261112.py --output reports/<fresh-directory>`.
A fresh clone needs separately retained original runtime models/data and credentials
to rerun the experiment; this Git snapshot is a scientific result/evidence archive.

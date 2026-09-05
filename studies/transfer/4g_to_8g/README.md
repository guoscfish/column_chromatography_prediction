# Final-source 4g→8g baseline

See [the complete report](TRANSFER_BASELINE_REPORT.md), `decision.json`, `protocol.json` and `results/` for per-seed/per-budget metrics, aggregates, normalized AULC, paired effects, trainable parameter counts and leakage audits.

Reproduce/resume from repository root in conda fish:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/studies/run_final_v2_transfer.py --execute --workers 2
```

The runner verifies frozen protocol/source/split hashes and reuses completed contexts. Interrupted neural fits resume optimizer/model/RNG state saved every 25 epochs. First execution completed every fit but its final status writer rejected NumPy integer seed/budget values; converting orchestration identifiers to native integers fixed the writer. All 20 completed contexts were hash-verified and reused, with no training rerun or method change.

The fixed source hash is `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. Source and adaptation checkpoint binaries remain local ignored runtime artifacts; a fresh clone needs the exact source checkpoint restored or reproduced. T1 manifests were reused byte-for-byte; T1 performance conclusions were not reused. No target threshold or active acquisition was applied.

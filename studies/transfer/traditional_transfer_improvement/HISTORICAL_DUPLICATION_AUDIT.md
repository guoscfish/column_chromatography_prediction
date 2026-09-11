# Historical Duplication Audit

Audit date: 2026-09-11. Repository HEAD: `5167094`.

| Method | Historical corresponding implementation | Equivalent? | Re-run decision / difference |
|---|---|---:|---|
| `head_only` | `4g_to_8g` target_head_only; `source_anchored_shared_transfer` target_head_only; cross-column and residual diagnostics | Yes | Do not re-run as a discovery experiment. Reproduce only as the frozen control in a new protocol. |
| `last1` / shallow FT | `source_anchored_shared_transfer` `standard_shallow_finetune` trains `backbone.convs.4`, condition branch, and target head | Functionally yes | Not novel. Current `last1` naming and current V2 parameter names are implementation differences only. |
| `last2` | `4g_to_8g` baseline and current cross-column protocol; current adaptation scope | Yes | Already tested; retain as control. |
| `full` | `4g_to_8g` baseline and full-data/headroom studies | Yes | Already tested; retain as control. |
| source replay | `source_anchored_shared_transfer` target-pass plus source replay, fixed BN statistics, `lambda=1` | Yes for the replay question | Do not repeat unless changing the replay ratio, source loss, or BN policy as a new hypothesis. |
| paper-style shallow/full | `paper_transfer_reproduction` and current `PaperStyleCurrentV2` | Yes at scope level | Existing result is the reference; a new run must change a declared optimization or loss condition. |

Historical optimization contract: Adam, `lr=1e-4`, `weight_decay=1e-5`, up to 500 epochs, patience 100, target-validation combined normalized RMSE checkpointing. The source-anchored study used raw-mL endpoint loss with `lambda=1`; target passes updated trainable BN statistics, while source replay evaluated with fixed BN running statistics and retained gradients. The 4g→8g protocol uses the same Adam scale and its declared BN/loss settings. Label normalization was not part of the historical source-anchored shallow replay loss.

Conclusion: `last1 vs last2 vs full` is not an untested scientific question. The highest-information remaining comparison is optimization regularization under the same frozen filtered ROW split: staged LP→FT, discriminative learning rates, dimensionally coherent endpoint weighting, and source-preserving L2-SP, each promoted only after validation evidence.

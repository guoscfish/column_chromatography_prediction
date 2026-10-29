# 研究记录：qgeognn_v2_row_cw_selector_extension_to_1005

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# CW factorial continuation to1005

User-authorized extension of all eight arms × two seeds through1005. Three new arms continue their own frozen653 histories with11 further batches each (66 fits); five controls reuse110 historical budget points after protocol/hash/lineage audit. Including the preceding24-fit653 continuation,90 new fits extend the original525 study. Zero paid LLM calls, zero new test labels.

Status: completed and delivery-audited. Frozen protocol: protocol.json; reuse manifest: REUSE_MANIFEST.json. Pipeline status: pipeline_state.json; per-seed progress: progress_seed_*.json; fit reservations/checkpoints remain under ignored runtime.

Primary descriptive extension metric: validation AULC525–1005, fixed before1005 extension outcomes. Secondary AULC653–1005, AULC333–1005, NRMSE1005 and V1/V2 RMSE/MAE/R². Original333–525 gate and intermediate525–653 result remain unchanged. Post-development exploration is not independent confirmation. Hard stop1005 regardless of effect.

Two independent fits may run concurrently after653, two CPU threads per fit. Per-trajectory quota11; interrupted attempts retain reservations. Training otherwise follows the original common frozen protocol.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/continue_cw_factorial_to1005.py
# Final verification only:
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_to1005.py
```

The supervisor refuses to silently restart an interrupted653 fit. Inspect its log and ledger before recovery; never erase a reservation to bypass the fit limit. Completed fits are reused without retraining.

## User stop amendment (current scope)

Whitening-LCMD stopped at525 for both seeds before any extension fit; preserve its historical result. Continue CW-MaxDet and CW-IVR only. Effective additional fit limits:16 through653 and44 thereafter,60 total. Seven arms × two seeds reach1005; Whitening's two curves end525. No late-window Whitening comparison is available. The original code/protocol is retained; the explicit stop amendment overrides execution membership and quota only.

Use `python scripts/studies/continue_cw_without_whitening.py` in the fish environment with `KMP_DUPLICATE_LIB_OK=TRUE`. Do not resume the old unamended supervisor.

## Explicit user resume

The subsequent user request resumes CW-MaxDet/CW-IVR only. Twelve extension fits are complete; one interrupted557 fit is restarted using the same frozen batch and initialization. See USER_RESUME.json and RESOURCE_ACCOUNTING_NOTE.md:60 completed fits plus one interrupted attempt,61 attempts maximum. Whitening remains stopped525. No extra LLM calls or test reveals.

## Verified delivery

Fourteen curves completed through1005; Whitening remains at525 for both seeds by user instruction.60 additional fits (16 to653 plus44 to1005);150 control points reused. Zero paid LLM calls or new test reveals. Original333–525 conclusions unchanged; extension exploratory. See the1005 RESULTS.md and DELIVERY_AUDIT.json for final effects and verification.

## Final accounting reconciliation (2026-10-10)

All60 additional fits are complete, with3 interrupted attempts recorded separately (63 total attempts). Earlier90-fit scope,61-attempt cap and blocked/pause descriptions are historical and superseded by the stop amendment, continue-until-complete authorization and final delivery. CW-IVR mean paired validation AULC525–1005 improvement is5.98%; CW-MaxDet is−3.80%; CW-LCMD is1.21%. Extension remains exploratory; the original confirmation gate is unchanged. Current curves are validation_curves.png in the1005 study, not paused_results. See RESOURCE_ACCOUNTING_NOTE.md and COMPLETION_RECHECK.json for final costs and verification limitations.

<a id="记录-2"></a>

## 记录 2：`RESOURCE_ACCOUNTING_NOTE.md`

# Final extension resource accounting

Completed:60 extension fits (16 to653 and44 thereafter), plus3 interrupted attempts,63 attempts total. The prior61-attempt limit was superseded by CONTINUE_UNTIL_COMPLETE_AUTHORIZATION.json. All three interrupted models remain archived; their frozen batches and original initialization were reused for successful scratch retries.

Interrupted attempts: seed6101/CW-IVR/557 (USER_RESUME.json; lost wall time at most99.9 seconds), seed157/CW-MaxDet/781 and seed6101/CW-MaxDet/845 (individual recovery_attempts receipts). Exact compute time for the latter two is unavailable; reservation-to-recovery duration includes idle time and must not be reported as training time.

Completed-fit training totals28279.668 seconds; acquisition totals3874.507 seconds. These sums exclude interrupted training costs, whose exact total is unknown; they are not elapsed makespan. No paid LLM calls or new test reveals. Whitening remains525.

Completion artifact hashes were rechecked against the existing delivery audits for all60 fits and validation curves. A fresh full audit invocation is blocked by subsequent shared-workspace changes to dual_expert_selector.py; no frozen-code check was bypassed and no unrelated work was reverted. Original successful delivery audits are preserved. See COMPLETION_RECHECK.json.

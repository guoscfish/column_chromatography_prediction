# Final extension resource accounting

Completed:60 extension fits (16 to653 and44 thereafter), plus3 interrupted attempts,63 attempts total. The prior61-attempt limit was superseded by CONTINUE_UNTIL_COMPLETE_AUTHORIZATION.json. All three interrupted models remain archived; their frozen batches and original initialization were reused for successful scratch retries.

Interrupted attempts: seed6101/CW-IVR/557 (USER_RESUME.json; lost wall time at most99.9 seconds), seed157/CW-MaxDet/781 and seed6101/CW-MaxDet/845 (individual recovery_attempts receipts). Exact compute time for the latter two is unavailable; reservation-to-recovery duration includes idle time and must not be reported as training time.

Completed-fit training totals28279.668 seconds; acquisition totals3874.507 seconds. These sums exclude interrupted training costs, whose exact total is unknown; they are not elapsed makespan. No paid LLM calls or new test reveals. Whitening remains525.

Completion artifact hashes were rechecked against the existing delivery audits for all60 fits and validation curves. A fresh full audit invocation is blocked by subsequent shared-workspace changes to dual_expert_selector.py; no frozen-code check was bypassed and no unrelated work was reverted. Original successful delivery audits are preserved. See COMPLETION_RECHECK.json.

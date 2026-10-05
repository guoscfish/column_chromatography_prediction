# 4G LLM Dual Expert Active Learning V1

## Scientific question

`free_llm32_dual_expert_v1` tests whether independent chemistry/chromatography reasoning and machine-learning/active-learning reasoning, followed by a free Scientific Planner, select labels that improve retrained QGeoGNN accuracy over the Row distribution. The experiment ends at the round-0 selection boundary in this implementation. No formal model call, freeze, label reveal, fit, or report was run while building it.

The only intended strategy variable relative to `free_llm32_scientist_v2` is reasoning architecture: Chemistry Scientist + ML Scientist -> Scientific Planner. The Row split, L333 initialization, candidate pool, QGeoGNN checkpoint, prediction schema, batch size 32, budget schedule, training configuration, label protocol, and evaluation plan are retained as controls for a future approved run.

## Information flow and isolation

For each seed and round, the host creates three separate read-only JSON relay contexts. Chemistry Scientist and ML Scientist receive the same legal state packet but independent transcript directories and independent catalog instances. They can query all current `U_t` candidates, observed `L_t` records, pending records, structures, conditions, descriptors, predictions, neutral numerical diagnostics, and trajectory-local measured feedback. They cannot see validation/test truth, future labels, other seeds, other methods, post-selection metrics, shell/files/web tools, or the other expert's memo.

The Planner receives both completed memos plus the state packet, then gets its own catalog instance and transcript. It may query again and can accept, reject, or supersede either memo. Expert recommendations are never copied into the final batch automatically. The Planner must explain chemistry takeaways, ML takeaways, agreements, disagreements, and its resolution.

There is no 16+16 split, no chemistry/ML quota, no molecule cap, no uncertainty/CW/novelty quota, no score sum, voting, rank aggregation, critic, judge, reranker, MaxDet, or post-selection strategy. The only execution constraints are exactly 32 unique legal IDs, current-pool membership, query/view/model-call budgets, and freeze-before-reveal provenance.

## Prompts

The complete, hashed prompts are stored verbatim in `chemistry_prompt.txt`, `ml_prompt.txt`, and `planner_prompt.txt`, and in `dual_expert_selector.py`. Chemistry explicitly covers 4 g silica normal-phase chromatography, PE/EA, V1/V2 semantics, chemical priors versus observed evidence, and hypothesis testing. ML explicitly treats uncertainty, coverage, residuals, novelty, redundancy, sensitivity, and information value as optional diagnostics rather than a formula. Planner explicitly owns the final decision and may choose experiments proposed by neither expert.

## Scientific memory and feedback

Future rounds will store only trajectory-local structured memory: the latest chemistry and ML memos, Planner synthesis and rationale, selected IDs, and measured feedback. Raw records remain queryable from the local catalog. Memory is checked for `(seed, method)` identity and has no future-label channel. A later protocol extension must reveal a batch only after `batch_freeze.json`, then attach measured V1/V2, premeasurement prediction errors, hypothesis relations, and expert/planner feedback. This branch currently hard-fails `advance` and `report` to prevent accidental labels or training.

## Budgets and audit

Per role and round: 24 relay queries, at most 480 distinct candidate views, and 12 model answers for each expert. The Planner has 16 model answers. Receipts record role, request hash, model/effort, backend provenance, usage when available, query results, and transcript hashes. The protocol records call/query/view budgets, prompt hashes, source hashes, and `647271df406cb30f606fd58de97b182b87fe2fd5` as the V2 baseline commit.

`ReadOnlyCatalog` rejects measured fields on candidate/pending cards, restricts measured endpoint and historical error queries to observed records, uses stable salted ID ordering and paging, and never reads validation/test labels. The final validator requires the packet hash, expert conflict synthesis, 32 viewed current candidates, unique IDs, reasons, bounded hypotheses, and no hidden reranking.

## Fairness and limitation

The architecture does not identify a causal contribution of chemistry knowledge. Prompt framing, role decomposition, additional model calls, and the Planner's extra context can each affect results. A confirmatory study should preregister matched backend/model/effort budgets, a single transport, independent seeds, and ablations that remove chemistry framing, ML framing, or expert memos while keeping the catalog and total call budget fixed. Historical test exposure in the repository also limits claims to development evidence.

## Worktree and formal command

Branch: `codex/llm-scientist-dual-expert-v1`.

Worktree: `../column_chromatography_prediction_dual_expert`.

Base: `647271df406cb30f606fd58de97b182b87fe2fd5`.

Study namespace: `studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/`.

After an explicit approval to run the first real selection, use:

```sh
/Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py select --seed 157 --method free_llm32_dual_expert_v1 --round 0
```

This command is intentionally not executed as part of implementation. It has no `advance` implementation; future label reveal and retraining require a separately reviewed protocol extension.

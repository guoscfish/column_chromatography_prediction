# CW-visible Free32 rerun

Five seeds 157,887,2357,6101,12203. L333 to L1485 in 36 batches of 32 per seed. Request gpt-6.1-sol / xhigh throughout. Scratch fit after every frozen batch, using the unchanged NoCW initialization, RNG, preprocessing, split, optimizer, epochs, patience and validation checkpoint rule. Initial L333 only is reused after byte/state hashes. New selections/history/models never import old Free/NoCW feedback. Reports never enter the selector. No test labels are accessed.

Complete full catalogs; lossless precision; historical Free salt, ID/order/encoding; same five-seed NoCW memory and direct validator. Candidate IDs and measured evidence IDs have separate validators. No quotas or numerical acquisition formula.

Eight answer attempts (one initial plus seven corrections), four transport attempts each; backoff 2/10/60 seconds. Counts persist across restarts. No retries for model/config/credential/terminal protocol violations. Ambiguous in-flight requests without durable completion stop for accounting. Durable completed responses and scratch fits are replayed locally. At most two local recoverable worker restarts per seed.

One seed-round worker and one new training fit at a time; queue by completed round count then fixed seed order. Existing NoCW retains resources and its original budgets/config. NoCW ADVANCING phases get training priority.

Milestones L525,L813,L1005,L1389,L1485 require all five seeds. Raw nonmonotone normalized trapezoidal AULC; never best-so-far. Paired comparisons only over five-seed common completed budgets. Historical Free keeps its mixed-model label. API randomness, dates and invisible backend changes prevent strict CW causal attribution.

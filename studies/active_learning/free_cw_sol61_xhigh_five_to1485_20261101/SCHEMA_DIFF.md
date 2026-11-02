# Necessary interface differences

Versus actual five-seed NoCW: restore cw_distance_percentile for each candidate and its original two-line definition; recompute packet binding hash. Skip NoCW-only negative scans on inputs, history, outputs and correction transcript; retain positive schemas and common direct checks. The private arm/version/output paths change. The visible historical study_version/method/salt remain identical. NoCW's inherited hybrid wording is retained to avoid unrelated prompt edits. Initial scientific memory is empty. The original memory-policy wording remains identical.

NoCW's currently authorized unbounded retry extensions are replaced here with a preregistered finite 8-answer x 4-transport bound. Frozen SSE uses the same request body, direct route, timeout and model parameters; adds durable response/usage/request-ID journals. Training completion is journaled before prediction export, so export failures reuse finished scratch training. Neither addition changes training or checkpoint selection.

New source snapshot changes only resources.ROOT to address read-only original data. Runtime adapters are process-local and never edit NoCW dependencies. Initial packet diffs are enumerated per candidate in offline_checks/packet_diff_seed_*.json.

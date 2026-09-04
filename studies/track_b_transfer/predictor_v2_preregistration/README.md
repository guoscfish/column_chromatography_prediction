# Predictor V2 condition-complete preregistration

Status: `IMPLEMENTATION_PREFLIGHT_COMPLETE / FORMAL_UNAUTHORIZED`.

This proposal responded prospectively to I0. Its recommended Option B has now been implemented as an engineering candidate and passed the separate preflight under `../predictor_v2_preflight/`. It did not modify the legacy QGeoGNN, reinterpret historical metrics, train a source predictor, adapt to 8g, or restart active transfer. The current legacy checkpoints and results remain frozen evidence under their recorded input contract.

## Objective and alternatives

Predictor V2 must make every intended chromatography condition explicit, typed, normalized, and empirically forward-reachable. Two implementation families must pass a preflight before architecture freeze:

| Option | Design | Legacy compatibility | Initialization/source identity | Main risk |
|---|---|---|---|---|
| A: direct clean full-schema encoder | Replace the ambiguous edge condition block with typed encoders for the full schema | New source model; legacy weights may be partially importable but not function-identical | No identity guarantee | Predictor changes are broad, making attribution and migration harder |
| B: legacy-compatible residual completion | Preserve the legacy graph path and add separate encoders only for previously ignored conditions, integrated through a zero-initialized residual | Loads the legacy anchor without changing its tensors | Must reproduce the legacy source function at initialization within frozen tolerance | New branch may learn slowly or remain underidentified |

Option B was implemented for engineering preflight because it isolates condition completion and preserves the exact legacy source function. The preflight fixed a 4D categorical solvent embedding, 16D hidden layer, 128D residual, and 2,332 added parameters without target outcomes or width comparison. It remains an engineering candidate, not the selected final predictor.

## Required encoding contract

- Keep categorical and continuous variables separate. Loading solvent uses a categorical embedding or explicit one-hot contract, never a numeric RBF interpretation of its code.
- Scale continuous loading amount (`Density * V`) and loading-solvent volume using statistics fit on source-training rows only. Persist those statistics and their provenance.
- Do not extend the old five-name RBF list and reuse its mostly 0–1 centers for unscaled mass or volume.
- Persist a machine-readable schema and its deterministic `input_schema_hash` in every V2 checkpoint.
- Require perturbation and gradient tests showing every intended feature reaches the forward output.
- If Option B is selected, require source-function identity immediately after installation and before any optimization.
- Version the model variant independently from `legacy_qgeognn_clean_reproduction_v1`.

## Preflight gate

The implementation preflight passed schema/type validation, deterministic hashes, per-feature forward and gradient reachability, normalization leakage checks, 3/3 legacy checkpoint loading, exact source-function identity, and separate nominal/gradient-bearing counts. Formal training still requires a new explicit authorization.

No 8g test truth may select the architecture, width, normalization, or integration point. Track C remains deferred.

## Consequences if V2 is later adopted

Mandatory reruns would be a new versioned 4g source baseline, source-member/UQ qualification, and a new 4g-to-8g transfer baseline under a separately frozen V2 contract. Recommended studies would include source-aware molecule/scaffold evaluation and acquisition representation ablations (graph only, conditions only, naive concatenation, block-balanced concatenation). Historical legacy T1/T1b-1, E2, and E4 runs do not need rerunning to preserve their original conclusions; they must simply remain labeled as legacy-contract evidence.

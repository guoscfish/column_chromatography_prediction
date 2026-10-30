"""Join derived read-only diagnostics; never import a study runner or label store."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ["seed", "round", "source"]
SOURCES = ("hybrid_cw16", "hybrid_llm16", "hybrid_all32", "free32", "cw_lcmd32", "raw_lcmd32")
STAGES = ("333-525", "525-653", "653-813")
CURVE_METHOD = dict(zip(SOURCES, ("cw16_llm16_scientist_v2", "cw16_llm16_scientist_v2",
                                 "cw16_llm16_scientist_v2", "free_llm32_scientist_v2",
                                 "center_width_lcmd", "raw_gradient_lcmd")))


def merge_profiles(behavior, geometry):
    """Refuse duplicate/missing keys and conflicting common metadata."""
    for frame in (behavior, geometry):
        if frame.duplicated(KEY).any():
            raise ValueError("duplicate profile key")
    if set(map(tuple, behavior[KEY].values)) != set(map(tuple, geometry[KEY].values)):
        raise ValueError("profile key sets differ")
    merged = behavior.merge(geometry, on=KEY, suffixes=("", "_geometry"), validate="one_to_one")
    for col in set(behavior) & set(geometry) - set(KEY):
        a, b = merged[col], merged[col + "_geometry"]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError(f"conflicting profile metadata: {col}")
        merged.drop(columns=col + "_geometry", inplace=True)
    if not merged.denominator.eq(merged.batch_size).all():
        raise ValueError("source denominator differs from batch size")
    return merged.sort_values(KEY).reset_index(drop=True)


def validate_profiles(frame):
    if len(frame) != 180 or set(frame.seed) != {157, 6101} or set(frame.source) != set(SOURCES):
        raise ValueError("incomplete two-seed six-source common window")
    if not frame.groupby(["seed", "source"]).size().eq(15).all():
        raise ValueError("incomplete fifteen-round source")
    if any(set(group["round"]) != set(range(15)) for _, group in frame.groupby(["seed", "source"])):
        raise ValueError("rounds must be exactly 0 through 14")
    if not frame.labels_before.eq(333 + 32 * frame["round"]).all():
        raise ValueError("label budget mismatch")
    if not frame.labels_after.eq(frame.labels_before + 32).all():
        raise ValueError("after budget must include the whole32 batch")
    expected_stage = pd.Series(np.where(frame.labels_before < 525, "333-525",
                              np.where(frame.labels_before < 653, "525-653", "653-813")), index=frame.index)
    if not frame.stage.eq(expected_stage).all():
        raise ValueError("stage does not match acquisition budget")
    expected = np.where(frame.source.isin(SOURCES[:2]), 16, 32)
    if not frame.denominator.eq(expected).all():
        raise ValueError("16/32 denominator mismatch")
    if not frame.strict_loading_union_count.eq(
        frame.strict_loading_prior_only_count + frame.strict_loading_batch_only_count + frame.strict_loading_both_count
    ).all():
        raise ValueError("loading union double count")
    for (seed, r), part in frame.groupby(["seed", "round"]):
        h = part.set_index("source")
        for count in ("strict_loading_prior_count", "strict_loading_union_count", "new_molecule_count"):
            if h.loc["hybrid_all32", count] != h.loc["hybrid_cw16", count] + h.loc["hybrid_llm16", count]:
                raise ValueError(f"hybrid source count is not additive: {seed}, {r}, {count}")
        # Same frozen space and same U minus all32: an exact geometric identity.
        for space in ("raw", "cw"):
            solo = h.loc["hybrid_cw16", f"{space}_common_remaining_U_coverage_distance_relative_drop"]
            marginal = h.loc["hybrid_llm16", f"{space}_common_remaining_U_marginal_after_other_relative_to_original_coverage"]
            union = h.loc["hybrid_all32", f"{space}_common_remaining_U_coverage_distance_relative_drop"]
            if not np.isclose(solo + marginal, union, atol=1e-10):
                raise ValueError("coverage decomposition mismatch")


def cap_sensitivity(frame):
    """Behavior-only strength audit; never consult NRMSE to pick K."""
    rows = []
    for (seed, stage), group in frame[frame.source.eq("free32")].groupby(["seed", "stage"]):
        for k in (8, 12):
            counts = group.strict_loading_union_count
            rows.append(dict(seed=int(seed), stage=stage, candidate_K=k, rounds=len(group),
                             rounds_over_cap=int((counts > k).sum()),
                             sum_count_excess=int(np.maximum(counts - k, 0).sum()),
                             fraction_rounds_over_cap=float((counts > k).mean()),
                             definition="strict_loading_union; prior L or final batch; no validation tuning",
                             status="DESIGN_CANDIDATE_ONLY_NOT_PREREGISTERED"))
    return pd.DataFrame(rows)


def add_performance(frame, curves):
    """Shared Hybrid outcomes belong to its all32 batch, never to a 16-point arm."""
    if curves.duplicated(["seed", "method", "labels"]).any():
        raise ValueError("duplicate curve budget")
    result = frame.copy()
    result["performance_trajectory"] = result.source.map(CURVE_METHOD)
    lookup = curves.set_index(["seed", "method", "labels"]).validation_nrmse
    for which in ("before", "after"):
        result[f"validation_nrmse_{which}"] = [lookup.loc[(r.seed, r.performance_trajectory, getattr(r, f"labels_{which}"))]
                                                  for r in result.itertuples()]
    result["next_fit_validation_improvement"] = result.validation_nrmse_before - result.validation_nrmse_after
    result["performance_attribution"] = np.where(result.source.isin(SOURCES[:2]),
        "whole Hybrid32 outcome; no source-specific causal performance", "own whole32 trajectory outcome")
    return result


def plot(frame, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(SOURCES, ("#258076", "#d37b29", "#2453a6", "#9d408d", "#6e6e6e", "#b3b3b3")))
    labels = dict(zip(SOURCES, ("Hybrid CW16", "Hybrid LLM16", "Hybrid all32", "Free32", "CW-LCMD32", "Raw LCMD32")))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for row, seed in enumerate((157, 6101)):
        for col, (metric, title) in enumerate((
                ("strict_loading_prior_fraction", "Strict loading vs prior L (primary)"),
                ("strict_loading_union_fraction", "Strict loading vs prior L or batch"),
                ("unique_molecules", "Unique molecules per source batch"))):
            ax = axes[row, col]
            for source in SOURCES[:4]:
                group = frame[(frame.seed.eq(seed)) & frame.source.eq(source)]
                vals = group.groupby("stage")[metric].mean().reindex(STAGES)
                ax.plot(range(3), vals, marker="o", color=colors[source], label=labels[source])
            ax.set_xticks(range(3), STAGES)
            ax.set_title(f"Seed {seed}: {title}", fontsize=10)
            ax.grid(alpha=.2)
            if col != 2:
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.set_ylim(0, 33)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen selection behavior: matched budgets, two seeds shown separately\n"
                 "CW16/LLM16 denominators=16; all32/Free denominators=32. Observational, different states.", fontsize=11)
    fig.savefig(out / "loading_and_molecules.png", dpi=180)
    fig.savefig(out / "loading_and_molecules.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    behavior = pd.read_csv(HERE / "behavior_profiles.csv")
    geometry = pd.read_csv(HERE / "geometry_profiles.csv")
    merged = merge_profiles(behavior, geometry)
    validate_profiles(merged)
    merged = add_performance(merged, pd.read_csv(HERE / "validation_rounds.csv"))
    merged.to_csv(HERE / "hybrid_selection_profiles.csv", index=False)
    metrics = ["strict_loading_prior_fraction", "strict_loading_union_fraction", "broad_loading_union_fraction",
               "legacy_v_prior_extension_fraction", "pe_extension_prior_fraction", "pe_extension_union_fraction",
               "unique_molecules", "new_molecule_fraction", "same_molecule_pairs", "exact_prior_tuple_repeat_fraction",
               "cw_selected_distance_percentile_mean", "raw_selected_distance_percentile_mean",
               "cw_common_remaining_U_coverage_distance_relative_drop", "raw_common_remaining_U_coverage_distance_relative_drop"]
    merged.groupby(["seed", "source"])[metrics].mean().to_csv(HERE / "combined_seed_summary.csv")
    merged.groupby(["seed", "source", "stage"])[metrics].mean().to_csv(HERE / "combined_stage_by_seed.csv")
    cap_sensitivity(merged).to_csv(HERE / "cap_behavior_sensitivity.csv", index=False)
    if not args.no_plots:
        plot(merged, HERE)
    record = dict(status="PASS", profiles=len(merged), seeds=[157, 6101], rounds_per_source=15,
                  sources=list(SOURCES), phase_b="PAUSED_BY_USER_DEFINED_EVIDENCE_GATE",
                  real_llm_calls=0, new_model_fits=0, new_labels_revealed=0,
                  limitations=["historical and version-confounded", "two independent seed pairs", "no causal loading effect identified"],
                  input_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("behavior_profiles.csv", "geometry_profiles.csv", "validation_rounds.csv")})
    (HERE / "assembly_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

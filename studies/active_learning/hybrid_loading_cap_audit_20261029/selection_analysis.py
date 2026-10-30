"""Read-only frozen Scientist V2 selection audit; no runners, truth tables or LLM.

The strict prior V definition is imported unchanged from the five-seed audit.
All condition metadata comes from legal frozen catalogs; own observed records
alone supply acquisition-time historical residuals. Batch peers always include
the full 32, including peers across CW/LLM source boundaries.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDIES = ROOT / "studies/active_learning"
REVISIONS = STUDIES / "qgeognn_v2_row_llm_scientist_v2/revisions"
HYBRID_OWNER = REVISIONS / "astra_high_full_pool_20261001"
FREE_CONTINUATION = REVISIONS / "free_sol_xhigh_continuation_20261012"
FREE = "free_llm32_scientist_v2"
HYBRID = "cw16_llm16_scientist_v2"
CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
AXES = {"pe": "PE/EA", "density": "Density g/ml", "v": "V/ul", "loading_type": "loading solvent", "loading_volume": "Volume of loading solvent/ul"}
LOADING_AXES = ("density", "v", "loading_type", "loading_volume")
STAGES = ((333, 525), (525, 653), (653, 813))
SOURCES = {}

spec = importlib.util.spec_from_file_location("five_seed_behavior", STUDIES / "free_lcmd_mechanism_diagnostics_20261029/selection_behavior_analysis.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = digest
    return digest


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).relative_to(ROOT))


def stage(budget):
    return next(f"{a}-{b}" for a, b in STAGES if a <= budget < b)


def free_owner(seed, rnd):
    return HYBRID_OWNER if rnd < (6 if seed == 157 else 10) else FREE_CONTINUATION


def cw_owner(rnd):
    name = ("qgeognn_v2_row_short_sequential_b32" if rnd < 3 else
            "qgeognn_v2_row_cw_lcmd_to_525" if rnd < 6 else
            "qgeognn_v2_row_cw_lcmd_to_653" if rnd < 10 else
            "qgeognn_v2_row_cw_lcmd_to_1005")
    return STUDIES / name


def key(row):
    return row["smiles"], tuple(row["conditions"][c] for c in CONDITIONS)


def differing_axes(a, b):
    if a["smiles"] != b["smiles"]:
        return None
    return tuple(short for short, field in AXES.items() if a["conditions"][field] != b["conditions"][field])


def point_flags(row, prior_rows, batch_rows):
    """Classify single-axis contrasts; extensions must be novel versus L_t.

    Simultaneous batch tuple duplicates are recorded separately. This definition
    does not give one row a privileged temporal position within the batch.
    """
    previous = [p for p in prior_rows if p["smiles"] == row["smiles"]]
    peers = [p for p in batch_rows if p["id"] != row["id"] and p["smiles"] == row["smiles"]]
    exact = any(key(p) == key(row) for p in previous)
    out = {"known_molecule": bool(previous), "new_molecule": not previous,
           "exact_prior_tuple_repeat": exact,
           "exact_batch_tuple_peer": any(key(p) == key(row) for p in peers),
           "same_molecule_batch_peer": bool(peers)}
    for scope, matches in (("prior", previous), ("batch", peers)):
        differences = [differing_axes(row, p) for p in matches]
        out[f"multiple_conditions_changed_{scope}"] = any(len(d) > 1 for d in differences)
        # Broad loading design allows PE/EA and other loading axes to change too.
        out[f"broad_loading_{scope}"] = not exact and any(set(d) & set(LOADING_AXES) for d in differences)
        for axis in AXES:
            out[f"{axis}_matched_contrast_{scope}"] = (axis,) in differences
            out[f"{axis}_extension_{scope}"] = out[f"{axis}_matched_contrast_{scope}"] and not exact
        out[f"strict_loading_{scope}"] = any(out[f"{axis}_extension_{scope}"] for axis in LOADING_AXES)
        out[f"pe_extension_{scope}"] = out[f"pe_extension_{scope}"]
    for prefix in ["strict_loading", "broad_loading", *[f"{axis}_extension" for axis in AXES]]:
        p, b = out[f"{prefix}_prior"], out[f"{prefix}_batch"]
        out.update({f"{prefix}_prior_only": p and not b, f"{prefix}_batch_only": b and not p,
                    f"{prefix}_both": p and b, f"{prefix}_union": p or b})
    old = legacy.local_flags(row, {row["smiles"]: previous} if previous else {})
    out["legacy_v_prior_extension"] = old["amount_curve_extension"]
    out["legacy_v_prior_interpolation"] = old["amount_interpolation"]
    assert out["legacy_v_prior_extension"] == out["v_extension_prior"]
    assert out["exact_prior_tuple_repeat"] == old["exact_condition_repeat"]
    return out


LOADING_TEXT = re.compile(r"loading|amount|density|dose|mass|\b\d+\s*(?:u[lL]|μ[lL]|microlit)", re.I)
PE_TEXT = re.compile(r"PE/EA|eluent|solvent.response|solvent.curve|solvent.ratio|elution.strength", re.I)


def reason_features(choice, row, meta, prior_ids, pending_ids, batch_ids, molecules):
    """Lexical proxies and exact ID checks; no inferred semantic truth claim."""
    text = choice.get("reason", "")
    refs = list(dict.fromkeys(re.findall(r"\b[0-9a-f]{20}\b", text)))
    known_refs = [i for i in refs if i in meta]
    mrefs = list(dict.fromkeys(re.findall(r"\bm\d+\b", text)))
    actual_mrefs = [i for i in mrefs if i in molecules and molecules[i]["smiles"] == row["smiles"]]
    allowed_ids = set(prior_ids) | set(pending_ids) | set(batch_ids)
    axes = [differing_axes(row, meta[i]) for i in known_refs if i in allowed_ids and i != row["id"]]
    reference_details = []
    for i in refs:
        scope = ("unknown_ID" if i not in meta else "prior_L_t" if i in set(prior_ids) else
                 "pending_CW16" if i in set(pending_ids) else "same_batch_LLM" if i in set(batch_ids) else "other_legal_candidate")
        diff = differing_axes(row, meta[i]) if i in meta else None
        reference_details.append({"reference_id": i, "scope": scope, "same_molecule": diff is not None,
                                  "differing_original_axes": ";".join(diff) if diff is not None else "",
                                  "strict_loading_support": diff is not None and len(diff) == 1 and diff[0] in LOADING_AXES and i in allowed_ids})
    return {"reason": text, "reason_loading_mention": bool(LOADING_TEXT.search(text)),
            "reason_pe_mention": bool(PE_TEXT.search(text)),
            "reason_referenced_ids": ";".join(refs), "reason_unknown_id_count": len(refs) - len(known_refs),
            "reason_prior_reference_count": sum(i in set(prior_ids) for i in known_refs),
            "reason_pending_reference_count": sum(i in set(pending_ids) for i in known_refs),
            "reason_batch_reference_count": sum(i in set(batch_ids) for i in known_refs),
            "reason_other_unselected_candidate_reference_count": sum(i not in allowed_ids for i in known_refs),
            "reason_same_molecule_reference_count": sum(d is not None for d in axes),
            "reason_strict_loading_reference_count": sum(d is not None and len(d) == 1 and d[0] in LOADING_AXES for d in axes),
            "reason_strict_pe_reference_count": sum(d == ("pe",) for d in axes),
            "reason_molecule_tokens": ";".join(mrefs),
            "reason_matching_molecule_token_count": len(actual_mrefs),
            "reason_other_molecule_token_count": sum(i in molecules and i not in actual_mrefs for i in mrefs),
            "reason_unknown_molecule_token_count": sum(i not in molecules for i in mrefs),
            "reason_reference_details": json.dumps(reference_details, ensure_ascii=False)}


def validate_llm_batch(directory):
    frozen = read(directory / "batch_freeze.json")
    catalog = read(directory / "catalog.json")
    selection = read(directory / "selection.json")
    packet = read(directory / "input.json")
    for name, expected in frozen["artifact_hashes"].items():
        assert sha(directory / name) == expected, (directory, name)
    assert frozen["new_batch_labels_revealed"] is False
    ids = frozen["batch_ids"]
    pending = frozen.get("pending_ids", [])
    choices = [c["id"] for c in selection["choices"]]
    assert len(ids) == len(set(ids)) == 32
    assert ids == pending + choices
    assert len(pending) == (16 if pending else 0)
    assert len(choices) == (16 if pending else 32)
    assert packet["selection_count"] == len(choices)
    assert packet["validation_test_records"] == 0
    assert set(pending) == {r["id"] for r in catalog["pending"]}
    prior_ids = [r["id"] for r in catalog["observed"]]
    assert len(prior_ids) == packet["active_label_count"]
    assert not set(prior_ids) & set(ids)
    for r in catalog["candidates"] + catalog["pending"]:
        assert not any(k in r for k in ("true_V1_ml", "true_V2_ml", "V1_ml", "V2_ml"))
    return frozen, catalog, selection, packet


def batch_profile(base, selected, batch, prior, pool, meta, choices, packet, chemistry, observed):
    prior_by_smiles = collections.defaultdict(list)
    batch_by_smiles = collections.defaultdict(list)
    for i in prior:
        prior_by_smiles[meta[i]["smiles"]].append(meta[i])
    for i in batch:
        batch_by_smiles[meta[i]["smiles"]].append(meta[i])
    rows, pairs = [], []
    pending = set(batch[:16]) if base["source"].startswith("hybrid") else set()
    allsm = {meta[i]["smiles"] for i in batch}
    cwsm = {meta[i]["smiles"] for i in pending}
    llmsm = {meta[i]["smiles"] for i in batch if i not in pending}
    priorsm = set(prior_by_smiles)
    for order, i in enumerate(selected):
        row = meta[i]
        flags = point_flags(row, prior_by_smiles[row["smiles"]], batch_by_smiles[row["smiles"]])
        record = {**base, "sample_id": i, "source_order": order, "batch_order": batch.index(i),
                  "smiles": row["smiles"], **row["conditions"], **flags,
                  "cw_distance_percentile": row.get("cw_distance_percentile", np.nan),
                  "pending_same_molecule": any(meta[j]["smiles"] == row["smiles"] for j in pending if j != i),
                  "pending_exact_tuple_peer": any(key(meta[j]) == key(row) for j in pending if j != i),
                  "pending_strict_loading_peer": any((d := differing_axes(row, meta[j])) is not None and len(d) == 1 and d[0] in LOADING_AXES for j in pending if j != i),
                  "loading_amount_mg": float(row["conditions"]["Density g/ml"]) * float(row["conditions"]["V/ul"])}
        if i in choices:
            record.update(reason_features(choices[i], row, meta, prior, pending, batch, packet.get("molecules", {})))
        rows.append(record)
        for scope, peers in (("prior", prior_by_smiles[row["smiles"]]), ("batch", batch_by_smiles[row["smiles"]])):
            for other in peers:
                if other["id"] == i:
                    continue
                axes = differing_axes(row, other)
                if len(axes) == 1:
                    axis = axes[0]
                    pairs.append({**base, "sample_id": i, "reference_id": other["id"], "reference_scope": scope,
                                  "reference_source": "prior_L_t" if scope == "prior" else ("hybrid_cw16" if other["id"] in pending else "hybrid_llm16") if pending else "same_batch",
                                  "axis": axis, "is_loading_axis": axis in LOADING_AXES,
                                  "is_extension": not flags["exact_prior_tuple_repeat"],
                                  "exact_prior_tuple_repeat": flags["exact_prior_tuple_repeat"],
                                  "selected_value": row["conditions"][AXES[axis]], "reference_value": other["conditions"][AXES[axis]],
                                  "smiles": row["smiles"]})
    counts = collections.Counter(meta[i]["smiles"] for i in selected)
    tuple_counts = collections.Counter(key(meta[i]) for i in selected)
    result = {**base, "denominator": len(selected), "batch_context_denominator": len(batch),
              "observed_count": len(prior), "candidate_pool_count": len(pool),
              "unique_molecules": len(counts), "same_molecule_pairs": sum(n * (n - 1) // 2 for n in counts.values()),
              "molecule_repeated_rows_beyond_first": len(selected) - len(counts),
              "max_rows_per_molecule": max(counts.values()),
              "distinct_molecule_condition_tuples": len(tuple_counts),
              "within_source_exact_tuple_pairs": sum(n * (n - 1) // 2 for n in tuple_counts.values()),
              "cw_llm_shared_molecule_count": len(cwsm & llmsm) if pending else np.nan,
              "cw_unique_new_molecule_count": len((cwsm - llmsm) - priorsm) if pending else np.nan,
              "llm_unique_new_molecule_count": len((llmsm - cwsm) - priorsm) if pending else np.nan,
              "batch_unique_new_molecule_count": len(allsm - priorsm)}
    flag_names = list(point_flags(meta[selected[0]], [], []))
    flag_names += ["pending_same_molecule", "pending_exact_tuple_peer", "pending_strict_loading_peer"]
    for name in flag_names:
        result[f"{name}_count"] = sum(r[name] for r in rows)
        result[f"{name}_fraction"] = result[f"{name}_count"] / len(rows)
    # Frozen catalog geometry is diagnostic; independent raw/CW recomputation is
    # supplied by geometry_analysis.py rather than ranking incomparable banks.
    result["selected_cw_percentile_mean"] = np.mean([r["cw_distance_percentile"] for r in rows])
    result["selected_cw_above_90_fraction"] = np.mean([r["cw_distance_percentile"] >= .9 for r in rows]) if not np.isnan(rows[0]["cw_distance_percentile"]) else np.nan
    result.update(legacy.high_error_features(selected, pool, observed, meta, chemistry))
    result["history_error_status"] = ("own prior frozen acquisition-time residuals; not current-model error" if observed and base["round"] > 0 else
                                      "initial state: no acquisition-time residual history" if base["round"] == 0 else "NA: control receipt has hashes, no own residual payload")
    for name in ("reason_loading_mention", "reason_pe_mention"):
        text_rows = [r for r in rows if name in r]
        result["reason_choice_denominator"] = len(text_rows)
        result[name + "_fraction"] = sum(r[name] for r in text_rows) / len(rows) if text_rows else np.nan
        result[name + "_among_llm_fraction"] = np.mean([r[name] for r in text_rows]) if text_rows else np.nan
    return result, rows, pairs


def main():
    sha(Path(__file__))
    sha(STUDIES / "free_lcmd_mechanism_diagnostics_20261029/selection_behavior_analysis.py")
    catalogs, metadata = {}, {}
    for seed in (157, 6101):
        c = read(HYBRID_OWNER / f"selections/seed_{seed}/{HYBRID}/round_00/catalog.json")
        catalogs[seed] = c
        for row in c["candidates"] + c["pending"] + c["observed"]:
            clean = {k: row[k] for k in ("id", "smiles", "conditions")}
            assert row["id"] not in metadata or metadata[row["id"]] == clean
            metadata[row["id"]] = clean
    chemistry = legacy.Chemistry(metadata)
    profiles, points, pairs = [], [], []
    for seed in (157, 6101):
        for method in (HYBRID, FREE, "center_width_lcmd", "raw_gradient_lcmd"):
            labeled = [r["id"] for r in catalogs[seed]["observed"]]
            pool0 = [r["id"] for r in catalogs[seed]["candidates"] + catalogs[seed]["pending"]]
            acquired = []
            for rnd in range(15):
                choices, packet, observed = {}, {}, []
                if method in (HYBRID, FREE):
                    owner = HYBRID_OWNER if method == HYBRID else free_owner(seed, rnd)
                    directory = owner / f"selections/seed_{seed}/{method}/round_{rnd:02d}"
                    frozen, raw, selection, packet = validate_llm_batch(directory)
                    batch = frozen["batch_ids"]
                    assert set(labeled) == {r["id"] for r in raw["observed"]}
                    observed = raw["observed"]
                    # Geometry changes each round; the chemical condition map is
                    # checked against L333's full legal candidate metadata.
                    meta = {r["id"]: r for k in ("observed", "pending", "candidates") for r in raw[k]}
                    assert all({k: r[k] for k in ("id", "smiles", "conditions")} == metadata[i] for i, r in meta.items())
                    pool = [r["id"] for r in raw["candidates"] + raw["pending"]]
                    choices = {r["id"]: r for r in selection["choices"]}
                    source_path = directory / "batch_freeze.json"
                    subsets = [("hybrid_cw16", frozen["pending_ids"]), ("hybrid_llm16", list(choices)), ("hybrid_all32", batch)] if method == HYBRID else [("free32", batch)]
                    catalog_path = directory / "catalog.json"
                else:
                    owner = cw_owner(rnd) if method == "center_width_lcmd" else STUDIES / "qgeognn_v2_row_sequential_b32"
                    control = "center_width_lcmd" if method == "center_width_lcmd" else "lcmd"
                    directory = owner / f"runtime/seed_{seed}/{control}/round_{rnd:02d}"
                    source_path = directory / "acquisition_artifacts/selected_next_batch.csv"
                    sha(source_path)
                    batch = pd.read_csv(source_path).sample_id.tolist()
                    pool = [i for i in pool0 if i not in set(acquired)]
                    meta = metadata
                    catalog_path = HYBRID_OWNER / f"selections/seed_{seed}/{HYBRID}/round_00/catalog.json"
                    subsets = [("cw_lcmd32" if method == "center_width_lcmd" else "raw_lcmd32", batch)]
                assert len(batch) == len(set(batch)) == 32
                assert not set(batch) & set(labeled)
                assert set(batch) <= set(pool)
                assert len(labeled) == 333 + 32 * rnd
                for source, selected in subsets:
                    base = {"seed": seed, "method": method, "round": rnd, "labels_before": len(labeled), "labels_after": len(labeled) + 32,
                            "source": source, "stage": stage(len(labeled)), "selection_source": rel(source_path),
                            "selection_sha256": sha(source_path), "catalog_source": rel(catalog_path), "catalog_sha256": sha(catalog_path),
                            "owner": owner.name}
                    profile, point_rows, pair_rows = batch_profile(base, selected, batch, labeled, pool, meta, choices, packet, chemistry, observed)
                    profiles.append(profile)
                    points.extend(point_rows)
                    pairs.extend(pair_rows)
                labeled += batch
                acquired += batch
    df, pts, contrasts = pd.DataFrame(profiles), pd.DataFrame(points), pd.DataFrame(pairs)
    df.to_csv(HERE / "behavior_profiles.csv", index=False)
    pts.to_csv(HERE / "selection_point_flags.csv", index=False)
    contrasts.to_csv(HERE / "loading_matched_contrasts.csv", index=False)
    refs = []
    for r in pts.query("source in ['free32','hybrid_llm16']").to_dict("records"):
        for item in json.loads(r["reason_reference_details"]):
            refs.append({**{k: r[k] for k in ("seed", "round", "source", "stage", "sample_id", "selection_source")}, **item, "reason": r["reason"]})
    pd.DataFrame(refs).to_csv(HERE / "reason_id_reference_audit.csv", index=False)
    selected_columns = ["denominator", "unique_molecules", "molecule_repeated_rows_beyond_first", "within_source_exact_tuple_pairs",
                        "strict_loading_prior_fraction", "strict_loading_batch_only_fraction", "strict_loading_both_fraction", "strict_loading_union_fraction",
                        "legacy_v_prior_extension_fraction", "pe_extension_prior_fraction", "pe_extension_union_fraction", "broad_loading_union_fraction",
                        "new_molecule_fraction", "pending_same_molecule_fraction", "pending_strict_loading_peer_fraction",
                        "selected_cw_percentile_mean", "cw_llm_shared_molecule_count", "cw_unique_new_molecule_count", "llm_unique_new_molecule_count",
                        "reason_loading_mention_fraction", "reason_pe_mention_fraction", "high8_same_molecule_fraction", "high8_same_molecule_enrichment"]
    grouped = df.groupby(["seed", "source", "stage"], as_index=False)[selected_columns].mean()
    grouped.to_csv(HERE / "behavior_stage_by_seed.csv", index=False)
    all_window = df.groupby(["seed", "source"], as_index=False)[selected_columns].mean()
    all_window.to_csv(HERE / "behavior_window_by_seed.csv", index=False)
    # Check the published 34.65% figure in its actual five-seed, 36-round window.
    old_profiles_path = STUDIES / "free_lcmd_mechanism_diagnostics_20261029/behavior_profiles.csv"
    sha(old_profiles_path)
    old_profiles = pd.read_csv(old_profiles_path)
    old_free = old_profiles.query("method == @FREE")
    match = old_free.query("seed in [157,6101] and round < 15").sort_values(["seed", "round"])
    current = df.query("source == 'free32'").sort_values(["seed", "round"])
    assert np.array_equal(match.amount_curve_extension_fraction.to_numpy(), current.legacy_v_prior_extension_fraction.to_numpy())
    text = pts.query("source in ['free32','hybrid_llm16']").copy()
    text_summary = []
    for (seed, source), g in text.groupby(["seed", "source"]):
        mentioned = g.reason_loading_mention.fillna(False).astype(bool)
        actual = g.strict_loading_union
        text_summary.append({"seed": int(seed), "source": source, "choices": len(g),
                             "loading_word_choices": int(mentioned.sum()), "loading_word_and_actual_strict": int((mentioned & actual).sum()),
                             "loading_word_without_strict": int((mentioned & ~actual).sum()), "strict_without_loading_word": int((~mentioned & actual).sum()),
                             "reason_unknown_id_count": int(g.reason_unknown_id_count.sum()),
                             "reason_other_molecule_token_count": int(g.reason_other_molecule_token_count.sum()),
                             "reason_unknown_molecule_token_count": int(g.reason_unknown_molecule_token_count.sum()),
                             "reason_exact_strict_loading_reference_total": int(g.reason_strict_loading_reference_count.sum())})
    pd.DataFrame(text_summary).to_csv(HERE / "reason_behavior_audit.csv", index=False)
    summary = {"window": "333-813; acquisitions r0..14; two seeds", "profiles": len(df), "point_source_records": len(pts),
               "independent_trajectory_selected_rows": len(pts.query("source not in ['hybrid_cw16','hybrid_llm16']")),
               "historical_alignment": {"five_seed_all_36_round_v_only_fraction": float(old_free.amount_curve_extension_fraction.mean()),
                                        "same_seed_same_window_exact_v_only_match": True,
                                        "published_34_65_not_this_window": True},
               "strict_loading_definition": "same SMILES, exactly one of V/Density/loading solvent volume/type differs, other four original conditions identical; selected tuple absent from prior L_t; prior-or-full-batch union counted once per selected row",
               "derived_mass": "Density * V (mg), descriptive only; not a sixth condition or separately counted axis",
               "batch_definition": "all 32 peers; source subset denominator 16, full denominator32; both = prior and batch, batch_only = batch and not prior",
               "replicate_policy": "full tuple already in prior L_t cannot be extension; simultaneous exact tuple peer flagged separately without assigning within-batch chronology",
               "broad_definition": "same molecule and any loading axis differs, including multiaxis changes; not strict matched contrast",
               "residual_boundary": "own prior acquisition-time frozen observed residuals only; control residuals NA; not current-model high-error region",
               "comparison_boundary": "observational matched seed/budget; L_t and models differ after round0; no causal significance test from non-independent rows/rounds",
               "dialog_v1": "excluded; independent old lineage, not stitched into V2",
               "all_window_by_seed_source": all_window.replace({np.nan: None}).to_dict("records"),
               "stage_by_seed_source": grouped.replace({np.nan: None}).to_dict("records"), "reason_proxy_audit": text_summary,
               "phase_b_gate": "PAUSE: internal LLM loading is not consistently lower; late seed6101 full hybrid is higher; no uniform K identified",
               "source_hashes": SOURCES}
    (HERE / "behavior_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_findings(df, all_window, grouped, text_summary, pts)
    print(all_window[["seed", "source", "strict_loading_prior_fraction", "strict_loading_union_fraction", "legacy_v_prior_extension_fraction", "pe_extension_union_fraction", "new_molecule_fraction", "unique_molecules"]].to_string(index=False))


def table(frame, columns):
    def cell(v):
        if pd.isna(v):
            return "NA"
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame[columns].itertuples(index=False, name=None)])


def write_findings(df, window, stages, text_summary, pts):
    columns = ["seed", "source", "strict_loading_prior_fraction", "strict_loading_union_fraction", "legacy_v_prior_extension_fraction", "pe_extension_union_fraction", "broad_loading_union_fraction", "new_molecule_fraction", "unique_molecules"]
    body = ["# Frozen Scientist V2 selection behavior audit", "",
            "Two seeds, acquisitions r0–14, 333→813. Fractions are per selected row; CW16 and LLM16 denominator16, Hybrid/Free/CW32/raw-LCMD denominator32. All source rows use the full32 peer context. These are matched-budget observational comparisons; rounds/points are not independent replicates, and no population significance or causal claim is made.", "",
            "## Whole-window results by seed", "", table(window, columns), "",
            "`strict_loading_prior` uses own prebatch L_t only. `strict_loading_union` adds full-batch peers and counts each point once. The legacy V-only metric exactly reproduces the previous five-seed audit in the same seed/window; the published34.65% is the other five-seed 333→1485 window. Density×V is a derived mass and receives no extra count.", "",
            "## Stage-specific strict loading", "", table(stages, ["seed", "stage", "source", "strict_loading_prior_fraction", "strict_loading_batch_only_fraction", "strict_loading_both_fraction", "strict_loading_union_fraction", "new_molecule_fraction"]), "",
            "## Interpretation and Phase B gate", "",
            "Free has more strict loading than Hybrid all32 on average in each seed, but not more than its internal LLM16 consistently. In seed157 their full-window union fractions are nearly identical; in seed6101 Hybrid LLM16 is higher. CW16 has zero prior-L_t strict loading at every round, so much of the full32 contrast is mechanically induced by the16-CW quota. Batch-only cross-source contrasts add a small nonzero CW contribution.", "",
            "The stage pattern is heterogeneous: seed157 early Free loading is lower than Hybrid LLM16; seed6101 middle and late Free loading is lower than Hybrid LLM16. In seed6101 at653–813 even Hybrid all32 has higher strict loading than Free. Original H3 asks whether pending CW exposure changes LLM loading preferences and does not specify a direction. Heterogeneous differences are observed, but they cannot be attributed to pending exposure because selection size, trajectory state and other inputs also differ. The stronger subhypothesis that pending CW uniformly lowers loading is not supported. H1/H4 cannot be inferred from historical performance curves. H2 remains compatible with complementary geometric and local designs, subject to the separate geometry audit.", "",
            "Loading is not uniformly the largest observed distinction. In seed157 Free-minus-Hybrid all32 strict-union loading differs by14.17 percentage points and PE/EA by4.79 points. In seed6101 PE/EA differs by25.63 points, larger than loading's11.46 points; Free also selects19.27 distinct molecules per batch versus Hybrid28.53. Broad loading-related design reverses the strict comparison: Free70.42%/72.92% versus Hybrid74.38%/75.00%. The method distinction includes response-curve design and molecule concentration, not only loading.", "",
            "Pause Phase B implementation under the user's stated gate. There is no defensible common main K from these two trajectories. If a future same-version paired acquisition study is separately authorized, K=8 or12 could be transparent25%/37.5% constraints, but neither is preregistered or selected by validation results here. Their heterogeneous binding rates are reported below; these are descriptive candidates, not a recommended main cap.", "",
            "## Source division, overlap and duplicates", "", table(window, ["seed", "source", "unique_molecules", "molecule_repeated_rows_beyond_first", "cw_llm_shared_molecule_count", "cw_unique_new_molecule_count", "llm_unique_new_molecule_count", "pending_same_molecule_fraction", "pending_strict_loading_peer_fraction", "within_source_exact_tuple_pairs"]), "",
            "A new molecule means absent from that method's own L_t. CW's coverage need not mean new chemistry: it operates in learned response-gradient geometry and often chooses known molecules in new condition regions. Molecule identity overlap with pending CW points is not identical point selection; frozen CW/LLM IDs are disjoint by construction. Counts of unique new molecules exclude molecules selected by the other source in the same batch.", "",
            "## Reasons versus actual IDs: lexical proxies", "", table(pd.DataFrame(text_summary), list(text_summary[0])), "",
            "Loading vocabulary is only a text proxy. `reason_loading_mention` is a conservative literal regex, not a judgment that a claim is true or causal. Actual strict classification uses selected IDs and five frozen conditions. Exact20-hex reference IDs, SMILES, and packet-local molecule tokens are audited independently. Other molecule tokens can be legitimate analog comparisons, so they are not labeled erroneous. Other unselected candidate references were legally visible but are not observed/pending/full-batch contrast support. A zero lexical match can still be a real matched contrast; a nonzero match can describe broad/multiaxis design rather than strict completion. Profile reason fractions use that source's16/32 denominator, while `_among_llm_fraction` uses explicit `reason_choice_denominator`.", "",
            "## Historical residual boundary", "",
            "Historical high8-neighborhood features use only each LLM trajectory's own prebatch observed acquisition-time errors. They describe whether choices revisit previously revealed errors; they do not establish current-model high-error neighborhoods. Initial333 rows lack acquisition-time errors, and CW/raw-LCMD receipts do not carry own frozen residual payloads, so unavailable control values remain NA. No candidate truth or held-out/test truth was loaded.", "",
            ""]
    axis_cols = [f"{a}_extension_union_count" for a in LOADING_AXES]
    axis_totals = df.groupby(["seed", "source"], as_index=False)[axis_cols + ["strict_loading_union_count"]].sum()
    body += ["## Single-axis extension totals", "", table(axis_totals, ["seed", "source", *axis_cols, "strict_loading_union_count"]), "",
             "One point can have different single-axis contrasts against different references. Axis counts therefore may overlap; the primary strict-loading union counts each selected point once. Density-only extensions are zero in this frozen window. CW32 and raw-gradient LCMD32 have zero strict loading in333→813; the historical6.65% raw-LCMD figure belongs to the longer five-seed window.", "",
             "## Own historical error neighborhoods", "", table(window, ["seed", "source", "high8_same_molecule_fraction", "high8_same_molecule_enrichment"]), "",
             "Across rounds with prior acquisitions, Hybrid LLM16 selects a molecule in its own historical high8-error set at17.86%/18.30%, compared with CW16 at4.46%/3.57%. Free is8.93%/14.73%. The pools and residual histories differ, so this supports a descriptive division of attention rather than a controlled effect of pending CW exposure.", ""]
    candidates = []
    for (seed, st), g in df.query("source == 'free32'").groupby(["seed", "stage"]):
        for k in (8, 12):
            candidates.append({"seed": seed, "stage": st, "candidate_K": k, "rounds": len(g), "rounds_above_K": int((g.strict_loading_union_count > k).sum()), "mean_count": g.strict_loading_union_count.mean(), "min": int(g.strict_loading_union_count.min()), "max": int(g.strict_loading_union_count.max())})
    pd.DataFrame(candidates).to_csv(HERE / "hypothetical_cap_binding_rates.csv", index=False)
    body += ["## Hypothetical cap binding rates (not preregistration)", "", table(pd.DataFrame(candidates), list(candidates[0]))]
    body += ["", "## Auditable examples", ""]
    for (seed, source), g in pts.query("source in ['free32','hybrid_llm16']").groupby(["seed", "source"]):
        actual = g[g.strict_loading_union & g.reason_loading_mention.fillna(False).astype(bool)]
        if len(actual):
            r = actual.iloc[0]
            body += [f"- Seed{seed} {source}, r{r['round']}, ID`{r.sample_id}`: strict loading union=True; reason: “{r.reason}”"]
        unsupported = g[g.reason_loading_mention.fillna(False).astype(bool) & ~g.strict_loading_union]
        if len(unsupported):
            r = unsupported.iloc[0]
            body += [f"- Seed{seed} {source}, r{r['round']}, ID`{r.sample_id}`: loading vocabulary but strict=False (may be broad design); reason: “{r.reason}”"]
    (HERE / "BEHAVIOR_FINDINGS.md").write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    main()

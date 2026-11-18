"""Recompute F04–F05 from saved metadata, selection records and audit tables."""
from collections import Counter, defaultdict
from csv import DictReader, DictWriter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import mean

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[2]
OUT = REPORT / "sources"
ACTIVE = ROOT / "studies/active_learning"
HYBRID = ACTIVE / "decision_mechanism_followup_20261117"
RETRY = ACTIVE / "volume_response_retry_audit_20261117"
METHODS = {
    "historical_free_mixed_model": "Historical CW-Free32",
    "new_free_cw_sol61_xhigh": "New CW-Free32",
    "nocw": "NoCW-Free32",
}
ARMS = {"lcmd_back16": "CW-LCMD 后16", "llm16": "Hybrid 的 LLM 后16"}
SEEDS = "157/887/2357/6101/12203"


def read_json(path, expected=None):
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if expected is not None:
        assert digest == expected, path
    return json.loads(data), digest


def source_name(path):
    return str(path.relative_to(ROOT))


def table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(DictReader(handle))


def write_table(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key(row, include_method=False):
    state = int(row["seed"]), int(row["round"])
    return (row["method"], *state) if include_method else state


def group_rows(rows, key_function):
    groups = defaultdict(list)
    for row in rows:
        groups[key_function(row)].append(row)
    return groups


def true(value):
    assert value in ("True", "False"), value
    return value == "True"


def close(a, b):
    return isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)


def direction(value):
    return 0 if abs(value) <= 1e-12 else (1 if value > 0 else -1)


def hybrid_rows():
    batches = table(HYBRID / "counterfactual_batches.csv")
    identity_records = read_json(HYBRID / "counterfactual_selected_ids.json")[0]
    identities = {key(row): row for row in identity_records}
    source_manifest = read_json(HYBRID / "counterfactual_input_manifest.json")[0]
    frozen_sources = {}
    for entry in source_manifest:
        path = Path(entry["path"])
        if path.name == "batch_freeze.json":
            seed = int(path.parts[-4].removeprefix("seed_"))
            round_number = int(path.parts[-2].removeprefix("round_"))
            assert (seed, round_number) not in frozen_sources
            frozen_sources[(seed, round_number)] = path, entry["sha256"]
    trace = group_rows(table(HYBRID / "counterfactual_lcmd_trace.csv"), key)
    candidates = group_rows(
        table(HYBRID / "counterfactual_candidates.csv"),
        lambda row: (*key(row), row["group"]),
    )
    metadata = {
        (*key(row), row["group"]): row
        for row in table(HYBRID / "counterfactual_independent_metadata.csv")
    }
    assert len(batches) == len(identities) == len(trace) == len(frozen_sources) == 71
    assert len(candidates) == len(metadata) == 142

    rows = []
    for saved in batches:
        state = key(saved)
        identity = identities[state]
        freeze_path, expected_freeze_sha = frozen_sources[state]
        freeze, freeze_sha = read_json(ROOT / freeze_path, expected_freeze_sha)
        selection, selection_sha = read_json(
            ROOT / freeze_path.with_name("selection.json"), freeze["artifact_hashes"]["selection.json"]
        )
        ordered_trace = sorted(trace[state], key=lambda row: int(row["selection_order"]))
        trace_ids = [row["sample_id"] for row in ordered_trace]
        assert [int(row["selection_order"]) for row in ordered_trace] == list(range(32))
        historical_ids = [choice["id"] for choice in selection["choices"]]
        assert len(set(trace_ids)) == 32 and len(set(historical_ids)) == 16
        assert trace_ids[:16] == freeze["pending_ids"] == identity["reconstructed_first16"]
        assert trace_ids[16:] == identity["reconstructed_back16"]
        assert historical_ids == identity["actual_llm_ids"]
        assert freeze["batch_ids"] == trace_ids[:16] + historical_ids
        overlap = len(set(trace_ids[16:]) & set(historical_ids))
        assert overlap == int(saved["llm_cf_candidate_overlap"])
        h_distance = float(saved["same_state_reference_before_h"])
        hn_distance = float(saved["same_state_reference_before_hn"])
        first_gain = (h_distance - hn_distance) / h_distance
        assert close(first_gain, saved["cw_first16_gain_relative_h"])
        assert int(saved["reference_n"]) == 128
        record = dict(
            seed=state[0], round=state[1], L_before=int(saved["L_before"]),
            L_after=int(saved["L_after"]), phase=saved["phase"],
            first16_exact=True, back16_overlap=overlap, reference_n=128,
            cw_first16_gain_relative_h=first_gain, freeze=str(freeze_path),
            freeze_sha256=freeze_sha, selection_sha256=selection_sha,
        )
        for arm, expected_ids in (("lcmd_back16", trace_ids[16:]), ("llm16", historical_ids)):
            arm_key = (*state, arm)
            slots = sorted(candidates[arm_key], key=lambda row: int(row["comparison_order"]))
            assert [row["sample_id"] for row in slots] == expected_ids
            assert all(not true(row["true_labels_read"]) for row in slots)
            volume_hn = sum(true(row["single_sample_volume_hn"]) for row in slots)
            pe_hn = sum(true(row["single_pe_hn"]) for row in slots)
            assert close(volume_hn / 16, metadata[arm_key]["sample_only_HN_fraction"])
            assert close(volume_hn / 16, saved[f"{arm}_single_sample_volume_hn_fraction"])
            assert close(pe_hn / 16, saved[f"{arm}_single_pe_hn_fraction"])
            known_h = round(16 * float(metadata[arm_key]["sample_only_H_fraction"]))
            assert close(known_h / 16, saved[f"{arm}_single_sample_volume_history_fraction"])
            volume_pairs = int(metadata[arm_key]["N_plus_group_sample_pairs"])
            assert volume_pairs == int(saved[f"{arm}_full32_single_sample_volume_pairs"])
            gain = float(saved[f"{arm}_cw_gain_after_n"])
            assert close(gain / h_distance, saved[f"{arm}_cw_gain_after_n_relative_h"])
            record.update({
                f"{arm}_volume_h": known_h,
                f"{arm}_volume_hn": volume_hn,
                f"{arm}_volume_within_full32_pairs": volume_pairs,
                f"{arm}_pe_hn": pe_hn,
                f"{arm}_pe_h": round(16 * float(saved[f"{arm}_single_pe_history_fraction"])),
                f"{arm}_cw_gain_relative_h": gain / h_distance,
            })
        rows.append(record)
    return rows


def failure_category(error):
    known = {
        "must select distinct legal supplied candidate IDs": "candidate_identity",
        "selection packet binding mismatch": "packet_binding",
        "unauthorized hypothesis fields": "field_whitelist",
        "wrong selection count or choice structure": "selection_count_or_structure",
    }
    if "cites unobserved evidence" in error:
        return "evidence_membership"
    return known[error]


def prefix_rows():
    batches = table(RETRY / "retry_case_batches.csv")
    repair = {
        key(row, True): row for row in table(RETRY / "local_repair_cases.csv")
    }
    repair_slots = group_rows(
        table(RETRY / "local_repair_candidates.csv"), lambda row: key(row, True)
    )
    frozen_audit = {
        key(row, True): row for row in table(HYBRID / "llm_display_prefix_batches.csv")
    }
    assert len(batches) == len(frozen_audit) == 540
    assert len(repair) == len(repair_slots) == 127

    rows = []
    for saved in batches:
        state = key(saved, True)
        freeze_path = ROOT / saved["source"]
        freeze, freeze_sha = read_json(freeze_path, frozen_audit[state]["source_sha256"])
        packet, packet_sha = read_json(
            freeze_path.with_name("input.json"), freeze["artifact_hashes"]["input.json"]
        )
        selection, selection_sha = read_json(
            freeze_path.with_name("selection.json"), freeze["artifact_hashes"]["selection.json"]
        )
        id_column = packet["candidates"]["columns"].index("id")
        pool = [row[id_column] for row in packet["candidates"]["rows"]]
        final_ids = [choice["id"] for choice in selection["choices"]]
        assert len(final_ids) == len(set(final_ids)) == 32
        assert final_ids == freeze["batch_ids"]
        prefix = final_ids == pool[:32]
        assert prefix == true(saved["prefix_final"]) == true(frozen_audit[state]["ordered_first32"])

        first_failure = "none"
        error_sha = ""
        if saved["first_failure_source"]:
            error_record, error_sha = read_json(ROOT / saved["first_failure_source"])
            assert error_record["error"] == saved["first_failure_error"]
            first_failure = failure_category(error_record["error"])
        assert first_failure == saved["first_failure_category"]

        exact_receipts = int(saved["accepted_receipt_exact_matches"])
        assert exact_receipts in (0, 1)
        accepted_receipt_sha = ""
        if exact_receipts == 1:
            accepted_path = freeze_path.with_name(f"turn_{int(float(saved['accepted_turn'])):02d}.json")
            accepted, accepted_receipt_sha = read_json(
                accepted_path, freeze["artifact_hashes"][accepted_path.name]
            )
            assert json.loads(accepted["answer"]) == selection

        first_receipt_sha = ""
        initial_choices = ""
        original_legal = ""
        final_retained = ""
        preserved = ""
        vacant = ""
        binding_fixed = ""
        science_review = ""
        gate_pass = ""
        if prefix:
            assert exact_receipts == 1
            first_path = ROOT / saved["first_receipt_source"]
            first_record, first_receipt_sha = read_json(
                first_path, freeze["artifact_hashes"][first_path.name]
            )
            first_answer = first_record["answer"]
            assert sha256(first_answer.encode()).hexdigest() == saved["first_answer_sha256"]
            first_ids = [choice["id"] for choice in json.loads(first_answer)["choices"]]
            assert first_ids != pool[:32] and len(first_ids) == 32
            first_legal = set(first_ids) & set(pool)
            initial_choices = len(first_ids)
            original_legal = len(first_legal)
            final_retained = len(first_legal & set(final_ids))
            assert original_legal == int(saved["first_unique_legal_n"])
            assert final_retained == int(saved["first_legal_retained_n"])
            repair_case = repair[state]
            slot_rows = sorted(repair_slots[state], key=lambda row: int(row["original_order"]))
            assert [row["original_id"] for row in slot_rows] == first_ids
            kept_slots = [row for row in slot_rows if true(row["retained"])]
            assert all(
                row["original_choice_sha256"] == row["retained_choice_sha256"]
                and true(row["reason_unchanged"]) and true(row["hypothesis_reference_unchanged"])
                for row in kept_slots
            )
            assert {row["original_id"] for row in kept_slots} == first_legal
            preserved = len(kept_slots)
            vacant = initial_choices - preserved
            binding_fixed = true(repair_case["binding_restored_from_verified_request"])
            science_review = true(repair_case["science_review_needed"])
            gate_pass = true(repair_case["local_draft_passes_historical_gate"])
            assert preserved == int(repair_case["preserved_unique_legal_n"])
            assert vacant == int(repair_case["vacant_slots"])
            assert final_retained == int(repair_case["historical_final_retained_original_legal_n"])

        rows.append(dict(
            method=saved["method"], seed=state[1], round=state[2],
            L_before=int(saved["L_before"]), L_after=int(saved["L_after"]),
            prefix_final=prefix, first_failure=first_failure,
            accepted_exact_receipt=bool(exact_receipts), initial_choices=initial_choices,
            original_legal=original_legal, historical_retained=final_retained,
            local_repair_preserved=preserved, local_repair_vacant=vacant,
            local_repair_binding_fixed=binding_fixed, local_repair_science_review=science_review,
            local_repair_passes_gate=gate_pass, freeze=saved["source"], freeze_sha256=freeze_sha,
            input_sha256=packet_sha, selection_sha256=selection_sha, rejected_file_sha256=error_sha,
            first_receipt_sha256=first_receipt_sha, accepted_receipt_sha256=accepted_receipt_sha,
        ))
    return rows


def summarize_hybrid(rows):
    out = []
    for seed, states in sorted(group_rows(rows, lambda row: row["seed"]).items()):
        for scope, scope_rows in (
            ("all", states),
            ("common_L1453", [row for row in states if row["L_after"] <= 1453]),
        ):
            no_overlap = sum(row["back16_overlap"] == 0 for row in scope_rows)
            cw_directions = Counter(direction(
                row["llm16_cw_gain_relative_h"] - row["lcmd_back16_cw_gain_relative_h"]
            ) for row in scope_rows)
            volume_directions = Counter(direction(
                row["llm16_volume_hn"] - row["lcmd_back16_volume_hn"]
            ) for row in scope_rows)
            for arm, method in ARMS.items():
                out.append(dict(
                    seed=seed, scope=scope, method=method, method_key=arm,
                    L_start=min(row["L_before"] for row in scope_rows),
                    L_end=max(row["L_after"] for row in scope_rows), batches=len(scope_rows),
                    slots=16 * len(scope_rows), first16_exact=sum(
                        row["first16_exact"] for row in scope_rows
                    ),
                    no_overlap=no_overlap, cw_delta_negative=cw_directions[-1],
                    cw_delta_zero=cw_directions[0], cw_delta_positive=cw_directions[1],
                    volume_hn_delta_positive=volume_directions[1],
                    volume_hn_delta_zero=volume_directions[0],
                    volume_hn_delta_negative=volume_directions[-1],
                    cw_first16_gain_pct=100 * mean(
                        row["cw_first16_gain_relative_h"] for row in scope_rows
                    ),
                    volume_h_slots=sum(row[f"{arm}_volume_h"] for row in scope_rows),
                    volume_hn_slots=sum(row[f"{arm}_volume_hn"] for row in scope_rows),
                    volume_within_full32_pairs=sum(
                        row[f"{arm}_volume_within_full32_pairs"] for row in scope_rows
                    ),
                    pe_h_slots=sum(row[f"{arm}_pe_h"] for row in scope_rows),
                    pe_hn_slots=sum(row[f"{arm}_pe_hn"] for row in scope_rows),
                    cw_gain_pct=100 * mean(row[f"{arm}_cw_gain_relative_h"] for row in scope_rows),
                ))
    return out


def summarize_prefix(rows):
    out = []
    for method, states in (*group_rows(rows, lambda row: row["method"]).items(), ("all", rows)):
        prefix = [row for row in states if row["prefix_final"]]
        failures = Counter(row["first_failure"] for row in prefix)
        out.append(dict(
            method=METHODS.get(method, "全部 Free32"),
            method_key=method, batches=len(states), prefix_batches=len(prefix),
            first_failure_batches=sum(row["first_failure"] != "none" for row in states),
            prefix_candidate_identity=failures["candidate_identity"],
            prefix_packet_binding=failures["packet_binding"],
            accepted_exact_receipt=sum(row["accepted_exact_receipt"] for row in states),
            original_slots=sum(row["initial_choices"] for row in prefix),
            original_legal=sum(row["original_legal"] for row in prefix),
            historical_retained=sum(row["historical_retained"] for row in prefix),
            local_repair_preserved=sum(row["local_repair_preserved"] for row in prefix),
            local_repair_vacant=sum(row["local_repair_vacant"] for row in prefix),
            local_repair_binding_fixed=sum(row["local_repair_binding_fixed"] for row in prefix),
            local_repair_science_review=sum(row["local_repair_science_review"] for row in prefix),
            local_repair_passes_gate=sum(row["local_repair_passes_gate"] for row in prefix),
            first_31_legal=sum(row["original_legal"] == 31 for row in prefix),
        ))
    return out


def metric_rows(hybrid, prefix):
    rows = []
    h_source = "; ".join(source_name(HYBRID / name) for name in (
        "counterfactual_batches.csv", "counterfactual_candidates.csv",
        "counterfactual_independent_metadata.csv", "counterfactual_selected_ids.json",
        "counterfactual_lcmd_trace.csv", "counterfactual_input_manifest.json",
    ))
    p_source = "; ".join(source_name(path) for path in (
        RETRY / "retry_case_batches.csv", RETRY / "local_repair_cases.csv",
        RETRY / "local_repair_candidates.csv", HYBRID / "llm_display_prefix_batches.csv",
    ))

    def f04(metric, method, value, unit, n, seeds, budget):
        return dict(
            figure="F04", metric=metric, method=method, value=value, unit=unit, n=n,
            seeds=seeds, budget=budget, level="row selection batches",
            split="history and candidate metadata", version="20261117 Hybrid replay",
            frozen="on the same historical frozen state",
            comparability="paired within state, same ordered numeric first 16",
            limitation="two seeds; sequential states; no counterfactual response or Row gain",
            source=h_source,
        )

    for r in hybrid:
        budget = f"L{r['L_start']}–{r['L_end']}; {r['scope']}"
        for metric, value, unit, n in (
            ("Sample-volume-only slots relative to H+first16", 100 * r["volume_hn_slots"] / r["slots"], "%", r["slots"]),
            ("Sample-volume-only slots relative to H", 100 * r["volume_h_slots"] / r["slots"], "%", r["slots"]),
            ("PE/EA-only slots relative to H", 100 * r["pe_h_slots"] / r["slots"], "%", r["slots"]),
            ("Sample-volume-only pairs within full32", r["volume_within_full32_pairs"] / r["batches"], "pairs per batch", r["batches"]),
            ("Second-half CW coverage gain relative to H", r["cw_gain_pct"], "%", r["batches"]),
        ):
            rows.append(f04(metric, r["method"], value, unit, n, r["seed"], budget))
        if r["method_key"] == "llm16":
            rows.append(f04("First16 CW coverage gain relative to H", "shared CW-LCMD first16",
                            r["cw_first16_gain_pct"], "%", r["batches"], r["seed"], budget))

    full = [row for row in hybrid if row["scope"] == "all" and row["method_key"] == "llm16"]
    for metric, field in (
        ("Ordered first16 verified", "first16_exact"), ("Zero shared second-half IDs", "no_overlap"),
        ("CW gain delta LLM–LCMD below zero", "cw_delta_negative"),
        ("CW gain delta LLM–LCMD equal to zero", "cw_delta_zero"),
        ("CW gain delta LLM–LCMD above zero", "cw_delta_positive"),
        ("Volume share delta LLM–LCMD above zero", "volume_hn_delta_positive"),
        ("Volume share delta LLM–LCMD equal to zero", "volume_hn_delta_zero"),
    ):
        rows.append(f04(metric, "paired replay", sum(r[field] for r in full), "batches", 71,
                        "157/6101", "L333–1485 (157), L333–1453 (6101)"))

    for r in prefix:
        for metric, value, unit, n in (
            ("Ordered display prefix", 100 * r["prefix_batches"] / r["batches"], "%", r["batches"]),
            ("Exact accepted receipts", r["accepted_exact_receipt"], "batches", r["batches"]),
            ("Prefix with initial rejection", r["prefix_candidate_identity"] + r["prefix_packet_binding"], "batches", r["prefix_batches"]),
            ("Prefix first failure candidate identity", r["prefix_candidate_identity"], "batches", r["prefix_batches"]),
            ("Prefix first failure packet binding", r["prefix_packet_binding"], "batches", r["prefix_batches"]),
            ("Historical first legal IDs retained", 100 * r["historical_retained"] / r["original_legal"], "%", r["original_legal"]),
            ("Local repair original slots preserved", 100 * r["local_repair_preserved"] / r["original_slots"], "%", r["original_slots"]),
            ("Local repair vacancy count", r["local_repair_vacant"], "slots", r["original_slots"]),
            ("Local repair full validator pass", r["local_repair_passes_gate"], "batches", r["prefix_batches"]),
            ("Local repair science review pending", r["local_repair_science_review"], "batches", r["prefix_batches"]),
        ):
            rows.append(dict(
                figure="F05", metric=metric, method=r["method"], value=value, unit=unit,
                n=n, seeds=SEEDS, budget="L333–1485", level="row selection batches or slots",
                split="historical selection, no outcome", version="20261117 retry audit/dry-run",
                frozen="saved requests/receipts, offline repair drafts",
                comparability="observational; repair panel conditioned on prefix subset",
                limitation="no automatic fallback, no future repair success/Row gain inference",
                source=p_source,
            ))
    return rows


def main():
    hybrid = hybrid_rows()
    prefix = prefix_rows()
    hybrid_summary = summarize_hybrid(hybrid)
    prefix_summary = summarize_prefix(prefix)
    all_free = prefix_summary[-1]
    assert all_free["batches"] == 540 and all_free["prefix_batches"] == 127
    assert all_free["prefix_candidate_identity"] + all_free["prefix_packet_binding"] == 127
    assert all_free["local_repair_preserved"] + all_free["local_repair_vacant"] == all_free["original_slots"]
    assert len(hybrid) == 71
    write_table("f04_replay_states.csv", hybrid)
    write_table("f04_replay_seed_summary.csv", hybrid_summary)
    write_table("f05_prefix_batches.csv", prefix)
    write_table("f05_prefix_repair_summary.csv", prefix_summary)
    write_table("behavior_metrics_fragment.csv", metric_rows(hybrid_summary, prefix_summary))

    registered = [
        (HYBRID, "HYBRID_COUNTERFACTUAL.md counterfactual_batches.csv counterfactual_candidates.csv "
         "counterfactual_independent_metadata.csv counterfactual_selected_ids.json counterfactual_lcmd_trace.csv "
         "counterfactual_input_manifest.json counterfactual_independent_verification.json "
         "llm_display_prefix_batches.csv figures/hybrid_counterfactual_tradeoff.pdf"),
        (RETRY, "PREFIX_FAILURE_ANALYSIS.md LOCAL_REPAIR_DESIGN.md AUDIT_FIGURES.md retry_case_batches.csv "
         "local_repair_cases.csv local_repair_candidates.csv retry_audit_verification.json "
         "local_repair_verification.json audit_figures/c2_first_failure_and_prefix.pdf "
         "audit_figures/c4_local_repair_preservation_and_gaps.pdf"),
    ]
    write_table("f04_f05_input_hashes.csv", [
        {"figure": "F04" if directory == HYBRID and not name.startswith("llm_") else "F05",
         "source": source_name(directory / name), "sha256": sha256((directory / name).read_bytes()).hexdigest()}
        for directory, files in registered for name in files.split()
    ])
    print("F04 verified historical first 16, all 71 states")
    print("F05 verified 540 accepted selection orders, including 127 prefix cases")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only historical replay. Never imports acquisition, training or networking code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDIES = HERE.parent
RETRY = STUDIES / "volume_response_retry_audit_20261117"
OLD = STUDIES / "verifiable_scientific_agent_20261030"
SCIENCE = STUDIES / "comprehensive_history_diagnostics_20261112" / "tables"
sys.path.insert(0, str(RETRY))
from retry_validators import catalog_identity, error_category, load_validator  # noqa: E402

HISTORICAL = "historical_free_mixed_model"
HYBRID = "cw16_llm16_scientist_v2"
CW_FREE = "new_free_cw_sol61_xhigh"
NOCW = "nocw"
PROTECTED = ("input.json", "catalog.json", "contract.json", "selection.json")
TURN = re.compile(r"turn_(\d+)\.json$")
ATTEMPT = re.compile(r"attempt_\d+_(\d+)\.json$")
HEX_ID = re.compile(r"[0-9a-f]{20}$")
CASES = {(NOCW, 2357, 10), (NOCW, 157, 9), (CW_FREE, 887, 9), (CW_FREE, 887, 10),
         (CW_FREE, 887, 11), (CW_FREE, 157, 15), (CW_FREE, 12203, 16)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def write_csv(out, file, rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"no output rows for {file}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / file).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key_of(row):
    return row["method"], int(row["seed"]), int(row["round"])


def scope():
    prior_retry = list(csv_rows(RETRY / "retry_case_batches.csv"))
    old_hybrid = [x for x in csv_rows(OLD / "reliability_metrics.csv") if x["method"] == HYBRID]
    return {key_of(x): ROOT / (x.get("source") or x["batch_freeze_path"]) for x in prior_retry + old_hybrid}


def safe_plan(raw):
    if not isinstance(raw, str):
        return None, "no_complete_text"
    try:
        return json.loads(raw), ""
    except ValueError:
        return None, "format_json"


def choice_objects(plan):
    choices = plan.get("choices") if isinstance(plan, dict) else None
    if not isinstance(choices, list):
        return []
    return [(i, item) for i, item in enumerate(choices, 1)
            if isinstance(item, dict) and isinstance(item.get("id"), str)]


def hypotheses(plan):
    values = plan.get("hypotheses") if isinstance(plan, dict) else None
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def evidence(items, field):
    return {(h.get("id", ""), k) for h in items for k in h.get(field) or []
            if isinstance(h.get("id"), str) and isinstance(k, str)}


def sizes(plan):
    return [len(hypotheses(plan)), *(len(evidence(hypotheses(plan), f)) for f in
             ("supporting_observed_ids", "contradicting_observed_ids"))]


def transport_status(value):
    if value.get("success") is True:
        return "recovered_local_completion" if value.get("recovered_local_completion") else "complete_terminal"
    error = str(value.get("error_type") or value.get("exception_type") or "")
    status = value.get("http_status")
    if not status:
        match = re.search(r"\b(?:401|403|408|429|500|502|503|504)\b", text(value.get("error_message") or value.get("error")))
        status = match.group(0) if match else None
    if status:
        return f"HTTP_{status}"
    upper = error.lower()
    if "timeout" in upper:
        return "timeout"
    if "connect" in upper:
        return "connection"
    description = text(value.get("error_message") or value.get("error")).lower()
    if "sse" in upper or "frame" in upper or "sse" in description or "frame" in description:
        return "SSE_frame"
    if ("incomplete" in upper or "empty" in upper or "protocol" in upper
            or "incomplete" in description or "empty response" in description):
        return "incomplete_or_protocol"
    if "served" in upper or "mismatch" in upper:
        return "model_or_binding_mismatch"
    if "interrupt" in upper or "unknown" in upper:
        return "interrupted_unknown_provider_outcome"
    return error if error.isidentifier() else "unclassified_failure"


def archived_ledgers(by_directory):
    cw = ROOT / "reports/free_cw_final_20261112"
    nc = ROOT / "reports/nocw_retry_evidence_20261117"
    sources = {
        **{cw / n: d["sha256"] for n, d in read(cw / "publication_manifest.json")["lossless_archives"].items()
           if n.startswith("receipts/")},
        **{nc / n: d["sha256"] for n, d in read(nc / "manifest.json")["lossless_archives"].items()},
    }
    entries, counters, terminal, receipt_requests = defaultdict(dict), defaultdict(Counter), set(), {}
    archive_results = {}
    for path, expected in sorted(sources.items()):
        digest = sha_file(path)
        if expected != digest:
            raise ValueError(f"archive hash mismatch: {path}")
        files = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                folder, _, file = row["path"].rpartition("/")
                payload = row["raw_utf8"]
                if sha(payload.encode("utf-8")) != row["sha256"]:
                    raise ValueError(f"archived record hash mismatch: {row['path']}")
                files += 1
                if folder not in by_directory:
                    continue
                if file in entries[folder]:
                    raise ValueError(f"duplicate archived record: {row['path']}")
                entries[folder][file] = row["sha256"]
                if ATTEMPT.fullmatch(file):
                    value = json.loads(payload)
                    counters[folder][transport_status(value)] += 1
                    terminal.add((folder, file.removesuffix(".json")))
                    if value.get("request_sha256"):
                        receipt_requests.setdefault(folder, Counter())[(int(file.split("_")[1]), value["request_sha256"])] += 1
        archive_results[str(path.relative_to(ROOT))] = {"sha256": digest, "record_hashes_verified": files}
    for folder, files in entries.items():
        for file in files:
            if file.startswith("attempt_") and file.endswith(".started.json"):
                if (folder, file.removesuffix(".started.json")) not in terminal:
                    counters[folder]["started_without_terminal"] += 1
    return entries, counters, receipt_requests, archive_results


def available(directory, names, archives, checks, gaps):
    for name, expected in sorted(names.items()):
        path = directory / name
        # Verify source bytes for scientific artifacts and role-qualified replies
        # and use the verified lossless archive for the high-volume receipt ledger.
        if name in archives and name.startswith(("attempt_", "stream_")):
            digest = archives[name]
            checks["freeze_archive"] += 1
        elif path.is_file():
            digest = sha_file(path)
            checks["freeze_local"] += 1
        elif name in archives:
            digest = archives[name]
            checks["freeze_archive"] += 1
        else:
            gaps.append({"kind": "frozen_artifact_unavailable", "path": str(path.relative_to(ROOT))})
            continue
        if digest != expected:
            gaps.append({"kind": "frozen_artifact_hash_mismatch", "path": str(path.relative_to(ROOT)),
                         "expected": expected, "actual": digest})


def early_inventory(selected, gaps):
    targets = (STUDIES / "qgeognn_v2_row_llm_scientist_v2",
               STUDIES / "qgeognn_v2_row_llm_nocw_ablation")
    known = {(path.parts[-4], path.parts[-3], path.parts[-2], read(path)["artifact_hashes"]["selection.json"])
             for path in selected.values()}
    originals = set(selected.values())
    for target in targets:
        for path in sorted(target.rglob("batch_freeze.json")):
            if path in originals:
                continue
            f = read(path)
            identity = (*path.parts[-4:-1], f.get("artifact_hashes", {}).get("selection.json"))
            status = "duplicate_or_imported" if identity in known else "separate_protocol_or_unbound_scope"
            if f.get("dry_run") or "dry_run" in path.parts:
                status = "mock"
            elif path.parts[-3] not in {"llm32_nocw", "cw16_llm16_scientist_v2", "free_llm32_scientist_v2"}:
                continue
            elif not (path.parent / "label_access_receipt.json").is_file() and status != "duplicate_or_imported":
                status = "frozen_selection_without_reveal_receipt"
            gaps.append({"kind": status, "path": str(path.relative_to(ROOT)), "sha256": sha_file(path)})
        for path in sorted(target.rglob("input.json")):
            if "selections" in path.parts and "round_" in path.parent.name and not (path.parent / "batch_freeze.json").exists():
                gaps.append({"kind": "unfrozen_or_incomplete_inventory_only", "path": str(path.relative_to(ROOT))})


def screened_hybrid():
    """Audit six actual early conversation decisions as a separate protocol."""
    study = STUDIES / "qgeognn_v2_row_llm16_screen"
    rows = []
    for path in sorted(study.glob("selections/seed_*/cw16_llm16/round_*/batch_freeze.json")):
        folder = path.parent
        freeze = read(path)
        for name, expected in freeze["artifact_hashes"].items():
            if sha_file(folder / name) != expected:
                raise ValueError(f"screened Hybrid binding mismatch: {folder / name}")
        packet = read(folder / "packet.json")
        response = read(folder / "response.json")
        verified = read(folder / "validated_response.json")
        eligible_ids = {item["id"] for item in packet["candidates"]}
        pending_ids = [item["id"] for item in packet["pending_CW16"]]
        raw_choices = [item["id"] for _, item in choice_objects(response)]
        final = [item["id"] for _, item in choice_objects(verified)]
        if (len(pending_ids) != 16 or len(eligible_ids) != 128 or len(final) != 16 or len(set(final)) != 16
                or pending_ids + final != freeze["batch_ids"] or set(final) - eligible_ids
                or response["packet_sha256"] != freeze["artifact_hashes"]["packet.json"]):
            raise ValueError(f"screened Hybrid role or request binding mismatch: {folder}")
        rows.append({"source": str(path.relative_to(ROOT)), "sha256": sha_file(path),
                     "artifacts": freeze["artifact_hashes"], "response_exact_choice_order": raw_choices == final,
                     "first_response_legal": not verified["response_failed"], "fallback_count": verified["fallback_count"],
                     "eligible_proposals": len(eligible_ids), "pending_rows": len(pending_ids),
                     "provider_receipt_available": False})
    return rows


def topology(packet, name):
    table = packet[name]
    cols = table["columns"]
    return [dict(zip(cols, row)) for row in table["rows"]]


def text(value):
    return value if isinstance(value, str) else ""


def memory_counts(packet, prior_state, final, base):
    mem = packet.get("memory", {})
    last = prior_state.get("feedback", {})
    previous_h = mem.get("previous_hypotheses", [])
    recent = mem.get("recent_outcomes", [])
    tops = mem.get("high_error_measured_examples", [])
    top_ids = {item.get("candidate_id") for item in tops}
    seen_rounds = prior_state.get("seen_rounds", {})
    ages = [base["round"] - seen_rounds[i] for i in top_ids if i in seen_rounds]
    final_h = hypotheses(final)
    same_id = {h.get("id"): h for h in previous_h}
    current_content = {text(h.get("content")) for h in previous_h}
    support = evidence(final_h, "supporting_observed_ids")
    oppose = evidence(final_h, "contradicting_observed_ids")
    used_ids = {reference for _, reference in support | oppose}
    previous_ids = {row.get("candidate_id") for row in recent}
    return {
        **base, "prior_feedback_exact": recent == last.get("records") if last else None,
        "prior_hypotheses_exact": previous_h == last.get("hypotheses") if last else None,
        "recent_count": len(recent), "top8_count": len(tops), "top8_age_known": len(ages),
        "top8_max_age_rounds": max(ages, default=None), "top8_mean_age_rounds": mean(ages) if ages else None,
        "evidence_rows": len(used_ids), "previous_feedback_rows_cited": len(used_ids & previous_ids),
        "top8_rows_cited": len(used_ids & top_ids), "support_distinct_id_hypothesis_pairs": len(support),
        "counterevidence_distinct_id_hypothesis_pairs": len(oppose),
        "exact_claim_repeated": sum(text(h.get("content")) in current_content for h in final_h),
        "same_hypothesis_id_changed_text": sum(
            h.get("id") in same_id and h.get("content") != same_id[h["id"]].get("content") for h in final_h),
        "previous_rejected_exact_claim_reappears": sum(any(
            prev.get("status") == "rejected" and prev.get("content") == h.get("content") for prev in previous_h
        ) for h in final_h),
    }


def audit():
    batches = scope()
    split = STUDIES / "qgeognn_v2_row_sequential_b32/splits/row_seed_157.csv"
    canonical_ids = {x["sample_id"] for x in csv_rows(split)}
    earlier_context = {(key_of(x), int(x["turn"])): x for x in csv_rows(RETRY / "retry_request_contexts.csv")}
    folder_keys = {str(path.parent.relative_to(ROOT)): key for key, path in batches.items()}
    archived, counters, transport_digests, archive_summary = archived_ledgers(folder_keys)
    validators = {m: load_validator(HISTORICAL if m == HYBRID else m) for m in (HISTORICAL, NOCW, CW_FREE, HYBRID)}
    gaps, covers, diag, repairs, memory, bindings = [], [], [], [], [], []
    checks = Counter()
    prior = {}
    for key, path in sorted(batches.items()):
        method, seed, rnd = key
        d = path.parent
        folder = str(d.relative_to(ROOT))
        base = {"method": method, "seed": seed, "round": rnd, "L_before": 333 + 32 * rnd, "source": folder}
        status = {"batch_freeze_sha256": sha_file(path) if path.is_file() else ""}
        if not path.is_file():
            gaps.append({"kind": "freeze_unavailable", "path": str(path.relative_to(ROOT))})
            continue
        freeze = read(path)
        if freeze.get("dry_run") or freeze.get("new_batch_labels_revealed") is not False:
            raise ValueError(f"expected a pre-reveal real freeze: {path}")
        names = freeze.get("artifact_hashes", {})
        start_gap = len(gaps)
        available(d, names, archived.get(folder, {}), checks, gaps)
        critical_gap = any(gap["path"].rsplit("/", 1)[-1] in PROTECTED for gap in gaps[start_gap:])
        # Mandatory local scientific files may not be published in a fresh clone.
        if critical_gap or not all((d / name).is_file() and name in names for name in PROTECTED):
            gaps.append({"kind": "core_scientific_files_unavailable_or_invalid", "path": folder})
            continue
        packet, catalog, contract, final = [read(d / n) for n in PROTECTED]
        for n in ("feedback.json", "label_access_receipt.json"):
            if not (d / n).is_file():
                gaps.append({"kind": "post_selection_artifact_unavailable", "path": f"{folder}/{n}"})
        retained_feedback = (d / "feedback.json").is_file() and (d / "label_access_receipt.json").is_file()
        feedback = read(d / "feedback.json") if retained_feedback else {}
        receipt = read(d / "label_access_receipt.json") if retained_feedback else {}
        # Only pre-existing historical feedback is read. The truth table, label store
        # and validation/test predictions are never opened.
        status["feedback_sha256"] = sha_file(d / "feedback.json") if retained_feedback else ""
        status["label_access_receipt_sha256"] = sha_file(d / "label_access_receipt.json") if retained_feedback else ""
        selected = set(freeze["batch_ids"])
        identity = catalog_identity(packet)
        candidates = list(identity.candidates)
        rank = {cid: i for i, cid in enumerate(candidates, 1)}
        observed, pending = set(identity.observed), set(identity.pending)
        k = packet["selection_count"]
        if set(freeze["pending_ids"]) != pending:
            raise ValueError(f"frozen pending roles mismatch: {folder}")
        packed = topology(packet, "candidates")
        for name, ids in (("candidates", set(candidates)), ("observed", observed), ("pending", pending)):
            if {r["id"] for r in catalog[name]} != ids:
                raise ValueError(f"catalog {name} differs from packet: {folder}")
        if (set(contract["L_t_ids"]) != observed or set(contract["U_t_ids"]) != set(candidates) | pending
                or packet["validation_test_records"] != 0 or len(selected) != 32
                or selected - (set(candidates) | pending)
                or packet["packet_sha256"] != stable({s: v for s, v in packet.items() if s != "packet_sha256"})):
            raise ValueError(f"scientific identity/packet mismatch: {folder}")
        if retained_feedback and (receipt["batch_freeze_sha256"] != status["batch_freeze_sha256"]
                                  or feedback["batch_freeze_sha256"] != status["batch_freeze_sha256"]
                                  or receipt["feedback_sha256"] != stable(feedback)
                                  or {row["candidate_id"] for row in feedback["records"]} != selected
                                  or receipt["audit"]["roles"] != "u0"):
            raise ValueError(f"post-freeze feedback binding mismatch: {folder}")
        old = prior.get((method, seed), {})
        final_choices = {c["id"]: c for _, c in choice_objects(final)}
        validate = validators[method][0]
        if (pending | set(final_choices)) != selected:
            raise ValueError(f"frozen selection/LLM plan mismatch: {folder}")
        try:
            validate(final, identity, packet)
            final_legal = True
        except (ValueError, TypeError, KeyError):
            final_legal = False
        turns, seen_errors, exact_turns = [], set(), 0
        starts_and_ends = counters.get(folder, Counter())
        failed_request_matches = 0
        for n, digest in sorted(names.items(), key=lambda entry: (
                int(TURN.fullmatch(entry[0]).group(1)) if TURN.fullmatch(entry[0]) else -1, entry[0])):
            match = TURN.fullmatch(n)
            if not match or not (d / n).is_file():
                continue
            if sha_file(d / n) != digest:
                continue
            turn = int(match.group(1))
            reply = read(d / n)
            plan, parse_failure = safe_plan(reply.get("answer"))
            if parse_failure == "no_complete_text":
                continue
            error = parse_failure
            full = False
            if not error:
                try:
                    full = validate(plan, identity, packet) == final
                except (ValueError, TypeError, KeyError) as exc:
                    error = error_category(exc)
            exact_turns += full
            original_request = d / f"request_{turn:02}.json"
            request_hash = reply.get("request_sha256", "")
            request_bytes = None
            request_mode = "raw_request_unavailable"
            config = {}
            if original_request.is_file():
                req = read(original_request)
                config = req["config"]
                request_bytes = original_request.stat().st_size
                request_mode = "raw_request"
                if request_hash != stable(req) or json.loads(req["messages"][1]["content"]) != packet:
                    raise ValueError(f"request or full input mismatch: {original_request}")
            elif (key, turn) in earlier_context:
                previous = earlier_context[key, turn]
                if previous["request_hash_matches_receipt"] == "True":
                    request_bytes = int(previous["messages_utf8_bytes"])
                    request_mode = "prior_exact_reconstruction_messages_only"
            failed_request_matches += sum(num for (t, dig), num in transport_digests.get(folder, {}).items()
                                          if t == turn and dig != request_hash)
            ids = choice_objects(plan)
            valid = {c["id"]: c for _, c in ids if c["id"] in rank}
            invalid = Counter()
            legal_seen, current_errors = set(), set()
            for pos, c in ids:
                cid = c["id"]
                flags = []
                if cid in legal_seen:
                    flags.append("duplicate")
                legal_seen.add(cid)
                if cid in old.get("acquired", ()):
                    flags.append("already_selected_previous_round")
                if cid in observed:
                    flags.append("already_observed")
                elif cid in pending:
                    flags.append("pending_reserved")
                elif cid not in rank:
                    flags.append("outside_pool_canonical_id" if cid in canonical_ids else "unknown_ID")
                for e in flags:
                    err_key = (cid, e)
                    invalid[e] += 1
                    current_errors.add(err_key)
                    diag.append({**base, "turn": turn, "position": pos, "id": cid, "error": e,
                                 "id_len": len(cid), "looks_historical_hex20": bool(HEX_ID.fullmatch(cid)),
                                 "repeats_previous_answer_error": err_key in seen_errors})
            seen_errors = current_errors
            hs = hypotheses(plan)
            known_h = {h["id"] for h in hs if isinstance(h.get("id"), str)}
            structure = plan.get("choices") if isinstance(plan, dict) else None
            issues = Counter()
            if not isinstance(structure, list) or len(structure) != k:
                issues["invalid_choice_count_or_missing_list"] += 1
            if isinstance(structure, list) and len(ids) != len(structure):
                issues["choice_object_without_candidate_id"] += len(structure) - len(ids)
            if isinstance(plan, dict) and plan.get("packet_sha256") != packet["packet_sha256"]:
                issues["packet_binding"] += 1
            for field, classification in (("supporting_observed_ids", "bad_support_membership"),
                                          ("contradicting_observed_ids", "bad_counterevidence_membership")):
                issues[classification] += sum(r not in observed for _, r in evidence(hs, field))
            issues["unclosed_hypothesis_reference"] += sum(isinstance(c.get("hypothesis_id"), str)
                and bool(c["hypothesis_id"]) and c["hypothesis_id"] not in known_h for _, c in ids)
            diag.extend({**base, "turn": turn, "error": name, "occurrences": n}
                        for name, n in issues.items() if n)
            usage = reply.get("provenance", {}).get("usage", {})
            provenance = reply.get("provenance", {})
            turns.append({**base, "turn": turn, "error": error or "none", "exact_final": full,
                "valid": valid, "plan": plan, "choice_order": [c["id"] for _, c in ids],
                "invalid_count": sum(any(c["id"] == x[0] for x in current_errors) for _, c in ids),
                "request_bytes": request_bytes, "request_mode": request_mode, "request_sha256": request_hash,
                "served_model": provenance.get("served_model"),
                "requested_model": config.get("model") or earlier_context.get((key, turn), {}).get("requested_model"),
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "elapsed_seconds": reply.get("elapsed_seconds"),
                "response_source": n, "response_sha256": digest})
        if failed_request_matches:
            gaps.append({"kind": "transport_receipt_request_hash_mismatch", "path": folder,
                         "count": failed_request_matches})
        if not turns:
            gaps.append({"kind": "no_completed_turn_evidence", "path": folder})
        first = turns[0] if turns and turns[0]["turn"] == 0 else {}
        first_legal = first.get("error") == "none" if first else None
        budget = base["L_before"]
        bounds = next((phase for phase in ((333, 525), (525, 813), (813, 1005), (1005, 1485))
                       if phase[0] <= budget < phase[1]))
        final_ranks = [rank[x] for x in final_choices]
        if folder not in counters:
            for legacy_attempt in sorted(d.glob("retries/turn_*/attempt_*.json")):
                starts_and_ends[transport_status(read(legacy_attempt))] += 1
        known_transport = sum(v for cat, v in starts_and_ends.items() if cat != "started_without_terminal")
        covers.append({**base, **status, "phase": f"{bounds[0]}-{bounds[1]}",
            "first_plan_legal": first_legal, "final_plan_legal": final_legal,
            "exact_raw_final_matches": exact_turns, "label_receipt_verified": retained_feedback,
            "first_error": first.get("error"), "first_illegal_choice_occurrences": first.get("invalid_count"),
            "first_packet_bytes": (d / "input.json").stat().st_size, "first_request_bytes": first.get("request_bytes"),
            "first_request_size_basis": first.get("request_mode"), "first_input_tokens": first.get("input_tokens"),
            "complete_responses": len(turns), "logical_corrections_after_first": max(0, len(turns) - 1),
            "physical_terminal_records": known_transport, "uncertain_started": starts_and_ends["started_without_terminal"],
            "transport_basis": "legacy_failure_ledger_lower_bound" if method in {HISTORICAL, HYBRID} else "verified_attempt_archive",
            "accepted_ordered_first_k": list(final_choices) == candidates[:k],
            "accepted_any_first_k": sum(i <= k for i in final_ranks),
            "accepted_mean_normalized_position": mean(i / len(candidates) for i in final_ranks),
            "accepted_mean_one_based_position": mean(final_ranks),
            "adjacent_same_molecule_candidate_edges": sum(a["molecule_id"] == b["molecule_id"]
                for a, b in zip(packed, packed[1:])),
            "candidate_edges": len(candidates) - 1,
            "complete_output_tokens": sum(t["output_tokens"] or 0 for t in turns),
            "complete_response_count_with_token_usage": sum(t["output_tokens"] is not None for t in turns),
            "transport_counts": dict(starts_and_ends),
            "response_summaries": [{k: v for k, v in t.items() if k not in
                {*base, "valid", "plan", "choice_order"}} for t in turns]})
        latest = first.get("valid", {})
        if first and len(turns) > 1:
            kept = latest.keys() & final_choices.keys()
            hi, si, ci = sizes(first["plan"])
            hf, sf, cf = sizes(final)
            repairs.append({**base, "responses": len(turns), "initial_error": first["error"],
                "initial_option_count": len(latest), "legal_ids_retained": len(kept),
                "all_legal_replaced": bool(latest) and not kept, "same_legal_reasons": sum(
                    latest[x].get("reason") == final_choices[x].get("reason") for x in kept),
                "strategy_changed": (first["plan"] or {}).get("batch_strategy") != final.get("batch_strategy")
                    if isinstance(first["plan"], dict) else None,
                "initial_hypotheses": hi, "final_hypotheses": hf, "initial_support_edges": si,
                "final_support_edges": sf, "initial_counterevidence_edges": ci, "final_counterevidence_edges": cf,
                "support_edges_retained": len(evidence(hypotheses(first["plan"]), "supporting_observed_ids") &
                                              evidence(hypotheses(final), "supporting_observed_ids")),
                "counterevidence_edges_retained": len(evidence(hypotheses(first["plan"]), "contradicting_observed_ids") &
                                                      evidence(hypotheses(final), "contradicting_observed_ids"))})
        memory.append(memory_counts(packet, old, final, base))
        if key in CASES:
            bindings.append({**base, "evidence_grade": "direct_complete_responses",
                             "case": "predeclared_round", "detail": "see repair_drift_summary.csv and narrative",
                             "turns": len(turns), "reported_first_error": first.get("error")})
        seen_rounds = old.get("seen_rounds", {})
        seen_rounds.update({item["candidate_id"]: rnd for item in feedback.get("records", [])})
        prior[method, seed] = {"feedback": feedback, "seen_rounds": seen_rounds,
                               "acquired": old.get("acquired", set()) | selected}
    early_inventory(batches, gaps)
    return batches, covers, diag, repairs, memory, bindings, gaps, checks, archive_summary


def rates(covers):
    groups = defaultdict(list)
    for batch in covers:
        groups[batch["method"], batch["seed"], batch["phase"]].append(batch)
    for (method, seed, phase), rows in sorted(groups.items()):
        seen = [row for row in rows if row["first_plan_legal"] is not None]
        total = lambda field: sum(row[field] for row in rows)
        nfirst = sum(row["first_plan_legal"] for row in seen)
        first_legal = [r["first_input_tokens"] for r in seen if r["first_plan_legal"] and r["first_input_tokens"]]
        first_bad = [r["first_input_tokens"] for r in seen if not r["first_plan_legal"] and r["first_input_tokens"]]
        yield {"method": method, "seed": seed, "phase": phase, "frozen_batches": len(rows),
               "first_response_denominator": len(seen), "first_legal": nfirst,
               "first_legality": nfirst / len(seen) if seen else None,
               "final_legal": total("final_plan_legal"), "final_frozen_denominator": len(rows),
               "exact_raw_final_plans": sum(r["exact_raw_final_matches"] == 1 for r in rows),
               "complete_responses": total("complete_responses"),
               "logical_corrections_after_first": total("logical_corrections_after_first"),
               "physical_terminal_records": total("physical_terminal_records"),
               "uncertain_started_records": total("uncertain_started"),
               "transport_basis": rows[0]["transport_basis"],
               "mean_first_packet_bytes": mean(row["first_packet_bytes"] for row in rows),
               "first_tokens_median_given_legal": median(first_legal) if first_legal else None,
               "first_tokens_median_given_invalid": median(first_bad) if first_bad else None,
               "complete_output_tokens_subtotal": total("complete_output_tokens"),
               "complete_response_count_with_token_usage": total("complete_response_count_with_token_usage"),
               "accepted_ordered_first_k": total("accepted_ordered_first_k"),
               "accepted_in_first_k_occurrences": total("accepted_any_first_k"),
               "mean_accepted_normalized_display_position": mean(r["accepted_mean_normalized_position"] for r in rows),
               "candidate_adjacent_same_structure_fraction": total("adjacent_same_molecule_candidate_edges")
               / total("candidate_edges")}


def transport(covers):
    groups = defaultdict(Counter)
    for batch in covers:
        key = (batch["method"], batch["seed"], batch["phase"], batch["transport_basis"])
        groups[key].update(batch["transport_counts"])
    for (method, seed, phase, basis), counts in sorted(groups.items()):
        for label, count in sorted(counts.items()):
            yield {"method": method, "seed": seed, "phase": phase, "basis": basis,
                   "status": label, "record_count": count,
                   "paid_requests": None, "billing_evidence": "not_in_local_receipts"}


def review_binding_cases(direct):
    yield from direct
    for row in csv_rows(SCIENCE / "reasoning_flag_adjudications.csv"):
        yield {"method": row["method"], "seed": row["seed"], "round": row["round"],
               "L_before": row["labeled_size_before"], "source": row["selection_source"],
               "evidence_grade": "prior_human_review_of_exact_frozen_selection",
               "case": row["flag"], "detail": row["manual_adjudication"],
               "row_id": row["identifier"], "claimed_molecule": row["claimed_first_molecule"],
               "actual_molecule": row["actual_molecule"]}


def authority_sources():
    paths = []
    for study in (STUDIES / "nocw_vs_cw_lcmd_five_to1485_20261112",
                  STUDIES / "free_cw_sol61_xhigh_five_to1485_20261101",
                  STUDIES / "qgeognn_v2_row_llm_nocw_ablation"):
        for pattern in ("*authorization*.json", "*/authorization.json", "protocol.lock.json",
                        "source_snapshot.json", "revisions/*/*amendment*.json", "*/prefix_import.json"):
            paths.extend(study.glob(pattern))
    for name in (HISTORICAL, NOCW, CW_FREE):
        paths.extend(load_validator(name)[1])
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    out = parser.parse_args().output.resolve()
    if out != HERE and out.is_relative_to(ROOT):
        parser.error("choose the study directory or an output outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    batches, covers, diag, repairs, memory, semantic, gaps, checks, archives = audit()
    sources = (OLD / "reliability_metrics.csv", RETRY / "retry_case_batches.csv",
               RETRY / "retry_request_contexts.csv", SCIENCE / "reasoning_flag_adjudications.csv",
               STUDIES / "qgeognn_v2_row_sequential_b32/splits/row_seed_157.csv")
    write_csv(out, "reliability_by_method_seed_stage.csv", rates(covers))
    write_csv(out, "identity_error_taxonomy.csv", diag)
    write_csv(out, "semantic_binding_cases.csv", review_binding_cases(semantic))
    write_csv(out, "repair_drift_summary.csv", repairs)
    write_csv(out, "memory_diagnostics.csv", memory)
    write_csv(out, "transport_failure_summary.csv", transport(covers))
    manifest = {
        "scope": "full-catalog Free32, CW-visible Free32, NoCW32, and CW16+LLM16 using the authoritative revision owners",
        "cohort": dict(Counter(key[0] for key in batches)), "audited_batches": len(covers),
        "eligibility_note": "Frozen observed outcomes only. Legacy SDK ledgers are failure-only; payment is unobserved.",
        "input_sources": {str(p.relative_to(ROOT)): sha_file(p) for p in (*sources, *authority_sources())},
        "archive_sources": archives, "freeze_checks": dict(checks),
        "screened_hybrid_external_conversation": screened_hybrid(),
        "inventory_and_gaps": gaps, "batch_evidence": covers,
        "new_provider_requests": 0, "new_label_reveals": 0, "new_training_fits": 0,
    }
    # JSON separators avoid another long repeated CSV of receipt IDs, paths and hashes.
    (out / "historical_attempt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print("Audited", len(covers), "batches.", "Archive records:", sum(x["record_hashes_verified"] for x in archives.values()))
    print("Inventory", dict(Counter(x["kind"] for x in gaps)))


if __name__ == "__main__":
    main()

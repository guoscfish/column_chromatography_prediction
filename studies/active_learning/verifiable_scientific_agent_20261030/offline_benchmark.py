"""Replay frozen complete answers with the unchanged validator and diagnostics.

Usage (from the repository root):
  python3 studies/active_learning/verifiable_scientific_agent_20261030/offline_benchmark.py

Only the supplied manifest and its frozen response/packet/contract files are opened.
Candidate/observed table values other than ID columns are not used for decisions.
The script has no model client, optimizer, label-store or dataset imports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from verifiable_planner import (FrozenContext, Unanalyzable, baseline_validate,
                               diagnose, format_only_repair, frozen_validator_bundle,
                               parse_complete_json)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_path(value, manifest_path):
    path = Path(value)
    if path.is_absolute():
        return path
    if (ROOT / path).is_file():
        return ROOT / path
    return manifest_path.parent / path


def context_for(entry, manifest_path):
    packet = read_json(resolve_path(entry["packet_path"], manifest_path))
    contract = read_json(resolve_path(entry["contract_path"], manifest_path))
    variant = entry.get("variant", contract.get("variant", "direct"))
    validator_kind = {"direct": "direct", "nocw": "nocw_five", "nocw_five": "nocw_five",
                      "nocw_paired": "nocw_paired"}.get(entry.get("validator_kind", "direct"))
    if validator_kind is None:
        raise ValueError("unknown manifest validator kind")
    # Free/Hybrid use validate_direct, regardless of any incidental protocol fields.
    if entry.get("validator_kind") == "direct":
        variant = "direct"
    if variant not in ("direct", "no_cw", "cw_aware"):
        raise ValueError(f"manifest must specify exact validator variant: {variant}")
    return FrozenContext.from_packet(packet, labeled_ids=contract["L_t_ids"],
                                     unlabeled_ids=contract["U_t_ids"], variant=variant,
                                     validator_kind=validator_kind)


def answer_for(entry, manifest_path):
    path = resolve_path(entry["response_path"], manifest_path)
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if entry.get("response_sha256") != sha256:
        raise ValueError("response file differs from the audit manifest SHA256")
    record = json.loads(data)
    if not isinstance(record, dict) or not isinstance(record.get("answer"), str):
        raise ValueError("complete turn wrapper must contain an answer string")
    if record.get("success") is False or entry.get("completion") not in ("frozen_artifact", "frozen_complete"):
        raise ValueError("not a frozen complete response")
    return record["answer"]


def verify_entry_inputs(entry, manifest_path, snapshot, verified):
    for key in ("response_path", "packet_path", "contract_path", "batch_freeze_path"):
        path = resolve_path(entry[key], manifest_path)
        name = str(path.relative_to(ROOT))
        expected = snapshot["files"][name]["sha256"]
        if name not in verified:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"frozen input SHA256 mismatch: {name}")
            verified[name] = actual
        if key == "batch_freeze_path" and verified[name] != entry["frozen_sha256"]:
            raise ValueError("batch freeze differs from the response manifest")


def benchmark_entry(entry, manifest_path, bundle):
    context = context_for(entry, manifest_path)
    answer = answer_for(entry, manifest_path)
    row = {key: entry.get(key) for key in ("method", "seed", "round", "attempt", "response_path")}
    row.update({"variant": context.variant, "validator_kind": context.validator_kind,
                "historical_accepted": entry.get("accepted"),
                "audited_original_validator_legal": entry.get("original_validator_legal"),
                "response_sha256": entry["response_sha256"], "context_sha256": context.binding_sha256,
                "candidate_count": len(context.candidate_ids), "labeled_count": len(context.labeled_ids),
                "observed_count": len(context.observed_ids), "required_choices": context.selection_count,
                "complete_frozen_response": True})
    start = time.perf_counter()
    legacy_plan = None
    try:
        # Exactly the production parser, including its legacy duplicate-key behavior.
        legacy_plan = json.loads(answer)
        a_legal, a_error = baseline_validate(legacy_plan, context, bundle)
    except (ValueError, TypeError) as error:
        a_legal, a_error = False, f"{type(error).__name__}: {error}"
    a_ms = (time.perf_counter() - start) * 1000
    reliable_plan = None
    parse_error = ""
    try:
        reliable_plan = parse_complete_json(answer)
        parse_reliable = True
    except Unanalyzable as error:
        parse_reliable, parse_error = False, str(error)
    start = time.perf_counter()
    if parse_reliable:
        diagnosis = diagnose(reliable_plan, context, bundle)
        # Both branches use authoritative original acceptance, never a stricter schema.
        b_legal = diagnosis.authoritative_legal
        findings = diagnosis.findings
        structurally_patchable = not diagnosis.full_replanning_required
        valid_candidate_count = len(diagnosis.legal_candidate_indices)
        valid_choice_count = len(diagnosis.legal_choice_indices)
        missing = diagnosis.missing_count
        candidate_structure_reliable = diagnosis.analyzable
    else:
        # Ambiguous legacy JSON must not be promoted to a determined scientific batch.
        b_legal = a_legal
        findings = [{"code": "unreliable_complete_parse", "path": "", "message": parse_error,
                     "mutable": False}]
        structurally_patchable = False
        valid_candidate_count = valid_choice_count = missing = None
        candidate_structure_reliable = False
    b_ms = (time.perf_counter() - start) * 1000
    if a_legal != b_legal:
        raise AssertionError("diagnostic B changed the authoritative A acceptance")
    start = time.perf_counter()
    c_plan, c_action, c_parse_error = None, "unrepaired", ""
    try:
        repaired, c_action = format_only_repair(answer)
        c_plan = parse_complete_json(repaired)
        c_legal, c_error = baseline_validate(c_plan, context, bundle)
    except Unanalyzable as error:
        c_legal, c_error, c_parse_error = False, "no safe complete interpretation", str(error)
    c_ms = (time.perf_counter() - start) * 1000
    # Format repair is semantically transparent; compare parsed whole objects when
    # the raw response is strict JSON. For a fence, its complete interior is source.
    unchanged = (c_plan == reliable_plan) if parse_reliable else (
        True if c_action == "unwrap_single_json_fence" else None)
    row.update({"A_legal": a_legal, "B_legal": b_legal, "C_legal": c_legal,
                "A_error": a_error, "C_error": c_error,
                "raw_json_reliably_parseable": parse_reliable,
                "candidate_choice_structure_reliable": candidate_structure_reliable,
                "raw_parse_error": parse_error, "C_parse_error": c_parse_error,
                "diagnostic_codes": "|".join(sorted({x["code"] for x in findings})),
                "diagnostic_finding_count": len(findings),
                "legal_candidate_occurrences": valid_candidate_count,
                "legal_candidate_subset_scope": ("whole_choices_structure" if candidate_structure_reliable else
                    "individually_identifiable_fields_only" if parse_reliable else "unavailable"),
                "fully_legal_choice_occurrences": valid_choice_count,
                "missing_count": missing,
                "structural_localization_possible": structurally_patchable,
                "scientific_intent_preservation_verified": False,
                "C_format_operation": c_action, "C_entire_parsed_plan_unchanged": unchanged,
                "requires_new_planner_answer_after_C": not c_legal,
                "A_ms": a_ms, "B_ms_including_original_validator": b_ms,
                "B_additional_ms_estimate": b_ms-a_ms,
                "C_ms_including_original_validator": c_ms})
    detail = {"response_path": entry["response_path"], "findings": findings,
              "raw_batch_analysis": ("reliable_complete_JSON_choices" if candidate_structure_reliable else
                  "complete_JSON_only_partial_or_unavailable_choice_analysis" if parse_reliable else
                  "unavailable_ambiguous_or_unparseable"),
              "local_patch_assessment": ("full_replanning_required_for_safe_protocol" if not parse_reliable else
                  "original_already_legal" if a_legal else
                  "new_scientific_answer_and_explicit_intent_review_needed" if structurally_patchable else
                  "full_replanning_required"),
              "no_actual_patch_or_model_call_performed": True}
    return row, detail


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    output = []
    for method, subset in sorted(groups.items()):
        n = len(subset)
        a, b, c = (sum(bool(x[f"{arm}_legal"]) for x in subset) for arm in ("A", "B", "C"))
        output.append({"method": method, "complete_responses": n,
                       "A_legal": a, "B_legal": b, "C_legal": c,
                       "A_legal_rate": a/n, "B_legal_rate": b/n, "C_legal_rate": c/n,
                       "A_B_agreement": sum(x["A_legal"] == x["B_legal"] for x in subset)/n,
                       "A_original_audit_agreement": sum(x["A_legal"] == x["audited_original_validator_legal"] for x in subset)/n,
                       "safe_format_rescues": sum(not x["A_legal"] and x["C_legal"] for x in subset),
                       "legacy_accepted_but_ambiguous_strict_parse": sum(x["A_legal"] and not x["raw_json_reliably_parseable"] for x in subset),
                       "unreliable_raw_parse": sum(not x["raw_json_reliably_parseable"] for x in subset),
                       "need_new_planner_answer_after_C": sum(x["requires_new_planner_answer_after_C"] for x in subset),
                       "failed_but_structurally_localizable": sum(not x["A_legal"] and x["structural_localization_possible"] for x in subset),
                       "mean_A_ms": statistics.mean(x["A_ms"] for x in subset),
                       "mean_B_ms_including_original_validator": statistics.mean(x["B_ms_including_original_validator"] for x in subset),
                       "median_B_additional_ms_estimate": statistics.median(x["B_additional_ms_estimate"] for x in subset),
                       "mean_C_ms_including_original_validator": statistics.mean(x["C_ms_including_original_validator"] for x in subset)})
    return output


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "response_manifest.jsonl")
    parser.add_argument("--snapshot", type=Path, default=HERE / "history_snapshot_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--limit", type=int, default=None, help="debug only; omitted for the final benchmark")
    args = parser.parse_args(argv)
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        entries = entries[:args.limit]
    bundle = frozen_validator_bundle(ROOT)
    snapshot = read_json(args.snapshot)
    verified = {}
    rows, details, excluded = [], [], []
    for number, entry in enumerate(entries):
        try:
            verify_entry_inputs(entry, args.manifest, snapshot, verified)
            row, detail = benchmark_entry(entry, args.manifest, bundle)
            rows.append(row)
            details.append(detail)
        except (ValueError, KeyError, TypeError, OSError) as error:
            excluded.append({"response_path": entry.get("response_path"),
                             "reason": f"{type(error).__name__}: {error}"})
        if (number+1) % 50 == 0:
            print(f"Replayed {number+1}/{len(entries)} frozen complete responses", flush=True)
    summary = summarize(rows)
    report = {"schema": "offline_validator_benchmark_v1", "manifest": str(args.manifest),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "snapshot_manifest_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
              "snapshot_captured_at": snapshot["captured_at"],
              "verified_frozen_input_files": len(verified),
              "validator_source_sha256": bundle["source_sha256"],
              "manifest_entries": len(entries), "analyzed_complete_responses": len(rows),
              "excluded_entries": excluded, "debug_limit": args.limit,
              "method_summary": summary,
              "diagnostic_code_counts": dict(Counter(code for row in rows for code in row["diagnostic_codes"].split("|") if code)),
              "external_model_calls": 0, "new_training_runs": 0, "new_label_reveals": 0,
              "interpretation": "Response-level replay only; B acceptance equals A by design. C changes only complete single JSON fences. "
              "Mock localized patches are separate tests, not observed rescues. No AL learning gain is measured.",
              "timing_note": "Single local process wall-clock; B includes original validation. Paired B-A estimates can be negative from timing noise. "
              "AST load, disk reads, manifest parsing and context construction excluded from arm timing.",
              "response_details": details}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "offline_benchmark_responses.csv", rows)
    write_csv(args.output_dir / "offline_benchmark_summary.csv", summary)
    (args.output_dir / "offline_benchmark_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"analyzed": len(rows), "excluded": len(excluded), "methods": summary}, ensure_ascii=False, indent=2))
    return 0 if not excluded else 1


if __name__ == "__main__":
    raise SystemExit(main())

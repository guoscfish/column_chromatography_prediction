"""Review already audited prose citations; never query a label table or infer intent.

The decisions below are explicit case annotations, not a text classifier.  Only the
four unknown-ID occurrences in the existing two-seed, first-15-round audit enter.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "studies/active_learning/hybrid_loading_cap_audit_20261029/reason_id_reference_audit.csv"
ANNOTATIONS = {
    ("hybrid_llm16", "157", "9", "96e2b82e29c9f4fe8548"): (
        "explicit_exclusion", "明确说该未知ID不作为类比依据；不能算伪造实测证据。"),
    ("free32", "157", "5", "966254b5be3c8ecfa2ee"): (
        "ambiguous_disclaimer", "句子同时出现measured与unavailable，随后改引其他观测；语义含混，不能确认为肯定实测引用。"),
    ("hybrid_llm16", "6101", "9", "0b247863fd6e8eaf44d8"): (
        "asserted_measured_anchor_not_in_catalog", "明确声称该ID测得8.34/45.06；旧审计将其定位为catalog中不存在，证据关系不可追溯。未查询其他标签来补救。"),
    ("hybrid_llm16", "6101", "12", "18f616a941bed3e03c3f"): (
        "asserted_anchor_not_in_catalog", "把该ID称为para-bromoanisole的50/1 anchor；旧审计中ID不在catalog，需复核，不能自动更正或确认其实测含义。"),
}


def main():
    raw = SOURCE.read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    unknown = [r for r in rows if r["scope"] == "unknown_ID"]
    found = {(r["source"], r["seed"], r["round"], r["reference_id"]) for r in unknown}
    if found != set(ANNOTATIONS):
        raise ValueError("Existing citation audit changed; review new cases explicitly")
    reviewed = []
    for row in unknown:
        key = (row["source"], row["seed"], row["round"], row["reference_id"])
        category, note = ANNOTATIONS[key]
        reviewed.append({**row, "review_category": category, "review_note": note,
                         "historical_validator_failure": "not_inferred",
                         "learning_value": "not_evaluated"})
    with (HERE / "scientific_issue_review.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
    summary = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "prior two-seed Free32/HybridLLM16 audit, rounds 0..14 only",
        "reference_occurrences": len(rows),
        "scope_counts": dict(Counter(r["scope"] for r in rows)),
        "unknown_reference_occurrences": len(unknown),
        "manual_review_counts": dict(Counter(r["review_category"] for r in reviewed)),
        "new_label_reads": 0,
        "warning": "Occurrence denominator is citations, not batches or experiments; no universal semantic failure rate is estimated.",
    }
    (HERE / "scientific_issue_review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

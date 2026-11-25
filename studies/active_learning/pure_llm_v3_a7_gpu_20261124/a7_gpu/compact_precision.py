"""Read-only precision and encoding audit for a captured A7 Responses request."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

from .compact_payload import (
    ROLES, WIDTHS, _raw, candidate_formats, compact_packet, dumps, expand_packet,
    grouped_table_text, read_grouped_table, round_view,
)


TOKENIZER = "o200k_base"
BPE_NAME = "fb374d419588a4632f3f557e76b4b70aebbca790"
BPE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def local_tokenizer(directory):
    import tiktoken

    if hashlib.sha256((directory / BPE_NAME).read_bytes()).hexdigest() != BPE_SHA256:
        raise ValueError("the public o200k_base cache is required and must match its SHA256")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(directory)
    return tiktoken.get_encoding(TOKENIZER)


def _tie_pairs(values):
    counts = np.unique(values, axis=0, return_counts=True)[1]
    return int(np.sum(counts * (counts - 1) // 2))


def compare_values(before, after):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape or before.size == 0:
        raise ValueError("precision audit requires nonempty paired values")
    errors = abs(after - before)
    shift = abs(rankdata(before) - rankdata(after))
    pair_count = before.size * (before.size - 1) // 2
    before_ties, after_ties = _tie_pairs(before), _tie_pairs(after)
    shared_ties = _tie_pairs(np.column_stack((before, after)))
    comparables = pair_count - before_ties - after_ties + shared_ties
    if comparables:
        tau = kendalltau(before, after).statistic
        difference = round(tau * np.sqrt((pair_count - before_ties) * (pair_count - after_ties)))
        inversions = (comparables - difference) // 2
    else:
        inversions = 0
    return {
        "n": before.size,
        "max_abs": float(errors.max()),
        "p50_abs": float(np.quantile(errors, .5)),
        "p95_abs": float(np.quantile(errors, .95)),
        "p99_abs": float(np.quantile(errors, .99)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_midrank_shift": float(shift.max()),
        "p95_abs_midrank_shift": float(np.quantile(shift, .95)),
        "changed_midranks": int(np.count_nonzero(shift)),
        "reversed_pairs": int(inversions),
        "tied_pair_fraction_before": before_ties / max(1, pair_count),
        "tied_pair_fraction_after": after_ties / max(1, pair_count),
        "new_tied_pairs": after_ties - shared_ties,
        "negative_before": int(np.count_nonzero(before < 0)),
        "negative_after": int(np.count_nonzero(after < 0)),
    }


def precision_metrics(packet, decimals, *, preserve_crossings=False, endpoint_scales=None, validation_reference=None):
    before = _raw(packet)
    rounded = round_view(packet, decimals, preserve_crossings=preserve_crossings)
    after = _raw(rounded)
    results = {}
    for role in ROLES:
        fields = [field for field in packet["table_columns"][role] if field.startswith(
            ("pred_", "premeasurement_prediction_", "premeasurement_error_")) or field in (
                *WIDTHS, "historical_abs_error_ml", "max_similarity_to_observed", "max_similarity_to_pending")]
        for field in fields:
            pairs = [(row[field], view[field]) for (_, row), (_, view) in zip(before[role], after[role], strict=True)
                     if type(row[field]) in (float, int)]
            if pairs:
                results[f"{role}.{field}"] = compare_values(*zip(*pairs))
    groups = packet["molecules"]
    for field in sorted({name for group in groups.values() for name in group["descriptors"]}):
        pairs = [(group["descriptors"][field], rounded["molecules"][key]["descriptors"][field])
                 for key, group in groups.items() if field in group["descriptors"]]
        results[f"descriptor.{field}"] = compare_values(*zip(*pairs))
    crossing = {}
    for target in ("V1", "V2"):
        cols = [f"pred_{target}_{q}_ml" for q in ("q10", "q50", "q90")]
        if set(cols) <= set(packet["table_columns"]["candidates"]):
            original_crossed, displayed_crossed = [], []
            for row_set, flags in ((before["candidates"], original_crossed),
                                   (after["candidates"], displayed_crossed)):
                for _, row in row_set:
                    low, center, high = (row[name] for name in cols)
                    flags.append(not low <= center <= high)
            crossing[target] = {"before": sum(original_crossed), "after": sum(displayed_crossed),
                                "changed": sum(a != b for a, b in zip(original_crossed, displayed_crossed, strict=True))}
    if {"pred_V1_q50_ml", "pred_V2_q50_ml"} <= set(packet["table_columns"]["candidates"]):
        signed = [[row["pred_V1_q50_ml"] > row["pred_V2_q50_ml"] for _, row in rows["candidates"]]
                  for rows in (before, after)]
        crossing["V2_q50_lt_V1_q50"] = {
            "before": sum(signed[0]), "after": sum(signed[1]),
            "changed": sum(a != b for a, b in zip(*signed, strict=True)),
        }
    exceptions = 0
    if preserve_crossings:
        normal = _raw(round_view(packet, decimals))
        exceptions = sum(row != guarded for (_, row), (_, guarded) in zip(
            normal["candidates"], after["candidates"], strict=True))
    bound = None
    if endpoint_scales is not None and validation_reference is not None:
        normalized = [results[f"candidates.pred_{target}_q50_ml"]["rmse"] / scale
                      for target, scale in zip(("V1", "V2"), endpoint_scales, strict=True)]
        perturbation = float(np.sqrt(np.mean(np.square(normalized))))
        bound = {"validation_combined_nrmse": validation_reference, "endpoint_scales_ml": endpoint_scales,
                 "q50_display_combined_normalized_rms": perturbation,
                 "q50_display_rms_over_validation_nrmse": perturbation / validation_reference,
                 "predictions_and_validation_are_distinct_populations": True}
    return {"fields": results, "crossed_quantiles": crossing, "crossing_guarded_rows": exceptions,
            "validation_scale_comparison": bound}


def audit_request(job, tokenizer, *, endpoint_scales=None, validation_reference=None):
    envelope = job.get("request", job)
    payload = envelope["payload"]
    messages = payload["input"]
    packet = json.loads(messages[1]["content"])
    original = dumps(packet, sort_keys=True)
    roundtrip = {}
    for keep in (False, True):
        packed = compact_packet(packet, keep_widths=keep)
        roundtrip[f"grouped_lossless_keep_widths_{keep}"] = dumps(expand_packet(packed), sort_keys=True) == original
        if keep:
            roundtrip["grouped_lossless_tsv"] = dumps(expand_packet(read_grouped_table(
                grouped_table_text(packed))), sort_keys=True) == original
    for label, decimals, guard in (("3", 3, False), ("2", 2, False), ("3_guarded", 3, True)):
        view = round_view(packet, decimals, preserve_crossings=guard)
        roundtrip[f"grouped_{label}_to_display_view"] = dumps(
            expand_packet(compact_packet(view)), sort_keys=True) == dumps(view, sort_keys=True)
    if not all(roundtrip.values()):
        raise ValueError("wire views failed round-trip comparison")
    formats = {}
    for name, variant in candidate_formats(messages).items():
        formats[name] = {
            "content_tokens_estimate": sum(len(tokenizer.encode_ordinary(message["content"])) for message in variant),
            "wire_bytes": len(dumps({**payload, "input": variant}).encode()),
        }
    reference = formats["existing_grouped_json"]
    for value in formats.values():
        value["token_reduction"] = 1 - value["content_tokens_estimate"] / reference["content_tokens_estimate"]
        value["wire_reduction"] = 1 - value["wire_bytes"] / reference["wire_bytes"]
    return {
        "seed": packet["seed"], "round": packet["round"], "request_id": envelope.get("request_id"),
        "estimate": "o200k_base content tokens; model tokenizer, chat framing and provider accounting unverified",
        "tokenizer_sha256": BPE_SHA256, "roundtrip": roundtrip, "formats": formats,
        "precision": {name: precision_metrics(packet, decimals, preserve_crossings=guard,
                                              endpoint_scales=endpoint_scales, validation_reference=validation_reference)
                      for name, decimals, guard in (("3", 3, False), ("2", 2, False), ("3_guarded", 3, True))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_json", type=Path, help="existing spool job JSON")
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    parser.add_argument("--endpoint-scales", type=float, nargs=2)
    parser.add_argument("--validation-reference", type=float)
    args = parser.parse_args()
    if (args.endpoint_scales is None) != (args.validation_reference is None):
        parser.error("pass endpoint scales and validation reference together")
    if args.endpoint_scales and (min(args.endpoint_scales) <= 0 or args.validation_reference <= 0):
        parser.error("scales and validation reference must be positive")
    data = audit_request(json.loads(args.request_json.read_text()), local_tokenizer(args.tokenizer_cache),
                         endpoint_scales=args.endpoint_scales, validation_reference=args.validation_reference)
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

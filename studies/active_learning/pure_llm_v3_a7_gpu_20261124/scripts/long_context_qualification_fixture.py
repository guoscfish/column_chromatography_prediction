"""Prepare a synthetic A7/M2 wire fixture locally. This script has no API client."""

import argparse
import json
import os
from pathlib import Path
from random import Random
import sys

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import (
    compact_packet, expand_packet, packet_messages, round_view,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.offline_context_audit import offline_tokenizer


def _reagent(index, rng):
    head = f"970{index:04d}{index % 100:02d}"
    check = sum((i + 1) * int(digit) for i, digit in enumerate(reversed(head))) % 10
    family = ("aliphatic", "aromatic", "ketone", "ester", "ether")[index % 5]
    name = f"Synthetic {family} reagent {index:03d} lot {rng.randrange(10000, 99999)}"
    percent = (99, 98.5, 98, 97, 95)[index % 5]
    return [f"{head[:-2]}-{head[-2:]}-{check}", name, f"{percent}%", percent / 100, "numeric"]


def _molecule(index, rng):
    # A deterministic branch pattern supplies distinct fictional catalog structures.
    chain = "".join("C(O)" if index & (1 << bit) else "C" for bit in range(8))
    smiles = ("CC(C)C", "COCC", "CCN(C)C", "OC(=O)C")[index % 4] + chain + "O"
    return {"smiles": smiles, "descriptors": {
        "HBA": rng.randrange(1, 9), "HBD": rng.randrange(0, 5),
        "LogP": round(rng.uniform(-0.8, 6.0), 5), "MW_g_mol": round(rng.uniform(108, 510), 3),
        "TPSA_A2": round(rng.uniform(12, 145), 2),
    }, "observed": [], "candidates": []}


def _quantiles(center, rng):
    return (max(0, center - rng.uniform(0.6, 2.4)), center, center + rng.uniform(0.6, 4.0))


def _row(kind, index, group, offset, order, rng, reagent_count, similarity, molecule_novelty):
    ratios = (3, 5, 8, 10, 12, 15, 20, 30, 50, 80)
    ratio = ratios[(group + offset) % len(ratios)]
    sample = rng.choice((30, 50, 100, 150, 200, 300, 400, 500))
    density = round(rng.uniform(0.68, 1.56), 3)
    solvent = rng.choice(("DMF", "DCM", "THF"))
    volume = rng.choice((20, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000))
    flow = rng.choice((10, 12, 14, 16, 18))
    reagent = (3 * group + offset % 3) % reagent_count
    molecule_signal = (group % 19) * 0.68 + ratio * 0.27
    v1, v2 = molecule_signal + rng.uniform(0.2, 2.5), molecule_signal + rng.uniform(2.5, 12.0)
    q1, q2 = _quantiles(v1, rng), _quantiles(v2, rng)
    observed = kind == "observed"
    values = {
        "id": f"{kind[0].upper()}-{index:04d}", "conditions": [f"{ratio}/1", str(density), str(sample), solvent,
                                                               str(volume), str(flow), 1.0 / (ratio + 1), reagent],
        "V1_ml": round(v1 + rng.uniform(-0.15, 0.4), 5), "V2_ml": round(v2 + rng.uniform(-0.4, 0.8), 5),
        "V1_quantile_width_ml": q1[2] - q1[0], "V2_quantile_width_ml": q2[2] - q2[0],
        "condition_novelty": 0 if observed else rng.randrange(0, 2), "loading_amount_mg": density * sample,
        "max_similarity_to_observed": 1.0 if observed else similarity,
        "molecule_novelty": 0 if observed else molecule_novelty,
        "record_kind": "initial_L333",
    }
    for target, quantiles in (("V1", q1), ("V2", q2)):
        values.update({f"pred_{target}_{level}_ml": value for level, value in zip(("q10", "q50", "q90"), quantiles)})
    return [values[field] for field in order]


def synthetic_packet(template, *, seed):
    """Use only the frozen table schema and group sizes from the captured request."""
    rng = Random(seed)
    assert (template["layout"], template["round"], template["selection_count"]) == ("B", 0, 32)
    shape = [(len(group.get("observed", [])), len(group.get("candidates", [])))
             for group in template["molecules"].values()]
    rng.shuffle(shape)
    reagent_count = len(template["reagents"])
    packet = {"seed": seed, "round": 0, "selection_count": 32, "layout": "B", "compression": "lossless",
              "candidate_count": sum(c for _, c in shape), "observed_count": sum(o for o, _ in shape),
              "condition_columns": list(template["condition_columns"]),
              "condition_constants": {"column_specs": "synthetic 4 g column"},
              "reagent_columns": list(template["reagent_columns"]),
              "reagents": [_reagent(index, rng) for index in range(reagent_count)],
              "table_columns": {key: list(value) for key, value in template["table_columns"].items()},
              "memory": {"mode": "M2", "latest_feedback": [], "previous_feedback_interpretation": "",
                         "hypotheses": [], "archived_hypotheses": []}, "molecules": {}}
    seen = {"observed": 0, "candidates": 0}
    for group_index, (observed, candidates) in enumerate(shape, 1):
        group = _molecule(group_index, rng)
        similarity = 1.0 if observed else rng.uniform(0.32, 0.92)
        for role, size in (("observed", observed), ("candidates", candidates)):
            for offset in range(size):
                seen[role] += 1
                group[role].append(_row(role, seen[role], group_index, offset, packet["table_columns"][role],
                                        rng, reagent_count, similarity, int(observed == 0)))
        packet["molecules"][f"M-{group_index:04d}"] = group
    assert (seen["observed"], seen["candidates"]) == (333, 2997)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path, help="archived v3 seed157 round0 job")
    parser.add_argument("--tokenizer-cache", required=True, type=Path, help="verified local o200k_base cache")
    parser.add_argument("--precision", type=int, choices=(0, 2, 3), default=3, help="0 keeps exact numeric values")
    parser.add_argument("--drop-derived-widths", action="store_true", help="reconstruct exact widths in the decoder")
    parser.add_argument("--synthetic-seed", type=int, default=7417)
    parser.add_argument("--output", type=Path, help="optional private JSON fixture; nothing is sent")
    args = parser.parse_args()

    archived = json.loads(args.job.read_text())["request"]
    baseline = archived["payload"]
    original = json.loads(baseline["input"][1]["content"])
    assert archived["payload_sha256"] == digest(baseline)
    assert archived["binding"]["packet_sha256"] == digest(original)
    assert (original["seed"], original["round"], original["candidate_count"], original["observed_count"]) == (157, 0, 2997, 333)

    synthetic = synthetic_packet(original, seed=args.synthetic_seed)
    encode = offline_tokenizer(args.tokenizer_cache).encode_ordinary
    summary = {}
    for name, packet in (("real", original), ("synthetic", synthetic)):
        guarded = args.precision == 3
        display = round_view(packet, args.precision, preserve_crossings=guarded) if args.precision else packet
        compact = compact_packet(display, keep_widths=not args.drop_derived_widths)
        assert expand_packet(compact) == display
        messages = baseline["input"]
        checks = {}
        if name == "synthetic":
            check_ids = ("C-0121", "C-1501", "C-2871")
            columns = display["table_columns"]["candidates"]
            id_index, q50_index = columns.index("id"), columns.index("pred_V2_q50_ml")
            checks = {row[id_index]: row[q50_index] for group in display["molecules"].values()
                      for row in group["candidates"] if row[id_index] in check_ids}
            assert set(checks) == set(check_ids)
            messages = [dict(message) for message in messages]
            messages[0]["content"] += (
                "\n\nThis is a fictional diagnostic catalog. All data rows are generated. "
                "As a delivery check only, use the first choice whose ID is outside "
                + ", ".join(check_ids) + ". Add typed candidate prediction_comparison references to "
                "all three and append their IDs and visible V2 q50 values to its reason. "
                "Do not change your choices or the JSON schema for this check."
            )
        wire = {**baseline, "input": packet_messages(messages, compact,
                                                     feature_precision=args.precision or None,
                                                     crossing_guard=guarded)}
        summary[name] = {"wire_payload_bytes": len(dumps(wire).encode()),
                         "content_tokens_o200k_base": sum(len(encode(message["content"])) for message in wire["input"]),
                         "candidates": packet["candidate_count"], "observed": packet["observed_count"],
                         "output_choices": packet["selection_count"]}
        if checks:
            summary[name]["expected_delivery_checks"] = checks
        if name == "synthetic" and args.output:
            os.umask(0o077)
            with args.output.open("x") as output:
                output.write(dumps(wire) + "\n")
    print(dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

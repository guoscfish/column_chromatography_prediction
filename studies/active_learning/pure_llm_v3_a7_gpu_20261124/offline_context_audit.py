"""Measure already-staged A7 payloads and simulate context growth without labels or API calls."""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.context_builder import (
    SIGNALS, TARGETS, decode_context, encode_context,
)
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MEASUREMENT_FIELDS
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps


VOCAB_NAME = "fb374d419588a4632f3f557e76b4b70aebbca790"
VOCAB_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
KINDS = (
    "system_prompt", "molecule_dictionary_and_descriptors", "reagent_dictionary", "experiment_conditions",
    "candidate_ids", "quantile_predictions", "quantile_width", "novelty_and_similarity", "observed_labels",
    "historical_predictions_and_errors", "scientific_memory", "json_structure", "other",
)
STRUCTURE = KINDS.index("json_structure")
OTHER = KINDS.index("other")


def offline_tokenizer(cache):
    """Refuse to let tiktoken fall back to an implicit network download."""
    asset = Path(cache).resolve() / VOCAB_NAME
    if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != VOCAB_SHA256:
        raise ValueError("supply the validated local o200k_base vocabulary cache")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(asset.parent)
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def read_payload(path):
    source = json.loads(Path(path).read_bytes())
    if "request" in source:
        request = source["request"]
    elif "payload" in source:
        request = source
    else:
        return source, None
    if request["payload_sha256"] != digest(request["payload"]):
        raise ValueError("saved request payload hash mismatch")
    return request["payload"], request["request_id"]


def field_kind(name, role):
    if name == "id" and role == "candidates":
        return KINDS.index("candidate_ids")
    if name == "conditions":
        return KINDS.index("experiment_conditions")
    if name.startswith("pred_") and any(q in name for q in ("_q10_", "_q50_", "_q90_")):
        return KINDS.index("quantile_predictions")
    if "quantile_width" in name:
        return KINDS.index("quantile_width")
    if "novelty" in name or "similarity" in name:
        return KINDS.index("novelty_and_similarity")
    if name in ("V1_ml", "V2_ml"):
        return KINDS.index("observed_labels")
    if name.startswith(("premeasurement_", "historical_")):
        return KINDS.index("historical_predictions_and_errors")
    return OTHER


class Attribution:
    def __init__(self, packet):
        self.packet = packet
        self.parts = []
        self.owners = bytearray()
        self.wire = Counter()
        self.numbers = defaultdict(Counter)
        self.empty = Counter()

    def add(self, line, owner=STRUCTURE):
        self.parts.append(line)
        self.owners.extend(bytes((owner,)) * len(line.encode()))
        # Escaping is additive across substrings of the outer input.content string.
        self.wire[owner] += len(dumps(line).encode()) - 2

    def emit(self, value, owner=OTHER, name="other", path=()):
        if value is None or value == "" or value == [] or value == {}:
            form = "null" if value is None else type(value).__name__
            self.empty[f"{name}:{form}"] += 1
        if isinstance(value, dict):
            self.add("{")
            for index, (key, child) in enumerate(value.items()):
                if index:
                    self.add(",")
                child_owner = owner
                if not path:
                    child_owner = {
                        "reagent_columns": KINDS.index("reagent_dictionary"),
                        "reagents": KINDS.index("reagent_dictionary"),
                        "condition_constants": KINDS.index("experiment_conditions"),
                        "condition_columns": KINDS.index("experiment_conditions"),
                        "memory": KINDS.index("scientific_memory"),
                    }.get(key, OTHER)
                if path == ("molecules",) or (
                    len(path) == 2 and path[0] == "molecules" and key in ("smiles", "descriptors")
                ):
                    child_owner = KINDS.index("molecule_dictionary_and_descriptors")
                if path[:1] == ("molecules",) and len(path) == 2 and key in ("observed", "candidates"):
                    child_owner = OTHER
                self.add(dumps(key), child_owner if path == ("molecules",) else STRUCTURE)
                self.add(":")
                self.emit(child, child_owner, key, path + (key,))
            self.add("}")
        elif isinstance(value, list):
            self.add("[")
            for index, child in enumerate(value):
                if index:
                    self.add(",")
                child_owner, child_name = owner, name
                if len(path) == 4 and path[0] == "molecules" and path[2] in ("observed", "candidates"):
                    child_name = self.packet["table_columns"][path[2]][index]
                    child_owner = field_kind(child_name, path[2])
                elif name == "conditions" and owner == KINDS.index("experiment_conditions"):
                    child_name = "conditions." + self.packet["condition_columns"][index]
                self.emit(child, child_owner, child_name, path + (index,))
            self.add("]")
        else:
            lexeme = dumps(value)
            if type(value) is float:
                self.numbers[name][len(lexeme)] += 1
            if isinstance(value, str):
                self.add('"')
                self.add(lexeme[1:-1], owner)
                self.add('"')
            else:
                self.add(lexeme, owner)


def _percentile(histogram, fraction):
    needed = max(1, int(sum(histogram.values()) * fraction + 0.999999999))
    for length, count in sorted(histogram.items()):
        needed -= count
        if needed <= 0:
            return length
    raise ValueError("empty numeric histogram")


def _length_statistics(histogram):
    count = sum(histogram.values())
    return {
        "count": count, "min": min(histogram), "max": max(histogram),
        "average": round(sum(length * count for length, count in histogram.items()) / count, 3),
        **{f"p{int(q * 100)}": _percentile(histogram, q) for q in (0.5, 0.9, 0.99)},
        "characters_histogram": dict(sorted(histogram.items())),
    }


def structure_statistics(packet):
    rows = decode_context(packet)
    all_rows = [row for table in rows.values() for row in table.values()]
    conditions = Counter(dumps(row["conditions"], sort_keys=True) for row in all_rows)
    condition_values = {col: len({dumps(row["conditions"][col]) for row in all_rows})
                        for col in all_rows[0]["conditions"]}
    repeated_features = {}
    for field in sorted(SIGNALS | {"V1_quantile_width_ml", "V2_quantile_width_ml"}):
        values = Counter(dumps(row[field]) for row in rows["candidates"].values() if field in row)
        if values:
            repeated_features[field] = {"unique": len(values), "most_common_count": values.most_common(1)[0][1]}
    molecules = packet["molecules"]
    return {
        "condition_rows": len(all_rows), "unique_exact_conditions": len(conditions),
        "repeated_condition_instances_beyond_first": len(all_rows) - len(conditions),
        "repeated_condition_groups": sum(value > 1 for value in conditions.values()),
        "condition_cardinality_by_field": condition_values,
        "empty_role_tables": {role: sum(not item[role] for item in molecules.values()) for role in rows},
        "repeated_candidate_features": repeated_features,
        "q50_aliases_already_recovered_by_existing_codec": True,
        "exactly_derivable_widths": {target: sum(
            dumps(row[f"{target}_quantile_width_ml"])
            == dumps(row[f"pred_{target}_q90_ml"] - row[f"pred_{target}_q10_ml"])
            for row in rows["candidates"].values()) for target in TARGETS},
        "exactly_derivable_loading_amounts": sum(dumps(row["loading_amount_mg"])
            == dumps(float(row["conditions"]["density"]) * float(row["conditions"]["sample_volume"]))
            for row in all_rows),
    }


def audit_payload(payload, tokenizer):
    messages = payload["input"]
    if len(messages) != 2 or [row["role"] for row in messages] != ["system", "user"]:
        raise ValueError("audit expects the two-message A7 request")
    system, user = [row["content"] for row in messages]
    packet = json.loads(user)
    if packet["layout"] != "B":
        raise ValueError("audit expects the sent layout-B packet")
    annotation = Attribution(packet)
    annotation.emit(packet)
    if "".join(annotation.parts) != user:
        raise ValueError("actual stored user message does not round-trip to identical serialized bytes")
    source_bytes = Counter(annotation.owners)
    source_bytes[0] = len(system.encode())
    tokens = Counter({0: len(tokenizer.encode_ordinary(system))})
    cursor = crossings = 0
    for fragment in tokenizer.decode_tokens_bytes(tokenizer.encode_ordinary(user)):
        sites = annotation.owners[cursor:cursor + len(fragment)]
        if len(sites) != len(fragment) or not sites:
            raise ValueError("token bytes do not match exact prompt byte offsets")
        if len(set(sites)) > 1:
            crossings += 1
        tokens[Counter(sites).most_common(1)[0][0]] += 1
        cursor += len(fragment)
    if cursor != len(annotation.owners):
        raise ValueError("token bytes do not reconcile to user message")
    wire_size = len(dumps(payload).encode())
    annotation.wire[0] = len(dumps(system).encode()) - 2
    annotation.wire[STRUCTURE] += wire_size - sum(annotation.wire.values())
    model_bytes, model_tokens = sum(source_bytes.values()), sum(tokens.values())
    histogram = Counter()
    for values in annotation.numbers.values():
        histogram.update(values)
    distribution = {field: _length_statistics(values) for field, values in sorted(annotation.numbers.items())}
    return {
        "seed": packet["seed"], "round": packet["round"], "payload_sha256": digest(payload),
        "candidate_count": packet["candidate_count"], "observed_count": packet["observed_count"],
        "tokenizer_proxy": tokenizer.name, "model_content_tokens": model_tokens, "model_content_bytes": model_bytes,
        "wire_payload_bytes": wire_size, "cross_field_token_count": crossings,
        "field_statistics": {category: {
            "content_bytes": source_bytes[index], "content_bytes_pct": round(100 * source_bytes[index] / model_bytes, 5),
            "wire_bytes": annotation.wire[index], "wire_bytes_pct": round(100 * annotation.wire[index] / wire_size, 5),
            "tokens": tokens[index], "tokens_pct": round(100 * tokens[index] / model_tokens, 5),
        } for index, category in enumerate(KINDS)},
        "float_lexeme_lengths": {"all": _length_statistics(histogram), "by_field": distribution},
        "empty_value_occurrences": dict(annotation.empty), "record_structure": structure_statistics(packet),
    }


def synthetic_later_payload(payload, stage, *, hypotheses_per_round=2):
    """Build a sizing fixture from the sent packet's public fields, without any true new observations."""
    if type(stage) is not int or not 0 <= stage <= 36:
        raise ValueError("sizing stage must be in 0..36")
    if type(hypotheses_per_round) is not int or hypotheses_per_round < 0:
        raise ValueError("synthetic hypotheses per round must be a nonnegative integer")
    if not stage:
        return copy.deepcopy(payload)
    packet = json.loads(payload["input"][1]["content"])
    if (packet["round"], packet["observed_count"], packet["candidate_count"]) != (0, 333, 2997):
        raise ValueError("synthetic growth requires the original L333/U2997 packet")
    tables = decode_context(packet)
    candidates, observed = tables["candidates"], tables["observed"]
    rng = random.Random(157)
    moved = sorted(candidates)
    rng.shuffle(moved)
    feedback = []
    for index, candidate_id in enumerate(moved[:32 * stage]):
        candidate = candidates.pop(candidate_id)
        row = {key: candidate[key] for key in ("molecule_id", "smiles", "descriptors", "conditions",
                                                  "max_similarity_to_pending", *sorted(SIGNALS))}
        observed_id = "O" + candidate_id[1:]
        row.update(id=observed_id, record_kind="acquired_with_premeasurement_prediction", acquisition_round=index // 32)
        for target in TARGETS:
            predicted = candidate[f"pred_{target}_q50_ml"]
            # The synthetic residual models string length only; it is never a stand-in for a true label.
            invented = max(0.0, round(predicted + rng.uniform(-0.8, 0.8), 6))
            row.update({f"{target}_ml": invented, f"premeasurement_prediction_{target}_ml": predicted,
                        f"premeasurement_error_{target}_ml": invented - predicted})
        observed[observed_id] = row
        if index // 32 == stage - 1:
            feedback.append({"observed_id": observed_id, **{k: row[k] for k in MEASUREMENT_FIELDS}})
    latest = {"mode": "M2", "latest_feedback": feedback,
              "previous_feedback_interpretation": "Fixture only. " + (
                  "The previous synthetic batch exercises measurement, prediction, and residual serialization. " * 4),
              "hypotheses": [], "archived_hypotheses": []}
    observed_ids = sorted(observed)
    for index in range(stage * hypotheses_per_round):
        archived = index % 3 == 0 and index < (stage - 1) * hypotheses_per_round
        origin = index // hypotheses_per_round
        claim = f"Synthetic claim {index}. A public condition marker accompanies a simulated endpoint shift."
        support = rng.sample(observed_ids, 8)
        item = {"persistent_hypothesis_id": f"H-fixture-{index:03d}", "version": 2 if archived else 1,
                "current_claim": claim, "status": "unresolved", "last_updated_round": origin + archived,
                "supporting_observed_ids": support[:6], "contradicting_observed_ids": support[6:]}
        if archived:
            latest["archived_hypotheses"].append(item)
        else:
            latest["hypotheses"].append({**item, "original_claim": claim, "created_round": origin,
                "basis": "model_behavior", "confidence": "low", "archived": False,
                "unresolved_question": "Synthetic question about a possible condition interaction.",
                "relevance_to_model_learning": "Exercise versioned scientific memory serialization.",
                "revision_reason": "Synthetic fixture generated without an experimental label lookup."})
    envelope = encode_context(SimpleNamespace(seed=packet["seed"], round=stage, candidates=candidates, observed=observed),
                              layout="B", compress=True, memory=latest)
    result = copy.deepcopy(payload)
    result["input"][1]["content"] = dumps(envelope)
    return result


def payload_size(payload, tokenizer):
    return {"wire_payload_bytes": len(dumps(payload).encode()), "model_content_tokens": sum(
        len(tokenizer.encode_ordinary(item["content"])) for item in payload["input"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="actual local job/request/archive or payload JSON")
    parser.add_argument("--tokenizer-cache", required=True, type=Path)
    parser.add_argument("--five-seed-preflight", type=Path)
    parser.add_argument("--compare-compact", action="store_true", help="also size each compact encoding")
    args = parser.parse_args()
    if args.compare_compact:
        from .a7_gpu.compact_payload import candidate_formats

    tokenizer = offline_tokenizer(args.tokenizer_cache)
    payload, request_id = read_payload(args.request)
    report = audit_payload(payload, tokenizer)
    report["request_id"] = request_id
    report["synthetic_growth"] = []
    for stage in (0, 1, 6, 15, 21, 35, 36):
        model = {"stage": stage, "observed_count": 333 + 32 * stage, "candidate_count": 2997 - 32 * stage,
                 "provenance": "actual_sent" if stage == 0 else "synthetic_no_new_labels"}
        model["scenarios"] = {}
        for count in ((0,) if stage == 0 else (0, 2, 8)):
            candidate = synthetic_later_payload(payload, stage, hypotheses_per_round=count)
            sizes = payload_size(candidate, tokenizer)
            if args.compare_compact:
                sizes["compact_encodings"] = {name: payload_size({**candidate, "input": messages}, tokenizer)
                    for name, messages in candidate_formats(candidate["input"]).items()}
            model["scenarios"][str(count)] = sizes
        report["synthetic_growth"].append(model)
    if args.five_seed_preflight:
        saved = json.loads(args.five_seed_preflight.read_bytes())
        ratio = report["model_content_tokens"] / report["wire_payload_bytes"]
        report["five_seed_preflight"] = {seed: {**entry, "scaled_157_token_estimate": round(ratio * entry["payload_bytes"])}
                                         for seed, entry in saved["seeds"].items()}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Generate full production-state requests with explicitly synthetic model replies."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import re

from .context_builder import decode_context
from .frozen_state import load_historical
from .records import digest, dumps, once, read
from .repair_protocol import build_repair_request
from .runner import ROOT, PROTOCOL_PATH, _next_messages, _validation, select_stage, stage_state
from .transport_manager import ProviderReply, TransportConfig, make_request


def mock_plan(stage):
    """Deterministic test fixture. This code is not imported by the live selector."""
    rows = decode_context(stage.packet)["candidates"]
    examples = list(rows.values())[:32]
    plan = {"batch_strategy": "MOCK protocol fixture, with no experimental recommendation.",
            "choices": [], "hypothesis_updates": [],
            "feedback_interpretation": "" if stage.context.round == 0 else
                "MOCK payload illustrates the distinction between measured feedback and current predictions.",
            "batch_review": "MOCK checks slots and references."}
    for row in examples:
        plan["choices"].append({"candidate_id": row["id"], "molecule_id": row["molecule_id"],
            "condition_anchor": {"PE/EA": row["conditions"]["PE/EA"]},
            "reason": "Synthetic fixture for protocol validation.", "hypothesis_id": None, "references": []})
    if stage.context.round:
        observed = list(stage.context.observed)
        support, contradiction = observed[:2]
        plan["hypothesis_updates"] = [{
            "hypothesis_id": "H-mock", "operation": "create", "expected_version": None,
            "claim": "Synthetic fixture proposition for scientific memory.",
            "revision_reason": "Demonstrate support and contradiction representation.",
            "basis": "mixed", "supporting_observed_ids": [support],
            "contradicting_observed_ids": [contradiction], "status": "unresolved", "confidence": "low",
            "unresolved_question": "What would discriminate this synthetic proposition?",
            "relevance_to_model_learning": "Demonstrates traceability only.",
        }]
        plan["choices"][0]["hypothesis_id"] = "H-mock"
        plan["choices"][0]["references"] = [{"reference_id": support, "reference_kind": "observed",
                                              "relationship": "observed_evidence"}]
    if not _validation(stage, plan).valid:
        raise ValueError(_validation(stage, plan).as_dict())
    return plan


def _metrics(request, encode):
    payload = dumps(request["payload"])
    return {"payload_bytes_utf8": len(payload.encode()),
            "payload_characters": len(payload),
            "message_content_characters": sum(len(m["content"]) for m in request["payload"]["input"]),
            "lexical_proxy_units": len(re.findall(r"[A-Za-z]{1,4}|[0-9]{1,3}|[^A-Za-z0-9\s]", payload)),
            "estimated_o200k_base_payload_tokens": len(encode(payload)) if encode else None,
            "estimator": "tiktoken/o200k_base on serialized payload, not provider tokens" if encode else None}


def generate(out, *, tokenize=False):
    config = replace(TransportConfig(**read(ROOT / PROTOCOL_PATH)["transport"]),
                     provider="mock", sdk_version="offline")
    state0, state1 = (load_historical(ROOT, 157, j) for j in range(2))
    first, next_stage = (stage_state(ROOT, state) for state in (state0, state1))
    request0 = make_request(first.binding, 0, first.messages, config)
    request1 = make_request(next_stage.binding, 0, next_stage.messages, config)
    plan = mock_plan(next_stage)
    invalid = copy.deepcopy(plan)
    invalid["choices"][7]["candidate_id"] = "C-9999"
    contract = build_repair_request(invalid, validation=_validation(next_stage, invalid),
                                    binding=next_stage.binding)
    patch = {"mode": "patch", "repair_id": contract["repair_id"], "reason": "Synthetic fixture.",
             "replace_choices": [{"slot": 7, "choice": plan["choices"][7]}],
             "replace_hypothesis_updates": [], "replace_fields": {}, "append_choices": []}
    repair = make_request(next_stage.binding, 1, _next_messages(next_stage, contract), config)
    sample = {"example_kind": "mock_response", "source": "synthetic", "answer": invalid}
    def simulate(request, _config):
        index = request["logical_index"]
        return ProviderReply(dumps((invalid, patch)[index]), f"mock-response-{index}", config.served_model, {})
    result = select_stage(next_stage, out / "mock_run", config, simulate, mock=True)
    once(out / "example_round0_request.json", {"example_kind": "offline_request", **request0})
    once(out / "example_round1_request.json", {"example_kind": "offline_request", **request1})
    once(out / "example_local_repair_request.json", {"example_kind": "offline_request", **repair})
    once(out / "example_response.json", {"example_kind": "mock_response", "source": "synthetic", "answer": plan})
    once(out / "example_initial_rejected_response.json", sample)
    once(out / "example_local_repair_response.json", {"example_kind": "mock_response", "answer": patch})
    memory = read(out / "mock_run" / "memory.json")
    snapshot = {"example_kind": "mock_memory", "seed": 157, "round": 1,
                "hypotheses": memory["hypotheses"], "accepted_plan_sha256": digest(plan),
                "batch_freeze_sha256": digest(result)}
    once(out / "example_memory_update.json", snapshot)
    if tokenize:
        import tiktoken
        encode = tiktoken.get_encoding("o200k_base").encode_ordinary
    else:
        encode = None
    sizes = []
    for state in (state0, state1):
        for arm in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
            stage = stage_state(ROOT, state, arm=arm)
            request = make_request(stage.binding, 0, stage.messages, config)
            sizes.append({"round": state.round, "arm": arm, **_metrics(request, encode)})
    once(out / "serialization_comparison.json", sizes)
    print(dumps({"stage": "mock_verified", "mock_choices": len(result["batch_ids"]),
                 "paid_requests": 0, "new_labels": 0, "training_runs": 0,
                 "items": len(sizes), "size_range": [min(r["payload_bytes_utf8"] for r in sizes),
                                                     max(r["payload_bytes_utf8"] for r in sizes)]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    generate(args.out, tokenize=args.tokenize)


if __name__ == "__main__":
    main()

"""Verify the fresh seed 887 L429 request against its exact archived predecessor."""

import argparse
import hashlib
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig, check_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX, SCHEMA


BOUNDARY_FIELDS = ("arm", "seed", "round", "checkpoint_sha256", "state_sha256", "split_sha256",
                   "mapping_sha256", "packet_sha256", "wire_packet_sha256", "wire_representation")


def _request(record):
    request = record.get("request", record)
    check_request(request, request["binding"], TransportConfig(**request["configuration"]))
    if request["logical_index"] != 0:
        raise ValueError("both jobs must be the first logical request at the L429 boundary")
    messages = request["payload"]["input"]
    if len(messages) != 2 or any(
        set(message) != {"role", "content"} or message["role"] != role or not isinstance(message["content"], str)
        for message, role in zip(messages, ("system", "user"), strict=True)
    ):
        raise ValueError("expected exactly one system message and one user message")
    return request, messages


def verify_prompt_successor(old_request_or_job, new_request_or_job, approved_addendum):
    """Return only identities and sizes if the approved prompt is the sole input change."""
    old, previous = _request(old_request_or_job)
    new, staged = _request(new_request_or_job)
    old_binding, new_binding = old["binding"], new["binding"]
    if (old_binding["arm"], old_binding["seed"], old_binding["round"]) != ("A7", 887, 3):
        raise ValueError("original job is outside seed 887 round 3")
    if any(old_binding[key] != new_binding[key] for key in BOUNDARY_FIELDS):
        raise ValueError("the original checkpoint or A7 packet binding changed")
    if old["configuration"] != new["configuration"]:
        raise ValueError("the original provider configuration changed")
    if staged[1] != previous[1]:
        raise ValueError("the compact user packet must match the original exactly")

    compact = parse_response(previous[1]["content"])
    if (compact["wire_schema"], compact["seed"], compact["round"], compact["observed_count"],
            compact["candidate_count"], compact["selection_count"], compact["memory"]["mode"]) != (
                SCHEMA, 887, 3, 429, 2901, 32, "M2"
            ) or digest(compact) != old_binding["wire_packet_sha256"]:
        raise ValueError("the original user packet is not the bound A7/M2 L429 packet")
    old_prompt = previous[0]["content"]
    addendum = approved_addendum.strip()
    if not addendum or not old_prompt.endswith(FORMAT_SUFFIX):
        raise ValueError("expected the approved addendum and original compact format instructions")
    expected = old_prompt.removesuffix(FORMAT_SUFFIX) + "\n\n" + addendum + FORMAT_SUFFIX
    if staged[0]["content"] != expected:
        raise ValueError("the staged system prompt contains changes outside the approved addendum")

    for binding, messages in ((old_binding, previous), (new_binding, staged)):
        if (digest(messages) != binding["base_messages_sha256"]
                or digest(messages[0]["content"].removesuffix(FORMAT_SUFFIX)) != binding["prompt_sha256"]):
            raise ValueError("base message and prompt hashes must match their request binding")
    if old["request_id"] == new["request_id"]:
        raise ValueError("the approved addendum must generate a fresh request identity")
    return {"seed": 887, "round": 3, "active_labels": 429, "remaining_candidates": 2901,
            "roles": ["system", "user"], "old_request_id": old["request_id"], "new_request_id": new["request_id"],
            "wire_packet_sha256": new_binding["wire_packet_sha256"], "new_payload_sha256": new["payload_sha256"],
            "prior_round_response_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="archived logical-0 request or bridge job JSON")
    parser.add_argument("successor", type=Path, help="newly staged logical-0 request or bridge job JSON")
    parser.add_argument("--approved-addendum", type=Path, required=True)
    parser.add_argument("--approved-addendum-sha256", required=True)
    args = parser.parse_args()
    addendum = args.approved_addendum.read_bytes()
    approved_sha = hashlib.sha256(addendum).hexdigest()
    if approved_sha != args.approved_addendum_sha256:
        parser.error("approved addendum file does not match its recorded SHA-256")
    try:
        receipt = verify_prompt_successor(read(args.original), read(args.successor), addendum.decode("utf-8"))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(dumps({**receipt, "approved_addendum_sha256": approved_sha}))


if __name__ == "__main__":
    main()

"""Frozen A/B/C input envelopes and an offline-only mock transport."""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

from controller import stable_hash


_COMPONENTS = ("labeled_ids", "candidate_ids", "scientific_memory", "candidate_input",
               "candidate_metadata", "baseline_packet", "selector_protocol",
               "llm_configuration", "tool_contract")


def freeze_shared_inputs(*, checkpoint_sha256: str, labeled_ids: Sequence[str],
                         candidate_ids: Sequence[str], scientific_memory: Any,
                         candidate_input: Any, candidate_metadata: Any,
                         baseline_packet: Any, selector_protocol: Any,
                         llm_configuration: Any, tool_contract: Any) -> dict[str, Any]:
    """Bind complete baseline inputs; caller must enforce existing label permissions.

    Hashing proves byte-equivalent JSON components, not semantic label provenance.
    Checkpoint digest must be produced by the caller from its frozen local file.
    """
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256):
        raise ValueError("Checkpoint requires an explicit SHA256 digest")
    components = {"labeled_ids": list(labeled_ids), "candidate_ids": list(candidate_ids),
                  "scientific_memory": scientific_memory, "candidate_input": candidate_input,
                  "candidate_metadata": candidate_metadata, "baseline_packet": baseline_packet,
                  "selector_protocol": selector_protocol, "llm_configuration": llm_configuration,
                  "tool_contract": tool_contract}
    components = copy.deepcopy(components)
    manifest = {"schema_version": 1, "checkpoint_sha256": checkpoint_sha256,
                "components_sha256": {key: stable_hash(components[key]) for key in _COMPONENTS}}
    return {"frozen_state_manifest": manifest, "baseline_components": components}


def validate_shared_freeze(shared: Mapping[str, Any]) -> None:
    if set(shared) != {"frozen_state_manifest", "baseline_components"}:
        raise ValueError("Unexpected frozen shared payload fields")
    manifest, components = shared["frozen_state_manifest"], shared["baseline_components"]
    if set(components) != set(_COMPONENTS) or set(manifest["components_sha256"]) != set(_COMPONENTS):
        raise ValueError("Required frozen input component is missing")
    if not re.fullmatch(r"[a-f0-9]{64}", manifest["checkpoint_sha256"]):
        raise ValueError("Frozen checkpoint digest is invalid")
    for key in _COMPONENTS:
        if stable_hash(components[key]) != manifest["components_sha256"][key]:
            raise ValueError("Frozen input component changed: " + key)


def build_arm_inputs(shared_base_payload: Mapping[str, Any], gp_information: Any) -> dict[str, Any]:
    validate_shared_freeze(shared_base_payload)
    # Arm labels live outside the model input. Initial B and C inputs are identical.
    inputs = {arm: {"shared_base_payload": copy.deepcopy(dict(shared_base_payload)),
                    "gp_information": None if arm == "A" else copy.deepcopy(gp_information),
                    "controller_feedback": []} for arm in ("A", "B", "C")}
    assert_frozen_input_invariants(inputs)
    return inputs


def assert_frozen_input_invariants(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if set(inputs) != {"A", "B", "C"}:
        raise ValueError("All three frozen arms are required")
    for arm in ("A", "B", "C"):
        if set(inputs[arm]) != {"shared_base_payload", "gp_information", "controller_feedback"}:
            raise ValueError("Unregistered model input difference in arm " + arm)
        validate_shared_freeze(inputs[arm]["shared_base_payload"])
    base_hashes = {arm: stable_hash(inputs[arm]["shared_base_payload"]) for arm in inputs}
    if len(set(base_hashes.values())) != 1:
        raise ValueError("A/B/C baseline, state, memory, tools or model configuration differ")
    if inputs["A"]["gp_information"] is not None:
        raise ValueError("Arm A must not receive additional GP information")
    gp_hashes = {arm: stable_hash(inputs[arm]["gp_information"]) for arm in ("B", "C")}
    if gp_hashes["B"] != gp_hashes["C"]:
        raise ValueError("B/C GP information differs")
    if inputs["A"]["controller_feedback"] or inputs["B"]["controller_feedback"]:
        raise ValueError("Only arm C can receive controller feedback")
    return {"shared_base_sha256": base_hashes["A"], "B_C_gp_information_sha256": gp_hashes["B"],
            "base_input_identical": True, "B_C_gp_information_identical": True,
            "C_only_feedback_count": len(inputs["C"]["controller_feedback"])}


def add_mock_controller_feedback(inputs: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    assert_frozen_input_invariants(inputs)
    result = copy.deepcopy(dict(inputs))
    result["C"]["controller_feedback"].append(copy.deepcopy(dict(feedback)))
    assert_frozen_input_invariants(result)
    return result


class MockTransport:
    """A finite in-memory response queue; cannot import or invoke an LLM client."""
    def __init__(self, responses: Sequence[Mapping[str, Any]]):
        self._responses = copy.deepcopy(list(responses))
        self.calls: list[dict[str, Any]] = []

    def respond(self, model_input: Mapping[str, Any]) -> dict[str, Any]:
        if not self._responses:
            raise RuntimeError("Mock response queue exhausted; no network fallback exists")
        response = self._responses.pop(0)
        self.calls.append({"input_sha256": stable_hash(model_input), "input": copy.deepcopy(model_input),
                           "response_sha256": stable_hash(response)})
        return copy.deepcopy(response)

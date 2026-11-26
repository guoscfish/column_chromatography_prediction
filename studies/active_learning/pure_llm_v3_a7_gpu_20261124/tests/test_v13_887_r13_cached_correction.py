from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, response_output_text
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_887_r13_cached_correction as cache


@pytest.fixture
def archived(monkeypatch):
    reason = ("Compare C-0400 (M-0061, PE/EA 2/1, 150 µL) with O-2659 (M-0061, PE/EA 0/1, 150 µL) "
              "to assess the within-compound PE/EA response. O-3192 (M-0062, PE/EA 2/1, 100 µL) "
              "provides a cross-molecule analogy.")
    patch = {"candidate_id": cache.repair.CANDIDATE, "reference_index": 0,
             "reference": cache.repair.REQUIRED_REFERENCE, "reason": reason}
    text = dumps(patch)
    raw = {"id": cache.RESPONSE, "status": "completed", "model": "gpt-6.2-astra",
           "reasoning": {"effort": "xhigh"},
           "output": [{"type": "message", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": " "}], "id": "fixture-placeholder"},
                      {"type": "reasoning", "summary": [{"type": "summary_text", "text": "do not consume this"}]},
                      {"type": "message", "role": "assistant", "phase": "final_answer", "status": "completed",
                       "content": [{"type": "output_text", "text": text}]}],
           "usage": {"input_tokens": cache.USAGE["input_tokens"], "output_tokens": cache.USAGE["output_tokens"],
                     "total_tokens": cache.USAGE["total_tokens"],
                     "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 29}}}
    raw_sha = digest(raw)
    capture = {"attempt_id": cache.ATTEMPT, "job_sha256": cache.JOB,
               "metadata": {"raw_response_sha256": raw_sha, "provider_request_id": cache.PROVIDER_REQUEST_ID,
                            "model_version": None, "reported_reasoning_effort": "xhigh"}}
    for name, value in (("RAW", raw_sha), ("PLACEHOLDER_ITEM", digest(raw["output"][0])),
                        ("FINAL_ITEM", digest(raw["output"][2])), ("TEXT", digest(text)), ("MAC_CAPTURE", digest(capture))):
        monkeypatch.setattr(cache, name, value)
    transport = SimpleNamespace(provider="token4research", requested_model="gpt-6.2-astra",
                                served_model="gpt-6.2-astra", reasoning_effort="xhigh")
    model = {"provider": "token4research", "requested_model": "gpt-6.2-astra", "served_model": "gpt-6.2-astra",
             "requested_reasoning_effort": "xhigh", "reported_reasoning_effort": "xhigh", "reported_model_version": None}
    return raw, transport, model, patch, text


def test_sole_explicit_final_preserves_raw_bytes_original_usage_and_model(archived):
    raw, transport, model, _, text = archived
    original = deepcopy(raw)
    reply = cache._extract_final(raw, transport, model)
    assert raw == original and digest(raw) == cache.RAW
    assert reply["text"] == text == response_output_text(original["output"][1:])
    assert reply["response_id"] == cache.RESPONSE and reply["provider_request_id"] == cache.PROVIDER_REQUEST_ID
    assert reply["usage"] == cache.USAGE and reply["served_model"] == model["served_model"]


@pytest.mark.parametrize("change", ["nonspace", "tag_as_commentary", "absent_final", "unverified_final", "second_final"])
def test_no_substantive_untagged_or_other_final_may_enter_local_import(archived, monkeypatch, change):
    raw, transport, model, _, _ = archived
    if change == "nonspace":
        raw["output"][0]["content"][0]["text"] = "this text cannot be dropped"
    elif change == "tag_as_commentary":
        raw["output"][0]["phase"] = "commentary"
    elif change == "absent_final":
        raw["output"].pop()
    elif change == "unverified_final":
        raw["output"][2]["phase"] = None
    else:
        raw["output"].append(deepcopy(raw["output"][2]))
    monkeypatch.setattr(cache, "RAW", digest(raw))
    monkeypatch.setattr(cache, "PLACEHOLDER_ITEM", digest(raw["output"][0]))
    if len(raw["output"]) > 2:
        monkeypatch.setattr(cache, "FINAL_ITEM", digest(raw["output"][2]))
    with pytest.raises(PermissionError, match="placeholder|item count"):
        cache._extract_final(raw, transport, model)


def test_exact_local_scientific_projection_is_the_only_allowed_patch(archived, monkeypatch):
    _, _, _, patch, _ = archived
    choices = [{"candidate_id": f"C-{index:04}", "references": [], "reason": "original"} for index in range(32)]
    choices[11] = {"candidate_id": cache.repair.CANDIDATE, "molecule_id": cache.repair.MOLECULE,
                   "references": [cache.repair.WRONG_REFERENCE, cache.repair.UNCHANGED_REFERENCE], "reason": "original"}
    normalized = {"choices": choices, "hypothesis_updates": [{"statement": "original update"}]}
    monkeypatch.setattr(cache.repair, "ORDER", digest([row["candidate_id"] for row in choices]))
    monkeypatch.setattr(cache.repair, "PLAN", digest(normalized))
    monkeypatch.setattr(cache, "PATCH", digest(patch))
    expected = deepcopy(normalized)
    expected["choices"][11]["references"][0] = patch["reference"]
    expected["choices"][11]["reason"] = patch["reason"]
    monkeypatch.setattr(cache, "PROJECTED", digest(expected))
    assert cache.project(patch, original=normalized) == expected
    with pytest.raises(PermissionError, match="archived final"):
        cache.project({**patch, "reason": patch["reason"].replace("assess", "test")}, original=normalized)


def test_zero_post_scope_rejects_any_provider_transport_or_bridge_reservation(tmp_path):
    assert cache.BUDGET["maximum_physical_attempts"] == 0 and cache.repair.TARGET == 781
    with pytest.raises(TransportStop, match="no provider transport"):
        cache._no_provider()
    cache._require_no_bridge(tmp_path)
    outbox = tmp_path / "bridge" / "outbox"
    outbox.mkdir(parents=True)
    (outbox / "unexpected.json").write_text("{}")
    with pytest.raises(PermissionError, match="zero-POST"):
        cache._require_no_bridge(tmp_path)

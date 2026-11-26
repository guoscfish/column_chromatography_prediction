from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _observed_alias_projection
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_887_r13_targeted_repair as repair


@pytest.fixture
def archived_plan(monkeypatch):
    def reference(identity):
        return {"reference_id": identity, "observed_id": identity, "reference_kind": "observed",
                "relationship": "condition_response"}

    choices = [{"candidate_id": f"C-{index:04}", "reason": f"scientific reason {index}",
                "references": [reference("O-0001"), reference("O-0002")]} for index in range(32)]
    choices[-1]["references"] = []
    choices[repair.CHOICE_INDEX] = {
        "candidate_id": repair.CANDIDATE, "molecule_id": repair.MOLECULE, "condition_anchor": {"PE/EA": "2/1"},
        "hypothesis_id": None, "reason": "original reason",
        "references": [reference(repair.OBSERVED), deepcopy(repair.UNCHANGED_REFERENCE)],
    }
    raw = {"choices": choices, "hypothesis_updates": [{"statement": "archived"}],
           "batch_review": "original review", "batch_strategy": "original strategy"}
    normalized, paths = _observed_alias_projection(raw)
    assert len(paths) == repair.REMOVED_ALIASES
    monkeypatch.setattr(repair, "ANSWER", digest(dumps(raw)))
    monkeypatch.setattr(repair, "ORIGINAL_PLAN", digest(raw))
    monkeypatch.setattr(repair, "PLAN", digest(normalized))
    monkeypatch.setattr(repair, "ORDER", digest([choice["candidate_id"] for choice in choices]))
    monkeypatch.setattr(repair, "ALIAS_PATHS", digest(paths))
    return raw, normalized


@pytest.fixture
def correction():
    return {"candidate_id": repair.CANDIDATE, "reference_index": repair.REFERENCE_INDEX,
            "reference": deepcopy(repair.REQUIRED_REFERENCE),
            "reason": ("Compare C-0400 (M-0061, PE/EA 2/1, 150 µL) with O-2659 (M-0061, PE/EA 0/1, 150 µL) "
                       "to test the within-compound PE/EA response. O-3192 (M-0062, PE/EA 2/1, 100 µL) "
                       "provides a cross-molecule analogy.")}


def test_exact_alias_projection_and_single_reference_repair_preserve_choices_and_memory(archived_plan, correction):
    raw, normalized = archived_plan
    assert repair._original_plan(dumps(raw)) == normalized
    assert sum("observed_id" in ref for choice in raw["choices"] for ref in choice["references"]) == 61
    assert all("observed_id" not in ref for choice in normalized["choices"] for ref in choice["references"])
    projected = repair.project_reply(correction, original=normalized)
    expected = deepcopy(normalized)
    expected["choices"][11]["reason"] = correction["reason"]
    expected["choices"][11]["references"][0] = correction["reference"]
    assert projected == expected
    assert projected["choices"][11]["references"][1] == repair.UNCHANGED_REFERENCE
    assert [row["candidate_id"] for row in projected["choices"]] == [row["candidate_id"] for row in raw["choices"]]


def test_archive_rejects_conflicting_observed_alias_even_with_matching_original_hashes(archived_plan, monkeypatch):
    original, _ = archived_plan
    original["choices"][0]["references"][0]["observed_id"] = "O-0900"
    text = dumps(original)
    monkeypatch.setattr(repair, "ANSWER", digest(text))
    monkeypatch.setattr(repair, "ORIGINAL_PLAN", digest(original))
    with pytest.raises(PermissionError, match="exact observed mirrors"):
        repair._original_plan(text)


@pytest.mark.parametrize("change", (
    {"candidate_id": "C-0040"}, {"reference_index": True}, {"reference_index": 1},
    {"choices": []}, {"hypothesis_updates": []}, {"reference": repair.WRONG_REFERENCE},
    {"reference": {**repair.REQUIRED_REFERENCE, "reference_id": repair.SAME_MOLECULE_OBSERVED}},
    {"reference": {**repair.REQUIRED_REFERENCE, "changed_axis": "sample_volume"}},
    {"reason": "C-0400 and O-3192 have the same structure and test a loading change."},
    {"reason": "O-2659 is the measured 2/1 100 µL point for C-0400."},
))
def test_patch_cannot_modify_any_other_choice_or_preserve_the_false_reason(archived_plan, correction, change):
    with pytest.raises(ResponseFormatError, match="sole affected"):
        repair.project_reply({**correction, **change}, original=archived_plan[1])


def test_exact_original_payload_with_only_the_signed_scoped_guide_and_one_attempt():
    original = {"request_id": repair.REQUEST, "payload_sha256": repair.PAYLOAD,
                "binding": {field: field for field in repair.DATA_BINDINGS},
                "payload": {"model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"}, "stream": True, "store": False,
                            "input": [{"role": "system", "content": "Original science" + FORMAT_SUFFIX},
                                      {"role": "user", "content": "Exact archived scientific input"}]}}
    metadata = repair._metadata(original)
    system = repair.patch_system(original)
    request = {"request_id": "b" * 64, "logical_index": 0,
               "binding": {**original["binding"], "prompt_sha256": digest(system), "fixed_reference_repair": metadata},
               "payload": {**original["payload"], "input": [{"role": "system", "content": system + FORMAT_SUFFIX},
                                                           original["payload"]["input"][1]]}}
    repair.verify_patch_request(request, original, metadata, attempt=0)
    assert metadata["removed_alias_count"] == 61 and metadata["reference_index"] == 0
    for changed, attempt in ((request, 1), ({**request, "request_id": repair.REQUEST}, 0),
                             ({**request, "logical_index": 1}, 0),
                             ({**request, "payload": {**request["payload"], "input": [
                                 request["payload"]["input"][0], {"role": "user", "content": "Changed packet"}]}}, 0),
                             ({**request, "payload": {**request["payload"], "reasoning": {"effort": "max"}}}, 0),
                             ({**request, "binding": {**request["binding"], "packet_sha256": "changed"}}, 0)):
        with pytest.raises(PermissionError):
            repair.verify_patch_request(changed, original, metadata, attempt=attempt)

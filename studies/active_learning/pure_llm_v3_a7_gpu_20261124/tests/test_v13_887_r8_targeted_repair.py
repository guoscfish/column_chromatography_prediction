from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_887_r8_targeted_repair as repair


@pytest.fixture
def source_plan(monkeypatch):
    choices = [{"candidate_id": f"C-{index:04}", "reason": f"scientific reason {index}"} for index in range(31)]
    choices.append({"candidate_id": repair.CANDIDATE, "molecule_id": repair.MOLECULE,
                    "condition_anchor": {"PE/EA": "20/1"}, "hypothesis_id": None, "reason": "original reason",
                    "references": [deepcopy(repair.WRONG_REFERENCE)]})
    plan = {"choices": choices, "hypothesis_updates": [], "batch_review": "archive", "batch_strategy": "archive"}
    monkeypatch.setattr(repair, "PLAN", digest(plan))
    monkeypatch.setattr(repair, "ANSWER", digest(dumps(plan)))
    monkeypatch.setattr(repair, "ORDER", digest([choice["candidate_id"] for choice in choices]))
    return plan


@pytest.fixture
def patch():
    return {"candidate_id": repair.CANDIDATE, "reference_index": 0, "reference": deepcopy(repair.REQUIRED_REFERENCE),
            "reason": ("Compare C-1232 at 100 µL and 10 mL/min with O-2392 at 150 µL and 5 mL/min "
                       "to assess joint response to sample volume and flow.")}


def test_projection_retains_every_field_besides_the_affected_reference_and_reason(source_plan, patch):
    assert repair._original_plan(dumps(source_plan)) == source_plan
    projected = repair.project_reply(patch, original=source_plan)
    expected = deepcopy(source_plan)
    expected["choices"][31]["reason"] = patch["reason"]
    expected["choices"][31]["references"][0] = patch["reference"]
    assert projected == expected
    assert source_plan["choices"][31]["references"][0] == repair.WRONG_REFERENCE
    assert source_plan["choices"][31]["reason"] == "original reason"
    assert [x["candidate_id"] for x in projected["choices"]] == [x["candidate_id"] for x in source_plan["choices"]]


@pytest.mark.parametrize("changes", (
    {"candidate_id": "C-0199"}, {"reference_index": True}, {"reference_index": 1}, {"choices": []},
    {"reference": {**repair.REQUIRED_REFERENCE, "changed_axis": "sample_volume"}},
    {"reference": {**repair.REQUIRED_REFERENCE, "reference_id": "O-0003"}},
    {"reference": {**repair.REQUIRED_REFERENCE, "reference_kind": "candidate"}},
    {"reference": repair.WRONG_REFERENCE},
    {"reason": "C-1232 at 100 µL and 5 mL/min versus O-2392 at 150 µL and 10 mL/min."},
    {"reason": "O-2392 at 100 µL and 10 mL/min versus C-1232 at 150 µL and 5 mL/min."},
    {"reason": "C-1232 at 100 µL and 10 mL/min versus O-2392 at 150 µL and 5 mL/min isolates flow."},
    {"reason": "C-1232 at 100 µL and 10 mL/min versus O-2392 at 150 µL and 5 mL/min or 20 mL/min."},
))
def test_other_candidates_references_and_false_condition_claims_are_rejected(source_plan, patch, changes):
    with pytest.raises(ResponseFormatError):
        repair.project_reply({**patch, **changes}, original=source_plan)


def test_only_approved_guide_may_change_in_new_provider_request():
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
    for altered, attempt in ((request, 1), (request, False), ({**request, "request_id": repair.REQUEST}, 0),
                             ({**request, "logical_index": 1}, 0),
                             ({**request, "payload": {**request["payload"], "reasoning": {"effort": "max"}}}, 0),
                             ({**request, "binding": {**request["binding"], "wire_packet_sha256": "changed"}}, 0),
                             ({**request, "payload": original["payload"]}, 0)):
        with pytest.raises(PermissionError):
            repair.verify_patch_request(altered, original, metadata, attempt=attempt)

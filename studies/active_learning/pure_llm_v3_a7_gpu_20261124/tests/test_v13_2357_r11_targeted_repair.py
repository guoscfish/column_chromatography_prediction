from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_2357_r11_targeted_repair as dispatch,
    run_v13_2357_r11_targeted_repair as repair,
)


@pytest.fixture
def plan(monkeypatch):
    choices = [{"candidate_id": f"C-{index:04}", "references": [], "reason": f"reason {index}"}
               for index in range(32)]
    choices[repair.CHOICE_INDEX] = {
        "candidate_id": repair.CANDIDATE, "molecule_id": repair.MOLECULE, "hypothesis_id": repair.INVALID_HYPOTHESIS,
        "reason": "two observed samples support this contrast",
        "references": [{"reference_id": "O-1681", "reference_kind": "observed",
                        "relationship": "single_factor_contrast", "changed_axis": "sample_volume"}],
        "condition_anchor": {"PE/EA": "20/1"},
    }
    original = {"choices": choices, "hypothesis_updates": [], "batch_strategy": "preserve",
                "batch_review": "preserve", "feedback_interpretation": "preserve"}
    monkeypatch.setattr(repair, "PLAN", digest(original))
    monkeypatch.setattr(repair, "ANSWER", digest(dumps(original)))
    monkeypatch.setattr(repair, "ORDER", digest([entry["candidate_id"] for entry in choices]))
    return original


def test_projection_changes_only_rank_32_optional_hypothesis_id(plan):
    archived = deepcopy(plan)
    assert repair._original_plan(dumps(plan)) == archived
    expected = deepcopy(plan)
    expected["choices"][31]["hypothesis_id"] = None
    assert repair.project_reply({"candidate_id": repair.CANDIDATE, "hypothesis_id": None}, original=plan) == expected
    assert archived == plan


@pytest.mark.parametrize("patch", [
    {"candidate_id": "C-0000", "hypothesis_id": None},
    {"candidate_id": repair.CANDIDATE, "hypothesis_id": repair.INVALID_HYPOTHESIS},
    {"candidate_id": repair.CANDIDATE, "hypothesis_id": "H-AcetoacetateLoad10"},
    {"candidate_id": repair.CANDIDATE, "hypothesis_id": None, "reason": "changed"},
    {"candidate_id": repair.CANDIDATE, "hypothesis_id": None, "hypothesis_updates": []},
    {"candidate_id": repair.CANDIDATE},
])
def test_projection_denies_any_other_scope(plan, patch):
    with pytest.raises(ResponseFormatError, match="sole authorized"):
        repair.project_reply(patch, original=plan)


def test_projection_denies_mutated_archived_candidate_order(plan):
    plan["choices"][0], plan["choices"][1] = plan["choices"][1], plan["choices"][0]
    with pytest.raises(PermissionError, match="archived 2357 provider plan"):
        repair.project_reply({"candidate_id": repair.CANDIDATE, "hypothesis_id": None}, original=plan)


def test_only_one_scoped_request_with_unchanged_original_user_packet_and_model():
    old = {"request_id": repair.REQUEST, "payload_sha256": repair.PAYLOAD,
           "binding": {field: field for field in repair.DATA_BINDINGS},
           "payload": {"model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"}, "stream": True, "store": False,
                       "input": [{"role": "system", "content": "Original science" + FORMAT_SUFFIX},
                                 {"role": "user", "content": "Archived scientific packet"}]}}
    system = repair.patch_system(old)
    meta = repair._metadata(old)
    req = {"request_id": "new", "logical_index": 0,
           "binding": {**old["binding"], "prompt_sha256": digest(system), "fixed_reference_repair": meta},
           "payload": {**old["payload"], "input": [{"role": "system", "content": system + FORMAT_SUFFIX},
                                                 old["payload"]["input"][1]]}}
    assert meta["repair_index"] == 31 and meta["field"] == "hypothesis_id"
    repair.verify_patch_request(req, old, meta, attempt=0)
    for changed, attempt in [(req, 1), ({**req, "request_id": repair.REQUEST}, 0), ({**req, "logical_index": 1}, 0),
                             ({**req, "payload": {**req["payload"], "reasoning": {"effort": "max"}}}, 0),
                             ({**req, "binding": {**req["binding"], "packet_sha256": "changed"}}, 0)]:
        with pytest.raises(PermissionError):
            repair.verify_patch_request(changed, old, meta, attempt=attempt)


def test_dispatcher_denies_original_completed_physical_attempt(monkeypatch):
    monkeypatch.setattr(dispatch.Dispatcher, "validate",
                        lambda _self, _job: ({}, None, {"attempt_id": repair.ATTEMPT}))
    dispatcher = dispatch.RepairDispatcher.__new__(dispatch.RepairDispatcher)
    with pytest.raises(PermissionError, match="cannot be submitted again"):
        dispatcher.validate({})

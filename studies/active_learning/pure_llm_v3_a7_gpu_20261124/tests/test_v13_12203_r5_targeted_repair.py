from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_12203_r5_targeted_repair as dispatch,
    run_v13_12203_r5_targeted_repair as repair,
)


@pytest.fixture
def archived(monkeypatch):
    choices = [{"candidate_id": f"C-{index:04}", "reason": "unaltered reason", "references": []} for index in range(32)]
    for scope in repair.REPAIRS:
        choices[scope["choice_index"]] = {
            "candidate_id": scope["candidate"], "molecule_id": scope["molecule"],
            "reason": "archived old reason", "references": [deepcopy(scope["old_reference"]), {"retained": True}],
            "hypothesis_id": None, "condition_anchor": {"PE/EA": "20/1"},
        }
    original = {"choices": choices, "hypothesis_updates": [], "batch_review": "the same", "batch_strategy": "the same"}
    monkeypatch.setattr(repair, "PLAN", digest(original))
    monkeypatch.setattr(repair, "ANSWER", digest(dumps(original)))
    monkeypatch.setattr(repair, "ORDER", digest([choice["candidate_id"] for choice in choices]))
    patch = {"repairs": [{"candidate_id": row["candidate"], "reference_index": row["ref_index"],
                          "reference": deepcopy(row["corrected_reference"]), "reason": row["reason"]}
                         for row in repair.REPAIRS]}
    return original, patch


def test_only_three_reference_and_reason_fields_change(archived):
    original, patch = archived
    before = deepcopy(original)
    assert repair._original_plan(dumps(original)) == before
    expected = deepcopy(original)
    for index, item in zip((12, 15, 30), patch["repairs"], strict=True):
        choice = expected["choices"][index]
        choice["references"][0] = item["reference"]
        choice["reason"] = item["reason"]
    assert repair.project_reply(patch, original=original) == expected
    assert original == before


@pytest.mark.parametrize("wrong", [
    {"candidate_id": "C-new"}, {"reference_index": True}, {"reference_index": 1},
    {"reference": {"reference_id": "O-new"}}, {"reason": "states the wrong comparison"},
    {"hypothesis_id": None}, {"references": []},
])
def test_repair_cannot_change_candidates_extras_or_condition_facts(archived, wrong):
    original, patch = archived
    patch["repairs"][0].update(wrong)
    with pytest.raises(ResponseFormatError):
        repair.project_reply(patch, original=original)


def test_repair_cannot_reorder_patches_or_the_original_choices(archived):
    original, patch = archived
    with pytest.raises(ResponseFormatError):
        repair.project_reply({"repairs": list(reversed(patch["repairs"]))}, original=original)
    original["choices"].reverse()
    with pytest.raises(PermissionError, match="archived 12203"):
        repair.project_reply(patch, original=original)


def test_verified_patch_reuses_packet_and_model_with_one_physical_slot():
    original = {"request_id": repair.REQUEST, "payload_sha256": repair.PAYLOAD,
                "binding": {field: field for field in repair.DATA_BINDINGS},
                "payload": {"model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"}, "stream": True, "store": False,
                            "input": [{"role": "system", "content": "Original science" + FORMAT_SUFFIX},
                                      {"role": "user", "content": "Archived scientific input"}]}}
    system = repair.patch_system(original)
    metadata = repair._metadata(original)
    request = {"request_id": "new", "logical_index": 0,
               "binding": {**original["binding"], "prompt_sha256": digest(system), "fixed_reference_repair": metadata},
               "payload": {**original["payload"], "input": [{"role": "system", "content": system + FORMAT_SUFFIX},
                                                           original["payload"]["input"][1]]}}
    repair.verify_patch_request(request, original, metadata, attempt=0)
    assert [item["choice_index"] for item in metadata["repairs"]] == [12, 15, 30]
    for changed, attempt in [(request, 1), ({**request, "request_id": repair.REQUEST}, 0),
                             ({**request, "logical_index": 1}, 0),
                             ({**request, "payload": {**request["payload"], "model": "different"}}, 0)]:
        with pytest.raises(PermissionError):
            repair.verify_patch_request(changed, original, metadata, attempt=attempt)


def test_reviewed_history_refuses_wrong_canonical_proof_before_archive_reads():
    incorrect = {key: None for key in repair.ARCHIVE_FIELDS}
    with pytest.raises(PermissionError, match="W72 reviewed archive proof"):
        repair._reviewed_archive(None, SimpleNamespace(predecessor=incorrect))


def test_dispatcher_denies_original_completed_attempt(monkeypatch):
    monkeypatch.setattr(dispatch.Dispatcher, "validate",
                        lambda _self, _job: ({}, None, {"attempt_id": repair.ATTEMPT}))
    dispatcher = dispatch.RepairDispatcher.__new__(dispatch.RepairDispatcher)
    with pytest.raises(PermissionError, match="cannot be submitted again"):
        dispatcher.validate({})

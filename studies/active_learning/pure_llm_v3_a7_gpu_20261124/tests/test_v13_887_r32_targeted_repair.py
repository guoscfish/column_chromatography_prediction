from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_887_r32_targeted_repair as dispatch,
    run_v13_887_r32_targeted_repair as repair,
)


@pytest.fixture
def archived_plan(monkeypatch):
    choices = [{"candidate_id": f"C-{index:04}", "reason": f"archived reason {index}", "references": []}
               for index in range(32)]
    choices[repair.CHOICE_INDEX] = {
        "candidate_id": repair.CANDIDATE, "molecule_id": repair.MOLECULE, "condition_anchor": {"PE/EA": "0/1"},
        "hypothesis_id": None, "reason": "archived DCM vs PE. " + repair.PRESERVED_REASON,
        "references": [deepcopy(repair.WRONG_REFERENCE), *deepcopy(repair.UNCHANGED_REFERENCES)],
    }
    plan = {"choices": choices, "hypothesis_updates": [{"statement": "archived"}],
            "batch_review": "archived review", "batch_strategy": "archived strategy"}
    monkeypatch.setattr(repair, "ANSWER", digest(dumps(plan)))
    monkeypatch.setattr(repair, "PLAN", digest(plan))
    monkeypatch.setattr(repair, "ORDER", digest([choice["candidate_id"] for choice in choices]))
    return plan


@pytest.fixture
def scoped_patch():
    return {"candidate_id": repair.CANDIDATE, "reference_index": 0, "reference": deepcopy(repair.REQUIRED_REFERENCE),
            "reason": ("Compare the DCM-loaded C-1591 with the PE-loaded O-1261 under neat EA "
                       "to test the effect of loading solvent. " + repair.PRESERVED_REASON)}


def test_archive_and_projection_preserve_all_choices_and_memory(archived_plan, scoped_patch):
    assert repair._original_plan(dumps(archived_plan)) == archived_plan
    original = deepcopy(archived_plan)
    expected = deepcopy(archived_plan)
    expected["choices"][22]["references"][0] = scoped_patch["reference"]
    expected["choices"][22]["reason"] = scoped_patch["reason"]
    assert repair.project_reply(scoped_patch, original=archived_plan) == expected
    assert archived_plan == original


@pytest.mark.parametrize("change", (
    {"candidate_id": "C-0000"}, {"reference_index": 1}, {"reference_index": True}, {"choices": []},
    {"hypothesis_updates": []}, {"reference": repair.WRONG_REFERENCE},
    {"reference": {**repair.REQUIRED_REFERENCE, "reference_id": "O-2975"}},
    {"reference": {**repair.REQUIRED_REFERENCE, "changed_axis": "PE/EA"}},
    {"reference": {**repair.REQUIRED_REFERENCE, "origin": "another choice"}},
    {"reason": "A different molecule under many simultaneous changes."},
    {"reason": ("Compare the PE-loaded C-1591 with the DCM-loaded O-1261 under neat EA "
                "to test the effect of loading solvent. " + repair.PRESERVED_REASON)},
    {"reason": "Compare the DCM-loaded C-1591 with the PE-loaded O-1261 under neat EA to test the effect of loading solvent."},
))
def test_projection_rejects_scope_or_fact_changes(archived_plan, scoped_patch, change):
    with pytest.raises(ResponseFormatError, match="sole affected"):
        repair.project_reply({**scoped_patch, **change}, original=archived_plan)


def test_original_batch_change_rejected_even_with_other_valid_shapes(archived_plan, scoped_patch):
    changed = deepcopy(archived_plan)
    changed["choices"][0], changed["choices"][1] = changed["choices"][1], changed["choices"][0]
    with pytest.raises(PermissionError, match="cached 887 provider plan"):
        repair.project_reply(scoped_patch, original=changed)
    with pytest.raises(PermissionError, match="32 ordered"):
        repair._original_plan(dumps(changed))


def test_only_the_new_scoped_prompt_one_physical_and_full_original_wire_body():
    original = {"request_id": repair.REQUEST, "payload_sha256": repair.PAYLOAD,
                "binding": {field: field for field in repair.DATA_BINDINGS},
                "payload": {"model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"}, "stream": True, "store": False,
                            "input": [{"role": "system", "content": "Original science" + FORMAT_SUFFIX},
                                      {"role": "user", "content": "Archived scientific input"}]}}
    system = repair.patch_system(original)
    metadata = repair._metadata(original)
    request = {"request_id": "b" * 64, "logical_index": 0,
               "binding": {**original["binding"], "prompt_sha256": digest(system), "fixed_reference_repair": metadata},
               "payload": {**original["payload"], "input": [{"role": "system", "content": system + FORMAT_SUFFIX},
                                                           original["payload"]["input"][1]]}}
    repair.verify_patch_request(request, original, metadata, attempt=0)
    assert metadata["repair_index"] == 22 and metadata["changed_axis"] == "loading_solvent"
    for changed, attempt in ((request, 1), ({**request, "request_id": repair.REQUEST}, 0),
                             ({**request, "logical_index": 1}, 0),
                             ({**request, "payload": {**request["payload"], "input": [
                                 request["payload"]["input"][0], {"role": "user", "content": "Changed packet"}]}}, 0),
                             ({**request, "payload": {**request["payload"], "reasoning": {"effort": "max"}}}, 0),
                             ({**request, "binding": {**request["binding"], "packet_sha256": "changed"}}, 0),
                             ({**request, "binding": {**request["binding"], "normalize_observed_reference_aliases": True}}, 0),
                             ({**request, "binding": {**request["binding"], "fixed_reference_repair": {}}}, 0)):
        with pytest.raises(PermissionError):
            repair.verify_patch_request(changed, original, metadata, attempt=attempt)


def test_dispatcher_never_reuses_original_attempt(monkeypatch):
    monkeypatch.setattr(dispatch.Dispatcher, "validate",
                        lambda self, job: ({}, None, {"attempt_id": repair.ATTEMPT}))
    dispatcher = dispatch.RepairDispatcher.__new__(dispatch.RepairDispatcher)
    with pytest.raises(PermissionError, match="cannot be submitted again"):
        dispatcher.validate({})

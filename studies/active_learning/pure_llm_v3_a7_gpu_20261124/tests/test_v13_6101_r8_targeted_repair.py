from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_6101_r8_targeted_repair as dispatch,
    run_v13_6101_r8_targeted_repair as repair,
)


@pytest.fixture
def archived_plan(monkeypatch):
    choices = [{"candidate_id": f"C-{index:04}", "reason": f"archived reason {index}", "references": []}
               for index in range(32)]
    for target in repair.REPAIRS:
        choices[target["index"]] = {
            "candidate_id": target["candidate"], "molecule_id": target["molecule"], "condition_anchor": {"PE/EA": "0/1"},
            "hypothesis_id": None, "reason": "archived reason",
            "references": [repair._reference(target, target["old_relationship"]),
                           {"reference_id": "O-5555", "reference_kind": "observed", "relationship": "observed_evidence"}],
        }
    plan = {"choices": choices, "hypothesis_updates": [{"statement": "archived"}],
            "batch_review": "archived review", "batch_strategy": "archived strategy"}
    monkeypatch.setattr(repair, "ANSWER", digest(dumps(plan)))
    monkeypatch.setattr(repair, "PLAN", digest(plan))
    monkeypatch.setattr(repair, "ORDER", digest([choice["candidate_id"] for choice in choices]))
    return plan


@pytest.fixture
def scoped_patch():
    return {"repairs": [{"candidate_id": target["candidate"], "reference_index": 0,
                         "reference": repair._reference(target, "cross_molecule_analogy"),
                         "reason": target["reason"].replace("µL", "uL").replace("Compare", "Assess", 1)}
                        for target in repair.REPAIRS]}


def test_projection_preserves_all_ordered_choices_and_scientific_memory(archived_plan, scoped_patch):
    assert repair._original_plan(dumps(archived_plan)) == archived_plan
    expected = deepcopy(archived_plan)
    for item, target in zip(scoped_patch["repairs"], repair.REPAIRS, strict=True):
        expected["choices"][target["index"]]["references"][0] = item["reference"]
        expected["choices"][target["index"]]["reason"] = item["reason"]
    assert repair.project_reply(scoped_patch, original=archived_plan) == expected
    assert digest(archived_plan) == repair.PLAN


@pytest.mark.parametrize("change", (
    {"candidate_id": "C-0000"}, {"reference_index": 1}, {"reference_index": True},
    {"reference": {"reference_id": "O-0124", "reference_kind": "observed", "relationship": "replicate"}},
    {"reference": {"reference_id": "O-1543", "reference_kind": "observed", "relationship": "cross_molecule_analogy"}},
    {"reference": {"reference_id": "O-0124", "reference_kind": "candidate", "relationship": "cross_molecule_analogy"}},
    {"reason": "C-1496 replicates O-0124 and isolates a loading-solvent effect."},
    {"reason": repair.REPAIRS[0]["reason"].replace("100", "150")}, {"choices": []},
))
def test_refuses_changes_outside_single_relationship_and_reason_for_each_choice(archived_plan, scoped_patch, change):
    scoped_patch["repairs"][0].update(change)
    with pytest.raises(ResponseFormatError, match="two scoped"):
        repair.project_reply(scoped_patch, original=archived_plan)


def test_rejects_extra_missing_or_reordered_patch_items(archived_plan, scoped_patch):
    for change in ({"repairs": scoped_patch["repairs"][:1]}, {"repairs": scoped_patch["repairs"] * 2},
                   {"repairs": list(reversed(scoped_patch["repairs"]))}, {**scoped_patch, "hypothesis_updates": []}):
        with pytest.raises(ResponseFormatError, match="two scoped"):
            repair.project_reply(change, original=archived_plan)


def test_rejects_an_archive_whose_32_choices_were_reordered(archived_plan, scoped_patch):
    changed = deepcopy(archived_plan)
    changed["choices"][0], changed["choices"][1] = changed["choices"][1], changed["choices"][0]
    with pytest.raises(PermissionError, match="cached 6101 provider plan"):
        repair.project_reply(scoped_patch, original=changed)
    with pytest.raises(PermissionError, match="32 ordered"):
        repair._original_plan(dumps(changed))


def test_same_wire_body_and_one_distinct_physical_request():
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
    assert [item["choice_index"] for item in metadata["repairs"]] == [16, 17]
    for changed, attempt in ((request, 1), ({**request, "request_id": repair.REQUEST}, 0),
                             ({**request, "logical_index": 1}, 0),
                             ({**request, "payload": {**request["payload"], "input": [
                                 request["payload"]["input"][0], {"role": "user", "content": "Changed packet"}]}}, 0),
                             ({**request, "payload": {**request["payload"], "reasoning": {"effort": "max"}}}, 0),
                             ({**request, "binding": {**request["binding"], "packet_sha256": "changed"}}, 0),
                             ({**request, "binding": {**request["binding"], "normalize_observed_reference_aliases": True}}, 0)):
        with pytest.raises(PermissionError):
            repair.verify_patch_request(changed, original, metadata, attempt=attempt)


@pytest.mark.parametrize("attempt_id", (repair.P0_ATTEMPT, repair.ATTEMPT))
def test_dispatcher_refuses_both_archived_physical_attempts(monkeypatch, attempt_id):
    monkeypatch.setattr(dispatch.Dispatcher, "validate", lambda self, job: ({}, None, {"attempt_id": attempt_id}))
    dispatcher = dispatch.RepairDispatcher.__new__(dispatch.RepairDispatcher)
    with pytest.raises(PermissionError, match="cannot be submitted again"):
        dispatcher.validate({})


@pytest.mark.parametrize("extra", (None, "physical", "logical"))
def test_construct_requires_exactly_the_two_historical_physical_slots(tmp_path, monkeypatch, extra):
    old = tmp_path / repair.PARENT.name
    seed = old / f"seed_{repair.SEED}"
    active = seed / "round_08"
    turn = active / "turn_00"
    turn.mkdir(parents=True)
    for index, attempt in enumerate((repair.P0_ATTEMPT, repair.ATTEMPT)):
        (turn / f"attempt_{index:02d}.started.json").write_text(
            dumps({"attempt": index, "attempt_id": attempt, "request_id": repair.REQUEST})
        )
    if extra == "physical":
        (turn / "attempt_02.started.json").write_text(dumps({"attempt": 2}))
    if extra == "logical":
        other = active / "turn_01"
        other.mkdir()
        (other / "request.json").write_text(dumps({"logical_index": 1}))

    monkeypatch.setattr(repair.relay.bounded, "_archived", lambda *args: (object(), object()))
    monkeypatch.setattr(repair.relay, "_retired", lambda *args, **kwargs: (seed, active))
    inspected = []

    def inspect_digest(path):
        inspected.append(path)
        return "not-a-pinned-digest"

    monkeypatch.setattr(repair, "file_digest", inspect_digest)
    message = {None: "terminal or decision receipt changed", "physical": "different reservations",
               "logical": "another logical proposal"}[extra]
    with pytest.raises(PermissionError, match=message):
        repair.construct(tmp_path / "release", tmp_path / "previous-release", old, tmp_path / repair.RUN_NAME)
    if extra is None:
        assert turn / "attempt_01.terminal.json" in inspected
        assert turn / "attempt_00.terminal.json" not in inspected
    else:
        assert inspected == []

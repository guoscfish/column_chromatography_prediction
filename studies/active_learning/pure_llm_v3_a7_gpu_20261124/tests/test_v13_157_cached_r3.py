from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import CONDITION_AXES
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_157_cached_r3 as mac,
    run_v13_157_cached_r3 as gpu,
)


@pytest.fixture
def reviewed_reply(monkeypatch):
    choices = []
    paths = []
    for index in range(32):
        identity = "C-2926" if index == 18 else f"C-{1000 + index}"
        references = []
        for slot in range(2 if index < 22 else 1):
            exceptional = (index, slot) == (18, 0)
            observed = "O-0959" if exceptional else f"O-{2000 + 2 * index + slot}"
            reference = {"reference_id": observed, "reference_kind": "observed", "observed_id": observed,
                         "relationship": "observed_evidence"}
            paths.append(f"/choices/{index}/references/{slot}/observed_id")
            if index < 25 and slot == 0:
                reference["candidate_id"] = identity
                paths.append(f"/choices/{index}/references/{slot}/candidate_id")
            if exceptional:
                reference.update(reference_kind="single_factor_contrast", relationship="single_factor_contrast", changed_axis="PE/EA")
            references.append(reference)
        choices.append({"candidate_id": identity, "molecule_id": "M-0130", "references": references})
    plan = {"choices": choices}
    normalized = deepcopy(plan)
    normalized["choices"][18]["references"][0]["reference_kind"] = "observed"
    for choice in normalized["choices"]:
        for reference in choice["references"]:
            reference.pop("observed_id")
            reference.pop("candidate_id", None)
    text = dumps(plan)
    raw = {"id": gpu.RESPONSE_ID, "status": "completed", "output": [{"phase": "final_answer", "type": "message",
        "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}
    for name, value in {"RAW_SHA": digest(raw), "FINAL_SHA": digest(text), "PARSED_SHA": digest(plan),
                        "PLAN_SHA": digest(normalized), "PATHS_SHA": digest(paths),
                        "ORDER_SHA": digest([choice["candidate_id"] for choice in choices])}.items():
        monkeypatch.setattr(gpu, name, value)
    conditions = dict.fromkeys(CONDITION_AXES, "")
    observed = {"O-0959": {"molecule_id": "M-0130", "conditions": {**conditions, "PE/EA": "2/1"}}}
    candidates = {"C-2926": {"molecule_id": "M-0130", "conditions": {**conditions, "PE/EA": "1/1"}}}
    stage = SimpleNamespace(context=SimpleNamespace(observed=observed, candidates=candidates))
    return raw, normalized, stage


def test_cached_projection_preserves_the_ordered_plan_and_checks_the_real_axis(reviewed_reply):
    raw, expected, stage = reviewed_reply
    original, normalized, paths = gpu._normalized(raw, stage)
    assert normalized == expected and len(paths) == 79
    assert [choice["candidate_id"] for choice in original["choices"]] == [c["candidate_id"] for c in normalized["choices"]]
    stage.context.observed["O-0959"]["conditions"]["flow"] = "2"
    with pytest.raises(PermissionError, match="recorded comparison"):
        gpu._normalized(raw, stage)


def test_cached_round_cannot_enter_the_mac_provider_dispatcher():
    dispatcher = mac.FutureDispatcher.__new__(mac.FutureDispatcher)
    with pytest.raises(PermissionError, match="local cached decision"):
        dispatcher._check_boundary({}, {"attempt": 0})
    assert gpu.IMPORT_BUDGET == {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
    assert gpu.BUDGET["maximum_physical_attempts"] == 192

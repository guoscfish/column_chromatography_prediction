from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prompt_targeted_repair as repair


def test_targeted_wire_only_adds_the_registered_system_guide():
    original = {
        "request_id": "archived",
        "binding": dict.fromkeys(repair.DATA_BINDINGS, "unchanged"),
        "payload": {"input": [{"role": "system", "content": "Original prompt" + repair.FORMAT_SUFFIX},
                              {"role": "user", "content": "Original state"}],
                    "model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"}, "store": False},
    }
    system = repair._updated_system(original)
    assert system == "Original prompt\n\n" + repair.repair_guide()
    profile = {"ordered_candidate_ids_sha256": "ordered"}
    job = deepcopy(original)
    job.update(logical_index=0, request_id="targeted")
    job["binding"].update(seed=887, round=4, prompt_sha256=digest(system), fixed_reference_repair=profile)
    job["payload"]["input"][0]["content"] = system + repair.FORMAT_SUFFIX
    validator = SimpleNamespace(validate=lambda value: (value, None, {"attempt": 0}))
    assert repair._validate(validator, job, original, profile, system) is job

    def wrong_message(candidate):
        candidate["payload"]["input"][1]["content"] = "Different data"

    def wrong_repair(candidate):
        candidate["binding"]["fixed_reference_repair"] = {"ordered_candidate_ids_sha256": "not ordered"}

    def old_answer(candidate):
        candidate["payload"]["input"].append({"role": "assistant", "content": "Prior answer"})

    def changed_model(candidate):
        candidate["payload"]["model"] = "different"

    for change in (wrong_message, wrong_repair, old_answer, changed_model):
        modified = deepcopy(job)
        change(modified)
        with pytest.raises(PermissionError, match="approved user data"):
            repair._validate(validator, modified, original, profile, system)
    retry = SimpleNamespace(validate=lambda value: (value, None, {"attempt": 1}))
    with pytest.raises(PermissionError, match="approved user data"):
        repair._validate(retry, job, original, profile, system)

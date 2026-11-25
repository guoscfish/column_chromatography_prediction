from copy import deepcopy
import json

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.targeted_reference_repair import (
    CHOICE_INDEX, normalize_original, project_reply,
)


def archived_plan():
    choices = []
    for index in range(32):
        row = {"candidate_id": f"C-{index:04}", "molecule_id": "M-0073", "condition_anchor": {"PE/EA": "5/1"},
               "reason": "original wording", "references": []}
        for slot in range(3 if index < 5 else 2):
            observed = f"O-{index * 3 + slot:04}"
            row["references"].append({"reference_id": observed, "reference_kind": "observed",
                                      "relationship": "condition_response", "observed_id": observed})
        choices.append(row)
    choices[CHOICE_INDEX]["candidate_id"] = "C-0249"
    choices[CHOICE_INDEX]["references"][0].update(reference_id="O-2702", observed_id="O-2702",
                                                      relationship="multifactor_contrast")
    return {"choices": choices, "hypothesis_updates": ["unchanged"], "batch_review": "unchanged"}


def patch_reply():
    return {"reference": {"reference_id": "O-2702", "reference_kind": "observed",
                          "relationship": "single_factor_contrast", "changed_axis": "flow"},
            "reason": "Compared with O-2702, only flow changes from 5 to 10 mL/min, isolating its effect on elution."}


def test_one_reference_patch_keeps_the_entire_selected_batch():
    original = archived_plan()
    projected, count = normalize_original(json.dumps(original))
    before = deepcopy(projected)
    result = project_reply(patch_reply(), plan=projected)
    assert count == 69
    assert projected == before
    assert [row["candidate_id"] for row in result["choices"]] == [row["candidate_id"] for row in original["choices"]]
    for index, choice in enumerate(result["choices"]):
        if index == CHOICE_INDEX:
            assert choice == {**before["choices"][index], "reason": patch_reply()["reason"],
                              "references": [patch_reply()["reference"], *before["choices"][index]["references"][1:]]}
        else:
            assert choice == before["choices"][index]
    assert result["hypothesis_updates"] == original["hypothesis_updates"]


def test_does_not_resolve_a_conflicting_alias_or_change_the_approved_comparison():
    original = archived_plan()
    original["choices"][0]["references"][0]["observed_id"] = "O-9999"
    with pytest.raises(ValueError, match="conflicts"):
        normalize_original(json.dumps(original))
    plan, _ = normalize_original(json.dumps(archived_plan()))
    wrong = patch_reply()
    wrong["reference"]["changed_axis"] = "purity"
    with pytest.raises(ResponseFormatError, match="scope"):
        project_reply(wrong, plan=plan)
    wrong = patch_reply()
    wrong["reason"] += " It also changes the solvent volume."
    with pytest.raises(ResponseFormatError, match="flow-only"):
        project_reply(wrong, plan=plan)

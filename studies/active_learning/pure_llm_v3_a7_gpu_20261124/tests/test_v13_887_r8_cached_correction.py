from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_887_r8_cached_correction as cached


def test_one_exact_negated_phrase_patch_retains_original_reason_and_32_order(monkeypatch):
    repair = cached.repair
    choices = [{"candidate_id": f"C-{index:04}", "reason": f"original {index}"} for index in range(31)]
    choices.append({"candidate_id": repair.CANDIDATE, "references": [repair.WRONG_REFERENCE], "reason": "original"})
    original = {"choices": choices, "hypothesis_updates": [], "batch_strategy": "unchanged"}
    patch = {"candidate_id": repair.CANDIDATE, "reference_index": 0, "reference": repair.REQUIRED_REFERENCE,
             "reason": ("C-1232: sample volume 100 µL, flow 10 mL/min; O-2392: sample volume 150 µL, flow 5 mL/min, "
                        "so the comparison combines sample-volume and flow changes and isolates neither effect.")}
    expected = deepcopy(original)
    expected["choices"][31].update(reason=patch["reason"], references=[patch["reference"]])
    monkeypatch.setattr(cached, "PATCH", digest(patch))
    monkeypatch.setattr(cached, "PROJECTED", digest(expected))
    monkeypatch.setattr(repair, "PLAN", digest(original))
    assert repair._checked_reason(patch["reason"]) is False
    assert cached.project(patch, original=original) == expected
    assert original["choices"][31]["reason"] == "original"
    for changed in ({**patch, "reference_index": 1}, {**patch, "reason": patch["reason"].replace("neither", "one")}):
        with pytest.raises(PermissionError, match="only this exact"):
            cached.project(changed, original=original)
        monkeypatch.setattr(cached, "PATCH", digest(changed))
        with pytest.raises(PermissionError, match="negated-phrase exception"):
            cached.project(changed, original=original)
        monkeypatch.setattr(cached, "PATCH", digest(patch))


def test_import_has_no_remote_transport_or_outbox(tmp_path):
    assert cached.BUDGET == {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
    cached._require_no_bridge(tmp_path)
    with pytest.raises(TransportStop, match="no provider transport"):
        cached._no_provider()
    queued = tmp_path / "bridge/outbox/outer-attempt.json"
    queued.parent.mkdir(parents=True)
    queued.write_text("{}")
    with pytest.raises(PermissionError, match="zero-POST"):
        cached._require_no_bridge(tmp_path)

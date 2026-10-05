import json

import pytest

from src.qgeognn_al.active_learning_v2.dual_expert_selector import (
    CHEMISTRY_SYSTEM_PROMPT, ML_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT,
    validate_expert_memo, validate_planner_selection, trajectory_memory,
)
from src.qgeognn_al.active_learning_v2.scientist_selector import ReadOnlyCatalog
from src.qgeognn_al.active_learning_v2.dual_expert_transport import run_agent, settings


def _row(i, observed=False):
    value = {"id": str(i), "smiles": "CCO" if i % 2 else "CCN",
             "conditions": {"PE/EA": 10 + i, "Density g/ml": 1.0, "V/ul": 10 + i,
                            "loading solvent": "DCM", "Volume of loading solvent/ul": 100},
             "pred_V1_ml": 10.0 + i, "pred_V2_ml": 20.0 + i,
             "pred_V1_q10_ml": 9.0 + i, "pred_V1_q50_ml": 10.0 + i, "pred_V1_q90_ml": 11.0 + i,
             "pred_V2_q10_ml": 19.0 + i, "pred_V2_q50_ml": 20.0 + i, "pred_V2_q90_ml": 21.0 + i,
             "cw_distance_percentile": i / 100}
    if observed:
        value.update({"V1_ml": 10.0, "V2_ml": 20.0, "record_kind": "initial_L333"})
    return value


def _catalog():
    return ReadOnlyCatalog(candidates=[_row(i) for i in range(40)], observed=[_row(100, True)], pending=[], salt="test")


def _planner_packet():
    return {"packet_sha256": "packet", "selection_count": 32, "round": 0}


def test_role_prompts_are_distinct_and_free_of_fixed_allocation():
    assert CHEMISTRY_SYSTEM_PROMPT != ML_SYSTEM_PROMPT != PLANNER_SYSTEM_PROMPT
    for prompt in (CHEMISTRY_SYSTEM_PROMPT, ML_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT):
        assert "validation/test" in prompt
        assert "quota" in prompt.lower()
    assert "16" not in PLANNER_SYSTEM_PROMPT
    assert "final batch" in CHEMISTRY_SYSTEM_PROMPT


def test_expert_memo_cannot_be_final_selection():
    with pytest.raises(ValueError, match="cannot contain final selection"):
        validate_expert_memo({"type": "memo", "role": "chemistry_scientist", "choices": []}, "chemistry_scientist")


def test_planner_can_reject_expert_recommendation_and_choose_current_pool():
    catalog = _catalog()
    catalog.initial_cards()
    first = catalog.query("search_candidates", {"limit": 24})
    catalog.query("search_candidates", {"offset": first["next_offset"], "limit": 24})
    choices = [{"id": str(i), "reason": "planner found a more informative contrast", "knowledge_basis": [],
                "related_hypothesis_id": None} for i in range(8, 40)]
    response = {"type": "selection", "packet_sha256": "packet",
                "expert_synthesis": {"chemistry_takeaways": "chemistry suggested ids 0-7",
                    "ml_takeaways": "ML suggested ids 0-7", "agreements": "limited", "disagreements": "planner differs",
                    "planner_resolution": "select ids 8-39 after checking the pool"},
                "batch_strategy": "contrastive coverage", "choices": choices, "hypotheses": [],
                "batch_rationale": "The planner independently selected a legal contrast set.",
                "expected_learning_value": "generalization", "feedback_interpretation": ""}
    assert len(validate_planner_selection(response, catalog, _planner_packet())["choices"]) == 32


def test_expert_and_planner_transcripts_are_independent(tmp_path):
    catalog = _catalog()
    catalog.initial_cards()
    calls = []

    def expert_call(messages, config):
        calls.append(messages[0]["content"])
        return json.dumps({"type": "memo", "role": "chemistry_scientist", "state_summary": "observed only"}), {"mock": True}

    memo = run_agent("chemistry_scientist", {"packet_sha256": "x"}, catalog,
                     tmp_path / "chemistry", settings(backend="responses"), call=expert_call)
    assert memo["role"] == "chemistry_scientist"
    with pytest.raises(ValueError):
        run_agent("ml_scientist", {"packet_sha256": "x"}, catalog, tmp_path / "ml",
                  settings(backend="responses"), call=lambda *_: (json.dumps(memo), {"mock": True}))
    assert (tmp_path / "chemistry" / "memo.json").exists()
    assert not (tmp_path / "ml" / "memo.json").exists()
    assert calls[0] == CHEMISTRY_SYSTEM_PROMPT


def test_memory_rejects_foreign_seed_and_has_no_future_label_field():
    with pytest.raises(ValueError, match="foreign"):
        trajectory_memory([{"seed": 6101, "method": "other", "records": []}], 157, "free_llm32_dual_expert_v1")
    memory = trajectory_memory([], 157, "free_llm32_dual_expert_v1")
    assert "future_labels" not in json.dumps(memory)

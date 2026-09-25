"""Synthetic boundary and recovery tests; these are not experiment results."""
import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.full_pool_selector import ReadOnlyCatalog, validate_selection
from src.qgeognn_al.active_learning_v2.full_pool_study import _feedback_records


def row(value, *, observed=False):
    result = {"id": str(value), "smiles": "CC" if value % 2 else "CCC",
              "conditions": {"PE/EA": "1:1", "Density g/ml": 1.0, "V/ul": 5,
                             "loading solvent": "PE", "Volume of loading solvent/ul": 10}}
    if observed:
        result.update({"V1_ml": 1.5, "V2_ml": 3.0})
    else:
        result.update({"pred_V1_ml": 1.0, "pred_V2_ml": 2.0})
    return result


def catalog():
    return ReadOnlyCatalog([row(i) for i in range(17, 161)], [row(0, observed=True)],
                           [row(i) for i in range(1, 17)], [str(i) for i in range(17, 145)])


def test_full_pool_search_and_outside_128_selection():
    cat = catalog()
    assert cat.overview()["eligible_count"] == 144
    assert cat.query("get_candidates", {"ids": ["145", "1", "0", "999"]})["matches"][0]["id"] == "145"
    assert cat.query("get_candidates", {"ids": ["1", "0", "999"]})["matches"] == []
    choices = [{"id": str(i), "reason": "condition contrast", "hypothesis_id": None} for i in range(145, 161)]
    assert len(validate_selection({"choices": choices, "hypotheses": []}, cat)["choices"]) == 16
    assert not set(choice["id"] for choice in choices) & set(cat.reference_ids)
    assert cat.query("search_candidates", {"same_molecule_as": "0", "descriptor": "LogP",
                                                "min_value": 100})["matches"] == []


def test_catalog_never_exposes_unobserved_truth_or_other_context():
    cat = catalog()
    found = cat.query("search_candidates", {"similar_to": "0", "limit": 5})["matches"]
    assert found and all("V1_ml" not in item and "V2_ml" not in item for item in found)
    assert cat.query("get_observed", {"ids": ["0", "17"]})["matches"][0]["V1_ml"] == 1.5
    with pytest.raises(ValueError, match="unknown catalog operation"):
        cat.query("read_file", {"path": "/some/repository/file"})
    other = ReadOnlyCatalog([row(18)], [dict(row(7, observed=True), V1_ml=99)], [row(1)], ["18"])
    assert cat.query("get_observed", {"ids": ["7"]})["matches"] == []
    assert other.query("get_observed", {"ids": ["7"]})["matches"][0]["V1_ml"] == 99


def test_strict_choice_and_hypothesis_boundaries():
    cat = catalog()
    good = [{"id": str(i), "reason": "informative"} for i in range(17, 33)]
    with pytest.raises(ValueError, match="distinct legal"):
        validate_selection({"choices": good[:-1] + [dict(good[0])]}, cat)
    with pytest.raises(ValueError, match="unobserved"):
        validate_selection({"choices": good, "hypotheses": [{"id": "h1", "content": "effect",
            "supporting_observed_ids": ["17"], "contradicting_observed_ids": [],
            "evidence_strength": "weak", "next_discriminating_question": "contrast?"}]}, cat)


def test_feedback_uses_frozen_premeasurement_prediction_and_matching_id():
    cards = [row(i) for i in range(1, 33)]
    for i, card in enumerate(cards):
        card["pred_V1_ml"] = float(i)
        card["pred_V2_ml"] = float(i + 2)
    truth = np.asarray([[i + .5, i + 3.0] for i in range(32)])
    batch = [str(i) for i in range(1, 33)]
    choices = [{"id": str(i), "reason": "hypothesis", "hypothesis_id": "h"} for i in range(17, 33)]
    feedback = _feedback_records(batch, cards, truth, choices)
    assert feedback[0]["source"] == "CW" and feedback[16]["source"] == "LLM"
    assert all(item["error_V1_ml"] == -.5 and item["error_V2_ml"] == -1.0 for item in feedback)
    assert feedback[16]["candidate_id"] == "17" and feedback[16]["reason"] == "hypothesis"


def test_frozen_batch_resume_does_not_call_selector_twice(tmp_path, monkeypatch):
    from src.qgeognn_al.active_learning_v2 import full_pool_study as study

    cat = catalog()
    cards = list(cat.pending.values()) + list(cat.candidates.values())
    class Context:
        seed = 157
        protocol_hash = "protocol"
        def ids(self, indices):
            return [str(i) for i in indices]

    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "banks", lambda *args: (np.zeros((161, 2)), np.zeros((161, 2))))
    monkeypatch.setattr(study, "shortlist", lambda *args: {"candidate_positions": list(range(17, 145))})
    monkeypatch.setattr(study, "_catalog", lambda *args: (catalog(), cards))
    monkeypatch.setattr(study, "configured", lambda: True)
    calls = []
    def selector(*args):
        calls.append(1)
        return {"packet_sha256": args[0]["packet_sha256"],
                "choices": [{"id": str(i), "reason": "condition contrast"} for i in range(145, 161)],
                "hypotheses": []}
    monkeypatch.setattr(study, "run_openai_selector", selector)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("synthetic")
    args = (Context(), study.METHODS[1], 0, np.array([0]), np.arange(1, 161),
            np.array([[1.5, 3.0]]), checkpoint, tmp_path, [])
    first = study._acquire(*args)
    second = study._acquire(*args)
    assert first[0] == second[0]
    assert len(calls) == 1
    assert first[0][16:] == [str(i) for i in range(145, 161)]
    frozen = study.read(first[-1] / "batch_freeze.json")
    assert not frozen["new_batch_labels_revealed"]
    assert len(set(frozen["batch_ids"])) == 32

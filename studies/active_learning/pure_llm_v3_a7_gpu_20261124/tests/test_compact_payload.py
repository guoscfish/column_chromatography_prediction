import json

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import (
    candidate_formats, compact_packet, dumps, expand_packet, grouped_table_text,
    packet_messages, read_grouped_table, round_view,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_precision import compare_values


@pytest.fixture
def packet():
    cond = ["2/1", "1.00", "001", "PE", "300", "10", 1 / 3, 0]
    candidate = ["C-0001", cond, 0.98, 1.0, .15147, .201554, .382736, .382736 - .15147]
    observed = ["O-0002", cond, "initial_L333", 1.123456789, -.5168417]
    return {
        "seed": 157, "round": 0, "layout": "B", "compression": "lossless", "selection_count": 32,
        "condition_columns": ["PE/EA", "density", "sample_volume", "loading_solvent",
                              "loading_solvent_volume", "flow", "EA_fraction", "reagent_index"],
        "condition_constants": {"column_specs": "fixture column"},
        "reagent_columns": ["CAS", "name", "purity_raw", "purity_fraction", "purity_kind"],
        "reagents": [["fixture-CAS", "fixture name", ">98%", None, "grade"]],
        "molecules": {"M-0001": {"smiles": "CC", "descriptors": {"MW_g_mol": 20.12345, "HBA": 0},
                                 "observed": [observed], "candidates": [candidate]}},
        "table_columns": {"candidates": ["id", "conditions", "max_similarity_to_observed", "loading_amount_mg",
                                          "pred_V1_q10_ml", "pred_V1_q50_ml", "pred_V1_q90_ml", "V1_quantile_width_ml"],
                          "observed": ["id", "conditions", "record_kind", "V1_ml", "premeasurement_error_V1_ml"]},
        "candidate_count": 1, "observed_count": 1,
        "memory": {"mode": "M2", "latest_feedback": [{"text": "fixture note", "observed_value": 3.14159265358979}]},
    }


@pytest.mark.parametrize("keep_widths", [True, False])
def test_lossless_round_trip_preserves_raw_types_and_memory(packet, keep_widths):
    original = dumps(packet, sort_keys=True)
    compact = compact_packet(packet, keep_widths=keep_widths)
    assert dumps(packet, sort_keys=True) == original
    assert dumps(expand_packet(compact), sort_keys=True) == original
    assert dumps(read_grouped_table(grouped_table_text(compact)), sort_keys=True) == dumps(compact, sort_keys=True)
    assert compact["ea_fraction_by_ratio"] == {"2/1": 1 / 3}
    condition_index = compact["table_columns"]["candidates"].index("conditions")
    conditions = compact["molecules"]["M-0001"]["candidates"][0][condition_index]
    assert "001" in conditions and 300 in conditions
    assert "EA_fraction" not in compact["condition_columns"]


def test_round_view_only_changes_designated_inputs(packet):
    updated = round_view(packet, 3)
    before = packet["molecules"]["M-0001"]
    after = updated["molecules"]["M-0001"]
    assert after["observed"][0][:4] == before["observed"][0][:4]
    assert after["observed"][0][4] == -.517
    assert after["candidates"][0][:4] == before["candidates"][0][:4]
    assert after["candidates"][0][4:] == [.151, .202, .383, .232]
    assert after["descriptors"] == {"MW_g_mol": 20.123, "HBA": 0}
    assert updated["memory"] == packet["memory"]
    messages = [{"role": "system", "content": "fixture scientific prompt"},
                {"role": "user", "content": dumps(packet)}]
    encoded = packet_messages(messages, compact_packet(updated), feature_precision=3)
    assert json.loads(encoded[1]["content"])["compression"] == "rounded_display"
    assert expand_packet(json.loads(encoded[1]["content"])) == updated
    assert candidate_formats(messages)["existing_grouped_json"] == messages


def test_rank_audit_counts_ties_and_reversals():
    outcome = compare_values([0, 1, 2, 3], [1, 0, 2, 2])
    assert outcome["max_abs"] == 1
    assert outcome["new_tied_pairs"] == 1
    assert outcome["reversed_pairs"] == 1


def test_crossing_guard_preserves_original_triplet_when_rounding_would_hide_it(packet):
    group = packet["molecules"]["M-0001"]
    group["candidates"][0][4:] = [.15149, .15140, .382736, .382736 - .15149]
    plain = round_view(packet, 3)["molecules"]["M-0001"]["candidates"][0]
    assert plain[4:6] == [.151, .151]
    guarded = round_view(packet, 3, preserve_crossings=True)
    assert guarded["molecules"]["M-0001"]["candidates"][0] == group["candidates"][0]

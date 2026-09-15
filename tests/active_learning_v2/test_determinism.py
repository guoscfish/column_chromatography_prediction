from __future__ import annotations

import pandas as pd
import torch

from scripts.studies.run_qgeognn_v2_4g_row_lcmd_pilot import (
    decision_gate,
    initialization_seed,
    sketch_seed,
    training_seed,
)
from src.qgeognn_al.active_learning_v2.gradient_features import state_dict_hash
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.training.predictor import seed_everything


def _normalization() -> ConditionNormalization:
    return ConditionNormalization(0, 1, 0, 1, "4g", "source_train", 10, "a" * 64, "b" * 64)


def test_all_arms_get_identical_initial_parameter_hash_within_seed() -> None:
    seed = initialization_seed(73)
    hashes = []
    for _arm in range(7):
        seed_everything(seed)
        hashes.append(state_dict_hash(build_predictor(_normalization())))
    assert len(set(hashes)) == 1


def test_seed_derivations_are_fixed_and_distinct() -> None:
    assert initialization_seed(73) == 2_000_076
    assert training_seed(73) == 3_000_090
    assert sketch_seed(73) == 4_000_110
    assert len({initialization_seed(73), training_seed(73), sketch_seed(73)}) == 3


def test_strong_positive_decision_gate_uses_frozen_thresholds() -> None:
    summaries = pd.DataFrame(
        {
            "outer_seed": [73, 311, 1297, 4093, 8191],
            "LCMD_gain": [0.20, 0.19, 0.18, 0.17, 0.01],
            "Random_gain_median": [0.10] * 5,
            "LCMD_minus_Random_mean": [0.10, 0.09, 0.08, 0.07, -0.09],
            "LCMD_minus_Random_median": [0.10, 0.09, 0.08, 0.07, -0.09],
        }
    )
    arms = []
    for seed in summaries.outer_seed:
        arms.append(
            {
                "outer_seed": seed,
                "arm": "lcmd",
                "combined_normalized_RMSE": 0.80,
                "V1_RMSE": 8.0,
                "V2_RMSE": 16.0,
            }
        )
        for control in range(5):
            arms.append(
                {
                    "outer_seed": seed,
                    "arm": f"random_control_{control}",
                    "combined_normalized_RMSE": 0.90,
                    "V1_RMSE": 9.0,
                    "V2_RMSE": 18.0,
                }
            )
    decision = decision_gate(summaries, pd.DataFrame(arms))
    assert decision["directional_wins"] == 4
    assert decision["decision"] == "STRONG_POSITIVE"
    assert decision["full_learning_curve_recommended"] is True

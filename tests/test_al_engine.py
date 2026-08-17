from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.al_engine import (
    ActiveLearningState,
    SourceFreeTrainConfig,
    TrainConfig,
    canonical_json_hash,
    initialize_round_state,
    load_round_state,
    random_query,
    save_round_state,
)


class TrainConfigTests(unittest.TestCase):
    def test_frozen_default_is_valid_and_hash_is_stable(self) -> None:
        first = TrainConfig()
        second = TrainConfig()
        first.validate_frozen_predictor()
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(first.config_hash, canonical_json_hash(json.loads(json.dumps(first.__dict__))))

    def test_frozen_setting_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen Gate 0"):
            TrainConfig(learning_rate=1e-3).validate_frozen_predictor()

    def test_source_free_contract_is_explicit_and_distinct(self) -> None:
        config = SourceFreeTrainConfig()
        config.validate_frozen_predictor()
        self.assertNotEqual(config.config_hash, TrainConfig().config_hash)
        self.assertEqual(config.transfer_mode, "full_source_free")
        self.assertEqual(config.scaler_policy, "fixed_l0_train")

    def test_source_free_contract_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "E2 contract"):
            SourceFreeTrainConfig(initialization_policy="checkpoint").validate_frozen_predictor()


class ActiveLearningStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = initialize_round_state(
            labeled_ids=["l0", "l1"],
            pool_ids=[f"p{i}" for i in range(20)],
            checkpoint="anchor.pt",
            seed=525,
            split_hash="split-hash",
            config_hash="config-hash",
        )

    def test_resume_matches_continuous_random_queries(self) -> None:
        continuous_1 = random_query(self.initial, 4, checkpoint="round-1.pt")
        continuous_2 = random_query(continuous_1, 4, checkpoint="round-2.pt")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_round_state(path, continuous_1)
            resumed_1 = load_round_state(path, "split-hash", "config-hash")
            resumed_2 = random_query(resumed_1, 4, checkpoint="round-2.pt")
        self.assertEqual(continuous_2, resumed_2)
        self.assertEqual(len(continuous_2.labeled_ids), len(set(continuous_2.labeled_ids)))
        self.assertFalse(set(continuous_2.labeled_ids) & set(continuous_2.pool_ids))

    def test_hash_mismatch_refuses_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_round_state(path, self.initial)
            with self.assertRaisesRegex(ValueError, "split hash changed"):
                load_round_state(path, expected_split_hash="different")

    def test_duplicate_or_requery_state_is_invalid(self) -> None:
        invalid = ActiveLearningState(
            version=1,
            round=1,
            labeled_ids=["x", "x"],
            pool_ids=["y"],
            selected_ids=["x"],
            checkpoint="x.pt",
            seed=1,
            rng_state=self.initial.rng_state,
            split_hash="s",
            config_hash="c",
        )
        with self.assertRaisesRegex(ValueError, "internally unique"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()

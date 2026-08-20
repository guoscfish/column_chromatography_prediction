from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.al_engine import SourceFreeTrainConfig
from scripts.run_e2_4g_active_learning import (
    OUTER_SEEDS,
    audit_compound_partitions,
    collect_outputs,
    common_reference_batch_diversity,
    batch_composition,
    partition_path_for_split,
    validate_formal_protocol,
    write_static_config,
)


class E2SplitModeTests(unittest.TestCase):
    def test_partition_routing(self) -> None:
        for seed in OUTER_SEEDS:
            self.assertEqual(partition_path_for_split("row", seed).name, f"e2_4g_row_seed_{seed}.csv")
            self.assertEqual(
                partition_path_for_split("compound", seed).name,
                f"e2_4g_compound_seed_{seed}.csv",
            )

    def test_compound_static_config_has_correct_stage_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_static_config(output, SourceFreeTrainConfig(), 8, 25, "compound")
            payload = json.loads((output / "config.json").read_text())
            self.assertEqual(payload["stage"], "E2_4g_active_learning_compound_pilot")
            self.assertEqual(payload["split_mode"], "compound")
            self.assertEqual(set(payload["partitions"]), {"42", "525", "1101"})
            for seed, item in payload["partitions"].items():
                self.assertIn(f"compound_seed_{seed}.csv", item["path"])
                self.assertEqual(len(item["sha256"]), 64)

    def test_compound_collection_excludes_row_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, seed in (("row", 42), ("compound", 525)):
                round_dir = root / f"{mode}_seed_{seed}" / "random" / "round_0"
                round_dir.mkdir(parents=True)
                (round_dir / "round_metrics.json").write_text(
                    json.dumps({"outer_seed": seed, "strategy": "random", "round": 0})
                )
                (round_dir / "queried_batch_diagnostics.json").write_text(
                    json.dumps({"seed": seed, "strategy": "random", "round": 0})
                )
                pd.DataFrame([{"outer_seed": seed, "strategy": "random", "round": 0, "member_index": 0}]).to_csv(
                    round_dir / "convergence.csv", index=False
                )
            metrics, queried, convergence = collect_outputs(root, "compound")
            self.assertEqual(metrics["outer_seed"].tolist(), [525])
            self.assertEqual(queried["seed"].tolist(), [525])
            self.assertEqual(convergence["outer_seed"].tolist(), [525])

    def test_all_compound_partitions_have_no_test_compound_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = audit_compound_partitions(Path(directory), SourceFreeTrainConfig())
            overlap = audit.filter(regex="_overlap$")
            self.assertFalse(overlap.to_numpy().any())

    def test_formal_protocol_guard(self) -> None:
        validate_formal_protocol(500, 100, 8, 25)
        for values in ((499, 100, 8, 25), (500, 99, 8, 25), (500, 100, 7, 25), (500, 100, 8, 24)):
            with self.assertRaisesRegex(ValueError, "Formal E2 protocol"):
                validate_formal_protocol(*values)

    def test_batch_composition_records_compound_concentration_and_conditions(self) -> None:
        frame = pd.DataFrame(
            {
                "canonical_smiles": ["A", "A", "B"],
                "loading solvent": ["DCM", "DCM", "MeOH"],
                "PE/EA": ["10/1", "10/1", "5/1"],
                "Density g/ml": [1.0, 1.0, 0.9],
                "V/ul": [10, 10, 20],
                "Volume of loading solvent/ul": [100, 100, 200],
                "Flow mL/min": [20, 20, 25],
            }
        )
        result = batch_composition(frame)
        self.assertEqual(result["selected_unique_compounds"], 2)
        self.assertEqual(result["max_samples_per_compound"], 2)
        self.assertAlmostEqual(result["compound_hhi"], 5 / 9)
        self.assertEqual(result["selected_unique_condition_keys"], 2)

    def test_row_common_reference_audit_uses_existing_frozen_outputs(self) -> None:
        output = Path("experiments/e2_4g_active_learning")
        result, summary = common_reference_batch_diversity(output, "row")
        self.assertEqual(len(result), len(OUTER_SEEDS) * 4 * 8)
        self.assertEqual(set(result["outer_seed"]), set(OUTER_SEEDS))
        self.assertIn("paired_effect", set(summary["record_type"]))


if __name__ == "__main__":
    unittest.main()

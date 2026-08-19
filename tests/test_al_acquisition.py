from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.al_acquisition import (
    farthest_first_select,
    hybrid_select,
    signal_agreement_rows,
    top_score_select,
)


class AcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = ["d", "c", "b", "a"]
        self.candidates = np.asarray([[1.0], [2.0], [8.0], [9.0]])
        self.reference = np.asarray([[0.0]])

    def test_farthest_first_updates_distance_and_is_deterministic(self) -> None:
        first = farthest_first_select(self.ids, self.candidates, self.reference, 2)
        second = farthest_first_select(self.ids, self.candidates, self.reference, 2)
        self.assertEqual(first, ["a", "c"])
        self.assertEqual(first, second)

    def test_score_ties_use_sample_id(self) -> None:
        selected = top_score_select(self.ids, np.ones(4), 2)
        self.assertEqual(selected, ["a", "b"])

    def test_hybrid_fraction_is_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            hybrid_select(
                self.ids,
                np.arange(4),
                self.candidates,
                self.reference,
                1,
                prefilter_fraction=0.5,
            )

    def test_signal_agreement_has_all_pairs_and_overlaps(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(20)],
                "quantile_width": np.arange(20),
                "ensemble_score": np.arange(20),
                "latent_distance": np.arange(20)[::-1],
            }
        )
        rows = signal_agreement_rows(frame, {"stage": "test"})
        self.assertEqual(len(rows), 3)
        qe = next(row for row in rows if row["signal_B"] == "ensemble")
        self.assertEqual(qe["spearman"], 1.0)
        self.assertEqual(qe["top10_overlap"], 1.0)


if __name__ == "__main__":
    unittest.main()

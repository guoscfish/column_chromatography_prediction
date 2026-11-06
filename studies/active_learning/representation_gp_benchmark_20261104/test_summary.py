"""Synthetic receipts only: aggregation, incomplete states, and ID alignment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_results import (acquisition_summary, load_completed, paired_tables, summarize)


def metric(seed, budget, rep, kernel, init, endpoint, value):
    return dict(seed=seed, budget=budget, representation=rep, kernel=kernel, initialization=init,
                endpoint=endpoint, normalized_rmse_L0=value, rmse=value*2, mae=value,
                nlpd=value+1, coverage_90=.8, coverage_95=.9, mean_interval_width_90=4.,
                mean_interval_width_95=5., standardized_error_rms=1.3, fit_seconds=.1, inference_seconds=.01)


def acquisition(seed=1, budget=333, rep="gnn", ids=None, values=None):
    ids = ids or ["A", "B", "C", "D"]
    values = values or [1., 2., 3., 4.]
    selected = np.argsort(-np.array(values))[:2].tolist()
    method = dict(batch_information=1., mean_absolute_batch_latent_correlation=.2,
                  selected_candidate_positions=selected, selected_ids=[ids[i] for i in selected])
    return dict(seed=seed, budget=budget, representation=rep, kernel="matern52", initialization=None,
                endpoint="V1", candidate_positions=list(range(4)), candidate_ids=ids,
                latent_variance=values, initial_ivr_scores=values, initial_information_scores=values,
                methods={name: dict(method) for name in ("maxvar", "conditional_ivr", "batch_information_gain")})


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_gp_summary_")
        self.study = Path(self.temp.name)
        self.write(self.study / "config.json", dict(seeds=[1, 2], budgets=[333, 525]))
        for seed, budget, value in ((1, 333, 1.), (1, 525, 4.), (2, 333, 2.)):
            directory = self.study / "results" / f"seed_{seed}_L{budget}"
            ms, fits, contrasts = [], [], []
            for rep in ("gnn", "fusion"):
                for kernel, inits in (("matern52", [None]), ("projected", [17, 29, 43])):
                    for init in inits:
                        for endpoint in ("V1", "V2"):
                            projection_delta = (-.2 if seed == 1 else .2) if kernel == "projected" else 0
                            variation = {None: 0, 17: -.1, 29: 0, 43: .1}[init]
                            v = value + projection_delta + variation + (.1 if rep == "fusion" else 0)
                            row = metric(seed, budget, rep, kernel, init, endpoint, v)
                            ms.append(row)
                            fit = {k: row[k] for k in ("seed", "budget", "representation", "kernel", "initialization", "endpoint")}
                            # Reproduce the old audit.kernel overwrite to test repair.
                            fit.update(kernel="matern52", trainable_projection=kernel == "projected",
                                       optimizer_success=kernel != "projected", optimizer={"boundary_hit": False},
                                       numerical_jitter=0., noise_variance=.3, signal_variance=2., lengthscale=1.,
                                       negative_log_marginal_likelihood=10., training_seconds=.1)
                            fits.append(fit)
                            contrasts.append({k: row[k] for k in ("seed", "budget", "representation", "kernel", "initialization", "endpoint")})
                            contrasts[-1].update(axis="sample_volume", id_a="A", id_b="B", distance_train_rms=.1,
                                true_delta_ml=1., gp_contrast_absolute_error_ml=.5, gnn_contrast_absolute_error_ml=.7)
                            if kernel == "projected":contrasts[-1]["learned_distance_train_rms"] = .2
            self.write(directory / "metrics.json", ms)
            self.write(directory / "fits.json", fits)
            self.write(directory / "contrasts.json", contrasts)
            self.write(directory / "acquisitions.json", [acquisition(seed,budget,"gnn"), acquisition(seed,budget,"fusion")])
            self.write(directory / "robustness.json", [dict(seed=seed,budget=budget,nearest_10_overlap_mean=.7,
                template_distance_train_rms_mean=.2,V1_variance_rank_spearman=.8,V2_variance_rank_spearman=.9)])
            self.write(directory / "complete.json", dict(seed=seed,budget=budget,endpoint_fits=len(fits),model_variants=8,
                test_truth_access_count=0,new_gnn_fits=0,al_trajectories_started=0))
        # A partial state is intentionally invalid JSON and must never be read.
        incomplete = self.study / "results/seed_2_L525"
        incomplete.mkdir()
        (incomplete / "metrics.json").write_text("do not open this unfinished output")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_seed_hierarchy_prevents_state_pseudoreplication(self):
        result = summarize(self.study, plots=False)
        row = next(r for r in result["overall_metrics"] if r["representation"] == "gnn" and r["kernel"] == "matern52")
        self.assertAlmostEqual(row["normalized_rmse_L0_mean"], 2.25)
        self.assertEqual(row["seed_count"], 2)
        self.assertEqual(result["coverage"]["completed_states"], 3)
        self.assertEqual(result["coverage"]["expected_states"], 4)
        self.assertTrue(result["partial_results"])
        self.assertEqual(result["coverage"]["incomplete_directories_excluded"], ["seed_2_L525"])
        self.assertTrue((self.study / "summary/metrics.csv").exists())
        self.assertEqual(result["test_truth_reads"], 0)

    def test_initializations_are_averaged_before_seed_comparison(self):
        result = summarize(self.study, plots=False)
        row = next(r for r in result["paired_comparisons"] if r["comparison"] == "gnn: projection - frozen")
        self.assertAlmostEqual(row["normalized_rmse_L0_delta_mean"], 0., places=10)
        self.assertEqual(row["primary_seed_win_fraction"], .5)
        self.assertEqual(row["paired_seed_budget_count"], 3)
        self.assertEqual(row["seed_count"], 2)
        opt = next(r for r in result["optimizer"] if r["representation"] == "gnn" and r["kernel"] == "projected")
        self.assertEqual(opt["nonconverged_fit_count"], 18)
        init = next(r for r in result["projection_initialization"] if r["representation"] == "gnn")
        self.assertAlmostEqual(init["normalized_rmse_L0_initialization_sd_mean"], .1)

    def test_fusion_projection_pairs_match_initialization(self):
        rows = []
        for rep, entries in (("gnn", [(17,100.),(29,0.)]), ("fusion", [(17,0.),(43,1000.)])):
            for init, value in entries:
                for endpoint in ("V1", "V2"):
                    rows.append(metric(1,333,rep,"projected",init,endpoint,value))
        _, _, paired = paired_tables(rows)
        row = next(r for r in paired if r["comparison"] == "Fusion projected - GNN projected")
        self.assertEqual(row["normalized_rmse_L0_delta_mean"], -100.)

    def test_acquisition_scores_align_by_candidate_id(self):
        a = acquisition(rep="gnn")
        b = acquisition(rep="fusion",ids=["D","C","B","A"],values=[4.,3.,2.,1.])
        _, comparisons, result = acquisition_summary([a,b])
        self.assertEqual(len(comparisons), 3)
        np.testing.assert_allclose(comparisons.initial_score_rank_spearman, 1.)
        np.testing.assert_allclose(comparisons.selected_batch_overlap, 1.)
        self.assertEqual(result["excluded_comparisons"], [])

    def test_candidate_universe_mismatch_fails(self):
        with self.assertRaises(ValueError):
            acquisition_summary([acquisition(rep="gnn"),acquisition(rep="fusion",ids=["A","B","C","X"])])

    def test_known_fit_metadata_merge_bug_is_repaired_without_writing_inputs(self):
        path = self.study / "results/seed_1_L333/fits.json"
        rows = json.loads(path.read_text())
        for row in rows:
            row["seed"] = row["initialization"] if row["initialization"] is not None else 17
        self.write(path, rows)
        original = path.read_bytes()
        result = summarize(self.study, plots=False)
        self.assertEqual(path.read_bytes(), original)
        repairs = result["coverage"]["receipt_identity_repairs"]
        self.assertTrue(any(r["state"] == "seed_1_L333" and "seed:" in r["fields"] for r in repairs))
        rows[0]["seed"] = 999
        self.write(path, rows)
        with self.assertRaises(ValueError):
            load_completed(self.study)

    def test_nonfit_state_metadata_is_never_repaired(self):
        path = self.study / "results/seed_1_L333/metrics.json"
        rows = json.loads(path.read_text())
        rows[0]["seed"] = 17
        self.write(path, rows)
        with self.assertRaises(ValueError):
            load_completed(self.study)

    def test_scientific_geometry_normalizes_with_result_only_scale(self):
        result = summarize(self.study, plots=False)
        row = next(r for r in result["contrast"] if r["representation"] == "gnn" and r["kernel"] == "matern52")
        self.assertAlmostEqual(row["mean_absolute_response_delta_L0_mean"], .5)
        self.assertAlmostEqual(row["mean_contrast_absolute_error_L0_mean"], .25)
        self.assertEqual(row["compressed_pair_fraction_mean"], 0.)
        self.assertEqual(result["scientific_geometry"]["scatter_budget"], 525)
        self.assertTrue((self.study / "summary/geometry_scatter_points.csv").exists())

    def test_complete_marker_requires_all_artifacts(self):
        (self.study / "results/seed_1_L333/fits.json").unlink()
        with self.assertRaises(ValueError):
            load_completed(self.study)

    def test_summary_does_not_overwrite_model_results(self):
        with self.assertRaises(ValueError):
            summarize(self.study, self.study / "results/summary", plots=False)

    def test_empty_partial_study_is_supported(self):
        empty = self.study / "empty"
        empty.mkdir()
        result = summarize(empty, plots=False)
        self.assertEqual(result["coverage"]["completed_states"], 0)
        self.assertEqual(result["overall_metrics"], [])

    def test_standalone_english_plots_render(self):
        result = summarize(self.study, self.study / "plot_output", plots=True)
        self.assertGreaterEqual(len(result["figures"]), 8)
        for name in result["figures"]:
            self.assertGreater((self.study / "plot_output" / name).stat().st_size, 1000)
        layout = json.loads((self.study / "plot_output/figure_layout_qa.json").read_text())
        self.assertGreaterEqual(len(layout), 4)
        self.assertTrue(all(r["footer_legend_overlaps"] == 0 and r["footer_axis_label_overlaps"] == 0 for r in layout))
        # JSON export must contain neither nonstandard NaN nor Infinity literals.
        text = (self.study / "plot_output/summary.json").read_text()
        self.assertNotIn(": NaN", text)
        self.assertNotIn(": Infinity", text)


if __name__ == "__main__":
    unittest.main()

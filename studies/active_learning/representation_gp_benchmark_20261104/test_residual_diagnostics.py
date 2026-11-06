"""Synthetic-only residual mathematics, binning and access-scope tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from residual_diagnostics import endpoint_bins, load_state_bins, run


class ResidualDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="synthetic_residual_diagnostic_")
        self.study=Path(self.temp.name)
        self.write(self.study/"config.json",dict(seeds=[1,2],budgets=[333,525]))
        for seed,budget,value in ((1,333,1.),(1,525,4.),(2,333,2.)):
            directory=self.study/"results"/f"seed_{seed}_L{budget}"
            metrics,predictions=[],[]
            for kernel,inits in (("matern52",[None]),("projected",[17,29,43])):
                for init in inits:
                    delta={None:0.,17:-.1,29:0.,43:.1}[init]
                    v=value+delta
                    for endpoint in ("V1","V2"):
                        base=dict(seed=seed,budget=budget,representation="gnn",kernel=kernel,initialization=init,endpoint=endpoint)
                        metrics.append({**base,"rmse":2*v,"normalized_rmse_L0":v})
                        for i in range(10):
                            predictions.append({**base,"id":f"row{i:02d}","truth_ml":2*v,"mean_ml":0.,"observation_variance_ml2":4.})
            self.write(directory/"complete.json",dict(seed=seed,budget=budget,endpoint_fits=len(metrics),evaluation_rows=10,
                        test_truth_access_count=0,new_gnn_fits=0,al_trajectories_started=0))
            self.write(directory/"metrics.json",metrics)
            self.write(directory/"predictions.json",predictions)
        incomplete=self.study/"results/seed_2_L525"
        incomplete.mkdir()
        (incomplete/"predictions.json").write_text("must not read unfinished state")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_z_math_and_observation_noise_are_used(self):
        rows=[dict(id=str(i),truth_ml=float(i+1),mean_ml=0.,observation_variance_ml2=float((i+1)**2)) for i in range(10)]
        bins=endpoint_bins(rows,2.)
        self.assertEqual([b["n_observations"] for b in bins],[2]*5)
        for b in bins:
            self.assertAlmostEqual(b["z_rms"],1.)
            self.assertEqual(b["coverage_90"],1.)
            self.assertEqual(b["coverage_95"],1.)
        self.assertAlmostEqual(bins[0]["mean_sigma_L0"],.75)
        self.assertAlmostEqual(bins[0]["normalized_rmse"],np.sqrt(2.5)/2)

    def test_bin_memberships_depend_only_on_predicted_variance_and_id(self):
        rows=[dict(id=str(i),truth_ml=float(i),mean_ml=0.,observation_variance_ml2=1.) for i in range(10)]
        before=endpoint_bins(rows,1.)
        changed=[{**r,"truth_ml":99999.-r["truth_ml"]} for r in reversed(rows)]
        after=endpoint_bins(changed,1.)
        self.assertEqual([b["member_ids_sha256"] for b in before],[b["member_ids_sha256"] for b in after])
        self.assertNotEqual(before[0]["z_rms"],after[0]["z_rms"])

    def test_hierarchical_aggregation_and_partial_state_exclusion(self):
        result=run(self.study,plots=False)
        fixed=[r for r in result["bin_summary"] if r["kernel"]=="matern52"]
        self.assertEqual(len(fixed),5)
        for row in fixed:
            self.assertAlmostEqual(row["z_rms_mean"],2.25)
            self.assertAlmostEqual(row["mean_sigma_L0_mean"],1.)
            self.assertEqual(row["seed_count"],2)
        projected=[r for r in result["bin_summary"] if r["kernel"]=="projected"]
        np.testing.assert_allclose([r["z_rms_mean"] for r in projected],2.25)
        self.assertEqual(result["coverage"]["completed_states"],3)
        self.assertTrue(result["partial_results"])
        self.assertEqual(result["coverage"]["incomplete_directories_excluded"],["seed_2_L525"])
        self.assertEqual(result["original_data_reads"],0)
        self.assertEqual(result["test_truth_reads"],0)
        self.assertTrue(all("predictions.json" in r["path"] or "metrics.json" in r["path"] or "complete.json" in r["path"] for r in result["source_files"]))

    def test_duplicate_ids_and_nonpositive_variance_fail(self):
        rows=[dict(id="A",truth_ml=1.,mean_ml=0.,observation_variance_ml2=1.)]*2
        with self.assertRaises(ValueError):endpoint_bins(rows,1.)
        with self.assertRaises(ValueError):endpoint_bins([{**rows[0],"observation_variance_ml2":0.}],1.)
        with self.assertRaises(ValueError):endpoint_bins(rows,1.,bin_count=4)

    def test_state_identity_mismatch_is_rejected(self):
        path=self.study/"results/seed_1_L333/predictions.json"
        rows=json.loads(path.read_text());rows[0]["seed"]=999
        self.write(path,rows)
        with self.assertRaises(ValueError):load_state_bins(self.study)

    def test_rerun_deduplicates_and_does_not_modify_inputs_or_main_summary(self):
        path=self.study/"results/seed_1_L333/predictions.json"
        before=path.read_bytes()
        (self.study/"summary").mkdir()
        sentinel=self.study/"summary/summary.json";sentinel.write_text("do not change main summary")
        first=run(self.study,plots=False);second=run(self.study,plots=False)
        self.assertEqual(first,second)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(sentinel.read_text(),"do not change main summary")

    def test_standard_plot_and_footer_layout(self):
        result=run(self.study,plots=True)
        self.assertEqual(len(result["figures"]),2)
        self.assertEqual(result["figure_layout_qa"]["footer_legend_overlaps"],0)
        for name in result["figures"]:
            self.assertGreater((self.study/"summary"/name).stat().st_size,1000)


if __name__=="__main__":unittest.main()

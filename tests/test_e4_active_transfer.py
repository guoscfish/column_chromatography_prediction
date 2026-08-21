import json, unittest
from pathlib import Path
import numpy as np

from scripts.al_engine import initialize_round_state, random_query
from scripts.al_acquisition import fit_standardizer, transform_standardized
from scripts.run_e4_active_transfer import acquire, ensemble_scores, partition_context, primary_quantile_width, representation_from_primary, validate_dry_run

ROOT=Path(__file__).resolve().parents[1]

class E4Tests(unittest.TestCase):
    def test_roles_disjoint_and_cover_574(self):
        for protocol in ("A","B"):
            for seed in (42,525,1101):
                frame,roles=partition_context(protocol,seed)
                self.assertEqual(len(frame),574)
                sets=list(map(set,roles.values()))
                self.assertFalse(any(sets[i]&sets[j] for i in range(4) for j in range(i+1,4)))
                self.assertEqual(len(set().union(*sets)),574)

    def test_source_compatibility_and_conversion(self):
        audit=json.loads((ROOT/"experiments/e4_active_transfer_preregistration/source_compatibility_audit.json").read_text())
        self.assertTrue(audit["core_pass"])
        self.assertEqual(audit["legacy_to_monotonic_crossing_rate"],0)
        self.assertTrue(audit["checkpoint_hashes_distinct"])
        self.assertFalse(any(audit["member_q50_pairwise_identical"].values()))

    def test_acquisition_contracts(self):
        ids=[f"s{i:03d}" for i in range(40)]; rng=np.random.default_rng(1)
        scores={"ensemble":np.arange(40.),"quantile_width":np.arange(40.)[::-1],"pool_rep":rng.normal(size=(40,4)),"train_rep":rng.normal(size=(8,4))}
        for strategy in ("random","coverage","ensemble","hybrid","quantile_width"):
            chosen,extra=acquire(strategy,ids,scores,10,7)
            self.assertEqual(len(chosen),10); self.assertEqual(len(set(chosen)),10); self.assertLessEqual(set(chosen),set(ids))
            if strategy=="hybrid": self.assertLessEqual(set(chosen),set(extra["ensemble_top25_candidates"]))
        self.assertEqual(acquire("quantile_width",ids,scores,10,7)[0],ids[:10])

    def test_primary_member_coordinate_system(self):
        h42_train=np.array([[0.,0.],[2.,2.]]); h42_pool=np.array([[1.,3.],[4.,5.]])
        h525_pool=h42_pool+100.; h1101_pool=h42_pool-50.
        conditions_train=np.array([[0.],[2.]]); conditions_pool=np.array([[1.],[3.]])
        train,pool,audit=representation_from_primary(h42_train,h42_pool,conditions_train,conditions_pool)
        hmean,hscale=fit_standardizer(h42_train)
        expected=transform_standardized(h42_pool,hmean,hscale)
        averaged=transform_standardized(np.mean([h42_pool,h525_pool,h1101_pool],axis=0),hmean,hscale)
        np.testing.assert_allclose(pool[:,:2],expected)
        self.assertFalse(np.allclose(pool[:,:2],averaged))
        self.assertEqual(audit["primary_member_seed"],42); self.assertTrue(audit["no_cross_member_embedding_average"])

    def test_standardized_ensemble_definition(self):
        q=np.array([[[2.,8.],[1.,2.]],[[4.,4.],[3.,6.]],[[8.,12.],[5.,10.]]])
        scales={"V1":2.,"V2":4.}
        expected=np.var(q[:,:,0]/2.,axis=0,ddof=1)+np.var(q[:,:,1]/4.,axis=0,ddof=1)
        np.testing.assert_allclose(ensemble_scores(q,scales),expected)

    def test_primary_normalized_quantile_width(self):
        import pandas as pd
        table=pd.DataFrame({"V1_q10":[1.,2.],"V1_q90":[5.,8.],"V2_q10":[2.,4.],"V2_q90":[10.,12.]})
        expected=.5*((table.V1_q90-table.V1_q10)/2.+(table.V2_q90-table.V2_q10)/4.)
        np.testing.assert_allclose(primary_quantile_width(table,{"V1":2.,"V2":4.}),expected)

    def test_non_hybrid_metadata_is_null(self):
        for strategy in ("random","coverage","ensemble","quantile_width"):
            row=validate_dry_run(strategy,[str(x) for x in range(10)],{},set(map(str,range(20))),set())
            self.assertIsNone(row["hybrid_within_ensemble_top25"])

    def test_resume_equality(self):
        state=initialize_round_state(["l0"],[f"u{i}" for i in range(30)],"source",42,"split","config")
        one=random_query(state,10); two=random_query(one,10)
        payload=json.loads(json.dumps(one.__dict__)); restarted=type(one)(**payload)
        resumed=random_query(restarted,10)
        self.assertEqual(two.__dict__,resumed.__dict__)

    def test_smoke_freeze_reinit_and_no_test_leakage(self):
        out=ROOT/"experiments/e4_protocol_a_engineering_smoke"
        if not (out/"smoke_decision.json").exists(): self.skipTest("smoke still running")
        freeze=json.loads((out/"parameter_freeze_audit.json").read_text())
        self.assertTrue(all(x["frozen_unchanged"] and x["trainable_changed"] for x in freeze))
        fits=json.loads((out/"fit_results.json").read_text())
        source_hash={x["member_seed"]:x["init_source_sha256"] for x in fits if x["round"]==0}
        self.assertTrue(all(x["init_source_sha256"]==source_hash[x["member_seed"]] for x in fits))
        config=json.loads((out/"config.json").read_text())
        self.assertIn("never scaler/early-stop/checkpoint/score/acquisition/pass-fail",config["test_usage"])
        self.assertTrue(json.loads((out/"resume_audit.json").read_text())["pass"])

if __name__ == "__main__": unittest.main()

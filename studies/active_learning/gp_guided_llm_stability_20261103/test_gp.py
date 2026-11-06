"""Mathematical and access-boundary tests; synthetic data only."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from gp import LinearGP, batch_information, fixed_kernel_signals, probability_metrics, stable_cholesky
from offline import Reader, evaluate_seed, upstream_stable_hash


class GPTests(unittest.TestCase):
    def test_posterior_matches_direct_conditioning(self):
        rng=np.random.default_rng(26)
        x=rng.normal(size=(25,5));y=rng.normal(size=25);q=rng.normal(size=(7,5))
        gp=LinearGP().fit(x,y)
        mean,cov=gp.predict(q,full_covariance=True)
        c=gp.signal*x@x.T+gp.noise*np.eye(len(x))
        cross=gp.signal*q@x.T
        np.testing.assert_allclose(mean,gp.mean+cross@np.linalg.solve(c,y-gp.mean),atol=1e-9)
        np.testing.assert_allclose(cov,gp.signal*q@q.T-cross@np.linalg.solve(c,cross.T),atol=1e-9)
        self.assertEqual(gp.x.dtype,np.float64)
        self.assertEqual(gp.audit['numerical_jitter'],0.)
        self.assertGreater(gp.noise,0.)

    def test_joint_information_is_not_sum_of_variances(self):
        duplicate=np.ones((2,2));independent=np.eye(2)
        a,_=batch_information(duplicate,1.);b,_=batch_information(independent,1.)
        self.assertAlmostEqual(a,.5*np.log(3))
        self.assertAlmostEqual(b,np.log(2))
        self.assertLess(a,b)
        self.assertAlmostEqual(np.trace(duplicate),np.trace(independent))

    def test_joint_information_chain_rule(self):
        covariance=np.array([[2.,.8],[.8,1.]])
        information,_=batch_information(covariance,.3)
        first=.5*np.log1p(2/.3)
        conditional_second=1.-.8**2/(2.+.3)
        self.assertAlmostEqual(information,first+.5*np.log1p(conditional_second/.3))

    def test_invalid_covariances_fail(self):
        for matrix in (np.diag([-.2,1]),np.array([[1.,.4],[.1,1.]]),np.array([[np.nan]])):
            with self.assertRaises(ValueError):batch_information(matrix,1.)
        with self.assertRaises(ValueError):batch_information(np.eye(2),0.)

    def test_jitter_does_not_hide_indefiniteness(self):
        chol,jitter=stable_cholesky(np.ones((3,3)))
        self.assertGreater(jitter,0.)
        self.assertLessEqual(jitter,1e-8)
        with self.assertRaises(ValueError):stable_cholesky(np.diag([-1.,1.]))

    def test_fixed_ivr_score_matches_existing_first_step(self):
        path=HERE.parents[2]/'src/qgeognn_al/active_learning_v2/ivr.py'
        spec=importlib.util.spec_from_file_location('historical_ivr',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(32);x=rng.normal(size=(48,6))
        variance,score=fixed_kernel_signals(x,12)
        historical=module.conditional_batch_ivr(x,np.arange(12),np.arange(12,48),1)
        selected=12+historical['selected_candidate_positions'][0]
        self.assertEqual(selected,12+int(np.argmax(score[12:])))
        self.assertAlmostEqual(score[selected],historical['trace'][0]['score'],places=12)
        self.assertAlmostEqual(variance[selected],historical['trace'][0]['conditional_variance'],places=12)

    def test_gp_no_evaluation_label_argument(self):
        # A fitted posterior depends only on its training data and query features.
        rng=np.random.default_rng(24);x=rng.normal(size=(20,4));y=rng.normal(size=20)
        gp=LinearGP().fit(x[:10],y[:10]);before=gp.predict(x[10:])
        y[10:]=1e9
        after=gp.predict(x[10:])
        np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])

    def test_probability_metrics_include_observation_noise(self):
        metrics=probability_metrics(np.zeros(4),np.zeros(4),np.ones(4))
        self.assertEqual(metrics['coverage_90'],1.)
        self.assertAlmostEqual(metrics['nlpd'],.5*np.log(2*np.pi))
        with self.assertRaises(ValueError):probability_metrics([1],[1],[0])

    def test_reader_rejects_unlisted_paths_before_open(self):
        reader=Reader()
        with self.assertRaises(ValueError):reader.read(HERE.parents[2]/'dataset/do_not_open.csv')
        with self.assertRaises(ValueError):reader.read(HERE.parents[2]/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/selections/not_listed/feedback.json')
        for path in ('test/truth.csv','studies/active_learning/qgeognn_v2_row_llm_scientist_v2/fake/banks.npz',
                     'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/selections/seed_887/free_cw_sol61_xhigh_rerun32/round_06/feedback.json'):
            with self.assertRaises(ValueError):reader.read(HERE.parents[2]/path)

    def test_forward_evaluator_cannot_fit_future_response_or_risk_targets(self):
        rng=np.random.default_rng(75)
        data={'y0':rng.normal(size=(333,2)), 'y':rng.normal(size=(192,2)),
              'pred':rng.normal(size=(192,2)), 'features':{'raw':rng.normal(size=(525,4))},
              'batches':[list(range(333+32*r,365+32*r)) for r in range(6)],
              'ids':[str(i) for i in range(525)]}
        plan={'log_variance_bounds':[-9.,9.], 'random_batch_seed':101,'random_batch_count':2}
        before=evaluate_seed(1,data,plan)
        data['y'][96:]+=100.
        after=evaluate_seed(1,data,plan)
        for old,new in zip(before[1],after[1]):
            if old['target']=='fixed_r0_log_risk' or old['round']<3:
                self.assertEqual(old['mean'],new['mean'])
                self.assertEqual(old['observation_variance'],new['observation_variance'])
        old_risk=[f for f in before[2] if f['target']=='fixed_r0_log_risk']
        new_risk=[f for f in after[2] if f['target']=='fixed_r0_log_risk']
        self.assertEqual(old_risk,new_risk)

    def test_production_hash_convention(self):
        self.assertEqual(upstream_stable_hash({'b':2,'a':1}),upstream_stable_hash({'a':1,'b':2}))
        self.assertNotEqual(upstream_stable_hash({'a':1}),upstream_stable_hash({'a':2}))

    def test_offline_has_no_production_runtime_imports(self):
        tree=ast.parse((HERE/'offline.py').read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        for name in names:
            self.assertFalse(any(s in name for s in ('runner','oracle','openai','requests','torch','training')))


if __name__=='__main__':
    unittest.main()

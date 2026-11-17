"""Independent dense Schur/logdet oracle; synthetic labels are never sampled."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acquisition_v2 import (GaussianStateWithBALD, IndependentStreamingCache,
                            batch_target_information, select_streaming_batch)


class DensePosterior:
    def __init__(self, covariance, noise, q):
        self.k = np.asarray(covariance, dtype=float)
        self.noise_cov = np.asarray(noise, dtype=float)
        self.n_outputs = q
        self.row_ids = np.array([f"r{i}" for i in range(len(self.k)//q)])

    def cov(self, a, b, noisy=False):
        q = self.n_outputs
        ia = (np.asarray(a)[:, None]*q+np.arange(q)).ravel()
        ib = (np.asarray(b)[:, None]*q+np.arange(q)).ravel()
        result = self.k[np.ix_(ia, ib)].copy()
        if noisy:
            for i, ai in enumerate(a):
                for j, bj in enumerate(b):
                    if ai == bj:
                        result[i*q:(i+1)*q,j*q:(j+1)*q] += self.noise_cov
        return result

    def latent_diag(self, positions):
        return np.stack([self.cov([p], [p]) for p in positions])


def conditioned(p, a, b, selected):
    out = p.cov(a, b)
    if len(selected):
        out -= p.cov(a, selected) @ np.linalg.solve(p.cov(selected, selected, noisy=True), p.cov(selected, b))
    return out


def oracle_scores(p, candidates, refs, selected=(), objective="epig"):
    results = []
    for candidate in candidates:
        if candidate in selected:
            results.append(-np.inf)
            continue
        sx = conditioned(p, [candidate], [candidate], selected)+p.noise_cov
        if objective == "bald":
            results.append(.5*(np.linalg.slogdet(sx)[1]-np.linalg.slogdet(p.noise_cov)[1]))
            continue
        scores = []
        for ref in refs:
            st = conditioned(p, [ref], [ref], selected)+p.noise_cov
            cross = conditioned(p, [ref], [candidate], selected)
            after = st-cross @ np.linalg.solve(sx, cross.T)
            scores.append(.5*(np.linalg.slogdet(st)[1]-np.linalg.slogdet(after)[1]) if objective == "epig" else np.trace(st-after))
        results.append(float(np.mean(scores)))
    return np.asarray(results)


def posterior(seed=97, n=11, q=2, independent=False, noise=0.2):
    rng = np.random.default_rng(seed)
    if independent:
        covariance = np.zeros((n*q,n*q))
        for out in range(q):
            x = rng.normal(size=(n,5))
            covariance[out::q,out::q] = x@x.T/5 + np.eye(n)*1e-5
    else:
        x = rng.normal(size=(n*q, 8))
        covariance = x@x.T/8+np.eye(n*q)*1e-5
    return DensePosterior(covariance, np.diag(noise*np.geomspace(.1,10,q)), q)


def run():
    errors = []
    for q in (1,2,3):
        for noise in (.001,.2,20):
            p = posterior(q=q,noise=noise)
            candidates, refs = np.arange(2,11), np.array([0,2,3,8,10])
            for objective in ("epig","ivr","bald"):
                state = GaussianStateWithBALD(p,candidates,refs,chunk_size=2)
                selected=[]
                for _ in range(5):
                    actual = state.scores(objective)
                    expected = oracle_scores(p,candidates,refs,selected,objective)
                    finite = np.isfinite(expected)
                    error = float(np.max(np.abs(actual[finite]-expected[finite])))
                    errors.append({"q":q,"noise":noise,"objective":objective,"step":len(selected),"absolute_error":error})
                    np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=1e-8)
                    winner=int(np.argmax(expected)); selected.append(int(candidates[winner]));state.condition(winner)
    streaming=[]
    for q in (1,2,3):
        p = posterior(q=q,independent=True)
        with tempfile.TemporaryDirectory() as directory:
            cache=IndependentStreamingCache(p,np.arange(2,11),directory,target_chunk=2,candidate_chunk=3)
            for objective in ("epig","ivr","bald"):
                result=select_streaming_batch(cache,np.arange(11),objective=objective,batch_size=5)
                chosen=[]
                for row in result["trace"]:
                    oracle=oracle_scores(p,np.arange(2,11),np.arange(11),chosen,objective)
                    assert row["candidate_index"] == int(np.argmax(oracle))
                    np.testing.assert_allclose(row["conditional_score"],max(oracle),rtol=1e-9,atol=1e-10)
                    chosen.append(row["row_position"])
                dense_information=0.
                for ref in range(11):
                    st=p.cov([ref],[ref])+p.noise_cov
                    cross=p.cov([ref],chosen)
                    after=st-cross@np.linalg.solve(p.cov(chosen,chosen,noisy=True),cross.T)
                    dense_information += .5*(np.linalg.slogdet(st)[1]-np.linalg.slogdet(after)[1])/11
                actual_information=batch_target_information(p,np.arange(11),chosen,chunk=2)
                np.testing.assert_allclose(actual_information,dense_information,atol=1e-10)
                if objective == "epig":
                    np.testing.assert_allclose(sum(r["conditional_score"] for r in result["trace"]),dense_information,atol=1e-10)
                streaming.append({"q":q,"objective":objective,"selected":chosen,"batch_information":actual_information})
    result={"status":"PASS","dense_conditioned_score_comparisons":len(errors),
            "maximum_absolute_error":max(e["absolute_error"] for e in errors),
            "errors":errors,"streaming_comparisons":streaming,"real_labels_read":0}
    (HERE/"math_oracle_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"errors","streaming_comparisons"}}))

if __name__=="__main__":
    run()

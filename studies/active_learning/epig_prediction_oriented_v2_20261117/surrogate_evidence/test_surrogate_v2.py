"""Independent synthetic checks for new group/conditional-score machinery."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]/"src"))
import numpy as np
from scipy.linalg import block_diag
from qgeognn_al.active_learning_v2.epig_models import GaussianProcessSurrogate
from surrogate_v2 import GroupedTransform,StudyGP
from surrogate_audit import fold_diagnostics,nll


def test_group_weighting_is_applied_after_train_standardization():
    x=np.random.default_rng(11).normal(size=(50,31))*np.arange(1,32)
    t=GroupedTransform(None).fit(x)
    expected=(x-x.mean(0))/x.std(0)*np.repeat(1/np.sqrt([16,6,3,6]),[16,6,3,6])
    np.testing.assert_allclose(t.transform(x),expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(t.transform(x).mean(0),0,atol=1e-12)
    np.testing.assert_allclose(t.transform(x).std(0),np.repeat(1/np.sqrt([16,6,3,6]),[16,6,3,6]),atol=1e-12)


def test_study_ungrouped_gp_reproduces_original_posterior():
    rng=np.random.default_rng(13);x=rng.normal(size=(35,9));y=rng.normal(size=(35,2));xt=rng.normal(size=(11,9))
    old=GaussianProcessSurrogate("matern32",max_components=6,maxiter=60,restarts=1).fit(x,y)
    new=StudyGP(grouped=False,max_components=6,maxiter=60,restarts=1).fit(x,y)
    a,b=old.posterior(xt),new.posterior(xt)
    np.testing.assert_allclose(a.mean,b.mean,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(a.cov(np.arange(11),np.arange(11),noisy=True),b.cov(np.arange(11),np.arange(11),noisy=True),rtol=1e-7,atol=1e-7)


def test_row_conditional_logscore_matches_dense_schur_oracle():
    rng=np.random.default_rng(17);a=rng.normal(size=(12,12));cov=a@a.T+.2*np.eye(12)
    y=rng.normal(size=(6,2));mean=rng.normal(size=(6,2));result=fold_diagnostics(y,mean,cov)
    for row in range(6):
        target=np.arange(2*row,2*row+2);other=np.setdiff1d(np.arange(12),target)
        cross=cov[np.ix_(target,other)];observed=cov[np.ix_(other,other)]
        condition_mean=mean.ravel()[target]+cross@np.linalg.solve(observed,(y-mean).ravel()[other])
        condition_cov=cov[np.ix_(target,target)]-cross@np.linalg.solve(observed,cross.T)
        oracle=nll(y[row:row+1],condition_mean[None,:],condition_cov)
        np.testing.assert_allclose(result["conditional_row_nll"][row],oracle,rtol=1e-12,atol=1e-12)


def test_row_independent_covariance_has_zero_cross_sample_logscore_change():
    rng=np.random.default_rng(19);blocks=[]
    for _ in range(7):
        a=rng.normal(size=(2,2));blocks.append(a@a.T+.5*np.eye(2))
    y=rng.normal(size=(7,2));mean=rng.normal(size=(7,2))
    result=fold_diagnostics(y,mean,block_diag(*blocks))
    assert abs(result["conditional_minus_unconditional"])<1e-12

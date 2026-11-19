"""Synthetic mathematics/scale check; never an AL learning-performance result."""
from study_support import ROOT
import json,time,resource
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from qgeognn_al.active_learning_v2.epig_models import GaussianProcessSurrogate,BayesianLinearSurrogate
from qgeognn_al.active_learning_v2.epig import select_gaussian_batch,score_gaussian_acquisition
rng=np.random.default_rng(20261117)
x=rng.normal(size=(45,7)); y=np.column_stack([np.sin(x[:,0])+0.1*rng.normal(size=len(x)),x[:,1]-0.4*x[:,0]+0.1*rng.normal(size=len(x))])
records=[]
for model in [GaussianProcessSurrogate(maxiter=20),BayesianLinearSurrogate(maxiter=20,coregionalized=True)]:
 p=model.fit(x[:20],y[:20]).posterior(x,row_ids=np.arange(len(x)))
 candidates=np.arange(20,45);refs=np.arange(45)
 result=select_gaussian_batch(p,candidates,refs,batch_size=5,chunk_size=7,labeled_positions=np.arange(20))
 selected=result['selected_positions'];cov=p.cov(selected,selected,noisy=True); factor=cho_factor(cov,lower=True)
 direct=0.
 for target in refs:
  st=p.cov(np.array([target]),np.array([target]))+p.noise_cov
  cross=p.cov(np.array([target]),np.array(selected))
  after=st-cross@cho_solve(factor,cross.T)
  direct+=.5*(np.linalg.slogdet(st)[1]-np.linalg.slogdet(after)[1])/len(refs)
 actual=result['audit']['conditional_batch_information_nats']
 assert np.isclose(actual,direct,rtol=1e-10,atol=1e-12),(actual,direct)
 records.append({'family':model.audit_['family'],'selected_positions':selected,'cached_batch_information_nats':actual,'direct_dense_batch_information_nats':direct,'absolute_error':abs(actual-direct),'seconds':result['audit']['elapsed_seconds']})
# Scale benchmark uses an actual fitted BLR posterior; labels are synthetic only.
x=rng.normal(size=(3330,16)); y=np.column_stack([x[:333,0]+.1*rng.normal(size=333),x[:333,1]+.1*rng.normal(size=333)])
p=BayesianLinearSurrogate(coregionalized=True,maxiter=20).fit(x[:333],y).posterior(x)
call_counts={'calls':0,'largest_block_entries':0,'candidate_dense_calls':0}
base_cov=p.cov
def counted_cov(a,b,noisy=False):
 call_counts['calls']+=1;call_counts['largest_block_entries']=max(call_counts['largest_block_entries'],len(a)*len(b)*4)
 if len(a)==len(b) and len(a)>2048:call_counts['candidate_dense_calls']+=1
 return base_cov(a,b,noisy=noisy)
p.cov=counted_cov
result=select_gaussian_batch(p,np.arange(333,3330),np.arange(2048),batch_size=32,chunk_size=128,labeled_positions=np.arange(333))
assert len(result['selected_positions'])==len(set(result['selected_positions']))==32
assert call_counts['candidate_dense_calls']==0
records.append({'family':'synthetic_fitted_blr_coregionalized_scale_benchmark','universe_rows':3330,'candidate_rows':2997,'reference_rows':2048,'batch_size':32,'cache_bytes':result['audit']['cache_bytes'],'seconds':result['audit']['elapsed_seconds'],'process_peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**call_counts})
print(json.dumps(records,indent=2))

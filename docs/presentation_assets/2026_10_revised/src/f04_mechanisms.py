"""Auditable constructed examples for F04, executed by unmodified selectors.

No model, checkpoint, experimental features, targets or study lifecycle is used.
The two-dimensional inputs are deliberately constructed teaching examples.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import types
import numpy as np

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data/illustrative'


def original_selectors():
    manifest=json.loads((DATA/'F04_source_manifest.json').read_text())
    for item in manifest:
        p=BASE/item['snapshot']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('F04 selector snapshot modified: '+str(p))
    name='_f04_audited_selectors'
    if name not in sys.modules:
        pkg=types.ModuleType(name)
        pkg.__path__=[str(DATA/'F04_selector_source')]
        sys.modules[name]=pkg
    previous=sys.dont_write_bytecode
    sys.dont_write_bytecode=True
    try:
        lcmd=importlib.import_module(name+'.lcmd').lcmd_tp_select
        maxdet=importlib.import_module(name+'.maxdet').conditional_gradient_maxdet
        ivr=importlib.import_module(name+'.ivr').conditional_batch_ivr
    finally:
        sys.dont_write_bytecode=previous
    return lcmd,maxdet,ivr


def assignments(pool,centers,excluded=()):
    squared=np.sum((pool[:,None,:]-centers[None,:,:])**2,axis=2)
    nearest=np.argmin(squared,axis=1)
    distance=squared[np.arange(len(pool)),nearest]
    available=~np.isin(np.arange(len(pool)),excluded)
    return {'center':nearest.tolist(),'squared_distance':distance.tolist(),
            'cluster_count':np.bincount(nearest[available],minlength=len(centers)).tolist(),
            'cluster_score':np.bincount(nearest[available],weights=distance[available],
                                         minlength=len(centers)).tolist()}


def examples():
    lcmd,maxdet,ivr=original_selectors()
    # The smaller cluster has much larger squared-distance mass.
    train=np.array([[-.8,0.],[.5,0.]])
    pool=np.array([[-.95,.15],[-.9,-.15],[-.65,.2],[-.7,-.2],[-.75,.05],
                   [-.55,-.05],[.8,.6],[1.25,.6],[1.15,-.5]])
    lc=lcmd(pool,train,2)
    first=int(lc.selected_pool_positions[0])
    lcmd_data={'train':train.tolist(),'pool':pool.tolist(),
               'selected':lc.selected_pool_positions.tolist(),'trace':list(lc.trace),
               'initial':assignments(pool,train),
               'after_first':assignments(pool,np.vstack([train,pool[first]]),(first,))}
    assert lcmd_data['initial']['cluster_count']==[6,3]
    assert lcmd_data['selected']==[7,8]
    assert np.allclose(lcmd_data['initial']['cluster_score'],[.260,2.045])
    assert np.isclose(max(lcmd_data['after_first']['cluster_score']),
                      lc.trace[1]['largest_cluster_score'])

    # A covered long direction is less informative; the first pick changes rank.
    train=np.array([[3.,0.],[3.,0.],[3.,0.]])
    pool=np.array([[3.4,.1],[.3,2.4],[.1,2.0],[1.6,1.7],[2.6,.5]])
    raw=np.vstack([train,pool]); n=len(train)
    md=maxdet(raw,np.arange(n),np.arange(n,len(raw)),2)
    x=raw/md.normalization_scale
    a=np.eye(2)+x[:n].T@x[:n]
    conditioned=np.linalg.solve(np.linalg.cholesky(a),x[n:].T).T
    first=int(md.selected_candidate_positions[0]); v=x[n+first]
    c_after=np.linalg.inv(a+np.outer(v,v))
    gain_after=np.log1p(np.einsum('ij,jk,ik->i',x[n:],c_after,x[n:]))
    maxdet_data={'train':train.tolist(),'pool':pool.tolist(),'names':list('PQRST'),
                'scale':md.normalization_scale,'conditioned_pool':conditioned.tolist(),
                'gain_before':md.initial_marginal_gains.tolist(),
                'gain_after_first':gain_after.tolist(),
                'selected':md.selected_candidate_positions.tolist(),'trace':list(md.trace)}
    assert maxdet_data['selected']==[1,0]
    assert int(np.argsort(-md.initial_marginal_gains)[1])==2
    assert np.isclose(gain_after[0],md.trace[1]['marginal_logdet_gain'])

    # P has the largest local proxy variance; Q improves more reference rows.
    train=np.array([[.2,0.],[.1,.1]])
    pool=np.array([[0.,2.8],[1.9,.1],[1.7,0.],[1.6,-.1],
                   [1.55,.15],[1.8,-.1],[1.75,.1]])
    raw=np.vstack([train,pool]); n=len(train)
    iv=ivr(raw,np.arange(n),np.arange(n,len(raw)),2)
    scale=float(np.sqrt(np.mean(np.sum(raw**2,axis=1))))
    x=raw/scale
    a=np.eye(2)+x[:n].T@x[:n]
    c=np.linalg.inv(a); m=x.T@x/len(x)
    variance=np.einsum('ij,jk,ik->i',x,c,x)
    scores=np.einsum('ij,jk,kl,lm,im->i',x[n:],c,m,c,x[n:])/(1+variance[n:])
    first=iv['selected_candidate_positions'][0]; v=x[n+first]
    after=np.einsum('ij,jk,ik->i',x,np.linalg.inv(a+np.outer(v,v)),x)
    ivr_data={'train':train.tolist(),'pool':pool.tolist(),'names':list('PQRSTUV'),
              'scale':scale,'variance_before':variance.tolist(),'variance_after_first':after.tolist(),
              'scores_before':scores.tolist(),'selected':iv['selected_candidate_positions'],
              'trace':iv['trace'],'reference_rows':len(raw),
              'prior_precision':1.,'noise_variance':1.,'reference_weight':1./len(raw)}
    assert ivr_data['selected']==[1,0]
    assert variance[n]>variance[n+1] and scores[0]<scores[1]
    assert np.isclose(variance.mean(),iv['trace'][0]['integrated_variance_before'])
    assert np.isclose(after.mean(),iv['trace'][0]['integrated_variance_after'])
    assert np.isclose(variance.mean()-after.mean(),scores[first])
    result={'kind':'constructed 2D teaching examples, not experimental data or a projection',
            'execution':'verbatim repository selectors; no model or targets loaded',
            'shown_picks':2,'experimental_feature_dimension':512,'experimental_batch_size':32,
            'lcmd':lcmd_data,'maxdet':maxdet_data,'ivr':ivr_data}
    (DATA/'F04_mechanism_examples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    r=examples()
    print('Original selectors verified on 2D constructed features:',
          'LCMD',r['lcmd']['selected'],'MaxDet',r['maxdet']['selected'],'IVR',r['ivr']['selected'])

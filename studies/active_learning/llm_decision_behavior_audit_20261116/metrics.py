"""Target-free, frozen-metadata behavioral metrics and numerical primitives."""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

def hamming(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return np.mean(a[:,None,:]!=b[None,:,:],axis=2)

def continuous_distance(a,b,ranges):
    a=np.asarray(a,float);b=np.asarray(b,float);ranges=np.asarray(ranges,float)
    if np.any(ranges<=0): raise ValueError('Scales must be positive')
    return np.mean(np.abs(a[:,None,:]-b[None,:,:])/ranges,axis=2)

def rarity(codes,history,cardinalities,normalized=False):
    codes=np.asarray(codes);history=np.asarray(history,int)
    result=np.zeros(len(codes),float)
    for j,n in enumerate(cardinalities):
        counts=np.bincount(codes[history,j],minlength=max(n,int(codes[:,j].max())+1))
        contribution=-np.log((counts[codes[:,j]]+1)/(len(history)+n))
        result+=contribution/np.log(len(history)+n) if normalized else contribution
    return result/codes.shape[1]

def coverage_gain(before,new_distances):
    before=np.asarray(before,float)
    return float(np.mean(before-np.minimum(before,np.min(new_distances,axis=-1))))

def percentile(actual,random):
    random=np.asarray(random,float)
    ties=np.isclose(random,actual,rtol=1e-10,atol=1e-12)
    return float((np.sum((random<actual)&~ties)+.5*np.sum(ties))/len(random))

def random_batches(pool,n,size,seed):
    pool=np.asarray(pool,int)
    if len(pool)<size:raise ValueError('Not enough eligible candidates')
    rng=np.random.default_rng(seed)
    return np.array([rng.choice(pool,size,replace=False) for _ in range(n)])

def aulc(budgets,loss):
    x=np.asarray(budgets,int);y=np.asarray(loss,float)
    if len(x)<2 or np.any(np.diff(x)!=32):raise ValueError('Incomplete budget grid')
    return float(np.sum((y[:-1]+y[1:])*.5*np.diff(x))/(x[-1]-x[0]))

def align_gain(curves,before,after):
    if after-before!=32 or before not in curves or after not in curves:
        raise ValueError('Missing exact adjacent evaluation')
    return curves[before]-curves[after]

def conditional_logit(groups,l2=1.):
    groups=[np.asarray(g,float) for g in groups]
    k=groups[0].shape[1]
    def objective(beta):
        loss=.5*l2*np.dot(beta,beta);grad=l2*beta.copy()
        for x in groups:
            z=x@beta;lp=logsumexp(z);p=np.exp(z-lp)
            loss+=lp-z[0];grad+=p@x-x[0]
        return loss,grad
    fit=minimize(objective,np.zeros(k),jac=True,method='L-BFGS-B')
    return fit

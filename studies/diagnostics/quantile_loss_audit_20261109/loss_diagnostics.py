"""Read-only checkpoint loss/gradient measurements; no optimizer step is exposed."""
from __future__ import annotations
import hashlib
import numpy as np
import torch

COMPONENTS=('q10_pinball','q50_mse','q90_pinball','cross_q10_q50','cross_q50_q90')


def target_components(truth, prediction):
    if prediction.ndim!=2 or prediction.shape[1]!=3 or truth.shape!=(len(prediction),):
        raise ValueError('one target and three raw predictions required')
    low=truth-prediction[:,0];high=truth-prediction[:,2]
    return dict(zip(COMPONENTS,[torch.maximum(.1*low,-.9*low).mean(),torch.square(truth-prediction[:,1]).mean(),
                               torch.maximum(.9*high,-.1*high).mean(),torch.relu(prediction[:,0]-prediction[:,1]).mean(),
                               torch.relu(prediction[:,1]-prediction[:,2]).mean()]))


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        a=value.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(repr(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def measure_gradients(model,batches,*,row_count):
    if model.training or row_count<1:raise ValueError('frozen eval-mode model and explicit authorized row count required')
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and (n.startswith('backbone.') or n.startswith('condition_branch.'))]
    if not named:raise ValueError('qualified backbone/condition parameter groups required')
    parameters=[p for _,p in named];sizes=[p.numel() for p in parameters];offset=np.cumsum([0]+sizes)
    groups={g:np.concatenate([np.arange(offset[i],offset[i+1]) for i,(n,_) in enumerate(named) if n.startswith(g+'.')]) for g in ['backbone','condition_branch']}
    gradients={(target,component):np.zeros(offset[-1],dtype=np.float64) for target in ['V1','V2'] for component in COMPONENTS}
    losses={key:0. for key in gradients};before=state_hash(model);seen=0
    for atom,angle,truth in batches:
        output=model(atom,angle)
        if output.shape!=(len(truth),6) or truth.shape[1]!=2:raise ValueError('authorized batch shape mismatch')
        weight=len(truth)/row_count;seen+=len(truth)
        terms=[(target,component,value) for endpoint,target in enumerate(['V1','V2']) for component,value in target_components(truth[:,endpoint],output[:,3*endpoint:3*endpoint+3]).items()]
        for index,(target,component,value) in enumerate(terms):
            values=torch.autograd.grad(value,parameters,retain_graph=index<len(terms)-1,allow_unused=True)
            flat=np.concatenate([np.zeros(size,dtype=np.float64) if grad is None else grad.detach().cpu().reshape(-1).numpy().astype(np.float64) for grad,size in zip(values,sizes)])
            gradients[target,component]+=weight*flat;losses[target,component]+=weight*float(value.detach())
    if seen!=row_count or state_hash(model)!=before:raise RuntimeError('frozen model/state or authorized row count changed')
    rows=[];competition=[]
    for target in ['V1','V2']:
        total=sum((gradients[target,c] for c in COMPONENTS),start=np.zeros(offset[-1],dtype=np.float64));gradients[target,'total']=total;losses[target,'total']=sum(losses[target,c] for c in COMPONENTS)
        for component in [*COMPONENTS,'total']:
            for group,positions in groups.items():rows.append({'target':target,'component':component,'parameter_group':group,'mean_loss':losses[target,component],'gradient_l2':float(np.linalg.norm(gradients[target,component][positions]))})
    for group,positions in groups.items():
        a,b=gradients['V1','total'][positions],gradients['V2','total'][positions];an,bn=np.linalg.norm(a),np.linalg.norm(b);dot=float(a@b)
        competition.append({'parameter_group':group,'V1_gradient_l2':float(an),'V2_gradient_l2':float(bn),'V2_to_V1_gradient_norm':float(bn/an) if an else None,
                            'cosine':dot/float(an*bn) if an and bn else None,'dot_product':dot,'gradient_conflict':dot<0,
                            'V1_dot_combined_gradient':float(a@(a+b)),'V2_dot_combined_gradient':float(b@(a+b)),
                            'mode':'frozen-checkpoint eval gradient; not actual Adam update or training-trajectory dominance'})
    return {'components':rows,'competition':competition,'state_before':before,'state_after':state_hash(model),'parameters_updated':False,'optimizer_steps':0}

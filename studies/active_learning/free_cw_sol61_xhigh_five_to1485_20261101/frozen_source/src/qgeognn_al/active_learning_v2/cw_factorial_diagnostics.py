"""Selection-only small-row projection and historical provenance diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from ..models import load_predictor_checkpoint
from ..training.predictor import loader_pair, atomic_json
from ..artifacts import sha256_file
from .cw_factorial_study import Context, STUDY, BASE, PARENT, read
from .cw_factorial import SEEDS, DIMENSIONS, PROJECTION_SEEDS, select
from .gradient_features import countsketch_mapping


def small_selection_audit():
    torch.set_num_threads(2)
    rows=[]
    for seed in SEEDS:
        c=Context(seed); model=load_predictor_checkpoint(c.anchor); model.eval()
        indices=np.r_[c.roles['l0'][:4],c.roles['u0'][:4]]
        pars=tuple(p for p in model.parameters() if p.requires_grad)
        jac=[]
        for a,b in zip(*loader_pair(c.atom,c.angle,indices,1)):
            out=model(a,b); pair=[]
            for col in (1,4):
                grads=torch.autograd.grad(out[0,col],pars,retain_graph=col==1,allow_unused=True)
                pair.append(torch.cat([torch.zeros(p.numel()) if g is None else g.detach().reshape(-1).cpu() for p,g in zip(pars,grads)]).numpy().astype(float))
            jac.append(pair)
        jac=np.array(jac)
        f={k:np.einsum('ab,nbp->nap',t,jac).reshape(8,-1) for k,t in c.transforms.items()}
        exact={}
        for k,x in f.items():
            gram=x@x.T; eig,vec=np.linalg.eigh(gram)
            reduced=vec*np.sqrt(np.maximum(eig,0))
            exact[k]={sel:select(reduced,4,sel,2)[0].tolist() for sel in ('lcmd','maxdet','ivr')}
        for dim in DIMENSIONS:
            for ps in PROJECTION_SEEDS:
                buckets,signs=countsketch_mapping(jac.shape[-1],dim,ps)
                for k,x in f.items():
                    sk=np.vstack([np.bincount(buckets.numpy(),weights=row*signs.numpy(),minlength=dim) for row in x])
                    # Use the exact factor of this small sketch Gram, avoiding a large
                    # feature-space solve; objectives depend on the same row kernel.
                    eig,vec=np.linalg.eigh(sk@sk.T); reduced=vec*np.sqrt(np.maximum(eig,0))
                    for sel in ('lcmd','maxdet','ivr'):
                        picks=select(reduced,4,sel,2)[0].tolist()
                        a,b=set(picks),set(exact[k][sel])
                        rows.append({'seed':seed,'dimension':dim,'projection_seed':ps,'representation':k,'selector':sel,
                                     'pool_positions':json.dumps(picks),'exact_pool_positions':json.dumps(exact[k][sel]),
                                     'jaccard_to_uncompressed':len(a&b)/len(a|b)})
        del f,jac
    pd.DataFrame(rows).to_csv(STUDY/'small_row_selection_audit.csv',index=False)


def historical_sketch_audit():
    entries=[]
    for seed in SEEDS:
        expected=read(PARENT/f'qgeognn_v2_row_short_sequential_b32/runtime/seed_{seed}/center_width_lcmd/round_00/acquisition_artifacts/gradient_audit.json')['gradient_audit']['sketch_mapping_sha256']
        paths=[]
        for name,pattern in [
            ('raw_lcmd',BASE/f'runtime/seed_{seed}/lcmd'),
            ('raw_maxdet',PARENT/f'qgeognn_v2_row_maxdet_b32/runtime/seed_{seed}/gradient_maxdet'),
            ('raw_ivr',PARENT/f'qgeognn_v2_row_kernel_ivr_b32/runtime/seed_{seed}'),
            ('cw_lcmd',PARENT/f'qgeognn_v2_row_short_sequential_b32/runtime/seed_{seed}/center_width_lcmd')]:
            paths += [(name,p) for p in pattern.glob('round_*/**/gradient_audit.json') if int(p.parts[len(pattern.parts)].split('_')[-1])<=5]
        from .cw_lcmd_extension_study import STUDY as extension
        paths += [('cw_lcmd',p) for p in (extension/f'runtime/seed_{seed}/center_width_lcmd').glob('round_*/**/gradient_audit.json') if int(p.parts[-3].split('_')[-1])<=5]
        for arm,p in paths:
            record=read(p); a=record.get('gradient_audit',record)
            assert a['sketch_mapping_sha256']==expected
            assert a['sketch_dimension']==512 and a['sketch_seed']==seed+4000037
            entries.append({'seed':seed,'arm':arm,'path':str(p),'sha256':sha256_file(p),'mapping_sha256':expected})
    atomic_json(STUDY/'historical_sketch_audit.json',{'entries':entries,'all_mapping_matches':True})

if __name__=='__main__':
    historical_sketch_audit()
    small_selection_audit()

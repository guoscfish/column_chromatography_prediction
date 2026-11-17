"""Execute the frozen controlled study; all writes stay in the new directory."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from data_boundary import HERE, ROOT, LEGACY, dump, file_hash, load_state, representations
from gp_training import ControlledProjectionGP, FixedRepresentationGP, regression_metrics, variance_temperature
from gp_models import acquisition_diagnostics
from qgeognn_al.active_learning_v2.benchmark_protocol import load_features

METHODS = ('original','converged','regularized','noise_floor','staged')
CONFIG = {'seeds':[157,887], 'budgets':[333,813], 'initializations':[17,29],
          'methods':list(METHODS), 'representations':['qwen','fusion'], 'maxiter':200,
          'ridge':.01,'noise_floor':.02,'stage_iterations':100,'wall_cap_seconds':2400,
          'protocol_sha256':file_hash(HERE/'PROTOCOL.md')}


def acq_summary(model, candidates, frame):
    started=time.perf_counter()
    result=acquisition_diagnostics(model,candidates,batch_size=32,candidate_limit=256)
    for method, details in result['methods'].items():
        selected=frame.iloc[details['selected_candidate_positions']]
        details['unique_molecules']=int(selected.canonical_smiles.nunique())
        details['condition_unique_counts']={k:int(selected[k].nunique()) for k in ('PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul')}
        details['numeric_ranges']={k:[float(selected[k].min()),float(selected[k].max())] for k in ('Density g/ml','V/ul','Volume of loading solvent/ul')}
    result['seconds']=time.perf_counter()-started
    return result


def overlap(a,b):
    return {m:len(set(v['selected_candidate_positions'])&set(b['methods'][m]['selected_candidate_positions']))/32
            for m,v in a['methods'].items()}


def run_one(state,data,subsets,reps,name,method,init,j,metadata,features):
    n=state.budget;cs=slice(n,n+256);vs=slice(n+256,None)
    x=reps[name];fit=subsets['fit'];cal=subsets['calibration'];check=subsets['check']
    identity={'data_seed':state.seed,'budget_state':n,'endpoint':f'V{j+1}','representation':name,'method':method,'initialization':init}
    path=HERE/'phase1'/(f'{state.seed}_{n}_{name}_{method}_{init}_V{j+1}.json')
    if path.exists():
        return
    started=time.perf_counter()
    try:
        if method=='frozen':
            model=FixedRepresentationGP(max_iter=200,seed=17)
        else:
            model=ControlledProjectionGP(method=method,max_iter=40 if method=='original' else 200,seed=init)
        model.fit(x[fit],data['train_y'][fit,j])
        metrics={};predictions={}
        cm,cl=model.predict(x[cal]);alpha=variance_temperature(data['train_y'][cal,j],cm,cl+model.noise)
        for role,pos,y in [('check',check,data['train_y'][check,j]),('development_validation',vs,data['validation_y'][:,j])]:
            mu,latent=model.predict(x[pos]);obs=latent+model.noise
            metrics[role]={'raw':regression_metrics(y,mu,obs),'temperature':regression_metrics(y,mu,alpha*obs)}
            for values in metrics[role].values():
                values['normalized_rmse_L0']=values['rmse']/state.audit['endpoint_scales'][f'V{j+1}']
            predictions[role]={'mean':mu.tolist(),'latent_variance':latent.tolist(),'observation_variance':obs.tolist(),
                               'truth':y.tolist()}
        candidate_frame=features.iloc[data['indices'][cs]].reset_index(drop=True)
        acquisition=acq_summary(model,x[cs],candidate_frame)
        robustness=None
        if name in ('qwen','fusion'):
            xb=reps[name+'_template_b']
            am,av=model.predict(x[cs]);bm,bv=model.predict(xb[cs])
            b_acq=acq_summary(model,xb[cs],candidate_frame)
            rho=float(spearmanr(av,bv).statistic) if np.ptp(av)>0 and np.ptp(bv)>0 else None
            robustness={'training_template':'A_only','mean_absolute_change_ml':float(np.mean(np.abs(am-bm))),
                        'variance_spearman':rho,'top32_overlap':overlap(acquisition,b_acq),
                        'template_b_acquisitions':b_acq}
        audit=model.audit
        if 'covariance_condition_number' not in audit:
            eig=np.linalg.eigvalsh(model.signal*model._base_kernel(model.x,model.x))
            audit['covariance_condition_number']=float((eig[-1]+model.noise)/(max(eig[0],0)+model.noise))
            audit['noise_signal_ratio']=float(model.noise/model.signal)
        dump(path,{**identity,'status':'finite_fit','training_rows':len(fit),'calibration_rows':len(cal),'check_rows':len(check),
                   'calibration_temperature':alpha,'metrics':metrics,'predictions':predictions,'acquisitions':acquisition,
                   'template_robustness':robustness,'fit_audit':audit,'elapsed_seconds':time.perf_counter()-started,
                   'representation_provenance':('frozen_unsupervised_Qwen' if name=='qwen' else 'historical_GNN_saw_all_Lt_conditional_GP_only')})
    except Exception as exc:
        dump(path,{**identity,'status':'failed','error':repr(exc),'elapsed_seconds':time.perf_counter()-started})
    print(path.name,'done',round(time.perf_counter()-started,2),flush=True)


def expanded_reps(data, fit):
    a,fa=representations(data,fit)
    length=len(data['indices'])
    joined={'indices':np.r_[data['indices'],data['indices']], 'gnn':np.r_[data['gnn'],data['gnn']],
            'llm_conditions':np.r_[data['llm_conditions'],data['llm_template_b']]}
    b,_=representations(joined,fit)
    a['qwen_template_b']=data['llm_template_b'];a['fusion_template_b']=b['fusion'][length:]
    for name in ('gnn','raw','cw','structured'):
        a[name]=data[name]
    return a,fa


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    configpath=HERE/'phase1_config.json'
    if configpath.exists() and json.loads(configpath.read_text())!=CONFIG:
        raise ValueError('frozen phase1 configuration drift')
    if not configpath.exists():
        dump(configpath,CONFIG)
        dump(HERE/'phase1_freeze.json',{'config_sha256':file_hash(configpath),'code_hashes':{p.name:file_hash(p) for p in (HERE/'gp_training.py',HERE/'data_boundary.py',Path(__file__))}})
    freeze=json.loads((HERE/'phase1_freeze.json').read_text())
    if freeze['config_sha256']!=file_hash(configpath) or any(file_hash(HERE/p)!=h for p,h in freeze['code_hashes'].items()):
        raise ValueError('frozen phase1 code drift')
    features=load_features(ROOT/'experiments/e0_4g_baseline/canonical_4g.csv')
    start=time.monotonic();count=0
    for seed in CONFIG['seeds']:
        for budget in CONFIG['budgets']:
            state,data,meta,subsets=load_state(seed,budget)
            reps,fa=expanded_reps(data,subsets['fit'])
            roleids={role:np.asarray(state.labeled_ids)[pos].tolist() for role,pos in subsets.items()}
            dump(HERE/'phase1'/f'{seed}_{budget}_identity.json',{'state':state.audit,'cache_sha256':meta['cache_sha256'],
                  'roles':roleids,'fusion_preprocessing':fa,'candidate_ids':meta['candidate_ids'],
                  'candidate_canonical_indices':data['indices'][budget:budget+256].tolist(),
                  'test_truth_access_count':0,'future_U_labels_read':0,'source_response_reads':0})
            for name in ('qwen','gnn','structured','raw','cw'):
                for j in range(2):
                    run_one(state,data,subsets,reps,name,'frozen',17,j,meta,features)
                    count+=1
                    if args.smoke:
                        dump(HERE/'phase1_smoke_receipt.json',{'fits':count,'elapsed_seconds':time.monotonic()-start});return
            for method in METHODS:
                for name in CONFIG['representations']:
                    for init in CONFIG['initializations']:
                        for j in range(2):
                            if time.monotonic()-start>CONFIG['wall_cap_seconds']:
                                dump(HERE/'phase1_completion.json',{'complete':False,'reason':'fixed wall cap','elapsed_seconds':time.monotonic()-start});return
                            run_one(state,data,subsets,reps,name,method,init,j,meta,features);count+=1
    dump(HERE/'phase1_completion.json',{'complete':True,'fits_in_matrix':count,'elapsed_seconds':time.monotonic()-start,
         'new_gnn_fits':0,'al_trajectories':0,'downloads':0,'paid_api_calls':0,'test_truth_access_count':0})

if __name__=='__main__':
    main()

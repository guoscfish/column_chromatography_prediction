"""Full-L_t numerical replay, distinct from the molecularly held-out benchmark."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from data_boundary import HERE, LEGACY, dump, file_hash, load_state, representations
from gp_training import ControlledProjectionGP, regression_metrics


def main():
    config={'seed':157,'budgets':[333,813],'representations':['qwen','fusion'],
            'endpoint':'V1','initialization':17,'methods':['original','converged'],
            'scope':'full_Lt_mechanism_replay_not_independent_benchmark','protocol_sha256':file_hash(HERE/'PROTOCOL.md')}
    dump(HERE/'full_l_config.json',config)
    rows=[]
    for budget in config['budgets']:
        state,data,meta,subsets=load_state(157,budget)
        reps,_=representations(data,np.arange(budget))
        for name in config['representations']:
            legacy_name='llm_conditions' if name=='qwen' else name
            legacy=json.loads((LEGACY/'results'/f'seed_157_L{budget}'/(legacy_name+'_projected_init17.json')).read_text())
            prior=[r for r in legacy['predictions'] if r['endpoint']=='V1']
            for method in config['methods']:
                path=HERE/'full_l'/f'157_{budget}_{name}_{method}_17_V1.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()));continue
                x=reps[name]
                model=ControlledProjectionGP(method=method,max_iter=40 if method=='original' else 200,seed=17).fit(x[:budget],data['train_y'][:,0])
                mu,lv=model.predict(x[budget+256:]);obs=lv+model.noise
                tm,tv=model.predict(x[:budget])
                replay={key:float(np.max(np.abs(value-np.array([r[key] for r in prior])))) for key,value in [('mean_ml',mu),('latent_variance_ml2',lv),('observation_variance_ml2',obs)]}
                row={'data_seed':157,'budget_state':budget,'endpoint':'V1','representation':name,'method':method,'initialization':17,
                     'fit_audit':model.audit,'training_rmse':float(np.sqrt(np.mean((tm-data['train_y'][:,0])**2))),
                     'development_validation':regression_metrics(data['validation_y'][:,0],mu,obs),
                     'historical_A40_prediction_max_abs_difference':replay,
                     'scope':config['scope']}
                if method=='original' and not all(v<1e-6 for v in replay.values()):
                    raise ValueError('full-Lt original replay disagrees with sealed history')
                dump(path,row);rows.append(row)
                print(budget,name,method,model.audit['optimizer_success'],model.audit['noise_variance'],flush=True)
    dump(HERE/'full_l_completion.json',{'complete':True,'fits':len(rows),'full_l_config_sha256':file_hash(HERE/'full_l_config.json')})

if __name__=='__main__':
    main()

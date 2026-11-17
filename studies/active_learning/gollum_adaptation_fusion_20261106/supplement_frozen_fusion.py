"""Prespecified capacity reference omitted from the main frozen baseline loop."""
from data_boundary import HERE, ROOT, dump, file_hash, load_state
from run_phase1 import expanded_reps, run_one
from qgeognn_al.active_learning_v2.benchmark_protocol import load_features


def main():
    config={'seeds':[157,887],'budgets':[333,813],'endpoints':['V1','V2'],'method':'frozen','representation':'fusion',
            'maxiter':200,'reason':'add same-split frozen early fusion reference; no hyperparameter tuning or replacement of frozen matrix',
            'main_protocol_sha256':file_hash(HERE/'PROTOCOL.md')}
    dump(HERE/'frozen_fusion_supplement_config.json',config)
    features=load_features(ROOT/'experiments/e0_4g_baseline/canonical_4g.csv')
    for seed in config['seeds']:
        for budget in config['budgets']:
            state,data,meta,subsets=load_state(seed,budget)
            reps,_=expanded_reps(data,subsets['fit'])
            for j in range(2):
                run_one(state,data,subsets,reps,'fusion','frozen',17,j,meta,features)
    dump(HERE/'frozen_fusion_supplement_completion.json',{'complete':True,'fits':8})

if __name__=='__main__':
    main()

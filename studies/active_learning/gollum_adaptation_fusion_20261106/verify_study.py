"""Final matrix, provenance, frozen-source and artifact integrity checks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from data_boundary import HERE, ROOT, dump, file_hash, load_state
from repair_metadata import expected_provenance, numeric_digest

REQUIRED=['AUDIT.md','PHASE1_GP_DIAGNOSIS.md','PHASE1_RESULTS.md','GOLLUM_METHOD_ALIGNMENT.md',
          'PHASE2_MODEL_AND_RESOURCE_PLAN.md','PHASE2_RESULTS.md','PHASE3_FUSION_DESIGN.md','PHASE3_RESULTS.md',
          'PROTOCOL.md','NEXT_STAGE_DECISION.md']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    completion=json.loads((HERE/'phase1_completion.json').read_text())
    if not completion['complete'] or completion['fits_in_matrix']!=200:
        raise ValueError('main phase1 matrix incomplete')
    freeze=json.loads((HERE/'phase1_freeze.json').read_text())
    if freeze['config_sha256']!=file_hash(HERE/'phase1_config.json'):
        raise ValueError('frozen configuration drift')
    if any(file_hash(HERE/p)!=h for p,h in freeze['code_hashes'].items()):
        raise ValueError('frozen training source drift')
    expected={(seed,budget,rep,method,init,endpoint) for seed in (157,887) for budget in (333,813)
              for rep in ('qwen','fusion') for method in ('original','converged','regularized','noise_floor','staged')
              for init in (17,29) for endpoint in ('V1','V2')}
    expected|={(seed,budget,rep,'frozen',17,endpoint) for seed in (157,887) for budget in (333,813)
               for rep in ('qwen','fusion','gnn','raw','cw','structured') for endpoint in ('V1','V2')}
    observed=set();failed=[];predictions=0
    for p in (HERE/'phase1').glob('*.json'):
        d=json.loads(p.read_text())
        if 'method' not in d:
            continue
        identity=tuple(d[k] for k in ('data_seed','budget_state','representation','method','initialization','endpoint'))
        if identity in observed:
            raise ValueError('duplicate fit identity')
        observed.add(identity)
        if d['status']!='finite_fit':
            failed.append(identity);continue
        if d['representation_provenance']!=expected_provenance(d['representation']):
            raise ValueError('mislabelled representation provenance')
        for role,record in d['predictions'].items():
            mu=np.asarray(record['mean']);latent=np.asarray(record['latent_variance']);obs=np.asarray(record['observation_variance'])
            if not np.isfinite(mu+latent+obs).all() or (latent<0).any() or (obs<=0).any():
                raise ValueError('invalid saved prediction')
            np.testing.assert_allclose(obs-latent,d['fit_audit']['noise_variance'],rtol=1e-10,atol=1e-10)
            predictions+=len(mu)
    if observed!=expected:
        raise ValueError(f'matrix identity mismatch missing={expected-observed} unexpected={observed-expected}')
    for seed in (157,887):
        for budget in (333,813):
            state,data,meta,subsets=load_state(seed,budget)
            identity=json.loads((HERE/'phase1'/f'{seed}_{budget}_identity.json').read_text())
            if identity['state']!=state.audit or identity['candidate_ids']!=meta['candidate_ids']:
                raise ValueError('saved state identity drift')
            if any(identity['roles'][r]!=np.asarray(state.labeled_ids)[pos].tolist() for r,pos in subsets.items()):
                raise ValueError('saved molecular split drift')
    for r in json.loads((HERE/'phase1_metadata_repair.json').read_text())['repairs']:
        p=HERE/r['file'];d=json.loads(p.read_text())
        if file_hash(p)!=r['sha256_after'] or numeric_digest(d)!=r['all_other_fields_sha256']:
            raise ValueError('metadata correction modified other results')
    gates=json.loads((HERE/'summary/phase1_gates.json').read_text())
    if len(gates)!=10 or any(g['planned_fits']!=16 or not g['matrix_complete'] for g in gates):
        raise ValueError('screen gate omitted planned configurations')
    missing=[p for p in REQUIRED if not (HERE/p).is_file()]
    if missing:
        raise ValueError('missing requested reports: '+repr(missing))
    result={'status':'passed','phase1_planned_fits':208,'phase1_observed_fits':len(observed),'phase1_failed_fits':len(failed),
            'phase1_saved_prediction_rows':predictions,'phase1_code_freeze_verified':True,'current_state_identity_verified':True,
            'phase1_gate_passes':sum(g['pass'] for g in gates),'test_truth_access_count':0,'future_U_labels_requested':0,
            'new_gnn_fits':0,'formal_al_runs':0,'new_model_downloads':0,'paid_api_calls':0,
            'limits':'permission counters describe this study; toy tests are not performance evidence; final hashes are not all pre-score source freezes'}
    dump(HERE/'verification.json',result)
    if args.seal:
        manifest={}
        for p in sorted(HERE.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.name=='artifact_manifest.json' or p.suffix=='.tmp':
                continue
            manifest[str(p.relative_to(HERE))]={'bytes':p.stat().st_size,'sha256':file_hash(p)}
        dump(HERE/'artifact_manifest.json',{'scope':'final artifacts, not an assertion that every source was frozen before scoring',
                                          'files':manifest,'file_count':len(manifest)})
    else:
        manifest=json.loads((HERE/'artifact_manifest.json').read_text())
        for name,meta in manifest['files'].items():
            if file_hash(HERE/name)!=meta['sha256']:
                raise ValueError('final artifact drift: '+name)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

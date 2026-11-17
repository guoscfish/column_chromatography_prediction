#!/usr/bin/env python3
"""Independent scalar checks on fixed first/phase-end/final rounds.

No import of analyze_model_signals and no raw label store. This is intentionally
a different summation/grouping path using only accepted archived feedback and
its verified premeasurement catalog.
"""
import csv, gzip, hashlib, itertools, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
RUN=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
SEEDS=[157,887,2357,6101,12203]
ROUNDS=[0,5,14,20,35]
keys=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
with (OUT/'tables/model_batch_metrics.csv').open() as f:
    derived={(int(x['seed']),int(x['round'])):x for x in csv.DictReader(f)}
with (OUT/'tables/model_same_batch_single_factor_response.csv').open() as f:
    pair_derived=list(csv.DictReader(f))
checks=[]; pair_counts={}
for seed in SEEDS:
    archived={}
    p=ROOT/f'reports/free_cw_final_20261112/scientific_artifacts/seed_{seed}_results.jsonl.gz'
    with gzip.open(p,'rt') as f:
        for l in f:
            d=json.loads(l)
            if d['path'].endswith(('/feedback.json','/contract.json')):
                assert hashlib.sha256(d['raw_utf8'].encode()).hexdigest()==d['sha256']
                archived[d['path']]=json.loads(d['raw_utf8'])
    for r in range(36):
        base=RUN/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}';rel=str(base.relative_to(ROOT))
        con=archived[rel+'/contract.json']; feedback=archived[rel+'/feedback.json']; raw=(base/'catalog.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==con['catalog_sha256']; cat=json.loads(raw)
        candidates={x['id']:x for x in cat['candidates']+cat['pending']}
        selected=[candidates[x['candidate_id']] for x in feedback['records']]
        phase='333-525' if r<6 else '525-813' if r<15 else '813-1005' if r<21 else '1005-1485'
        for a,b in itertools.combinations(selected,2):
            if a['smiles']!=b['smiles']:continue
            diff=[k for k in keys if a['conditions'][k]!=b['conditions'][k]]
            if len(diff)==1:
                k=(seed,phase,diff[0]);pair_counts[k]=pair_counts.get(k,0)+1
        if r not in ROUNDS:continue
        for t in [1,2]:
            errors=[]; covered=0; negative=0
            for rec in feedback['records']:
                q=candidates[rec['candidate_id']];e=rec[f'pred_V{t}_ml']-rec[f'true_V{t}_ml'];errors.append(e)
                covered+=int(q[f'pred_V{t}_q10_ml']<=rec[f'true_V{t}_ml']<=q[f'pred_V{t}_q90_ml'])
                negative+=int(q[f'pred_V{t}_q90_ml']<q[f'pred_V{t}_q10_ml'])
            n=len(errors); row=derived[seed,r]
            for name,val in {'rmse_ml':math.sqrt(math.fsum(e*e for e in errors)/n),
                             'mae_ml':math.fsum(abs(e) for e in errors)/n,
                             'bias_pred_minus_true_ml':math.fsum(errors)/n,
                             'coverage80_raw':covered/n,'negative_width_rate':negative/n}.items():
                assert math.isclose(float(row[f'selected_v{t}_{name}']),val,rel_tol=1e-11,abs_tol=1e-11),(seed,r,t,name)
            checks.append({'seed':seed,'round':r,'target':f'V{t}','n':n,'scalar_metric_checks':5,'status':'PASS'})
for row in pair_derived:
    k=(int(row['seed']),row['phase'],row['axis'])
    assert int(row['n_pairs'])==pair_counts[k]
assert sum(pair_counts.values())==893
result={'status':'PASS','fixed_sampling_rule':'Every seed × rounds 0,5,14,20,35; selected before calculation',
        'batches_scalar_checked':25,'endpoint_checks':len(checks),'scalar_checks':sum(x['scalar_metric_checks'] for x in checks),
        'all_180_batches_pair_counts_independently_rebuilt':True,'single_factor_pair_events':sum(pair_counts.values()),
        'checks':checks,'labels_source':'accepted feedback only; catalogs match archived contract SHA',
        'not_a_test_of_causal_validity':True}
(OUT/'verification/model_independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

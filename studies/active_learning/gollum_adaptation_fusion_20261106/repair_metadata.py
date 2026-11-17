"""Repair only a provenance label, retaining before/after and numeric hashes."""
import hashlib
import json
from data_boundary import HERE, dump, file_hash


def expected_provenance(representation):
    if representation=='qwen':
        return 'frozen_unsupervised_Qwen'
    if representation=='structured':
        return 'structured_response_independent_covariates'
    return 'historical_GNN_saw_all_Lt_conditional_GP_only'


def numeric_digest(record):
    retained={k:v for k,v in record.items() if k!='representation_provenance'}
    return hashlib.sha256(json.dumps(retained,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    path=HERE/'phase1_metadata_repair.json'
    previous=json.loads(path.read_text()) if path.exists() else {'repairs':[]}
    rows=previous['repairs']
    for p in sorted((HERE/'phase1').glob('*.json')):
        d=json.loads(p.read_text())
        if 'representation_provenance' not in d:
            continue
        expected=expected_provenance(d['representation'])
        if d['representation_provenance']==expected:
            continue
        before=file_hash(p);old=d['representation_provenance'];digest=numeric_digest(d)
        d['representation_provenance']=expected
        dump(p,d)
        if digest!=numeric_digest(json.loads(p.read_text())):
            raise ValueError('metadata repair changed numerical results')
        rows.append({'file':str(p.relative_to(HERE)),'old_value':old,'new_value':expected,
                     'sha256_before':before,'sha256_after':file_hash(p),'all_other_fields_sha256':digest})
    dump(path,{'reason':'run_one assigned all non-Qwen metadata to historical GNN; structured covariates are response-independent',
               'scope':'metadata_only_no_model_prediction_metric_or_selection_changes','repairs':rows,
               'frozen_runner_unchanged':True,'rerun_note':'apply this idempotent repair after the frozen runner, before final reporting'})

if __name__=='__main__':
    main()

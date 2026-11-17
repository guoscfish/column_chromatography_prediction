#!/usr/bin/env python3
"""Seal or recheck this isolated study; never start a fit or contact a provider."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,ast,io,tokenize
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
DOCS=['README.md','MODEL_ARCHITECTURE_AUDIT.md','VOLUME_SENSITIVITY.md','MODEL_COMPARISON.md','RETRY_CONTEXT_AUDIT.md','PREFIX_FAILURE_ANALYSIS.md','LOCAL_REPAIR_DESIGN.md','LLM_MODEL_DISCOVERY_ROADMAP.md','FINAL_DECISION.md']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text())

def main(check=False):
    if check:
        seal=load(OUT/'VERIFICATION.json')
        for entry in seal['inputs']+seal['outputs']:
            assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
        assert seal['status']=='PASS'
        print(json.dumps(dict(status='PASS',inputs=len(seal['inputs']),outputs=len(seal['outputs']),new_fits=seal['new_fits'],new_llm_calls=0)))
        return
    for name in DOCS:assert (OUT/name).is_file(),name
    gate=load(OUT/'DATA_FEASIBILITY.json');assert gate['status']=='PASS'
    for name,digest in gate['training_code_sha256'].items():assert sha(OUT/name)==digest
    planned={(v['seed'],v['kind']) for v in gate['planned_fits']}
    starts=list((OUT/'fits').glob('seed_*/*/started.json'));audits=list((OUT/'fits').glob('seed_*/*/fit_audit.json'))
    assert len(starts)==len(audits)==len(planned)==15
    assert not list((OUT/'fits').glob('seed_*/*/FAILED.json'))
    assert {(load(p)['seed'],load(p)['kind']) for p in starts}==planned
    assert all(load(p)['data_gate_sha256']==sha(OUT/'DATA_FEASIBILITY.json') for p in starts)
    actual=set();details=[]
    for p in audits:
        a=load(p);d=load(OUT/f'data/seed_{a["seed"]}/identity.json');dest=p.parent
        actual.add((a['seed'],a['kind']));assert a['status']=='COMPLETE'
        assert not a['holdout_used_for_early_stopping'] and not a['test_truth_used'] and not a['unobserved_targets_used']
        assert a['training_config']==d['training_config'] and a['initialization_seed']==d['initialization_seed']
        for key in ['train_ids_hash','check_ids_hash','validation_ids_hash']:assert a[key]==d[key]
        assert sha(dest/'best.pt')==a['checkpoint_sha256']
        history=pd.read_csv(dest/'history.csv');minimum=history.loc[history.validation_selection_score.idxmin()]
        assert len(history)==a['epochs_run'] and int(minimum.epoch)==a['best_epoch']
        assert abs(float(minimum.validation_selection_score)-a['best_validation_nrmse'])<1e-12
        assert a['epochs_run']==d['training_config']['maximum_epochs'] or a['epochs_run']-a['best_epoch']==d['training_config']['patience']
        for scope in ['validation','check']:
            pred=pd.read_csv(dest/f'{scope}_predictions.csv');assert pred.sample_id.tolist()==d[scope+'_ids']
            values=pred.drop(columns='sample_id').to_numpy(float)
            assert values.shape==(len(d[scope+'_ids']),6) and np.isfinite(values).all() and (values>=0).all()
        expected=next(v for v in gate['pretraining_checks'] if v['seed']==a['seed'])
        assert a['post_init_torch_rng_hash']==expected['post_init_torch_rng_hash'][a['kind']]
        assert a['common_initial_hash']==expected['common_initial_hash'][a['kind']]
        details.append({k:a[k] for k in ['seed','kind','best_epoch','epochs_run','checkpoint_sha256','parameter_count']})
    assert actual==planned
    sources={}
    for name in ['DATA_FEASIBILITY.json','sensitivity_manifest.json','sensitivity_summary_manifest.json','molecule_summary_manifest.json','retry_audit_inputs.json','local_repair_inputs.json']:
        manifest=load(OUT/name)
        for entry in manifest.get('sources',manifest.get('inputs',[])):
            path=entry['path'];digest=entry['sha256']
            assert path not in sources or sources[path]==digest,('conflicting input identities',path)
            sources[path]=digest
    sources['experiments/e0_4g_baseline/canonical_4g.csv']=gate['canonical_covariate_source_sha256']
    for seed in sorted({s for s,_ in planned}):
        identity=load(OUT/f'data/seed_{seed}/identity.json')
        partition=pd.read_csv(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv')
        outer=set(partition.loc[partition.role.isin(['l0','u0']),'sample_id'])
        history=set(identity['history_ids']);train=set(identity['train_ids']);held=set(identity['check_ids'])
        assert history<=outer and history==train|held and not train&held
        assert identity['validation_ids']==partition.loc[partition.role=='validation','sample_id'].tolist()
        assert not set(partition.loc[partition.role=='test','sample_id'])&(history|set(identity['validation_ids']))
        linked={identity['graphs_path']:identity['graphs_sha256'],
            str((OUT/f'data/seed_{seed}/metadata.csv').relative_to(ROOT)):identity['metadata_sha256'],
            str((OUT/f'data/seed_{seed}/authorized_labels.npz').relative_to(ROOT)):identity['authorized_labels_sha256']}
        for path,digest in linked.items():
            assert path not in sources or sources[path]==digest,('linked identity conflict',path)
            sources[path]=digest
    for path,digest in sources.items():assert sha(ROOT/path)==digest,('source changed',path)
    qa=load(OUT/'training_independent_qa.json')
    assert qa['status']=='PASS_WITH_FIXED_VALIDATION_ASSOCIATIONS'
    for row in qa['per_seed']:
        assert row['training_same_series_overlap_n']==row['training_exact_condition_overlap_n']==0
        assert row['fixed_target_scales_match_historical'] and row['original_training_config_matches']
    for path,digest in qa['source_sha256'].items():assert sha(ROOT/path)==digest,('QA source changed',path)
    results_qa=load(OUT/'RESULTS_INDEPENDENT_QA.json')
    assert results_qa['status']=='PASS' and results_qa['new_fits']==results_qa['model_inference_calls']==results_qa['api_calls']==0
    assert sha(OUT/'results_independent_qa.py')==results_qa['qa_script_sha256']
    for entry in results_qa['input_sources']:
        path,digest=entry['path'],entry['sha256'];assert sha(ROOT/path)==digest,('results QA input changed',path)
        assert path not in sources or sources[path]==digest
        sources[path]=digest
    repair=load(OUT/'local_repair_verification.json')
    assert repair['retained_original_unique_legal']==repair['original_unique_legal']==3634
    assert repair['vacancies']==430 and repair['ready_after_local_draft']==0 and repair['final_historical_validator_failures']==0
    retry=load(OUT/'retry_audit_inputs.json');assert retry['batches']==540 and retry['prefix_batches']==127 and retry['new_API_calls']==0
    tests=load(OUT/'test_result.json');assert tests['exit_code']==0
    assert sha(OUT/'test_stdout.txt')==tests['stdout_sha256']
    for path,digest in tests['tests_sha256'].items():assert sha(ROOT/path)==digest,('tests changed after run',path)
    code_paths=list(OUT.rglob('*.py'))
    for path in code_paths:
        source=path.read_text();ast.parse(source,filename=str(path))
        assert all(token.string.isascii() for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type==tokenize.COMMENT)
    outputs=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name in ['VERIFICATION.json'] or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
        outputs.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    seal=dict(status='PASS',git_head=revision,git_status=status,new_fits=15,maximum_authorized_fits=15,new_llm_calls=0,
        full_AL_rollouts=0,test_or_unrevealed_numeric_target_values_projected=0,
        label_boundary='H1005 own observed catalogs; original validation via role-gated streaming numeric projection. CSV strings and hashed bytes pass through the file reader; no test/unrevealed numeric targets projected, returned or scored.',
        dedicated_check_labels_used_for_selection=False,validation_overlap_limitation_disclosed=True,
        tests=tests,static_syntax_pass_scripts=len(code_paths),all_new_code_comments_ascii=True,
        independent_results_qa='PASS: 30 prediction files, nine table comparisons at tolerance 2e-12; no model inference or fitting',
        per_seed_numeric_target_roles_restricted_to_history_and_original_validation=True,
        fit_receipts=sorted(details,key=lambda v:(v['seed'],v['kind'])),
        inputs=[dict(path=p,sha256=h) for p,h in sorted(sources.items())],outputs=outputs,
        scope='Checks local artifacts and recorded client payloads; cannot attest remote provider internals or physical causality.')
    (OUT/'VERIFICATION.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',inputs=len(sources),outputs=len(outputs),new_fits=15,new_llm_calls=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)

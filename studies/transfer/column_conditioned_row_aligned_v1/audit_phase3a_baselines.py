"""Metadata/identity-only reference eligibility planning; no endpoint or score reads."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase3a as s


def audit():
    schedule=s.p.row_schedule()
    registers=[]
    benchmark=s.old.PARENT
    global_freeze=s.read_json(benchmark/'all_predictions_frozen.json')
    s.require(global_freeze['protocol_sha256']==s.old.sha(benchmark/'protocol.json'),'baseline protocol changed')
    for seed in s.SEEDS:
        for c in s.TARGETS:
            part=schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)]
            roles={role:part.loc[part.role.eq(role),'sample_id'].astype(str).tolist() for role in ['gradient_train','validation','test']}
            hashes={role:s.old.digest(sorted(ids)) for role,ids in roles.items()}
            run=benchmark/f'runtime/contexts/{c}/row/seed_{seed}'
            frozen=s.read_json(run/'frozen.json')
            rel=str((run/'frozen.json').relative_to(benchmark))
            s.require(global_freeze['files'][rel]==s.old.sha(run/'frozen.json'),'scale freeze receipt changed')
            for name,digest in frozen['files'].items():
                s.require(s.old.sha(run/name)==digest,'scale frozen file changed')
            contract=frozen['contract']
            for role in roles:
                s.require(contract[role+'_ids_hash']==hashes[role], 'scale role identity mismatch')
            ids=pd.read_csv(run/'predictions_blind.csv.gz',usecols=['sample_id'],dtype=str).sample_id.tolist()
            s.require(ids==roles['test'],'scale prediction identity mismatch')
            registers.append(dict(column=c,outer_seed=seed,method='scale_only',eligibility='MATCHED_FULL_ROW_WORKFLOW_REFERENCE',
                                  prediction_path=str((run/'predictions_blind.csv.gz').relative_to(s.ROOT)),
                                  prediction_sha256=s.old.sha(run/'predictions_blind.csv.gz'),
                                  prediction_usecols=['sample_id','scale_only_V1','scale_only_V2'],
                                  gradient_train_count=len(roles['gradient_train']),test_count=len(ids),
                                  gradient_train_ids_sha256=hashes['gradient_train'],test_ids_sha256=hashes['test'],
                                  selection='no hyperparameter/outer selector; train-only scale fit',
                                  note='4g is source-pretraining only; A1/A2 add source_train auxiliary supervision; workflow comparison'))
    traditional=s.ROOT/'studies/transfer/traditional_transfer_converged_baseline_v1'
    manifest=s.read_json(traditional/'PREDICTION_FREEZE_MANIFEST.json')
    protocol=s.read_json(traditional/'PROTOCOL.json')
    s.require(manifest['protocol_sha256']==s.old.sha(traditional/'PROTOCOL.json')
              and protocol['source_checkpoint_sha256']==s.old.SOURCE_SHA
              and protocol['split_manifest_sha256']==s.old.SPLIT_SHA,'traditional reference authority mismatch')
    for seed in s.SEEDS:
        for c in s.TARGETS:
            part=schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)]
            roles={role:part.loc[part.role.eq(role),'sample_id'].astype(str).tolist() for role in ['gradient_train','validation','test']}
            run=traditional/f'runtime/{c}/seed_{seed}'
            rel=str((run/'context_fit_complete.json').relative_to(traditional))
            s.require(manifest['context_fit_complete_sha256'][rel]==s.old.sha(run/'context_fit_complete.json'),'traditional freeze receipt changed')
            receipt=s.read_json(run/'context_fit_complete.json')
            for name,digest in receipt['files'].items():
                s.require(s.old.sha(run/name)==digest,'traditional frozen file changed')
            context=s.read_json(run/'context_audit.json')
            for role,ids in roles.items():
                s.require(context['roles'][role]['ids_hash']==s.old.digest(sorted(ids)) and context['roles'][role]['count']==len(ids),'traditional role mismatch')
            for method in ['P0','P1']:
                path=run/method/'test_predictions_blind.csv.gz'
                ids=pd.read_csv(path,usecols=['sample_id'],dtype=str).sample_id.tolist()
                s.require(ids==roles['test'],'traditional prediction identity mismatch')
                registers.append(dict(column=c,outer_seed=seed,method=method,eligibility='MATCHED_FULL_ROW_DIFFERENT_INNER_CV_REFERENCE',
                                      prediction_path=str(path.relative_to(s.ROOT)),prediction_sha256=s.old.sha(path),
                                      selection='GroupKFold five-fold medians, fixed FULL refit; not current Row-CV selection',
                                      gradient_train_count=len(roles['gradient_train']),test_count=len(ids),
                                      gradient_train_ids_sha256=s.old.digest(sorted(roles['gradient_train'])),test_ids_sha256=s.old.digest(sorted(ids))))
    hier=s.ROOT/'studies/transfer/hier_cw_semantic_repair'
    manifest=s.read_json(hier/'prediction_freeze_manifest.json')
    mismatches=[]
    for rel,digest in manifest['files'].items():
        actual=s.old.sha(s.ROOT/rel)
        if actual!=digest:
            mismatches.append({'path':rel,'expected_sha256':digest,'actual_sha256':actual})
    s.require(manifest['source_checkpoint_sha256']==s.old.SOURCE_SHA
              and manifest['split_manifest_sha256']==s.old.sha(hier/'split_manifest.csv'),'HIER authority mismatch')
    hierarchy_split=pd.read_csv(hier/'split_manifest.csv',usecols=['column','protocol','outer_seed','role','sample_id'],dtype={'sample_id':str})
    for seed in s.SEEDS:
        target=schedule.loc[schedule.outer_seed.eq(seed)&schedule.role.eq('gradient_train')]
        train_union=s.old.digest(sorted(target.sample_id.astype(str)))
        for c in s.TARGETS:
            for role in ['gradient_train','validation','test']:
                expected=schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)&schedule.role.eq(role),'sample_id'].astype(str).tolist()
                actual=hierarchy_split.loc[hierarchy_split.outer_seed.eq(seed)&hierarchy_split.protocol.eq('row')&hierarchy_split.column.eq(c)&hierarchy_split.role.eq(role),'sample_id'].tolist()
                s.require(len(actual)==len(set(actual))==len(expected) and set(actual)==set(expected),'HIER focal role mismatch')
            run=hier/f'runtime/contexts/{c}/row/seed_{seed}'
            context=s.read_json(run/'fit_audit.json')
            s.require(context['train_ids_sha256']==train_union and not context['outer_validation_used'] and not context['test_truth_used'],'HIER training scope mismatch')
            path=run/'predictions_blind.csv.gz'
            ids=pd.read_csv(path,usecols=['sample_id'],dtype=str).sample_id.tolist()
            expected=schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)&schedule.role.eq('test'),'sample_id'].astype(str).tolist()
            s.require(len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected),'HIER prediction identity mismatch')
            registers.append(dict(column=c,outer_seed=seed,method='HIER_CW_SHARED_LAMBDA_CORRECTED',
                                  eligibility='EXCLUDED_FROZEN_PROVENANCE_MISMATCH' if mismatches else 'MATCHED_FULL_FOCAL_ROW_DIFFERENT_INNER_SPLITTER_REFERENCE',
                                  prediction_path=str(path.relative_to(s.ROOT)),prediction_sha256=s.old.sha(path),
                                  prediction_usecols=['sample_id','HIER_CW_SHARED_LAMBDA_CORRECTED_V1','HIER_CW_SHARED_LAMBDA_CORRECTED_V2'],
                                  test_count=len(ids),test_ids_sha256=s.old.digest(sorted(ids)),gradient_train_union_sha256=train_union,
                                  selection='pooled Row KFold seeded by outer seed; differs from per-column sorted KFold seed=42',
                                  note='full split file includes 8g and different row order; byte hash differs; focal role sets/two-column fitted union match; sample_id alignment required; model fits only 25g/40g'))
    payload={'scope':'metadata, file hashes and sample_id only; no observed outer responses or score tables',
             'reference_contexts_verified':sum(not row['eligibility'].startswith('EXCLUDED') for row in registers),
             'reference_contexts_inspected':len(registers),'frozen_provenance_mismatches':mismatches,
             'outer_response_cells_parsed':0,'reference_scores_computed':False,
             'references':registers,'excluded':[
                 {'method':'Affine','reason':'only historical B=100/raw-population references found; no compatible FULL Row freeze identified'},
                 {'method':'paper_style_current_v2 / conditional_EA / target_only_full','reason':'historical outer-validation-based selector differs from internal-only model selection'}]
                 +([{'method':'HIER_CW_SHARED_LAMBDA_CORRECTED','reason':'frozen provenance file SHA256 mismatch; do not compare or regenerate historical artifacts','mismatches':mismatches}] if mismatches else []),
             'evaluation_rule':'reverify these hashes before Phase 3B; recompute using Phase 3A FULL gradient_train scales/tail masks; complete workflow comparisons, no architecture-only causal claim'}
    s.old.write_json(s.STUDY/'PHASE3A_BASELINE_COMPATIBILITY.json',payload,immutable=True)
    return {'reference_contexts_verified':payload['reference_contexts_verified'],'HIER_excluded':bool(mismatches),'scores_computed':False}


if __name__=='__main__':
    print(audit())

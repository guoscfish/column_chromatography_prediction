"""Independent pair-count and frozen prediction-difference QA; no target reads."""
from pathlib import Path
import numpy as np
import pandas as pd
import predictor_response_diagnostics as main
import study_support as s

out=main.OUT
pairs=pd.read_csv(out/'input_matched_pairs.csv',dtype={'sample_id_a':str,'sample_id_b':str})
frame=main.feature_frame()
assert all(main.ea_fraction(v)==.5 for v in ('1/1','5/5','10/10','.5/.5'))
assert all(main.ea_fraction(v)==.25 for v in ('3/1','6/2','1.5/.5'))
aliases=frame.groupby('EA_fraction')['PE/EA'].nunique()
alias_groups=int((aliases>1).sum())
normalized_equal_pair_checks=0
max_error=0.
checks=[]
for seed in main.SEEDS:
    split=s.partition(seed)
    allowed=split.loc[split.role.isin(['l0','u0']),'canonical_index'].to_numpy(int)
    inputs=frame.iloc[allowed]
    z=np.load(out/f'predictions_seed_{seed}.npz',allow_pickle=False)
    pred=pd.DataFrame(z['q50'].astype(np.float64),index=z['canonical_indices'],columns=['V1','V2'])
    scales=s.read(out/f'predictions_seed_{seed}.json')['input']['target_scales']
    for factor,active in main.FACTOR_COLUMNS.items():
        fixed=['canonical_smiles']+[c for c in main.CONDITIONS if c!=active]
        n=inputs.groupby(fixed).size().to_numpy()
        all_combinations=int(np.sum(n*(n-1)//2))
        levels=inputs.groupby(fixed+[active]).size().to_numpy()
        equal_level=int(np.sum(levels*(levels-1)//2))
        expected=all_combinations-equal_level
        g=pairs.loc[pairs.seed.eq(seed)&pairs.factor.eq(factor)]
        assert len(g)==expected
        a=g.canonical_index_a.to_numpy(int);b=g.canonical_index_b.to_numpy(int)
        assert set(a)<=set(allowed) and set(b)<=set(allowed)
        assert not g.duplicated(['sample_id_a','sample_id_b']).any()
        for column in fixed:
            assert np.all(frame.iloc[a][column].to_numpy()==frame.iloc[b][column].to_numpy())
        assert np.all(frame.iloc[a][active].to_numpy()!=frame.iloc[b][active].to_numpy())
        difference=pred.loc[b].to_numpy()-pred.loc[a].to_numpy()
        saved=g[['pred_delta_V1_ml','pred_delta_V2_ml']].to_numpy()
        err=float(np.max(np.abs(difference-saved))) if len(g) else 0.
        max_error=max(max_error,err)
        np.testing.assert_allclose(difference,saved,atol=1e-12,rtol=0)
        normalized=np.abs(difference)/np.array([scales['V1'],scales['V2']])
        np.testing.assert_allclose(normalized,g[['pred_abs_delta_V1_L333std','pred_abs_delta_V2_L333std']],atol=1e-12,rtol=0)
        if factor=='eluent_EA_fraction':
            assert np.all(frame.iloc[a]['EA_fraction'].to_numpy()!=frame.iloc[b]['EA_fraction'].to_numpy())
            normalized_equal_pair_checks+=len(g)
        checks.append({'seed':seed,'factor':factor,'pairs':expected,'independent_count_formula_passed':True,'metadata_match_passed':True})
summary_path=out/'selected_pair_summary.csv'
try:
    selected_summary=pd.read_csv(summary_path)
except pd.errors.EmptyDataError:
    selected_summary=pd.DataFrame()
selection_checks=0
for rec in selected_summary.itertuples():
    directory=main.HERE/f'acquisition/seed_{rec.seed}_{rec.family}_P1_M1024'
    batch=set(map(str,s.read(directory/'batches.json')[rec.method]['selected_row_ids']))
    split=s.partition(rec.seed)
    history=set(split.loc[split.role.eq('l0'),'sample_id'].astype(str))
    rows=pairs.loc[pairs.seed.eq(rec.seed)&pairs.factor.eq(rec.factor)]
    # Direct Python set predicates are independent of production pandas masks.
    keep=[]
    for pair in rows.itertuples():
        a,b=pair.sample_id_a,pair.sample_id_b
        if rec.scope=='selected_within_batch':
            use=a in batch and b in batch
        elif rec.scope=='selected_to_L333':
            use=(a in batch and b in history) or (b in batch and a in history)
        else:
            use=a in batch or b in batch
        if use:keep.append(pair)
    assert len(keep)==rec.pairs
    if keep:
        retained=pd.DataFrame([p._asdict() for p in keep])
        assert abs(retained.groupby('series_id').pred_combined_delta_L333std.mean().mean()-rec.series_equal_mean_combined_delta_L333std)<1e-12
        assert abs(retained.groupby('molecule_id').pred_combined_delta_L333std.mean().mean()-rec.molecule_equal_mean_combined_delta_L333std)<1e-12
        assert abs(np.mean([p.pred_combined_delta_L333std for p in keep])-rec.mean_combined_delta_L333std)<1e-12
    selection_checks+=1
s.dump(out/'independent_qa.json',{'status':'PASS','qa_script_sha256':s.digest(Path(__file__)),
    'pairs':len(pairs),'seed_factor_checks':checks,'selection_summary_checks':selection_checks,
    'maximum_prediction_difference_error_ml':max_error,'equivalent_eluent_synthetic_tests_passed':True,'actual_EA_levels_with_multiple_ratio_spellings':alias_groups,'eluent_pairs_checked_different_actual_fraction':normalized_equal_pair_checks,'target_reads':0,'new_fits':0,
    'method':'pair counts by nChoose2 minus equal-level nChoose2; exact metadata checks; direct indexed float64 difference of original saved float32 q50; independent Python set selection predicates',
    'inputs':{p.name:s.digest(p) for p in [out/'input_matched_pairs.csv',summary_path]}})
artifacts=[p for p in out.iterdir() if p.is_file() and p.name!='analysis_manifest.json']
artifacts += [main.HERE/'PREDICTOR_MATCH_DIAGNOSTICS.md',Path(main.__file__),Path(__file__)]
s.dump(out/'analysis_manifest.json',{'schema':'post-freeze-independent-analysis-manifest-v1',
    'science_freeze_sha256':s.digest(main.HERE/'freeze.json'),
    'science_implementation_modified':False,'new_neural_fits':0,'target_reads':0,
    'artifacts_sha256':{str(p.relative_to(s.ROOT)):s.digest(p) for p in artifacts}})
print('INDEPENDENT_QA_PASS',len(pairs),'pairs;',selection_checks,'selection scopes;',max_error,'mL maximum error')

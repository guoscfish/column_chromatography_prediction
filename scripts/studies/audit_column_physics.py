#!/usr/bin/env python3
"""Metadata census plus explicitly training-only normalization gate; no model fit."""
import json
from pathlib import Path
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from rdkit import Chem
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.transfer.column_physics import packing_mass, context_matrix
from src.qgeognn_al.transfer.scaling_audit import match_key
from src.qgeognn_al.artifacts import sha256_file
from scripts.studies.run_scaling_failure_audit import selected_truth, FEATURES
OUT = ROOT/'studies/transfer/physics_column_conditioned_transfer'
OLD = ROOT/'studies/transfer/cross_column'
SOURCE = ROOT/'experiments/e0_4g_baseline/canonical_4g.csv'
COLS = ['4g','8g','25g','40g']

def table(df):
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'audit_frozen.json').exists():
        raise RuntimeError('audit already frozen; do not overwrite')
    frames, census, flows, pairs = {}, [], [], []
    for c in COLS:
        path = ROOT/'dataset'/f'dataset_{c}.csv'
        d = pd.read_csv(path, usecols=['smiles','column_specs',*FEATURES[2:]])
        def canonical(s):
            m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
            return Chem.MolToSmiles(m) if m is not None else None
        d['canonical_smiles'] = d.smiles.map(canonical)
        x = context_matrix(d)
        d['EA_fraction'] = d['PE/EA'].map(lambda s: float(s.split('/')[1])/sum(map(float,s.split('/'))))
        d['loading_mass_mg'] = x[:,2]*x[:,0]
        frames[c] = d
        census.append(dict(column=c, rows=len(d), compounds=d.canonical_smiles.nunique(), invalid_smiles=d.canonical_smiles.isna().sum(), column_specs=';'.join(sorted(d.column_specs.unique())), packing_mass_g=x[0,0], EA_min=d.EA_fraction.min(), EA_max=d.EA_fraction.max(), loading_mass_min_mg=d.loading_mass_mg.min(), loading_mass_max_mg=d.loading_mass_mg.max(), loading_solvent_min_ul=d['Volume of loading solvent/ul'].min(), loading_solvent_max_ul=d['Volume of loading solvent/ul'].max()))
        for f,g in d.groupby('Flow mL/min'):
            flows.append(dict(column=c,flow_ml_min=f,rows=len(g),fraction=len(g)/len(d),compounds=g.canonical_smiles.nunique()))
    for a,b in combinations(COLS,2):
        record = dict(source=a,target=b,shared_compounds=len(set(frames[a].canonical_smiles.dropna())&set(frames[b].canonical_smiles.dropna())))
        for relaxed in [False,True]:
            keys = {match_key(r,relaxed=relaxed) for r in frames[a].dropna(subset='canonical_smiles').to_dict('records')}
            record['relaxed_rows' if relaxed else 'exact_rows'] = sum(match_key(r,relaxed=relaxed) in keys for r in frames[b].dropna(subset='canonical_smiles').to_dict('records'))
        pairs.append(record)
    census,flows,pairs = map(pd.DataFrame,[census,flows,pairs])
    for name,d in [('metadata_census',census),('flow_distribution',flows),('raw_pair_coverage',pairs)]:
        d.to_csv(OUT/f'{name}.csv',index=False)
    # Frozen first-seed budget100 gradient labels only: never union overlapping seed roles.
    schedule = pd.read_csv(OLD/'splits/schedule_manifest.csv')
    source_split = pd.read_csv(ROOT/'studies/predictor/final_4g_qualification/splits/row_seed_42.csv')
    source_ids = source_split.loc[source_split.split.eq('train'),'sample_id'].tolist()
    training,ids_record = {},{}
    for c in COLS:
        p = SOURCE if c=='4g' else OLD/'data_audit'/f'canonical_{c}.csv'
        ids = source_ids if c=='4g' else schedule.loc[schedule.column.eq(c)&schedule.protocol.eq('compound')&schedule.outer_seed.eq(769539383)&schedule.planned_budget.eq(100)&schedule.role.eq('gradient_train'),'sample_id'].tolist()
        d = pd.read_csv(p,usecols=[*FEATURES,'column_specs']).set_index('sample_id').loc[ids].reset_index()
        y = selected_truth(p,ids)
        d['V1'],d['V2'] = y[:,0],y[:,1]
        d['center'],d['width'] = y.mean(1),y[:,1]-y[:,0]
        d['EA_bin'] = pd.cut(d['PE/EA'].map(lambda s: float(s.split('/')[1])/sum(map(float,s.split('/')))),[-.01,.1,.5,1.01],labels=['low','mid','high'])
        d['packing_mass_g'] = d.column_specs.map(packing_mass)
        training[c]=d
        ids_record[c]=ids
    distributions,discrepancy,matched = [],[],[]
    for c,d in training.items():
        for target in ['V1','V2','center','width']:
            for norm in [False,True]:
                v = d[target]/d.packing_mass_g if norm else d[target]
                for dim in ['overall','EA_bin','Flow mL/min']:
                    groups = [('all',d.index)] if dim=='overall' else [(str(k),g.index) for k,g in d.groupby(dim,observed=True)]
                    for level,idx in groups:
                        z=v.loc[idx]
                        distributions.append(dict(column=c,target=target,normalization='mass_normalized' if norm else 'raw',dimension=dim,level=level,n=len(z),mean=z.mean(),median=z.median(),std=z.std(),cv=z.std()/abs(z.mean()) if abs(z.mean())>1e-12 else np.nan))
    dist=pd.DataFrame(distributions)
    for target in ['V1','V2','center','width']:
        for c in COLS[1:]:
            for norm in [False,True]:
                a,b=training['4g'].copy(),training[c].copy()
                if norm:
                    a[target]/=a.packing_mass_g;b[target]/=b.packing_mass_g
                for keys in [['canonical_smiles'],['canonical_smiles','EA_bin'],['canonical_smiles','Flow mL/min']]:
                    aa=a.groupby(keys,observed=True)[target].mean();bb=b.groupby(keys,observed=True)[target].mean()
                    joined=pd.concat([aa.rename('source'),bb.rename('target')],axis=1).dropna()
                    relative=(joined.target-joined.source).abs()/joined.source.abs().clip(lower=.5/(4 if norm else 1))
                    matched.append(dict(column=c,target=target,normalization='mass_normalized' if norm else 'raw',matching='+'.join(keys),groups=len(joined),mean_absolute_difference=(joined.target-joined.source).abs().mean(),median_relative_discrepancy=relative.median()))
        for norm in ['raw','mass_normalized']:
            means=dist.loc[dist.target.eq(target)&dist.normalization.eq(norm)&dist.dimension.eq('overall'),'mean']
            discrepancy.append(dict(target=target,normalization=norm,cross_column_mean_cv=means.std()/abs(means.mean())))
    matched=pd.DataFrame(matched); discrepancy=pd.DataFrame(discrepancy)
    for name,d in [('training_target_distributions',dist),('training_matched_discrepancy',matched),('normalization_discrepancy',discrepancy)]: d.to_csv(OUT/f'{name}.csv',index=False)
    scales=pd.read_csv(ROOT/'studies/transfer/scaling_failure_audit/scale_stability_by_seed.csv')
    print('scale columns',scales.columns.tolist())
    scales['packing_mass_ratio']=scales.column.map({'8g':2.,'25g':6.25,'40g':10.})
    coeff='scale' if 'scale' in scales else 'coefficient'
    print(scales.head().to_string(index=False))
    # Persist source coefficients unchanged; derived differences are descriptive only.
    candidates=[n for n in ['scale','coefficient','fitted_scale','a'] if n in scales]
    if not candidates: raise ValueError('coefficient schema must be inspected')
    scales['absolute_difference']=abs(scales[candidates[0]]-scales.packing_mass_ratio)
    scales['relative_difference']=scales.absolute_difference/scales.packing_mass_ratio
    scales.to_csv(OUT/'scale_physics_comparison.csv',index=False)
    audit='''# Physical metadata audit\n\nMetadata census uses every raw row without target-label reads. Normalization and center/width evidence below uses only source-train and the frozen first compound-seed budget100 gradient-train rows, not a union of seed labels. Full outcome distributions are deferred until global prediction freeze; audit labels are previously purchased developmental data.\n\n'''+table(census)+'\n\n## All flow values\n\n'+table(flows)+'\n\n## Raw pairing (direction source → target)\n\n'+table(pairs)+'''\n\nExact matching uses canonical molecule, rational EA composition, loading solvent, density, sample volume, loading-solvent volume and flow; relaxed ignores flow only. Pair coverage is metadata availability, not free source-test labels. 8g is explicitly 4g+4g, with unspecified physical connection geometry. Source/target canonical eligibility differs from raw census (especially 25g); inherited ledgers remain unchanged.\n\nUnits: packing_mass_g [g] is nominal label mass; flow_ml_min [mL/min] is volumetric flow; loading_mass_mg = density[g/mL] × sample_volume[uL]; loading_mass_mg_per_g [mg/g]; loading_solvent_ul_per_g [uL/g]; flow_per_g [mL/min/g] is an engineering proxy. V1/V2/center/width are mL; dividing by mass gives mL/g, not dimensionless and not CV. Center is an elution-window position proxy, not peak apex; width is an elution-window/dispersion proxy, not measured peak variance.\n'''
    (OUT/'PHYSICAL_METADATA_AUDIT.md').write_text(audit)
    (OUT/'COLUMN_METADATA_GAPS.md').write_text('''# Column metadata gaps\n\nREAL_COLUMN_VOLUME_SCALE_NOT_IDENTIFIABLE_FROM_CURRENT_REPOSITORY_METADATA\n\nAvailable: raw column_specs, nominal packing masses 4/8/25/40 g, flow, loading quantities, solvent composition and molecule identifiers. 8g is Silica-CS 4g+4g; nominal summed packing mass is 8 g, not evidence of a single cartridge.\n\nRepository clues exist and must not be called absent: application/QGeoGNN.py:1653–1655 hardcodes 8g diameter/length/density 1.5/13.2/0.4458; :1720–1722 hardcodes 2.15/15.6/0.5248 in a legacy dataset path; :3852–3854 hardcodes 4g 1.5/6.6/0.4458. scripts/run_g0_4_paper_style_transfer.py:58–59 repeats 4g/8g tuples. These are implementation constants, without verified physical units, measurement provenance, lot/product IDs, bed-versus-housing definition or void fraction. Do not treat numerical names as verified physical measurements.\n\nMissing verified metadata: measured packed-bed volume, void/dead volume and tracer method; inner diameter and actual bed length; particle-size distribution; measured silica bulk density; manufacturer and exact model/lot; tubing/connection volume for 4g+4g. No internet catalog was substituted. Ask experimental staff to verify code constants against records. No true CV baseline, CV normalization, F/A linear velocity or causal mass/flow interpretation is permitted.\n\nPriorities: crossed mass × flow experiments, independent batch/repeat IDs and exact-condition repeats; high-retention tail and source-unseen compounds. Same raw rows do not establish independent experimental repeats.\n''')
    pivot=discrepancy.pivot(index='target',columns='normalization',values='cross_column_mean_cv')
    improves=bool((pivot.mass_normalized < pivot.raw).all())
    gate='PHYSICAL_NORMALIZATION_PARTIALLY_SUPPORTED' if improves else 'PHYSICAL_NORMALIZATION_NOT_SUPPORTED'
    (OUT/'NORMALIZATION_AUDIT.md').write_text('# Normalization audit\n\n'+gate+'\n\nTraining-only cross-column CV of means:\n\n'+table(discrepancy)+'\n\nCompound/EA/flow matched comparisons:\n\n'+table(matched)+'\n\nWithin-column CV is algebraically unchanged by division by a constant mass. Cross-column location discrepancy, not within-column CV, is the relevant test. Matching is descriptive and composition/flow/geometry remain confounded. Partial support permits one fixed mass-normalized arm; it does not validate void-volume proportionality. Full-data outcome census is post-freeze only.\n')
    summary=scales.groupby(['column','protocol','target','budget'])[candidates[0]].agg(['mean','std']).reset_index() if 'budget' in scales else scales.groupby(['column','protocol','target'])[candidates[0]].agg(['mean','std']).reset_index()
    (OUT/'SCALE_PHYSICS_AUDIT.md').write_text('# Scale physics audit\n\nFrozen coefficients are read, never refit or optimized. See scale_physics_comparison.csv for every context, absolute and relative difference from packing_mass_ratio (2, 6.25, 10).\n\n'+table(summary)+'\n\nOLS a = sum(x*y)/sum(x²) weights ratios by x². The historical audit attributes 46–70% of mean weight to the training upper decile; this is not an independent physical law. V1/V2 discrepancies from the same nominal mass factor rule out exact universal mass proportionality. Partial mass-scale structure supports a prespecified mass-scale-plus-residual diagnostic, not a claim that mass is causal. No center/width OLS calibration is trained. Training matched distribution ratios characterize their different behavior; no raw chromatogram supports a peak-apex claim.\n')
    (OUT/'audit_training_ids.json').write_text(json.dumps(ids_record,indent=2))
    inputs=[ROOT/'dataset'/f'dataset_{c}.csv' for c in COLS]+[SOURCE,OLD/'splits/schedule_manifest.csv',Path(__file__),ROOT/'src/qgeognn_al/transfer/column_physics.py']
    files=list(OUT.glob('*.md'))+list(OUT.glob('*.csv'))+[OUT/'audit_training_ids.json']
    (OUT/'audit_frozen.json').write_text(json.dumps(dict(normalization_gate=gate,normalized_arm=improves,physical_residual_arm=True,test_labels_read=False,files={str(p.relative_to(ROOT)):sha256_file(p) for p in inputs+files}),indent=2))
    print(gate)
if __name__=='__main__': main()

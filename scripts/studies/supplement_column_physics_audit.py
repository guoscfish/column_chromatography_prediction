"""Training-only center/width ratios and metadata provenance supplement before fits."""
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.studies.audit_column_physics import OUT,OLD,SOURCE,COLS,table
from scripts.studies.run_scaling_failure_audit import FEATURES,selected_truth
from src.qgeognn_al.transfer.scaling_audit import match_key
from src.qgeognn_al.transfer.column_physics import packing_mass
from src.qgeognn_al.artifacts import sha256_file

def main():
    if (OUT/'protocol.json').exists(): raise RuntimeError('supplement must precede model protocol freeze')
    ids=json.loads((OUT/'audit_training_ids.json').read_text());data={};rows=[];coverage=[]
    for c in COLS:
        path=SOURCE if c=='4g' else OLD/'data_audit'/f'canonical_{c}.csv'
        all_meta=pd.read_csv(path,usecols=[*FEATURES,'column_specs'])
        for flow,g in all_meta.groupby('Flow mL/min'):
            coverage.append(dict(column=c,flow_ml_min=flow,canonical_rows=len(g),compounds=g.canonical_smiles.nunique()))
        d=all_meta.set_index('sample_id').loc[ids[c]].reset_index()
        y=selected_truth(path,ids[c]);d['V1'],d['V2']=y[:,0],y[:,1];d['center']=y.mean(1);d['width']=y[:,1]-y[:,0]
        d['EA_bin']=d['PE/EA'].map(lambda s: float(s.split('/')[1])/sum(map(float,s.split('/')))).map(lambda x:'low' if x<=.1 else 'mid' if x<=.5 else 'high')
        d['loading_mass_mg']=d['Density g/ml']*d['V/ul']
        d['loading_bin']=pd.cut(d.loading_mass_mg,[-np.inf,50,100,np.inf],labels=['<=50','50-100','>100'])
        data[c]=d
    for c in COLS[1:]:
        for mode in ['exact','relaxed']:
            mapping=defaultdict(list)
            for r in data['4g'].to_dict('records'):mapping[match_key(r,relaxed=mode=='relaxed')].append(r)
            paired=[]
            for r in data[c].to_dict('records'):
                matches=mapping.get(match_key(r,relaxed=mode=='relaxed'),[])
                if not matches:continue
                entry={k:r[k] for k in ['sample_id','canonical_smiles','EA_bin','Flow mL/min','loading_bin']}
                for t in ['V1','V2','center','width']:
                    base=np.mean([m[t] for m in matches]);entry[t+'_source']=base;entry[t+'_target']=r[t];entry[t+'_ratio']=r[t]/base if abs(base)>=.5 else np.nan
                paired.append(entry)
            p=pd.DataFrame(paired)
            for dim in ['overall','EA_bin','Flow mL/min','loading_bin']:
                groups=[('all',p)] if dim=='overall' else [] if p.empty else p.groupby(dim,observed=True)
                for level,g in groups:
                    for t in ['V1','V2','center','width']:
                        ratio=g[t+'_ratio'].dropna() if len(g) else pd.Series(dtype=float)
                        rows.append(dict(column=c,matching=mode,dimension=dim,level=str(level),target=t,rows=len(g),compounds=g.canonical_smiles.nunique() if len(g) else 0,ratio_rows=len(ratio),median_ratio=ratio.median(),ratio_q25=ratio.quantile(.25),ratio_q75=ratio.quantile(.75),packing_mass_ratio=float(c[:-1])/4))
    r=pd.DataFrame(rows);r.to_csv(OUT/'training_center_width_scales.csv',index=False)
    cv=pd.DataFrame(coverage);cv.to_csv(OUT/'canonical_flow_distribution.csv',index=False)
    overall=r.loc[r.dimension.eq('overall')]
    (OUT/'PHYSICAL_AUDIT_SUPPLEMENT.md').write_text('# Pre-model physical audit supplement\n\nThis supplement precedes all model fitting and uses exactly audit_training_ids.json. Ratio is the median of observed target/source-train matched values (source exact repeats averaged), with source denominator floor 0.5 mL. These are descriptive paired ratios, not fitted calibration coefficients. Center is not peak apex; width is not peak variance.\n\n'+table(overall)+'\n\nFull EA, volumetric-flow and prespecified loading-mass bins (<=50, 50–100, >100 mg) are in training_center_width_scales.csv. No matching at identical flow is available for 25g/40g, so their relaxed ratios do not isolate scale from flow.\n\nCanonical flow census:\n\n'+table(cv)+'\n\nRaw and canonical censuses are distinct. Current branch context exposes source flow variation already present in the inherited source training data; original V2 did not explicitly consume flow. Thus improvements cannot uniquely distinguish new source-flow information from cross-column conditioning. No feature ablation is appended.\n')
    paths=[Path(__file__),OUT/'PHYSICAL_AUDIT_SUPPLEMENT.md',OUT/'training_center_width_scales.csv',OUT/'canonical_flow_distribution.csv']
    (OUT/'audit_supplement_frozen.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha256_file(p) for p in paths}},indent=2))
    print(table(overall))
if __name__=='__main__':main()

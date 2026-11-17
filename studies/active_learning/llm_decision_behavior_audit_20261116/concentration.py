#!/usr/bin/env python3
"""Explicit consecutive-round locality summaries in fixed metadata space."""
from collections import Counter
from pathlib import Path
import json,lzma,sys
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101'))
from condition_metrics import _canonical_conditions

def main():
    data=json.loads(lzma.decompress((OUT/'states.json.xz').read_bytes()));meta=data['metadata'];rows=[]
    for (m,s),group in pd.DataFrame(data['states']).groupby(['method','seed']):
        previous_molecule=None;previous_condition=None;mol_streak=cond_streak=0;previous_set=set()
        for b in group.sort_values('round').to_dict('records'):
            selected=b['selected_ids'];mols=[meta[v]['smiles'] for v in selected]
            conditions=[tuple(map(str,_canonical_conditions(meta[v]))) for v in selected]
            mc=Counter(mols);cc=Counter(conditions)
            dominant_molecule=sorted(mc,key=lambda v:(-mc[v],v))[0]
            dominant_condition=sorted(cc,key=lambda v:(-cc[v],v))[0]
            mol_streak=mol_streak+1 if dominant_molecule==previous_molecule else 1
            cond_streak=cond_streak+1 if dominant_condition==previous_condition else 1
            current=set(mols);union=current|previous_set
            rows.append(dict(method=m,seed=int(s),round=b['round'],L_before=b['L_before'],
                dominant_molecule=dominant_molecule,dominant_molecule_fraction=mc[dominant_molecule]/32,
                same_modal_molecule_consecutive_rounds=mol_streak,
                dominant_condition=json.dumps(dominant_condition),dominant_condition_fraction=cc[dominant_condition]/32,
                same_modal_condition_consecutive_rounds=cond_streak,
                previous_batch_molecule_jaccard=len(current&previous_set)/len(union) if previous_set else None,
                interpretation='Exact modal identity streak; tie broken lexically; does not establish harmful concentration'))
            previous_molecule=dominant_molecule;previous_condition=dominant_condition;previous_set=current
    pd.DataFrame(rows).to_csv(OUT/'concentration_dynamics.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['method','seed']).agg(
        max_same_modal_molecule_rounds=('same_modal_molecule_consecutive_rounds','max'),
        max_same_modal_condition_rounds=('same_modal_condition_consecutive_rounds','max'),
        mean_previous_molecule_jaccard=('previous_batch_molecule_jaccard','mean'),
        mean_max_molecule_fraction=('dominant_molecule_fraction','mean')).reset_index()
    summary.to_csv(OUT/'concentration_campaigns.csv',index=False)
    print('Concentration trajectories',len(rows))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Add molecule-stratified descriptions from the already frozen response pairs."""
from pathlib import Path
import hashlib,json,sys
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'studies/active_learning/decision_mechanism_followup_20261117'))
from model_learnability import pair_numbers

def main():
    source=OUT/'frozen_response_pairs.csv';pairs=pd.read_csv(source)
    rows=[];keys=['seed','budget','axis','molecule','target']
    for key,group in pairs.groupby(keys):
        for threshold in [.1,.5,1.]:
            rows.append(dict(zip(keys,key),n_rows=len(set(group.low_id)|set(group.high_id)),
                **pair_numbers(group.actual_delta_ml,group.predicted_delta_ml,threshold,.1)))
    table=pd.DataFrame(rows);table.to_csv(OUT/'frozen_response_molecule_summary.csv',index=False)
    inputs=[source,Path(__file__),ROOT/'studies/active_learning/decision_mechanism_followup_20261117/model_learnability.py',
        ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101/condition_metrics.py']
    manifest=dict(sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        molecule_endpoint_groups=len(rows)//3,new_fits=0,new_llm_calls=0,
        scope='Observed in-sample histories of the original ten checkpoints; molecule groups and pairs are correlated descriptions, not independent replications.')
    (OUT/'molecule_summary_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Molecule-stratified groups',len(rows)//3)

if __name__=='__main__':main()

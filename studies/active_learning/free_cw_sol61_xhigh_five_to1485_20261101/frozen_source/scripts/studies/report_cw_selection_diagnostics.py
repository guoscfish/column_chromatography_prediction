#!/usr/bin/env python
"""Selection coverage and same-state L333 overlap; no label access."""
from pathlib import Path
from itertools import combinations
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import numpy as np
from src.qgeognn_al.active_learning_v2.benchmark_protocol import load_features
from src.qgeognn_al.active_learning_v2.cw_factorial_study import STUDY, BASE, PARENT
from src.qgeognn_al.active_learning_v2.cw_factorial import ARMS, SEEDS
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_study import STUDY as EXTENSION

def read(p):return json.loads(p.read_text())
data=load_features().set_index('sample_id');rows=[];batches={}
for seed in SEEDS:
    partition=pd.read_csv(BASE/f'splits/row_seed_{seed}.csv');l0=partition.loc[partition.role.eq('l0'),'sample_id'].tolist()
    for arm in (*ARMS,'raw_lcmd','cw_lcmd','raw_maxdet','raw_ivr','random'):
        labeled=list(l0)
        for r in range(6):
            diagnostic={}
            if arm in ARMS:
                d=STUDY/f'runtime/seed_{seed}/{arm}/round_{r:02d}';p=d/'batch_freeze.json'
                if not p.exists():break
                f=read(p);ids=f['ids'];assert f['contract']['ordered_l']==labeled
                diagnostic=read(d/'selection_diagnostics.json')
            elif arm=='raw_ivr':
                p=PARENT/f'qgeognn_v2_row_kernel_ivr_b32/runtime/seed_{seed}/round_{r:02d}/selection.json';ids=read(p)['selected_ids']
            else:
                if arm in ('raw_lcmd','random'):base=BASE/f'runtime/seed_{seed}/{"lcmd" if arm=="raw_lcmd" else "random"}'
                elif arm=='raw_maxdet':base=PARENT/f'qgeognn_v2_row_maxdet_b32/runtime/seed_{seed}/gradient_maxdet'
                else:base=(PARENT/'qgeognn_v2_row_short_sequential_b32' if r<3 else EXTENSION)/f'runtime/seed_{seed}/center_width_lcmd'
                folder='acquisition' if arm=='raw_maxdet' else 'acquisition_artifacts'
                ids=pd.read_csv(base/f'round_{r:02d}'/folder/'selected_next_batch.csv').sample_id.astype(str).tolist()
            assert len(ids)==32 and len(set(ids))==32 and not set(ids)&set(labeled)
            frame=data.loc[ids];observed=data.loc[labeled]
            old_molecules=set(observed.canonical_smiles)
            condition_columns=['canonical_smiles','PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
            old_conditions=set(map(tuple,observed[condition_columns].to_numpy()))
            row={'seed':seed,'arm':arm,'source_budget':333+32*r,'batch_size':32,
                 'unique_structures':frame.canonical_smiles.nunique(),
                 'new_structure_rows':sum(x not in old_molecules for x in frame.canonical_smiles),
                 'new_structure_condition_rows':sum(x not in old_conditions for x in map(tuple,frame[condition_columns].to_numpy())),
                 'unique_eluent_ratios':frame['PE/EA'].nunique(),'unique_loading_solvents':frame['loading solvent'].nunique(),
                 'effective_rank':diagnostic.get('effective_rank',np.nan)}
            for col,short in [('Density g/ml','density'),('V/ul','sample_volume'),('Volume of loading solvent/ul','loading_volume')]:
                row[short+'_min']=frame[col].min();row[short+'_max']=frame[col].max()
            if 'norm_quantiles' in diagnostic:
                for name,val in zip(('min','q25','median','q75','max'),diagnostic['norm_quantiles']):row['gradient_norm_'+name]=val
            rows.append(row)
            if r==0:batches[(seed,arm)]=set(ids)
            labeled+=ids
pd.DataFrame(rows).to_csv(STUDY/'selection_coverage.csv',index=False)
overlaps=[]
for seed in SEEDS:
    arms=[a for s,a in batches if s==seed]
    for a,b in combinations(arms,2):
        x,y=batches[(seed,a)],batches[(seed,b)]
        overlaps.append({'seed':seed,'source_budget':333,'arm_a':a,'arm_b':b,'intersection':len(x&y),'Jaccard':len(x&y)/len(x|y),'same_model_state':True})
pd.DataFrame(overlaps).to_csv(STUDY/'same_state_L333_overlap.csv',index=False)
print(json.dumps({'coverage_batches':len(rows),'same_state_pairs':len(overlaps),'labels_read':0}))

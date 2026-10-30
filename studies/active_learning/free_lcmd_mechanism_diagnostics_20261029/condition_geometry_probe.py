"""Feature-only matched-condition distances in existing representative banks.

This targeted diagnostic reads frozen condition metadata and existing banks. It
never reads unselected responses, runs inference, extracts gradients, or selects
new experiments. Fixed representatives: seeds 157/887, rounds 0/10/20/30.
"""
from __future__ import annotations
from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import geometry_analysis as g

CONDITIONS=('PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul')
FIELDS=('V/ul','PE/EA','Volume of loading solvent/ul')
SEEDS=(157,887)
ROUNDS=(0,10,20,30)


def matched_pairs(rows,field):
    groups=defaultdict(list)
    for j,row in enumerate(rows):
        key=(row['smiles'],)+tuple(row['conditions'][c] for c in CONDITIONS if c!=field)
        groups[key].append(j)
    return np.array([(i,j) for members in groups.values() for i,j in itertools.combinations(members,2)
                     if rows[i]['conditions'][field]!=rows[j]['conditions'][field]],dtype=int)


def run():
    summaries=[];examples=[]
    for seed in SEEDS:
        partition=pd.read_csv(g.BASE/f'splits/row_seed_{seed}.csv')
        catalog=g.read(g.owner(seed,1)/f'selections/seed_{seed}/{g.FREE}/round_00/catalog.json')
        meta={row['id']:{k:row[k] for k in ('id','smiles','conditions')} for kind in ('candidates','observed') for row in catalog[kind]}
        canonical_to_id=dict(zip(partition.canonical_index.astype(int),partition.sample_id.astype(str)))
        for r in ROUNDS:
            for method,reader in ((g.FREE,g.free_inputs),('lcmd',g.lcmd_inputs)):
                arrays,n,selected,p,_,audit=reader(seed,r,partition)
                bank=g.ROOT/audit['bank_path']
                if method==g.FREE:
                    indices=g.read(bank.with_suffix('.json'))['contract']['indices']
                else:
                    with np.load(bank) as z:indices=z['canonical_indices'].tolist()
                rows=[meta[canonical_to_id[i]] for i in indices]
                for field in FIELDS:
                    pairs=matched_pairs(rows,field)
                    assert len(pairs)>0,(seed,r,field)
                    i,j=pairs.T
                    L_U=((i<n)&(j>=n))|((j<n)&(i>=n))
                    eligible=set(np.where(i[L_U]>=n,i[L_U],j[L_U]).tolist())
                    selected_global=set((n+p).tolist())
                    for space,features in arrays.items():
                        x=features.astype(np.float64)
                        rms=float(np.sqrt(np.square(x-x.mean(0)).sum(1).mean()))
                        distance=np.linalg.norm(x[i]-x[j],axis=1)/rms
                        key={'seed':seed,'method':method,'round':r,'labels_before':n,'space':space,'varied_condition':field}
                        result={**key,'matched_pair_count':len(distance),'distance_to_bank_RMS_min':float(distance.min()),
                                'distance_to_bank_RMS_p10':float(np.quantile(distance,.1)),
                                'distance_to_bank_RMS_median':float(np.median(distance)),
                                'distance_to_bank_RMS_p90':float(np.quantile(distance,.9)),
                                'distance_to_bank_RMS_max':float(distance.max()),
                                'exact_zero_fraction':float((distance==0).mean()),
                                'below_1e_minus3_bank_RMS_fraction':float((distance<1e-3).mean()),
                                'matched_L_U_pairs':int(L_U.sum()),'eligible_U_rows':len(eligible),
                                'eligible_U_selected_count':len(selected_global&eligible),
                                'selected_eligible_U_fraction':len(selected_global&eligible)/32,
                                'bank_path':audit['bank_path'],'bank_sha256':g.sha(bank)}
                        summaries.append(result)
                        order=np.argsort(distance)
                        for tag,k in [('min',order[0]),('median',order[len(order)//2]),('max',order[-1])]:
                            a,b=rows[i[k]],rows[j[k]]
                            examples.append({**key,'example':tag,'id_a':a['id'],'id_b':b['id'],'smiles':a['smiles'],
                                             'conditions_a':a['conditions'],'conditions_b':b['conditions'],
                                             'distance_to_bank_RMS':float(distance[k]),'bank_path':audit['bank_path']})
    frame=pd.DataFrame(summaries)
    frame.to_csv(g.OUT/'matched_condition_geometry_probe.csv',index=False)
    pivot=frame.pivot(index=['seed','method','round','space'],columns='varied_condition',values='distance_to_bank_RMS_median')
    ratios=(pivot['PE/EA']/pivot['V/ul']).rename('median_PE_contrast_distance_divided_by_median_amount_contrast_distance').reset_index()
    report={'analysis':'fixed_representative_frozen_bank_condition_probe','seeds':list(SEEDS),'rounds':list(ROUNDS),
            'source_labels_before':[333+32*r for r in ROUNDS],
            'coverage':'2 seeds x 4 states x (Free raw/CW + LCMD raw) x 3 matched-condition axes = 72 rows',
            'pair_definition':'same canonical SMILES and all recorded conditions identical except varied_condition; varied condition values must differ',
            'normalization':'Euclidean feature distance / RMS centered feature norm over the same current L+U bank',
            'statistical_status':'descriptive geometry probe; pairs are dependent, no inferential pair-level sample size',
            'source_truth_table_read':False,'new_model_or_gradient_computation':False,
            'ratio_summary':{'min':float(ratios.iloc[:,-1].min()),'median':float(ratios.iloc[:,-1].median()),'max':float(ratios.iloc[:,-1].max())},
            'state_ratios':ratios.to_dict('records'),'summary_rows':summaries,'representative_pairs':examples,'source_sha256':g.HASHES}
    selection_records=g.OUT/'selection_records.csv'
    point_geometry=g.OUT/'geometry_selected_points.csv'
    if selection_records.exists() and point_geometry.exists():
        g.sha(selection_records);g.sha(point_geometry)
        keys=['seed','method','round','selection_order','sample_id']
        behavior=pd.read_csv(selection_records,usecols=keys+['amount_curve_extension'])
        geometry=pd.read_csv(point_geometry,usecols=keys+['raw_distance_percentile','cw_distance_percentile'])
        joined=behavior.merge(geometry,on=keys,validate='one_to_one')
        linked=joined.groupby(['seed','method','amount_curve_extension']).agg(
            selected_points=('sample_id','size'),raw_percentile_mean=('raw_distance_percentile','mean'),
            cw_percentile_mean=('cw_distance_percentile','mean')).reset_index()
        linked.to_csv(g.OUT/'behavior_geometry_amount_link.csv',index=False)
        report['selected_amount_geometry_link']=json.loads(linked.to_json(orient='records'))
    (g.OUT/'matched_condition_geometry_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    series=[(g.FREE,'raw','Free raw'),(g.FREE,'cw','Free CW'),('lcmd','raw','LCMD raw')]
    colors=['#c76b42','#b59747','#356d9a']
    for k,(method,space,label) in enumerate(series):
        subset=frame[(frame.method==method)&(frame.space==space)]
        for j,field in enumerate(FIELDS):
            values=subset[subset.varied_condition==field].distance_to_bank_RMS_median.to_numpy()
            x=j+(k-1)*.19+np.linspace(-.035,.035,len(values))
            axs[0].scatter(x,values,s=20,color=colors[k],alpha=.6,label=label if j==0 else None)
    axs[0].set_yscale('log');axs[0].set_xticks(range(3),['Amount','PE/EA','Loading volume'])
    axs[0].set_ylabel('Median matched-pair distance / current-bank RMS')
    axs[0].set_title('Representative frozen states; 2 seeds x 4 rounds',fontsize=10)
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.15)
    if 'linked' in locals():
        subset=linked[linked.method.eq(g.FREE)]
        for seed,rows in subset.groupby('seed'):
            rows=rows.sort_values('amount_curve_extension',ascending=False)
            axs[1].plot([0,1],rows.raw_percentile_mean,marker='o',alpha=.7,label=str(seed))
        axs[1].set_xticks([0,1],['Amount-extension picks','Other Free picks'],fontsize=9)
        axs[1].set_ylim(0,1);axs[1].set_ylabel('Mean raw distance percentile within own pool')
        axs[1].set_title('All Free selected points; paired seed means',fontsize=10)
        axs[1].legend(fontsize=7,ncol=3);axs[1].grid(alpha=.15)
    fig.suptitle('Frozen geometry resolves solvent contrasts far more than matched loading contrasts\nDescriptive diagnostics; validation trajectories; no new model/gradient computation',fontsize=11)
    fig.savefig(g.OUT/'matched_condition_geometry_signal.png',dpi=180)
    fig.savefig(g.OUT/'matched_condition_geometry_signal.svg');plt.close(fig)
    print(frame.groupby(['method','space','varied_condition'])[['distance_to_bank_RMS_median','below_1e_minus3_bank_RMS_fraction','exact_zero_fraction']].mean().to_string())
    print('PE / amount median-distance ratios',report['ratio_summary'])
    return frame,report


if __name__=='__main__':run()

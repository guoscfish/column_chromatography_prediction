#!/usr/bin/env python3
"""Read-only selected-feedback audit; never opens a canonical label table/model.

Full quantiles and conditions come from SHA-verified preselection catalogs;
truth only from the matching accepted batch's archived feedback. Export only
aggregates, never complete catalogs, packets, or row-level measured responses.
"""
from __future__ import annotations
import gzip, hashlib, itertools, json, math, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/chromatography_history_mpl')
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
RUN = ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
ARCH = ROOT/'reports/free_cw_final_20261112/scientific_artifacts'
SEEDS = [157, 887, 2357, 6101, 12203]
METHOD = 'new_free_cw_sol61_xhigh'
COND = ['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
DESC = ['MolWt','LogP','TPSA','HBD','HBA','aromatic_fraction','hetero_atoms']
PHASES = ['333-525','525-813','813-1005','1005-1485']
MANIFEST = []

def sha(b): return hashlib.sha256(b).hexdigest()
def add_input(p, b=None, role='selected_feedback_or_preselection_features'):
    b = p.read_bytes() if b is None else b
    MANIFEST.append({'path':str(p.relative_to(ROOT)), 'sha256':sha(b),'bytes':len(b),'role':role})
    return b
def phase(r): return PHASES[0 if r<6 else 1 if r<15 else 2 if r<21 else 3]
def rho(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return np.nan
    return float(spearmanr(x,y).statistic)
def partial_rho(x,y,controls):
    if len(x)<8 or np.ptp(x)==0 or np.ptp(y)==0: return np.nan
    z=np.column_stack([np.ones(len(x))]+[rankdata(controls[c]) for c in controls])
    a,b=rankdata(x),rankdata(y)
    ea=a-z@np.linalg.lstsq(z,a,rcond=None)[0]; eb=b-z@np.linalg.lstsq(z,b,rcond=None)[0]
    return float(np.corrcoef(ea,eb)[0,1])
def signal_metrics(signal,error):
    s,e=np.asarray(signal,float),np.asarray(error,float); n=len(e); k=math.ceil(.2*n)
    top=np.argsort(-s,kind='stable')[:k]; hard=np.argsort(-e,kind='stable')[:k]
    hardlabel=np.zeros(n);hardlabel[hard]=1
    fractions=np.linspace(0,.9,10); observed=[]; oracle=[]
    for f in fractions:
        keep=max(1,n-int(np.floor(f*n)))
        observed.append(e[np.argsort(s,kind='stable')[:keep]].mean())
        oracle.append(np.sort(e)[:keep].mean())
    return {'rho':rho(s,e), 'top20_error_lift':float(e[top].mean()/max(e.mean(),1e-12)),
            'hard_top20_auroc':float(roc_auc_score(hardlabel,s)),
            'ause':float(np.trapezoid((np.array(observed)-oracle)/max(e.mean(),1e-12),fractions))}
def desc(smiles):
    mol=Chem.MolFromSmiles(smiles); assert mol is not None
    atoms=list(mol.GetAtoms());return {'MolWt':Descriptors.MolWt(mol),'LogP':Descriptors.MolLogP(mol),
      'TPSA':rdMolDescriptors.CalcTPSA(mol),'HBD':Lipinski.NumHDonors(mol),'HBA':Lipinski.NumHAcceptors(mol),
      'aromatic_fraction':sum(a.GetIsAromatic() for a in atoms)/len(atoms),
      'hetero_atoms':sum(a.GetAtomicNum() not in (1,6) for a in atoms),
      'scaffold':MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
      'molecule_key':sha(smiles.encode())[:16]}
def write_csv(name,rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/'tables'/name,index=False);return df
def endpoint_stats(g,target):
    e=g[f'e{target}'].to_numpy(float); q10=g[f'q{target}_10'];q50=g[f'q{target}_50'];q90=g[f'q{target}_90'];y=g[f'y{target}']
    ordered=np.sort(np.column_stack([q10,q50,q90]),axis=1)
    return {'n':len(g),'rmse_ml':float(np.sqrt(np.mean(e**2))),'mae_ml':float(np.mean(abs(e))),
      'bias_pred_minus_true_ml':float(e.mean()),'coverage80_raw':float(((q10<=y)&(y<=q90)).mean()),
      'q10_gt_q50_rate':float((q10>q50).mean()),'q50_gt_q90_rate':float((q50>q90).mean()),
      'negative_width_rate':float((q10>q90).mean()),'signed_width_mean_ml':float((q90-q10).mean()),
      'negative_q50_rate':float((q50<0).mean()),
      'sorting_q50_changed_rate':float(np.mean(ordered[:,1]!=q50)),
      'sorting_q50_mae_ml':float(np.mean(abs(y-ordered[:,1]))),
      'sorting_q50_rmse_ml':float(np.sqrt(np.mean((y-ordered[:,1])**2))),
      'sorting_coverage80':float(np.mean((ordered[:,0]<=y)&(y<=ordered[:,2]))),
      'pinball_q10_ml':float(np.mean(np.maximum(.1*(y-q10),-.9*(y-q10)))),
      'pinball_q90_ml':float(np.mean(np.maximum(.9*(y-q90),-.1*(y-q90))))}

def main():
    for name in ['tables','figures','verification']: (OUT/name).mkdir(exist_ok=True)
    frozen_protocol=json.loads((RUN/'protocol.json').read_text())
    frozen_scales={v['seed']:v['preprocessing']['target_scales'] for v in frozen_protocol['initial_states']}
    allrows=[]; batch=[]; reliab=[]; correlations=[]; pairrows=[]; duplicates=[]; seen_desc={}; archive_checked=0
    for seed in SEEDS:
        ap=ARCH/f'seed_{seed}_results.jsonl.gz';add_input(ap,role='published_lossless_scientific_archive')
        archived={}
        with gzip.open(ap,'rt') as f:
            for line in f:
                o=json.loads(line);name=Path(o['path']).name
                if name in {'feedback.json','contract.json','batch_freeze.json'}:
                    assert sha(o['raw_utf8'].encode())==o['sha256'];archive_checked+=1
                    archived[o['path']]=json.loads(o['raw_utf8'])
        per_seed=[];all_observed={}; sigma=None
        for r in range(36):
            base=RUN/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}'
            rel=str(base.relative_to(ROOT));c=archived[rel+'/contract.json'];fb=archived[rel+'/feedback.json'];freeze=archived[rel+'/batch_freeze.json']
            raw=add_input(base/'catalog.json');assert sha(raw)==c['catalog_sha256']
            cat=json.loads(raw); candidates={v['id']:v for v in cat['candidates']+cat['pending']}
            assert set(candidates)==set(c['U_t_ids'])
            assert set(x['id'] for x in cat['observed'])==set(c['L_t_ids'])
            assert len(fb['records'])==len(freeze['batch_ids'])==32
            assert [v['candidate_id'] for v in fb['records']]==freeze['batch_ids']
            assert not set(freeze['batch_ids']) & set(c['L_t_ids'])
            if r==0:
                sigma=np.array([frozen_scales[seed][t] for t in ['V1','V2']],dtype=float)
                recomputed=np.std(np.array([[v['V1_ml'],v['V2_ml']] for v in cat['observed']],dtype=np.float32),axis=0,ddof=0)
                assert np.allclose(sigma,recomputed,rtol=2e-6)
            for v in cat['observed']:
                all_observed[v['id']]={'smiles':v['smiles'],'conditions':v['conditions'],'y1':v['V1_ml'],'y2':v['V2_ml']}
            rows=[]
            for v,fr in zip(fb['records'],freeze['premeasurement_predictions']):
                ca=candidates[v['candidate_id']];sm=ca['smiles'];co=ca['conditions']
                if sm not in seen_desc: seen_desc[sm]=desc(sm)
                pe,ea=map(float,co['PE/EA'].split('/'))
                d={'method':METHOD,'seed':seed,'round':r,'labeled_size_before':333+32*r,'phase':phase(r),
                   'id':v['candidate_id'],'smiles':sm,'conditions':co,**seen_desc[sm],
                   'PE_EA':co['PE/EA'],'EA_fraction':ea/(pe+ea),'sample_volume_ul':float(co['V/ul']),
                   'loading_amount_density_x_volume':float(co['Density g/ml'])*float(co['V/ul']),
                   'loading_solvent':co['loading solvent'],'loading_solvent_volume_ul':float(co['Volume of loading solvent/ul']),
                   'cw_percentile':float(ca['cw_distance_percentile'])}
                for t in [1,2]:
                    assert v[f'pred_V{t}_ml']==fr[f'pred_V{t}_ml']==ca[f'pred_V{t}_q50_ml']
                    assert np.isclose(v[f'error_V{t}_ml'],v[f'pred_V{t}_ml']-v[f'true_V{t}_ml'],atol=1e-7)
                    d.update({f'y{t}':v[f'true_V{t}_ml'],f'e{t}':v[f'error_V{t}_ml'],f'abs_e{t}':abs(v[f'error_V{t}_ml']),
                      f'norm_e{t}':v[f'error_V{t}_ml']/sigma[t-1]})
                    for q in [10,50,90]:d[f'q{t}_{q}']=float(ca[f'pred_V{t}_q{q}_ml'])
                    d[f'w{t}']=d[f'q{t}_90']-d[f'q{t}_10'];d[f'prednorm{t}']=d[f'q{t}_50']/sigma[t-1]
                d['norm_abs_combined']=math.sqrt(.5*(d['norm_e1']**2+d['norm_e2']**2))
                d['norm_width_combined']=math.sqrt(.5*((max(d['w1'],0)/sigma[0])**2+(max(d['w2'],0)/sigma[1])**2))
                d['cross_endpoint_span']=d['q2_90']-d['q1_10'];rows.append(d)
                all_observed[d['id']]={'smiles':sm,'conditions':co,'y1':d['y1'],'y2':d['y2']}
            g=pd.DataFrame(rows);allrows+=rows;per_seed+=rows
            b={'method':METHOD,'seed':seed,'round':r,'labeled_size_before':333+32*r,'phase':phase(r),
               'selected_n':32,'selected_error_combined_nrmse':float(np.sqrt(np.mean(g.norm_abs_combined**2))),
               'selected_abs_normalized_error_mean':g.norm_abs_combined.mean(),
               'selected_residual_asymmetry_v2_minus_v1':float(np.mean(abs(g.norm_e2))-np.mean(abs(g.norm_e1))),
               'selected_cw_percentile_mean':g.cw_percentile.mean(),
               'selected_v1_v2_residual_spearman':rho(g.e1,g.e2),
               'selected_pred_v2_below_v1_rate':float((g.q2_50<g.q1_50).mean())}
            for t in [1,2]:
                b.update({f'selected_v{t}_{k}':v for k,v in endpoint_stats(g,t).items() if k!='n'})
                for sig in [f'w{t}','cw_percentile',f'q{t}_50','norm_width_combined','cross_endpoint_span']:
                    reliab.append({'seed':seed,'round':r,'phase':phase(r),'target':f'V{t}','signal':sig,
                                   **signal_metrics(g[sig],g[f'abs_e{t}'])})
            batch.append(b)
            for a,z in itertools.combinations(['w1','w2','cw_percentile','q1_50','q2_50','norm_width_combined','cross_endpoint_span'],2):
                correlations.append({'seed':seed,'round':r,'phase':phase(r),'signal_a':a,'signal_b':z,'rho':rho(g[a],g[z]),'population':'selected32_same_state'})
            # A local response comparison is valid only if both points share THIS pre-batch checkpoint.
            for a,z in itertools.combinations(rows,2):
                if a['smiles']!=z['smiles']:continue
                differing=[k for k in COND if a['conditions'][k]!=z['conditions'][k]]
                if len(differing)!=1:continue
                axis=differing[0]
                if axis=='PE/EA':
                    if a['EA_fraction']>z['EA_fraction']:a,z=z,a
                elif axis!='loading solvent' and float(a['conditions'][axis])>float(z['conditions'][axis]):a,z=z,a
                for t in [1,2]:
                    actual=z[f'y{t}']-a[f'y{t}'];pred=z[f'q{t}_50']-a[f'q{t}_50']
                    pairrows.append({'seed':seed,'round':r,'phase':phase(r),'axis':axis,'target':f'V{t}',
                        'pair_key':sha('|'.join(sorted([a['id'],z['id']])).encode())[:16],
                        'molecule_key':a['molecule_key'],'actual_abs_delta_ml':abs(actual),'pred_abs_delta_ml':abs(pred),
                        'difference_error_ml':abs(actual-pred),'sign_agreement':int(np.sign(actual)==np.sign(pred)),
                        'actual_increases_with_axis':int(actual>0),'pred_increases_with_axis':int(pred>0),
                        'pred_nearly_flat_abs_lt_0p1ml':int(abs(pred)<.1),'observed_abs_gt_1ml':int(abs(actual)>1)})
        # Identical recorded conditions among observed rows only. Distinct IDs are not proof of real replication.
        groups={}
        for i,v in all_observed.items():groups.setdefault((v['smiles'],tuple((k,v['conditions'][k]) for k in COND)),[]).append(v)
        for vals in groups.values():
            if len(vals)<2:continue
            for t in [1,2]:
                y=[v[f'y{t}'] for v in vals]
                duplicates.append({'seed':seed,'target':f'V{t}','n_rows':len(vals),'n_pairs':len(vals)*(len(vals)-1)//2,
                  'range_ml':max(y)-min(y),'sd_ml':float(np.std(y,ddof=1)),
                  'group_key':sha(json.dumps([vals[0]['smiles'],vals[0]['conditions']],sort_keys=True).encode())[:16],
                  'all_equal':int(max(y)==min(y)), 'molecule_key':desc(vals[0]['smiles'])['molecule_key']})
    df=pd.DataFrame(allrows);bm=write_csv('model_batch_metrics.csv',batch)
    rm=write_csv('signals_selected_batch_reliability.csv',reliab)
    write_csv('signals_selected_batch_correlations.csv',correlations)
    grouped=[]
    for (s,p),g in df.groupby(['seed','phase'],sort=False):
        for t in [1,2]:grouped.append({'seed':s,'phase':p,'target':f'V{t}',**endpoint_stats(g,t)})
    gs=write_csv('model_seed_phase_errors.csv',grouped)
    write_csv('signals_seed_phase_reliability.csv',rm.groupby(['seed','phase','target','signal'],sort=False).agg(
        n_batches=('round','size'),rho=('rho','mean'),top20_error_lift=('top20_error_lift','mean'),
        hard_top20_auroc=('hard_top20_auroc','mean'),ause=('ause','mean')).reset_index())
    feats=[]
    for (seed,p),g in df.groupby(['seed','phase'],sort=False):
        for name in DESC+['EA_fraction','sample_volume_ul','loading_solvent_volume_ul','cw_percentile','norm_width_combined','cross_endpoint_span']:
            feats.append({'seed':seed,'phase':p,'feature':name,'n_selected_rows':len(g),'n_molecules':g.molecule_key.nunique(),
                'rho_error':rho(g[name],g.norm_abs_combined),'partial_rho_error':partial_rho(g[name],g.norm_abs_combined,
                g[['prednorm1','prednorm2','cw_percentile','round']] if name!='cw_percentile' else g[['prednorm1','prednorm2','round']])})
    write_csv('model_feature_error_correlations.csv',feats)
    strata=[]
    for axis in ['PE_EA','loading_solvent','sample_volume_ul','loading_solvent_volume_ul']:
        for (seed,p,value),g in df.groupby(['seed','phase',axis],sort=False):
            for t in [1,2]:strata.append({'seed':seed,'phase':p,'axis':axis,'value':value,'target':f'V{t}',
                                       'n_molecules':g.molecule_key.nunique(),**endpoint_stats(g,t)})
    write_csv('model_condition_error_strata.csv',strata)
    molecules=[]
    for (seed,mol),g in df.groupby(['seed','molecule_key']):
        for t in [1,2]:molecules.append({'seed':seed,'molecule_key':mol,'target':f'V{t}',
                    **{k:g.iloc[0][k] for k in DESC},'scaffold':g.iloc[0].scaffold,**endpoint_stats(g,t)})
    write_csv('model_molecule_error_strata.csv',molecules)
    pairs=pd.DataFrame(pairrows)
    if len(pairs):
        # Export pair summaries only: measured responses remain in the existing archive.
        ps=pairs.groupby(['seed','phase','axis','target']).agg(n_pairs=('seed','size'),n_unique_pairs=('pair_key','nunique'),n_molecules=('molecule_key','nunique'),
           actual_abs_delta_ml=('actual_abs_delta_ml','mean'),pred_abs_delta_ml=('pred_abs_delta_ml','mean'),
           difference_error_ml=('difference_error_ml','mean'),sign_agreement=('sign_agreement','mean'),
           actual_increases_with_axis=('actual_increases_with_axis','mean'),pred_increases_with_axis=('pred_increases_with_axis','mean'),
           pred_nearly_flat_abs_lt_0p1ml=('pred_nearly_flat_abs_lt_0p1ml','mean'),observed_abs_gt_1ml=('observed_abs_gt_1ml','mean')).reset_index()
        write_csv('model_same_batch_single_factor_response.csv',ps)
        write_csv('model_single_factor_unique_counts.csv',pairs.groupby(['axis','target']).agg(n_pair_events=('seed','size'),
          n_unique_pairs=('pair_key','nunique'),n_molecules=('molecule_key','nunique'),n_seeds=('seed','nunique')).reset_index())
    du=pd.DataFrame(duplicates)
    if len(du):
        write_csv('model_identical_recorded_conditions.csv',du.groupby(['seed','target']).agg(
           n_groups=('seed','size'),n_rows=('n_rows','sum'),n_pairs=('n_pairs','sum'),n_molecules=('molecule_key','nunique'),
           mean_range_ml=('range_ml','mean'),median_range_ml=('range_ml','median'),max_range_ml=('range_ml','max'),
           equal_groups=('all_equal','sum')).reset_index())
    # Plots deliberately label selected-row population; no claim of validation calibration.
    fig,ax=plt.subplots(1,3,figsize=(14,4.1))
    for ti,t in enumerate(['V1','V2']):
        q=gs[gs.target.eq(t)]
        for s,g in q.groupby('seed'):
            g=g.set_index('phase').reindex(PHASES)
            ax[ti].plot(range(4),g.coverage80_raw,marker='o',alpha=.75,label=str(s))
        ax[ti].axhline(.8,ls='--',color='black');ax[ti].set_title(t+' raw q10-q90 coverage')
        ax[ti].set_xticks(range(4),PHASES,rotation=20);ax[ti].set_ylim(0,1);ax[ti].set_ylabel('Selected-row fraction');ax[ti].legend(fontsize=7)
    r=rm.groupby(['seed','target','signal']).rho.mean().reset_index()
    labels=['width','CW distance','q50']
    for i,t in enumerate(['V1','V2']):
        for j,si in enumerate([f'w{i+1}','cw_percentile',f'q{i+1}_50']):
            v=r[r.target.eq(t)&r.signal.eq(si)].rho
            ax[2].scatter(np.full(len(v),j+i*.22),v,label=t if j==0 else None,s=26,color=['tab:blue','tab:orange'][i])
    ax[2].set_xticks(np.arange(3)+.11,labels);ax[2].axhline(0,color='gray',lw=.5);ax[2].set_ylabel('Mean within-batch error Spearman');ax[2].legend();ax[2].set_title('5 seed means; 36 batches/seed')
    fig.suptitle('CW-Free selected feedback: 5 seeds, 180 batches, 5760 selections (not independent rows)',fontsize=11)
    fig.text(.5,.005,'Source: SHA-verified preselection catalogs + accepted feedback. Phases are pre/post budget windows; intervals and errors in mL.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(OUT/'figures/model_quantile_reliability.png',dpi=170);plt.close(fig)
    if len(pairs):
        axes=sorted(pairs.axis.unique());fig,ax=plt.subplots(1,2,figsize=(12,4))
        for ti,t in enumerate(['V1','V2']):
            q=pairs[pairs.target.eq(t)].groupby(['seed','axis'])[['actual_abs_delta_ml','pred_abs_delta_ml','difference_error_ml']].mean().groupby('axis').mean().reindex(axes)
            for j,(col,label) in enumerate([('actual_abs_delta_ml','Observed |delta|'),('pred_abs_delta_ml','Predicted |delta|'),('difference_error_ml','Difference error')]):
                ax[ti].bar(np.arange(len(axes))+(j-1)*.24,q[col],.24,label=label)
            axislabels={'PE/EA':'PE/EA\n714 pairs, 5 seeds','V/ul':'Sample vol.\n165 pairs, 5 seeds',
                        'Volume of loading solvent/ul':'Loading vol.\n7 pairs, 2 seeds','loading solvent':'Solvent type\n7 pairs, 4 seeds'}
            ax[ti].set_xticks(range(len(axes)),[axislabels[a] for a in axes]);ax[ti].set_title(t);ax[ti].set_ylabel('mL; equal means over available seeds');ax[ti].legend(fontsize=8)
        fig.suptitle(f'Same molecule, same batch checkpoint, one changed recorded factor: {len(pairs)//2} pairs')
        fig.text(.5,.005,'Source: CW-Free accepted feedback; unequal counts by factor; shared molecules/pairs are descriptive, not independent replicates.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'figures/model_local_response.png',dpi=170);plt.close(fig)
    # Fixed-rule examples of actual local curves, never chosen by residual size.
    local_groups={}
    other=[k for k in COND if k!='PE/EA']
    for row in allrows:
        k=(row['seed'],row['round'],row['molecule_key'],tuple((a,row['conditions'][a]) for a in other))
        local_groups.setdefault(k,[]).append(row)
    qualified=[(k,v) for k,v in local_groups.items() if len({x['EA_fraction'] for x in v})>=3]
    fixed=[]; curve_meta=[]
    for s in SEEDS:
        options=sorted([(k,v) for k,v in qualified if k[0]==s],key=lambda kv:(kv[0][1],kv[0][2],repr(kv[0][3])))
        if options:
            k,v=options[0];v=sorted(v,key=lambda x:(x['EA_fraction'],x['id']));fixed.append((k,v))
            curve_meta.append({'seed':s,'round':k[1],'molecule_key':k[2],'n_selected_points':len(v),
              'candidate_ids':';'.join(x['id'] for x in v),'rule':'first round, then lexical molecule SHA; >=3 EA levels; all other recorded conditions identical'})
    if fixed:
        fig,axes=plt.subplots(2,len(fixed),figsize=(3.1*len(fixed),6),squeeze=False)
        for j,(k,v) in enumerate(fixed):
            for ti,t in enumerate([1,2]):
                a=axes[ti,j];a.plot([x['EA_fraction'] for x in v],[x[f'y{t}'] for x in v],'o-',label='Observed',color='black')
                a.plot([x['EA_fraction'] for x in v],[x[f'q{t}_50'] for x in v],'s--',label='Pre-batch model',color='tab:blue')
                a.set_title(f'seed {k[0]}, round {k[1]:02d}\n{k[2][:8]} / V{t}',fontsize=9)
                a.set_xlabel('EA volume fraction');a.set_ylabel('mL');a.legend(fontsize=7)
        fig.suptitle('Fixed-rule same-batch local curves; 5 illustrative cases, not a prevalence estimate')
        fig.text(.5,.005,'Same molecule and all non-PE/EA recorded conditions; only accepted historical responses. Source: frozen catalogs + feedback.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.03,1,.94]);fig.savefig(OUT/'figures/model_fixed_local_curves.png',dpi=170);plt.close(fig)
        write_csv('model_fixed_local_curve_metadata.csv',curve_meta)
    # Heatmap gives seed consistency and explicitly shows negative/weak associations.
    f=pd.DataFrame(feats).groupby(['seed','feature']).rho_error.mean().unstack('seed').reindex(DESC+['EA_fraction','sample_volume_ul','loading_solvent_volume_ul'])
    fig,ax=plt.subplots(figsize=(7.6,5));im=ax.imshow(f.to_numpy(),vmin=-.5,vmax=.5,cmap='coolwarm',aspect='auto')
    ax.set_yticks(range(len(f)),f.index);ax.set_xticks(range(5),f.columns);fig.colorbar(im,ax=ax,label='Mean phase Spearman with normalized selected error')
    for i in range(len(f)):
        for j in range(5):ax.text(j,i,f'{f.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Chemical/condition error associations: selected feedback only');ax.set_xlabel('Seed; 4 phase correlations averaged per seed')
    fig.tight_layout();fig.savefig(OUT/'figures/model_chemical_error.png',dpi=170);plt.close(fig)
    sc=pd.DataFrame(correlations);labels=['w1','w2','cw_percentile','q1_50','q2_50','norm_width_combined','cross_endpoint_span'];mat=np.eye(len(labels))
    for (a,b),g in sc.groupby(['signal_a','signal_b']):mat[labels.index(a),labels.index(b)]=mat[labels.index(b),labels.index(a)]=g.groupby('seed').rho.mean().mean()
    fig,ax=plt.subplots(figsize=(7.6,6));im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(len(labels)):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Within-batch Spearman; 5 seed means');ax.set_title('Preselection signal overlap in 180 selected batches (n=32 each)');fig.tight_layout();fig.savefig(OUT/'figures/signals_selected_correlation.png',dpi=170);plt.close(fig)
    summary={'selected_events':len(df),'batches':len(bm),'seeds':SEEDS,'unique_row_ids':int(df.id.nunique()),
      'unique_molecules':int(df.molecule_key.nunique()),'archive_records_hash_verified':archive_checked,
      'catalog_hash_verified':180,'same_batch_single_factor_pairs':len(pairs)//2,'selected_feedback_only':True,
      'raw_candidate_labels_read_by_script':0,'test_truth_reads_by_script':0,
      'test_truth_reads':'not_claimed_session_wide; see label_access_incident.json',
      'training_calls':0,'llm_calls':0,
      'identical_condition_group_events':int(len(du)/2),
      'identical_condition_unique_groups':int(du.group_key.nunique()) if len(du) else 0,
      'eligible_same_batch_three_level_curve_groups':len(qualified),'fixed_rule_curves_plotted':len(fixed),
      'global_selected_stats':{f'V{t}':endpoint_stats(df,t) for t in [1,2]}}
    # Independent scalar recomputation and edge cases for the key residual and AUSE formulas.
    for t in [1,2]:
        scalar=math.sqrt(sum(float(x)**2 for x in df[f'e{t}'])/len(df))
        assert math.isclose(scalar,summary['global_selected_stats'][f'V{t}']['rmse_ml'],rel_tol=1e-12)
    assert signal_metrics(np.arange(32),np.arange(32))['ause']==0
    assert math.isnan(rho(np.ones(4),np.arange(4)))
    assert phase(5)=='333-525' and phase(6)=='525-813' and phase(20)=='813-1005' and phase(21)=='1005-1485'
    assert all(df.groupby('seed').size()==1152)
    for p in [ROOT/'src/qgeognn_al/training/predictor.py',ROOT/'src/qgeognn_al/active_learning_v2/runner.py',ROOT/'src/qgeognn_al/active_learning_v2/benchmark_protocol.py',ROOT/'scripts/run_e0_4g_baseline.py',ROOT/'src/qgeognn_al/schemas/conditions.py',RUN/'protocol.json']:
        add_input(p,role='source_semantics_only')
    for relative in [
        'studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md',
        'studies/diagnostics/quantile_loss_audit_20261109/quantile_metrics.csv',
        'studies/diagnostics/quantile_loss_audit_20261109/incremental.csv',
        'studies/active_learning/representation_gp_benchmark_20261104/COMPARISON_REPORT.md',
        'studies/active_learning/representation_gp_benchmark_20261104/REPRESENTATION_REPORT.md',
        'studies/active_learning/representation_gp_benchmark_20261104/NEXT_STAGE_DECISION.md',
        'studies/active_learning/gp_guided_llm_stability_20261103/OFFLINE_REPORT.md',
        'studies/active_learning/gollum_adaptation_fusion_20261106/PHASE1_RESULTS.md',
        'studies/active_learning/gollum_adaptation_fusion_20261106/NEXT_STAGE_DECISION.md',
        'studies/active_learning/qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md',
        'studies/active_learning/qgeognn_v2_ivr_mechanism_audit/gate_b/FINAL_REPORT.md']:
        add_input(ROOT/relative,role='reused_historical_report_not_new_measurement')
    add_input(Path(__file__).resolve(),role='analysis_script')
    (OUT/'verification/model_analysis_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'verification/model_input_manifest.json').write_text(json.dumps({'inputs':MANIFEST,'semantics':'No source label table opened; catalogs are preselection state and feedback exactly accepted batch. No individual response export.'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

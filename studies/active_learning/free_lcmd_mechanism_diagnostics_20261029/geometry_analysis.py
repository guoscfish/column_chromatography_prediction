"""Reporting-only geometry of frozen Free LLM32 / corrected raw-gradient LCMD32.

Reads no target table, no predictions, no checkpoint contents, and calls no model,
selector, or gradient extractor. Checkpoints are streamed only for SHA-256 audit.
Banks are immutable inputs. Within-state ranks and relative coverage are used
for cross-round summaries; raw distances are retained only for state-level audit.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

# Set before importing numerical libraries to avoid oversubscribing the host.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/free_lcmd_geometry_matplotlib')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
EXT = BASE.parent / 'qgeognn_v2_row_lcmd_to1485_20261026'
REVISIONS = BASE.parent / 'qgeognn_v2_row_llm_scientist_v2/revisions'
FREE = 'free_llm32_scientist_v2'
SEEDS = (157, 887, 2357, 6101, 12203)
OLD = ((21,'sol_xhigh_hybrid_free_to1005_20261013'),(26,'sol_xhigh_all_strategies_to1165_20261014'),
       (30,'sol_xhigh_all_strategies_to1293_20261015'),(34,'sol_xhigh_all_strategies_to1421_20261017'),
       (36,'sol_xhigh_all_strategies_to1549_20261018'))
NEW = ((21,'sol61_xhigh_free_five_seeds_to1005_20261023'),(36,'sol61_xhigh_free_five_seeds_to1485_20261024'))
CUTS = {887:33, 12203:24}
HASHES: dict[str,str] = {}


def relative(path):
    return str(Path(path).relative_to(ROOT))


def sha(path):
    path = Path(path)
    key = relative(path)
    if key not in HASHES:
        with path.open('rb') as handle:
            HASHES[key] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return HASHES[key]


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def owner(seed, fit_round):
    if seed in CUTS and fit_round > CUTS[seed]:
        return REVISIONS / 'sol61_xhigh_free_to1485_extra3_20261026'
    return REVISIONS / next(name for end,name in (OLD if seed in (157,6101) else NEW) if fit_round <= end)


def stage_of(labels):
    for a,b in ((333,525),(525,653),(653,813),(813,1005),(1005,1197),(1197,1485)):
        if a <= labels < b:
            return f'{a}-{b}'
    raise ValueError(labels)


def squared_distances(left, right):
    a,b = np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    return np.maximum(np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None,:]-2*a@b.T,0)


def effective_rank(features):
    """Entropy effective rank from centered singular values, plus participation rank.

    E-rank uses singular-value probabilities (Roy/Vetterli convention); the
    covariance participation rank uses eigenvalues. They are not interchangeable.
    """
    x = np.asarray(features,dtype=np.float64)
    x = x-x.mean(0,keepdims=True)
    gram = x@x.T if len(x) <= x.shape[1] else x.T@x
    eigen = np.maximum(np.linalg.eigvalsh(gram),0)
    if not eigen.max() > 0:
        return 0.,0.,0.
    eigen = eigen[eigen > eigen.max()*1e-12]
    singular = np.sqrt(eigen)
    p = singular/singular.sum()
    return float(np.exp(-(p*np.log(p)).sum())),float(eigen.sum()**2/np.square(eigen).sum()),float(eigen.max()/eigen.sum())


def geometry(features, labeled_count, selected_positions):
    """State metrics, with selected_positions relative to the unlabelled pool."""
    x=np.asarray(features,dtype=np.float64); n=int(labeled_count)
    pool=x[n:]; p=np.asarray(selected_positions,dtype=int); batch=pool[p]
    assert x.ndim==2 and np.isfinite(x).all() and len(set(p))==len(p)==32
    before_sq=squared_distances(pool,x[:n]).min(1)
    before=np.sqrt(before_sq)
    percentile=pd.Series(before).rank(method='average',pct=True).to_numpy()
    # Coverage uses the same preselection U in denominator and numerator.
    # Remaining-U sensitivity excludes selected rows to avoid automatic zeros.
    after_sq=np.minimum(before_sq,squared_distances(pool,batch).min(1))
    after_sq[p]=0.0  # Exact self-distance; avoid BLAS cancellation at selected centers.
    after=np.sqrt(after_sq)
    keep=np.ones(len(pool),dtype=bool);keep[p]=False
    bb=squared_distances(batch,batch);np.fill_diagonal(bb,np.inf)
    nn=np.sqrt(bb.min(1))
    before_mean=float(before.mean());before_sq_mean=float(before_sq.mean())
    bank_rms=float(np.sqrt(np.square(x-x.mean(0,keepdims=True)).sum(1).mean()))
    er,pr,top=effective_rank(x);ber,bpr,btop=effective_rank(batch)
    def ratio(a,b):return float(a/b) if b else None
    result={
        'selected_to_L_distance_mean':float(before[p].mean()),
        'selected_to_L_distance_median':float(np.median(before[p])),
        'selected_to_L_distance_p10':float(np.quantile(before[p],.1)),
        'selected_to_L_distance_p90':float(np.quantile(before[p],.9)),
        'selected_distance_percentile_mean':float(percentile[p].mean()),
        'selected_distance_percentile_median':float(np.median(percentile[p])),
        'selected_distance_percentile_p10':float(np.quantile(percentile[p],.1)),
        'selected_distance_percentile_p90':float(np.quantile(percentile[p],.9)),
        'selected_top_decile_fraction':float((percentile[p]>=.9).mean()),
        'pool_preselection_coverage_distance_mean':before_mean,
        'bank_centered_feature_RMS':bank_rms,
        'pool_coverage_mean_to_bank_RMS':ratio(before_mean,bank_rms),
        'pool_coverage_p50_to_bank_RMS':ratio(np.median(before),bank_rms),
        'pool_coverage_p90_to_bank_RMS':ratio(np.quantile(before,.9),bank_rms),
        'pool_coverage_p99_to_bank_RMS':ratio(np.quantile(before,.99),bank_rms),
        'pool_already_covered_at_1e_minus6_bank_RMS_fraction':float((before<=bank_rms*1e-6).mean()),
        'coverage_gain_to_bank_RMS':ratio(before_mean-after.mean(),bank_rms),
        'remaining_U_coverage_gain_to_bank_RMS':ratio(before[keep].mean()-after[keep].mean(),bank_rms),
        'batch_nn_distance_mean':float(nn.mean()),
        'batch_nn_to_pool_coverage_ratio':ratio(nn.mean(),before_mean),
        'batch_nn_to_selected_L_distance_ratio':ratio(nn.mean(),before[p].mean()),
        'batch_nn_closer_than_L_fraction':float((nn < before[p]).mean()),
        'batch_nn_less_than_half_L_fraction':float((nn < .5*before[p]).mean()),
        'batch_exact_feature_duplicate_fraction':float(1-len(np.unique(batch,axis=0))/len(batch)),
        'coverage_distance_relative_drop':ratio(before_mean-after.mean(),before_mean),
        'coverage_squared_distance_relative_drop':ratio(before_sq_mean-after_sq.mean(),before_sq_mean),
        'remaining_U_coverage_distance_relative_drop':ratio(before[keep].mean()-after[keep].mean(),before[keep].mean()),
        'centered_bank_effective_rank':er,'centered_bank_covariance_participation_rank':pr,
        'centered_bank_top_covariance_fraction':top,
        'centered_batch_effective_rank':ber,'centered_batch_covariance_participation_rank':bpr,
        'centered_batch_top_covariance_fraction':btop,
    }
    points={'to_L_distance':before[p],'distance_percentile':percentile[p],
            'batch_nn_distance':nn,'batch_nn_closer_than_L':(nn<before[p]).astype(int)}
    return result,points


def checkpoint_audit(path, expected):
    assert Path(path).is_file(),path
    assert sha(path)==expected,path


def free_inputs(seed,r,partition):
    selection=owner(seed,r+1)/f'selections/seed_{seed}/{FREE}/round_{r:02d}'
    contract=read(selection/'contract.json'); frozen=read(selection/'batch_freeze.json')
    for name in ('contract.json','catalog.json'):
        assert sha(selection/name)==frozen['artifact_hashes'][name],selection/name
    assert contract['seed']==seed and contract['round']==r
    L,U=contract['L_t_ids'],contract['U_t_ids']; selected=frozen['batch_ids']
    assert len(L)==333+32*r and len(selected)==len(set(selected))==32 and not frozen['pending_ids']
    assert not set(selected)&set(L) and set(selected)<=set(U)
    catalog=read(selection/'catalog.json')
    assert sha(selection/'catalog.json')==contract['catalog_sha256']
    assert [row['id'] for row in catalog['candidates']]==U
    idx=dict(zip(partition.sample_id.astype(str),partition.canonical_index.astype(int)))
    expected=[idx[i] for i in L+U]
    valid=[]
    for stage in dict.fromkeys((owner(seed,r+1),owner(seed,r))):
        bank=stage/f'runtime/seed_{seed}/{FREE}/round_{r:02d}/banks.npz'
        if bank.exists():
            meta=read(bank.with_suffix('.json'))
            if meta['contract']['checkpoint']==contract['checkpoint_sha256'] and meta['contract']['indices']==expected:
                assert sha(bank)==meta['sha256'];valid.append(bank)
    if not valid:
        raise FileNotFoundError(f'No checkpoint/ordered-ID-matched Free bank: seed {seed} round {r}')
    bank=valid[0]
    fitfreeze=read(owner(seed,r)/f'runtime/seed_{seed}/{FREE}/round_{r:02d}/prediction_freeze.json')
    assert fitfreeze['checkpoint_sha256']==contract['checkpoint_sha256']
    assert fitfreeze['L_t_ids_hash']==stable_hash(sorted(L))
    checkpoint_audit(fitfreeze['checkpoint_path'],contract['checkpoint_sha256'])
    with np.load(bank) as z:
        arrays={space:z[space] for space in ('raw','cw')}
    assert all(a.shape==(len(L)+len(U),512) for a in arrays.values())
    pos={i:j for j,i in enumerate(U)};p=np.array([pos[i] for i in selected],dtype=int)
    cw_seen=np.array([row['cw_distance_percentile'] for row in catalog['candidates']],dtype=float)
    audit={'seed':seed,'method':FREE,'round':r,'labels_before':len(L),'selection_directory':relative(selection),
           'bank_path':relative(bank),'bank_metadata_path':relative(bank.with_suffix('.json')),
           'checkpoint_path':relative(fitfreeze['checkpoint_path']),'checkpoint_hash_verified':True,
           'ordered_bank_ids_verified':True,'bank_hash_verified':True,'raw_available':True,'cw_available':True,
           'cw_status':'frozen_transformed_bank','selection_ids_hash':stable_hash(selected)}
    return arrays,len(L),selected,p,cw_seen,audit


def validate_lcmd_bank_contract(bank_contract, acquisition_input, seed, source_round):
    # Bank metadata augments the acquisition input. Every acquisition field,
    # including scales, state/ID hashes and round ownership, must agree.
    required={'outer_seed','method','source_round','acquisition_round','L_t_ids_hash','U_t_ids_hash',
              'evaluation_checkpoint_sha256','evaluation_checkpoint_state_hash','endpoint_scales',
              'batch_size','feature_recomputed_from_current_model','centers_are_current_L_t','test_truth_access_count'}
    assert required <= set(acquisition_input), 'Incomplete LCMD acquisition input'
    assert set(acquisition_input) <= set(bank_contract), 'Bank omits acquisition input fields'
    for key,value in acquisition_input.items():
        assert bank_contract[key]==value, f'LCMD bank/acquisition contract mismatch: {key}'
    expected={'outer_seed':int(seed),'method':'lcmd','source_round':int(source_round),
              'acquisition_round':int(source_round)+1,'batch_size':32,
              'feature_recomputed_from_current_model':True,'centers_are_current_L_t':True,
              'test_truth_access_count':0,'dimension':512,'sketch_seed':int(seed)+4_000_037,
              'kind':'current_full_network_q50_gradient_CountSketch'}
    for key,value in expected.items():
        assert bank_contract[key]==value, f'Unexpected LCMD bank field: {key}'
    assert bank_contract['scales']==[bank_contract['endpoint_scales'][k] for k in ('V1','V2')]


def lcmd_inputs(seed,r,partition):
    state_owner=BASE if r<=21 else EXT
    fitdir=state_owner/f'runtime/seed_{seed}/lcmd/round_{r:02d}'
    directory=(BASE if r<21 else EXT)/f'runtime/seed_{seed}/lcmd/round_{r:02d}'
    acq=directory/'acquisition_artifacts';bank=acq/'current_gradient_features.npz'
    contract=read(acq/'contract.json');meta=read(bank.with_suffix('.npz.contract.json'))
    for name,digest in contract['files'].items():
        # Acquisition artifacts only; no labels or model loading is needed.
        assert sha(directory/name)==digest,directory/name
    assert sha(bank)==meta['sha256']
    c=meta['contract']
    validate_lcmd_bank_contract(c,contract['input'],seed,r)
    assert meta['contract_hash']==stable_hash(c)
    L=np.load(fitdir/'labeled_ids.npy').astype(str).tolist()
    U=np.load(fitdir/'unlabeled_ids.npy').astype(str).tolist()
    assert len(L)==333+32*r and c['L_t_ids_hash']==stable_hash(sorted(L)) and c['U_t_ids_hash']==stable_hash(sorted(U))
    fit=read(fitdir/'model/fit_audit.json')
    checkpoint_audit(fitdir/'model/best.pt',c['evaluation_checkpoint_sha256'])
    assert fit['checkpoint_sha256']==c['evaluation_checkpoint_sha256']
    selected=pd.read_csv(acq/'selected_next_batch.csv').sample_id.astype(str).tolist()
    assert len(selected)==len(set(selected))==32 and contract['selected_ids_ordered_hash']==stable_hash(selected)
    idx=dict(zip(partition.sample_id.astype(str),partition.canonical_index.astype(int)))
    expected=np.array([idx[i] for i in L+U],dtype=int)
    assert c['ordered_indices']==expected.tolist()
    with np.load(bank) as z:
        assert np.array_equal(z['canonical_indices'],expected)
        raw=z['features']
    assert raw.shape==(len(expected),512) and c['kind']=='current_full_network_q50_gradient_CountSketch'
    assert c['centers_are_current_L_t'] and c['feature_recomputed_from_current_model']
    pos={i:j for j,i in enumerate(U)};p=np.array([pos[i] for i in selected],dtype=int)
    trace=pd.read_csv(acq/'lcmd_trace.csv')
    assert np.array_equal(trace.pool_position.to_numpy(),p)
    audit={'seed':seed,'method':'lcmd','round':r,'labels_before':len(L),'selection_directory':relative(acq),
           'bank_path':relative(bank),'bank_metadata_path':relative(bank.with_suffix('.npz.contract.json')),
           'checkpoint_path':relative(fitdir/'model/best.pt'),'checkpoint_hash_verified':True,
           'ordered_bank_ids_verified':True,'bank_hash_verified':True,'raw_available':True,'cw_available':False,
           'cw_status':'not_frozen_on_raw_LCMD_trajectory_no_reconstruction','selection_ids_hash':stable_hash(selected)}
    return {'raw':raw},len(L),selected,p,None,audit


def source_audit():
    frozen=ROOT/'reports/free_lcmd_five_seeds_to1485_20261026/frozen_sources/gradient_features.py'
    current=ROOT/'src/qgeognn_al/active_learning_v2/gradient_features.py'
    protocol=read(BASE/'protocol.json')
    old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(frozen.read_text()).body if isinstance(n,ast.FunctionDef)}
    new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(current.read_text()).body if isinstance(n,ast.FunctionDef)}
    lcmd=ROOT/'src/qgeognn_al/active_learning_v2/lcmd.py'
    return {'frozen_gradient_sha256':sha(frozen),'current_gradient_sha256':sha(current),
            'frozen_gradient_matches_original_protocol':sha(frozen)==protocol['code_hashes'][relative(current)],
            'shared_function_ast_identity':{name:body==new[name] for name,body in old.items()},
            'new_functions_only':sorted(set(new)-set(old)),
            'lcmd_source_sha256':sha(lcmd),'lcmd_matches_original_protocol':sha(lcmd)==protocol['code_hashes'][relative(lcmd)],
            'lcmd_semantics':'L_t installed as initial centers; cluster scores=sum squared nearest-center distances; farthest point in largest cluster; sequential new centers; stable input-order ties',
            'raw_definition':'CountSketch(concat(grad(V1_q50/s_V1), grad(V2_q50/s_V2))); full-network 512D; L0 endpoint scales',
            'cw_definition':'CountSketch(concat(grad((V1_q50+V2_q50)/(2*s_C)), grad((V2_q50-V1_q50)/s_W))); L0 population C/W scales',
            'cw_cannot_be_derived_from_512D_raw_sketch':'Endpoint-specific sketches mix/collide into one 512D vector; no invertible output transform exists on the stored raw sketch alone.'}


def plot(profiles,points,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={FREE:'#da704c','lcmd':'#356d9a'}
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    metrics=[('raw_selected_distance_percentile_mean','Raw: selected-to-L distance percentile'),
             ('raw_remaining_U_coverage_distance_relative_drop','Raw: remaining-U coverage relative drop'),
             ('raw_batch_nn_less_than_half_L_fraction','Raw: within-batch close-neighbor fraction'),
             ('raw_centered_batch_effective_rank','Raw: centered batch effective rank')]
    for ax,(metric,title) in zip(axs.flat,metrics):
        for method in (FREE,'lcmd'):
            subset=profiles[profiles.method.eq(method)]
            for seed,g in subset.groupby('seed'):
                ax.plot(g.labels_before,g[metric],color=colors[method],alpha=.15,lw=.8)
            mean=subset.groupby('labels_before')[metric].mean()
            ax.plot(mean.index,mean.values,color=colors[method],lw=2,label='Free LLM32' if method==FREE else 'LCMD32')
        for b in (653,1005):ax.axvline(b,color='.7',ls=':',lw=.8)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Labels before selection');ax.grid(alpha=.15)
    axs[0,0].set_ylim(0,1);axs[0,0].legend(fontsize=9)
    fig.suptitle('Frozen selection geometry; 5 paired seeds; state-normalized metrics\nDescriptive acquisition diagnostics; validation trajectories only',fontsize=12)
    fig.savefig(out/'geometry_raw_diagnostics.png',dpi=180);fig.savefig(out/'geometry_raw_diagnostics.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,metric,title in zip(axs,['raw_pool_coverage_mean_to_bank_RMS','raw_remaining_U_coverage_gain_to_bank_RMS'],
                               ['Remaining coverage distance / current-bank RMS','Coverage gain / current-bank RMS (exclude selections)']):
        for method in (FREE,'lcmd'):
            subset=profiles[profiles.method.eq(method)]
            for _,g in subset.groupby('seed'):
                ax.plot(g.labels_before,g[metric],color=colors[method],alpha=.15,lw=.8)
            mean=subset.groupby('labels_before')[metric].mean()
            ax.plot(mean.index,mean.values,color=colors[method],lw=2,label='Free LLM32' if method==FREE else 'LCMD32')
        for b in (653,1005):ax.axvline(b,color='.7',ls=':',lw=.8)
        ax.set_title(title,fontsize=9);ax.set_xlabel('Labels before selection');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8)
    fig.suptitle('Raw geometry: coverage denominators and gradual exhaustion\n5 paired seeds; each frozen model normalized separately; validation trajectories only',fontsize=11)
    fig.savefig(out/'geometry_coverage_denominator_sensitivity.png',dpi=180)
    fig.savefig(out/'geometry_coverage_denominator_sensitivity.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    free=profiles[profiles.method.eq(FREE)]
    for seed,g in free.groupby('seed'):
        axs[0].plot(g.labels_before,g.cw_selected_distance_percentile_mean,alpha=.3,lw=.9,label=str(seed))
    g=free.groupby('labels_before').cw_selected_distance_percentile_mean.mean()
    axs[0].plot(g.index,g.values,color='#a54325',lw=2.2,label='Mean, 5 seeds')
    axs[0].axhline(.5,color='.5',ls=':',lw=.8);axs[0].set_ylim(0,1)
    axs[0].set_title('Free selected CW distance percentile');axs[0].legend(fontsize=7,ncol=2)
    stage_order=['333-525','525-653','653-813','813-1005','1005-1197','1197-1485']
    values=[points[(points.method.eq(FREE))&(points.stage.eq(s))].cw_distance_percentile.to_numpy() for s in stage_order]
    axs[1].boxplot(values,tick_labels=stage_order,showfliers=False);axs[1].tick_params(axis='x',rotation=35)
    axs[1].set_ylim(0,1);axs[1].set_title('CW percentiles by selection stage (pooled picks)')
    for ax in axs:ax.grid(alpha=.15)
    axs[0].set_xlabel('Labels before selection');axs[1].set_xlabel('Descriptive distributions; no point-level inference')
    fig.suptitle('Free CW is an input signal; LCMD uses raw gradients\n5 seeds; frozen selection states; validation trajectories only',fontsize=11)
    fig.savefig(out/'geometry_free_cw_percentiles.png',dpi=180);fig.savefig(out/'geometry_free_cw_percentiles.svg');plt.close(fig)


def run(out=OUT,plots=True):
    out.mkdir(parents=True,exist_ok=True)
    profiles=[];points=[];audits=[]
    for seed in SEEDS:
        partition=pd.read_csv(BASE/f'splits/row_seed_{seed}.csv')
        # Split contains IDs, role, canonical index only; no responses are read.
        for method,reader in ((FREE,free_inputs),('lcmd',lcmd_inputs)):
            for r in range(36):
                arrays,n,ids,p,cw_seen,audit=reader(seed,r,partition)
                row={'seed':seed,'method':method,'round':r,'labels_before':n,'labels_after':n+32,'stage':stage_of(n),'batch_size':32,'pool_count':len(arrays['raw'])-n}
                selected=[dict(seed=seed,method=method,round=r,labels_before=n,stage=stage_of(n),selection_order=j,sample_id=i) for j,i in enumerate(ids)]
                for space,features in arrays.items():
                    metrics,point_metrics=geometry(features,n,p)
                    row.update({space+'_'+k:v for k,v in metrics.items()})
                    for k,values in point_metrics.items():
                        for j,value in enumerate(values):selected[j][space+'_'+k]=float(value)
                    if space=='cw':
                        # Recalculate the visible full-pool rank without rounding.
                        d=np.sqrt(squared_distances(features[n:],features[:n]).min(1))
                        ranks=pd.Series(d).rank(method='average',pct=True).to_numpy()
                        deviation=float(np.max(np.abs(ranks-cw_seen)))
                        assert deviation<=5.1e-6,(seed,r,deviation)
                        audit['cw_catalog_percentiles_verified']=True
                        audit['cw_catalog_percentile_max_rounding_difference']=deviation
                row.update(raw_bank_available=True,cw_bank_available='cw' in arrays)
                profiles.append(row);points.extend(selected);audits.append(audit)
            print(f'Geometry complete: seed={seed} method={method}',flush=True)
    profiles=pd.DataFrame(profiles);points=pd.DataFrame(points)
    assert len(profiles)==360 and len(points)==11520
    assert profiles.groupby(['seed','method']).size().eq(36).all()
    profiles.to_csv(out/'geometry_profiles.csv',index=False)
    points.to_csv(out/'geometry_selected_points.csv',index=False)
    metrics=[c for c in profiles if c.startswith(('raw_','cw_')) and c not in ('raw_bank_available','cw_bank_available')]
    summaries=profiles.groupby(['seed','method','stage'])[metrics].mean().reset_index()
    summaries.to_csv(out/'geometry_stage_by_seed.csv',index=False)
    summaries.groupby(['method','stage'])[metrics].agg(['mean','std']).to_csv(out/'geometry_stage_summary.csv')
    available={'analysis_type':'reporting_only_frozen_features','seeds':list(SEEDS),'expected_batches':360,
               'verified_batches':len(audits),'verified_raw_banks':sum(a['raw_available'] for a in audits),
               'verified_cw_banks':sum(a['cw_available'] for a in audits),'verified_checkpoint_hashes':sum(a['checkpoint_hash_verified'] for a in audits),
               'missing_raw_banks':[],'missing_cw_banks':[{'seed':a['seed'],'method':a['method'],'round':a['round'],'reason':a['cw_status']} for a in audits if not a['cw_available']],
               'no_gradient_recomputation':True,'no_model_training':True,'no_model_inference':True,'no_selector_rerun':True,
               'no_unselected_or_test_response_table_read':True,'source_audit':source_audit(),'batches':audits,
               'source_sha256':HASHES}
    (out/'geometry_availability.json').write_text(json.dumps(available,indent=2,ensure_ascii=False)+'\n')
    if plots:plot(profiles,points,out)
    return profiles,points,available


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();run(args.out,not args.no_plots)

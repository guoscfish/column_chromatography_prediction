"""Read-only geometry audit of frozen Scientist V2 and independent CW/raw controls.

No model, selector, gradient extractor, target table, or prediction table is loaded.
Checkpoints are streamed only to verify their SHA-256. All coverage counterfactuals
use the SAME frozen model/feature bank and preselection U within each state.
Cross-method comparisons after round 0 are observational, at matched budgets.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hybrid_geometry_matplotlib')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AL = ROOT / 'studies/active_learning'
SCIENTIST = AL / 'qgeognn_v2_row_llm_scientist_v2'
ASTRA = SCIENTIST / 'revisions/astra_high_full_pool_20261001'
SOL = SCIENTIST / 'revisions/free_sol_xhigh_continuation_20261012'
SEEDS = (157, 6101)
HYBRID = 'cw16_llm16_scientist_v2'
FREE = 'free_llm32_scientist_v2'
SOURCES = ('hybrid_cw16','hybrid_llm16','hybrid_all32','free32','cw_lcmd32','raw_lcmd32')
HASHES = {}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


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


def stage_of(n):
    return '333-525' if n < 525 else '525-653' if n < 653 else '653-813'


def squared_distances(left, right):
    a, b = np.asarray(left,dtype=np.float64), np.asarray(right,dtype=np.float64)
    return np.maximum(np.square(a).sum(1)[:,None] + np.square(b).sum(1)[None,:] - 2*a@b.T, 0.)


def effective_rank(features):
    x = np.asarray(features,dtype=np.float64)
    x = x-x.mean(0,keepdims=True)
    gram = x@x.T if len(x)<=x.shape[1] else x.T@x
    eigen = np.maximum(np.linalg.eigvalsh(gram),0)
    if not eigen.max()>0:
        return 0.,0.,0.
    eigen = eigen[eigen>eigen.max()*1e-12]
    singular = np.sqrt(eigen)
    p = singular/singular.sum()
    return float(np.exp(-(p*np.log(p)).sum())),float(eigen.sum()**2/np.square(eigen).sum()),float(eigen.max()/eigen.sum())


def ratio(a,b):
    return float(a/b) if b else None


def state_geometry(features, labeled_count):
    x = np.asarray(features,dtype=np.float64)
    n = int(labeled_count)
    assert x.ndim==2 and 0<n<len(x) and np.isfinite(x).all()
    before_sq = squared_distances(x[n:],x[:n]).min(1)
    before = np.sqrt(before_sq)
    return dict(x=x,n=n,pool=x[n:],before=before,before_sq=before_sq,
                percentile=pd.Series(before).rank(method='average',pct=True).to_numpy(),
                rms=float(np.sqrt(np.square(x-x.mean(0,keepdims=True)).sum(1).mean())),
                bank_rank=effective_rank(x))


def postselection_distance(state, selected_positions):
    p = np.asarray(selected_positions,dtype=int)
    assert len(p)>0 and len(set(p))==len(p) and min(p)>=0 and max(p)<len(state['pool'])
    after_sq = np.minimum(state['before_sq'],squared_distances(state['pool'],state['pool'][p]).min(1))
    after_sq[p]=0.  # Exact selected-center distance, without BLAS cancellation.
    return np.sqrt(after_sq)


def geometry(features, labeled_count, selected_positions, *, common_excluded=None, other_positions=None, state=None):
    """Variable-size generalization of the five-seed metrics, including 16-pick sets.

    Fixed full-U drop aligns with the previous report. remaining_U excludes this
    source's selected points. common_remaining_U excludes the full Hybrid32 union
    for BOTH sources, permitting additive same-denominator marginal attribution.
    Marginal gains are scaled by the PRESELECTION mean, not by the other's mean.
    """
    state = state or state_geometry(features,labeled_count)
    p = np.asarray(selected_positions,dtype=int)
    before, before_sq, pool = state['before'],state['before_sq'],state['pool']
    after = postselection_distance(state,p)
    batch = pool[p]
    keep = np.ones(len(pool),dtype=bool); keep[p]=False
    common = np.ones(len(pool),dtype=bool)
    common[np.asarray(p if common_excluded is None else common_excluded,dtype=int)]=False
    assert keep.any() and common.any()
    bb = squared_distances(batch,batch); np.fill_diagonal(bb,np.inf)
    nn = np.sqrt(bb.min(1)) if len(p)>1 else np.full(1,np.nan)
    mean = float(before.mean()); sqmean=float(before_sq.mean())
    er,pr,top=state['bank_rank']; ber,bpr,btop=effective_rank(batch)
    rank=state['percentile'][p]
    result={
        'selected_to_L_distance_mean':float(before[p].mean()),
        'selected_to_L_distance_median':float(np.median(before[p])),
        'selected_to_L_distance_p10':float(np.quantile(before[p],.1)),
        'selected_to_L_distance_p90':float(np.quantile(before[p],.9)),
        'selected_distance_percentile_mean':float(rank.mean()),
        'selected_distance_percentile_median':float(np.median(rank)),
        'selected_distance_percentile_p10':float(np.quantile(rank,.1)),
        'selected_distance_percentile_p90':float(np.quantile(rank,.9)),
        'selected_top_decile_fraction':float((rank>=.9).mean()),
        'pool_preselection_coverage_distance_mean':mean,
        'bank_centered_feature_RMS':state['rms'],
        'pool_coverage_mean_to_bank_RMS':ratio(mean,state['rms']),
        'coverage_distance_relative_drop':ratio(mean-after.mean(),mean),
        'coverage_squared_distance_relative_drop':ratio(sqmean-np.square(after).mean(),sqmean),
        'coverage_gain_to_bank_RMS':ratio(mean-after.mean(),state['rms']),
        'remaining_U_coverage_distance_relative_drop':ratio(before[keep].mean()-after[keep].mean(),before[keep].mean()),
        'remaining_U_coverage_gain_to_bank_RMS':ratio(before[keep].mean()-after[keep].mean(),state['rms']),
        'common_remaining_U_coverage_distance_relative_drop':ratio(before[common].mean()-after[common].mean(),before[common].mean()),
        'mechanical_selected_zeroing_drop_component':ratio(before[p].sum()/len(before),mean),
        'nonselected_drop_component_full_U_denominator':ratio((before[keep]-after[keep]).sum()/len(before),mean),
        'selected_zeroing_share_of_total_gain':ratio(before[p].sum(),(before-after).sum()),
        'batch_nn_distance_mean':float(nn.mean()),
        'batch_nn_to_pool_coverage_ratio':ratio(nn.mean(),mean),
        'batch_nn_to_selected_L_distance_ratio':ratio(nn.mean(),before[p].mean()),
        'batch_nn_closer_than_L_fraction':float((nn<before[p]).mean()),
        'batch_nn_less_than_half_L_fraction':float((nn<.5*before[p]).mean()),
        'batch_exact_feature_duplicate_fraction':float(1-len(np.unique(batch,axis=0))/len(batch)),
        'centered_bank_effective_rank':er,'centered_bank_covariance_participation_rank':pr,
        'centered_bank_top_covariance_fraction':top,'centered_batch_effective_rank':ber,
        'centered_batch_covariance_participation_rank':bpr,'centered_batch_top_covariance_fraction':btop,
    }
    pts={'to_L_distance':before[p],'distance_percentile':rank,'batch_nn_distance':nn}
    if other_positions is not None:
        other=np.asarray(other_positions,dtype=int)
        assert not set(p)&set(other)
        after_other=postselection_distance(state,other)
        union=np.minimum(after,after_other)
        marginal=after_other-union
        result.update({
            'marginal_after_other_relative_to_original_coverage':ratio(marginal.mean(),mean),
            'common_remaining_U_marginal_after_other_relative_to_original_coverage':ratio(marginal[common].mean(),before[common].mean()),
            'fraction_of_solo_gain_retained_after_other':ratio(marginal.mean(),(before-after).mean()),
            'cross_source_nn_closer_than_L_fraction':float((np.sqrt(squared_distances(batch,pool[other]).min(1))<before[p]).mean()),
        })
        pts['other_source_nn_distance']=np.sqrt(squared_distances(batch,pool[other]).min(1))
    return result,pts


def verify_ids(L,U,selected,n):
    assert len(L)==333+32*n and len(selected)==len(set(selected))==32
    assert len(L)==len(set(L)) and len(U)==len(set(U)) and not set(L)&set(U)
    assert set(selected)<=set(U)


def validate_bank_metadata(meta, expected_checkpoint, expected_indices):
    """Strict method-local checkpoint and ordered L+U match; no fallback state."""
    assert meta['contract']['checkpoint']==expected_checkpoint,'checkpoint mismatch'
    assert meta['contract']['indices']==expected_indices,'ordered L+U mismatch'


def llm_inputs(seed,r,method,partition):
    selection_owner = ASTRA if method==HYBRID or r<({157:6,6101:10}[seed]) else SOL
    directory=selection_owner/f'selections/seed_{seed}/{method}/round_{r:02d}'
    c=read(directory/'contract.json');f=read(directory/'batch_freeze.json')
    assert c['seed']==seed and c['round']==r and c['method']==method
    for fn in ('contract.json','catalog.json','input.json','selection.json'):
        assert sha(directory/fn)==f['artifact_hashes'][fn],directory/fn
    assert sha(directory/'catalog.json')==c['catalog_sha256']
    assert sha(directory/'input.json')==c['input_sha256']
    L,U,selected=c['L_t_ids'],c['U_t_ids'],f['batch_ids']
    verify_ids(L,U,selected,r)
    pending=f['pending_ids']
    assert len(pending)==(16 if method==HYBRID else 0)
    assert selected[:len(pending)]==pending
    cat=read(directory/'catalog.json')
    visible=cat['candidates']+cat['pending']
    byid={row['id']:row for row in visible}
    assert len(byid)==len(U) and set(byid)==set(U)
    assert [row['id'] for row in cat['candidates']]==[i for i in U if i not in set(pending)]
    idx=dict(zip(partition.sample_id.astype(str),partition.canonical_index.astype(int)))
    expected=[idx[i] for i in L+U]
    runtime=selection_owner/f'runtime/seed_{seed}/{method}/round_{r:02d}'
    bank=runtime/'banks.npz'
    fit=read(runtime/'prediction_freeze.json')
    assert fit['checkpoint_sha256']==c['checkpoint_sha256'] and fit['L_t_ids_hash']==stable_hash(sorted(L))
    assert fit['test_truth_access_count']==0
    assert sha(fit['checkpoint_path'])==c['checkpoint_sha256']
    audit=dict(seed=seed,method='hybrid' if method==HYBRID else 'free32',round=r,
               labels_before=len(L),selection_directory=relative(directory),
               checkpoint_path=relative(fit['checkpoint_path']),checkpoint_sha256=c['checkpoint_sha256'],
               checkpoint_hash_verified=True,L_ordered_hash=stable_hash(L),U_ordered_hash=stable_hash(U),
               L_set_hash=stable_hash(sorted(L)),U_set_hash=stable_hash(sorted(U)),
               test_truth_access_count=0,raw_available=False,cw_available=False)
    arrays={}
    if bank.exists():
        md=read(bank.with_suffix('.json'))
        validate_bank_metadata(md,c['checkpoint_sha256'],expected)
        assert sha(bank)==md['sha256']
        with np.load(bank) as z:
            arrays={space:z[space] for space in ('raw','cw')}
        assert all(x.shape==(len(expected),512) for x in arrays.values())
        audit.update(bank_path=relative(bank),bank_metadata_path=relative(bank.with_suffix('.json')),
                     ordered_bank_ids_verified=True,bank_hash_verified=True,raw_available=True,cw_available=True)
    else:
        audit.update(bank_status='NA: own method frozen bank absent; no reconstruction or other-method substitution')
    positions={i:j for j,i in enumerate(U)}
    groups={'hybrid_cw16':pending,'hybrid_llm16':selected[16:],'hybrid_all32':selected} if method==HYBRID else {'free32':selected}
    groups={k:np.array([positions[i] for i in ids],dtype=int) for k,ids in groups.items()}
    seen=np.array([byid[i]['cw_distance_percentile'] for i in U],dtype=float)
    extra={}
    if method==HYBRID:
        cw_mol={byid[i]['smiles'] for i in pending}; llm_mol={byid[i]['smiles'] for i in selected[16:]}
        extra={'cross_source_molecule_intersection_count':len(cw_mol&llm_mol),
               'llm_points_same_molecule_as_cw_fraction':sum(byid[i]['smiles'] in cw_mol for i in selected[16:])/16,
               'cw_points_same_molecule_as_llm_fraction':sum(byid[i]['smiles'] in llm_mol for i in pending)/16,
               'cross_source_molecule_jaccard':len(cw_mol&llm_mol)/len(cw_mol|llm_mol)}
    return arrays,L,U,selected,groups,seen,audit,extra


def cw_inputs(seed,r,partition):
    name='qgeognn_v2_row_short_sequential_b32' if r<3 else 'qgeognn_v2_row_cw_lcmd_to_525' if r<6 else 'qgeognn_v2_row_cw_lcmd_to_653' if r<10 else 'qgeognn_v2_row_cw_lcmd_to_1005'
    d=AL/name/f'runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
    acq=d/'acquisition_artifacts';c=read(acq/'contract.json');f=read(d/'round_freeze.json')
    path=d/'state.csv';sha(path);state=pd.read_csv(path)
    L=state[state.role.eq('labeled')].sort_values('position').sample_id.astype(str).tolist()
    U=state[state.role.eq('unlabeled')].sort_values('position').sample_id.astype(str).tolist()
    table=acq/'selected_next_batch.csv';sha(table);selected=pd.read_csv(table).sample_id.astype(str).tolist()
    verify_ids(L,U,selected,r)
    assert c['selected_ids_ordered_hash']==stable_hash(selected)
    assert c['selected_table_sha256']==sha(table)
    assert f['outgoing_selected_ids_hash']==stable_hash(selected)
    assert f['checkpoint_sha256']==c['input']['checkpoint_sha256']
    assert sha(f['checkpoint_path'])==f['checkpoint_sha256']
    for h,ids in [('L_t_ids_hash',L),('U_t_ids_hash',U)]:
        assert c['input'][h]==f['input'][h]==stable_hash(sorted(ids))
    assert c['input']['centers_are_all_current_L_t'] and c['input']['current_model_round']==r
    assert c['input']['outer_seed']==seed and c['input']['source_round']==r and c['input']['batch_size']==32
    assert c['input']['test_truth_access_count']==0
    bank=acq/('current_transformed_gradient_features.npz' if r<3 else 'current_center_width_gradient_features.npz')
    audit=dict(seed=seed,method='center_width_lcmd',round=r,labels_before=len(L),selection_directory=relative(acq),
               checkpoint_path=relative(f['checkpoint_path']),checkpoint_sha256=f['checkpoint_sha256'],
               checkpoint_hash_verified=True,L_ordered_hash=stable_hash(L),U_ordered_hash=stable_hash(U),
               L_set_hash=stable_hash(sorted(L)),U_set_hash=stable_hash(sorted(U)),test_truth_access_count=0,
               raw_available=False,cw_available=False,raw_status='NA: no raw bank frozen on independent CW trajectory; not borrowed')
    arrays={}
    if bank.exists():
        md=read(bank.with_suffix('.npz.contract.json'));mc=md['contract']
        idx=dict(zip(partition.sample_id.astype(str),partition.canonical_index.astype(int)))
        expected=[idx[i] for i in L+U]
        assert mc['ordered_current_indices']==expected and mc['ordered_current_ids_hash']==stable_hash(L+U)
        assert mc['checkpoint_sha256']==f['checkpoint_sha256']
        assert mc['L_t_ids_hash']==stable_hash(sorted(L)) and mc['U_t_ids_hash']==stable_hash(sorted(U))
        assert mc['source_round']==r and mc['method']=='center_width_lcmd' and mc['test_truth_access_count']==0
        assert sha(bank)==md['sha256']
        with np.load(bank) as z:
            assert z['canonical_indices'].tolist()==expected
            arrays['cw']=z['features']
        assert arrays['cw'].shape==(len(L)+len(U),512)
        audit.update(bank_path=relative(bank),bank_metadata_path=relative(bank.with_suffix('.npz.contract.json')),
                     ordered_bank_ids_verified=True,bank_hash_verified=True,cw_available=True)
    else:
        audit.update(cw_status='NA: own frozen CW bank absent; no reconstruction')
    pos={i:j for j,i in enumerate(U)}
    return arrays,L,U,selected,{'cw_lcmd32':np.array([pos[i] for i in selected],dtype=int)},None,audit,{}


def raw_inputs(seed,r,partition):
    """Use only the prior audit's read-only raw LCMD loader; NEVER its main()."""
    script=AL/'free_lcmd_mechanism_diagnostics_20261029/geometry_analysis.py'
    spec=importlib.util.spec_from_file_location('five_seed_frozen_geometry',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    arrays,n,selected,p,seen,audit=mod.lcmd_inputs(seed,r,partition)
    HASHES.update(mod.HASHES);sha(script)
    runtime=AL/f'qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/lcmd/round_{r:02d}'
    for fn in ('labeled_ids.npy','unlabeled_ids.npy'):sha(runtime/fn)
    L=np.load(runtime/'labeled_ids.npy').astype(str).tolist();U=np.load(runtime/'unlabeled_ids.npy').astype(str).tolist()
    verify_ids(L,U,selected,r)
    audit.update(method='raw_gradient_lcmd',L_ordered_hash=stable_hash(L),U_ordered_hash=stable_hash(U),
                 L_set_hash=stable_hash(sorted(L)),U_set_hash=stable_hash(sorted(U)),test_truth_access_count=0,
                 checkpoint_sha256=sha(audit['checkpoint_path'] if Path(audit['checkpoint_path']).is_absolute() else ROOT/audit['checkpoint_path']))
    return arrays,L,U,selected,{'raw_lcmd32':p},seen,audit,{}


def plot(profiles,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'hybrid_cw16':'Hybrid CW16','hybrid_llm16':'Hybrid LLM16','hybrid_all32':'Hybrid all32',
            'free32':'Free LLM32','cw_lcmd32':'CW-LCMD32','raw_lcmd32':'Raw LCMD32'}
    colors=dict(zip(SOURCES,['#23658b','#cb6b38','#603f83','#ca3f56','#397d53','#666666']))
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    metrics=[('cw_selected_distance_percentile_mean','CW: selected-to-L percentile'),
             ('cw_common_remaining_U_coverage_distance_relative_drop','CW: coverage drop, selected32 excluded'),
             ('raw_selected_distance_percentile_mean','Raw: selected-to-L percentile'),
             ('raw_common_remaining_U_coverage_distance_relative_drop','Raw: coverage drop, selected32 excluded')]
    for ax,(metric,title) in zip(axs.flat,metrics):
        for source in SOURCES:
            g=profiles[profiles.source.eq(source)]
            if metric not in g or g[metric].notna().sum()==0:continue
            for _,s in g.groupby('seed'):ax.plot(s.labels_before,s[metric],color=colors[source],lw=.7,alpha=.25)
            mean=g.groupby('labels_before')[metric].mean()
            ax.plot(mean.index,mean.values,color=colors[source],lw=1.8,label=labels[source])
        ax.set_title(title,fontsize=10);ax.set_xlabel('Labels before selection');ax.grid(alpha=.2)
        for x in (525,653):ax.axvline(x,color='.7',lw=.7,ls=':')
    axs[0,0].set_ylim(0,1);axs[1,0].set_ylim(0,1);axs[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Frozen method-local geometry, matched label budgets\nn = 2 seeds; repeated rounds are not independent replicates; descriptive comparisons',fontsize=11)
    fig.savefig(out/'geometry_division_of_labor.png',dpi=180);fig.savefig(out/'geometry_division_of_labor.svg');plt.close(fig)
    stages=['333-525','525-653','653-813']
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    h=profiles[profiles.source.isin(['hybrid_cw16','hybrid_llm16'])]
    for ax,space in zip(axs,['cw','raw']):
        for i,source in enumerate(['hybrid_cw16','hybrid_llm16']):
            for j,kind in enumerate(['common_remaining_U_coverage_distance_relative_drop','common_remaining_U_marginal_after_other_relative_to_original_coverage']):
                col=space+'_'+kind
                vals=h[h.source.eq(source)].groupby('stage')[col].mean().reindex(stages)
                x=np.arange(3)+(i*2+j-1.5)*.19
                ax.bar(x,vals,width=.18,color=colors[source],alpha=1 if j==0 else .45,
                       label=labels[source]+(' alone' if j==0 else ' after other'))
                for _,sub in h[h.source.eq(source)].groupby('seed'):
                    v=sub.groupby('stage')[col].mean().reindex(stages)
                    ax.scatter(x,v,color='black',s=9,zorder=3)
        ax.set_xticks(range(3),stages);ax.set_title(space.upper()+' same-state coverage attribution')
        ax.set_ylabel('Relative mean-distance drop, common remaining U');ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=7,ncol=2)
    fig.suptitle('CW16 / LLM16 solo and marginal contributions; union32 excluded\nn = 2 seeds (dots); rounds within stages are not independent replicates',fontsize=11)
    fig.savefig(out/'geometry_stage_marginal_coverage.png',dpi=180);fig.savefig(out/'geometry_stage_marginal_coverage.svg');plt.close(fig)


def markdown(frame):
    """Small dependency-free numeric Markdown table for the local Python env."""
    f=frame.reset_index()
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return 'NA' if pd.isna(value) else f'{value:.4f}'
        return str(value).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(str,f.columns))+' |',
                      '| '+' | '.join(['---']*len(f.columns))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])


def findings(profiles,audits,out):
    metrics=['cw_selected_distance_percentile_mean','raw_selected_distance_percentile_mean',
             'cw_common_remaining_U_coverage_distance_relative_drop','raw_common_remaining_U_coverage_distance_relative_drop']
    summary=profiles.groupby(['seed','source'])[metrics].mean()
    hmean=profiles[profiles.source.isin(['hybrid_cw16','hybrid_llm16'])].groupby(['seed','source']).mean(numeric_only=True)
    quick=[]
    for seed in SEEDS:
        cw=hmean.loc[(seed,'hybrid_cw16')];llm=hmean.loc[(seed,'hybrid_llm16')]
        total=profiles[profiles.seed.eq(seed)&profiles.source.eq('hybrid_all32')].cw_common_remaining_U_coverage_distance_relative_drop.mean()
        share=.5*(cw.cw_common_remaining_U_coverage_distance_relative_drop+cw.cw_common_remaining_U_marginal_after_other_relative_to_original_coverage)/total
        quick.append(f'- Seed {seed}: CW16 receives {100*share:.1f}% of Hybrid CW coverage gain under the symmetric mean of its solo and after-LLM marginal contributions. LLM16 retains {100*llm.cw_common_remaining_U_marginal_after_other_relative_to_original_coverage/llm.cw_common_remaining_U_coverage_distance_relative_drop:.1f}% of its solo gain after CW16. This is only a within-bank distance decomposition.')
    lines=['# Frozen geometry findings','',
           'Scope: Scientist V2, seeds 157 / 6101, acquisition rounds 0–14 (333→813 labels). All banks are method-local frozen inputs. No training, inference, selection, gradient reconstruction, or unseen/test target-table read occurred.',
           '', '## Main observations','',
           '- CW16 consistently selects high-distance points (mean CW percentiles 0.9408 / 0.9573 for seeds 157 / 6101) and supplies most Hybrid geometric coverage. The LLM16 component has lower mean percentiles (0.6804 / 0.6087), with a positive extra coverage gain after CW16.',
           '- LLM points sharing any molecule with the current CW16 are only 9.58% / 7.92%. This does not support a claim of widespread direct molecular overlap between the two halves.',
           '- Whole-Hybrid relative coverage exceeds Free in both seeds over 333–813, yet the late 653–813 interval reverses in seed 6101 (CW-space common remaining-U drop 0.0351 for Hybrid vs 0.0396 for Free). A uniform superiority claim is unsupported.',
           '- The late Hybrid LLM16 becomes local in gradient space, especially seed 6101: CW percentile 0.3574 and marginal after-CW relative coverage gain 0.0023 at 653–813. Geometry therefore supports a changing division of labor, not a claim that Hybrid LLM consistently explores more than Free.',
           *quick,
           '', '## Per-seed mean of round metrics','',markdown(summary),'',
           'Values compare ranks or relative within-state coverage. They do not put different models in a common geometry; cross-method comparisons after round 0 remain matched-budget observations. Each seed has 15 dependent rounds, not 15 independent replicates.',
           '', '## Same-state Hybrid attribution','']
    cols=['cw_common_remaining_U_coverage_distance_relative_drop',
          'cw_common_remaining_U_marginal_after_other_relative_to_original_coverage',
          'raw_common_remaining_U_coverage_distance_relative_drop',
          'raw_common_remaining_U_marginal_after_other_relative_to_original_coverage']
    h=profiles[profiles.source.isin(['hybrid_cw16','hybrid_llm16'])]
    lines+=[markdown(h.groupby(['seed','source'])[cols].mean()),'']
    lines+=['“Alone” adds a source to L_t. “After other” adds it after the other source on the same frozen bank. Both use U_t minus all32 as the same denominator and the original L_t coverage as the same normalizer. This is a geometric decomposition, not a model-performance causal effect. Solo gains overlap and must not be added; one solo plus the other marginal equals all32.',
            '', '## Stage detail','',markdown(h.groupby(['seed','stage','source'])[cols].mean()),'',
            '## Molecular overlap','',markdown(profiles[profiles.source.eq('hybrid_all32')].groupby('seed')[['cross_source_molecule_intersection_count','llm_points_same_molecule_as_cw_fraction','cw_points_same_molecule_as_llm_fraction','cross_source_molecule_jaccard']].mean()),'',
            'Overlap refers to molecules, not necessarily duplicate conditions or redundant experimental value. Geometry can show a close gradient region, but cannot establish that any selected experiment was unnecessary.',
            '', '## Coverage denominator sensitivity','',
            markdown(profiles.groupby(['seed','source'])[['cw_coverage_distance_relative_drop','cw_remaining_U_coverage_distance_relative_drop','cw_common_remaining_U_coverage_distance_relative_drop','cw_selected_zeroing_share_of_total_gain','raw_coverage_distance_relative_drop','raw_remaining_U_coverage_distance_relative_drop','raw_common_remaining_U_coverage_distance_relative_drop','raw_selected_zeroing_share_of_total_gain']].mean()),'',
            'Full-U coverage includes exact zero distance at chosen centers; `mechanical_selected_zeroing_drop_component` records that contribution explicitly. `remaining_U` excludes the source selection, matching the five-seed report. `common_remaining_U` excludes the entire32 for the two16 sources, ensuring a common denominator for attribution. Differences in small batch sizes and denominator alone do not show sample usefulness.',
            '', '## Availability and interpretation boundaries','',
            f'- Verified frozen states: {len(audits)}. Raw banks: {sum(a["raw_available"] for a in audits)}; CW banks: {sum(a["cw_available"] for a in audits)}.',
            '- Hybrid and Free each have raw and CW frozen banks. Independent CW-LCMD32 has CW only. Independent raw LCMD32 has raw only; unavailable spaces are NA and are never borrowed or reconstructed.',
            '- CW catalog ranks are recalculated on the full preselection U including pending16, then compared to catalog.candidates + catalog.pending; five-decimal rounding tolerance is 5.1e-6.',
            '- Effective ranks use centered singular-value entropy (and separately covariance participation rank), with no inference from 16-vs32 rank magnitudes alone.',
            '- No claim about current-model true high-error regions is made from feature distances. Such evidence requires already measured premeasurement residuals and is addressed separately.',
            '- Geometry does not identify loading as the largest difference or show that a loading cap will improve validation AULC. It provides an observational check of coverage allocation; loading behavior and paired prospective arms are needed for the scientific question.', '']
    (out/'GEOMETRY_FINDINGS.md').write_text('\n'.join(lines))


def run(out=OUT,plots=True):
    out.mkdir(parents=True,exist_ok=True)
    sha(Path(__file__))
    test_path=OUT/'test_geometry_analysis.py'
    if test_path.exists():sha(test_path)
    profiles=[];points=[];audits=[]
    for seed in SEEDS:
        path=AL/f'qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv'
        sha(path);partition=pd.read_csv(path)
        for method,reader in [(HYBRID,llm_inputs),(FREE,llm_inputs),('cw_lcmd32',cw_inputs),('raw_lcmd32',raw_inputs)]:
            for r in range(15):
                args=(seed,r,method,partition) if reader==llm_inputs else (seed,r,partition)
                arrays,L,U,ids,groups,seen,audit,extra=reader(*args)
                cached={space:state_geometry(x,len(L)) for space,x in arrays.items()}
                if 'cw' in cached and seen is not None:
                    delta=float(np.max(np.abs(cached['cw']['percentile']-seen)))
                    assert delta<=5.1e-6,(seed,r,method,delta)
                    audit.update(cw_catalog_percentiles_verified=True,cw_catalog_percentile_max_rounding_difference=delta)
                positions={i:j for j,i in enumerate(U)};union=np.array([positions[i] for i in ids],dtype=int)
                for source,p in groups.items():
                    method_name={'cw_lcmd32':'center_width_lcmd','raw_lcmd32':'raw_gradient_lcmd'}.get(method,method)
                    row=dict(seed=seed,method=method_name,source=source,round=r,labels_before=len(L),labels_after=len(L)+32,
                             stage=stage_of(len(L)),batch_size=len(p),pool_count=len(U),raw_bank_available='raw' in arrays,cw_bank_available='cw' in arrays,**extra)
                    selected=[dict(seed=seed,source=source,round=r,stage=stage_of(len(L)),labels_before=len(L),selection_order=j,sample_id=U[k],pool_position=int(k)) for j,k in enumerate(p)]
                    other=groups.get('hybrid_llm16' if source=='hybrid_cw16' else 'hybrid_cw16') if source in ['hybrid_cw16','hybrid_llm16'] else None
                    for space,features in arrays.items():
                        metrics,pt=geometry(features,len(L),p,common_excluded=union,other_positions=other,state=cached[space])
                        row.update({space+'_'+k:v for k,v in metrics.items()})
                        for k,values in pt.items():
                            for j,v in enumerate(values):selected[j][space+'_'+k]=float(v)
                    profiles.append(row);points.extend(selected)
                audits.append(audit)
            print(f'Geometry verified seed {seed}, method {method}',flush=True)
    profiles=pd.DataFrame(profiles);points=pd.DataFrame(points)
    assert len(profiles)==180 and len(points)==4800 and len(audits)==120
    assert profiles.groupby(['seed','source']).size().eq(15).all()
    profiles.to_csv(out/'geometry_profiles.csv',index=False);points.to_csv(out/'geometry_points.csv',index=False)
    metrics=[c for c in profiles if c.startswith(('raw_','cw_')) and c not in ['raw_bank_available','cw_bank_available']]
    profiles.groupby(['seed','source','stage'])[metrics].mean().to_csv(out/'geometry_stage_by_seed.csv')
    profiles.groupby(['seed','source'])[metrics].mean().to_csv(out/'geometry_seed_summary.csv')
    audit=dict(analysis_type='read_only_frozen_method_local_geometry',seeds=list(SEEDS),rounds=list(range(15)),
               repeated_rounds_are_independent=False,expected_states=120,verified_states=len(audits),
               verified_raw_banks=sum(a['raw_available'] for a in audits),verified_cw_banks=sum(a['cw_available'] for a in audits),
               verified_checkpoint_hashes=sum(a['checkpoint_hash_verified'] for a in audits),
               coverage_definition='mean Euclidean nearest-center distance; all relative drops normalized by original L_t mean; fixed preselection U_t',
               common_remaining_U_definition='exclude current full32 for Hybrid CW16/LLM16/all32; exclude own32 for other methods',
               unavailable_spaces=[{'seed':a['seed'],'method':a['method'],'round':a['round'],'space':s,'reason':a.get(s+'_status','no frozen method-local bank')} for a in audits for s in ['raw','cw'] if not a[s+'_available']],
               no_gradient_recomputation=True,no_model_training=True,no_model_inference=True,no_selector_rerun=True,
               no_unselected_or_test_response_table_read=True,batches=audits,source_sha256=HASHES)
    (out/'geometry_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    findings(profiles,audits,out)
    if plots:plot(profiles,out)
    return profiles,points,audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();run(args.out,not args.no_plots)

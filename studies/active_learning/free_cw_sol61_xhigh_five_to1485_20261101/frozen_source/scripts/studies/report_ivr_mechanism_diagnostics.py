#!/usr/bin/env python3
"""Descriptive post-hoc controls; never modify the frozen mechanism gates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'studies/active_learning/qgeognn_v2_ivr_mechanism_audit'


def main():
    decision = json.loads((STUDY/'decision.json').read_text())
    if decision['snapshots'] != 25:
        raise RuntimeError('complete frozen audit required before descriptive reporting')
    rows=[]
    block_rows=[]
    duplicates=[]
    for p in sorted((STUDY/'runtime').glob('seed_*/round_*/complete.json')):
        selections=json.loads((p.parent/'selections.json').read_text())
        variants=selections['variants']
        with np.load(p.parent/'bank.npz') as bank:
            block=bank['block'][0].reshape(3330,2,512,4).sum(axis=-1).astype(float)
            scalar=bank['scalar'][0].reshape(3330,512,4).sum(axis=-1)
            reference=bank['canonical_indices']
        original=ROOT/'studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/runtime'/p.parent.parent.name/p.parent.name/'selection.json'
        membership=json.loads(original.read_text())['input']
        lookup={int(v):i for i,v in enumerate(reference)}
        u=np.array([lookup[v] for v in membership['ordered_unlabeled_indices']])
        for name,selected,features in (
            ('scalar',variants['scalar_d512_s0']['selected_candidate_positions'],scalar),
            ('block',variants['block_d512_s0']['selected_candidate_positions'],block.reshape(3330,-1)),
            ('forward_backward',selections['forward_backward']['selected_candidate_positions'],scalar),
            ('lcmd',variants['scalar_d512_s0']['lcmd_selected'],scalar)):
            duplicates.append(dict(snapshot=str(p.parent.relative_to(STUDY)),method=name,
                repeated_gradient_rows=32-len(np.unique(features[u[selected]],axis=0)),
                repeated_candidate_indices=32-len(set(selected))))
        gram=np.einsum('ned,nfd->nef',block,block)
        eigen=np.maximum(np.linalg.eigvalsh(gram),0)
        effective=(eigen.sum(axis=1)**2)/np.sum(eigen**2,axis=1)
        denom=np.sqrt(gram[:,0,0]*gram[:,1,1])
        corr=np.divide(gram[:,0,1],denom,out=np.full(len(denom),np.nan),where=denom>0)
        block_rows.append(dict(seed=int(p.parent.parent.name.removeprefix('seed_')),
            round=int(p.parent.name.removeprefix('round_')),median_endpoint_cosine=float(np.nanmedian(corr)),
            median_block_effective_rank=float(np.median(effective)),
            median_second_eigen_fraction=float(np.median(eigen[:,0]/eigen.sum(axis=1)))))
        comparisons=pd.read_csv(p.parent/'comparisons.csv')
        selected=comparisons[(comparisons.representation=='scalar') & (comparisons.group!='other')]
        for v in selected.to_dict('records'):
            a=set(variants[v['a']]['lcmd_selected'])
            b=set(variants[v['b']]['lcmd_selected'])
            rows.append(dict(seed=v['seed'],round=v['round'],group=v['group'],a=v['a'],b=v['b'],
                             lcmd_overlap=len(a&b)/32,ivr_overlap=v['batch_overlap']))
    frame=pd.DataFrame(rows)
    pd.DataFrame(duplicates).to_csv(STUDY/'results/descriptive_batch_duplicates.csv',index=False)
    pd.DataFrame(block_rows).to_csv(STUDY/'results/descriptive_endpoint_geometry.csv',index=False)
    frame.to_csv(STUDY/'results/descriptive_lcmd_stability.csv',index=False)
    summary=frame.groupby('group')[['lcmd_overlap','ivr_overlap']].agg(['mean','median','min'])
    summary.to_csv(STUDY/'results/descriptive_lcmd_stability_summary.csv')
    print(summary.to_string())
    spectral=pd.read_csv(STUDY/'results/spectral_and_selection.csv')
    primary=spectral[(spectral.dimension==512)&(spectral.map_index==0)]
    print(primary.groupby('representation')[['effective_rank','precision_condition','elapsed_seconds']].agg(['min','median','max']).to_string())
    print('maximum direct-factor error:',spectral.direct_factor_max_error.max())
    mechanisms=pd.read_csv(STUDY/'results/mechanisms.csv')
    print('maximum historical feature error:',mechanisms.historical_relative_feature_error.max())
    print('minimum historical score correlation:',mechanisms.historical_score_spearman.min())
    print('descriptive LCMD analysis added after first snapshot; does not change frozen decision')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparisons=pd.read_csv(STUDY/'results/comparisons.csv')
    groups=[('scalar','cross_seed_512','Scalar: map seeds'),
            ('scalar','512_vs_2048_same_seed','Scalar: 512 vs 2048'),
            ('block','cross_seed_512','Two outputs: map seeds')]
    colors=['#237894','#a35a28','#47815a']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for (rep,group,label),color in zip(groups,colors):
        part=comparisons[(comparisons.representation==rep)&(comparisons.group==group)]
        axes[0].scatter(part.score_spearman,part.batch_overlap,s=15,alpha=.5,color=color,label=label)
        axes[1].scatter(part.batch_overlap,100*part.transferred_batch_relative_regret,s=15,alpha=.5,color=color)
    axes[0].set(xlabel='Initial score Spearman correlation',ylabel='Selected batch overlap fraction')
    axes[0].axhline(.75,color='#777777',linestyle='--',linewidth=1,label='Mean-overlap gate: 0.75')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Selected batch overlap fraction',ylabel='Transferred batch regret (% of batch gain)')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.suptitle('IVR: score stability, batch membership, and surrogate equivalence')
    figures=STUDY/'figures'
    figures.mkdir(exist_ok=True)
    fig.savefig(figures/'representation_stability.png',dpi=170)
    plt.close(fig)

    gate_a=STUDY/'gate_a'
    gate_b=STUDY/'gate_b'
    gate_a.mkdir(exist_ok=True)
    gate_b.mkdir(exist_ok=True)
    a_lines=[
        '# Gate A: representation stability', '',
        'Completed: five frozen seeds x rounds 1, 5, 10, 15, 20.',
        '300 scalar configurations (4 dimensions x 3 mappings x 25 states),',
        'plus 75 shared-parameter two-output configurations at 512D.', '',
        'The scalar audit covers 256, 512, 1024 and 2048 dimensions.',
        'Multi-output cross-dimension stability has NOT been measured.',
        'No new fits, labels, validation outcomes or test outcomes were used.', '',
        '| Representation | Comparison | Median rho | Mean batch overlap | Median batch regret | Gate |',
        '|---|---|---:|---:|---:|---|',
    ]
    for g in decision['gates']:
        a_lines.append(f"| {g['representation']} | {g['group']} | {g['median_score_spearman']:.6f} | {g['mean_batch_overlap_fraction']:.4f} | {g['median_relative_regret']:.4%} | {g['passed']} |")
    a_lines += [
        '', '## Numerical checks', '',
        f"Largest regularized precision condition number: {spectral.precision_condition.max():.3f}.",
        f"Largest direct versus incremental covariance-factor error: {spectral.direct_factor_max_error.max():.3e}.",
        f"Largest historical-feature reconstruction relative error: {mechanisms.historical_relative_feature_error.max():.3e}.",
        f"Smallest historical-score Spearman: {mechanisms.historical_score_spearman.min():.10f}.", '',
        'With mean per-experiment squared feature norm fixed to one,',
        '`P = I + sum_L J_i.T J_i` has minimum eigenvalue at least one and',
        'maximum eigenvalue at most `1 + sum_R ||J_i||_F^2 = 3331`.',
        'Thus a large condition number of the UNREGULARIZED Gram matrix does',
        'not imply an unstable inverse in the implemented posterior.', '',
        '## Descriptive LCMD control (added after first snapshot)', '',
    ]
    for group,part in frame.groupby('group'):
        a_lines.append(f"- {group}: mean LCMD overlap {part.lcmd_overlap.mean():.4f}; IVR {part.ivr_overlap.mean():.4f}.")
    a_lines += [
        '', 'A failed 0.75 mean-overlap gate is an engineering-gate result,',
        'not proof that an acquisition method is scientifically invalid.',
        'LCMD and transferred-objective controls contextualize this threshold;',
        'they do not retroactively change the sealed decision.', '',
        'Full spectra/scores and all selected candidates are stored under runtime;',
        'aggregate tables are in ../results. See ../PROTOCOL.md and ../config.json.',
    ]
    (gate_a/'FINAL_REPORT.md').write_text('\n'.join(a_lines)+'\n')
    b_lines=[
        '# Gate B: batch optimizer and multi-output information', '',
        'Completed: selection-only audits on all 25 fixed states. No one-step',
        'retraining or predictive-error evaluation was performed.', '',
        '## Fixed mathematics', '',
        '`J_x = [S grad(V1/s1); S grad(V2/s2)] / a`, using the SAME S for both',
        'endpoints and common `a^2 = mean_R ||J_raw||_F^2`.',
        '`C = (I + sum_L J_i.T J_i)^(-1)` and `M = mean_R J_z.T J_z`.',
        '`risk = tr(M C)`; unit noise and endpoint weights remain frozen.',
        '`delta(x) = tr((I + J_x C J_x.T)^(-1) J_x C M C J_x.T)`.',
        '`C_new = C - C J_x.T (I + J_x C J_x.T)^(-1) J_x C`.', '',
        'The BAIT-style optimizer selects 64 then removes 32 pending candidates.',
        'A removal uses `C_new = C + C J_x.T (I - J_x C J_x.T)^(-1) J_x C`.',
        'At each step it minimizes the risk increase; labeled rows are never removed.',
        'There is no theorem here that forward/backward dominates forward greedy.', '',
        '## Results', '',
        f"Positive forward/backward gains: {decision['fb_positive_snapshots']}/25.",
        f"Median forward/backward relative batch gain: {decision['fb_median_relative_batch_gain']:.6%}.",
        f"Maximum relative batch gain: {mechanisms.fb_relative_batch_gain.max():.6%}.",
        f"Minimum relative batch gain: {mechanisms.fb_relative_batch_gain.min():.6%}.",
        f"Multi-output IVR beats LCMD on its OWN surrogate: {decision['multioutput_positive_snapshots']}/25.", '',
        'Winning the objective a selector explicitly optimizes is a mechanism',
        'sanity check, not evidence that it improves test NRMSE.',
        'Scalar and block risk magnitudes cannot be compared directly.', '',
        '## Training decision', '',
        f"Frozen decision: `{decision['status']}`; candidate: `{decision['candidate']}`.",
        'No complete sequential run or COMPOUND experiment was launched here.',
        'Do not interpret a stability-gate failure as a negative predictive trial.',
        'See ../decision.json for each fixed criterion and ../results for paired data.',
    ]
    (gate_b/'FINAL_REPORT.md').write_text('\n'.join(b_lines)+'\n')


if __name__=='__main__':
    main()

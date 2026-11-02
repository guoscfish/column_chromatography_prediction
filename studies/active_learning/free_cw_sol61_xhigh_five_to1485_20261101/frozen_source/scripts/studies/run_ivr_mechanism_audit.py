#!/usr/bin/env python3
"""Run a sealed, label-free IVR kernel and batch-optimizer audit."""

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_ivr_audit_matplotlib")

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.training.predictor import atomic_json
from src.qgeognn_al.active_learning_v2.benchmark_protocol import sketch_seed
from src.qgeognn_al.active_learning_v2.block_ivr import (
    block_batch_ivr, integrated_risk, normalize_blocks, posterior_covariance,
)
from src.qgeognn_al.active_learning_v2.cache import seal_cache, verify_cache
from src.qgeognn_al.active_learning_v2.gradient_bank import extract_gradient_bank, fold_sketch
from src.qgeognn_al.active_learning_v2.ivr import conditional_batch_ivr
from src.qgeognn_al.active_learning_v2.ivr_study import exclusive_lock
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.sequential_runner import _assert_protected, _write_json_once

STUDY = ROOT / 'studies/active_learning/qgeognn_v2_ivr_mechanism_audit'
OLD = ROOT / 'studies/active_learning/qgeognn_v2_row_kernel_ivr_b32'


def read(path):
    return json.loads(path.read_text())


def emit(**values):
    print(json.dumps(values), flush=True)


def snapshot_paths(seed, round_index):
    base = OLD / 'runtime' / f'seed_{seed}'
    directory = base / f'round_{round_index:02d}'
    return dict(checkpoint=directory/'model/best.pt', selection=directory/'selection.json',
                features=directory/'gradient_features.npz', freeze=directory/'freeze.json',
                context=base/'context/context.json', graphs=base/'context/scrubbed_graphs.pt')


def prepare(test_report):
    if (STUDY/'seal.json').exists():
        return validate()
    if (STUDY/'runtime').exists():
        raise RuntimeError('cannot freeze protocol after execution')
    suites = list(ET.parse(test_report).getroot().iter('testsuite'))
    if not suites or sum(int(s.get('tests', 0)) for s in suites) < 17 or any(
        int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped')
    ):
        raise RuntimeError('passing non-skipped mathematics and mapping tests required')
    config = read(STUDY/'config.json')
    paths = [STUDY/'config.json', STUDY/'PROTOCOL.md', Path(__file__).resolve()]
    paths += list((ROOT/'src/qgeognn_al').rglob('*.py'))
    paths += [ROOT/'application/QGeoGNN.py', ROOT/'application/utils.py']
    paths += [ROOT/'tests/active_learning_v2'/name for name in ('test_block_ivr.py', 'test_gradient_bank.py')]
    (STUDY/'preflight_tests.xml').write_bytes(test_report.read_bytes())
    paths.append(STUDY/'preflight_tests.xml')
    inputs = {}
    for seed, r in itertools.product(config['outer_seeds'], config['rounds']):
        sources = snapshot_paths(seed, r)
        freeze = read(sources['freeze'])
        if freeze['test_truth_access_count'] != 0:
            raise RuntimeError('historical snapshot was not frozen blind')
        _assert_protected(ROOT, freeze['files'])
        for key in ('graphs', 'features'):
            receipt = sources[key].with_suffix(sources[key].suffix+'.contract.json')
            verify_cache(sources[key], read(receipt)['contract'])
            inputs[str(receipt.relative_to(ROOT))] = sha256_file(receipt)
        for p in sources.values():
            inputs[str(p.relative_to(ROOT))] = sha256_file(p)
    seal = dict(status='SEALED_BEFORE_AUDIT', files={str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
                inputs=inputs, new_training_fits=0, new_label_access=0, created_unix=time.time())
    _write_json_once(STUDY/'seal.json', seal)
    return {'status': seal['status'], 'snapshots': 25}


def validate():
    seal = read(STUDY/'seal.json')
    _assert_protected(ROOT, seal['files'])
    _assert_protected(ROOT, seal['inputs'])
    return {'status': seal['status']}


def spectrum(features, labeled):
    x, _ = normalize_blocks(features)
    flat = x.reshape(-1, x.shape[-1])
    eigen = np.maximum(eigvalsh(flat.T@flat/len(x)), 0)
    positive = eigen[eigen > eigen[-1]*1e-12]
    p = positive/positive.sum()
    lf = x[labeled].reshape(-1, x.shape[-1])
    if len(lf) < x.shape[-1]:
        le = np.maximum(eigvalsh(lf@lf.T), 0)
        lower = 0.
    else:
        le = np.maximum(eigvalsh(lf.T@lf), 0)
        lower = le[0]
    return dict(effective_rank=float(np.exp(-np.sum(p*np.log(p)))),
                participation_rank=float(eigen.sum()**2/np.sum(eigen**2)),
                numerical_rank=len(positive), top_eigenvalue_fraction=float(eigen[-1]/eigen.sum()),
                positive_spectrum_condition=float(positive[-1]/positive[0]),
                raw_rank_deficient=len(positive)<len(eigen),
                precision_condition=float((1+le[-1])/(1+lower))), eigen


def overlap(a, b):
    return len(set(a)&set(b))/len(a)


def compare_scores(a, b):
    return float(spearmanr(a, b).statistic)


def evaluate_snapshot(seed, r, graphs, config):
    sources = snapshot_paths(seed, r)
    destination = STUDY/'runtime'/f'seed_{seed}'/f'round_{r:02d}'
    destination.mkdir(parents=True, exist_ok=True)
    seal_hash = sha256_file(STUDY/'seal.json')
    done = destination/'complete.json'
    if done.exists():
        receipt = read(done)
        if receipt['seal_sha256'] != seal_hash:
            raise RuntimeError('snapshot seal changed')
        _assert_protected(ROOT, receipt['files'])
        emit(seed=seed, round=r, stage='reused_complete')
        return
    selection = read(sources['selection'])
    with np.load(sources['features']) as old:
        reference = old['canonical_indices'].copy()
        historical = old['features'].copy()
    lookup = {int(v): i for i, v in enumerate(reference)}
    l = np.array([lookup[v] for v in selection['input']['ordered_labeled_indices']])
    u = np.array([lookup[v] for v in selection['input']['ordered_unlabeled_indices']])
    if len(reference) != 3330 or len(np.unique(np.r_[l,u])) != 3330:
        raise RuntimeError('reference/partition mismatch')
    scales = read(sources['context'])['contract']['preprocessing']['target_scales']
    seeds = [sketch_seed(seed)+offset for offset in config['sketch_seed_offsets']]
    contract = dict(seal_sha256=seal_hash, seed=seed, round=r, checkpoint_sha256=sha256_file(sources['checkpoint']),
                    sketch_seeds=seeds, reference_indices=reference.tolist())
    bank_path = destination/'bank.npz'
    if verify_cache(bank_path, contract):
        with np.load(bank_path) as values:
            bank = {k: values[k] for k in values.files}
    else:
        model = load_predictor_checkpoint(sources['checkpoint'])
        emit(seed=seed, round=r, stage='gradients')
        bank, audit = extract_gradient_bank(model, *graphs, reference, tuple(scales[k] for k in ('V1','V2')), seeds,
            progress=lambda progress: emit(seed=seed, round=r, stage='gradients', **progress))
        temporary = destination/'bank.partial.npz'
        np.savez_compressed(temporary, **bank)
        temporary.replace(bank_path)
        seal_cache(bank_path, contract)
        atomic_json(destination/'gradient_audit.json', audit)
    legacy = fold_sketch(bank['scalar'][0], 512)
    relative = float(np.linalg.norm(legacy.astype(float)-historical)/np.linalg.norm(historical.astype(float)))
    if relative > config['numerical_checks']['historical_512_relative_frobenius_error_max']:
        raise RuntimeError(f'historical feature reconstruction failed: {relative}')
    old_result = block_batch_ivr(historical, l, u, 32)
    old_selector = conditional_batch_ivr(historical, l, u, 32)
    if old_result['selected_candidate_positions'] != old_selector['selected_candidate_positions']:
        raise RuntimeError('new scalar selector differs from historical implementation')
    variants, stats, spectra, score_arrays = {}, [], {}, {}
    for representation in ('scalar', 'block'):
        dimensions = config[f'{representation}_dimensions']
        for dimension, map_index in itertools.product(dimensions, range(3)):
            key = f'{representation}_d{dimension}_s{map_index}'
            emit(seed=seed, round=r, stage='selection', variant=key)
            features = fold_sketch(bank[representation][map_index], dimension)
            result = block_batch_ivr(features, l, u, 32)
            spectral, eigen = spectrum(features, l)
            scores = result.pop('initial_scores')
            spectra[key] = eigen
            score_arrays[key] = scores
            if representation == 'scalar':
                lc = lcmd_tp_select(features[u], features[l], 32).selected_pool_positions.tolist()
            else:
                # Common legacy LCMD on this exact state, not a different trajectory.
                lc = variants[f'scalar_d512_s{map_index}']['lcmd_selected']
            x, _ = normalize_blocks(features)
            lc_risk = integrated_risk(x, posterior_covariance(x, np.r_[l,u[lc]]))
            result['lcmd_selected'] = lc
            result['lcmd_risk_same_surrogate'] = lc_risk
            variants[key] = result
            stats.append(dict(seed=seed, round=r, representation=representation, dimension=dimension,
                              map_index=map_index, **spectral, **result['audit'],
                              lcmd_overlap=overlap(result['selected_candidate_positions'],lc),
                              lcmd_risk_same_surrogate=lc_risk))
    rho = compare_scores(old_result['initial_scores'],score_arrays['scalar_d512_s0'])
    if rho < config['numerical_checks']['historical_initial_score_spearman_min']:
        raise RuntimeError('historical score reconstruction failed')
    comparisons=[]
    for representation in ('scalar','block'):
        keys = [key for key in variants if key.startswith(representation)]
        for a,b in itertools.combinations(keys,2):
            _, da, sa = a.split('_')
            _, db, sb = b.split('_')
            if da != db and sa != sb:
                continue
            group = 'other'
            if da == db == 'd512':
                group = 'cross_seed_512'
            if {da,db} == {'d512','d2048'} and sa == sb:
                group = '512_vs_2048_same_seed'
            xa, _ = normalize_blocks(fold_sketch(bank[representation][int(sa[1:])],int(da[1:])))
            sel_a = variants[a]['selected_candidate_positions']
            sel_b = variants[b]['selected_candidate_positions']
            risk_b_in_a = integrated_risk(xa,posterior_covariance(xa,np.r_[l,u[sel_b]]))
            aa = variants[a]['audit']
            comparisons.append(dict(seed=seed,round=r,representation=representation,group=group,a=a,b=b,
                score_spearman=compare_scores(score_arrays[a],score_arrays[b]),
                top32_overlap=overlap(np.argsort(-score_arrays[a],kind='stable')[:32],np.argsort(-score_arrays[b],kind='stable')[:32]),
                batch_overlap=overlap(sel_a,sel_b),
                transferred_batch_relative_regret=(risk_b_in_a-aa['final_risk'])/(aa['initial_risk']-aa['final_risk'])))
    fb = block_batch_ivr(legacy,l,u,32,backward=True)
    fb.pop('initial_scores')
    scalar = variants['scalar_d512_s0']
    multi = variants['block_d512_s0']
    gain = (scalar['audit']['final_risk']-fb['audit']['final_risk'])/(scalar['audit']['initial_risk']-scalar['audit']['final_risk'])
    mechanism = dict(seed=seed,round=r,fb_relative_batch_gain=gain,
        multioutput_risk_gain_over_lcmd=multi['lcmd_risk_same_surrogate']-multi['audit']['final_risk'],
        multioutput_scalar_overlap=overlap(multi['selected_candidate_positions'],scalar['selected_candidate_positions']),
        historical_relative_feature_error=relative,historical_score_spearman=rho,
        scalar_block_kernel_cosine=kernel_cosine(legacy,fold_sketch(bank['block'][0],512).reshape(3330,-1)))
    pd.DataFrame(stats).to_csv(destination/'spectral_and_selection.csv',index=False)
    pd.DataFrame(comparisons).to_csv(destination/'comparisons.csv',index=False)
    atomic_json(destination/'mechanism.json',mechanism)
    atomic_json(destination/'selections.json',dict(variants=variants,forward_backward=fb))
    np.savez_compressed(destination/'spectra_and_scores.npz',**{f'eigen_{k}':v for k,v in spectra.items()},
                        **{f'score_{k}':v for k,v in score_arrays.items()})
    files = [p for p in destination.iterdir() if p.is_file() and p.name != 'complete.json']
    atomic_json(done,dict(seal_sha256=seal_hash,new_label_access=0,new_training_fits=0,
                         files={str(p.relative_to(ROOT)):sha256_file(p) for p in files}))
    emit(seed=seed,round=r,stage='complete',**{k:v for k,v in mechanism.items() if k not in ('seed','round')})


def kernel_cosine(a,b):
    cross = np.sum((a.astype(float).T@b.astype(float))**2)
    norm_a = np.sum((a.astype(float).T@a.astype(float))**2)
    norm_b = np.sum((b.astype(float).T@b.astype(float))**2)
    return float(cross/np.sqrt(norm_a*norm_b))


def run():
    validate()
    config = read(STUDY/'config.json')
    torch.set_num_threads(config['threads'])
    with exclusive_lock(STUDY/'runtime/worker.lock'):
        for seed in config['outer_seeds']:
            path = snapshot_paths(seed,config['rounds'][0])['graphs']
            graphs = torch.load(path,weights_only=False)
            if any(torch.count_nonzero(item.y) for item in graphs[0]):
                raise RuntimeError('nonzero graph labels')
            for r in config['rounds']:
                evaluate_snapshot(seed,r,graphs,config)
    return report()


def report():
    validate()
    config = read(STUDY/'config.json')
    snapshots=[]
    for seed,r in itertools.product(config['outer_seeds'],config['rounds']):
        p=STUDY/'runtime'/f'seed_{seed}'/f'round_{r:02d}'
        receipt=read(p/'complete.json')
        if receipt['seal_sha256'] != sha256_file(STUDY/'seal.json'):
            raise RuntimeError('snapshot seal mismatch')
        _assert_protected(ROOT,receipt['files'])
        snapshots.append(p)
    spectral=pd.concat([pd.read_csv(p/'spectral_and_selection.csv') for p in snapshots],ignore_index=True)
    comparisons=pd.concat([pd.read_csv(p/'comparisons.csv') for p in snapshots],ignore_index=True)
    mechanisms=pd.DataFrame([read(p/'mechanism.json') for p in snapshots])
    outputs=STUDY/'results'
    outputs.mkdir(exist_ok=True)
    spectral.to_csv(outputs/'spectral_and_selection.csv',index=False)
    comparisons.to_csv(outputs/'comparisons.csv',index=False)
    mechanisms.to_csv(outputs/'mechanisms.csv',index=False)
    rules=config['stability_gate']
    summary=[]
    for (representation,group),rows in comparisons[comparisons.group.ne('other')].groupby(['representation','group']):
        values=dict(representation=representation,group=group,comparisons=len(rows),
            median_score_spearman=float(rows.score_spearman.median()),p10_score_spearman=float(rows.score_spearman.quantile(.1)),
            mean_batch_overlap_fraction=float(rows.batch_overlap.mean()),p10_batch_overlap_fraction=float(rows.batch_overlap.quantile(.1)),
            median_relative_regret=float(rows.transferred_batch_relative_regret.median()))
        values['passed']=all(values[k.removesuffix('_min')] >= v for k,v in rules.items() if k.endswith('_min'))
        summary.append(values)
    s=pd.DataFrame(summary)
    s.to_csv(outputs/'stability_gate.csv',index=False)
    scalar_pass=bool(s[s.representation.eq('scalar')].passed.all())
    block_pass=bool(s[s.representation.eq('block')].passed.all())
    gate=config['mechanism_gate']
    multi_positive=int((mechanisms.multioutput_risk_gain_over_lcmd>1e-12).sum())
    fb_positive=int((mechanisms.fb_relative_batch_gain>1e-12).sum())
    fb_median=float(mechanisms.fb_relative_batch_gain.median())
    multi_pass=scalar_pass and block_pass and multi_positive>=gate['multioutput_positive_risk_gain_over_same_state_lcmd_snapshots_min']
    fb_pass=scalar_pass and fb_positive>=gate['fb_positive_gain_snapshots_min'] and fb_median>=gate['fb_median_relative_batch_gain_over_forward_min']
    candidate='multioutput_ivr' if multi_pass else ('scalar_forward_backward' if fb_pass else None)
    decision=dict(status='CANDIDATE_READY_FOR_SEPARATE_PREREGISTRATION' if candidate else 'STOP_BEFORE_GATE_C',
        candidate=candidate,snapshots=len(snapshots),scalar_stability_passed=scalar_pass,block_stability_passed=block_pass,
        multioutput_positive_snapshots=multi_positive,fb_positive_snapshots=fb_positive,fb_median_relative_batch_gain=fb_median,
        new_training_fits=0,new_label_access=0,predictive_effectiveness='NOT_EVALUATED',gates=summary)
    atomic_json(STUDY/'decision.json',decision)
    lines=['# IVR mechanism audit: final report','',f"Decision: `{decision['status']}`.",'',
        'All 25 frozen states were audited. No training or label reveal was performed.',
        'The gates measure reproducibility and surrogate behavior, not predictive improvement.','',
        '| Representation | Comparison | Median rho | P10 rho | Mean overlap | P10 overlap | Pass |',
        '|---|---|---:|---:|---:|---:|---|']
    for v in summary:
        lines.append(f"| {v['representation']} | {v['group']} | {v['median_score_spearman']:.4f} | {v['p10_score_spearman']:.4f} | {v['mean_batch_overlap_fraction']:.4f} | {v['p10_batch_overlap_fraction']:.4f} | {v['passed']} |")
    lines += ['',f'Forward/backward positive gain: {fb_positive}/25 snapshots; median relative batch gain {fb_median:.6f}.',
        f'Multi-output IVR beats same-state LCMD on its own surrogate: {multi_positive}/25 snapshots.',
        f'Candidate eligible for a separately sealed training experiment: {candidate}.','',
        'See PROTOCOL.md and config.json for formulas, fixed gates, limitations and representation scaling.',
        'Raw low-rank kernels do not establish numerical instability. Scalar/block objectives are not directly comparable.',
        'Marginal variance-reduction scores need not be monotone. No test-error gate was applied.',
        'No automatic dimension/noise/regularization search or COMPOUND expansion is allowed.']
    (STUDY/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--run',action='store_true')
    action.add_argument('--report',action='store_true')
    parser.add_argument('--test-report',type=Path)
    args=parser.parse_args()
    if args.prepare and args.test_report is None:
        parser.error('--prepare requires --test-report')
    emit(result=prepare(args.test_report) if args.prepare else run() if args.run else report())

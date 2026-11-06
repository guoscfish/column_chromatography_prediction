"""Reproduce frozen historical batch geometry; CPU only, no model/target reads.

First run: python run_geometry.py --capture
Later runs: python run_geometry.py   (reuses locked source choices and hashes)
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/nocw_geometry_mpl')

import argparse
from collections import Counter
from datetime import datetime, timezone
import lzma
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry_metrics import evaluate_geometry, fixed_reference_ids, self_check
import analyze

HERE = Path(__file__).resolve().parent
ROOT = analyze.ROOT
MANIFEST = HERE / 'geometry_input_manifest.json.xz'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(lzma.compress(payload, preset=6) if path.suffix == '.xz' else payload)


class Reader:
    def __init__(self, cutoff_ns, locked=None):
        self.cutoff_ns = cutoff_ns
        self.locked = locked
        self.files = {}

    def read(self, path, expected=None):
        path = Path(path)
        key = relative(path)
        before = path.stat()
        if before.st_mtime_ns > self.cutoff_ns:
            raise ValueError('file modified after the main snapshot cutoff: ' + key)
        a = path.read_bytes()
        middle = path.stat()
        b = path.read_bytes()
        after = path.stat()
        signature = lambda st: (st.st_ino, st.st_size, st.st_mtime_ns)
        if signature(before) != signature(middle) or signature(middle) != signature(after) or a != b:
            raise ValueError('file changed during double read: ' + key)
        digest = sha(a)
        if expected is not None and digest != expected:
            raise ValueError('source differs from frozen metadata: ' + key)
        if self.locked is not None:
            if key not in self.locked or self.locked[key]['sha256'] != digest:
                raise ValueError('source differs from geometry manifest: ' + key)
        if key in self.files and self.files[key]['sha256'] != digest:
            raise ValueError('source changed between dependent reads: ' + key)
        self.files[key] = {'path': key, 'sha256': digest, 'bytes': len(a),
            'mtime_ns': before.st_mtime_ns,
            'read_at_utc': datetime.now(timezone.utc).isoformat(), 'stable_double_read': True}
        return a

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))


def llm_candidates(batch):
    method, seed, r = batch['method'], batch['seed'], batch['round']
    name = analyze.FREE if method == 'free' else 'llm32_nocw'
    own = ROOT / batch['contract_source']
    runtime = Path(str(own.parent).replace('/selections/', '/runtime/'))
    candidates = [runtime / 'banks.json', analyze.llm_dirs(method, seed, r, True) / 'banks.json']
    revision = analyze.FREV if method == 'free' else analyze.NREV
    candidates += sorted(revision.glob(f'*/runtime/seed_{seed}/{name}/round_{r:02d}/banks.json'))
    if method == 'nocw' and r == 0:
        # The initial protocol explicitly reused this seed's Free L333 catalog
        # and predictor. Permit only an exact checkpoint + full ordered-ID match.
        owner = ('astra_high_full_pool_20261001' if seed in (157,6101)
                 else 'sol61_xhigh_free_five_seeds_to1005_20261023')
        candidates.append(analyze.FREV / owner /
            f'runtime/seed_{seed}/{analyze.FREE}/round_00/banks.json')
    return list(dict.fromkeys(candidates))


def locate(batch, reader, partition):
    method, seed, r = batch['method'], batch['seed'], batch['round']
    reverse = {int(row['canonical_index']): sid for sid, row in partition.items()
               if row['role'] in ('l0', 'u0')}
    lset, uset = set(batch['labeled_ids']), set(batch['unlabeled_ids'])
    if method in ('nocw','free'):
        errors = []
        for metadata_path in llm_candidates(batch):
            if not metadata_path.is_file():
                continue
            try:
                metadata = reader.json(metadata_path)
                contract = metadata['contract']
                if contract['checkpoint'] != batch['checkpoint_sha256']:
                    raise ValueError('checkpoint mismatch')
                indices = contract['indices']
                ids = [reverse[int(i)] for i in indices]
                n = len(lset)
                if len(ids) != len(set(ids)) or set(ids[:n]) != lset or set(ids[n:]) != uset:
                    raise ValueError('ordered L/U bank IDs mismatch')
                bank_path = metadata_path.with_suffix('.npz')
                return {'method': method, 'seed': seed, 'round': r,
                    'bank_path': relative(bank_path), 'metadata_path': relative(metadata_path),
                    'bank_sha256': metadata['sha256'], 'metadata_sha256': reader.files[relative(metadata_path)]['sha256'],
                    'shared_initial_bank': method == 'nocw' and '/free_llm32_scientist_v2/' in str(bank_path),
                    'spaces': ['raw_gradient', 'cw_gradient'], 'bank_kind': 'llm_raw_and_cw_frozen_gradient',
                    'row_ids': ids, 'canonical_indices': indices,
                    'state_checkpoint_sha256': contract['checkpoint'],
                    'bank_protocol': contract['protocol'], 'L_U_order_verified': True}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'path': relative(metadata_path), 'reason': str(exc)})
        raise ValueError('No complete checkpoint/ordered-LU-matched historical LLM bank; ' + json.dumps(errors))
    directory = analyze.num_dir(method, seed, r)
    acq = directory / 'acquisition_artifacts'
    name = 'current_representations.npz' if method == 'hybrid' else 'current_gradient_features.npz'
    metadata_path = acq / (name + '.contract.json')
    metadata = reader.json(metadata_path)
    acquisition = reader.json(ROOT / batch['contract_source'])
    c = metadata['contract']
    if metadata['contract_hash'] != stable(c):
        raise ValueError('numeric metadata contract hash mismatch')
    for key, value in acquisition['input'].items():
        if c.get(key) != value:
            raise ValueError('numeric bank/acquisition contract mismatch: ' + key)
    expected_kind = ('current_QGeoGNN_V2_128D_representation' if method == 'hybrid'
                     else 'current_full_network_q50_gradient_CountSketch')
    expected = {'outer_seed': seed, 'method': method, 'source_round': r,
        'acquisition_round': r+1, 'batch_size': 32,
        'evaluation_checkpoint_sha256': batch['checkpoint_sha256'],
        'L_t_ids_hash': analyze.ids_hash(batch['labeled_ids']),
        'U_t_ids_hash': analyze.ids_hash(batch['unlabeled_ids']),
        'centers_are_current_L_t': True, 'feature_recomputed_from_current_model': True,
        'test_truth_access_count': 0, 'kind': expected_kind}
    if method == 'lcmd':
        expected.update(dimension=512, sketch_seed=seed+4_000_037)
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError('numeric state contract mismatch: ' + key)
    indices = c['ordered_indices']
    ids = [reverse[int(i)] for i in indices]
    n = len(lset)
    if len(ids) != len(set(ids)) or set(ids[:n]) != lset or set(ids[n:]) != uset:
        raise ValueError('numeric ordered L/U bank IDs mismatch')
    return {'method': method, 'seed': seed, 'round': r, 'bank_path': relative(acq/name),
        'metadata_path': relative(metadata_path), 'bank_sha256': metadata['sha256'],
        'metadata_sha256': reader.files[relative(metadata_path)]['sha256'],
        'acquisition_contract_path': batch['contract_source'],
        'acquisition_contract_sha256': reader.files[batch['contract_source']]['sha256'],
        'shared_initial_bank': False, 'spaces': ['latent' if method == 'hybrid' else 'raw_gradient'],
        'bank_kind': expected_kind, 'row_ids': ids, 'canonical_indices': indices,
        'state_checkpoint_sha256': c['evaluation_checkpoint_sha256'], 'L_U_order_verified': True}


def read_bank(entry, batch, reader):
    metadata = reader.json(ROOT / entry['metadata_path'], entry['metadata_sha256'])
    if metadata['sha256'] != entry['bank_sha256']:
        raise ValueError('Bank hash differs from locked metadata')
    data = reader.read(ROOT / entry['bank_path'], entry['bank_sha256'])
    with np.load(io.BytesIO(data), allow_pickle=False) as bank:
        if batch['method'] in ('nocw','free'):
            arrays = {'raw_gradient': bank['raw'], 'cw_gradient': bank['cw']}
        else:
            if bank['canonical_indices'].tolist() != entry['canonical_indices']:
                raise ValueError('NPZ canonical row order differs from metadata')
            arrays = {entry['spaces'][0]: bank['features']}
    for space, features in arrays.items():
        dimension = 128 if space == 'latent' else 512
        if features.shape != (3330, dimension) or not np.isfinite(features).all():
            raise ValueError('Invalid complete finite feature matrix: ' + space)
    return arrays


def phase(L):
    return 'early_333_525' if L < 525 else 'middle_525_813' if L < 813 else 'late_813_common'


def summarize(rows, snapshot):
    frame = pd.DataFrame(rows)
    common = min(s['completed_L'] for s in snapshot['status'])
    frame['in_common_range'] = frame.L_after <= common
    frame.to_csv(HERE / 'geometry_metrics.csv', index=False)
    valid = frame[frame.available & frame.in_common_range].copy()
    metrics = ['batch_nn_distance_mean_random_percentile',
               'selected_to_l_distance_mean_random_percentile',
               'reference_coverage_distance_drop_random_percentile',
               'reference_coverage_relative_drop_random_percentile',
               'reference_coverage_relative_drop', 'reference_positive_before_n']
    group_keys = ['method','seed','space','phase']
    per_seed = valid.groupby(group_keys)[metrics].mean().reset_index()
    per_seed['n_batches'] = valid.groupby(group_keys).size().values
    per_seed['aggregation'] = 'seed'
    aggregate = []
    for keys, group in per_seed.groupby(['method','space','phase']):
        for stat in ('mean','std'):
            vals = getattr(group[metrics], stat)(**({'ddof':1} if stat=='std' else {})).to_dict()
            aggregate.append(dict(zip(['method','space','phase'], keys)) |
                {'seed':'ALL','aggregation':'group_'+stat,'n_seeds':len(group),**vals})
    pd.concat([per_seed,pd.DataFrame(aggregate)],ignore_index=True).to_csv(
        HERE / 'geometry_seed_phase_summary.csv', index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(13.8,4.8),sharey=True)
    displays = [('nocw','raw_gradient'),('free','raw_gradient'),('lcmd','raw_gradient'),('hybrid','latent')]
    names = {'nocw':'NoCW (raw gradient)','free':'Free (raw gradient)',
             'lcmd':'Gradient-LCMD (raw gradient)','hybrid':'Hybrid (latent)'}
    colors = {'nocw':'#E34A33','free':'#8055B8','hybrid':'#D89025','lcmd':'#148B88'}
    stages = ['early_333_525','middle_525_813','late_813_common']
    for ax, metric, title in zip(axes,metrics[:3],['Within-batch nearest distance',
             'Selected to labelled nearest distance','Fixed-reference coverage gain']):
        for j,(method,space) in enumerate(displays):
            means, errors = [], []
            for stage in stages:
                values = per_seed[(per_seed.method==method)&(per_seed.space==space)&(per_seed.phase==stage)][metric]
                means.append(values.mean());errors.append(values.std(ddof=1))
            ax.errorbar(np.arange(3)+(j-1.5)*.055, means, yerr=errors, label=names[method],
                color=colors[method], marker='o',linewidth=1.5,capsize=2,markersize=4)
        ax.axhline(.5,color='#888',ls='--',lw=.8)
        ax.set_xticks(range(3),['333-525','525-813',f'813-{common}'])
        ax.set_xlabel('L_before phase (left inclusive)');ax.set_title(title)
        ax.set_ylim(-.03,1.03);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Percentile vs 16 random batches\n(seed mean +/- seed SD; higher = larger metric)')
    axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Historical geometry, normalized within each method/checkpoint state',fontsize=12)
    fig.text(.5,.025,'Fixed 128 initial-pool reference IDs per seed; same IDs across methods and rounds.\n'
        'Gradient and latent spaces remain separate proxies; percentiles are not information gain.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(HERE / 'geometry_summary.png',dpi=180)
    plt.close(fig)
    return frame, per_seed, common


def main():
    global HERE, MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE, help='Directory with the fixed snapshot and derived outputs')
    parser.add_argument('--capture',action='store_true')
    args = parser.parse_args()
    HERE=args.out.resolve()
    MANIFEST=HERE/'geometry_input_manifest.json.xz'
    started = time.monotonic()
    snapshot_path = HERE / 'snapshot_inputs.json'
    data = analyze.load_snapshot_bytes(HERE)
    snapshot = json.loads(data)
    if args.capture:
        if MANIFEST.exists():
            raise FileExistsError('Geometry manifest already exists; rerun without --capture')
        manifest = {'version':1,'snapshot_path':relative(snapshot_path),'snapshot_sha256':sha(data),
            'snapshot_time':snapshot['snapshot_time'],'cutoff_ns':snapshot['cutoff_ns'],
            'reference_rule':'sha256(salt + NUL + stable_row_id); first 128 initial U0 IDs, fixed across all methods/rounds',
            'reference_by_seed':{s:fixed_reference_ids(ids,128,'nocw_free_selection_diagnostics_v1_seed_'+s)
                for s,ids in snapshot['pool'].items()},
            'random_seeds':list(range(90210,90226)),
            'reference_policy':'No removal of labelled or selected references; exact zeros by stable IDs; identical denominator for actual/random batches',
            'sources':{},'batches':[],'missing':[]}
        reader = Reader(snapshot['cutoff_ns'])
    else:
        manifest = json.loads(lzma.decompress(MANIFEST.read_bytes()))
        if sha(data) != manifest['snapshot_sha256']:
            raise ValueError('Compact snapshot differs from locked geometry manifest')
        reader = Reader(manifest['cutoff_ns'],manifest['sources'])
    entries = {(x['method'],x['seed'],x['round']):x for x in manifest['batches']}
    rows = []
    for b in snapshot['batches']:
        if not b['accepted'] or not b['completed_fit_validation']:
            continue
        key = (b['method'],b['seed'],b['round'])
        base = {k:b[k] for k in ('method','seed','round','L_before','L_after','source','contract_source')}
        base.update(phase=phase(b['L_before']),accepted=True,completed_fit_validation=True)
        try:
            if args.capture:
                entry = locate(b,reader,snapshot['partitions'][str(b['seed'])])
            else:
                if key not in entries:
                    prior = next((x for x in manifest['missing'] if tuple(x[k] for k in ('method','seed','round'))==key), None)
                    raise ValueError('Locked missing input: ' + str(prior))
                entry = entries[key]
            arrays = read_bank(entry,b,reader)
            if args.capture:
                manifest['batches'].append(entry)
            for space, features in arrays.items():
                metrics = evaluate_geometry(features,entry['row_ids'],b['labeled_ids'],b['unlabeled_ids'],
                    b['selected_ids'],manifest['reference_by_seed'][str(b['seed'])],manifest['random_seeds'])
                rows.append({**base,'space':space,'available':True,'missing_reason':'',
                    'bank_path':entry['bank_path'],'bank_sha256':entry['bank_sha256'],
                    'shared_initial_bank':entry['shared_initial_bank'],
                    'checkpoint_ids_hash_order_verified':True,**metrics})
        except (OSError,ValueError,KeyError,TypeError,AssertionError) as exc:
            reason = str(exc)
            if args.capture:
                manifest['missing'].append({**base,'reason':reason})
            for space in (['raw_gradient','cw_gradient'] if b['method'] in ('nocw','free')
                          else ['latent'] if b['method']=='hybrid' else ['raw_gradient']):
                rows.append({**base,'space':space,'available':False,'missing_reason':reason})
        if (len(rows)//25) != ((len(rows)-(2 if b['method'] in ('nocw','free') else 1))//25):
            print(f'Processed {len(rows)} geometry-space rows',flush=True)
    if args.capture:
        manifest['sources'] = reader.files
        manifest['capture_finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(MANIFEST,manifest)
    write_json(HERE/'geometry_missing_inputs.json',manifest['missing'])
    frame, per_seed, common = summarize(rows,snapshot)
    summary = {'common_completed_budget':common,'accepted_completed_batches':sum(
        b['accepted'] and b['completed_fit_validation'] for b in snapshot['batches']),
        'geometry_rows':len(frame),'available_rows':int(frame.available.sum()),
        'missing_rows':int((~frame.available).sum()),
        'available_by_method_space':frame[frame.available].groupby(['method','space']).size().to_dict(),
        'shared_initial_rows':int(frame.get('shared_initial_bank',pd.Series(dtype=bool)).fillna(False).sum()),
        'sources':len(manifest['sources']),'seconds':round(time.monotonic()-started,3),
        'self_check':self_check(),
        'safety':{'model_loads':0,'training_runs':0,'gpu_use':False,'llm_calls':0,'target_or_prediction_table_reads':0},
        'missing':manifest['missing']}
    summary['available_by_method_space'] = {'/'.join(k):int(v) for k,v in summary['available_by_method_space'].items()}
    write_json(HERE/'geometry_verification.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='missing'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

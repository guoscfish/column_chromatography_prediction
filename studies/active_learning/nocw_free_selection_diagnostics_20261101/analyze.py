"""Read-only historical acquisition diagnostics; no runner/model/label-store imports.

Capture complete, stable metadata once, then reproduce all tables from the compact
label-free snapshot. Geometry reads only hash-locked historical feature caches.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import lzma
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/nocw_free_diagnostics_mpl')
import numpy as np
import pandas as pd
from condition_metrics import compute_condition_metrics

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'src/qgeognn_al').is_dir())
SEEDS = (157, 887, 2357, 6101, 12203)
METHODS = ('nocw', 'free', 'hybrid', 'lcmd')
FREE = 'free_llm32_scientist_v2'
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
EXT = BASE.parent / 'qgeognn_v2_row_lcmd_to1485_20261026'
NREV = BASE.parent / 'qgeognn_v2_row_llm_nocw_ablation/revisions'
FREV = BASE.parent / 'qgeognn_v2_row_llm_scientist_v2/revisions'
OLD = ((21,'sol_xhigh_hybrid_free_to1005_20261013'),(26,'sol_xhigh_all_strategies_to1165_20261014'),
       (30,'sol_xhigh_all_strategies_to1293_20261015'),(34,'sol_xhigh_all_strategies_to1421_20261017'),
       (36,'sol_xhigh_all_strategies_to1549_20261018'))
NEW = ((21,'sol61_xhigh_free_five_seeds_to1005_20261023'),(36,'sol61_xhigh_free_five_seeds_to1485_20261024'))
NOLD = NREV / 'five_seed_nocw_to1005_20261029'
NEXT = NREV / 'five_seed_nocw_to1389_20261101'


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def digest(b):
    return hashlib.sha256(b).hexdigest()


def stable(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def ids_hash(ids):
    return stable(sorted(ids))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_snapshot_bytes(out):
    archive=out/'snapshot_inputs.json.xz'
    if archive.exists():
        return lzma.decompress(archive.read_bytes())
    packed = out/'snapshot_inputs.json.gz'
    return gzip.decompress(packed.read_bytes()) if packed.exists() else (out/'snapshot_inputs.json').read_bytes()


def load_snapshot(out):
    return json.loads(load_snapshot_bytes(out))


class SnapshotReader:
    def __init__(self):
        self.cutoff_ns = time.time_ns()
        self.cutoff = datetime.fromtimestamp(self.cutoff_ns / 1e9, ZoneInfo('Asia/Shanghai')).isoformat()
        self.manifest = {}
        self.missing = []

    def read(self, p, kind='json', expected=None, optional=False):
        p = Path(p)
        key = rel(p)
        try:
            before = p.stat()
            if before.st_mtime_ns > self.cutoff_ns:
                raise ValueError('file modified after snapshot cutoff')
            b = p.read_bytes()
            after = p.stat()
            # A second content read detects in-place writes even with unchanged size.
            b2 = p.read_bytes()
            final = p.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or (after.st_size,after.st_mtime_ns)!=(final.st_size,final.st_mtime_ns) or b != b2:
                raise ValueError('file changed while reading')
            sha = digest(b)
            if expected is not None and sha != expected:
                raise ValueError('content hash disagrees with frozen contract')
            if kind == 'json':
                value = json.loads(b)
            elif kind == 'csv':
                value = list(csv.DictReader(io.StringIO(b.decode())))
            elif kind == 'npy':
                value = np.load(io.BytesIO(b), allow_pickle=False).astype(str).tolist()
            else:
                value = b
            prior = self.manifest.get(key)
            if prior and prior['sha256'] != sha:
                raise ValueError('source changed between dependent reads')
            self.manifest[key] = {'path':key, 'sha256':sha, 'bytes':len(b),
                'mtime_utc':datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                'read_at_utc':datetime.now(timezone.utc).isoformat(), 'stable_double_read':True}
            return value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.missing.append({'path':key, 'reason':str(exc), 'optional':optional})
            return None


def free_owner(seed, fit_round):
    if seed in {887:33,12203:24} and fit_round > {887:33,12203:24}[seed]:
        return FREV / 'sol61_xhigh_free_to1485_extra3_20261026'
    return FREV / next(n for end,n in (OLD if seed in (157,6101) else NEW) if fit_round <= end)


def llm_dirs(method, seed, r, fit=False):
    name = FREE if method == 'free' else 'llm32_nocw'
    if method == 'free':
        owner = free_owner(seed, r if fit else r+1)
    else:
        owner = NOLD if r < (22 if fit else 21) else NEXT
    return owner / f'{"runtime" if fit else "selections"}/seed_{seed}/{name}/round_{r:02d}'


def num_dir(method, seed, r, fit=False):
    owner = BASE if method == 'hybrid' or r < (22 if fit else 21) else EXT
    return owner / f'runtime/seed_{seed}/{method}/round_{r:02d}'


def fit_read(reader, method, seed, r, required=True):
    directory = llm_dirs(method,seed,r,True) if method in ('nocw','free') else num_dir(method,seed,r,True)
    if method in ('nocw','free'):
        frozen = reader.read(directory/'prediction_freeze.json', optional=not required)
        if frozen is None:
            return None
        assert frozen['test_truth_access_count'] == 0
        path = Path(frozen['checkpoint_path']).parent/'fit_audit.json'
        audit = reader.read(path, optional=not required)
        if audit is None:
            return None
        assert audit['train_ids_hash'] == frozen['L_t_ids_hash']
        assert audit['checkpoint_sha256'] == frozen['checkpoint_sha256']
        assert audit['prediction_sha256'] == frozen['prediction_sha256']
    else:
        path = directory/'model/fit_audit.json'
        audit = reader.read(path, optional=not required)
        if audit is None:
            return None
    assert audit['train_rows'] == 333 + 32*r
    assert audit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
    return {'method':method,'seed':seed,'round':r,'L':333+32*r,'source':rel(path),**audit}


def capture(out):
    rd = SnapshotReader()
    snapshot = {'snapshot_time':rd.cutoff,'cutoff_ns':rd.cutoff_ns,'seeds':list(SEEDS),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'metadata':{},'initial':{},'pool':{},'partitions':{},'batches':[],'fits':[], 'status':[],
        'safety':{'new_llm_calls':0,'training_runs':0,'uncollected_response_reads':0,'gpu_use':False}}
    for seed in SEEDS:
        part = rd.read(BASE/f'splits/row_seed_{seed}.csv','csv')
        assert part is not None
        # Identity/split table has no response columns.
        p = {x['sample_id']:x for x in part}
        snapshot['partitions'][str(seed)] = p
        initial = [x['sample_id'] for x in part if x['role']=='l0']
        pool = [x['sample_id'] for x in part if x['role']=='u0']
        assert len(initial)==333 and len(pool)==2997
        snapshot['initial'][str(seed)] = initial
        snapshot['pool'][str(seed)] = pool
        # Frozen catalog has only allowed metadata/predictions on candidates.
        catalog = rd.read(llm_dirs('nocw',seed,0)/'catalog.json')
        assert catalog is not None
        for row in catalog['candidates']+catalog['pending']+catalog['observed']:
            clean = {k:row[k] for k in ('id','smiles','conditions')}
            assert row['smiles'] == p[row['id']]['canonical_smiles']
            assert row['id'] not in snapshot['metadata'] or snapshot['metadata'][row['id']]==clean
            snapshot['metadata'][row['id']] = clean
        assert {x['id'] for x in catalog['observed']}==set(initial)
        assert {x['id'] for x in catalog['candidates']+catalog['pending']}==set(pool)
        init_fit = None
        for method in METHODS:
            labeled = list(initial)
            f0 = fit_read(rd,method,seed,0)
            assert f0 is not None and f0['train_ids_hash']==ids_hash(labeled)
            if init_fit is None:
                init_fit = f0
            for key in ('initialization_hash','validation_ids_hash','checkpoint_sha256'):
                assert f0[key] == init_fit[key], (method,seed,key)
            snapshot['fits'].append(f0)
            maxrounds = 33 if method=='nocw' else 21 if method=='hybrid' else 36
            accepted = completed = rejected = 0
            for r in range(maxrounds):
                directory = llm_dirs(method,seed,r) if method in ('nocw','free') else num_dir(method,seed,r)
                if method in ('nocw','free'):
                    freeze = rd.read(directory/'batch_freeze.json', optional=True)
                    if freeze is None:
                        break
                    assert not freeze['pending_ids'] and not freeze.get('dry_run',False)
                    hashes = freeze['artifact_hashes']
                    contract = rd.read(directory/'contract.json',expected=hashes['contract.json'])
                    selection = rd.read(directory/'selection.json',expected=hashes['selection.json'])
                    cat = rd.read(directory/'catalog.json',expected=hashes['catalog.json'])
                    if any(x is None for x in (contract,selection,cat)):
                        break
                    selected = freeze['batch_ids']
                    assert selected==[x['id'] for x in selection['choices']]
                    assert set(contract['L_t_ids']) == set(labeled)
                    unlabeled = contract['U_t_ids']
                    assert set(unlabeled) == set(pool)-set(labeled)
                    assert {x['id'] for x in cat['observed']}==set(labeled)
                    assert [x['id'] for x in cat['candidates']+cat['pending']]==unlabeled
                    for row in cat['candidates']+cat['pending']+cat['observed']:
                        assert {k:row[k] for k in ('id','smiles','conditions')}==snapshot['metadata'][row['id']]
                    source = directory/'batch_freeze.json'
                    reject_n = sum(n.startswith('rejected_') for n in hashes)
                    checkpoint = contract['checkpoint_sha256']
                    assert freeze.get('checkpoint_sha256',checkpoint)==checkpoint
                    cfg = 'gpt-6.1-sol/xhigh'
                    if method=='free' and ((seed==157 and r<=5) or (seed==6101 and r<=9)):
                        cfg = 'Astra/high prefix; r5 seed157 authorized ID correction' if seed==157 and r==5 else 'Astra/high'
                    choice_note = {k:selection.get(k) for k in ('batch_rationale','batch_strategy','hypotheses')}
                else:
                    acq = directory/'acquisition_artifacts'
                    contract = rd.read(acq/'contract.json',optional=True)
                    if contract is None:
                        break
                    table = rd.read(acq/'selected_next_batch.csv','csv',expected=contract['files']['acquisition_artifacts/selected_next_batch.csv'])
                    if table is None:
                        break
                    selected = [x['sample_id'] for x in table]
                    assert stable(selected)==contract['selected_ids_ordered_hash']
                    assert contract['input']['L_t_ids_hash']==ids_hash(labeled)
                    unlabeled = [i for i in pool if i not in set(labeled)]
                    assert contract['input']['U_t_ids_hash']==ids_hash(unlabeled)
                    checkpoint = contract['input']['evaluation_checkpoint_sha256']
                    source = acq/'selected_next_batch.csv'
                    reject_n = 0
                    cfg = 'not_applicable'
                    choice_note = {}
                assert len(selected)==len(set(selected))==32
                assert not set(selected)&set(labeled) and set(selected)<=set(unlabeled)
                prior = snapshot['fits'][-1]
                assert prior['method']==method and prior['seed']==seed and prior['round']==r
                assert checkpoint==prior['checkpoint_sha256']
                accepted += 1
                rejected += reject_n
                nxt = fit_read(rd,method,seed,r+1,False)
                is_complete = nxt is not None
                if is_complete:
                    assert nxt['train_ids_hash']==ids_hash(labeled+selected)
                    assert nxt['initialization_hash']==init_fit['initialization_hash']
                    assert nxt['validation_ids_hash']==init_fit['validation_ids_hash']
                    snapshot['fits'].append(nxt)
                    completed += 1
                batch = {'method':method,'seed':seed,'round':r,'L_before':333+32*r,'L_after':365+32*r,
                    'selected_ids':selected,'labeled_ids':list(labeled),'unlabeled_ids':unlabeled,
                    'source':rel(source),'source_sha256':rd.manifest[rel(source)]['sha256'],
                    'contract_source':rel(directory/'contract.json' if method in ('nocw','free') else directory/'acquisition_artifacts/contract.json'),
                    'checkpoint_sha256':checkpoint,'accepted':True,'completed_fit_validation':is_complete,
                    'rejected_responses_before_acceptance':reject_n,'llm_config':cfg,'notes':choice_note,
                    'validation_nrmse_before':prior['best_validation_combined_normalized_rmse'],
                    'validation_nrmse_after':nxt['best_validation_combined_normalized_rmse'] if nxt else None}
                snapshot['batches'].append(batch)
                labeled.extend(selected)
                if not is_complete:
                    break
            snapshot['status'].append({'method':method,'seed':seed,'accepted_batches':accepted,
                'completed_batches':completed,'completed_L':333+32*completed,
                'accepted_L':333+32*accepted,'rejected_responses_in_accepted_batch_artifacts':rejected})
    # Inventory terminal request status in NoCW, without treating any response as a selection.
    # Prefix physical attempts belong to NOLD; NEXT contributes only extension rounds >=21.
    attempts = []
    for run in (NOLD,NEXT):
        candidates = list((run/'selections').glob('seed_*/llm32_nocw/round_*/attempt_*.json'))
        for path in sorted(candidates):
            if path.name.endswith('.started.json'):
                continue
            rnd = int(path.parent.name.split('_')[-1])
            if run==NEXT and rnd<21:
                continue
            value = rd.read(path,optional=True)
            if value is None:
                continue
            if not isinstance(value.get('success'),bool):
                rd.missing.append({'path':rel(path),'reason':'nonterminal attempt ignored','optional':True})
                continue
            attempts.append({'source':rel(path),'success':value['success'],'error_type':value.get('error_type'),
                'seed':int(path.parents[2].name.split('_')[-1]),'round':rnd})
    snapshot['nocw_terminal_attempts'] = attempts
    # Preserve and hash existing audit context; do not run their collectors/writers.
    audit_paths = [ROOT/'reports/nocw_progress_to1389_20261101/accepted_batch_index.json',
        ROOT/'reports/nocw_progress_to1389_20261101/fit_audits.json',
        ROOT/'studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/behavior_data_audit.json',
        ROOT/'studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/runtime_integrity_audit.json',
        ROOT/'studies/active_learning/verifiable_scientific_agent_20261030/history_audit_summary.json']
    for path in audit_paths:
        rd.read(path,optional=True)
    payload=(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    (out/'snapshot_inputs.json.xz').write_bytes(lzma.compress(payload,preset=6))
    write_json(out/'input_manifest.json',{'snapshot_time':rd.cutoff,'git_head':snapshot['git_head'],
        'policy':'Files newer than cutoff, changing during double-read, invalid or hash-conflicting are excluded. No original response table/checkpoint weights/prediction payload is opened.',
        'files':list(rd.manifest.values())})
    write_json(out/'missing_inputs.json',{'items':rd.missing})
    print('Captured',len(snapshot['batches']),'batches;',len(rd.manifest),'stable sources at',rd.cutoff,flush=True)


def phase(L):
    return 'early_333_525' if L<525 else 'middle_525_813' if L<813 else 'late_813_common'


CORE = ['unique_molecules','new_molecules','new_molecule_slot_fraction','max_molecule_slot_fraction',
        'cumulative_molecules_after','within_any_single_selected_fraction','cross_any_single_selected_fraction',
        'cross_only_pe_selected_fraction','cross_only_sample_volume_selected_fraction',
        'cross_only_loading_volume_selected_fraction','cross_exact_selected_fraction']


def summarize(frame,out,common):
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in ('seed','round','L_before','L_after')]
    rows = []
    paired = []
    for subset,seeds in [('all5',SEEDS),('sol_consistent3',(887,2357,12203))]:
        for window in ['early_333_525','middle_525_813','late_813_common','all_common']:
            x = frame[(frame.seed.isin(seeds)) & frame.in_common_range & frame.completed_fit_validation]
            if window!='all_common':
                x=x[x.phase==window]
            seedrows=[]
            for (m,s),g in x.groupby(['method','seed']):
                val={c:float(g[c].mean()) for c in numeric}
                val.update(cumulative_molecules_after=float(g.sort_values('L_after').iloc[-1].cumulative_molecules_after))
                val.update(new_molecules_total=int(g.new_molecules.sum()))
                for c in numeric:
                    if c.endswith('_pairs'):
                        val[c+'_total']=int(g[c].sum())
                row={'subset':subset,'aggregation':'seed','phase':window,'method':m,'seed':s,
                    'n_seeds':1,'n_batches':len(g),'L_end':int(g.L_after.max()),**val}
                rows.append(row);seedrows.append(row)
            temp=pd.DataFrame(seedrows)
            for m,g in temp.groupby('method'):
                for agg in ('mean','sd'):
                    columns=[c for c in g.select_dtypes(include=[np.number]).columns if c not in ('seed','n_seeds')]
                    vals={c:float(g[c].mean() if agg=='mean' else g[c].std(ddof=1)) for c in columns}
                    rows.append({'subset':subset,'aggregation':'group_'+agg,'phase':window,'method':m,'seed':'ALL', 'n_seeds':len(g),**vals})
            for comparator in ('free','hybrid','lcmd'):
                a=temp[temp.method=='nocw'].set_index('seed')
                b=temp[temp.method==comparator].set_index('seed')
                for metric in CORE:
                    delta=(a[metric]-b[metric]).dropna()
                    paired.append({'subset':subset,'phase':window,'contrast':'nocw_minus_'+comparator,
                        'metric':metric,'mean_delta':delta.mean(),'sd_delta':delta.std(ddof=1),'n_seeds':len(delta),
                        'positive_seeds':';'.join(map(str,delta[delta>1e-12].index)),
                        'negative_seeds':';'.join(map(str,delta[delta< -1e-12].index)),
                        'zero_seeds':';'.join(map(str,delta[delta.abs()<=1e-12].index))})
    pd.DataFrame(rows).to_csv(out/'seed_phase_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_seed_differences.csv',index=False)


def analyze(out):
    snap=load_snapshot(out)
    meta=snap['metadata']
    completed=[b for b in snap['batches'] if b['completed_fit_validation']]
    common=min(s['completed_L'] for s in snap['status'])
    assert common>=813
    rows=[]; cases=[]
    for b in snap['batches']:
        selected=b['selected_ids']; previous=b['labeled_ids']
        counts=pd.Series([meta[i]['smiles'] for i in selected]).value_counts()
        prevsm={meta[i]['smiles'] for i in previous}
        novel=set(counts.index)-prevsm
        metrics, examples=compute_condition_metrics(selected,previous,meta,return_examples=1)
        metrics.update({k.replace('_selected_slots_fraction','_selected_fraction'):v
                        for k,v in list(metrics.items()) if k.endswith('_selected_slots_fraction')})
        # Alias depends only on explicit module naming; preserve all raw pair metrics.
        base={k:v for k,v in b.items() if k not in ('selected_ids','labeled_ids','unlabeled_ids','notes')}
        row={**base,'phase':phase(b['L_before']),'in_common_range':b['L_after']<=common,
            'batch_size':len(selected),'unique_molecules':len(counts),'new_molecules':len(novel),
            'new_molecule_slots':sum(counts[s] for s in novel),
            'new_molecule_slot_fraction':sum(counts[s] for s in novel)/len(selected),
            'max_molecule_slots':int(counts.max()),'max_molecule_slot_fraction':counts.max()/len(selected),
            'molecule_slot_histogram':json.dumps({str(k):int(v) for k,v in counts.value_counts().sort_index().items()}),
            'cumulative_molecules_before':len(prevsm),'cumulative_molecules_after':len(prevsm|set(counts.index)),
            'unique_selected_ids':len(set(selected)),'overlap_previous_ids':len(set(selected)&set(previous)),
            'missing_flags':'geometry_separate; no_responses_used',**metrics}
        rows.append(row)
        if b['seed']==2357 and b['round'] in (3,10,17):
            cases.append({**base,'selection_note':b['notes'],'selected':[meta[i] for i in selected],
                'condition_examples':examples,'unique_molecules':len(counts),'new_molecules':len(novel),
                'max_molecule_slots':int(counts.max())})
    df=pd.DataFrame(rows)
    df.to_csv(out/'batch_metrics.csv',index=False)
    summarize(df,out,common)
    pd.DataFrame(snap['status']).to_csv(out/'completion_status.csv',index=False)
    pd.DataFrame(snap['fits']).to_csv(out/'validation_fits.csv',index=False)
    extended=[]
    for (method,L),g in df[df.completed_fit_validation].groupby(['method','L_after']):
        item={'method':method,'L_after':int(L),'n_seeds':len(g),
              'seeds':';'.join(map(str,sorted(g.seed))),
              'in_all_method_all_seed_common':bool(g.in_common_range.all())}
        for col in CORE+['validation_nrmse_after']:
            item[col+'_mean']=float(g[col].mean())
            item[col+'_sd']=float(g[col].std(ddof=1))
        extended.append(item)
    pd.DataFrame(extended).to_csv(out/'extended_trajectory_summary.csv',index=False)
    write_json(out/'representative_batches.json',{'rule':'Seed 2357 (middle seed in sorted prespecified seeds); round 3,10,17, one fixed interior budget in each descriptive phase; all four methods. No metric/performance filtering.', 'cases':cases})
    checks={'status':'PASS','common_completed_L':common,'batches_total':len(df),
        'completed_batches':len(completed),'accepted_only_batches':len(df)-len(completed),
        'main_batches':int((df.in_common_range & df.completed_fit_validation).sum()),
        'metadata_row_ids':len(meta),'metadata_molecule_keys':len({r['smiles'] for r in meta.values()}),
        'all_batches_32_unique_ids':bool((df.batch_size.eq(32)&df.unique_selected_ids.eq(32)).all()),
        'no_cross_round_repeats':bool(df.overlap_previous_ids.eq(0).all()),
        'main_phase_batch_counts':df[df.in_common_range].groupby('phase').size().to_dict(),
        'phase_rule':'L_before: [333,525), [525,813), [813,common_end). L_after<=common_end; boundary 525 belongs to middle, 813 to late.',
        'proportions':'selected slots / 32 per batch. Phase average first within each seed; across-seed mean and sample SD ddof=1. Pair totals also retained; single-match categories can overlap in slots.',
        'safety':snap['safety']}
    for (m,s),g in df.groupby(['method','seed']):
        assert g.sort_values('round').L_before.tolist()==list(range(333,333+32*len(g),32))
        assert g.new_molecules.sum()==g.iloc[-1].cumulative_molecules_after-g.iloc[0].cumulative_molecules_before
    write_json(out/'verification.json',checks)
    plot(df,pd.DataFrame(snap['fits']),out,common)
    print(json.dumps(checks,ensure_ascii=False),flush=True)


def plot(df,fits,out,common):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'nocw':'#db5137','free':'#7755aa','hybrid':'#cf941e','lcmd':'#168e8c'}
    names={'nocw':'NoCW LLM32','free':'Free LLM32','hybrid':'Numerical Hybrid32','lcmd':'Gradient-LCMD32'}
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,(metric,title) in zip(axs.flat,[('unique_molecules','Distinct molecules in each batch / 32 slots'),
        ('new_molecule_slot_fraction','Slots on previously unobserved molecules'),
        ('max_molecule_slot_fraction','Largest within-batch molecule share'),
        ('cumulative_molecules_after','Cumulative distinct molecules (includes initial 333)')]):
        for m in METHODS:
            x=df[(df.method==m)&df.in_common_range&df.completed_fit_validation]
            g=x.groupby('L_after' if metric=='cumulative_molecules_after' else 'L_before')[metric]
            mean=g.mean();sd=g.std(ddof=1)
            ax.plot(mean.index,mean,label=names[m],color=colors[m]);ax.fill_between(mean.index,mean-sd,mean+sd,color=colors[m],alpha=.10)
        ax.set_title(title,fontsize=10);ax.set_xlabel('L_after' if metric=='cumulative_molecules_after' else 'L_before')
        ax.axvline(525,color='gray',ls=':',lw=.7);ax.axvline(813,color='gray',ls=':',lw=.7);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle(f'Molecule coverage and concentration | common completed L{common}\nMean +/- sample SD across five seeds; descriptive historical trajectories',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'molecule_coverage.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,(metric,title) in zip(axs.flat,[('within_any_single_selected_fraction','Within batch: slots in any single-factor pair'),
        ('cross_only_pe_selected_fraction','Versus previous L: PE/EA-only pair slots'),
        ('cross_only_sample_volume_selected_fraction','Versus previous L: sample-volume-only pair slots'),
        ('cross_only_loading_volume_selected_fraction','Versus previous L: loading-volume-only pair slots')]):
        for m in METHODS:
            x=df[(df.method==m)&df.in_common_range&df.completed_fit_validation]
            g=x.groupby('L_before')[metric];mean=g.mean();sd=g.std(ddof=1)
            ax.plot(mean.index,mean,label=names[m],color=colors[m]);ax.fill_between(mean.index,mean-sd,mean+sd,color=colors[m],alpha=.10)
        ax.set_title(title,fontsize=10);ax.set_xlabel('L_before');ax.set_ylim(-.03,1.03);ax.grid(alpha=.15)
        ax.axvline(525,color='gray',ls=':',lw=.7);ax.axvline(813,color='gray',ls=':',lw=.7)
    axs[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('Exact same-molecule condition contrasts | 32 slots per batch\nAny-single uses union of slots; factor-specific slot fractions may overlap',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'condition_contrasts.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',action='store_true',help='Capture once into a NEW output directory; existing snapshots never overwritten.')
    parser.add_argument('--out',type=Path,default=HERE)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.capture:
        if any((args.out/name).exists() for name in ('snapshot_inputs.json','snapshot_inputs.json.gz','snapshot_inputs.json.xz')):
            raise SystemExit('Snapshot exists. Use --out NEW_SNAPSHOT_DIR or omit --capture.')
        capture(args.out)
    analyze(args.out)

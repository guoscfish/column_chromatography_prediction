#!/usr/bin/env python3
"""Build an explicitly projected, target-free historical state snapshot."""
from pathlib import Path
import argparse, csv, hashlib, io, json, lzma, subprocess
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101'
PAIR = ROOT / 'studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112'
CW = ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
SEEDS = [157, 887, 2357, 6101, 12203]
FILES = {}

def read(p):
    p = Path(p); b = p.read_bytes()
    assert b == p.read_bytes(), ('changing input', str(p))
    key = str(p.resolve().relative_to(ROOT))
    entry = {'path': key, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    assert key not in FILES or FILES[key] == entry
    FILES[key] = entry
    return b

def load(p):
    return json.loads(read(p))

def stable(ids, sort=False):
    v = sorted(ids) if sort else ids
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def rel(p):
    return str(Path(p).relative_to(ROOT))

def csvread(p):
    return list(csv.DictReader(io.StringIO(read(p).decode())))

def llm_state(method, seed, r, d):
    f = load(d/'batch_freeze.json'); c = load(d/'contract.json'); s = load(d/'selection.json')
    assert f['artifact_hashes']['contract.json'] == FILES[rel(d/'contract.json')]['sha256']
    assert f['artifact_hashes']['selection.json'] == FILES[rel(d/'selection.json')]['sha256']
    assert f['batch_ids'] == [v['id'] for v in s['choices']]
    assert not f['pending_ids']
    assert f['checkpoint_sha256'] == c['checkpoint_sha256']
    return dict(method=method, seed=seed, round=r, L_before=333+32*r, L_after=365+32*r,
        labeled_ids=c['L_t_ids'], unlabeled_ids=c['U_t_ids'], selected_ids=f['batch_ids'],
        source=rel(d/'batch_freeze.json'), source_sha256=FILES[rel(d/'batch_freeze.json')]['sha256'],
        contract_source=rel(d/'contract.json'), checkpoint_sha256=c['checkpoint_sha256'])

def cw_directory(seed, r):
    d = PAIR/f'cw_lcmd/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
    if (d/'fit_receipt.json').exists():
        return d
    names = ['qgeognn_v2_row_cw_lcmd_to_1005','qgeognn_v2_row_cw_lcmd_to_653',
             'qgeognn_v2_row_cw_lcmd_to_525','qgeognn_v2_row_short_sequential_b32']
    return next(ROOT/f'studies/active_learning/{n}/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}'
        for n in names if (ROOT/f'studies/active_learning/{n}/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}/round_freeze.json').exists())

def main(config):
    global OLD, PAIR, CW
    cfg=load(config)
    OLD=ROOT/cfg['sources']['old_snapshot_directory']
    PAIR=ROOT/cfg['sources']['paired_campaign_directory']
    CW=ROOT/cfg['sources']['new_cw_directory']
    snap=json.loads(lzma.decompress(read(OLD/'snapshot_inputs.json.xz')))
    states=[]
    keep=['seed','round','L_before','L_after','selected_ids','labeled_ids','unlabeled_ids',
          'source','source_sha256','contract_source','checkpoint_sha256','llm_config']
    for b in snap['batches']:
        if b['method']=='free' and b['completed_fit_validation'] and b['L_after']<=cfg['maximum_budget']:
            states.append(dict(method='historical_free_mixed_model',**{k:b[k] for k in keep if k in b}))
    for seed in SEEDS:
        for r in range(36):
            states.append(llm_state('nocw', seed, r, PAIR/f'nocw/selections/seed_{seed}/llm32_nocw/round_{r:02d}'))
            states.append(llm_state('new_free_cw_sol61_xhigh', seed, r,
                CW/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}'))
        rows=[]
        for r in range(37):
            d=cw_directory(seed,r); t=csvread(d/'state.csv'); f=load(d/'round_freeze.json')
            h=[v['sample_id'] for v in t if v['role']=='labeled']
            u=[v['sample_id'] for v in t if v['role']=='unlabeled']
            assert len(h)==333+32*r
            if 'input' in f:
                assert f['input']['L_t_ids_hash']==stable(h,True)
                assert f['input']['U_t_ids_hash']==stable(u,True)
            rows.append((h,u,d,f))
        for r in range(36):
            h,u,d,f=rows[r]; hh,uu,_,_=rows[r+1]; sel=hh[len(h):]
            assert hh[:len(h)]==h and len(sel)==32
            assert uu==[v for v in u if v not in set(sel)]
            assert f.get('selected_next_batch',sel)==sel
            assert f.get('outgoing_selected_ids_hash',stable(sel))==stable(sel)
            states.append(dict(method='cw_gradient_lcmd32', seed=seed, round=r,
                L_before=len(h), L_after=len(hh), labeled_ids=h, unlabeled_ids=u, selected_ids=sel,
                source=rel(d/'round_freeze.json'), source_sha256=FILES[rel(d/'round_freeze.json')]['sha256'],
                contract_source=rel(d/'input_contract.json'), checkpoint_sha256=f['checkpoint_sha256']))
    # Hybrid is independently reconstructed by the source-lineage module.
    hb=load(OUT/'hybrid_batches.json')
    if isinstance(hb,dict): hb=hb.get('batches',hb.get('states'))
    for b in hb:
        b=dict(b);b['method']='free16_lcmd16';states.append(b)
    for b in states:
        h,u,s=map(set,(b['labeled_ids'],b['unlabeled_ids'],b['selected_ids']))
        assert len(h)==b['L_before'] and len(s)==len(b['selected_ids'])==32
        assert not h&u and not h&s and s<=u
        assert h|u==set(snap['initial'][str(b['seed'])])|set(snap['pool'][str(b['seed'])])
        assert h|u<=set(snap['metadata'])
    states.sort(key=lambda b:(b['method'],b['seed'],b['round']))
    assert len({(b['method'],b['seed'],b['round']) for b in states})==len(states)
    data=dict(metadata=snap['metadata'], initial=snap['initial'],pool=snap['pool'],states=states,
              repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (OUT/'states.json.xz').write_bytes(lzma.compress(json.dumps(data,ensure_ascii=False).encode()))
    curves={}
    paths=[ROOT/p for p in cfg['sources']['saved_curve_tables']]
    for p in paths:
        for row in csvread(p):
            m={'cw':'cw_gradient_lcmd32'}.get(row['method'],row['method'])
            if m not in cfg['methods']:continue
            budget=int(row['budget']);seed=int(row['seed'])
            if budget>cfg['maximum_budget']:continue
            v=float(row.get('validation_nrmse') or row['validation_combined_normalized_rmse'])
            key=(m,seed,budget)
            if key in curves: assert abs(curves[key]['nrmse']-v)<1e-12
            curves[key]=dict(method=m,seed=seed,budget=budget,nrmse=v,source=rel(p))
    pd.DataFrame(curves.values()).sort_values(['method','seed','budget']).to_csv(OUT/'row_curves.csv',index=False)
    pd.DataFrame([dict(method=b['method'],seed=b['seed'],round=b['round'],L_before=b['L_before'],L_after=b['L_after'],
        selected_n=len(b['selected_ids']),history_n=len(b['labeled_ids']),pool_n=len(b['unlabeled_ids']),
        identity_verified=True,source=b['source']) for b in states]).to_csv(OUT/'state_coverage.csv',index=False)
    (OUT/'collection_manifest.json').write_text(json.dumps(dict(sources=list(FILES.values()),states=len(states),
        candidates=sum(len(b['selected_ids']) for b in states),new_fits=0,new_llm_calls=0,truth_tables_read=0),indent=2))
    print(json.dumps({'states':len(states),'selected_events':32*len(states),'curves':len(curves)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=OUT/'config.json')
    main(ap.parse_args().config)

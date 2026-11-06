"""Necessary delivery checks; no runner, model, GPU, response or remote imports."""
import hashlib
import json
import lzma
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import analyze
from condition_metrics import run_self_tests
from geometry_metrics import self_check

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run_self_tests()
    geometry_checks=self_check()
    snap=analyze.load_snapshot(HERE)
    data=pd.read_csv(HERE/'batch_metrics.csv')
    common=min(x['completed_L'] for x in snap['status'])
    assert set(data.method)==set(analyze.METHODS)
    assert set(data.seed)==set(analyze.SEEDS)
    assert len(data)==len(snap['batches'])==584
    assert data[['method','seed','round']].duplicated().sum()==0
    assert data.batch_size.eq(32).all() and data.unique_selected_ids.eq(32).all()
    assert data.overlap_previous_ids.eq(0).all()
    assert data.L_after.eq(data.L_before+32).all()
    assert data.phase.eq(data.L_before.map(analyze.phase)).all()
    primary=data[data.in_common_range & data.completed_fit_validation]
    assert common==973 and len(primary)==400
    assert primary.groupby(['method','seed']).size().eq(20).all()
    assert primary.groupby(['method','seed','phase']).size().value_counts().to_dict()=={6:20,9:20,5:20}
    assert not data.filter(regex='condition_.*(missing|incomplete)').to_numpy().any()
    # Reconstruct the entire historical ID lineage independent of CSV counters.
    for method in analyze.METHODS:
        for seed in analyze.SEEDS:
            observed=set(snap['initial'][str(seed)])
            available=set(snap['pool'][str(seed)])
            rows=sorted((b for b in snap['batches'] if b['method']==method and b['seed']==seed),key=lambda b:b['round'])
            for b in rows:
                selected=set(b['selected_ids'])
                assert len(selected)==32 and observed==set(b['labeled_ids'])
                assert selected<=available and selected.isdisjoint(observed)
                match=data[(data.method==method)&(data.seed==seed)&(data['round']==b['round'])].iloc[0]
                molecules={snap['metadata'][i]['smiles'] for i in selected}
                previous={snap['metadata'][i]['smiles'] for i in observed}
                assert match.unique_molecules==len(molecules)
                assert match.new_molecules==len(molecules-previous)
                assert match.cumulative_molecules_after==len(molecules|previous)
                observed.update(selected);available.difference_update(selected)
    # Verify the snapshot is exactly the one used for geometry and all 883 spaces are present.
    manifest=json.loads(lzma.decompress((HERE/'geometry_input_manifest.json.xz').read_bytes()))
    assert hashlib.sha256(analyze.load_snapshot_bytes(HERE)).hexdigest()==manifest['snapshot_sha256']
    geo=pd.read_csv(HERE/'geometry_metrics.csv')
    assert len(geo)==883 and geo.available.all()
    assert geo[['method','seed','round','space']].duplicated().sum()==0
    assert geo.random_n.eq(16).all() and geo.reference_n.eq(128).all()
    for seed,g in geo.groupby('seed'):
        assert g.reference_ids_sha256.nunique()==1
    reproduction=json.loads((HERE/'geometry_reproduction_verification.json').read_text())
    for name,expected in reproduction['output_sha256'].items():
        assert sha(HERE/name)==expected
    manual=json.loads((HERE/'manual_review.json').read_text())
    assert manual['status']=='PASS'
    # Required deliverables and the expected three figures are concrete files.
    required=['REPORT.md','batch_metrics.csv','seed_phase_summary.csv','input_manifest.json',
        'missing_inputs.json','analyze.py','condition_metrics.py','geometry_metrics.py','run_geometry.py',
        'molecule_coverage.png','condition_contrasts.png','geometry_summary.png']
    assert all((HERE/p).is_file() for p in required)
    result={'status':'PASS','checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'snapshot_time':snap['snapshot_time'],'git_head':snap['git_head'],'common_completed_L':common,
        'accepted_completed_batches':len(data),'main_batches':len(primary),'geometry_space_rows':len(geo),
        'condition_self_checks':True,'geometry_self_checks':geometry_checks,
        'all_lineages_and_molecule_counts_reconstructed':True,'phase_counts_per_seed':[6,9,5],
        'manual_independent_review':'manual_review.json (15 batches; 2688 summary values)',
        'geometry_locked_reproduction':True,
        'safety':snap['safety'],'original_worktree_mutations_by_this_analysis':0}
    (HERE/'final_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    hashes={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
    (HERE/'artifact_hashes.json').write_text(json.dumps({'files':hashes},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

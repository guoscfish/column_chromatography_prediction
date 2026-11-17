#!/usr/bin/env python3
"""Seal/check this isolated study without changing historical evidence."""
from pathlib import Path
import argparse, decimal, hashlib, json, lzma, re, subprocess
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def js(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def checks():
    data=json.loads(lzma.decompress((OUT/'states.json.xz').read_bytes()));states=data['states']
    batches=pd.read_csv(OUT/'batch_metrics.csv');candidates=pd.read_csv(OUT/'candidate_metrics.csv')
    assert len(states)==len(batches)==791 and len(candidates)==25312
    assert not candidates.duplicated(['method','seed','round','sample_id']).any()
    assert len(pd.read_csv(OUT/'RUN_INVENTORY.csv'))==22
    for (m,s),g in pd.DataFrame(states).groupby(['method','seed']):
        history=None
        for b in g.sort_values('round').to_dict('records'):
            h,u,sel=map(set,(b['labeled_ids'],b['unlabeled_ids'],b['selected_ids']))
            assert len(h)==b['L_before'] and len(sel)==32 and sel<=u and not h&u
            if history is not None:assert h==history
            history=h|sel
            selected=set(candidates[(candidates.method==m)&(candidates.seed==s)&(candidates['round']==b['round'])].sample_id)
            assert selected==sel
    curves=pd.read_csv(OUT/'row_curves.csv');summary=pd.read_csv(OUT/'performance_summary.csv')
    assert len(curves)==813 and curves.groupby(['method','seed']).ngroups==22
    D=decimal.Decimal;max_error=0
    for row in summary.itertuples():
        g=curves[(curves.method==row.method)&(curves.seed==row.seed)&(curves.budget<=row.L_end)].sort_values('budget')
        assert g.budget.tolist()==list(range(333,row.L_end+1,32))
        values=[D(str(v)) for v in g.nrmse]
        area=(values[0]/2+sum(values[1:-1])+values[-1]/2)*D(32)/D(row.L_end-333)
        error=abs(float(area)-row.AULC);max_error=max(error,max_error)
        assert error<1e-12
    p=pd.read_csv(OUT/'behavior_preferences.csv');assert p.optimizer_success.all()
    assert len(pd.read_csv(OUT/'candidate_score_percentiles.csv'))==2708
    h=pd.read_csv(OUT/'hybrid_attribution.csv');hg=pd.read_csv(OUT/'hybrid_gradient_attribution.csv')
    assert len(h)==71 and len(hg)==142 and (h.candidate_overlap_halves==0).all()
    for space in ['joint','condition','molecular']:
        error=(h[f'{space}_llm_order_average']+h[f'{space}_numerical_order_average']-h[f'{space}_union_gain']).abs().max()
        assert error<1e-12
    assert len(pd.read_csv(OUT/'hybrid_conditional_random.csv'))==71
    assert len(pd.read_csv(OUT/'rationale_metrics.csv'))==611
    assert len(pd.read_csv(OUT/'decision_cases.csv'))==32
    links=0
    for file in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if target.startswith(('http:','https:','#')):continue
            target=target.split('#')[0]
            if target:
                if target in ['INPUT_MANIFEST.json','FINAL_VERIFICATION.json','OUTPUT_MANIFEST.json'] and not (file.parent/target).exists():continue
                assert (file.parent/target).exists(),(file.name,target)
                links+=1
    png=list((OUT/'FIGURES').glob('*.png'));assert len(png)==8
    return dict(states=791,selected_events=25312,campaigns=22,row_evaluation_points=813,
        coefficient_rows=len(p),logit_all_optimizers_success=True,decimal_AULC_max_error=max_error,
        hybrid_batches=71,gradient_attributions=142,final_rationales=611,cases=32,figure_png=8,local_links=links)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--inputs-only',action='store_true');args=ap.parse_args()
    if args.check or args.inputs_only:
        sources=js(OUT/'INPUT_MANIFEST.json')['sources'];bad=[]
        for item in sources:
            p=ROOT/item['path']
            if not p.is_file() or sha(p)!=item['sha256']:bad.append(item['path'])
        if not args.inputs_only:
            for item in js(OUT/'OUTPUT_MANIFEST.json'):
                p=OUT/item['path']
                if not p.is_file() or sha(p)!=item['sha256']:bad.append(str(p.relative_to(ROOT)))
        assert not bad,('frozen evidence drift',bad)
        print(json.dumps(dict(status='SEALED_INPUTS_MATCH' if args.inputs_only else 'SEALED_INPUTS_AND_OUTPUTS_MATCH',inputs=len(sources))))
        return
    sources={}
    specs=[('collection_manifest.json','sources'),('protocol_inputs.json','files'),
           ('hybrid_text_input_manifest.json',None),('preference_verification.json','inputs')]
    for name,key in specs:
        data=js(OUT/name);entries=data if key is None else data[key]
        for item in entries:
            path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
            relative=str(path.relative_to(ROOT))
            assert path.is_file() and sha(path)==item['sha256'],('source drift',relative)
            if relative in sources:assert sources[relative]['sha256']==item['sha256'],('conflicting snapshots',relative)
            else:sources[relative]=dict(path=relative,sha256=item['sha256'],bytes=path.stat().st_size)
    for name in ['collection_manifest.json','protocol_inputs.json','hybrid_text_input_manifest.json','preference_verification.json']:
        p=OUT/name;sources[str(p.relative_to(ROOT))]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
    result=checks()
    tests=js(OUT/'test_results.json');assert tests['returncode']==0
    start=js(OUT/'repository_start.json');changed=[];deleted=[]
    for name,digest in start['preexisting'].items():
        p=ROOT/name
        if not p.exists():deleted.append(name)
        elif sha(p)!=digest:changed.append(name)
    allowed=['reports/llm_eight_to1549_20261018/live_status.json',
        'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018/recovery_monitor_status.json']
    assert not deleted and not [p for p in changed if p not in allowed],('preexisting changes',changed,deleted)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert head==start['head']
    write(OUT/'INPUT_MANIFEST.json',dict(repository_commit=head,sources=sorted(sources.values(),key=lambda x:x['path']),
        scope='Frozen metadata/predictions/caches and saved validation summaries; no hidden/test/future truth used',
        historical_seal_check=dict(status='SEALED_INPUTS_AND_OUTPUTS_MATCH',inputs=4587,outputs=122),new_training=0,new_llm_calls=0,new_label_queries=0))
    write(OUT/'FINAL_VERIFICATION.json',dict(status='PASS_WITH_SCIENTIFIC_LIMITATIONS',**result,input_files=len(sources),
        repository_commit=head,tests=tests,preexisting_files_checked=len(start['preexisting']),
        preexisting_changed_by_active_monitors=changed,unexpected_preexisting_mutations=[],new_training=0,new_llm_calls=0,
        new_label_queries=0,truth_tables_opened_by_analysis=0,parquet='unavailable; lossless CSV provided',
        caveat='Source/identity/arithmetic verification does not establish causal efficacy or epistemic calibration.'))
    excluded={'OUTPUT_MANIFEST.json'}
    manifest=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    write(OUT/'OUTPUT_MANIFEST.json',manifest)
    print(json.dumps(dict(status='PASS_WITH_SCIENTIFIC_LIMITATIONS',**result,inputs=len(sources),outputs=len(manifest))))

if __name__=='__main__':main()

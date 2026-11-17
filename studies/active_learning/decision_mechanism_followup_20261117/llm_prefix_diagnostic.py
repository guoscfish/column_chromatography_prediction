#!/usr/bin/env python3
"""Audit visible display-prefix choices and saved correction trajectories only."""
from __future__ import annotations
import argparse
import hashlib
import io
import itertools
import json
import lzma
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PREVIOUS=ROOT/'studies/active_learning/llm_decision_behavior_audit_20261116'
METHODS=['historical_free_mixed_model','new_free_cw_sol61_xhigh','nocw']
FILES={}
PREFIX_TEXT=re.compile(r'first\s+(?:32|thirty[- ]two)\b|(?:前|首)\s*32\s*(?:个|条|行)',re.I)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(path,expected=None):
    path=Path(path);raw=path.read_bytes();assert raw==path.read_bytes()
    digest=sha(raw)
    if expected:assert digest==expected,('frozen hash mismatch',str(path))
    FILES[str(path.relative_to(ROOT))]={'path':str(path.relative_to(ROOT)),'sha256':digest,'bytes':len(raw)}
    return raw


def load(path,expected=None):return json.loads(read(path,expected))


def stable(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def phase(budget):return '333-525' if budget<525 else '525-813' if budget<813 else '813-1005' if budget<1005 else '1005-1485'


def prefix_identity(selected,ordered_pool):
    assert len(selected)==len(set(selected))==32
    assert set(selected)<=set(ordered_pool)
    prefix=ordered_pool[:32]
    return {'ordered_first32':selected==prefix,'set_first32':set(selected)==set(prefix),
            'overlap_first32_n':len(set(selected)&set(prefix)),
            'selected_from_first64_fraction':sum(sid in set(ordered_pool[:64]) for sid in selected)/32,
            'mean_selected_packet_position_fraction':np.mean([ordered_pool.index(sid)/(len(ordered_pool)-1) for sid in selected])}


def parse_plan(receipt):
    answer=receipt.get('answer')
    try:answer=json.loads(answer) if isinstance(answer,str) else answer
    except (json.JSONDecodeError,TypeError):return None
    if not isinstance(answer,dict) or not isinstance(answer.get('choices'),list):return None
    if not all(isinstance(v,dict) and isinstance(v.get('id'),str) for v in answer['choices']):return None
    return answer


def plan_text(plan):
    return '\n'.join([str(plan.get(k,'')) for k in ['batch_strategy','batch_rationale','feedback_interpretation']]+
                     [str(v.get('reason','')) for v in plan['choices']])


def main(args):
    for relative in ['src/qgeognn_al/active_learning_v2/scientist_transport.py',
                     'scripts/studies/run_qgeognn_v2_row_llm_nocw_unbounded_v2.py',
                     'scripts/studies/run_qgeognn_v2_row_llm_nocw_unbounded_v3.py',
                     'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py']:
        read(ROOT/relative)
    snapshot=json.loads(lzma.decompress(read(args.snapshot)))
    states=[x for x in snapshot['states'] if x['method'] in METHODS]
    rows=[];turns=[];cases=[];prefix_states=[];initial_provenance=[]
    for number,state in enumerate(states):
        directory=(ROOT/state['source']).parent
        freeze=load(directory/'batch_freeze.json',state['source_sha256'])
        packet=load(directory/'input.json',freeze['artifact_hashes']['input.json'])
        selection=load(directory/'selection.json',freeze['artifact_hashes']['selection.json'])
        id_column=packet['candidates']['columns'].index('id')
        pool=[v[id_column] for v in packet['candidates']['rows']]
        assert len(pool)==len(set(pool))==len(state['unlabeled_ids']) and set(pool)==set(state['unlabeled_ids'])
        selected=[v['id'] for v in selection['choices']]
        assert selected==freeze['batch_ids']==state['selected_ids']
        names=sorted(n for n in freeze['artifact_hashes'] if Path(n).name.startswith('turn_') and n.endswith('.json') and not n.endswith('.started.json'))
        rejection_paths=sorted(directory.glob('rejected*.json'))
        row={k:state[k] for k in ['method','seed','round','L_before','L_after']}
        row.update(phase=phase(state['L_before']),**prefix_identity(selected,pool),saved_turn_artifact_n=len(names),
                   recorded_rejection_files_n=len(rejection_paths),has_recorded_rejection=bool(rejection_paths),
                   source=state['source'],source_sha256=state['source_sha256'],
                   packet_sha256=packet['packet_sha256'],
                   selected_ID_order_sha256=stable(selected),packet_first32_ID_order_sha256=stable(pool[:32]),
                   selection_without_packet_sha256=stable({k:v for k,v in selection.items() if k!='packet_sha256'}),
                   choice_reason_order_sha256=stable([v.get('reason','') for v in selection['choices']]))
        text=plan_text(selection);match=PREFIX_TEXT.search(text)
        row['final_text_explicit_prefix_rule_match']=bool(match)
        row['final_text_prefix_excerpt']=text[max(0,match.start()-100):match.end()+180] if match else ''
        # Inspect completed proposal contents for every initial state and every prefix case.
        if row['set_first32'] or state['round']==0:
            proposals=[]
            for name in names:
                receipt=load(directory/name,freeze['artifact_hashes'][name]);plan=parse_plan(receipt)
                if plan is None:continue
                ids=[v['id'] for v in plan['choices']];legal=[sid for sid in ids if sid in set(pool)]
                provenance=receipt.get('provenance',{});m=PREFIX_TEXT.search(plan_text(plan))
                turn={**{k:row[k] for k in ['method','seed','round','L_before','phase']},'turn_name':name,
                      'turn_source':str((directory/name).relative_to(ROOT)),'turn_sha256':FILES[str((directory/name).relative_to(ROOT))]['sha256'],
                      'request_sha256':receipt.get('request_sha256',''),'response_id_sha256':stable(provenance.get('response_id','')),
                      'served_model':provenance.get('served_model',''),'complete_proposal_choices':len(ids),
                      'unique_choice_ids':len(set(ids)),'legal_choice_occurrences':len(legal),'unique_legal_choices':len(set(legal)),
                      'unknown_ID_occurrences':sum(sid not in set(pool) and sid not in set(state['labeled_ids']) for sid in ids),
                      'previously_measured_ID_occurrences':sum(sid in set(state['labeled_ids']) for sid in ids),
                      'accepted_selection_exact_match':plan==selection,'final_ID_order_match':ids==selected,
                      'final_ID_set_overlap_n':len(set(legal)&set(selected)),
                      'ordered_first32':ids==pool[:32], 'set_first32':set(ids)==set(pool[:32]) and len(ids)==32,
                      'explicit_prefix_text_rule_match':bool(m),
                      'prefix_text_excerpt':plan_text(plan)[max(0,m.start()-80):m.end()+200] if m else '',
                      'choice_ID_order':json.dumps(ids),'choice_reason_order_sha256':stable([v.get('reason','') for v in plan['choices']])}
                turns.append(turn);proposals.append(turn)
                if plan==selection:
                    row['accepted_request_sha256']=turn['request_sha256'];row['accepted_response_id_sha256']=turn['response_id_sha256']
                    row['accepted_turn_sha256']=turn['turn_sha256'];row['accepted_turn_name']=name
            if row['set_first32']:
                prefix_states.append(state)
                accepted=[p for p in proposals if p['accepted_selection_exact_match']]
                previous=[]
                if len(accepted)==1:
                    at=next(j for j,p in enumerate(proposals) if p['accepted_selection_exact_match'])
                    previous=proposals[:at]
                corrections=[]
                for path in rejection_paths:
                    item=load(path,freeze['artifact_hashes'].get(path.name))
                    corrections.append({'path':str(path.relative_to(ROOT)),'sha256':FILES[str(path.relative_to(ROOT))]['sha256'],
                                        'error':str(item.get('error','')),'instruction':str(item.get('instruction','')),
                                        'explicit_first32_instruction':bool(PREFIX_TEXT.search(json.dumps(item,ensure_ascii=False)))})
                last=previous[-1] if previous else {}
                first=proposals[0] if proposals else {}
                cases.append({**row,'accepted_receipt_unique_match':len(accepted)==1,
                              'first_complete_proposal_source':first.get('turn_source',''),
                              'first_complete_proposal_sha256':first.get('turn_sha256',''),
                              'first_complete_proposal_choices_n':first.get('complete_proposal_choices'),
                              'first_unique_legal_choices_n':first.get('unique_legal_choices'),
                              'first_proposal_final_overlap_n':first.get('final_ID_set_overlap_n'),
                              'first_proposal_already_ordered_prefix':first.get('ordered_first32'),
                              'first_proposal_choice_ID_order':first.get('choice_ID_order',''),
                              'previous_complete_proposal_n':len(previous),
                              'previous_proposal_source':last.get('turn_source',''),
                              'previous_proposal_turn_sha256':last.get('turn_sha256',''),
                              'previous_proposal_choices_n':last.get('complete_proposal_choices'),
                              'previous_unique_legal_choices_n':last.get('unique_legal_choices'),
                              'previous_proposal_final_overlap_n':last.get('final_ID_set_overlap_n'),
                              'previous_proposal_choice_ID_order':last.get('choice_ID_order',''),
                              'final_choice_ID_order':json.dumps(selected),
                              'correction_records':json.dumps(corrections,ensure_ascii=False),
                              'correction_explicit_prefix_instruction':any(x['explicit_first32_instruction'] for x in corrections),
                              'final_batch_strategy_excerpt':selection.get('batch_strategy','')[:900],
                              'final_first_three_reasons':json.dumps([v.get('reason','') for v in selection['choices'][:3]],ensure_ascii=False)})
        rows.append(row)
        if number%100==0:print('display prefix states',number,'/',len(states),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'llm_display_prefix_batches.csv',index=False)
    pd.DataFrame(turns).to_csv(OUT/'llm_prefix_proposal_trajectories.csv',index=False)
    pd.DataFrame(cases).to_csv(OUT/'llm_prefix_cases.csv',index=False)
    frame.groupby(['method','seed','phase']).agg(batches=('round','size'),set_prefix_batches=('set_first32','sum'),
        ordered_prefix_batches=('ordered_first32','sum'),recorded_rejection_batches=('has_recorded_rejection','sum'),
        mean_first64_fraction=('selected_from_first64_fraction','mean')).reset_index().to_csv(OUT/'llm_prefix_seed_phase.csv',index=False)
    frame.groupby(['method','has_recorded_rejection']).agg(batches=('round','size'),set_prefix_batches=('set_first32','sum'),
        ordered_prefix_batches=('ordered_first32','sum')).reset_index().to_csv(OUT/'llm_prefix_by_rejection.csv',index=False)
    initial=frame[frame['round']==0].set_index(['method','seed']);pairwise=[]
    for seed in [157,887,2357,6101,12203]:
        for method_a,method_b in itertools.combinations(METHODS,2):
            a,b=initial.loc[(method_a,seed)],initial.loc[(method_b,seed)]
            selected_a=next(s['selected_ids'] for s in states if s['method']==method_a and s['seed']==seed and s['round']==0)
            selected_b=next(s['selected_ids'] for s in states if s['method']==method_b and s['seed']==seed and s['round']==0)
            pairwise.append({'seed':seed,'method_a':method_a,'method_b':method_b,
                'selected_ID_overlap_n':len(set(selected_a)&set(selected_b)),
                'ordered_ID_lists_identical':a.selected_ID_order_sha256==b.selected_ID_order_sha256,
                'selection_without_packet_identical':a.selection_without_packet_sha256==b.selection_without_packet_sha256,
                'choice_reason_order_identical':a.choice_reason_order_sha256==b.choice_reason_order_sha256,
                'accepted_request_hash_identical':a.accepted_request_sha256==b.accepted_request_sha256,
                'accepted_response_ID_hash_identical':a.accepted_response_id_sha256==b.accepted_response_id_sha256,
                'first32_method_a':bool(a.set_first32),'first32_method_b':bool(b.set_first32),
                'request_sha256_a':a.accepted_request_sha256,'request_sha256_b':b.accepted_request_sha256,
                'response_id_sha256_a':a.accepted_response_id_sha256,'response_id_sha256_b':b.accepted_response_id_sha256,
                'selection_without_packet_sha256_a':a.selection_without_packet_sha256,'selection_without_packet_sha256_b':b.selection_without_packet_sha256,
                'reason_order_sha256_a':a.choice_reason_order_sha256,'reason_order_sha256_b':b.choice_reason_order_sha256,
                'ordered_ID_hash_a':a.selected_ID_order_sha256,'ordered_ID_hash_b':b.selected_ID_order_sha256})
    pd.DataFrame(pairwise).to_csv(OUT/'initial_pairwise_provenance.csv',index=False)
    if (OUT/'condition_allocation.csv').exists() and (OUT/'condition_factor_allocation.csv').exists():
        left=pd.read_csv(io.BytesIO(read(OUT/'condition_allocation.csv')))
        left=left[(left.source_half=='all32')&left.method.isin(METHODS)]
        right=pd.read_csv(io.BytesIO(read(OUT/'condition_factor_allocation.csv'))).rename(columns={'factor':'axis'})
        joined=left.merge(right,on=['method','seed','round','L_before','phase','axis'],validate='one_to_one')
        maxhistory=float((joined.history_control_fraction-joined.history_single_factor_slot_fraction).abs().max())
        maxwithin=int((joined.within_pair_n-joined.within_batch_single_factor_pairs).abs().max())
        assert len(joined)==2700 and maxhistory==maxwithin==0
        qa={'status':'PASS','rows':len(joined),'maximum_history_fraction_difference':maxhistory,
            'maximum_within_pair_count_difference':maxwithin,
            'sources':[FILES[str((OUT/name).relative_to(ROOT))] for name in ['condition_allocation.csv','condition_factor_allocation.csv']]}
        (OUT/'llm_allocation_crosscheck.json').write_text(json.dumps(qa,indent=2)+'\n')
    if (OUT/'llm_score_adjustment.csv').exists():
        scores=pd.read_csv(io.BytesIO(read(OUT/'llm_score_adjustment.csv')))
        initial_summary=scores[scores['round']==0].groupby(['method','feature','adjustment']).agg(
            states=('round','size'),mean_selected_excess=('selected_excess_percentile','mean'),
            mean_projection_R2=('projection_R2','mean')).reset_index()
        initial_summary.to_csv(OUT/'llm_initial_adjustment_summary.csv',index=False)
        complete_u=scores[scores.feature=='U_proxy'].groupby(['method','seed','adjustment']).selected_excess_percentile.mean().groupby(['method','adjustment']).mean().reset_index()
        complete_u.to_csv(OUT/'llm_U_full_trajectory_adjustment.csv',index=False)
        os.environ.setdefault('MPLCONFIGDIR','/private/tmp/decision_llm_figure_mpl')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
        labels=['Historical CW-Free','New CW-Free','NoCW-Free'];colors=['#9467ad','#0085a1','#db6900']
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),layout='constrained')
        for j,(method,label,color) in enumerate(zip(METHODS,labels,colors)):
            g=frame[frame.method==method];fraction=g.set_first32.mean()*100
            axes[0].bar(j,fraction,color=color,alpha=.7)
            seed_rates=g.groupby('seed').set_first32.mean()*100
            axes[0].scatter(j+np.linspace(-.12,.12,len(seed_rates)),seed_rates,c=color,s=25,edgecolors='white',linewidth=.5)
            axes[0].text(j,fraction+1.2,f'{int(g.set_first32.sum())}/{len(g)}',ha='center',fontsize=10)
            stages=['raw','q50_polynomial','q50_molecule_conditions']
            y=[float(initial_summary[(initial_summary.method==method)&(initial_summary.feature=='U_proxy')&
                                     (initial_summary.adjustment==stage)].mean_selected_excess.iloc[0]) for stage in stages]
            axes[1].plot(range(3),y,'o-',color=color,label=label,lw=1.5,markersize=5)
        axes[0].set_xticks(range(3),labels,rotation=12);axes[0].set_ylabel('Accepted exact display-prefix batches (%)')
        axes[0].set_title('Correction-linked prefix behavior\nSmall points: five seed rates')
        axes[0].set_ylim(0,48)
        axes[1].set_xticks(range(3),['Raw','q50 controls','q50 + molecule\n+ condition controls'])
        axes[1].set_ylabel('Selected U-proxy excess over legal pool')
        axes[1].axhline(0,c='gray',ls=':',lw=.8);axes[1].set_title('Shared initial states: score composition\nDescriptive projection, not information value')
        axes[1].legend(frameon=False,fontsize=8)
        fig.suptitle('Two mechanisms to resolve before another full LLM rollout',fontsize=12)
        (OUT/'figures').mkdir(exist_ok=True)
        for extension in ['png','pdf','svg']:
            fig.savefig(OUT/'figures'/f'llm_correction_and_response_scale.{extension}',dpi=190,bbox_inches='tight')
        plt.close(fig)
    verified={'status':'PASS','states':len(frame),'prefix_set_n':int(frame.set_first32.sum()),'prefix_ordered_n':int(frame.ordered_first32.sum()),
              'prefix_after_recorded_rejection_n':int((frame.set_first32&frame.has_recorded_rejection).sum()),
              'prefix_case_exact_accepted_receipt_matches':int(sum(c['accepted_receipt_unique_match'] for c in cases)),
              'prefix_final_text_rule_matches':int(frame.loc[frame.set_first32,'final_text_explicit_prefix_rule_match'].sum()),
              'corrections_explicitly_request_first32_n':int(sum(c['correction_explicit_prefix_instruction'] for c in cases)),
              'prefix_cases_already_prefix_in_first_complete_proposal_n':int(sum(c['first_proposal_already_ordered_prefix'] for c in cases)),
              'new_llm_calls':0,'new_training':0,'test_truth_reads':0,'unobserved_truth_reads':0,
              'interpretation':'An observed accepted display-prefix choice; no automatic fallback, copying, intent, or learning-performance causality is inferred.',
              'sources':list(FILES.values())}
    (OUT/'llm_prefix_inputs.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verified.items() if k!='sources'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,default=PREVIOUS/'states.json.xz')
    main(parser.parse_args())

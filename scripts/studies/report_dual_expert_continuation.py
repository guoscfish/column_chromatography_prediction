#!/usr/bin/env python3
"""Render completed, validation-only dual-expert continuation results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.qgeognn_al.active_learning_v2 import dual_expert_study as study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',required=True)
    p.add_argument('--rounds',type=int,required=True,choices=range(3,len(study.BUDGETS)))
    p.add_argument('--seed',type=int,default=157,choices=study.SEEDS)
    a=p.parse_args();study.configure_revision(a.revision)
    record=study.report(a.seed,study.METHOD,a.rounds)
    path=Path(record['results']);out=path.parent;r=json.loads(path.read_text())
    styles={study.METHOD:('Dual expert: Astra -> Sol 6.1','#2463ba'),
        'free_llm32_scientist_v2':('Free LLM32','#7652ad'),
        'cw16_llm16_scientist_v2':('CW16 + LLM16','#087f8c')}
    curves={m:sorted([row for row in r['learning_curves'] if row['method']==m],key=lambda row:row['budget']) for m in styles}
    latest=study.BUDGETS[a.rounds];summary=[]
    common=min(curve[-1]['budget'] for curve in curves.values())
    endpoints=[{'method':m,'budget':curve[-1]['budget'],'validation_NRMSE':curve[-1]['validation_NRMSE']} for m,curve in curves.items()]
    for m,curve in curves.items():
        forward=[row for row in curve if 397<=row['budget']<=common]
        summary.append({'method':m,'latest_budget':common,'validation_NRMSE':forward[-1]['validation_NRMSE'],
            'validation_AULC_397_latest':float(np.trapezoid([row['validation_NRMSE'] for row in forward],
                [row['budget'] for row in forward])/(common-397))})
    unique={c['response_id']:c for c in r['llm_calls'] if c.get('response_id')}
    for response_path in study.STUDY.glob(f'selections/seed_{a.seed}/{study.METHOD}/round_*/agents/*/turn_[0-9][0-9].json'):
        receipt=json.loads(response_path.read_text())
        provenance=receipt.get('provenance',{})
        if provenance.get('response_id'):
            round_index=int(response_path.parent.parent.parent.name.split('_')[-1])
            unique[provenance['response_id']]={'round':round_index,**provenance}
    new=[c for c in unique.values() if c['round']>=2]
    cost={'completed_provider_responses_since_switch':len(new),
        'input_tokens':sum(c.get('usage',{}).get('input_tokens',0) for c in new),
        'output_tokens':sum(c.get('usage',{}).get('output_tokens',0) for c in new),
        'total_tokens':sum(c.get('usage',{}).get('total_tokens',0) for c in new),
        'network_failure_billing':'unknown; incomplete responses excluded from reported token totals',
        'monetary_cost':'use actual provider billing; not inferred from official prices'}
    (out/'comparison_summary.json').write_text(json.dumps({'summary':summary,'latest_endpoints':endpoints,'common_budget':common,'cost':cost,'switch_budget':397,
        'evidence':'single-seed mixed-model trajectory; not a paired Astra/Sol model-only comparison'},ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    for m,curve in curves.items():
        name,color=styles[m]
        axes[0].plot([row['budget'] for row in curve],[row['validation_NRMSE'] for row in curve],marker='o',label=name,color=color,linewidth=2)
    axes[0].axvline(397,color='#777777',ls=':',linewidth=1.2)
    axes[0].set_xlabel('Measured training labels');axes[0].set_ylabel('Validation NRMSE (lower is better)')
    axes[0].set_xticks(study.BUDGETS[:a.rounds+1]);axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    axes[0].set_title('Sol 6.1 / xhigh starts after 397 labels')
    ordered=sorted(summary,key=lambda row:row['validation_NRMSE'])
    names=['Dual expert' if row['method']==study.METHOD else styles[row['method']][0].replace(' + ','\n+ ') for row in ordered]
    values=[row['validation_NRMSE'] for row in ordered]
    bars=axes[1].bar(names,values,color=[styles[row['method']][1] for row in ordered],width=.6)
    axes[1].bar_label(bars,fmt='%.4f',padding=4);axes[1].set_ylim(0,max(values)*1.18)
    axes[1].set_title(f'Same budget: {common} labels');axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    fig.suptitle(f'Dual expert continuation | seed {a.seed}',fontsize=15,fontweight='bold')
    fig.text(.07,.015,'Validation only. Historical prefix retained; no test truth evaluation.\nModel and reasoning effort changed at 397; later differences also include label budget and selected samples.',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=[0,.1,1,.93]);fig.savefig(out/'comparison.png',dpi=180,bbox_inches='tight');fig.savefig(out/'comparison.svg',bbox_inches='tight');plt.close(fig)
    labels={study.METHOD:'双专家（397后Sol 6.1/xhigh）','free_llm32_scientist_v2':'Free','cw16_llm16_scientist_v2':'LLM+CW'}
    budgets=study.BUDGETS[2:a.rounds+1]
    lines=['# 双专家续跑：同预算验证结果\n',f'seed {a.seed}，397标签后切换gpt-6.1-sol/xhigh；最新完成{latest}标签。无新test评价。\n',
        '| 方法 | '+' | '.join(f'{b}标签' for b in budgets)+' |','| --- | '+' | '.join('---:' for b in budgets)+' |']
    for m,curve in curves.items():
        d={row['budget']:row['validation_NRMSE'] for row in curve}
        lines.append('| '+labels[m]+' | '+' | '.join(f'{d[b]:.6f}' if b in d else '—' for b in budgets)+' |')
    lines+=[f'\n指标越低越好；对照复用历史同seed/split/训练规则fit audit。三方法共同预算上限为{common}；缺失结果记作—，不外推。\n','![比较](comparison.png)\n',
        '从397继承原有checkpoint、方案、标签和反馈。前两轮选样是Astra/high，之后是Sol 6.1/xhigh；不是同一397状态下的Astra/Sol分叉对照，不能单独归因模型更换。\n',
        '所有原始回答、格式纠正和失败网络请求保留。对格式/证据问题只提供纠正反馈，不由host改写选点或证据；样本与证据全部通过校验后才冻结和揭示新标签。首个Sol轮次ML和第二个Sol轮次planner各登记一次额外格式修复，来源与原始回答在研究revision的provenance文件中。\n',
        f"模型切换后的已完成provider回答{len(new)}次，共{cost['total_tokens']:,} tokens；包含已完成的无效格式回答及尚未完成批次的回答。网络中断未获得完整usage，费用不估算。\n",
        '[原始指标与来源hash](results.json) · [区间AULC和成本](comparison_summary.json)\n',
        '模型设置参考：[OpenAI GPT-6.1 Sol文档](https://developers.openai.com/api/docs/models/gpt-6.1-sol)。实际provider回复的served_model已核对；无模型自动回退。\n']
    (out/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
    # Matplotlib emits trailing spaces in SVG paths; keep regenerated diffs clean.
    svg = out/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({'report':str(out/'README.md'),'summary':summary,'cost':cost},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

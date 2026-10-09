#!/usr/bin/env python
"""Update owned status pages and commit only verified CW extension delivery."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
S = ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005'
P = ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_653'


def main():
    decision = json.loads((S/'decision.json').read_text())
    amended = (S/'WHITENING_STOP_AMENDMENT.json').exists()
    total, first, last = (60,16,44) if amended else (90,24,66)
    assert decision['status'] == 'completed' and decision['new_fits525_1005'] == total
    assert json.loads((S/'DELIVERY_AUDIT.json').read_text())['verified_fits'] == last
    assert json.loads((P/'DELIVERY_AUDIT.json').read_text())['verified_fits'] == first
    summary = 'All16 curves completed through1005;90 additional fits from525 (24 to653 plus66 to1005),150 historical control points reused. No paid LLM calls or new test reveals. Original333–525 gate remains unchanged; the extension is exploratory.'
    if amended:
        summary = 'Fourteen curves completed through1005; Whitening remains at525 for both seeds by user instruction.60 additional fits (16 to653 plus44 to1005);150 control points reused. Zero paid LLM calls or new test reveals. Original333–525 conclusions unchanged; extension exploratory.'
    for directory in (S,P):
        p = directory/'README.md'
        text = p.read_text()
        if '\n## Verified delivery\n' not in text:
            p.write_text(text+'\n## Verified delivery\n\n'+summary+' See the1005 RESULTS.md and DELIVERY_AUDIT.json for final effects and verification.\n')
    p = S/'README.md'
    p.write_text(p.read_text().replace('Status: queued after the active653 run.', 'Status: completed and delivery-audited.'))
    p = ROOT/'docs/research/CW_GEOMETRY_AND_LLM_FEEDBACK_PLAN.md'
    text = p.read_text()
    marker = '\n## Completed user extension through1005\n'
    if marker not in text:
        text += marker+'\n'+summary+'\n\nP1 extended matrix: completed. P2 offline: completed; live ablations: planned/unrun. New training and LLM calls stop here. See `studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/RESULTS.md` for effects, paired differences, cost and verification-only commands.\n'
    text = text.replace('Status: planned, waiting for653.', 'Status: completed through1005 and delivery-audited.')
    text = text.replace('Current extension status: running;', 'Current extension status: completed;')
    p.write_text(text)
    p = ROOT/'docs/NEXT_STAGE_DECISION.md'
    text = p.read_text()
    start = text.find('用户随后授权追加四个round：')
    if start >= 0:
        end = text.index('\n\n', start)
        effects = []
        for method in ('cw_lcmd','cw_maxdet','cw_ivr'):
            values = [x['relative_improvement_percent'] for x in decision['effects'] if x['method']==method]
            effects.append(f'{method} {sum(values)/len(values):+.2f}%')
        replacement = ('用户授权的1005扩展已完成：七臂×两seed到1005，Whitening按用户要求停在525；新增60次拟合，复用150个历史对照点。探索性validation AULC525–1005平均相对改善：' if amended else '用户授权的1005扩展已完成：八臂×两seed到1005；新增90次拟合，复用150个历史对照点。探索性validation AULC525–1005平均相对改善：')+ '；'.join(effects)+'。原333–525结论与确认门槛不变，已停止新增训练及LLM调用。[完整扩展结果](../studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/RESULTS.md)。'
        text = text[:start]+replacement+text[end:]
    p.write_text(text)
    (S/'pipeline_state.json').write_text(json.dumps({'status':'completed','budget':1005,
        'new_fits_from525':total,'paid_llm_calls':0,'new_test_reveals':0},indent=2)+'\n')
    paths = ['docs/research/CW_GEOMETRY_AND_LLM_FEEDBACK_PLAN.md','docs/NEXT_STAGE_DECISION.md',
             str(S.relative_to(ROOT)),str(P.relative_to(ROOT))]
    # Explicit paths protect unrelated live LLM work in this shared checkout.
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    changed = subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT)
    if changed.returncode == 1:
        subprocess.run(['git','commit','--only','-m','study: deliver verified CW matrix through 1005','--',*paths],cwd=ROOT,check=True)
    elif changed.returncode != 0:
        raise RuntimeError('cannot inspect staged delivery')
    print(summary,flush=True)


if __name__ == '__main__':
    main()

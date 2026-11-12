from pathlib import Path
import csv,html,json,datetime,statistics
S=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction/studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101')
O=S/'reports'
a=json.loads((O/'final_verification_20261112.json').read_text())
assert a['status']=='FINAL_OFFLINE_AUDIT_PASSED'
names={'historical_free_mixed_model':'历史 Free（混合模型）','nocw':'NoCW','numerical_hybrid32':'数值 Hybrid32','gradient_lcmd32':'Gradient-LCMD32'}
def esc(x):return html.escape(str(x))
def fmt(x):return f"{x['mean']:.6f} ± {x['seed_std_ddof1']:.6f}"
def table(head,rows):
 return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CW-visible Free32 五 seed 终态报告</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:48px auto;padding:0 24px;color:#182436;background:#fafbfd}h1{font-size:32px}h2{margin-top:36px;font-size:23px}table{border-collapse:collapse;width:100%;background:white;margin:18px 0;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dce3eb;text-align:left;font-variant-numeric:tabular-nums}th{background:#edf4f5}a{color:#087f8c}img{width:100%;height:auto}p{max-width:1100px}.lead{font-size:20px;color:#087f8c}.muted{color:#536174}</style><body>']
parts+=['<h1>CW-visible Free32 · 五 seed 终态报告</h1>',f'<p class="lead">180 / 180 批全部接受并完成训练；五个 seed 均完成 36 轮，L333 → L1485。</p>',f'<p>157 最后一批完成于上海时间 2026年11月12日11:17:39。终态离线核验：{esc(a["checked_shanghai"])}。模型 gpt-6.1-sol，reasoning effort xhigh；累计传输尝试 173511，成功响应轮次 {a["successful_response_turns"]}。</p>']
parts+=['<h2>终态核验</h2>',f'<p>五条冻结轨迹，每条新增1152个不重复的训练样本；180个完成标记、每批完整输入与候选池、请求及纠错历史、成功回执、选点冻结、标签揭示回执、模型与预测冻结绑定均通过核验。共核对 {a["hashed_artifact_files"]} 个文件哈希。全部开始的请求均有本地回执；仅保留精确授权的五份历史401回执，没有新增未决请求。</p>','<p>报告与核验未发起选点请求、未训练模型、未读取测试真值，未将监控指标回填选点输入或记忆。价格未经核实，保留 usage 与回执，不推算货币费用。NoCW 未修改、停止或发送信号。</p>']
with (O/'validation_curves.csv').open() as f: rows=list(csv.DictReader(f))
current=[r for r in rows if r['method']=='new_free_cw_sol61_xhigh']
final={int(r['seed']):float(r['validation_selection_score']) for r in current if int(r['budget'])==1485}
fullareas={r['seed']:next(m['raw_validation_AULC'] for m in json.loads((O/'milestone_L1485.json').read_text())['seeds'] if m['seed']==r['seed']) for r in a['terminal_seed_results']}
parts.append(table(['seed','完成轮数','最终 L','终点验证 NRMSE','全程归一化 AULC','累计传输尝试'],[[r['seed'],36,1485,f"{final[r['seed']]:.6f}",f"{fullareas[r['seed']]:.6f}",r['transmission_attempts']] for r in a['terminal_seed_results']]))
parts+=['<h2>五 seed 共同预算节点</h2>','<p>下表均为固定同五个 seed 的均值 ± 样本标准差（ddof=1）。NRMSE及AULC均越低越好。</p>']
parts.append(table(['预算 L','终点验证 NRMSE','L333起原始曲线归一化 AULC'],[[m['budget'],fmt(m['validation_selection_score']),fmt(m['raw_validation_AULC'])] for m in a['milestones']]))
parts+=['<h2>公平共同预算比较</h2>','<p>每个参考方法分别采用同五个 seed 均已完成的共同预算。Δ = 本次 CW-visible Free32 − 参考方法；负值表示本次误差较低。AULC使用L333起的全部原始、非单调曲线做梯形积分，再除以预算跨度；没有平滑、累计最优或单调包络。</p>','<h3>全程原始曲线 AULC</h3>']
parts.append(table(['参考方法','共同 L','本次 AULC','参考 AULC','逐seed配对 Δ（均值 ± 样本SD）'],[[names[c['comparator']],c['budget'],fmt(c['raw_validation_AULC']['new_free']),fmt(c['raw_validation_AULC']['reference']),fmt(c['raw_validation_AULC']['paired_delta_new_minus_reference'])] for c in a['fair_comparisons']]))
parts+=['<h3>共同预算终点误差</h3>']
parts.append(table(['参考方法','共同 L','本次验证 NRMSE','参考验证 NRMSE','逐seed配对 Δ（均值 ± 样本SD）'],[[names[c['comparator']],c['budget'],fmt(c['validation_selection_score']['new_free']),fmt(c['validation_selection_score']['reference']),fmt(c['validation_selection_score']['paired_delta_new_minus_reference'])] for c in a['fair_comparisons']]))
parts+=['<p>本次全程AULC高于四个参考方法；共同预算终点NRMSE低于历史Free、NoCW与Gradient-LCMD，高于数值Hybrid。终点改善与全程表现是不同结果，不能互相替代。</p>','<h3>各 seed 配对差值</h3>']
with (O/'paired_differences.csv').open() as f: paired=list(csv.DictReader(f))
for c in a['fair_comparisons']:
 comp,budget=c['comparator'],c['budget'];index={(int(r['seed']),r['metric']):r for r in paired if r['comparator']==comp and int(r['budget'])==budget}
 parts.append(f'<h4>{esc(names[comp])} · L{budget}</h4>')
 parts.append(table(['seed','本次 AULC','参考 AULC','AULC Δ','终点 NRMSE Δ'],[[seed,f"{float(index[seed,'raw_validation_AULC']['new_free']):.6f}",f"{float(index[seed,'raw_validation_AULC']['reference']):.6f}",f"{float(index[seed,'raw_validation_AULC']['paired_delta_new_minus_reference']):+.6f}",f"{float(index[seed,'validation_selection_score']['paired_delta_new_minus_reference']):+.6f}"] for seed in (157,887,2357,6101,12203)]))
parts+=['<h2>完整学习曲线</h2>','<p><a href="final_curves_20261112/overall_curves.pdf">下载 PDF</a> · <a href="final_curves_20261112/overall_curves.svg">下载 SVG</a></p>','<img src="final_curves_20261112/overall_curves.png" alt="五seed完整原始验证学习曲线及参考方法均值">','<h2>历史标记与解释范围</h2>',f'<p>{esc(a["historical_free_model_mixing"])}</p>','<p>历史Free保留混合模型标记。API随机性、运行日期与不可见后端差异仍存在；这些比较不单独证明CW的因果效果。NoCW的共同预算为L1005，不外推到L1485。</p>','<h2>可复核原始产物</h2>','<ul>']
for name,label in [('final_verification_20261112.json','终态独立核验与文件哈希'),('final_runtime_validation_20261112.json','最终授权链与进程、锁释放核验'),('final_audit_20261112.py','本地离线核验脚本'),('REPORT.md','自动报告'),('comparison_summary.json','全部共同预算比较及均值/样本标准差'),('paired_differences.csv','逐seed、逐预算、逐指标配对值'),('validation_curves.csv','全部原始非单调验证曲线'),('completion.csv','各方法各seed实际完成预算'),('costs.json','完整传输回执与usage汇总')]:parts.append(f'<li><a href="{name}">{esc(label)}</a></li>')
parts+=['</ul><p class="muted">聊天、冻结输入、成功/失败回执、模型、报告和本地切换证据全部保留；本次工作未提交或推送Git远程。</p></body></html>']
p=O/'FINAL_REPORT_20261112.html';assert not p.exists();p.write_text('\n'.join(parts))
print(json.dumps({'report':str(p),'milestones':len(a['milestones']),'fair_comparisons':len(a['fair_comparisons'])},ensure_ascii=False))

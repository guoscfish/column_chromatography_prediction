#!/usr/bin/env python3
"""Plot only saved validation metrics from completed fits; never select or train."""
import argparse,csv,hashlib,importlib.util,json,statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
SEEDS=(157,887,2357,6101,12203)
METHODS=(('new_free_cw_sol61_xhigh','当前 CW-visible Free32','#087F8C'),
 ('historical_free_mixed_model','历史 Free32（混合模型）','#9A67A5'),
 ('nocw','NoCW Free32','#DA8436'),
 ('numerical_hybrid32','数值 Hybrid32','#547DBB'),
 ('gradient_lcmd32','Gradient LCMD32','#608C52'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def draw(out):
 if out.exists():raise RuntimeError('Use a fresh snapshot directory')
 spec=importlib.util.spec_from_file_location('saved_validation',ROOT/'scripts/studies/report_free_cw_sol61_five_20261101.py')
 report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
 captured=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
 receipts={seed:[json.loads(p.read_text()) for p in sorted((report.STUDY/'progress').glob(f'completed_seed_{seed}_round_*.json'))] for seed in SEEDS}
 counts={seed:len(v) for seed,v in receipts.items()};common_round=min(counts.values());common=333+32*common_round
 sources={};rows=[];curves={};identity={}
 for seed in SEEDS:
  assert [r['round'] for r in receipts[seed]]==list(range(counts[seed]))
  for receipt in receipts[seed]:
   for name,digest in receipt['files'].items():
    p=report.STUDY/name;assert sha(p)==digest,name;sources[str(p.relative_to(ROOT))]=digest
  for method,_,_ in METHODS:
   points=[]
   for rnd in range((counts[seed] if method==METHODS[0][0] else common_round)+1):
    p=report.saved_point(method,seed,rnd)
    if p is None:raise RuntimeError(f'Missing shared-budget fit: {method}/{seed}/{rnd}')
    audit=json.loads((ROOT/p['source']).read_text());ident=(audit['initialization_hash'],audit['validation_ids_hash'])
    if seed not in identity:identity[seed]=ident
    assert identity[seed]==ident,(method,seed,rnd)
    value=p['validation_selection_score'];assert abs(value-audit['best_validation_combined_normalized_rmse'])<1e-12
    row={'method':method,'seed':seed,'budget':p['budget'],'validation_nrmse':value,'source':p['source'],'fit_audit_sha256':sha(ROOT/p['source'])}
    points.append(row);rows.append(row);sources[p['source']]=row['fit_audit_sha256']
   curves[method,seed]=points
 means=[];paired=[]
 for method,_,_ in METHODS:
  for rnd in range(common_round+1):
   values=[curves[method,seed][rnd]['validation_nrmse'] for seed in SEEDS]
   means.append({'method':method,'budget':333+32*rnd,'n':5,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)})
 for method,_,_ in METHODS[1:]:
  for seed in SEEDS:
   current=curves[METHODS[0][0],seed][:common_round+1];reference=curves[method,seed]
   def area(points):return sum((a['validation_nrmse']+b['validation_nrmse'])*.5*(b['budget']-a['budget']) for a,b in zip(points,points[1:]))/(common-333)
   paired.append({'comparator':method,'seed':seed,'budget':common,'current_nrmse':current[-1]['validation_nrmse'],'reference_nrmse':reference[-1]['validation_nrmse'],'paired_delta':current[-1]['validation_nrmse']-reference[-1]['validation_nrmse'],'current_raw_normalized_aulc':area(current),'reference_raw_normalized_aulc':area(reference)})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 available={f.name for f in font_manager.fontManager.ttflist};font=next((f for f in ('Arial Unicode MS','PingFang SC','Heiti TC') if f in available),'DejaVu Sans')
 plt.rcParams.update({'font.family':font,'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,2,figsize=(16,7.5));fig.patch.set_facecolor('#FAFBFD')
 fig.text(.065,.945,'CW-visible Free32 · 当前整体学习曲线',fontsize=24,weight='bold',color='#182436')
 fig.text(.065,.897,f"{captured[:16].replace('T',' ')} 上海时间  |  完成训练 {sum(counts.values())}/180 批  |  仅统计已冻结的训练结果",fontsize=12,color='#536174')
 for method,label,color in METHODS:
  points=[p for p in means if p['method']==method];xs=[p['budget'] for p in points];ys=[p['mean'] for p in points];primary=method==METHODS[0][0]
  axes[0].plot(xs,ys,label=label,color=color,lw=3 if primary else 1.9,ls='-' if primary else '--',marker='o',ms=3.4 if primary else 2.4,zorder=4 if primary else 3)
  if primary:axes[0].fill_between(xs,[p['mean']-p['sample_sd'] for p in points],[p['mean']+p['sample_sd'] for p in points],color=color,alpha=.10,lw=0)
 axes[0].set_title(f'五 seed 公平比较 · 共同预算至 L{common}',loc='left',fontsize=14,pad=15)
 axes[0].legend(frameon=False,fontsize=10,loc='upper right');axes[0].set_xlim(325,common+10)
 colors=('#087F8C','#A66AAB','#DC8B37','#557FBD','#6D9656')
 for seed,color in zip(SEEDS,colors):
  points=curves[METHODS[0][0],seed];xs=[p['budget'] for p in points];ys=[p['validation_nrmse'] for p in points]
  axes[1].plot(xs,ys,label=f"seed {seed} · {counts[seed]}/36轮 · L{xs[-1]}",color=color,lw=2,marker='o',ms=3)
  axes[1].scatter(xs[-1],ys[-1],s=42,color=color,edgecolor='white',linewidth=.8,zorder=5)
 axes[1].set_title('当前实验 · 五个 seed 的完整已完成轨迹',loc='left',fontsize=14,pad=15)
 axes[1].legend(frameon=False,fontsize=9.6,loc='upper right');axes[1].set_xlim(325,1495)
 for ax in axes:
  ax.set_facecolor('white');ax.grid(axis='y',alpha=.18);ax.set_xlabel('已标注训练样本数 L');ax.set_ylabel('验证集 NRMSE（越低越好）')
 axes[0].set_xticks([333,525,653,813,common]);axes[1].set_xticks([333,525,813,1005,1197,1389,1485])
 fig.text(.065,.105,'左图：所有方法固定相同五个 seed、相同预算；浅色带为当前实验均值 ±1 个样本标准差。右图：每个 seed 画到实际完成位置。',fontsize=10,color='#536174')
 fig.text(.065,.06,'保留原始非单调曲线，未平滑、未取历史最优。历史 Free 保留混合模型标记；Gradient LCMD 为既有五 seed 基线。\n正在选点或计算的批次不计入；未读取测试真值，未向选点输入提供这些指标。',fontsize=9.5,color='#536174',linespacing=1.6)
 fig.subplots_adjust(left=.065,right=.98,top=.81,bottom=.205,wspace=.23)
 out.mkdir(parents=True)
 for ext in ('png','pdf','svg'):fig.savefig(out/f'overall_curves.{ext}',dpi=190,facecolor=fig.get_facecolor())
 plt.close(fig);table(out/'validation_curves.csv',rows);table(out/'five_seed_means.csv',means);table(out/'paired_differences.csv',paired)
 snapshot={'captured_shanghai':captured,'completed_rounds':counts,'common_budget':common,'total_completed':sum(counts.values()),'source_hashes':sources,'new_api_calls':0,'new_fits':0,'test_truth_reads':0,'selector_feedback':False,'raw_curves':True}
 (out/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in snapshot.items() if k!='source_hashes'},ensure_ascii=False))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();draw(args.output.resolve())

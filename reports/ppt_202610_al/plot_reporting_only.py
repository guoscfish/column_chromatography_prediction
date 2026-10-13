#!/usr/bin/env python3
"""Frozen-result reporting only. No repository imports, models, labels or selectors.

Read pinned Git objects, audit and export PNG/SVG + CSV. Never rerun a study report
entry point: some historical report functions open test truth/checkpoints.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = '42ab12600a353bba4a063ef5d1b623ee2f7b976a'
BASE = 'studies/active_learning/'
FA = BASE + 'qgeognn_v2_row_cw_selector_factorial/'
EX = BASE + 'qgeognn_v2_row_cw_selector_extension_to_1005/'
BR = BASE + 'qgeognn_v2_same_state_branching/'
METHODS = ['Random32', 'LCMD32', 'Hybrid traditional', 'Gradient MaxDet', 'CW32 (LCMD)', 'CW-IVR']
NAMES = {'Random32':'随机采样32', 'LCMD32':'LCMD32（Raw梯度）',
         'Hybrid traditional':'传统混合策略', 'Gradient MaxDet':'梯度MaxDet',
         'CW32 (LCMD)':'CW32（LCMD）', 'CW-IVR':'CW-IVR'}
# PKU red with restrained red variants, blue-gray and warm gray; line patterns
# supplement colors in the six-method overview.
RED, BLUE, WARM, INK = '#8C0000', '#617A8B', '#9A9086', '#282828'
COLORS = dict(zip(METHODS, [WARM, BLUE, '#9C6471', '#435666', RED, '#BB4A3E']))
STYLES = dict(zip(METHODS, ['--', '-', '-.', ':', '-', '--']))
SEEDS, GRID = (157,6101), np.arange(333,1006,32)
sources, checks = [], []

def check(name, condition, detail=''):
    checks.append({'check':name, 'passed':bool(condition), 'detail':detail})
    if not condition:
        raise AssertionError(name + ': ' + str(detail))

def git_bytes(path):
    return subprocess.check_output(['git','show',f'{REV}:{path}'],cwd=ROOT)

def read(path, role='source'):
    b = git_bytes(path)
    local = ROOT/path
    sources.append({'path':path, 'role':role, 'git_commit':REV,
                    'sha256':hashlib.sha256(b).hexdigest(), 'bytes':len(b),
                    'working_copy_matches_main':local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()})
    dest = OUT/'frozen_inputs'/path
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(b)
    return b

def frame(path):
    return pd.read_csv(io.BytesIO(read(path)))

def aulc(x,y):
    return float(np.trapezoid(y,x)/(x[-1]-x[0]))

def csv(df,name):
    df.to_csv(OUT/name,index=False,float_format='%.17g')

def setup():
    candidates=['/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/Library/Fonts/Arial Unicode.ttf']
    font=next((p for p in candidates if Path(p).exists()),None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    else:
        plt.rcParams['font.family']=['Noto Sans CJK SC','Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':18,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B4B1AD',
        'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ppt_202610_al',
        'legend.frameon':False})

def title(fig, main, sub):
    fig.text(.065,.927,main,fontsize=25,fontweight='bold',color=RED,va='center')
    fig.text(.065,.868,sub,fontsize=12.5,color='#646464',va='center')
    fig.add_artist(Line2D([.065,.935],[.903,.903],transform=fig.transFigure,color=RED,lw=1.2))

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300)
    fig.savefig(OUT/(name+'.svg'),metadata={'Date':None})
    plt.close(fig)

def axis(ax,small=False):
    ax.grid(axis='y',color='#EEECE9',lw=.7)
    ax.set_axisbelow(True)
    ax.set_xlim(333,1013)
    ax.set_xticks([333,429,525,653,813,1005] if not small else [333,525,653,1005])
    for n in [429,525,653,1005]:
        ax.axvline(n,color='#DAD6D1',ls=(0,(3,4)),lw=.9,zorder=0)

def load_and_audit():
    snapshot=json.loads(read('reports/active_learning_latest/snapshot.json'))
    for p in ['reports/active_learning_latest/REPORT.md', BASE+'qgeognn_v2_row_master_report/MASTER_REPORT.md',
              FA+'RESULTS.md',EX+'RESULTS.md',BR+'FINAL_REPORT.md',
              'scripts/report_active_learning_snapshot.py','scripts/studies/build_qgeognn_v2_row_master_report.py',
              FA+'reuse_compatibility.json',FA+'REUSE_MANIFEST.json',FA+'protocol.json',
              EX+'REUSE_GEOMETRY_AUDIT.json',EX+'COMPLETION_RECHECK.json',EX+'DELIVERY_AUDIT.json',
              EX+'RUN_STATE.json',EX+'GLOBAL_FREEZE.json',BR+'postfreeze_artifact_manifest.json']:
        read(p,'report/code/audit')
    code=['models/qgeognn_v2.py','active_learning_v2/coverage.py','active_learning_v2/gradient_features.py',
          'active_learning_v2/gradient_transforms.py','active_learning_v2/lcmd.py','active_learning_v2/maxdet.py',
          'active_learning_v2/ivr.py','active_learning_v2/cw_factorial.py','active_learning_v2/cw_factorial_study.py',
          'active_learning_v2/cw_factorial_to1005.py','active_learning_v2/sequential_acquisition.py',
          'active_learning_v2/uncertainty.py','active_learning_v2/fused_features.py',
          'active_learning_v2/fused_maxdet_runner.py',
          'active_learning_v2/sequential_runner.py','active_learning_v2/benchmark_protocol.py',
          'active_learning_v2/sequential_protocol.py','active_learning_v2/same_state_branching.py',
          'evaluation/point.py','training/predictor.py']
    for p in code: read('src/qgeognn_al/'+p,'method code')
    check('snapshot_split',snapshot['evaluation']=='validation_only')
    selected=[]
    for seed in SEEDS:
        for m in METHODS:
            points=snapshot['curves'][str(seed)][m]
            check(f'complete_grid/{seed}/{m}',[p['labels'] for p in points]==GRID.tolist())
            for p in points:
                selected.append({'seed':seed,'Method':m,'labels':p['labels'],'Combined_NRMSE':p['nrmse'],
                                 'split':'validation','batch_size':32,'source':'snapshot.json'})
    overview=pd.DataFrame(selected)
    check('overview_finite',np.isfinite(overview.Combined_NRMSE).all())
    # Match each plotted metric to a retained completed fit audit. Initial fits
    # are shared; select only numerical audit entries, never a checkpoint.
    paths={'Random32':[('qgeognn_v2_row_sequential_b32','random')],
           'LCMD32':[('qgeognn_v2_row_sequential_b32','lcmd')],
           'Hybrid traditional':[('qgeognn_v2_row_sequential_b32','hybrid')],
           'Gradient MaxDet':[('qgeognn_v2_row_maxdet_b32','gradient_maxdet')],
           'CW32 (LCMD)':[(n,'center_width_lcmd') for n in ['qgeognn_v2_row_short_sequential_b32','qgeognn_v2_row_cw_lcmd_to_525','qgeognn_v2_row_cw_lcmd_to_653','qgeognn_v2_row_cw_lcmd_to_1005']],
           'CW-IVR':[(n,'cw_ivr') for n in ['qgeognn_v2_row_cw_selector_factorial','qgeognn_v2_row_cw_selector_extension_to_653','qgeognn_v2_row_cw_selector_extension_to_1005']]}
    audit_rows=[]
    for row in overview.itertuples(index=False):
        matches=[]
        for p,a in snapshot['fit_audits'].items():
            if a['train_rows']!=row.labels or f'/seed_{row.seed}/' not in p: continue
            allowed=paths[row.Method] if row.labels!=333 else paths['Random32']
            if any(p.startswith(BASE+n+'/') and f'/{method}/' in p for n,method in allowed):
                matches.append((p,a))
        check(f'fit_receipt/{row.seed}/{row.Method}/{row.labels}',bool(matches))
        for p,a in matches:
            check('fit_metric/'+p,abs(a['best_validation_combined_normalized_rmse']-row.Combined_NRMSE)<1e-12)
            check('no_test_fit/'+p,a['test_labels_used_for_fit_or_checkpoint_selection']==0)
            audit_rows.append({'seed':row.seed,'Method':row.Method,'labels':row.labels,'source_path':p,
                               **a,'retained_raw_sha256':snapshot['source_hashes'][p]})
    fit=pd.DataFrame(audit_rows)
    for seed,g in fit.groupby('seed'):
        check(f'validation_identity/{seed}',g.validation_ids_hash.nunique()==1)
        check(f'model_identity/{seed}',set(g.model_variant)=={'qgeognn_v2'} and set(g.parameter_count)=={458952})
        check(f'validation_count/{seed}',set(g.validation_rows)=={416})
        check(f'initialization_seed/{seed}',set(g.initialization_seed)=={int(seed)+2_000_003})
    # Hash-only audit of existing, explicitly listed fit receipts is safe. No
    # metrics are collected from local runtime; Git snapshot remains the input.
    raw_checks=[]
    for p in sorted(set(fit.source_path)):
        local=ROOT/p
        ok=hashlib.sha256(local.read_bytes()).hexdigest()==snapshot['source_hashes'][p] if local.is_file() else None
        raw_checks.append({'path':p,'exists':local.is_file(),'matches_snapshot_hash':ok})
        if ok is not None: check('retained_fit_hash/'+p,ok)
    csv(pd.DataFrame(raw_checks),'retained_fit_hash_audit.csv')
    csv(fit,'overview_fit_receipts.csv')
    ext=frame(EX+'validation_curves.csv')
    ext_summary=frame(EX+'validation_summary.csv')
    factorial=frame(FA+'validation_curves.csv')
    factorial_summary=frame(FA+'validation_summary.csv')
    protocol=json.loads(git_bytes(FA+'protocol.json'))
    check('selector_normalization_contract',protocol['normalization']=={'lcmd':'none','maxdet':'current_L_RMS','ivr':'fixed_L_plus_U_RMS'})
    for seed in SEEDS:
        config=protocol['contexts'][str(seed)]['training_config']
        check(f'frozen_training_budget/{seed}',config['maximum_epochs']==1000 and config['patience']==100 and config['batch_size']==2048 and config['learning_rate']==.001 and config['head_changed'] is False)
    mapping={'raw_lcmd':'LCMD32','cw_lcmd':'CW32 (LCMD)','raw_maxdet':'Gradient MaxDet','cw_ivr':'CW-IVR','random':'Random32'}
    for (seed,arm),g in ext.groupby(['seed','arm']):
        g=g.sort_values('budget')
        if arm=='whitening_lcmd':
            check(f'whitening_stopped/{seed}',g.budget.tolist()==GRID[GRID<=525].tolist())
            continue
        check(f'extension_grid/{seed}/{arm}',g.budget.tolist()==GRID.tolist())
        sr=ext_summary[(ext_summary.seed==seed)&(ext_summary.arm==arm)].iloc[0]
        for start,key in [(333,'AULC333_1005'),(525,'AULC525_1005'),(653,'AULC653_1005')]:
            z=g[g.budget>=start]
            check(f'extension_aulc/{seed}/{arm}/{start}',abs(aulc(z.budget.to_numpy(),z.combined_normalized_RMSE.to_numpy())-sr[key])<1e-12)
        check(f'endpoint/{seed}/{arm}',abs(g.iloc[-1].combined_normalized_RMSE-sr.NRMSE1005)<1e-12)
        f=factorial[(factorial.seed==seed)&(factorial.arm==arm)].sort_values('budget')
        check(f'factorial_prefix/{seed}/{arm}',np.allclose(g[g.budget<=525].combined_normalized_RMSE,f.combined_normalized_RMSE,rtol=0,atol=1e-12))
        fs=factorial_summary[(factorial_summary.seed==seed)&(factorial_summary.arm==arm)].iloc[0]
        check(f'factorial_aulc/{seed}/{arm}',abs(aulc(f.budget.to_numpy(),f.combined_normalized_RMSE.to_numpy())-fs.AULC333_525)<1e-12)
        if arm in mapping:
            o=overview[(overview.seed==seed)&(overview.Method==mapping[arm])].sort_values('labels')
            check(f'snapshot_csv_match/{seed}/{arm}',np.allclose(o.Combined_NRMSE,g.combined_normalized_RMSE,rtol=0,atol=1e-12))
    delivery=json.loads(git_bytes(EX+'DELIVERY_AUDIT.json'))
    check('extension_delivery_hash',delivery['validation_curves_sha256']==hashlib.sha256(git_bytes(EX+'validation_curves.csv')).hexdigest())
    branches=frame(BR+'results/branch_results.csv')
    winners=frame(BR+'results/winner_matrix.csv')
    manifest=json.loads(git_bytes(BR+'postfreeze_artifact_manifest.json'))['files']
    for p in [BR+'results/branch_results.csv',BR+'results/winner_matrix.csv']:
        check('branch_manifest/'+p,manifest[p]==hashlib.sha256(git_bytes(p)).hexdigest())
    check('branch_count',len(branches)==24 and len(winners)==8)
    for key,g in branches.groupby(['seed','source_trajectory','anchor_budget']):
        check('branch_three_strategies/'+str(key),set(g.strategy)=={'gradient_lcmd','kernel_ivr','gradient_maxdet'} and len(g)==3)
        check('shared_anchor/'+str(key),g.NRMSE_anchor.nunique()==1)
        recompute=(g.NRMSE_anchor+2*g.NRMSE_plus32+g.NRMSE_plus64)/4
        check('short_aulc/'+str(key),np.allclose(recompute,g.short_AULC,rtol=0,atol=1e-12))
        w=winners[(winners.seed==key[0])&(winners.source_trajectory==key[1])&(winners.anchor_budget==key[2])].iloc[0]
        check('branch_winner/'+str(key),w.winner_short_AULC==g.loc[g.short_AULC.idxmin(),'strategy'])
    check('winner_counts',winners.winner_short_AULC.value_counts().to_dict()=={'kernel_ivr':4,'gradient_lcmd':4})
    return overview,ext,ext_summary,branches,winners,snapshot

def summaries(overview,ext,ext_summary,branches,winners):
    rows=[]
    for m in METHODS:
        for seed in SEEDS:
            g=overview[(overview.Method==m)&(overview.seed==seed)].sort_values('labels')
            rows.append({'Method':m,'seed':str(seed),'AULC333_1005':aulc(g.labels.to_numpy(),g.Combined_NRMSE.to_numpy()),
                         'NRMSE1005':g.iloc[-1].Combined_NRMSE,'split':'validation','batch_size':32})
        avg={k:np.mean([r[k] for r in rows[-2:]]) for k in ['AULC333_1005','NRMSE1005']}
        rows.append({'Method':m,'seed':'mean',**avg,'split':'validation','batch_size':32})
    summary=pd.DataFrame(rows)
    csv(overview,'fig10_validation_curves.csv');csv(summary,'fig10_summary.csv')
    wide=summary.pivot(index='Method',columns='seed',values=['AULC333_1005','NRMSE1005']).reindex(METHODS)
    wide.columns=[f'{a}_seed{b}' if b!='mean' else f'{a}_mean' for a,b in wide.columns]
    csv(wide.reset_index(),'fig10_summary_wide.csv')
    six=ext[ext.arm.str.match(r'(raw|cw)_(lcmd|ivr|maxdet)$')].copy()
    six['split']='validation';six['batch_size']=32
    csv(six,'fig11_validation_curves.csv')
    s=ext_summary[ext_summary.arm.isin(six.arm.unique())].copy()
    mean=s.groupby('arm',as_index=False)[['AULC333_1005','AULC525_1005','AULC653_1005','NRMSE1005']].mean()
    mean['seed']='mean'; s['seed']=s.seed.astype(str)
    s=pd.concat([s,mean],ignore_index=True);s['split']='validation'
    csv(s,'fig11_summary.csv')
    paired=[]
    for selector in ['lcmd','ivr','maxdet']:
        for seed in ['157','6101','mean']:
            raw=s[(s.seed==seed)&(s.arm=='raw_'+selector)].iloc[0].AULC333_1005
            cw=s[(s.seed==seed)&(s.arm=='cw_'+selector)].iloc[0].AULC333_1005
            paired.append({'selector':selector,'seed':seed,'Raw_AULC333_1005':raw,'CW_AULC333_1005':cw,
                           'CW_minus_Raw':cw-raw,'CW_relative_improvement':1-cw/raw,'split':'validation'})
    csv(pd.DataFrame(paired),'fig11_paired_effects.csv')
    mapping_rows=[]
    for selector in ['lcmd','ivr','maxdet']:
        for rep in ['raw','cw']:
            arm=rep+'_'+selector
            mapping_rows.append({'display_method':rep.upper()+'-'+selector.upper(),'csv_arm':arm,
                'representation':'scaled_q50_output_gradient' if rep=='raw' else 'scaled_center_width_output_gradient',
                'selector':selector,'is_latent':False,'changes_model_head':False,'dimension':512,
                'selector_normalization':{'lcmd':'none','ivr':'RMS_over_fixed_L_plus_U','maxdet':'RMS_over_current_L'}[selector],
                'historical_source':{'raw_lcmd':'qgeognn_v2_row_sequential_b32/lcmd','cw_lcmd':'qgeognn_v2_row_short_sequential_b32 + CW LCMD extensions',
                    'raw_ivr':'qgeognn_v2_row_kernel_ivr_b32','raw_maxdet':'qgeognn_v2_row_maxdet_b32/gradient_maxdet',
                    'cw_ivr':'CW factorial + selector extensions','cw_maxdet':'CW factorial + selector extensions'}[arm]})
    csv(pd.DataFrame(mapping_rows),'method_mapping_audit.csv')
    status=[{'experiment':m,'status':'completed','seeds':'157;6101','budget':'333–1005','split':'validation',
             'evidence':'frozen historical curve; current two-seed comparison is descriptive'} for m in METHODS]
    status += [{'experiment':'Raw/CW selector extension','status':'completed: 14 curves; 60 extra fits after525','seeds':'157;6101',
                'budget':'333–1005','split':'validation','evidence':'post-development exploration; original 36 factorial fits preserved'},
               {'experiment':'Whitening LCMD','status':'stopped by user at525','seeds':'157;6101','budget':'333–525','split':'validation','evidence':'excluded from complete-range plots'},
               {'experiment':'Same-state branching','status':'completed: 8 states; 24 branches; 48 historical fits','seeds':'157;6101',
                'budget':'429/653 anchors; +64 rollout','split':'test_historical_postfreeze','evidence':'development/mechanism; no controller trained'}]
    csv(pd.DataFrame(status),'experiment_status.csv')
    branches=branches.copy();branches['split']='test_historical_postfreeze';branches['rollout_labels']=64;branches['batch_size']=32
    csv(branches,'fig12_branch_summary.csv');csv(branches[(branches.seed==157)&(branches.anchor_budget==429)],'fig12_reversal_values.csv')
    csv(winners.assign(split='test_historical_postfreeze'),'fig12_winner_audit.csv')
    return summary,s,wide

def overview_plot(data,summary,wide):
    fig=plt.figure(figsize=(16,9))
    title(fig,'传统主动学习：随机基线落后，策略排序随评价口径变化',
          '验证集 Combined NRMSE  ·  seed 157 / 6101  ·  B=32  ·  完整区间 333–1005  ·  越低越好')
    ax=fig.add_axes([.085,.22,.59,.57]);axis(ax)
    for m in METHODS:
        ys=[]
        for seed in SEEDS:
            g=data[(data.Method==m)&(data.seed==seed)].sort_values('labels');ys.append(g.Combined_NRMSE.to_numpy())
            ax.plot(GRID,ys[-1],color=COLORS[m],alpha=.23,lw=1,ls=STYLES[m],zorder=2)
        ax.plot(GRID,np.mean(ys,axis=0),color=COLORS[m],lw=2.8,ls=STYLES[m],label=NAMES[m],zorder=3)
    ax.set_ylim(.29,1.065)
    ax.set_xlabel('当前已标注训练样本数',labelpad=13)
    ax.set_ylabel('验证集 Combined NRMSE（越低越好）',labelpad=13)
    ax.legend(loc='upper right',fontsize=11.5,ncol=2)
    means=summary[summary.seed=='mean'].set_index('Method')
    fig.text(.72,.772,'两 seed 算术均值',color=RED,fontsize=17,fontweight='bold')
    for i,(key,head) in enumerate([('AULC333_1005','全程 AULC 排序'),('NRMSE1005','1005 端点排序')]):
        top=.702-i*.267
        fig.text(.72,top,head+' ↓',fontsize=13,fontweight='bold')
        ranked=means.sort_values(key)
        for j,(m,r) in enumerate(ranked.iterrows()):
            fig.text(.72,top-.035-j*.030,f'{j+1}  {NAMES[m]}',fontsize=10.8,color=COLORS[m])
            fig.text(.94,top-.035-j*.030,f'{r[key]:.4f}',fontsize=10.8,ha='right',color=COLORS[m])
    fig.text(.085,.093,'粗线：两 seed 均值；浅色细线：各 seed 原始轨迹。细线不代表统计置信区间。',fontsize=12,color='#666666')
    fig.text(.085,.052,'淡虚线：429 / 525 / 653 / 1005。仅汇报冻结结果；CW 延伸为开发后探索，均值差异不代表统计显著性。',fontsize=10.8,color='#777777')
    save(fig,'fig10_overview')
    # A separate slide-ready table, with each metric's two seeds and mean.
    fig=plt.figure(figsize=(16,9))
    title(fig,'传统主动学习：完整区间 AULC 与最终端点',
          '验证集  ·  AULC = 梯形积分 / (1005−333)  ·  seed 157 / 6101  ·  算术均值  ·  越低越好')
    ax=fig.add_axes([.055,.23,.89,.55]);ax.axis('off')
    cell=[]
    for m in METHODS:
        row=[m]
        for metric in ['AULC333_1005','NRMSE1005']:
            for seed in ['157','6101','mean']:
                row.append(f'{summary[(summary.Method==m)&(summary.seed==seed)].iloc[0][metric]:.6f}')
        cell.append(row)
    headers=['Method','AULC\nseed 157','AULC\nseed 6101','AULC\n均值','NRMSE@1005\nseed 157','NRMSE@1005\nseed 6101','NRMSE@1005\n均值']
    table=ax.table(cellText=cell,colLabels=headers,cellLoc='center',colWidths=[.245]+[.126]*6,bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(12)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if r==0:cell.set_facecolor(RED);cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#F4F1EE' if r%2 else '#FCFBFA')
            if c in (3,6):cell.set_text_props(color=RED,fontweight='bold')
    fig.text(.065,.13,'AULC 衡量全程标签效率；最终端点衡量 1005 标签时的误差。两者的主动策略排序并不一致。',fontsize=15,color=RED)
    fig.text(.065,.075,'数据：最新 validation 冻结快照；不引用旧 master test 总表。两 seed 是描述性比较，非独立确认。',fontsize=12,color='#777777')
    save(fig,'fig10_summary_table')

def ablation_plot(data,summary):
    fig=plt.figure(figsize=(16,9))
    title(fig,'输出梯度表示的收益依赖具体选点准则',
          'Raw / CW × LCMD / IVR / MaxDet  ·  验证集  ·  seed 157 / 6101  ·  B=32  ·  越低越好')
    for i,selector in enumerate(['lcmd','ivr','maxdet']):
        ax=fig.add_axes([.077+i*.306,.25,.266,.51]);axis(ax,True)
        for rep,color in [('raw',BLUE),('cw',RED)]:
            ys=[]
            for seed in SEEDS:
                g=data[(data.seed==seed)&(data.arm==rep+'_'+selector)].sort_values('budget')
                ys.append(g.combined_normalized_RMSE.to_numpy())
                ax.plot(GRID,ys[-1],color=color,lw=.9,alpha=.24)
            ax.plot(GRID,np.mean(ys,axis=0),color=color,lw=2.8,label='Raw 原输出梯度' if rep=='raw' else 'CW 中心/宽度梯度')
        ax.axvline(525,color=WARM,ls='--',lw=1.2)
        ax.text(.15,1.03,'冻结开发',transform=ax.transAxes,fontsize=10,color=WARM)
        ax.text(.56,1.03,'探索性延伸',transform=ax.transAxes,fontsize=10,color=WARM)
        ax.set_title(selector.upper() if selector!='maxdet' else 'MaxDet',pad=36,color=INK)
        ax.set_ylim(.28,1.065);ax.set_xlabel('已标注训练样本数',fontsize=13)
        if i==0:ax.set_ylabel('验证集 Combined NRMSE ↓',fontsize=14)
        else:ax.set_yticklabels([])
    fig.legend(handles=[Line2D([0],[0],color=BLUE,lw=3,label='Raw 原输出梯度'),Line2D([0],[0],color=RED,lw=3,label='CW 中心/宽度梯度')],loc='lower center',bbox_to_anchor=(.5,.135),ncol=2,fontsize=13)
    fig.text(.065,.105,'粗线：两 seed 均值；浅色细线：各 seed 轨迹（非置信区间）。Raw 不等于 Latent；CW 不改变模型输出头。',fontsize=11.3,color='#666666')
    fig.text(.065,.058,'333–525 factorial 为冻结开发实验；525–1005 为后续探索性延伸。两个阶段不能合并为独立确认实验。',fontsize=11.3,color='#666666')
    save(fig,'fig11_factorial_curves')
    fig=plt.figure(figsize=(16,9))
    title(fig,'输出梯度表示的收益依赖具体选点准则',
          'Raw / CW 配对 AULC  ·  验证集 333–1005  ·  两 seed 算术均值  ·  越低越好')
    ax=fig.add_axes([.12,.26,.76,.50]);ax.grid(axis='y',color='#EEECE9');ax.set_axisbelow(True)
    positions=np.arange(3); width=.27
    for rep,color,offset,label in [('raw',BLUE,-width/2,'Raw 原输出梯度'),('cw',RED,width/2,'CW 中心/宽度梯度')]:
        values=[summary[(summary.seed=='mean')&(summary.arm==rep+'_'+sel)].iloc[0].AULC333_1005 for sel in ['lcmd','ivr','maxdet']]
        bars=ax.bar(positions+offset,values,width,color=color,label=label)
        for bar,val in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,val+.012,f'{val:.6f}',ha='center',fontsize=15,color=color,fontweight='bold')
        for j,sel in enumerate(['lcmd','ivr','maxdet']):
            for k,seed in enumerate(['157','6101']):
                y=summary[(summary.seed==seed)&(summary.arm==rep+'_'+sel)].iloc[0].AULC333_1005
                ax.scatter(j+offset+(k-.5)*.055,y,s=33,facecolor='white',edgecolor='#343434',marker='o' if k==0 else 's',zorder=4,linewidth=.9)
    ax.set_xticks(positions,['LCMD','IVR','MaxDet']);ax.set_ylim(0,.72)
    ax.set_ylabel('验证集 AULC333–1005（越低越好）',labelpad=14)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.12),ncol=2,fontsize=13)
    fig.text(.12,.174,'白色圆点 / 方点：seed 157 / 6101。柱从零开始；没有统计置信区间。',fontsize=12,color='#666666')
    fig.text(.12,.115,'CW 相对 Raw：LCMD 与 IVR 的平均 AULC 降低；MaxDet 的平均 AULC 升高。',fontsize=15,color=RED)
    fig.text(.12,.065,'333–525 为冻结开发实验，525–1005 为后续探索性延伸；两阶段不能合并为独立确认实验。',fontsize=11.5,color='#777777')
    save(fig,'fig11_paired_aulc')

def box(ax,x,y,w,h,text,color=RED,fill='#FBF5F3',size=14):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',edgecolor=color,facecolor=fill,lw=1.15))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK)

def arrow(ax,start,end,color=WARM):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,lw=1.05,color=color))

def branching_plot(branches):
    fig=plt.figure(figsize=(16,9))
    title(fig,'相同标签数，不同历史状态：策略优胜关系发生反转',
          'Same-state branching  ·  seed 157 / budget 429 示例  ·  历史冻结后 test short-AULC  ·  越低越好')
    ax=fig.add_axes([.045,.20,.91,.59]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.01,.96,'复用历史起始状态',fontsize=16,color=RED,fontweight='bold')
    ax.text(.365,.96,'同一状态分出三条策略',fontsize=16,color=RED,fontweight='bold')
    ax.text(.71,.96,'+64 区间 short-AULC ↓',fontsize=16,color=RED,fontweight='bold')
    names={'gradient_lcmd':'LCMD','kernel_ivr':'IVR','gradient_maxdet':'MaxDet'}
    for i,(source,state) in enumerate([('gradient_lcmd','A'),('kernel_ivr','B')]):
        mid=.70-i*.43
        box(ax,.01,mid-.083,.142,.166,f'历史 {names[source]}\n轨迹',BLUE,'#F3F6F8',14)
        arrow(ax,(.159,mid),(.204,mid))
        box(ax,.212,mid-.083,.135,.166,f'状态 {state}\n429 labels',RED,'#FBF5F3',15)
        ax.text(.28,mid-.127,'相同 L、U、checkpoint',ha='center',fontsize=9.5,color='#666666')
        group=branches[(branches.seed==157)&(branches.anchor_budget==429)&(branches.source_trajectory==source)].set_index('strategy')
        win=group.short_AULC.idxmin()
        for j,strategy in enumerate(['gradient_lcmd','kernel_ivr','gradient_maxdet']):
            cy=mid+.118-j*.118
            arrow(ax,(.355,mid),(.394,cy))
            box(ax,.403,cy-.042,.187,.082,f'{names[strategy]}   429 → 461 → 493',BLUE,'white',11)
            arrow(ax,(.6,cy),(.692,cy))
            val=group.loc[strategy,'short_AULC']
            ax.text(.71,cy,f'{names[strategy]:7s}  {val:.6f}',fontsize=16,color=RED if strategy==win else '#737373',va='center',fontweight='bold' if strategy==win else 'normal')
            if strategy==win:ax.text(.955,cy,'优胜',fontsize=12,color=RED,va='center',ha='right')
        ax.text(.495,mid-.218,'两轮，每轮 +32 标签',ha='center',fontsize=10.5,color='#666666')
    fig.text(.065,.162,'状态 A：IVR 优于 LCMD；状态 B：LCMD 优于 IVR。仅在同一个起始状态内比较三条分支。',fontsize=14,color=RED)
    fig.text(.065,.115,'策略偏好可能依赖当前 state/history，而非固定标签数量；当前数据不足以训练可靠 controller。',fontsize=16,color=INK)
    fig.text(.065,.071,'8 个起始状态 / 24 条分支 / 两种策略各胜出 4 次（LCMD / IVR；MaxDet 0 次）。',fontsize=10.5,color='#777777')
    fig.text(.065,.038,'评价 split：历史冻结后统一揭示的 test；本次只读取既有汇总。short-AULC = (e₀ + 2e₃₂ + e₆₄) / 4。开发机制证据。',fontsize=10.5,color='#777777')
    save(fig,'fig12_same_state_branching')

def method_diagram():
    fig=plt.figure(figsize=(16,9))
    title(fig,'样本表示与选点准则：两个可以拆开的层次',
          '连接箭头表示接口关系；不代表所有组合都已完成多轮实验验证')
    ax=fig.add_axes([.09,.23,.82,.52]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(0,.86,'第一层\n样本表示',fontsize=17,color=RED,ha='left',va='center')
    ax.text(0,.15,'第二层\n选点方法',fontsize=17,color=RED,ha='left',va='center')
    xs=[.24,.52,.80]
    for x,t in zip(xs,['Latent\n模型隐表示','Raw Gradient\n原输出梯度','CW Gradient\n中心/宽度梯度']):box(ax,x-.10,.72,.20,.26,t,RED,'#FBF5F3',17)
    for x,t in zip(xs,['LCMD','IVR','MaxDet']):box(ax,x-.10,.02,.20,.26,t,BLUE,'#F3F6F8',18)
    for x in xs:
        for y in xs:arrow(ax,(x,.71),(y,.29),'#C7C0B9')
    fig.text(.09,.13,'同一种表示可以接入不同 acquisition objective。Raw / CW × 三准则已有开发轨迹；Latent 的完整矩阵未验证。',fontsize=13,color='#666666')
    fig.text(.09,.075,'CW 是提取梯度时的坐标变换；保持 QGeoGNN-V2 架构与输出头相同。IVR 是核代理方差缩减。',fontsize=13,color='#666666')
    save(fig,'methods_two_layer')

def main():
    global REV
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-rev',default=REV);args=p.parse_args()
    REV=subprocess.check_output(['git','rev-parse',args.source_rev],cwd=ROOT,text=True).strip()
    setup()
    data,ext,ext_summary,branches,winners,snapshot=load_and_audit()
    summary,s,wide=summaries(data,ext,ext_summary,branches,winners)
    overview_plot(data,summary,wide);ablation_plot(ext,s);branching_plot(branches);method_diagram()
    csv(pd.DataFrame(sources),'input_sources.csv');csv(pd.DataFrame(checks),'numerical_audit.csv')
    report={'audited_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'git_main_commit':REV,'snapshot_captured_at':snapshot['captured_at'],
        'overview_points':len(data),'ablation_points':264,'branch_states':len(winners),'branches':len(branches),
        'passed_checks':len(checks),'failed_checks':0,'training_runs':0,'llm_calls':0,'acquisition_runs':0,'new_test_label_accesses':0,
        'read_policy':'pinned Git report objects + hash-only existing fit receipts; no labels/checkpoints/dataset reads',
        'limitations':['two development seeds; no confidence intervals','original factorial and exploratory extension are not independent confirmation',
                      'branch short-AULC is historical postfreeze test, not validation',
                      'retained extension completion recheck disclosed a blocked full audit rerun after unrelated shared-workspace code edits; stored hashes verified here']}
    (OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    outputs=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='OUTPUT_SHA256.csv']
    csv(pd.DataFrame(outputs),'OUTPUT_SHA256.csv')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

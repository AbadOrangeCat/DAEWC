from pathlib import Path
import sys,json,csv,numpy as np
import argparse
project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project))
p=argparse.ArgumentParser()
p.add_argument('--artifacts',type=Path,default=project/'revision_artifacts')
p.add_argument('--manuscript',type=Path,default=project/'manuscript')
args=p.parse_args()
from daewc.summarize import summarize,interval,NAMES
from daewc.reporting import source_change_macros
root=args.artifacts;out=args.manuscript;tables=out/'tables';tables.mkdir(exist_ok=True)
for name in ['local','budget_cv','random','bert_base','low_footprint','low_footprint_cv']:
 if (root/name/'runs').exists():summarize(root/name,root/'sequential' if name=='local' else None,root/'tables'/name)
d=json.loads((root/'tables/local/summary.json').read_text());g=d['groups'];domains=['health','public_affairs','entertainment'];dn={'source':'Source','health':'Health','public_affairs':'Public affairs','entertainment':'Entertainment'}
methods=['head','full','full_ewc','adapter','lora','lwf','daewc']
def esc(s):return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def f(v):return f'{v:.2f}'
def ci(r,p='target'):return f"{r[p+'_mean']:.2f} $\\pm$ {r[p+'_ci95_halfwidth']:.2f}" if r[p+'_ci95_halfwidth'] is not None else f(r[p+'_mean'])
def table(caption,label,columns,head,rows,notes='',long=False):
 begin=r'\begin{longtable}' if long else r'\begin{tabular}'
 end=r'\end{longtable}' if long else r'\end{tabular}'
 body=' & '.join(head)+r' \\ \midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+ '\n'+r'\bottomrule'
 if long:return '\\begingroup\\small\n'+begin+'{'+columns+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\\toprule'+'\n'+body+'\n'+end+'\n'+r'\endgroup'+'\n'+notes+'\n'
 return r'\begin{table}[htbp]\centering\small'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+begin+'{'+columns+'}\n'+r'\toprule'+'\n'+body+'\n'+end+'\n'+(r'\par\smallskip\footnotesize '+notes if notes else '')+'\n'+r'\end{table}'+'\n'
m=json.loads((root/'data/manifest.json').read_text())
rows=[]
for dom in ['source']+domains:
 c=m['counts'][dom];rows.append([dn[dom]]+['%s / %s'%(c[s]['0'],c[s]['1']) for s in ['train','dev','test']]+[len(m['matched_test_ids'][dom])])
(tables/'datasets.tex').write_text(table('Primary partitions after duplicate and label exclusions. Each split reports label 0 / label 1 counts.','tab:datasets','lrrrr',['Collection','Training','Development','Test','Matched test'],rows,'Development data from the targets never enter adaptation or model selection. The source uses its development labels for source checkpoint and threshold selection.'))
settings=[['Encoder (compact / base)','2 / 12 blocks; width 128 / 768; heads 2 / 12'],['Feed-forward width; maximum length','512 / 3072; 128 tokens'],['Source optimizer','AdamW; learning rate $5\\times10^{-5}$'],['Source schedule and stopping','Constant rate; at most 5 epochs; patience 2'],['Source weight decay; selection','0.01; development loss improvement $>10^{-6}$'],['Target optimizer and schedule','AdamW; constant rate; 80 updates; no early stopping'],['Target shared full-update rate','$2\\times10^{-5}$'],['Target shared calibration rate','$10^{-4}$'],['Target adapters, gates, LoRA, and heads','$10^{-3}$'],['Optimizer moments and epsilon','$(0.9,0.999)$; $10^{-8}$'],['Target weight decay','0; proximity and EWC are explicit'],['Training / evaluation batch size','32 / 128'],['Gradient-norm clipping','1.0'],['Adapter bottleneck; dropout','16 (smaller version: 2); 0.1'],['Domain-vector dimension; gate','16 (smaller version: 4); $2\\sigma(We+b)$'],['LoRA rank; scaling factor','8; 2 (LoRA alpha = 16)'],['Backbone/head dropout; LoRA branch','0.1 / 0.1; 0'],['EWC $\\lambda$; proximity $\\alpha$','100; 0.1'],['Fisher examples $N_F$; damping $\\epsilon$','256 source training examples; $10^{-8}$'],['LwF temperature; loss weight','2; 1'],['Target threshold; source threshold','0.5; source-development grid 0.05 to 0.95'],['Target cross-validation (separate)','2 folds; rate multipliers 0.25, 1, 4'],['Seeds; shots per class','42, 43, 44; 10, 20, 80, 160']]
(tables/'settings.tex').write_text(table('Consolidated settings. The primary configuration is fixed before target evaluation.','tab:settings','p{.39\\linewidth}p{.55\\linewidth}',['Setting','Value'],settings,'The same settings apply to standard BERT and random-initialization controls, except for encoder dimensions and initialization. The sensitivity grid separately varies calibration rate and Fisher weight.'))
rows=[]
for dom in domains:
 for method in methods:
  r=next(x for x in g if x['domain']==dom and x['shots']==80 and x['method']==method)
  rows.append([dn[dom] if method=='head' else '',NAMES[method],ci(r),ci(r,'source_change'),ci(r,'matched')])
(tables/'main80.tex').write_text(table('Fixed-configuration results at 80 labelled examples per class. Values are means $\\pm$ 95\\% seed intervals.','tab:main80','llrrr',['Target','Method','Target $F_1$ (\\%)','$\\Delta$Src (pp)','Matched $F_1$ (\\%)'],rows,'The public-affairs matched test has 20 items. Its seed interval does not represent uncertainty from sampling a new set of headlines. Full precision and all budgets are available in the appendix.'))
rows=[]
for dom in domains:
 for method in ['daewc','no_gate','no_ewc','no_proximity','no_regularizers']:
  r=next(x for x in g if x['domain']==dom and x['shots']==80 and x['method']==method)
  name={'daewc':'DAEWC','no_gate':'No gate','no_ewc':'No EWC','no_proximity':'No proximity','no_regularizers':'Neither penalty'}[method]
  rows.append([dn[dom] if method=='daewc' else '',name,ci(r),ci(r,'source_change')])
(tables/'ablations.tex').write_text(table('Controlled component removals at 80 examples per class with the compact pretrained encoder.','tab:ablations','llrr',['Target','Variant','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows,'All variants retain the same per-block adapters and shared calibration subset. Only the named component is removed.'))
# Complete main grid, with repeated headers over page breaks.
full=[]
for dom in domains:
 rows=[]
 for shots in [10,20,80,160]:
  for method in methods:
   r=next(x for x in g if x['domain']==dom and x['shots']==shots and x['method']==method)
   rows.append([shots if method=='head' else '',NAMES[method],ci(r),ci(r,'source_change')])
 full.append(table('Complete fixed-configuration results for '+dn[dom].lower()+'.','tab:full_'+dom,'llrr',['$K$ per class','Method','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows,long=True))
(tables/'full_results.tex').write_text(('\n'+r'\clearpage'+'\n').join(full))
# Zero shot and efficiency.
zero=[]
for domain in ['source']+domains:
 values=[]
 for seed in [42,43,44]:
  x=json.loads((root/f'local/source_seed{seed}/zero_shot.json').read_text())[domain]
  values.append(x['all']['macro_f1'] if domain=='source' else x['macro_f1'])
 q=interval(values);zero.append([dn[domain],f(q['mean'])+r' $\pm$ '+f(q['ci95_halfwidth'])])
(tables/'zero.tex').write_text(table('Source and zero-shot target performance before target adaptation.','tab:zero','lr',['Collection','Macro-$F_1$ (\\%)'],zero,'Source evaluation uses the frozen source-development threshold. Zero-shot target evaluation uses 0.5.'))
rows=[]
for method in methods:
 rs=[x for x in g if x['shots']==80 and x['method']==method]
 rows.append([NAMES[method],f"{rs[0]['trainable_parameters']:,}",f(np.mean([x['adapt_seconds_mean'] for x in rs]))])
(tables/'efficiency.tex').write_text(table('Compact-model adaptation cost at 80 examples per class.','tab:efficiency','lrr',['Method','Trainable parameters','Final adaptation (s)'],rows,'Times are descriptive local MPS measurements of 80 updates, excluding source training, Fisher estimation, and test evaluation. They are not controlled cross-hardware speed benchmarks. Cross-validation selection time is additional and separately logged.'))
# Feasibility: report the number of all domain-budget-seed observations, not a selected mean.
raw=[json.loads(p.read_text()) for p in (root/'local/runs').glob('*.json')]
from daewc.protocol import feasible
rows=[]
for method in methods:
 rs=[r for r in raw if r['method']==method]
 rows.append([NAMES[method]]+[f"{sum(feasible(r['delta_source_pp'],v) for r in rs)}/{len(rs)}" for v in [.5,1,2,5]])
(tables/'retention.tex').write_text(table('Loss-only retention feasibility across all 36 target--budget--seed cases per method.','tab:retention','lrrrr',['Method','0.5 pp','1 pp','2 pp','5 pp'],rows,'A run is feasible when its signed source change is greater than minus the tolerance. This table aggregates feasibility only; target performance remains available for every run. Both loss-only and absolute-band records are released.'))
seq=d['sequences'];rows=[]
for r in seq:
 rows.append([NAMES[r['method']],f(r['final_target_macro_f1_mean'])+r' $\pm$ '+f(r['final_target_macro_f1_ci95_halfwidth']),f(r['bwt_older_targets_pp_mean'])+r' $\pm$ '+f(r['bwt_older_targets_pp_ci95_halfwidth']),f(r['delta_source_pp_mean'])+r' $\pm$ '+f(r['delta_source_pp_ci95_halfwidth'])])
(tables/'sequential.tex').write_text(table('Sequential adaptation over all six domain orders. Each method has 18 sequences.','tab:sequential','lrrr',['Method','Final target $F_1$ (\\%)','Older-target BWT (pp)','$\\Delta$Src (pp)'],rows,'Intervals use the three seed-level averages across orders. Source is excluded from older-target backward transfer (BWT). The source-inclusive alternative and all intermediate matrices are released.'))
# Additional protocols: always reflect available run counts; final QA requires completion.
for name,label,title in [('budget_cv','cv','Within-budget cross-validation'),('random','random','Randomly initialized compact encoder'),('bert_base','base','Pretrained standard BERT-base')]:
 p=root/'tables'/name/'summary.json'
 if not p.exists():continue
 dd=json.loads(p.read_text());rows=[]
 for dom in domains:
  for method in methods:
   found=[x for x in dd['groups'] if x['domain']==dom and x['shots']==80 and x['method']==method]
   if found:
    r=found[0];rows.append([dn[dom] if method=='head' else '',NAMES[method],ci(r),ci(r,'source_change')])
 (tables/(name+'.tex')).write_text(table(title+' at 80 examples per class.','tab:'+label,'llrr',['Target','Method','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows,'The associated configuration, selection records where applicable, and complete predictions are included with the run artifacts.'))
# Complete stress grid, averaging domains within each seed.
from collections import defaultdict
mech=[json.loads(p.read_text()) for p in (root/'mechanism/runs').glob('*.json')];rows=[]
for lr in [1e-4,1e-3,1e-2]:
 for lam in [0.,100.,10000.]:
  selected=[r for r in mech if r['calibration_lr']==lr and r['lambda']==lam]
  if len(selected)!=9:continue
  source=interval([np.mean([r['delta_source_pp'] for r in selected if r['seed']==s]) for s in [42,43,44]])
  target=interval([np.mean([r['target']['all']['macro_f1'] for r in selected if r['seed']==s]) for s in [42,43,44]])
  rows.append([f'{lr:g}',f'{lam:g}',f(target['mean'])+r' $\pm$ '+f(target['ci95_halfwidth']),f(source['mean'])+r' $\pm$ '+f(source['ci95_halfwidth'])])
(tables/'mechanism.tex').write_text(table('Complete calibration-rate and Fisher-weight sensitivity grid at 80 examples per class.','tab:mechanism','rrrr',['Calibration rate','$\\lambda$','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows,'Values first average the three targets within each seed. Proximity remains fixed at 0.1. This is a descriptive intervention grid; no row is selected to replace the primary configuration. Per-domain records are supplied.'))
# Portable numerical macros, based on actual primary data.
r=next(x for x in raw if x['method']=='daewc');sourceinfo=json.loads((root/'local/source_seed42/source.json').read_text())
(out/'numbers.tex').write_text(f"\\newcommand{{\\TrainableCount}}{{{r['training']['trainable_parameters']:,}}}\n\\newcommand{{\\SharedCalibrationCount}}{{{sourceinfo['calibration_parameters']:,}}}\n\\newcommand{{\\SourceParameterCount}}{{{sourceinfo['total_parameters']:,}}}\n")
with (out/'numbers.tex').open('a') as stream:
 stream.write(source_change_macros(raw))
print('Tables regenerated.',{name:len(list((root/name/'runs').glob('*.json'))) for name in ['local','sequential','budget_cv','random','bert_base','mechanism']})

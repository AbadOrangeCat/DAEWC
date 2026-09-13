"""Generate the smaller-model, export, and second-test tables from complete records."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
project=Path(__file__).resolve().parents[1];sys.path.insert(0,str(project))
from daewc.summarize import interval,NAMES,summarize
p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,default=project/'revision_artifacts');p.add_argument('--manuscript',type=Path,default=project/'manuscript');args=p.parse_args()
root=args.artifacts;out=args.manuscript;tables=out/'tables';tables.mkdir(exist_ok=True)
domains=['health','public_affairs','entertainment'];titles={'health':'Health','public_affairs':'Public affairs','entertainment':'Entertainment'}
def fmt(v):
 return f"{v['mean']:.2f} $\\pm$ {v['ci95_halfwidth']:.2f}"
def table(caption,label,columns,header,rows,note='',long=False):
 body=' & '.join(header)+r' \\ \midrule'+'\n'
 data='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
 if long:
  return '\\begingroup\\small\n'+r'\begin{longtable}{'+columns+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\\toprule'+'\n'+body+r'\endfirsthead'+'\n'+r'\multicolumn{'+str(len(header))+r'}{l}{\textit{Continued from previous page}}\\\toprule'+'\n'+body+r'\endhead'+'\n'+data+'\n'+r'\bottomrule\end{longtable}\endgroup'+'\n'+note+'\n'
 return r'\begin{table}[htbp]\centering\small'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+body+data+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\par\smallskip\footnotesize '+note+'\n'+r'\end{table}'+'\n'
runs={name:[json.loads(p.read_text()) for p in (root/name/'runs').glob('*.json')] for name in ['local','low_footprint','budget_cv','low_footprint_cv']}
assert {name:len(rs) for name,rs in runs.items()}=={'local':288,'low_footprint':108,'budget_cv':126,'low_footprint_cv':36}
def values(directory,domain,method,shots=80,field='target'):
 rows=[r for r in runs[directory] if r['domain']==domain and r['method']==method and r['shots_per_class']==shots]
 assert len(rows)==3
 return interval([r['target']['all']['macro_f1'] if field=='target' else r['delta_source_pp'] for r in rows])
comparison=[('local','head','Head-only'),('local','full','Full fine-tuning'),('local','full_ewc','Full fine-tuning + EWC'),('local','adapter','Adapter, width 16'),('local','lora','LoRA, rank 8'),('local','lwf','LwF'),('local','daewc','DAEWC, widths 16/16'),('low_footprint','adapter','Adapter, width 2'),('low_footprint','daewc','DAEWC, widths 2/4')]
rows=[]
for directory,method,name in comparison:
 params=next(r['training']['trainable_parameters'] for r in runs[directory] if r['method']==method)
 rows.append([name,f'{params:,}']+[fmt(values(directory,d,method)) for d in domains])
(tables/'parameter_comparison.tex').write_text(table('Parameter cost and target performance under the fixed configuration at $K=80$. Widths denote adapter / domain-vector dimensions.','tab:parameter_comparison','lrrrr',['Method','Parameters','Health $F_1$','Public affairs $F_1$','Entertainment $F_1$'],rows,'Macro-$F_1$ is in percent. Intervals summarize the three paired seeds. All methods use exactly the same source checkpoints and target samples.'))
rows=[]
for domain in domains:
 for method in ['adapter','daewc']:
  for directory,protocol in [('low_footprint','Fixed'),('low_footprint_cv','Internal CV')]:
   rows.append([titles[domain],NAMES[method],protocol,fmt(values(directory,domain,method)),fmt(values(directory,domain,method,field='source'))])
(tables/'small80.tex').write_text(table('Smaller domain modules at 80 examples per class. Adapter width is 2 and the DAEWC domain-vector width is 4.','tab:small80','lllrr',['Target','Method','Protocol','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows,'The adapter-only control has 1,542 trainable parameters; smaller DAEWC has 6,154. Internal CV uses the same six candidate fits and label budget for each method.'))
rows=[]
for domain in domains:
 for method in ['daewc','no_gate','no_ewc','no_proximity','no_regularizers']:
  name={'daewc':'DAEWC','no_gate':'No gate','no_ewc':'No EWC','no_proximity':'No proximity','no_regularizers':'Neither penalty'}[method]
  rows.append([titles[domain],name,fmt(values('low_footprint',domain,method)),fmt(values('low_footprint',domain,method,field='source'))])
(tables/'small_ablations.tex').write_text(table('Complete component removals for the smaller architecture at $K=80$.','tab:small_ablations','llrr',['Target','Variant','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows))
rows=[]
for domain in domains:
 for shots in [10,20,80,160]:
  for method in ['adapter','daewc']:
   rows.append([titles[domain],shots,NAMES[method],fmt(values('low_footprint',domain,method,shots)),fmt(values('low_footprint',domain,method,shots,'source'))])
(tables/'small_full.tex').write_text(table('All fixed budgets for the smaller architecture.','tab:small_full','lllrr',['Target','$K$ per class','Method','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows,long=True))
rows=[]
for domain in domains:
 for shots in [10,80]:
  for method in ['adapter','daewc']:
   rows.append([titles[domain],shots,NAMES[method],fmt(values('low_footprint_cv',domain,method,shots)),fmt(values('low_footprint_cv',domain,method,shots,'source'))])
(tables/'small_cv.tex').write_text(table('All internal cross-validation budgets for the smaller architecture.','tab:small_cv','lllrr',['Target','$K$ per class','Method','Target $F_1$ (\\%)','$\\Delta$Src (pp)'],rows))
for name in ['low_footprint','low_footprint_cv']:
 summarize(root/name,root/'sequential_low' if name=='low_footprint' else None,root/'tables'/name)
sequences=[json.loads(p.read_text()) for p in (root/'sequential_low/runs').glob('*.json')]
if len(sequences)==72:
 rows=[]
 for method in ['adapter','lora','full','daewc']:
  selected=[r for r in sequences if r['method']==method]
  measures=[]
  for field in ['final_target_macro_f1','bwt_older_targets_pp','delta_source_pp']:
   measures.append(fmt(interval([np.mean([r[field] for r in selected if r['seed']==s]) for s in [42,43,44]])))
  rows.append([NAMES[method]]+measures)
 (tables/'sequential_small.tex').write_text(table('Sequential results with smaller domain modules across all six orders and three seeds.','tab:sequential_small','lrrr',['Method','Final target $F_1$ (\\%)','Older-target BWT (pp)','$\\Delta$Src (pp)'],rows,'Widths are 2/4 for DAEWC and 2 for adapter-only. Full fine-tuning and LoRA retain their reference architectures; their runs are repeated under this frozen configuration. Uncertainty is computed across three seed-level averages of the six orders.'))
confirmation=root/'tables/confirmation/second_test_summary.json'
if confirmation.exists():
 c=json.loads(confirmation.read_text());assert c['evaluations']==162
 for protocol in ['fixed','cv']:
  rows=[]
  for directory,method,name in comparison:
   group=protocol+('_low_footprint' if directory=='low_footprint' else '_reference')
   row=[name]
   for domain in domains:
    x=next(r for r in c['groups'] if (r['group'],r['domain'],r['method'])==(group,domain,method));row.append(fmt(x))
   rows.append(row)
  (tables/f'second_{protocol}.tex').write_text(table('Second-test results at $K=80$ using '+('fixed configurations' if protocol=='fixed' else 'within-budget model selection')+'.','tab:second_'+protocol,'lrrr',['Method','Health $F_1$','Public affairs $F_1$','Entertainment $F_1$'],rows,'Values are macro-$F_1$ percentages and 95\\% seed intervals. Second-test sample sizes are 137, 98, and 2,061. No labels from these partitions enter training or selection.'))
 rows=[]
 for r in c['paired']:
  m=r['comparison'].removeprefix('smaller DAEWC minus ')
  label='Adapter, width 2' if m=='small_adapter' else NAMES[m]
  rows.append(['Fixed' if r['protocol']=='fixed' else 'Internal CV',label,f"{r['mean_difference_pp']:.2f}",f"[{r['paired_bootstrap_ci95_low']:.2f}, {r['paired_bootstrap_ci95_high']:.2f}]"])
 (tables/'second_paired.tex').write_text(table('Smaller DAEWC minus every comparator on the second test, averaging the three domains equally.','tab:second_paired','llrr',['Protocol','Comparator','Difference (pp)','Paired 95\\% interval'],rows,'Intervals use 2,000 paired resamples of seeds and test items within each domain and label class. They condition on the tested domains and class counts. The full set of comparisons is retained; intervals are descriptive and unadjusted for multiple comparisons.'))
print('Extension tables rebuilt. Second-test tables available:',confirmation.exists(),'small sequences:',len(sequences))

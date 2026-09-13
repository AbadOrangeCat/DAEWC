"""Plot all specified parameter costs and fixed-protocol second-test scores."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
project=Path(__file__).resolve().parents[1];sys.path.insert(0,str(project))
from daewc.summarize import interval
p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,default=project/'revision_artifacts');p.add_argument('--manuscript',type=Path,default=project/'manuscript');args=p.parse_args()
runs=[json.loads(p.read_text()) for p in (args.artifacts/'confirmation/runs').glob('*.json')]
assert len(runs)==162
spec=[('reference','head','Head-only'),('reference','full','Full fine-tuning'),('reference','full_ewc','Full fine-tuning + EWC'),('reference','adapter','Adapter, width 16'),('reference','lora','LoRA, rank 8'),('reference','lwf','LwF'),('reference','daewc','DAEWC, widths 16/16'),('low_footprint','adapter','Adapter, width 2'),('low_footprint','daewc','DAEWC, widths 2/4')]
params=[];means=[];errors=[]
for group,method,label in spec:
 rows=[r for r in runs if r['group']=='fixed_'+group and r['method']==method]
 assert len(rows)==9
 x=interval([np.mean([r['scores']['macro_f1'] for r in rows if r['seed']==seed]) for seed in [42,43,44]])
 params.append(rows[0]['trainable_parameters']);means.append(x['mean']);errors.append(x['ci95_halfwidth'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(8.2,4.8),sharey=True,gridspec_kw={'width_ratios':[1.05,1.]})
y=np.arange(len(spec));colors=['#75808b']*len(spec);colors[4]='#5268a2';colors[-1]='#16745e';colors[-2]='#9a784a'
for j in range(len(spec)):
 axes[0].scatter(params[j],y[j],color=colors[j],s=43,zorder=3)
 axes[0].annotate(f'{params[j]:,}',(params[j],y[j]),xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
 axes[1].errorbar(means[j],y[j],xerr=errors[j],fmt='o',color=colors[j],capsize=3,markersize=5,linewidth=1.2)
axes[0].set_xscale('log');axes[0].set_xlim(170,3e7);axes[0].set_xticks([1e3,1e4,1e6],['1,000','10,000','1 million']);axes[0].set_yticks(y,[x[2] for x in spec]);axes[0].invert_yaxis()
axes[0].set_xlabel('Trainable parameters (log scale)');axes[0].set_title('Adaptation cost',pad=12)
axes[1].set_xlabel('Second-test mean macro-F1 (%)');axes[1].set_title('Target performance',pad=12)
for ax in axes:ax.grid(axis='x',alpha=.16);ax.tick_params(axis='y',length=0)
fig.tight_layout(w_pad=1.5)
out=args.manuscript/'figures';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'parameter_comparison.pdf',bbox_inches='tight');fig.savefig(out/'parameter_comparison.png',dpi=180,bbox_inches='tight')

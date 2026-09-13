from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project))
p=argparse.ArgumentParser()
p.add_argument('--artifacts',type=Path,default=project/'revision_artifacts')
p.add_argument('--manuscript',type=Path,default=project/'manuscript')
args=p.parse_args()
from daewc.summarize import NAMES
root=args.artifacts;out=args.manuscript/'figures';out.mkdir(parents=True,exist_ok=True)
g=json.loads((root/'tables/local/summary.json').read_text())['groups']
methods=['head','full','full_ewc','adapter','lora','lwf','daewc']
colors=['#686868','#c2463b','#cf9250','#3b9670','#73549a','#c2669a','#185e91']
domains=['health','public_affairs','entertainment'];names=['Health statements','Public-affairs headlines','Entertainment headlines']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(8.0,3.3),sharex=True)
for ax,domain,title in zip(axes,domains,names):
 for method,color in zip(methods,colors):
  rows=sorted([x for x in g if x['domain']==domain and x['method']==method],key=lambda x:x['shots'])
  ax.plot([x['shots'] for x in rows],[x['target_mean'] for x in rows],marker='o',markersize=3,color=color,label=NAMES[method],linewidth=2 if method=='daewc' else 1.1)
 ax.set_xscale('log',base=2);ax.set_xticks([10,20,80,160],[10,20,80,160]);ax.set_xlabel('Labelled examples per class');ax.set_title(title,pad=10);ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Target macro-F1 (%)')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.04))
fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/'learning_curves.pdf',bbox_inches='tight');fig.savefig(out/'learning_curves.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(8.0,3.3))
for ax,domain,title in zip(axes,domains,names):
 for method,color in zip(methods,colors):
  r=next(x for x in g if x['domain']==domain and x['shots']==80 and x['method']==method)
  # Duplicate full/EWC target means are shown at their actual source changes.
  ax.scatter(r['source_change_mean'],r['target_mean'],color=color,label=NAMES[method],s=48 if method=='daewc' else 27,marker='D' if method=='daewc' else 'o',zorder=3)
 ax.axvline(-1,color='#bbbbbb',ls=':',lw=1);ax.axvline(0,color='#dddddd',lw=.8);ax.set_title(title,pad=10);ax.set_xlabel('Signed source change (percentage points)');ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Target macro-F1 (%)')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.04))
fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/'pareto.pdf',bbox_inches='tight');fig.savefig(out/'pareto.png',dpi=180,bbox_inches='tight');plt.close(fig)

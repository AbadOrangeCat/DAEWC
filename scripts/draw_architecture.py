from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import argparse
p=argparse.ArgumentParser()
p.add_argument('--manuscript',type=Path,default=Path(__file__).resolve().parents[1]/'manuscript')
args=p.parse_args()
out=args.manuscript/'figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.0,4.8));ax.set_xlim(0,10);ax.set_ylim(0,7.2);ax.axis('off')
colors={'frozen':'#edf1f7','calibration':'#fff0cd','domain':'#e3f3ed'}
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.06',linewidth=.8,edgecolor='#526276',facecolor=color))
 ax.text(x+.14,y+h-.19,title,fontweight='bold',va='top',fontsize=9)
 ax.text(x+.14,y+h-.54,body,va='top',fontsize=8.3,linespacing=1.4)
box(.2,5.9,4.55,1.05,'1. Input text and known domain','One statement or headline;\ndomain supplied externally','white')
box(5.15,5.9,4.6,1.05,'2. Tokens and shared embeddings','Fixed WordPiece vocabulary; padding mask','white')
ax.add_patch(FancyBboxPatch((.15,1.98),9.65,3.5,boxstyle='round,pad=.06',linewidth=1,edgecolor='#526276',facecolor='#fbfcfe'))
ax.text(.38,5.21,'3. Repeat for each Transformer block',fontweight='bold',fontsize=10)
box(.4,3.5,4.05,1.2,'Shared block','Attention and feed-forward matrices\nFrozen during target adaptation',colors['frozen'])
box(.4,2.25,4.05,.93,'Shared calibration subset','Block biases and LayerNorm scales\nTrainable; protected by both penalties',colors['calibration'])

box(6.05,3.5,3.45,1.2,'Domain residual adapter','Bottleneck residual correction\nTrainable for the current domain',colors['domain'])
box(6.05,2.25,3.45,.93,'Domain feature gate','Learned domain vector\nFeature scaling for that domain',colors['domain'])
ax.text(.45,1.73,'Source inference bypasses target adapters and gates. Shared calibration can affect source scores.',fontsize=8.3)
box(.2,.2,4.55,1.08,'4. Masked mean pooling','Mean of non-padding token vectors','white')
box(5.15,.2,4.6,1.08,'5. Domain head and prediction','Probability of dataset label 1\nDomain-specific linear classifier',colors['domain'])
fig.tight_layout(pad=.5)
fig.savefig(out/'architecture.pdf',bbox_inches='tight');fig.savefig(out/'architecture.png',dpi=180,bbox_inches='tight')

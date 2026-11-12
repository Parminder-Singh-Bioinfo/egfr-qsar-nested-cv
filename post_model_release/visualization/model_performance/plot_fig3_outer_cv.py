"""New portable plot of saved outer-fold results; original plot source was not located."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True, type=Path)
p.add_argument('--output-dir', required=True, type=Path)
args = p.parse_args()
out = args.output_dir
if out.exists() and any(out.iterdir()):
    raise FileExistsError('Output directory must be new or empty')
df = pd.read_csv(args.input)
datasets = ['Single-protein', 'Cell-based']
reps = ['2D', '3D', 'Morgan FP', '2D + 3D', '2D + 3D + Morgan FP']
if len(df) != 50 or df.duplicated(['Dataset','Representation','Outer_Fold']).any():
    raise ValueError('Expected 50 unique dataset/representation/fold rows')
for ds in datasets:
    for rep in reps:
        if set(df.loc[df.Dataset.eq(ds) & df.Representation.eq(rep),'Outer_Fold']) != {1,2,3,4,5}:
            raise ValueError('Missing five-fold group: '+ ds + '/' +rep)
algos = ['Extra Trees','KNN','LightGBM','Random Forest','SVM-RBF','XGBoost']
colors = dict(zip(algos, ['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9']))
markers = dict(zip(algos, ['o','s','^','D','v','P']))
if not set(df.Algorithm).issubset(colors):
    raise ValueError('Unexpected algorithm; add an explicit color and marker')
def symbol_area(count):
    return (float(count)/65.0)**2
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'], 'font.size':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','xtick.labelsize':7,'ytick.labelsize':7})
fig, axes = plt.subplots(3,2,figsize=(7.5,7.7),sharex=True,sharey='row')
fig.subplots_adjust(left=.085,right=.985,bottom=.138,top=.888,wspace=.085,hspace=.24)
summaries = []
pr_label = 'AP'
for row, (metric,label) in enumerate([('MCC','Held-out outer-fold MCC'),('PR_AUC','Held-out outer-fold '+pr_label),('ROC_AUC','Held-out outer-fold ROC-AUC')]):
    for col, ds in enumerate(datasets):
        ax = axes[row,col]
        for j,rep in enumerate(reps):
            if j % 2 == 0:
                ax.axvspan(j-.5,j+.5,color='0.96',zorder=0)
            d = df[df.Dataset.eq(ds) & df.Representation.eq(rep)].sort_values('Outer_Fold')
            for offset, item in zip(np.linspace(-.30,.30,5),d.itertuples()):
                ax.scatter(j+offset,getattr(item,metric),color=colors[item.Algorithm],marker=markers[item.Algorithm],s=symbol_area(item.Actual_Selected_Features),edgecolor='white',linewidth=.3,zorder=3)
            mean,sd = d[metric].mean(),d[metric].std(ddof=1)
            ax.errorbar(j,mean,yerr=sd,fmt='none',color='black',capsize=2.5,lw=.8,zorder=4)
            ax.hlines(mean,j-.36,j+.36,color='black',lw=1.25,zorder=4)
            summaries.append(dict(Dataset=ds,Representation=rep,Metric=metric,Mean=mean,SD=sd))
        if col == 0:
            ax.set_ylabel(label)
        ax.set_xlim(-.5,4.5)
        ax.set_xticks(range(len(reps)))
        ax.set_xticklabels(['2D','3D','Morgan','2D\n+ 3D','2D\n+ 3D\n+ Morgan'])
        ax.grid(axis='y',alpha=.4,lw=.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-.14 if col==0 else -.06,1.06,'ABCDEF'[2*row+col],transform=ax.transAxes,fontweight='bold',fontsize=11)
        if row == 0:
            ax.set_title(ds,fontsize=9,pad=14)
    vals=df[metric]
    margin=(vals.max()-vals.min())*.05
    axes[row,0].set_ylim(vals.min()-margin,vals.max()+margin)
handles = [Line2D([],[],marker=markers[a],color=colors[a],ls='',label=a,markersize=5) for a in algos]
handles += [Line2D([],[],marker='+',color='black',ls='',label='Mean ± SD (5 folds)',markersize=9)]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.992),ncol=7,frameon=False,fontsize=7.5,columnspacing=1.05,handletextpad=.35)
sizes=[200,350,500]
size_handles=[plt.scatter([],[],s=symbol_area(n),color='0.5',edgecolors='white',linewidth=.3,label=str(n)) for n in sizes]
fig.legend(handles=size_handles,title='Selected features',loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=8,title_fontsize=8)
out.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','svg','png','tiff']:
    fig.savefig(out / ('Fig3_outer_cv.'+ext),dpi=600 if ext=='tiff' else 300,bbox_inches='tight')
plt.close(fig)
pd.DataFrame(summaries).to_csv(out/'outer_cv_summary.csv',index=False)
(out/'caption_Fig3.txt').write_text('Figure 3. Held-out outer-fold performance during nested grouped cross-validation. Panels A and B show Matthews correlation coefficient (MCC), C and D show average precision (AP), and E and F show ROC-AUC for the single-protein and cell-based datasets, respectively. Individual symbols represent the models evaluated in five outer folds; color and shape identify the algorithm and symbol size represents actual selected molecular features. Black markers and error bars indicate mean ± sample standard deviation across five outer folds. AP was calculated using average_precision_score.\n')
(out/'plot_manifest.json').write_text(json.dumps({'input':str(args.input.resolve()),'sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'source':'New plotting code matched to supplied manuscript image; point-area scale inferred from raster, not recovered original source','model_training_performed':False,'precision_recall_display_label':pr_label,'precision_recall_metric':'average_precision_score','algorithm_colors':colors,'algorithm_markers':markers,'point_area_formula':'(actual_selected_features / 65)**2','feature_size_legend':sizes},indent=2)+'\n')

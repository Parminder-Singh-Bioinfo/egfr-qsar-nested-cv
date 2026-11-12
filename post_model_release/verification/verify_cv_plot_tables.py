"""Compare frozen plotting tables with the original saved search and fold files."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--projects-root',required=True,type=Path)
p.add_argument('--input-dir',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
inner=pd.read_csv(a.input_dir/'all_350_inner_cv_results.csv')
outer=pd.read_csv(a.input_dir/'combined_single_protein_cell_based_outer_cv.csv')
dev=pd.read_csv(a.input_dir/'combined_development_search_plot_data.csv')
names={'extra_trees':'Extra Trees','knn':'KNN','lightgbm':'LightGBM','logistic':'Logistic Regression','random_forest':'Random Forest','svm_rbf':'SVM-RBF','xgboost':'XGBoost'}
reps={'2d':'2D','3d':'3D','morgan':'Morgan FP','2d_3d':'2D + 3D','2d_3d_morgan':'2D + 3D + Morgan FP'}
checks=[]
def check(name,ok):
    checks.append({'check':name,'passed':bool(ok)})
    if not ok:
        raise AssertionError(name)
def equal(x,y):
    return np.allclose(x,y,atol=1e-10,rtol=0,equal_nan=True)
for dataset,folder in [('Single-protein','egfr_source_holdout/results/final_nested_qsar/full_cutoff_7'),('Cell-based','egfr_cell_based_source_holdout/results/final_nested_qsar/literature_multisource_token_supported/full_cutoff_7')]:
    for rep,label in reps.items():
        base=a.projects_root/folder/rep
        fold=pd.read_csv(base/'metrics/nested_outer_fold_metrics.csv').sort_values('outer_fold')
        view=outer[outer.Dataset.eq(dataset)&outer.Representation.eq(label)].sort_values('Outer_Fold')
        check(dataset+'/'+rep+'/five folds',len(fold)==len(view)==5 and set(view.Outer_Fold)=={1,2,3,4,5})
        for raw,col in [('mcc','MCC'),('pr_auc','PR_AUC'),('roc_auc','ROC_AUC'),('budget','Feature_Budget'),('selected_molecular_features','Actual_Selected_Features')]:
            check(dataset+'/'+rep+'/'+col,equal(fold[raw],view[col]))
        check(dataset+'/'+rep+'/fold algorithms',list(fold.model.map(names))==list(view.Algorithm))
        parts=[]
        for k in range(1,6):
            table=pd.read_csv(base/('inner_search/outer_fold_%d_search.csv'%k))
            table['outer_fold']=k
            parts.append(table)
        raw=pd.concat(parts,ignore_index=True)
        winners=raw.loc[raw.groupby(['outer_fold','model','budget']).inner_mean_mcc.idxmax()]
        group=winners.groupby(['model','budget']).inner_mean_mcc.agg(['mean','std','count'])
        label_inner=label.replace(' FP','')
        view=inner[inner.Dataset.eq(dataset)&inner.Representation.eq(label_inner)]
        check(dataset+'/'+rep+'/35 inner cells',len(view)==35)
        for item in view.itertuples():
            entry=group.loc[(item.Algorithm,item.Budget)]
            check(dataset+'/'+rep+'/inner/'+item.Algorithm+'/'+str(item.Budget),entry['count']==5 and equal([entry['mean'],entry['std']],[item.Mean_MCC,item.SD_MCC]))
        raw=pd.read_csv(base/'inner_search/final_winner_development_search.csv')
        best=raw.sort_values('inner_mean_mcc',ascending=False).drop_duplicates(['model','budget'])
        view=dev[dev.Dataset.eq(dataset)&dev.Representation.eq(label)]
        check(dataset+'/'+rep+'/35 development cells',len(view)==len(best)==35 and view.Selected_Final_Model.sum()==1)
        for item in view.itertuples():
            match=best[best.model.map(names).eq(item.Algorithm)&best.budget.eq(item.Feature_Budget)]
            check(dataset+'/'+rep+'/development/'+item.Algorithm+'/'+str(item.Feature_Budget),len(match)==1 and equal([match.iloc[0].inner_mean_mcc,match.iloc[0].inner_sd_mcc],[item.Mean_Inner_CV_MCC,item.SD_Inner_CV_MCC]))
        selected=view[view.Selected_Final_Model].iloc[0]
        check(dataset+'/'+rep+'/selected highest development MCC',equal([selected.Mean_Inner_CV_MCC],[best.inner_mean_mcc.max()]))
a.output.write_text(json.dumps({'all_passed':True,'checks':checks,'model_training_performed':False},indent=2)+'\n')
print('Passed',len(checks),'checks against original saved CV files')

"""Reproduce a full-versus-ablated comparison table from saved results only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args()
    reference=args.reference_dir.expanduser().resolve()
    output=args.output_dir.expanduser().resolve()
    names=['ablation_comparison.csv','ablation_comparison.md','comparison_manifest.json']
    if any((output/name).exists() for name in names):
        p.error('An output file already exists. Choose a new output directory.')
    summary_path=reference/'ablation_external_summary.csv'
    summary=pd.read_csv(summary_path)
    if set(summary.Dataset)!={'single_protein','cell_based'} or summary.Dataset.duplicated().any():
        p.error('Expected one ablation summary row for each verified dataset.')
    inputs=[summary_path]
    rows=[]
    for dataset,label,features,n in [('single_protein','Single-protein',300,2524),('cell_based','Cell-based',438,192)]:
        baseline_path=reference/(dataset+'_full_external_metrics.csv')
        prediction_path=reference/(dataset+'_external_predictions_without_OHE.csv')
        inputs.extend([baseline_path,prediction_path])
        baseline=pd.read_csv(baseline_path)
        baseline=baseline.loc[baseline.analysis_group.eq('Total')]
        if len(baseline)!=1:
            raise ValueError('Expected exactly one full-model Total row for '+dataset)
        full=baseline.iloc[0]
        ablated=summary.set_index('Dataset').loc[dataset]
        pred=pd.read_csv(prediction_path)
        if len(pred)!=n or full['rows']!=n or ablated.External_N!=n:
            raise ValueError('Unexpected external cohort size for '+dataset)
        if ablated.Mutation_features_removed!=6 or ablated.Molecular_features!=features:
            raise ValueError('Unexpected feature counts for '+dataset)
        if not np.array_equal(pred.external_row_index.to_numpy(),np.arange(n)):
            raise ValueError('Unexpected ablation row indexing for '+dataset)
        y=pred.observed_activity.to_numpy(dtype=int)
        predicted=pred.predicted_activity.to_numpy(dtype=int)
        probability=pred.probability_active.to_numpy(dtype=float)
        if not set(np.unique(y))<={0,1} or not set(np.unique(predicted))<={0,1} or not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
            raise ValueError('Invalid saved predictions for '+dataset)
        measured={'MCC':matthews_corrcoef(y,predicted),'ROC_AUC':roc_auc_score(y,probability),
                  'AP':average_precision_score(y,probability)}
        for metric,value in measured.items():
            if not np.isclose(value,ablated[metric],atol=1e-10,rtol=0):
                raise ValueError('Saved summary does not match predictions: '+dataset+'/'+metric)
        tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
        for key,value in {'TN':tn,'FP':fp,'FN':fn,'TP':tp}.items():
            if int(ablated[key])!=value: raise ValueError('Confusion-count mismatch: '+dataset+'/'+key)
        row={'Dataset':label,'Algorithm':ablated.Algorithm,'Molecular_features':features,
             'Mutation_features_removed':6,'Development_N':int(ablated.Development_N),'External_N':n}
        for metric,source in [('MCC','mcc'),('ROC_AUC','roc_auc'),('AP','pr_auc')]:
            row[metric+'_full']=float(full[source])
            row[metric+'_ablated']=float(ablated[metric])
            row[metric+'_full_minus_ablated']=float(full[source]-ablated[metric])
        rows.append(row)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'ablation_comparison.csv',index=False)
    lines=['# Mutation-feature ablation comparison','',
        '| Dataset | Molecular features | Mutation indicators removed | Full MCC | Ablated MCC | MCC decrease | Full ROC-AUC | Ablated ROC-AUC | Full AP | Ablated AP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[r['Dataset'],str(r['Molecular_features']),'6']
        values += [f"{r[k]:.3f}" for k in ['MCC_full','MCC_ablated','MCC_full_minus_ablated',
            'ROC_AUC_full','ROC_AUC_ablated','AP_full','AP_ablated']]
        lines.append('| '+' | '.join(values)+' |')
    lines+=['','Decreases are full minus ablated, calculated before rounding. AP is average precision. ',
        'Ablation uses fixed selected molecular features and hyperparameters, with development-only retraining. ',
        'This table was rebuilt from existing saved results; no classifier was fitted.']
    (output/'ablation_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={'method':'Replay from saved metrics and ablated external predictions',
        'model_training_performed':False,'saved_ablation_metrics_verified':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs],
        'removed_indicators':['C797S','L858R','L861Q','T790M','Other_EGFR_mutation','Mutation_unspecified'],
        'numpy_version':np.__version__,'pandas_version':pd.__version__}
    (output/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Comparison table rebuilt; both saved ablation summaries match their external predictions. No models trained.')

if __name__=='__main__': main()

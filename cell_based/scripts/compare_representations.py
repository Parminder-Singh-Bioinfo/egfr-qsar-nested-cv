#!/usr/bin/env python3
"""Compare nested-CV and frozen external results across five representations."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ORDER = ["2d", "3d", "morgan", "2d_3d", "2d_3d_morgan"]
LABEL = {"2d":"2D", "3d":"3D", "morgan":"Morgan FP", "2d_3d":"2D + 3D", "2d_3d_morgan":"2D + 3D + Morgan FP"}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--results-root",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path); p.add_argument("--dpi",type=int,default=300); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); cv=[]; ext=[]; winners=[]
    for rep in ORDER:
        root=a.results_root/rep
        c=pd.read_csv(root/"metrics/nested_cv_summary.csv"); e=pd.read_csv(root/"metrics/strict_external_metrics_total_inside_outside_ad.csv"); w=pd.read_csv(root/"metrics/development_selected_winner.csv")
        c["representation"]=rep; e["representation"]=rep; w["representation"]=rep
        cv.append(c); ext.append(e); winners.append(w)
    cv=pd.concat(cv,ignore_index=True); ext=pd.concat(ext,ignore_index=True); winners=pd.concat(winners,ignore_index=True)
    cv["representation_label"]=cv.representation.map(LABEL); ext["representation_label"]=ext.representation.map(LABEL)
    cv.to_csv(a.output_dir/"nested_cv_all_representations.csv",index=False); ext.to_csv(a.output_dir/"external_total_inside_outside_all_representations.csv",index=False); winners.to_csv(a.output_dir/"development_selected_configurations.csv",index=False)
    # Both input tables contain a display-label column.  Merging them therefore
    # creates representation_label_external/representation_label_cv.  Rebuild
    # one canonical label after the merge so downstream plotting is stable.
    total=ext.loc[ext.analysis_group.eq("Total")].drop(columns="representation_label").merge(
        cv.drop(columns="representation_label"),
        on="representation",
        suffixes=("_external","_cv"),
    )
    total["representation_label"]=total["representation"].map(LABEL)
    total.to_csv(a.output_dir/"nested_cv_vs_total_external.csv",index=False)
    sns.set_theme(style="whitegrid",context="talk"); order=[LABEL[x] for x in ORDER]
    long=total.melt(id_vars="representation_label",value_vars=["mean_outer_mcc","mcc"],var_name="evaluation",value_name="MCC")
    long["evaluation"]=long.evaluation.map({"mean_outer_mcc":"Nested grouped CV","mcc":"Strict external total"})
    fig,ax=plt.subplots(figsize=(14,7)); sns.barplot(data=long,x="representation_label",y="MCC",hue="evaluation",order=order,ax=ax,palette=["#4C78A8","#E15759"]); ax.axhline(0,color="black",lw=1); ax.tick_params(axis="x",rotation=18); ax.set(xlabel="Molecular representation",ylabel="MCC",title="Nested CV versus frozen external validation"); fig.tight_layout(); fig.savefig(a.output_dir/"nested_cv_vs_external_mcc.png",dpi=a.dpi,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,7)); sns.barplot(data=ext,x="representation_label",y="mcc",hue="analysis_group",order=order,ax=ax,palette=["#4C78A8","#59A14F","#E15759"]); ax.axhline(0,color="black",lw=1); ax.tick_params(axis="x",rotation=18); ax.set(xlabel="Molecular representation",ylabel="MCC",title="Frozen external performance by common structural AD"); fig.tight_layout(); fig.savefig(a.output_dir/"external_mcc_total_inside_outside_ad.png",dpi=a.dpi,bbox_inches="tight"); plt.close(fig)
    for metric, ylabel, filename in [
        ("roc_auc", "ROC-AUC", "external_roc_auc_total_inside_outside_ad.png"),
        ("pr_auc", "PR-AUC", "external_pr_auc_total_inside_outside_ad.png"),
    ]:
        fig,ax=plt.subplots(figsize=(14,7))
        sns.barplot(data=ext,x="representation_label",y=metric,hue="analysis_group",order=order,ax=ax,palette=["#4C78A8","#59A14F","#E15759"])
        ax.axhline(0.5,color="black",lw=1,ls="--" if metric == "roc_auc" else ":")
        ax.tick_params(axis="x",rotation=18)
        ax.set(
            xlabel="Molecular representation",
            ylabel=ylabel,
            title="Frozen external {} by common structural AD".format(ylabel),
        )
        ax.set_ylim(0,1)
        fig.tight_layout(); fig.savefig(a.output_dir/filename,dpi=a.dpi,bbox_inches="tight"); plt.close(fig)
    print(total[["representation_label","mean_outer_mcc","mcc","roc_auc","pr_auc"]].to_string(index=False)); print("Saved:",a.output_dir)
if __name__=="__main__": main()

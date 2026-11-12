#!/usr/bin/env python3
"""Leakage-safe grouped-CV QSAR training and frozen external validation.

Feature filtering and ANOVA/FDR ranking are fitted independently inside every
development training fold.  ``InChIKey_STD`` defines the groups.  External
data are transformed only with preprocessing learned from development data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


KEY = "InChIKey_STD"
TARGET = "Activity_binary"
CONTEXT = "Mutation_clean"
BASE_MUTATIONS = ["C797S", "L858R", "L861Q", "T790M"]
MUTATIONS = BASE_MUTATIONS + ["Other_EGFR_mutation", "Mutation_unspecified"]
PREFIXES = {
    "2d": ("rdkit_2d_", "mordred_2d_", "padel_2d_"),
    "3d": ("rdkit_3d_", "mordred_3d_", "padel_3d_"),
    "morgan": ("rdkit_morgan_",),
}
LEAKAGE_COLUMNS = {
    "Activity_binary_input", "Activity", "Activity_binary_full",
    "Activity_binary", "Activity_rule", "Activity_binary_high_confidence",
    "High_confidence_status", "pIC50", "IC50_geomean_nM", "IC50_median_nM",
}
SCALED_MODELS = {"logistic", "svm", "svm_rbf", "knn"}
# scikit-learn tree ensembles internally convert inputs to float32.  Values
# outside this conservative range are therefore unusable even when they are
# technically finite float64 values.  Treat them exactly like missing
# descriptor values and impute them from the development-training data.
FLOAT32_SAFE_ABS = float(np.finfo(np.float32).max / 10.0)


def numeric_feature_frame(frame, columns):
    """Return numeric descriptor values with unusable values replaced by NaN."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.mask(values.abs() > FLOAT32_SAFE_ABS)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", required=True, type=Path)
    p.add_argument("--external", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--representation", required=True, choices=["2d", "3d", "morgan", "2d_3d", "2d_3d_morgan"])
    p.add_argument("--analysis-name", required=True)
    p.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200, 300, 500])
    p.add_argument("--models", nargs="+", default=["logistic", "svm_rbf", "random_forest", "extra_trees", "lightgbm", "xgboost", "knn"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--missing-threshold", type=float, default=0.30)
    p.add_argument("--correlation-threshold", type=float, default=0.90)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def normalize_context(series):
    return (series.astype("string").fillna("NOMUTATION").str.upper()
            .str.replace(" ", "", regex=False).str.replace("_", "", regex=False)
            .replace({"": "NOMUTATION", "NAN": "NOMUTATION", "NONE": "NOMUTATION"}))


def add_mutations(frame):
    frame = frame.copy()
    context = normalize_context(frame[CONTEXT])
    frame[CONTEXT] = context
    for mutation in BASE_MUTATIONS:
        frame[mutation] = context.str.contains(mutation, regex=False, na=False).astype("int8")
    unspecified = context.str.contains("UNDEFINED|UNSPECIFIED|UNKNOWN", regex=True, na=False)
    frame["Mutation_unspecified"] = unspecified.astype("int8")
    known_no_mutation = {"NOMUTATION", "WILDTYPE", "WT"}
    known_tokens = set(BASE_MUTATIONS)

    def contains_other(value):
        if value in known_no_mutation or any(word in value for word in ["UNDEFINED", "UNSPECIFIED", "UNKNOWN"]):
            return 0
        tokens = [token for token in value.replace("+", ",").replace(";", ",").split(",") if token]
        return int(any(token not in known_tokens for token in tokens))

    frame["Other_EGFR_mutation"] = context.map(contains_other).astype("int8")
    return frame


def read_data(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    required = [KEY, TARGET, CONTEXT]
    missing = [c for c in required if c not in frame]
    if missing:
        raise KeyError("{} lacks {}".format(path, missing))
    frame[KEY] = frame[KEY].astype("string").str.strip().str.upper()
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("{}: {} must be binary".format(path, TARGET))
    frame[TARGET] = y.astype("int8")
    return add_mutations(frame)


def feature_groups(frame, representation):
    requested = {
        "2d": ["2d"], "3d": ["3d"], "morgan": ["morgan"],
        "2d_3d": ["2d", "3d"], "2d_3d_morgan": ["2d", "3d", "morgan"],
    }[representation]
    groups = {}
    for name in requested:
        columns = [c for c in frame.columns if c.startswith(PREFIXES[name])]
        if not columns:
            raise ValueError("No {} molecular features detected".format(name))
        forbidden = sorted(set(columns) & LEAKAGE_COLUMNS)
        if forbidden:
            raise ValueError("Outcome leakage found among features: {}".format(forbidden))
        groups[name] = columns
    return groups


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.minimum(q_ranked, 1.0)
    return q


def filter_rank_group(train, columns, y, args):
    values = numeric_feature_frame(train, columns)
    missing_rate = values.isna().mean()
    keep = missing_rate.index[missing_rate <= args.missing_threshold].tolist()
    values = values[keep]
    medians = values.median().fillna(0.0)
    values = values.fillna(medians)
    variance = values.var(ddof=0)
    keep = variance.index[variance > 1e-12].tolist()
    values, medians = values[keep], medians[keep]
    if values.empty:
        raise ValueError("All features failed missingness/variance filters")

    f_scores, p_values = f_classif(values.to_numpy(dtype=float), y)
    q_values = bh_fdr(p_values)
    ranking = pd.DataFrame({
        "feature": values.columns, "f_score": np.nan_to_num(f_scores, nan=-np.inf),
        "p_value": np.nan_to_num(p_values, nan=1.0), "q_value": q_values,
    }).sort_values(["q_value", "f_score"], ascending=[True, False]).reset_index(drop=True)
    significant = ranking.loc[ranking.q_value <= args.fdr_alpha]
    candidates = significant.feature.tolist() if len(significant) else ranking.feature.tolist()

    # Keep the best-ranked member of highly correlated pairs. Because columns
    # are processed by training-fold ANOVA rank, the more target-associated
    # descriptor is retained.
    accepted = []
    for feature in candidates:
        if not accepted:
            accepted.append(feature)
            continue
        correlations = values[accepted].corrwith(values[feature]).abs()
        if not (correlations >= args.correlation_threshold).any():
            accepted.append(feature)
    ranking["passed_fdr"] = ranking.q_value <= args.fdr_alpha
    ranking["passed_correlation"] = ranking.feature.isin(accepted)
    return accepted, medians.to_dict(), ranking


def allocation(names, budget):
    names = list(names)
    if len(names) == 1:
        return {names[0]: budget}
    if len(names) == 2:
        return {names[0]: budget // 2, names[1]: budget - budget // 2}
    base = budget // 3
    result = {name: base for name in names}
    result["morgan"] += budget - sum(result.values())
    return result


def select_features(train, y, groups, budget, args):
    ranked, medians, audits = {}, {}, []
    for name, columns in groups.items():
        accepted, group_medians, audit = filter_rank_group(train, columns, y, args)
        ranked[name] = accepted
        medians.update(group_medians)
        audit.insert(0, "feature_group", name)
        audits.append(audit)
    wanted = allocation(groups.keys(), budget)
    selected = {name: ranked[name][:min(wanted[name], len(ranked[name]))] for name in groups}
    remaining = min(budget, sum(len(x) for x in ranked.values())) - sum(len(x) for x in selected.values())
    while remaining:
        changed = False
        for name in groups:
            if len(selected[name]) < len(ranked[name]) and remaining:
                selected[name].append(ranked[name][len(selected[name])])
                remaining -= 1
                changed = True
        if not changed:
            break
    flat = [feature for name in groups for feature in selected[name]]
    return flat, {f: float(medians[f]) for f in flat}, pd.concat(audits, ignore_index=True), {k: len(v) for k, v in selected.items()}


def matrix(frame, features, medians):
    missing = [c for c in features if c not in frame]
    if missing:
        raise ValueError("Dataset lacks selected features: {}".format(missing[:10]))
    molecular = numeric_feature_frame(frame, features)
    for feature in features:
        molecular[feature] = molecular[feature].fillna(medians[feature])
    mutation = frame[MUTATIONS].apply(pd.to_numeric, errors="raise")
    result = np.hstack([molecular.to_numpy(dtype=float), mutation.to_numpy(dtype=float)])
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values remain after frozen median imputation")
    if np.abs(result).max(initial=0.0) > FLOAT32_SAFE_ABS:
        raise ValueError("Values outside the safe float32 range remain after preprocessing")
    return result


def models(args):
    result = {}
    requested = set(args.models)
    if "logistic" in requested:
        result["logistic"] = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=args.random_state)
    # ``svm`` is retained as a backwards-compatible alias. Both names use RBF.
    if "svm_rbf" in requested:
        result["svm_rbf"] = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=args.random_state)
    elif "svm" in requested:
        result["svm"] = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=args.random_state)
    if "random_forest" in requested:
        result["random_forest"] = RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_split=8, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=args.n_jobs, random_state=args.random_state)
    if "extra_trees" in requested:
        result["extra_trees"] = ExtraTreesClassifier(n_estimators=500, max_depth=14, min_samples_split=6, min_samples_leaf=3, max_features="sqrt", bootstrap=True, class_weight="balanced", n_jobs=args.n_jobs, random_state=args.random_state)
    if "lightgbm" in requested:
        try:
            from lightgbm import LGBMClassifier
            result["lightgbm"] = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, max_depth=6, min_child_samples=40, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0, class_weight="balanced", n_jobs=args.n_jobs, random_state=args.random_state, verbosity=-1)
        except ImportError:
            print("Warning: LightGBM unavailable; skipped", file=sys.stderr)
    if "xgboost" in requested:
        try:
            from xgboost import XGBClassifier
            # Conservative regularization is used to limit train-validation gaps.
            result["xgboost"] = XGBClassifier(
                n_estimators=500, learning_rate=0.03, max_depth=4,
                min_child_weight=8, subsample=0.80, colsample_bytree=0.70,
                reg_alpha=0.5, reg_lambda=8.0, gamma=0.1,
                objective="binary:logistic", eval_metric="logloss",
                n_jobs=args.n_jobs, random_state=args.random_state,
            )
        except ImportError:
            print("Warning: XGBoost unavailable; skipped", file=sys.stderr)
    if "knn" in requested:
        result["knn"] = KNeighborsClassifier(
            n_neighbors=11, weights="distance", metric="minkowski", p=2,
            n_jobs=args.n_jobs,
        )
    if not result:
        raise ValueError("No requested model is available")
    return result


def fit(name, estimator, X, y):
    scaler = StandardScaler() if name in SCALED_MODELS else None
    X_fit = scaler.fit_transform(X) if scaler else X
    model = clone(estimator)
    if name == "xgboost":
        negatives = max(int((np.asarray(y) == 0).sum()), 1)
        positives = max(int((np.asarray(y) == 1).sum()), 1)
        model.set_params(scale_pos_weight=negatives / positives)
    model = model.fit(X_fit, y)
    return model, scaler


def probability(model, scaler, X):
    transformed = scaler.transform(X) if scaler else X
    return model.predict_proba(transformed)[:, 1]


def metrics(y, prob, threshold):
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "mcc": matthews_corrcoef(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def bootstrap_ci(frame, y, prob, args):
    if args.bootstrap <= 0:
        return {}
    rng = np.random.RandomState(args.random_state)
    keys = frame[KEY].astype(str).to_numpy()
    unique = np.unique(keys)
    mapping = {key: np.flatnonzero(keys == key) for key in unique}
    samples = {name: [] for name in ["mcc", "roc_auc", "pr_auc", "balanced_accuracy"]}
    for _ in range(args.bootstrap):
        chosen = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([mapping[key] for key in chosen])
        result = metrics(np.asarray(y)[index], np.asarray(prob)[index], args.threshold)
        for name in samples:
            if np.isfinite(result[name]): samples[name].append(result[name])
    output = {}
    for name, values in samples.items():
        output[name + "_ci_low"] = np.percentile(values, 2.5) if values else np.nan
        output[name + "_ci_high"] = np.percentile(values, 97.5) if values else np.nan
    return output


def plot_outputs(summary, predictions, output, args):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, group in summary.groupby("model"):
        group = group.sort_values("budget")
        ax.errorbar(group.budget, group.mean_mcc, yerr=group.sd_mcc, marker="o", capsize=3, label=name)
    ax.set(xlabel="Selected molecular features", ylabel="Grouped 5-fold CV MCC", title=args.analysis_name + " – " + args.representation)
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "figures" / "cv_mcc_by_selected_features.png", dpi=args.dpi); plt.close(fig)

    y = predictions.observed.to_numpy(); p = predictions.probability_active.to_numpy()
    fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label="AUC = {:.3f}".format(roc_auc_score(y, p))); axes[0].plot([0,1],[0,1], "--", color="grey")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Strict external ROC"); axes[0].legend()
    axes[1].plot(recall, precision, label="AP = {:.3f}".format(average_precision_score(y, p))); axes[1].axhline(y.mean(), ls="--", color="grey", label="Prevalence = {:.3f}".format(y.mean()))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Strict external precision–recall"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "figures" / "strict_external_roc_pr.png", dpi=args.dpi); plt.close(fig)


def main():
    args = parse_args(); start = time.time()
    for folder in ["metrics", "selected_features", "final_models", "predictions", "figures", "audit"]:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    development, external = read_data(args.development), read_data(args.external)
    overlap = set(development[KEY]) & set(external[KEY])
    if overlap:
        raise ValueError("Strict external data contain {} development compounds".format(len(overlap)))
    groups = feature_groups(development, args.representation)
    external_missing = [c for cols in groups.values() for c in cols if c not in external]
    if external_missing:
        raise ValueError("External data lack {} molecular columns".format(len(external_missing)))
    y = development[TARGET].to_numpy(dtype=int); compound_groups = development[KEY].astype(str).to_numpy()
    estimators = models(args)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    splits = list(splitter.split(np.zeros(len(y)), y, compound_groups))
    fold_rows, selection_rows = [], []
    total = len(args.budgets) * len(splits) * len(estimators); done = 0
    print("Development: {} rows, {} compounds; external: {} rows, {} compounds".format(len(development), development[KEY].nunique(), len(external), external[KEY].nunique()), flush=True)
    print("Molecular groups: {}".format({k: len(v) for k,v in groups.items()}), flush=True)
    for budget in args.budgets:
        for fold_number, (train_idx, test_idx) in enumerate(splits, 1):
            train, test = development.iloc[train_idx], development.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            selected, medians, audit, group_allocation = select_features(train, y_train, groups, budget, args)
            audit["budget"], audit["fold"] = budget, fold_number; selection_rows.append(audit)
            X_train, X_test = matrix(train, selected, medians), matrix(test, selected, medians)
            for model_name, estimator in estimators.items():
                model, scaler = fit(model_name, estimator, X_train, y_train)
                train_result = metrics(y_train, probability(model, scaler, X_train), args.threshold)
                test_result = metrics(y_test, probability(model, scaler, X_test), args.threshold)
                row = {"analysis": args.analysis_name, "representation": args.representation, "budget": budget, "model": model_name, "fold": fold_number, "selected_molecular_features": len(selected), "mutation_features": len(MUTATIONS), "final_feature_count": len(selected)+len(MUTATIONS), "train_mcc": train_result["mcc"], "train_validation_gap": train_result["mcc"]-test_result["mcc"]}
                row.update(test_result); fold_rows.append(row); done += 1
                elapsed = time.time()-start; eta=(elapsed/done)*(total-done)/60
                print("[{}/{}] {} budget={} fold={} MCC={:.3f} gap={:.3f} ETA={:.1f}m".format(done,total,model_name,budget,fold_number,test_result["mcc"],row["train_validation_gap"],eta), flush=True)
    folds = pd.DataFrame(fold_rows); folds.to_csv(args.output_dir / "metrics/all_fold_metrics.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(args.output_dir / "selected_features/fold_feature_audits.csv", index=False)
    summary = folds.groupby(["analysis","representation","budget","model"], as_index=False).agg(mean_mcc=("mcc","mean"),sd_mcc=("mcc","std"),mean_balanced_accuracy=("balanced_accuracy","mean"),mean_roc_auc=("roc_auc","mean"),mean_pr_auc=("pr_auc","mean"),mean_f1=("f1","mean"),mean_train_mcc=("train_mcc","mean"),mean_train_validation_gap=("train_validation_gap","mean"))
    summary.to_csv(args.output_dir / "metrics/model_summary_metrics.csv", index=False)
    winner = summary.sort_values(["mean_mcc","mean_roc_auc"], ascending=False).iloc[0]
    pd.DataFrame([winner]).to_csv(args.output_dir / "metrics/cv_selected_winner.csv", index=False)
    budget, model_name = int(winner.budget), str(winner.model)
    selected, medians, final_audit, group_allocation = select_features(development, y, groups, budget, args)
    final_audit.to_csv(args.output_dir / "selected_features/final_selected_feature_audit.csv", index=False)
    pd.DataFrame({"feature": selected}).to_csv(args.output_dir / "selected_features/final_selected_features.csv", index=False)
    X_dev, X_ext = matrix(development, selected, medians), matrix(external, selected, medians)
    model, scaler = fit(model_name, estimators[model_name], X_dev, y)
    bundle = {"model": model, "scaler": scaler, "selected_molecular_features": selected, "mutation_features": MUTATIONS, "imputation_medians": medians, "threshold": args.threshold, "representation": args.representation, "analysis_name": args.analysis_name, "model_name": model_name, "budget": budget, "feature_group_allocation": group_allocation, "target": TARGET, "key": KEY}
    model_path = args.output_dir / "final_models/{}__{}__budget_{}.joblib".format(args.representation,model_name,budget); joblib.dump(bundle, model_path, compress=3)
    external_y = external[TARGET].to_numpy(dtype=int); external_prob = probability(model, scaler, X_ext)
    external_result = {"analysis":args.analysis_name,"representation":args.representation,"model":model_name,"budget":budget,"rows":len(external),"unique_compounds":external[KEY].nunique(),"active_prevalence":external_y.mean(),"pr_auc_no_skill_baseline":external_y.mean()}
    external_result.update(metrics(external_y, external_prob, args.threshold)); external_result.update(bootstrap_ci(external,external_y,external_prob,args))
    pd.DataFrame([external_result]).to_csv(args.output_dir / "metrics/strict_external_metrics.csv", index=False)
    predictions = external[[KEY,CONTEXT]].copy(); predictions["observed"]=external_y; predictions["probability_active"]=external_prob; predictions["predicted"]=(external_prob>=args.threshold).astype(int)
    predictions.to_csv(args.output_dir / "predictions/strict_external_predictions.csv", index=False)
    plot_outputs(summary,predictions,args.output_dir,args)
    configuration=vars(args).copy(); configuration={k:(str(v) if isinstance(v,Path) else v) for k,v in configuration.items()}; configuration["mutation_features"]=MUTATIONS; configuration["blocked_leakage_columns"]=sorted(LEAKAGE_COLUMNS)
    (args.output_dir / "run_configuration.json").write_text(json.dumps(configuration,indent=2)+"\n")
    print("CV-selected winner: {} budget={}; strict external MCC={:.3f}, ROC-AUC={:.3f}, PR-AUC={:.3f}".format(model_name,budget,external_result["mcc"],external_result["roc_auc"],external_result["pr_auc"]), flush=True)
    print("Results: {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
